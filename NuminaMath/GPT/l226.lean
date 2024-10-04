import Mathlib

namespace quotient_of_501_div_0_point_5_l226_226740

theorem quotient_of_501_div_0_point_5 : 501 / 0.5 = 1002 := by
  sorry

end quotient_of_501_div_0_point_5_l226_226740


namespace altitudes_sine_inequality_l226_226599

theorem altitudes_sine_inequality 
  (h_a h_b h_c r : ℝ) 
  (A B C : ℝ) 
  (h_a_def : h_a = 2 * (1 / (A) )) 
  (h_b_def : h_b = 2 * (1 / (B) ))
  (h_c_def : h_c = 2 * (1 / (C) ))
  (sin_def : Π x, sin x = x) :
  h_a * sin A + h_b * sin B + h_c * sin C ≤ 9 * real.sqrt 3 / 2 * r :=
begin
  sorry
end

end altitudes_sine_inequality_l226_226599


namespace limit_at_one_third_l226_226797

noncomputable def delta (ε : ℝ) : ℝ := ε / 3

theorem limit_at_one_third :
  ∀ (ε > 0), ∃ (δ > 0), ∀ (x : ℝ),
    0 < abs (x - 1 / 3) ∧ abs (x - 1 / 3) < δ →
    abs ((3 * x^2 + 17 * x - 6) / (x - 1 / 3) - 19) < ε :=
by
  intros ε hε
  use delta ε
  split
  · linarith
  intros x hx
  sorry

end limit_at_one_third_l226_226797


namespace factorize_cubic_l226_226869

theorem factorize_cubic (a : ℝ) : a^3 - 16 * a = a * (a + 4) * (a - 4) :=
sorry

end factorize_cubic_l226_226869


namespace smaller_octagon_area_fraction_l226_226731

theorem smaller_octagon_area_fraction (A B C D E F G H : Point) (O : Point) :
  is_regular_octagon A B C D E F G H →
  is_center O A B C D E F G H →
  let A' := midpoint A B,
      B' := midpoint B C,
      C' := midpoint C D,
      D' := midpoint D E,
      E' := midpoint E F,
      F' := midpoint F G,
      G' := midpoint G H,
      H' := midpoint H A in
  is_octa_center O A' B' C' D' E' F' G' H' →
  (area_of_octagon A B C D E F G H) * (1 / 4) = area_of_octagon A' B' C' D' E' F' G' H' :=
by
  -- Sorry, proof is omitted.
  sorry

end smaller_octagon_area_fraction_l226_226731


namespace tickets_spent_on_beanie_l226_226469

-- Define the initial number of tickets Jerry had.
def initial_tickets : ℕ := 4

-- Define the number of tickets Jerry won later.
def won_tickets : ℕ := 47

-- Define the current number of tickets Jerry has.
def current_tickets : ℕ := 49

-- The statement of the problem to prove the tickets spent on the beanie.
theorem tickets_spent_on_beanie :
  initial_tickets + won_tickets - 2 = current_tickets := by
  sorry

end tickets_spent_on_beanie_l226_226469


namespace vika_pairs_exactly_8_ways_l226_226346

theorem vika_pairs_exactly_8_ways :
  ∃ d : ℕ, (d ∣ 30) ∧ (Finset.card (Finset.filter (λ d, d ∣ 30) (Finset.range 31)) = 8) := 
sorry

end vika_pairs_exactly_8_ways_l226_226346


namespace area_ratio_of_smaller_octagon_l226_226722

theorem area_ratio_of_smaller_octagon
    (A B C D E F G H : ℝ × ℝ) -- Coordinates of vertices of the larger octagon
    (P Q R S T U V W : ℝ × ℝ) -- Coordinates of vertices of the smaller octagon
    (regular_octagon : ∀ (X Y Z W U V T S : ℝ × ℝ), regular_octo X Y Z W U V T S)  -- Predicate for regular octagon
    (midpoints_joined : ∀ (X Y : ℝ × ℝ), midpoint X Y) : -- Condition that midpoints form the smaller octagon
  area (smaller_octo P Q R S T U V W) = (3 : ℝ) / 4 * area (larger_octo A B C D E F G H) :=
sorry

end area_ratio_of_smaller_octagon_l226_226722


namespace compute_g_five_times_l226_226233

def g : ℝ → ℝ :=
  λ x, if x ≥ 0 then -x^3 else x + 10

theorem compute_g_five_times :
  g (g (g (g (g 2)))) = -8 :=
by
  sorry

end compute_g_five_times_l226_226233


namespace solution_set_of_inequality_l226_226322

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l226_226322


namespace find_number_l226_226183

theorem find_number (x : ℝ) (h : 0.65 * x = 0.05 * 60 + 23) : x = 40 :=
sorry

end find_number_l226_226183


namespace describe_shape_l226_226258

open Complex

theorem describe_shape (m n : ℝ) (z : ℂ) (h_eqns : |z + Complex.I * n| + |z - Complex.I * m| = n ∧ |z + Complex.I * n| - |z - Complex.I * m| = -m) : 
  ∃ F1 F2, F1 = (0, -n) ∧ F2 = (0, m) ∧ ((ellipse_with_foci F1 F2 z ∧ hyperbola_with_foci F1 F2 z) := 
begin
  sorry
end

end describe_shape_l226_226258


namespace card_paiting_modulus_l226_226366

theorem card_paiting_modulus (cards : Finset ℕ) (H : cards = Finset.range 61 \ {0}) :
  ∃ d : ℕ, ∀ n ∈ cards, ∃! k, (∀ x ∈ cards, (x + n ≡ k [MOD d])) ∧ (d ∣ 30) ∧ (∃! n : ℕ, 1 ≤ n ∧ n ≤ 8) :=
sorry

end card_paiting_modulus_l226_226366


namespace houses_with_garage_l226_226200

theorem houses_with_garage (h70 : 70 = 70) (hP : 40 = 40) (hGP : 35 = 35) (hN : 15 = 15) :
  (exists G : ℕ, G + 40 - 35 = 55 ∧ G = 50) :=
  exists.intro 50 (by simp)

end houses_with_garage_l226_226200


namespace enclosed_shape_area_l226_226300

noncomputable def enclosedArea : ℝ :=
  ∫ x in -1..1, (1 - x^2)

theorem enclosed_shape_area :
  enclosedArea = 4 / 3 :=
by 
  sorry

end enclosed_shape_area_l226_226300


namespace integer_solutions_count_l226_226168

theorem integer_solutions_count :
  let eq : Int -> Int -> Int := fun x y => 6 * y ^ 2 + 3 * x * y + x + 2 * y - 72
  ∃ (sols : List (Int × Int)), 
    (∀ x y, eq x y = 0 → (x, y) ∈ sols) ∧
    (∀ p ∈ sols, ∃ x y, p = (x, y) ∧ eq x y = 0) ∧
    sols.length = 4 :=
by
  sorry

end integer_solutions_count_l226_226168


namespace convert_to_spherical_l226_226488

variables {x y z : ℝ}
def point_rectangular := (x, y, z)
def point_spherical := (ρ θ φ : ℝ)

theorem convert_to_spherical (x := 2 * Real.sqrt 2) (y := -2 * Real.sqrt 2) (z := 2) 
    (hρ : ρ = Real.sqrt ((2 * Real.sqrt 2)^2 + (-2 * Real.sqrt 2)^2 + 2^2) = 2 * Real.sqrt 5)
    (hφ : φ = Real.arccos (1 / (Real.sqrt 5)))
    (hθ : θ = 7 * Real.pi / 6) :
    point_spherical ρ θ φ = (2 * Real.sqrt 5, 7 * Real.pi / 6, Real.arccos (1 / Real.sqrt 5)) :=
by
  sorry

end convert_to_spherical_l226_226488


namespace smaller_octagon_area_fraction_l226_226725

theorem smaller_octagon_area_fraction (A B C D E F G H : Point)
  (midpoints_joined : Boolean)
  (regular_octagon : RegularOctagon A B C D E F G H)
  (smaller_octagon : Octagon (midpoint (A, B)) (midpoint (B, C)) (midpoint (C, D)) 
                              (midpoint (D, E)) (midpoint (E, F)) (midpoint (F, G))
                              (midpoint (G, H)) (midpoint (H, A))) :
  midpoints_joined → regular_octagon → 
  (area smaller_octagon) = (3 / 4) * (area regular_octagon) :=
by
  sorry

end smaller_octagon_area_fraction_l226_226725


namespace number_divisible_by_5_l226_226394

theorem number_divisible_by_5 (n : ℕ) (h : String.length (nat.digits 10 n) = 3 ∧
    (nat.digits 10 n).head = 3 ∧
    (nat.digits 10 n).get_last (nat.succ_ne_zero 2) = 5):
    ∃ k : ℕ, k ≤ 9 ∧ n = 300 + 10 * k + 5 := 
    sorry

end number_divisible_by_5_l226_226394


namespace tan_15_eq_sqrt3_l226_226022

theorem tan_15_eq_sqrt3 :
  (1 + Real.tan (Real.pi / 12)) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
sorry

end tan_15_eq_sqrt3_l226_226022


namespace no_poly_degree_3_satisfies_conditions_l226_226159

theorem no_poly_degree_3_satisfies_conditions :
  ¬∃ (f : ℝ → ℝ), (∃ (a b c : ℝ), a ≠ 0 ∧ f = λ x, a * x ^ 3 + b * x + c) ∧
  (∀ x, f (x ^ 2) = (f x) ^ 2) ∧
  (∀ x, (f (x)) ^ 2 = f (f x)) := by
  sorry

end no_poly_degree_3_satisfies_conditions_l226_226159


namespace four_digit_square_condition_l226_226238

theorem four_digit_square_condition (n : ℕ) (h1 : n >= 1000) (h2 : n < 10000) (h3 : ∀ d ∈ Int.digits (Int.ofNat n), d < 6) :
  ∃ k l : ℕ, (n = k * k) ∧ (∃ m : ℕ, m = l * l ∧ Int.digits (Int.ofNat m) = List.map (λ x => x + 1) (Int.digits (Int.ofNat n))) :=
begin
  use 45,
  use 56,
  split,
  { exact 2025 },
  { use 3136,
    split,
    { exact 3136 },
    { simp,
      have : Int.digits (Int.ofNat 2025) = [2, 0, 2, 5], by sorry,
      exact this, }
  }
end

end four_digit_square_condition_l226_226238


namespace asymptote_of_hyperbola_l226_226208

theorem asymptote_of_hyperbola (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0)
    (intersects : ∃ A B : ℝ × ℝ, A ≠ B ∧
        (A.1^2 / a^2 - A.2^2 / b^2 = 1 ∧ A.1^2 = 2 * p * A.2) ∧
        (B.1^2 / a^2 - B.2^2 / b^2 = 1 ∧ B.1^2 = 2 * p * B.2)) :
    (∀ (F : ℝ × ℝ), ∀ (O : ℝ × ℝ), F = (0, p / 2) ∧ O = (0, 0) →
    (abs (dist (classical.some intersects) F) + abs (dist (classical.some (classical.some_spec intersects)) F) = 4 * abs (dist O F))) →
    ∃ m : ℝ, m = Real.sqrt 2 / 2 ∧ ∀ x, y = m * x ∨ y = -m * x :=
by
  sorry

end asymptote_of_hyperbola_l226_226208


namespace pallet_weight_l226_226445

theorem pallet_weight (box_weight : ℕ) (num_boxes : ℕ) (total_weight : ℕ) 
  (h1 : box_weight = 89) (h2 : num_boxes = 3) : total_weight = 267 := by
  sorry

end pallet_weight_l226_226445


namespace compare_sequences_l226_226643

theorem compare_sequences (a b s t u v : ℝ) (h0 : 0 < a) (h1 : a < b)
  (h2 : s = (2*a + b) / 3) (h3 : t = (a + 2*b) / 3)
  (h4 : u = a * Real.cbrt (b / a)) (h5 : v = a * (Real.cbrt (b / a))^2):
  (st * (s + t) > uv * (u + v)) :=
begin
  let st := s * t,
  let uv := u * v,
  let x := st * (s + t),
  let y := uv * (u + v),
  sorry,
end

end compare_sequences_l226_226643


namespace problem1_problem2_l226_226149

noncomputable def A := {x : ℝ | 2 ≤ x ∧ x ≤ 6}
noncomputable def B := {x : ℝ | 3 * x - 7 ≥ 8 - 2 * x}
noncomputable def C (a : ℝ) := {x : ℝ | x ≤ a}

theorem problem1 : (A ∩ B)ᶜ = {x : ℝ | x < 3 ∨ x > 6} :=
by
  sorry

theorem problem2 (a : ℝ) (h : A ⊆ C a) : 6 ≤ a :=
by
  sorry

end problem1_problem2_l226_226149


namespace sum_min_max_prime_factors_of_1365_l226_226775

-- Define a function to check if a number is prime (for manual handling in proof)
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the number we're working with
def number := 1365

-- Define the list of known prime factors
def prime_factors : list ℕ := [3, 5, 7, 13]

-- Ensure all elements of prime_factors are prime
lemma all_primes : ∀ p ∈ prime_factors, is_prime p := by
  intros p h
  cases h
  case or.inl h_0 => exact nat.prime_three
  case or.inr h_0 => cases h_0
    case or.inl h_1 => exact nat.prime_five
    case or.inr h_1 => cases h_1
      case or.inl h_2 => exact nat.prime_seven
      case or.inr h_2 => exact nat.prime_thirteen

-- Define the sum of the largest and smallest prime factors
def sum_of_min_and_max : ℕ := list.minimum' prime_factors + list.maximum' prime_factors

-- Theorem stating that this sum is 16
theorem sum_min_max_prime_factors_of_1365 : sum_of_min_and_max = 16 := by
  sorry

end sum_min_max_prime_factors_of_1365_l226_226775


namespace find_y_coordinate_of_Q_l226_226660

noncomputable def y_coordinate_of_Q 
  (P R T S : ℝ × ℝ) (Q : ℝ × ℝ) (areaPentagon areaSquare : ℝ) : Prop :=
  P = (0, 0) ∧ 
  R = (0, 5) ∧ 
  T = (6, 0) ∧ 
  S = (6, 5) ∧ 
  Q.fst = 3 ∧ 
  areaSquare = 25 ∧ 
  areaPentagon = 50 ∧ 
  (1 / 2) * 6 * (Q.snd - 5) + areaSquare = areaPentagon

theorem find_y_coordinate_of_Q : 
  ∃ y_Q : ℝ, y_coordinate_of_Q (0, 0) (0, 5) (6, 0) (6, 5) (3, y_Q) 50 25 ∧ y_Q = 40 / 3 :=
sorry

end find_y_coordinate_of_Q_l226_226660


namespace factor_diff_of_squares_l226_226078

theorem factor_diff_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4 * y) * (5 + 4 * y) := 
sorry

end factor_diff_of_squares_l226_226078


namespace find_interval_x_l226_226871

theorem find_interval_x (x : ℝ) :
  (x^2 + 7 * x < 12) ↔ (x ∈ set.Ioo (-4 : ℝ) (-3 : ℝ)) :=
by
  sorry

end find_interval_x_l226_226871


namespace slope_of_line_determined_by_solutions_eq_l226_226391

theorem slope_of_line_determined_by_solutions_eq :
  ∀ (x y : ℝ), (4 / x + 5 / y = 0) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro x y h
  use -5 / 4
  sorry

end slope_of_line_determined_by_solutions_eq_l226_226391


namespace quadratic_completion_l226_226487

theorem quadratic_completion (a b : ℤ) (h_eq : (x : ℝ) → x^2 - 10 * x + 25 = 0) :
  (∃ a b : ℤ, ∀ x : ℝ, (x + a) ^ 2 = b) → a + b = -5 := by
  sorry

end quadratic_completion_l226_226487


namespace max_diff_intersection_points_l226_226492

theorem max_diff_intersection_points :
  let f₁ := λ x : ℝ => 5 - x^2 + 2 * x^3
  let f₂ := λ x : ℝ => 3 + 2 * x^2 + 2 * x^3
  ∀ x₁ x₂, f₁ x₁ = f₂ x₁ → f₁ x₂ = f₂ x₂ → 
      abs ((5 - (2 / 3) + (4 * (sqrt 6) / 9)) - (5 - (2 / 3) - (4 * (sqrt 6) / 9))) = (8 * sqrt 6) / 9 :=
by
  intros f₁ f₂ x₁ x₂ H₁ H₂
  sorry

end max_diff_intersection_points_l226_226492


namespace sum_of_y_coordinates_on_y_axis_l226_226854

noncomputable def center := (-4, 3)
noncomputable def radius := 5

theorem sum_of_y_coordinates_on_y_axis 
  (C : Point)
  (center_c : C = center) 
  (radius_c : ∀ (P : Point), ((P.1 + 4)^2 + (P.2 - 3)^2 = 25) → (dist C P = radius)):
  ∀ (P1 P2 : Point), (P1.1 = 0) ∧ (P2.1 = 0) 
                      ∧ ((P1.1 + 4)^2 + (P1.2 - 3)^2 = 25)
                      ∧ ((P2.1 + 4)^2 + (P2.2 - 3)^2 = 25) 
                      → P1.2 + P2.2 = 6 :=
begin
  sorry
end

end sum_of_y_coordinates_on_y_axis_l226_226854


namespace range_of_y_l226_226307

section
variable (x : ℝ) (y : ℝ)

def f (x : ℝ) : ℝ := 2 ^ x
def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_of_y :
  (∃ x, (1 / 8 : ℝ) ≤ x ∧ x ≤ 4 ∧ y = g (1 / x) * g (4 * x)) ↔ -8 ≤ y ∧ y ≤ 1 :=
by
  sorry
end

end range_of_y_l226_226307


namespace find_n_value_l226_226093

noncomputable def roots_of_ratio_condition (d e f : ℝ) : ℝ :=
  let n := (4 * d) / (2 * d - e)
  in if 2 * d - e ≠ 0 then n else 0

theorem find_n_value {d e f : ℝ} (hyp : ∀ y : ℝ, (y^2 + 2 * d * y) / (e * y + f) = (roots_of_ratio_condition d e f) / (roots_of_ratio_condition d e f - 2) → y = -y) :
  roots_of_ratio_condition d e f = (4 * d) / (2 * d - e) := by
  -- Using conditions and deriving n
  sorry

end find_n_value_l226_226093


namespace calculate_expression_l226_226535

variable (f g : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = - g (-x)

theorem calculate_expression 
  (hf : is_even_function f)
  (hg : is_odd_function g)
  (hfg : ∀ x : ℝ, f x - g x = x ^ 3 + x ^ 2 + 1) :
  f 1 + g 1 = 1 :=
  sorry

end calculate_expression_l226_226535


namespace cos_beta_eq_sqrt2_div_2_l226_226243

theorem cos_beta_eq_sqrt2_div_2 (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : cos α = sqrt 5 / 5) 
  (h4 : sin (α - β) = sqrt 10 / 10) : 
  cos β = sqrt 2 / 2 := 
sorry

end cos_beta_eq_sqrt2_div_2_l226_226243


namespace fraction_of_loss_correct_l226_226450

-- Define the cost price and selling price of the apple.
def cost_price : ℝ := 16
def selling_price : ℝ := 15 

-- Define the loss as the difference between cost price and selling price.
def loss : ℝ := cost_price - selling_price

-- Define the fraction of loss compared to the cost price.
def fraction_of_loss : ℝ := loss / cost_price

theorem fraction_of_loss_correct :
  fraction_of_loss = 1 / 16 := 
  by
    -- Placeholder for the proof
    sorry

end fraction_of_loss_correct_l226_226450


namespace hyperbola_equation_l226_226557

open Real

theorem hyperbola_equation (a b : ℝ) (h : a^2 + b^2 = 25) (P : (ℝ × ℝ)) (hP : P = (2,1)) :
  P.2 = b / a * P.1 → (a = 2 * sqrt 5 ∧ b = sqrt 5) ∧ (C : ℝ → ℝ → Prop := λ x y, x^2 / 20 - y^2 / 5 = 1) :=
by
  sorry

end hyperbola_equation_l226_226557


namespace normal_vector_for_plane_l226_226820

-- Define the points A, B, and C
def A : Euclidean 3 := (-1, 0, 1)
def B : Euclidean 3 := (1, 1, 2)
def C : Euclidean 3 := (2, -1, 0)

-- Define the vector u
def u : Euclidean 3 := (0, 1, -1)

-- Prove that u is a normal vector to the plane passing through A, B, and C
theorem normal_vector_for_plane :
  ∃ u : Euclidean 3, u = (0, 1, -1) ∧ 
  is_normal_vector A B C u := sorry

end normal_vector_for_plane_l226_226820


namespace number_of_squares_is_five_l226_226479

-- A function that computes the number of squares obtained after the described operations on a piece of paper.
def folded_and_cut_number_of_squares (initial_shape : Type) (folds : ℕ) (cuts : ℕ) : ℕ :=
  -- sorry is used here as a placeholder for the actual implementation
  sorry

-- The main theorem stating that after two folds and two cuts, we obtain five square pieces.
theorem number_of_squares_is_five (initial_shape : Type) (h_initial_square : initial_shape = square)
  (h_folds : folds = 2) (h_cuts : cuts = 2) : folded_and_cut_number_of_squares initial_shape folds cuts = 5 :=
  sorry

end number_of_squares_is_five_l226_226479


namespace problem1_problem2_l226_226927

variable (k : ℝ)

-- Definitions of proposition p and q
def p (k : ℝ) : Prop := ∀ x : ℝ, x^2 - k*x + 2*k + 5 ≥ 0

def q (k : ℝ) : Prop := (4 - k > 0) ∧ (1 - k < 0)

-- Theorem statements based on the proof problem
theorem problem1 (hq : q k) : 1 < k ∧ k < 4 :=
by sorry

theorem problem2 (hp_q : p k ∨ q k) (hp_and_q_false : ¬(p k ∧ q k)) : 
  (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) :=
by sorry

end problem1_problem2_l226_226927


namespace smallest_x_l226_226398

open Classical
noncomputable theory

def conditions (x : ℕ) : Prop :=
  x % 3 = 2 ∧ x % 4 = 3 ∧ x % 5 = 4

theorem smallest_x : ∃ (x : ℕ), conditions x ∧ (∀ (y : ℕ), conditions y → x ≤ y) ∧ x = 59 :=
by {
  sorry
}

end smallest_x_l226_226398


namespace find_second_factor_l226_226694

noncomputable def HCF : Nat := 20
noncomputable def larger_number : Nat := 460
noncomputable def factor1 : Nat := 21

def is_lcm_of_factors (HCF : Nat) (factor1 : Nat) (factor2 : Nat) (n m : Nat) : Prop :=
  n = HCF * factor1 * factor2 ∧ m = HCF * (some_factor n larger_number)

theorem find_second_factor
  (HCF : Nat) (larger_number : Nat) (factor1 : Nat)
  (h1 : GCD larger_number (some_number) = HCF)
  (h2 : larger_number = HCF * 23) :
  ∃ factor2, is_lcm_of_factors HCF factor1 factor2 larger_number some_number →
             factor2 = 23 := sorry

end find_second_factor_l226_226694


namespace range_of_a_l226_226961

noncomputable def universal_set : Set ℝ := {x | true}

variables {a : ℝ}

def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a^2 + 2}

def p (x : ℝ) : Prop := x ∈ ({x | 2 < x ∧ x < 3 * a + 1})

def q (x : ℝ) : Prop := x ∈ B a

def necessary_condition : Prop := ∀ x : ℝ, p x → q x

theorem range_of_a (a : ℝ) :
  necessary_condition →
  (a ≤ sqrt (a^2 + 2) ∧ 1/3 < a) ∨ (a < 1/3) :=
by
  -- Skip proof part
  sorry

end range_of_a_l226_226961


namespace most_popular_computer_l226_226789

-- Define the sales data for each computer over the years
def sales_A : ℕ × ℕ × ℕ := (600, 610, 590)
def sales_B : ℕ × ℕ × ℕ := (590, 650, 700)
def sales_C : ℕ × ℕ × ℕ := (650, 670, 660)

-- Define the trend function to check if sales are consistently increasing
def is_continuously_increasing (sales : ℕ × ℕ × ℕ) : Prop :=
  sales.1 < sales.2 ∧ sales.2 < sales.3

-- Theorem stating that computer B is the most popular based on sales trend
theorem most_popular_computer :
  is_continuously_increasing sales_B ∧
  ¬ is_continuously_increasing sales_A ∧
  ¬ is_continuously_increasing sales_C :=
by
  sorry

end most_popular_computer_l226_226789


namespace smallest_integer_cube_root_form_l226_226638

theorem smallest_integer_cube_root_form (m : ℕ) (h : ∃ (n : ℕ) (r : ℝ), m = (n + r)^3 ∧ 0 < r ∧ r < 1 / 500) : 
  let n := 13 in n = 13 :=
by
  sorry

end smallest_integer_cube_root_form_l226_226638


namespace measure_of_angle_B_l226_226803

-- Defining the points and the triangle
variables (A B C P Q : Type) [point A] [point B] [point C] [point P] [point Q]

-- Defining the angles
variables (angleA angleB angleC angleP angleQ : measure angle)

-- Given conditions
axiom isosceles_triangle : is_isosceles_triangle A B C AC
axiom points_on_segments : on_segment P CB ∧ on_segment Q AB
axiom equal_segments : AC = AP ∧ AP = PQ ∧ PQ = QB ∧ QB = BC

-- Goal: Prove that angle B is 60 degrees
theorem measure_of_angle_B : angleB = 60 :=
by { sorry }

end measure_of_angle_B_l226_226803


namespace equal_chances_iff_odd_k_l226_226801

-- Define the set of the first 100 positive integers
def first100 := { x : ℕ | 1 ≤ x ∧ x ≤ 100 }

-- Define the game and winning conditions
def A_wins (s : Finset ℕ) : Prop := s.sum % 2 = 0
def B_wins (s : Finset ℕ) : Prop := s.sum % 2 = 1

-- Define the condition under which the winning chances are equal
def equal_winning_chances (k : ℕ) : Prop := 
∀ s : Finset ℕ, s.card = k → (A_wins s ↔ B_wins s)

-- Theorem statement
theorem equal_chances_iff_odd_k (k : ℕ) : equal_winning_chances k ↔ k % 2 = 1 := by
  sorry

end equal_chances_iff_odd_k_l226_226801


namespace valid_three_digit_numbers_count_l226_226176

theorem valid_three_digit_numbers_count : 
∑ n in Finset.range 1000, (100 ≤ n ∧ n < 1000 ∧ 
  ¬((n / 100 = (n % 100) / 10 ∧ (n % 100) / 10 ≠ n % 10) ∨ 
    ((n % 100) / 10 = n % 10 ∧ n / 100 ≠ (n % 100) / 10))) → 738 :=
by
  sorry

end valid_three_digit_numbers_count_l226_226176


namespace smaller_octagon_area_fraction_l226_226734

theorem smaller_octagon_area_fraction (A B C D E F G H : Point) (O : Point) :
  is_regular_octagon A B C D E F G H →
  is_center O A B C D E F G H →
  let A' := midpoint A B,
      B' := midpoint B C,
      C' := midpoint C D,
      D' := midpoint D E,
      E' := midpoint E F,
      F' := midpoint F G,
      G' := midpoint G H,
      H' := midpoint H A in
  is_octa_center O A' B' C' D' E' F' G' H' →
  (area_of_octagon A B C D E F G H) * (1 / 4) = area_of_octagon A' B' C' D' E' F' G' H' :=
by
  -- Sorry, proof is omitted.
  sorry

end smaller_octagon_area_fraction_l226_226734


namespace find_L_l226_226625

variable (L : ℕ) -- L represents the number of movies L&J Productions produces in a year

-- Johnny TV makes 25 percent more movies than L&J Productions each year
def johnnyTV_produces (L : ℕ) : ℕ := 1.25 * L

-- The total number of movies produced by both companies in one year
def total_produces_per_year (L : ℕ) : ℕ := L + johnnyTV_produces L

-- The total number of movies produced by both companies in five years
def total_produces_in_five_years (L : ℕ) : ℕ := 5 * total_produces_per_year L

-- Given condition: the two companies produce 2475 movies in five years
axiom combined_five_years_production : total_produces_in_five_years L = 2475

-- Prove the number of movies L&J Productions produces in a year
theorem find_L : L = 220 :=
by
  sorry

end find_L_l226_226625


namespace inclination_of_shortest_chord_through_point_l226_226084

noncomputable def angle_of_inclination_of_shortest_chord
  (x y : ℝ) (h : x^2 + y^2 + 2*x - 4*y = 0)
  (p : ℝ × ℝ) (hp : p = (0, 1)) : ℝ :=
  π / 4

-- Theorem statement
theorem inclination_of_shortest_chord_through_point :
  ∀ (x y : ℝ), (x^2 + y^2 + 2*x - 4*y = 0) →
  ∀ (p : ℝ × ℝ), (p = (0, 1)) →
  angle_of_inclination_of_shortest_chord x y (x^2 + y^2 + 2*x - 4*y = 0) p p = π / 4
:= by
  intros x y h p hp
  rw hp
  exact sorry

end inclination_of_shortest_chord_through_point_l226_226084


namespace trace_of_A_l226_226244

noncomputable def A (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, 2], ![-3, d]]

noncomputable def A_inv (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let det_inv := (a * d + 6)⁻¹
  ![![d * det_inv, -2 * det_inv], ![3 * det_inv, a * det_inv]]

theorem trace_of_A (a d : ℝ) (h : A a d - A_inv a d = 0) : Matrix.trace (Fin 2) ℝ ℝ (A a d) = a + d := by
  -- Proof goes here
  sorry

end trace_of_A_l226_226244


namespace range_of_f_at_most_7_l226_226237

theorem range_of_f_at_most_7 (f : ℤ × ℤ → ℝ)
  (H : ∀ (x y m n : ℤ), f (x + 3 * m - 2 * n, y - 4 * m + 5 * n) = f (x, y)) :
  ∃ (s : Finset ℝ), s.card ≤ 7 ∧ ∀ (a : ℤ × ℤ), f a ∈ s :=
sorry

end range_of_f_at_most_7_l226_226237


namespace slope_of_line_determined_by_solutions_eq_l226_226390

theorem slope_of_line_determined_by_solutions_eq :
  ∀ (x y : ℝ), (4 / x + 5 / y = 0) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro x y h
  use -5 / 4
  sorry

end slope_of_line_determined_by_solutions_eq_l226_226390


namespace taco_beef_per_taco_l226_226014

open Real

theorem taco_beef_per_taco
  (total_beef : ℝ)
  (sell_price : ℝ)
  (cost_per_taco : ℝ)
  (profit : ℝ)
  (h1 : total_beef = 100)
  (h2 : sell_price = 2)
  (h3 : cost_per_taco = 1.5)
  (h4 : profit = 200) :
  ∃ (x : ℝ), x = 1/4 := 
by
  -- The proof will go here.
  sorry

end taco_beef_per_taco_l226_226014


namespace handshakes_total_l226_226032

def num_couples : ℕ := 15
def total_people : ℕ := 30
def men : ℕ := 15
def women : ℕ := 15
def youngest_man_handshakes : ℕ := 0
def men_handshakes : ℕ := (14 * 13) / 2
def men_women_handshakes : ℕ := 15 * 14

theorem handshakes_total : men_handshakes + men_women_handshakes = 301 :=
by
  -- Proof goes here
  sorry

end handshakes_total_l226_226032


namespace arithmetic_mean_midpoint_l226_226277

theorem arithmetic_mean_midpoint (a b : ℝ) : ∃ m : ℝ, m = (a + b) / 2 ∧ m = a + (b - a) / 2 :=
by
  sorry

end arithmetic_mean_midpoint_l226_226277


namespace total_animal_eyes_l226_226981

/-
Prove that the total number of animal eyes in the pond is 56,
given the conditions:
  - There are 18 snakes.
  - There are 10 alligators.
  - Each snake has 2 eyes.
  - Each alligator has 2 eyes.
-/

def snakes : Nat := 18
def alligators : Nat := 10
def eyes_per_snake : Nat := 2
def eyes_per_alligator : Nat := 2

theorem total_animal_eyes (snakes alligators eyes_per_snake eyes_per_alligator : Nat) :
  snakes = 18 -> 
  alligators = 10 -> 
  eyes_per_snake = 2 -> 
  eyes_per_alligator = 2 -> 
  snakes * eyes_per_snake + alligators * eyes_per_alligator = 56 := 
by
  intros hsnakes halligators heyes_per_snake heyes_per_alligator
  rw [hsnakes, halligators, heyes_per_snake, heyes_per_alligator]
  simp
  exact rfl

end total_animal_eyes_l226_226981


namespace sum_of_steps_digits_l226_226661

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Statement
theorem sum_of_steps_digits :
  ∃ n : ℕ, (Int.natAbs (n / 3).ceil - Int.natAbs (n / 4).ceil = 15) ∧ sum_of_digits n = 19 := sorry

end sum_of_steps_digits_l226_226661


namespace bob_smallest_number_l226_226020

theorem bob_smallest_number (n : ℕ) (alice_num : ℕ) (h₁ : alice_num = 30) 
  (h₂ : ∀ p, prime p → p ∣ alice_num → p ∣ n) (h₃ : 5 ∣ n) : n = 30 :=
sorry

end bob_smallest_number_l226_226020


namespace regular_pay_is_correct_l226_226605

def regular_week_days : ℕ := 5
def daily_work_hours : ℕ := 8
def overtime_pay_per_hour : ℝ := 3.20
def total_earnings_in_4weeks : ℝ := 432.0
def total_hours_worked_in_4weeks : ℕ := 175

def regular_hours_per_week (days : ℕ) (hours : ℕ) : ℕ := days * hours
def total_regular_hours_in_4weeks (weeks : ℕ) (hours_per_week : ℕ) : ℕ := weeks * hours_per_week

def overtime_hours (total_hours : ℕ) (regular_hours : ℕ) : ℕ := total_hours - regular_hours
def overtime_earnings (overtime_hours : ℕ) (overtime_pay : ℝ) : ℝ := overtime_hours * overtime_pay
def regular_earnings (total_earnings : ℝ) (overtime_earnings : ℝ) : ℝ := total_earnings - overtime_earnings

def pay_per_hour (regular_earnings : ℝ) (regular_hours : ℕ) : ℝ := regular_earnings / regular_hours

theorem regular_pay_is_correct :
  let weeks := 4 in
  let reg_hours_per_week := regular_hours_per_week regular_week_days daily_work_hours in
  let total_reg_hours := total_regular_hours_in_4weeks weeks reg_hours_per_week in
  let overtime_hours_worked := overtime_hours total_hours_worked_in_4weeks total_reg_hours in
  let overtime_earn := overtime_earnings overtime_hours_worked overtime_pay_per_hour in
  let reg_earn := regular_earnings total_earnings_in_4weeks overtime_earn in
  pay_per_hour reg_earn total_reg_hours = 2.40 :=
by
  sorry

end regular_pay_is_correct_l226_226605


namespace cost_per_meter_of_fencing_l226_226747

theorem cost_per_meter_of_fencing
  (area : ℕ)
  (ratio_length_width : ℕ × ℕ)
  (total_cost_rupees : ℝ) :
  (ratio_length_width = (3, 4)) →
  (area = 8748) →
  (total_cost_rupees = 94.5) →
  let total_cost_paise := total_cost_rupees * 100;
  let x := (area / (ratio_length_width.1 * ratio_length_width.2)).sqrt in
  let length := ratio_length_width.1 * x;
  let width := ratio_length_width.2 * x;
  let perimeter := 2 * (length + width) in
  let cost_per_meter := total_cost_paise / perimeter in
  cost_per_meter = 25 :=
by
  sorry

end cost_per_meter_of_fencing_l226_226747


namespace common_projection_same_l226_226503

-- Definitions of given vectors
def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (1, 5)

-- Definition of the projection function
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  let k := dot_uv / dot_vv
  (k * v.1, k * v.2)

-- The common projection p
noncomputable def p : ℝ × ℝ := (3, 3)

-- The proof statement
theorem common_projection_same (v : ℝ × ℝ) (hv : v ≠ (0, 0)) 
(hpa : proj a v = p) (hpb : proj b v = p) : p = (3, 3) :=
sorry

end common_projection_same_l226_226503


namespace determine_C_l226_226457

theorem determine_C (A B D : ℚ) (r1 r2 r3 r4 r5 r6 : ℚ)
  (h_pos_int : ∀ i, i ∈ {r1, r2, r3, r4, r5, r6} → i ∈ ℤ ∧ 1 ≤ i)
  (h_sum_15 : r1 + r2 + r3 + r4 + r5 + r6 = 15)
  (h_poly : ∀ z : ℚ, z^6 - 15*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36 = 
            (z - r1) * (z - r2) * (z - r3) * (z - r4) * (z - r5) * (z - r6)) :
  C = -92 :=
by
  sorry

end determine_C_l226_226457


namespace find_k_l226_226898

theorem find_k : ∃ k : ℝ, (∀ x y : ℝ, (x, y) = (4, -3) → 1 - k * x = -3 * y) ↔ k = -2 :=
by {
  use -2,
  intros x y h,
  cases h,
  simp,
  sorry
}

end find_k_l226_226898


namespace find_vertex_D_l226_226568

structure Point where
  x : ℤ
  y : ℤ

def vector_sub (a b : Point) : Point :=
  Point.mk (a.x - b.x) (a.y - b.y)

def vector_add (a b : Point) : Point :=
  Point.mk (a.x + b.x) (a.y + b.y)

def is_parallelogram (A B C D : Point) : Prop :=
  vector_sub B A = vector_sub D C

theorem find_vertex_D (A B C D : Point)
  (hA : A = Point.mk (-1) (-2))
  (hB : B = Point.mk 3 (-1))
  (hC : C = Point.mk 5 6)
  (hParallelogram: is_parallelogram A B C D) :
  D = Point.mk 1 5 :=
sorry

end find_vertex_D_l226_226568


namespace first_problem_second_problem_l226_226048

variable (x : ℝ)

-- Proof for the first problem
theorem first_problem : 6 * x^3 / (-3 * x^2) = -2 * x := by
sorry

-- Proof for the second problem
theorem second_problem : (2 * x + 3) * (2 * x - 3) - 4 * (x - 2)^2 = 16 * x - 25 := by
sorry

end first_problem_second_problem_l226_226048


namespace find_g3_l226_226308

noncomputable def g : ℝ → ℝ := sorry

theorem find_g3 (h : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = x) : g 3 = 1 :=
sorry

end find_g3_l226_226308


namespace range_of_m_l226_226522

noncomputable def f (x m : ℝ) : ℝ := x^3 - 3 * x + 2 + m

theorem range_of_m :
  ∃ (m : ℝ), 0 < m ∧ m < 3 + 4 * Real.sqrt 2 ∧
    ∃ (a b c : ℝ), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (0 ≤ c ∧ c ≤ 2) ∧
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
    let fa := f a m in
    let fb := f b m in
    let fc := f c m in
    fa^2 + fb^2 = fc^2 ∨ fa^2 + fc^2 = fb^2 ∨ fb^2 + fc^2 = fa^2 :=
begin
  sorry
end

end range_of_m_l226_226522


namespace int_solution_count_l226_226167

theorem int_solution_count :
  let count_solutions (eq : ℤ → ℤ → Bool) : Nat :=
    Finset.card (Finset.filter (λ ⟨(y, x)⟩, eq y x) Finset.univ.prod Finset.univ)
  count_solutions (λ y x, 6 * y^2 + 3 * y * x + x + 2 * y + 180 = 0) = 6 :=
sorry

end int_solution_count_l226_226167


namespace hexagon_planting_schemes_l226_226982

theorem hexagon_planting_schemes (n m : ℕ) (h : n = 4 ∧ m = 6) : 
  ∃ k, k = 732 := 
by sorry

end hexagon_planting_schemes_l226_226982


namespace percentage_increase_l226_226192

noncomputable def percentMoreThan (a b : ℕ) : ℕ :=
  ((a - b) * 100) / b

theorem percentage_increase (x y z : ℕ) (h1 : z = 300) (h2 : x = 5 * y / 4) (h3 : x + y + z = 1110) :
  percentMoreThan y z = 20 := by
  sorry

end percentage_increase_l226_226192


namespace graph_shifted_right_by_pi_over_12_l226_226330

def f (x : ℝ) : ℝ := Real.cos (2 * x)
def g (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)

theorem graph_shifted_right_by_pi_over_12 :
  ∀ x : ℝ, g x = f (x - (Real.pi / 12)) :=
by
  sorry

end graph_shifted_right_by_pi_over_12_l226_226330


namespace problem1_problem2_l226_226265

-- Definitions of the sets A and B
def set_A (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 4
def set_B (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

-- Problem 1: If A ∩ B ≠ ∅, find the range of a
theorem problem1 (a : ℝ) : (∃ x : ℝ, set_A x ∧ set_B x a) → a ≤ -1 / 2 ∨ a = 2 :=
sorry

-- Problem 2: If A ∩ B = B, find the value of a
theorem problem2 (a : ℝ) : (∀ x : ℝ, set_B x a → set_A x) → a ≤ -1 / 2 ∨ a ≥ 2 :=
sorry

end problem1_problem2_l226_226265


namespace area_ratio_of_smaller_octagon_l226_226714

theorem area_ratio_of_smaller_octagon (A B C D E F G H P Q R S T U V W : Point) 
  (h1 : is_regular_octagon A B C D E F G H)
  (h2 : midpoint A B = P) (h3 : midpoint B C = Q) (h4 : midpoint C D = R)
  (h5 : midpoint D E = S) (h6 : midpoint E F = T) (h7 : midpoint F G = U)
  (h8 : midpoint G H = V) (h9 : midpoint H A = W):
  area (octagon A B C D E F G H) / area (octagon P Q R S T U V W) = 4 := sorry

end area_ratio_of_smaller_octagon_l226_226714


namespace number_of_valid_pairing_ways_l226_226344

-- Define a natural number as a condition.
def is_natural (n : ℕ) : Prop := 0 < n

-- Define that 60 cards can be paired with the same modulus difference.
def pair_cards_same_modulus_difference (d : ℕ) (k : ℕ) : Prop :=
  60 = 2 * d * k

-- Define what it means for d to be a divisor of 30.
def is_divisor_of_30 (d : ℕ) : Prop :=
  ∃ k, 30 = d * k

theorem number_of_valid_pairing_ways :
  (finset.univ.filter is_divisor_of_30).card = 8 :=
begin
  sorry
end

end number_of_valid_pairing_ways_l226_226344


namespace intersection_product_PE_PF_l226_226613

noncomputable def curve1 : set (ℝ × ℝ) := {p | ∃ t : ℝ, p.1 = 4 * t^2 ∧ p.2 = 4 * t}
noncomputable def curve2 : set (ℝ × ℝ) := {p | p.1 - p.2 - 1 = 0}

theorem intersection_product_PE_PF :
  let points := {(x, y) | y^2 = 4 * x ∧ x - y - 1 = 0},
      mid_AB := (3, 2),
      perpendicular_bisector := λ (t : ℝ), (3 - (real.sqrt 2 / 2) * t, 2 + (real.sqrt 2 / 2) * t),
      intersects := {p : ℝ × ℝ | ∃ t, let q := perpendicular_bisector t in q.1 = 4 * p.2 ∧ q.2 = 4 * p.1}
  in ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ intersects t1 ∧ intersects t2 ∧ t1 * t2 = 16 := sorry

end intersection_product_PE_PF_l226_226613


namespace recent_quarter_revenue_l226_226019

theorem recent_quarter_revenue :
  let revenue_year_ago : Float := 69.0
  let percentage_decrease : Float := 30.434782608695656
  let decrease_in_revenue : Float := revenue_year_ago * (percentage_decrease / 100)
  let recent_quarter_revenue := revenue_year_ago - decrease_in_revenue
  recent_quarter_revenue = 48.0 := by
  sorry

end recent_quarter_revenue_l226_226019


namespace find_b_l226_226184

-- Define the variables and the given condition
variables (m c a b k : ℝ)
hypothesis (h : m = (c^2 * a * b) / (a - k * b))

-- State the theorem to be proved
theorem find_b (m c a k : ℝ) (h : m = (c^2 * a * b) / (a - k * b)) : 
  b = (m * a) / (c^2 * a + m * k) :=
by
  sorry

end find_b_l226_226184


namespace intersection_A_B_l226_226563

def A := {x : ℝ | x < -1 ∨ x > 1}
def B := {x : ℝ | Real.log x / Real.log 2 > 0}

theorem intersection_A_B:
  A ∩ B = {x : ℝ | x > 1} :=
by
  sorry

end intersection_A_B_l226_226563


namespace polynomial_evaluation_l226_226044

theorem polynomial_evaluation :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 :=
by
  sorry

end polynomial_evaluation_l226_226044


namespace binomial_expansion_value_calculation_result_final_result_l226_226041

theorem binomial_expansion_value :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = (7 + 1)^4 := 
sorry

theorem calculation_result :
  (7 + 1)^4 = 4096 := 
sorry

theorem final_result :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by
  calc
    7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = (7 + 1)^4 := binomial_expansion_value
    ... = 4096 := calculation_result

end binomial_expansion_value_calculation_result_final_result_l226_226041


namespace find_x_l226_226441

theorem find_x
  (x : ℝ)
  (h1 : (x - 2)^2 + (15 - 5)^2 = 13^2)
  (h2 : x > 0) : 
  x = 2 + Real.sqrt 69 :=
sorry

end find_x_l226_226441


namespace vika_card_pairs_l226_226369

theorem vika_card_pairs : 
  let numbers := finset.range 61 \ finset.singleton 0 in
  let divs := {d | d ∈ finset.divisors 30} in
  numbers.card = 60 →
  ∀ d ∈ divs, ∀ pair : finset (ℕ × ℕ),
    pair.card = 30 →
    finset.forall₂ pair (λ x y, |x.1 - x.2| % d = |y.1 - y.2| % d) → 
    ∃ (number_of_pairs : ℕ), number_of_pairs = 8 :=
by 
  intro numbers divs hc hd hp hpairs,
  sorry

end vika_card_pairs_l226_226369


namespace number_of_divisors_not_divisible_by_5_or_3_l226_226965

-- Definition of the number 300 and its prime factorization
def n : ℕ := 300
def prime_factorization (n : ℕ) : Prop :=
  (∃ a b c : ℕ, n = 2^a * 3^b * 5^c) ∧ (0 ≤ a) ∧ (a ≤ 2) ∧ (0 ≤ b) ∧ (b ≤ 1) ∧ (0 ≤ c) ∧ (c ≤ 2)

-- The main statement about the number of positive divisors of 300 that are not divisible by 5 or 3
theorem number_of_divisors_not_divisible_by_5_or_3 :
  (∃ count : ℕ, count = 3 ∧ ∀ d : ℕ,
    d ∣ n → (¬ (5 ∣ d ∨ 3 ∣ d)) → count = 3) := sorry

end number_of_divisors_not_divisible_by_5_or_3_l226_226965


namespace sqrt_domain_l226_226587

theorem sqrt_domain (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (x - 2)) ↔ (x ≥ 2) :=
by
  -- proof goes here
  sorry

end sqrt_domain_l226_226587


namespace N_is_midpoint_CD_l226_226828

-- Define the necessary geometry types and concepts
variables {Point : Type} [InnerProductSpace ℝ Point]
variables (A B C D M N : Point)
variables (AB CD : Line Point)
variables (adn mdn bcn mnc : Angle Point)

-- Define the conditions
def is_midpoint (P Q R : Point) : Prop := dist P Q = dist P R
def is_parallel (L1 L2 : Line Point) : Prop := ∀ {p q : Point}, p ∈ L1 → q ∈ L2 → InnerProductSpace.angle p q = 0
def Angle_A (P Q R : Point) : Angle Point := sorry -- Angle definition
def Angle_B (P Q R : Point) : Angle Point := sorry -- Angle definition

-- The proof problem statement
theorem N_is_midpoint_CD
  (h1 : AB ∥ CD)
  (h2 : is_midpoint M A B)
  (h3 : N ∈ segment CD)
  (h4 : Angle_A A D N = (1 / 2) * Angle_B M N C)
  (h5 : Angle_A B C N = (1 / 2) * Angle_B M N D) : is_midpoint N C D :=
sorry

end N_is_midpoint_CD_l226_226828


namespace cube_parallel_edge_pairs_l226_226576

theorem cube_parallel_edge_pairs : 
  (∀ (c : Cube), cube.edges c = 12 → cube.dimensions c = 3 → total_parallel_edge_pairs c = 18) := 
by sorry

end cube_parallel_edge_pairs_l226_226576


namespace root_of_quadratic_eq_l226_226089

open Complex

theorem root_of_quadratic_eq :
  ∃ z1 z2 : ℂ, (z1 = 3.5 - I) ∧ (z2 = -2.5 + I) ∧ (∀ z : ℂ, z^2 - z = 6 - 6 * I → (z = z1 ∨ z = z2)) := 
sorry

end root_of_quadratic_eq_l226_226089


namespace trajectory_of_P_l226_226187

noncomputable def distance (P Q : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def distance_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * P.1 + b * P.2 + c) / real.sqrt (a^2 + b^2)

theorem trajectory_of_P 
  (P : ℝ × ℝ)
  (h : distance P (2, 0) + 1 = distance_to_line P 1 0 3) :
  P.2^2 = 8 * P.1 :=
by
  sorry

end trajectory_of_P_l226_226187


namespace integer_solutions_count_l226_226170

theorem integer_solutions_count :
  let eq : Int -> Int -> Int := fun x y => 6 * y ^ 2 + 3 * x * y + x + 2 * y - 72
  ∃ (sols : List (Int × Int)), 
    (∀ x y, eq x y = 0 → (x, y) ∈ sols) ∧
    (∀ p ∈ sols, ∃ x y, p = (x, y) ∧ eq x y = 0) ∧
    sols.length = 4 :=
by
  sorry

end integer_solutions_count_l226_226170


namespace area_triangle_ABG_l226_226537

theorem area_triangle_ABG :
  ∀ (A B C D E F G : ℝ×ℝ) (h1 h2 h3 h4 h5 h6 h7 h8: Prop),
    (A = (0, 0)) →
    (D = (8, 0)) →
    (C = (8, 6)) →
    (B = (0, 6)) →
    -- E is the midpoint of AD
    (E = ((A.1 + D.1) / 2, 0)) →
    -- F is on BC such that BF = 2/3 * BC
    (F = ((2/3) * C.1, C.2)) →
    -- G is a point such that the area of triangle DEG is equal to the area of triangle CFG
    (∃ h : ℝ, G = (D.1, h) ∧ 2 * h = 8 - (4 * (6 - h) / 3)) →
    let area (x1 y1 x2 y2 x3 y3 : ℝ) := 0.5 * ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))) in
    area (A.1) (A.2) (B.1) (B.2) (G.1) (G.2) = 24 :=
by
  intros A B C D E F G h1 h2 h3 h4 h5 h6 h7 h8 hA hD hC hB hE hF hG area
  simp at hE hF hG
  sorry -- proof not required

end area_triangle_ABG_l226_226537


namespace gcd_of_75_and_360_l226_226878

theorem gcd_of_75_and_360 : Nat.gcd 75 360 = 15 := by
  sorry

end gcd_of_75_and_360_l226_226878


namespace marked_price_percentage_l226_226452

variables (L M: ℝ)

-- The store owner purchases items at a 25% discount of the list price.
def cost_price (L : ℝ) := 0.75 * L

-- The store owner plans to mark them up such that after a 10% discount on the marked price,
-- he achieves a 25% profit on the selling price.
def selling_price (M : ℝ) := 0.9 * M

-- Given condition: cost price is 75% of selling price
theorem marked_price_percentage (h : cost_price L = 0.75 * selling_price M) : 
  M = 1.111 * L :=
by 
  sorry

end marked_price_percentage_l226_226452


namespace derivative_odd_function_l226_226266

theorem derivative_odd_function (a b c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^3 + b * x^2 + c * x + 2) 
    (h_deriv_odd : ∀ x, deriv f (-x) = - deriv f x) : a^2 + c^2 ≠ 0 :=
by
  sorry

end derivative_odd_function_l226_226266


namespace function_solution_l226_226499

theorem function_solution (f : ℝ → ℝ) (H : ∀ x y : ℝ, 1 < x → 1 < y → f x - f y = (y - x) * f (x * y)) :
  ∃ k : ℝ, ∀ x : ℝ, 1 < x → f x = k / x :=
by
  sorry

end function_solution_l226_226499


namespace max_value_of_f_l226_226881

def f (x : ℝ) : ℝ := cos (2 * x) + 5 * cos (π / 2 - x)

theorem max_value_of_f : ∃ x : ℝ, (f x = 4) ∧ (∀ y : ℝ, f y ≤ 4) :=
sorry

end max_value_of_f_l226_226881


namespace ordered_triples_count_l226_226448

theorem ordered_triples_count : 
  let b := 3003
  let side_length_squared := b * b
  let num_divisors := (2 + 1) * (2 + 1) * (2 + 1) * (2 + 1)
  let half_divisors := num_divisors / 2
  half_divisors = 40 := by
  sorry

end ordered_triples_count_l226_226448


namespace basketball_shooting_l226_226197

theorem basketball_shooting (missed_shots : ℕ → Prop)
  (h_missed_shots : ∀ i, i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → missed_shots i)
  (h_miss_count : 2 = (set.univ.filter missed_shots).card) :
  ¬ ∃ scores : ℕ, scores = 35 ∧ scores = 55 - (∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.filter (λ i, ¬ missed_shots i)) := by
  sorry

end basketball_shooting_l226_226197


namespace ratio_A_B_l226_226062

noncomputable def A : ℝ :=
  ∑' (n : ℕ) in ({ n | n % 6 ≠ 0 ∧ n % 2 = 0 }).filter (λ n, even n), (1:ℝ) / (n ^ 2)

noncomputable def B : ℝ :=
  ∑' (k : ℕ), (1:ℝ) / (6 * (k + 1)) ^ 2 * (-1) ^ k

theorem ratio_A_B :
  A / B = 37 :=
sorry

end ratio_A_B_l226_226062


namespace exists_12x12_square_l226_226989

noncomputable def real_board_60_60 (board : ℕ → ℕ → ℝ) : Prop :=
  (∀ i j, |board i j| ≤ 1) ∧
  (∑ i, ∑ j, board i j = 600)

theorem exists_12x12_square 
  (board : ℕ → ℕ → ℝ)
  (hboard : real_board_60_60 board) :
  ∃ (a b : ℕ) (s : ℝ), a ≤ 48 ∧ b ≤ 48 ∧
  (s = ∑ i in finset.range 12, ∑ j in finset.range 12, board (a + i) (b + j)) ∧
  |s| ≤ 24 :=
sorry

end exists_12x12_square_l226_226989


namespace find_p_fulfill_condition_l226_226544

noncomputable def P_fulfills_angle_condition (p : ℝ) : Prop :=
  ∀ (A B : ℝ × ℝ), A ≠ B → (A.1^2 / 4 + A.2^2 = 1) → (B.1^2 / 4 + B.2^2 = 1) → 
  (∃ F : ℝ × ℝ, F = (Real.sqrt 3, 0) ∧ A.x = Real.sqrt 3 ∧ B.x = Real.sqrt 3) →
  let P := (p, 0) in
  (angle A P F = angle B P F)

theorem find_p_fulfill_condition : P_fulfills_angle_condition (Real.sqrt 3) :=
sorry

end find_p_fulfill_condition_l226_226544


namespace card_paiting_modulus_l226_226363

theorem card_paiting_modulus (cards : Finset ℕ) (H : cards = Finset.range 61 \ {0}) :
  ∃ d : ℕ, ∀ n ∈ cards, ∃! k, (∀ x ∈ cards, (x + n ≡ k [MOD d])) ∧ (d ∣ 30) ∧ (∃! n : ℕ, 1 ≤ n ∧ n ≤ 8) :=
sorry

end card_paiting_modulus_l226_226363


namespace matchsticks_20th_stage_l226_226752

theorem matchsticks_20th_stage :
  let a1 := 3
  let d := 3
  let a20 := a1 + 19 * d
  a20 = 60 := by
  sorry

end matchsticks_20th_stage_l226_226752


namespace fish_population_l226_226198

theorem fish_population (N : ℕ) (hN : (2 / 50 : ℚ) = (30 / N : ℚ)) : N = 750 :=
by
  sorry

end fish_population_l226_226198


namespace prime_factors_correct_sum_of_largest_and_smallest_prime_factors_l226_226778

/-- The number we are considering is 1365. -/
def number := 1365

/-- Hypothesis: The prime factors of 1365 are exactly 3, 5, 7, and 13. -/
def prime_factors_of_1365 : List ℕ := [3, 5, 7, 13]

/-- Hypothesis: The prime factors are prime numbers. -/
theorem prime_factors_correct (n : ℕ) (hn : n ∈ prime_factors_of_1365) : Nat.Prime n :=
begin
  rw prime_factors_of_1365 at hn,
  fin_cases hn,
  { exact Nat.prime_of_nat 3 },
  { exact Nat.prime_of_nat 5 },
  { exact Nat.prime_of_nat 7 },
  { exact Nat.prime_of_nat 13 },
end

/-- We state the final proof we need to show that the sum is 16. -/
theorem sum_of_largest_and_smallest_prime_factors : 
  let min_prime := List.minimum prime_factors_of_1365
  let max_prime := List.maximum prime_factors_of_1365
  min_prime + max_prime = 16 :=
begin
  sorry
end

end prime_factors_correct_sum_of_largest_and_smallest_prime_factors_l226_226778


namespace vika_card_pairs_l226_226371

theorem vika_card_pairs : 
  let numbers := finset.range 61 \ finset.singleton 0 in
  let divs := {d | d ∈ finset.divisors 30} in
  numbers.card = 60 →
  ∀ d ∈ divs, ∀ pair : finset (ℕ × ℕ),
    pair.card = 30 →
    finset.forall₂ pair (λ x y, |x.1 - x.2| % d = |y.1 - y.2| % d) → 
    ∃ (number_of_pairs : ℕ), number_of_pairs = 8 :=
by 
  intro numbers divs hc hd hp hpairs,
  sorry

end vika_card_pairs_l226_226371


namespace remainder_T_mod_1000_l226_226436

def T_m (m : ℕ) : ℕ :=
  match m with
  | 1 => 1
  | 2 => 3 * T_m 1
  | 3 => 3 * T_m 2
  | 4 => 3 * T_m 3
  | _ => if m < 4 then T_m (m - 1) else 4 * T_m (m - 1)

def T : ℕ := T_m 9

theorem remainder_T_mod_1000 : T % 1000 = 296 := by
  sorry

end remainder_T_mod_1000_l226_226436


namespace vika_pairs_exactly_8_ways_l226_226350

theorem vika_pairs_exactly_8_ways :
  ∃ d : ℕ, (d ∣ 30) ∧ (Finset.card (Finset.filter (λ d, d ∣ 30) (Finset.range 31)) = 8) := 
sorry

end vika_pairs_exactly_8_ways_l226_226350


namespace vika_card_pairs_l226_226370

theorem vika_card_pairs : 
  let numbers := finset.range 61 \ finset.singleton 0 in
  let divs := {d | d ∈ finset.divisors 30} in
  numbers.card = 60 →
  ∀ d ∈ divs, ∀ pair : finset (ℕ × ℕ),
    pair.card = 30 →
    finset.forall₂ pair (λ x y, |x.1 - x.2| % d = |y.1 - y.2| % d) → 
    ∃ (number_of_pairs : ℕ), number_of_pairs = 8 :=
by 
  intro numbers divs hc hd hp hpairs,
  sorry

end vika_card_pairs_l226_226370


namespace percentage_gain_loss_l226_226442

theorem percentage_gain_loss (price : ℝ) (gain_loss_amount : ℝ) (x : ℝ) :
  price = 675958 → gain_loss_amount = 1.96 →
  x = (gain_loss_amount / (2 * price)) * 100 →
  x ≈ 0.0145 :=
by
  intros
  sorry

end percentage_gain_loss_l226_226442


namespace custom_op_evaluation_l226_226581

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_evaluation : custom_op 6 5 - custom_op 5 6 = -4 := by
  sorry

end custom_op_evaluation_l226_226581


namespace integer_solutions_of_equation_l226_226161

theorem integer_solutions_of_equation:
  (number_of_int_solutions (λ x y, 6 * y^2 + 3 * x * y + x + 2 * y + 180) = 6) :=
begin
  sorry
end

end integer_solutions_of_equation_l226_226161


namespace at_least_one_positive_l226_226246

theorem at_least_one_positive (x y z : ℝ) :
  let a := x^2 - 2*x + Real.pi / 2
  let b := y^2 - 2*y + Real.pi / 3
  let c := z^2 - 2*z + Real.pi / 6
  in a > 0 ∨ b > 0 ∨ c > 0 := by
  sorry

end at_least_one_positive_l226_226246


namespace probability_arithmetic_sequence_l226_226900

-- Define the problem conditions
def set_of_numbers := {n : ℕ | 1 ≤ n ∧ n ≤ 20}
def is_arithmetic_sequence (a b c : ℕ) : Prop := a + c = 2 * b
def is_valid_subset (s : Set ℕ) := s ⊆ set_of_numbers ∧ s.card = 3

theorem probability_arithmetic_sequence :
  (∑ s in (Finset.powersetLen 3 (Finset.range 21)), 
    if is_arithmetic_sequence s else 0)
  / (Finset.card (Finset.powersetLen 3 (Finset.range 21))) = (1 : ℚ) / 38 :=
  sorry

end probability_arithmetic_sequence_l226_226900


namespace verify_equation_l226_226646

noncomputable def sides_of_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def inradius (a b c : ℝ) (s : ℝ) : ℝ :=
  √(s * (s - a) * (s - b) * (s - c)) / s

noncomputable def circumradius (a b c : ℝ) (K : ℝ) : ℝ :=
  (a * b * c) / (4 * K)

theorem verify_equation (a b c r R : ℝ) (h : sides_of_triangle a b c)
  (p : ℝ := semiperimeter a b c) :
  (a * b * c = 4 * R * r * p) ∧
  (r^2 + p^2 + 4 * R * r = (a * b + b * c + c * a)) ∧
  (p * r ≠ 0) →
  (1 / (a * b) + 1 / (b * c) + 1 / (c * a) = 1 / (2 * r * R)) :=
sorry

end verify_equation_l226_226646


namespace monotonic_decreasing_interval_l226_226735

def f (x : ℝ) : ℝ := Real.log (x^2 - x - 2)

def t (x : ℝ) : ℝ := x^2 - x - 2

theorem monotonic_decreasing_interval :
  {x : ℝ | t x > 0} = {x : ℝ | x < -1} ∨ {x : ℝ | x > 2} →
  ∀ x1 x2 : ℝ, x1 < x2 → f(x2) < f(x1) ↔ x1 < -1 :=
sorry

end monotonic_decreasing_interval_l226_226735


namespace hexagon_vertex_to_center_length_l226_226086

theorem hexagon_vertex_to_center_length :
  ∀ (s : ℝ), s = 12 → let altitude := (s * real.sqrt 3) / 2 in
  let median := (2 * altitude) / 3 in
  median = 4 * real.sqrt 3 :=
by
  intros s hs
  let altitude := (s * real.sqrt 3) / 2
  let median := (2 * altitude) / 3
  rw hs at *
  have haltitude : altitude = 6 * real.sqrt 3 := by
    rw [altitude, mul_div_cancel' _ (two_ne_zero : (2 : ℝ) ≠ 0)]
    norm_num
  rw haltitude at median
  norm_num [median]
  sorry

end hexagon_vertex_to_center_length_l226_226086


namespace opposite_terminal_sides_l226_226597

theorem opposite_terminal_sides (α β : ℝ) (k : ℤ) (h : ∃ k : ℤ, α = β + 180 + k * 360) :
  α = β + 180 + k * 360 :=
by sorry

end opposite_terminal_sides_l226_226597


namespace dice_sum_is_4_l226_226756

-- Defining the sum of points obtained from two dice rolls
def sum_of_dice (a b : ℕ) : ℕ := a + b

-- The main theorem stating the condition we need to prove
theorem dice_sum_is_4 (a b : ℕ) (h : sum_of_dice a b = 4) :
  (a = 3 ∧ b = 1) ∨ (a = 1 ∧ b = 3) ∨ (a = 2 ∧ b = 2) :=
sorry

end dice_sum_is_4_l226_226756


namespace det_abs_eq_one_l226_226629

variable {n : ℕ}
variable {A : Matrix (Fin n) (Fin n) ℤ}
variable {p q r : ℕ}
variable (hpq : p^2 = q^2 + r^2)
variable (hodd : Odd r)
variable (hA : p^2 • A ^ p^2 = q^2 • A ^ q^2 + r^2 • 1)

theorem det_abs_eq_one : |A.det| = 1 := by
  sorry

end det_abs_eq_one_l226_226629


namespace smile_area_eq_2pi_l226_226823

-- Geometry setup
structure Point :=
(x : ℝ)
(y : ℝ)

def radiusAB : ℝ := 2
def centerC : Point := ⟨0, 0⟩
def pointD : Point := ⟨1, 1⟩ -- Since CD = 1 and it is perpendicular to AB, D is at (1, 1)

-- Assume E and F are on the circle when CD extends fully
def pointE : Point := ⟨2, 2⟩ -- Placeholder
def pointF : Point := ⟨-2, 2⟩ -- Placeholder

-- The problem to prove
theorem smile_area_eq_2pi :
  let semicircle_area (r : ℝ) := (1 / 2) * π * r^2 in
  let full_circle_area (r : ℝ) := π * r^2 in
  semicircle_area radiusAB - semicircle_area radiusAB = 2π :=
by
  sorry

end smile_area_eq_2pi_l226_226823


namespace smallest_x_l226_226399

theorem smallest_x (x : ℕ) (h₁ : x % 3 = 2) (h₂ : x % 4 = 3) (h₃ : x % 5 = 4) : x = 59 :=
by
  sorry

end smallest_x_l226_226399


namespace slope_of_given_line_l226_226384

theorem slope_of_given_line : ∀ (x y : ℝ), (4 / x + 5 / y = 0) → (y = (-5 / 4) * x) := 
by 
  intros x y h
  sorry

end slope_of_given_line_l226_226384


namespace seventh_term_in_geometric_sequence_l226_226884

-- Define the geometric sequence conditions
def first_term : ℝ := 3
def second_term : ℝ := -1/2
def common_ratio : ℝ := second_term / first_term

-- Define the formula for the nth term of the geometric sequence
def nth_term (a r : ℝ) (n : ℕ) : ℝ := a * r^(n-1)

-- The Lean statement for proving the seventh term in the geometric sequence
theorem seventh_term_in_geometric_sequence :
  nth_term first_term common_ratio 7 = 1 / 15552 :=
by
  -- The proof is to be filled in.
  sorry

end seventh_term_in_geometric_sequence_l226_226884


namespace frequency_of_8th_group_l226_226494

theorem frequency_of_8th_group :
  let sample_size := 100
  let freq1 := 15
  let freq2 := 17
  let freq3 := 11
  let freq4 := 13
  let freq_5_to_7 := 0.32 * sample_size
  let total_freq_1_to_4 := freq1 + freq2 + freq3 + freq4
  let remaining_freq := sample_size - total_freq_1_to_4
  let freq8 := remaining_freq - freq_5_to_7
  (freq8 / sample_size = 0.12) :=
by
  sorry

end frequency_of_8th_group_l226_226494


namespace integer_roots_of_polynomial_l226_226876

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, (x^3 - 3 * x^2 - 13 * x + 15 = 0) → (x = -3 ∨ x = 1 ∨ x = 5) :=
by
  sorry

end integer_roots_of_polynomial_l226_226876


namespace sum_real_imag_z_l226_226153

noncomputable def z1 : ℂ := 2 * complex.I
noncomputable def z2 : ℂ := 1 - complex.I
noncomputable def z : ℂ := z1 / conj z2

theorem sum_real_imag_z : z.re + z.im = 2 :=
  sorry

end sum_real_imag_z_l226_226153


namespace constant_term_in_expansion_l226_226687

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := (nat.factorial n) / (nat.factorial k * nat.factorial (n - k))

theorem constant_term_in_expansion : 
  let expr := (fun (x : ℚ) => (x^2 + 1/x)^6) in 
  is_constant_term (expr 1) 15 := 
begin
  sorry
end

end constant_term_in_expansion_l226_226687


namespace smallest_x_l226_226404

theorem smallest_x (x : ℕ) : (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) → x = 59 :=
by
  intro h
  sorry

end smallest_x_l226_226404


namespace find_x_l226_226080

theorem find_x (x : ℝ) (hx_nonneg : 0 ≤ x) (h : (⌊x⌋₊ : ℝ) * x = 108) : x = 10.8 :=
by
  -- Proof goes here
  sorry

end find_x_l226_226080


namespace find_a_8_l226_226991

variable {α : Type*} [LinearOrderedField α]
variables (a : ℕ → α) (n : ℕ)

-- Definition of an arithmetic sequence
def is_arithmetic_seq (a : ℕ → α) := ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition
def given_condition (a : ℕ → α) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- Main theorem to prove
theorem find_a_8 (h_arith : is_arithmetic_seq a) (h_cond : given_condition a) : a 8 = 24 :=
  sorry

end find_a_8_l226_226991


namespace power_multiplication_l226_226037

theorem power_multiplication :
  2^4 * 5^4 = 10000 := 
by
  sorry

end power_multiplication_l226_226037


namespace katie_books_ratio_l226_226061

theorem katie_books_ratio
  (d : ℕ)
  (k : ℚ)
  (g : ℕ)
  (total_books : ℕ)
  (hd : d = 6)
  (hk : ∃ k : ℚ, k = (k : ℚ))
  (hg : g = 5 * (d + k * d))
  (ht : total_books = d + k * d + g)
  (htotal : total_books = 54) :
  k = 1 / 2 :=
by
  sorry

end katie_books_ratio_l226_226061


namespace find_y_l226_226180

theorem find_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := 
by 
  sorry

end find_y_l226_226180


namespace mod_pow_equiv_one_l226_226279

theorem mod_pow_equiv_one (k m α : ℤ) : (1 + k * m)^(m^(α - 1)) ≡ 1 [ZMOD m^α] :=
by
  sorry

end mod_pow_equiv_one_l226_226279


namespace int_solution_count_l226_226164

theorem int_solution_count :
  let count_solutions (eq : ℤ → ℤ → Bool) : Nat :=
    Finset.card (Finset.filter (λ ⟨(y, x)⟩, eq y x) Finset.univ.prod Finset.univ)
  count_solutions (λ y x, 6 * y^2 + 3 * y * x + x + 2 * y + 180 = 0) = 6 :=
sorry

end int_solution_count_l226_226164


namespace cheryl_walking_speed_l226_226480

theorem cheryl_walking_speed (H : 12 = 6 * v) : v = 2 := 
by
  -- proof here
  sorry

end cheryl_walking_speed_l226_226480


namespace spent_on_burgers_l226_226578

noncomputable def money_spent_on_burgers (total_allowance : ℝ) (movie_fraction music_fraction ice_cream_fraction : ℝ) : ℝ :=
  let movie_expense := (movie_fraction * total_allowance)
  let music_expense := (music_fraction * total_allowance)
  let ice_cream_expense := (ice_cream_fraction * total_allowance)
  total_allowance - (movie_expense + music_expense + ice_cream_expense)

theorem spent_on_burgers : 
  money_spent_on_burgers 50 (1/4) (3/10) (2/5) = 2.5 :=
by sorry

end spent_on_burgers_l226_226578


namespace total_depreciation_correct_residual_value_correct_sales_price_correct_l226_226418

-- Definitions and conditions
def initial_cost := 500000
def max_capacity := 100000
def jul_bottles := 200
def aug_bottles := 15000
def sep_bottles := 12300

def depreciation_per_bottle := initial_cost / max_capacity

-- Part (a)
def total_depreciation_jul := jul_bottles * depreciation_per_bottle
def total_depreciation_aug := aug_bottles * depreciation_per_bottle
def total_depreciation_sep := sep_bottles * depreciation_per_bottle
def total_depreciation := total_depreciation_jul + total_depreciation_aug + total_depreciation_sep

theorem total_depreciation_correct :
  total_depreciation = 137500 := 
by sorry

-- Part (b)
def residual_value := initial_cost - total_depreciation

theorem residual_value_correct :
  residual_value = 362500 := 
by sorry

-- Part (c)
def desired_profit := 10000
def sales_price := residual_value + desired_profit

theorem sales_price_correct :
  sales_price = 372500 := 
by sorry

end total_depreciation_correct_residual_value_correct_sales_price_correct_l226_226418


namespace largest_m_exists_l226_226915

variable {α : Type*} [linear_ordered_field α]

def f (x : α) := (1/4 : α) * x^2 + (1/2 : α) * x - (1/4 : α)

theorem largest_m_exists
  (h1 : ∀ x : α, f (x - 4) = f (2 - x))
  (h2 : ∀ x : α, f x ≥ x)
  (h3 : ∀ x : α, x ∈ Ioo 0 2 → f x ≤ ((x + 1) / 2)^2)
  (h4 : ∃ y : α, ∀ x : α, f x ≥ f y ∧ (∀ z : α, f z ≠ 0 → f y = 0)) :
  ∃ t : α, ∀ x : α, x ∈ Icc 1 9 → f (x + t) ≤ x :=
sorry

end largest_m_exists_l226_226915


namespace calculate_simple_interest_l226_226659

-- Defining the conditions based on the problem

-- Principal amount (P), Rate of interest (r) and Time period (t) are required
-- We know that:
-- 1. Compound Interest (C.I.) after 2 years is $693
-- 2. Rate of interest (r) is 10%

-- Let P be the principal amount and A be the amount after the interest is applied
-- Time period (t) is 2 years

def P : ℝ := 693 / 0.21  -- From solution step, P = 693 / 0.21

def r : ℝ := 10

def t : ℕ := 2

def CI : ℝ := 693

noncomputable def SI := (P * r * t) / 100

-- Problem statement in Lean
theorem calculate_simple_interest :
  SI = 660 := sorry

end calculate_simple_interest_l226_226659


namespace monotonic_intervals_axis_of_symmetry_center_of_symmetry_l226_226141

def f (x : ℝ) : ℝ := sqrt 2 * Real.cos (4 * x - π / 4) + 1

theorem monotonic_intervals :
  (∀ k : ℤ, ∀ x, (π / 16 + k * π / 2 <= x ∧ x <= 5 * π / 16 + k * π / 2) → decreasing_on f {x | π / 16 + k * π / 2 <= x ∧ x <= 5 * π / 16 + k * π / 2}) ∧
  (∀ k : ℤ, ∀ x, (5 * π / 16 + k * π / 2 <= x ∧ x <= 9 * π / 16 + k * π / 2) → increasing_on f {x | 5 * π / 16 + k * π / 2 <= x ∧ x <= 9 * π / 16 + k * π / 2}) :=
sorry

theorem axis_of_symmetry :
  ∀ k : ℤ, ∀ x, x = π / 16 + k * π / 4 ↔ is_axis_of_symmetry f x :=
sorry

theorem center_of_symmetry :
  ∀ k : ℤ, ∀ (a : ℝ), a = (3 * π / 16 + k * π / 4, 1) ↔ is_center_of_symmetry f a :=
sorry

end monotonic_intervals_axis_of_symmetry_center_of_symmetry_l226_226141


namespace area_triangle_N1N2N3_l226_226999

variables {A B C D E F N1 N2 N3 : Type}

-- Assume segments CD, AE, and BF are all 1/3 of their respective sides
variable (hCD : CD = (1 / 3) * BC)
variable (hAE : AE = (1 / 3) * AC)
variable (hBF : BF = (1 / 3) * AB)

-- Assume the segment ratio AN_2 : N_2N_1 : N_1D = 3 : 3 : 1
variable (hRatio : AN2 : N2N1 : N1D = 3 : 3 : 1)

-- Let's assume the area notation
variable (S_ABC : ℝ)
variable (S_triangle : Triangle → ℝ)

-- Definitions for triangles
axiom S_AE : S_triangle (triangle A E CD) = (1 / 3) * S_ABC
axiom S_BF : S_triangle (triangle B F AB) = (1 / 3) * S_ABC
axiom S_CD : S_triangle (triangle C D BC) = (1 / 3) * S_ABC

-- The areas of triangles \triangle CDN_1, \triangle AEV_2, \triangle BFV_3
axiom S_CDN1 : S_triangle (triangle C D N1) = (1 / 21) * S_ABC
axiom S_AEV2 : S_triangle (triangle A E V2) = (1 / 21) * S_ABC
axiom S_BFV3 : S_triangle (triangle B F V3) = (1 / 21) * S_ABC

-- Main statement to prove the area of \triangle N_1N_2N_3
theorem area_triangle_N1N2N3 (S_ABC : ℝ) :
  S_triangle (triangle N1 N2 N3) = (1 / 7) * S_ABC :=
sorry

end area_triangle_N1N2N3_l226_226999


namespace probability_symmetric_interval_l226_226946

namespace NormalDistributionProof

open ProbabilityTheory

noncomputable def normal_distribution (μ σ : ℝ) : measure ℝ :=
  measure_theory.measure_Gaussian μ (σ^2)

variables {σ : ℝ} (hpos : 0 < σ)

theorem probability_symmetric_interval
  (h : ∀ x, (measure_theory.measure_Gaussian 0 (σ^2)).to_outer_measure.caratheodory.guard_set 
    -∅ {y : ℝ | y > 2} = 0.023) : 
  (measure_theory.measure_Gaussian 0 (σ^2)).to_outer_measure.caratheodory.guard_set -∅ 
    {y : ℝ | -2 ≤ y ∧ y ≤ 2} = 0.954 := 
by
  sorry

end NormalDistributionProof

end probability_symmetric_interval_l226_226946


namespace smallest_x_l226_226395

open Classical
noncomputable theory

def conditions (x : ℕ) : Prop :=
  x % 3 = 2 ∧ x % 4 = 3 ∧ x % 5 = 4

theorem smallest_x : ∃ (x : ℕ), conditions x ∧ (∀ (y : ℕ), conditions y → x ≤ y) ∧ x = 59 :=
by {
  sorry
}

end smallest_x_l226_226395


namespace y_percent_of_x_equals_1250_l226_226278

noncomputable def x : ℕ := 1000
def y : ℕ := 125

theorem y_percent_of_x_equals_1250 (hx : y = 125) (hy : y = 0.125 * x) : 
  (y * x) / 100 = 1250 := 
sorry

end y_percent_of_x_equals_1250_l226_226278


namespace find_integer_roots_l226_226874

open Int Polynomial

def P (x : ℤ) : ℤ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_integer_roots : {x : ℤ | P x = 0} = {-3, 1, 5} := by
  sorry

end find_integer_roots_l226_226874


namespace error_estimate_first_three_terms_error_estimate_first_four_terms_l226_226950

noncomputable def alternating_series := ∑ n in (Set.univ : Set ℕ), (-1)^(n+1) * (1/(n^2))

def partial_sum (n : ℕ) : ℚ :=
∑ k in Finset.range n, (-1)^(k + 1) * (1/(k + 1)^2)

def error_term (n : ℕ) : ℚ :=
alternating_series - partial_sum n

theorem error_estimate_first_three_terms :
  partial_sum 3 = 31 / 36 ∧ (-1/16 < error_term 3 ∧ error_term 3 < 0) := by
  sorry

theorem error_estimate_first_four_terms :
  partial_sum 4 = 115 / 144 ∧ (0 < error_term 4 ∧ error_term 4 < 1/25) := by
  sorry

end error_estimate_first_three_terms_error_estimate_first_four_terms_l226_226950


namespace find_x_l226_226215

-- Definition of x, the problem parameters, and the proof goal
theorem find_x (x : ℝ) (h1 : 9 * x * x + 36 * x * x + 9 * x * x = 1000) : 
  x = 10 * real.sqrt 3 / 3 :=
begin
  sorry,
end

end find_x_l226_226215


namespace angle_A_condition_1_angle_A_condition_2_angle_A_condition_3_triangle_area_l226_226941

noncomputable def area_of_triangle (a b c : ℝ) := (1/4) * real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))

noncomputable def incircle_radius (a b c : ℝ) := (2 * area_of_triangle a b c) / (a + b + c)

theorem angle_A_condition_1 (a b c : ℝ) (h : real.sin a * real.cos b - real.sin B * real.cos A = real.sin c - real.sin B) :
  A = π / 3 :=
sorry

theorem angle_A_condition_2 (a b c : ℝ) (h : real.tan c = (real.tan A + real.tan b) / (real.sqrt 3 * real.tan b - 1)) :
  A = π / 3 :=
sorry

theorem angle_A_condition_3 (a b c : ℝ) (h : (1/2) * b * c * real.sin A = (1/2) * a * (b * real.sin B + c * real.sin c - a * real.sin A)) :
  A = π / 3 :=
sorry

theorem triangle_area (b c : ℝ) (h1 : a = 8)
  (h2 : incircle_radius a b c = real.sqrt 3)
  (Hlawofsin : area_of_triangle a b c = 11 * real.sqrt 3) :
  area_of_triangle 8 b c = 11 * real.sqrt 3 :=
sorry

end angle_A_condition_1_angle_A_condition_2_angle_A_condition_3_triangle_area_l226_226941


namespace normal_distribution_probability_l226_226539

noncomputable def P (a b : ℝ) : ℝ := 
  -- The probability distribution function to be defined later as needed.
  sorry 

variable (X : ℝ → Prop)
variable (μ σ : ℝ)

-- Given conditions
axiom h₁ : ∀ X, X = NormalDist μ σ
axiom h₂ : μ = 3
axiom h₃ : σ^2 > 0
axiom P_X_lt_2 : P(X, -∞, 2) = 0.3

-- The proof that P(2 < X < 4) = 0.4
theorem normal_distribution_probability :
  P(X, 2, 4) = 0.4 :=
sorry

end normal_distribution_probability_l226_226539


namespace volume_of_regular_triangular_pyramid_l226_226892

-- Define the parameters and conditions
variables (R α : Real)

-- Define the conclusion to be proven
theorem volume_of_regular_triangular_pyramid (hR : R > 0) (hα : 0 < α ∧ α < π / 2) : 
  let V := (1 / 4) * R^3 * sqrt 3 * (sin (2 * α))^3 * tan α
  V = (1 / 4) * R^3 * sqrt 3 * (sin 2*α)^3* (tan α) := 
sorry

end volume_of_regular_triangular_pyramid_l226_226892


namespace integer_solutions_of_equation_l226_226160

theorem integer_solutions_of_equation:
  (number_of_int_solutions (λ x y, 6 * y^2 + 3 * x * y + x + 2 * y + 180) = 6) :=
begin
  sorry
end

end integer_solutions_of_equation_l226_226160


namespace projection_eq_minus_4_l226_226534

variable (a b : ℝ^3)

def norm (v : ℝ^3) := Real.sqrt (v.dot v)

-- Given conditions as assumptions
axiom norm_a : norm a = 5
axiom norm_b : norm b = 3
axiom dot_ab : a.dot b = -12

-- Define the projection of a on b
def projection (a b : ℝ^3) : ℝ :=
  (a.dot b) / (norm b)

-- The theorem to prove
theorem projection_eq_minus_4 (a b : ℝ^3) :
  projection a b = -4 :=
by
  -- Filling in the proof is not required
  sorry

end projection_eq_minus_4_l226_226534


namespace expression_of_fn_l226_226905

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
if n = 0 then x else f (n - 1) x / (1 + n * x)

theorem expression_of_fn (n : ℕ) (x : ℝ) (hn : 1 ≤ n) : f n x = x / (1 + n * x) :=
sorry

end expression_of_fn_l226_226905


namespace factorization_identity_l226_226583

theorem factorization_identity 
  (p q r s t u : ℤ)
  (h : 729*y^3 + 64 = (p*y^2 + q*y + r) * (s*y^2 + t*y + u)) :
  p = 27 ∧ q = 4 ∧ r = 0 ∧ s = 729 ∧ t = -108 ∧ u = 16 → 
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 543106 :=
begin
  sorry
end

end factorization_identity_l226_226583


namespace vika_card_pairing_l226_226351

theorem vika_card_pairing :
  ∃ (d ∈ {1, 2, 3, 5, 6, 10, 15, 30}), ∃ (k : ℕ), 60 = 2 * d * k :=
by sorry

end vika_card_pairing_l226_226351


namespace count_valid_permutations_l226_226633

open Finset

def is_odd_pair_sum (a b : ℕ) : Prop :=
  (a + b) % 2 = 1

def valid_permutation (p : Finset (ℕ × ℕ)) : Prop :=
  ∃ (a b c d ∈ (Finset.ofList [1, 2, 3, 4])) (h_ab : (a, b) ∈ p) (h_cd : (c, d) ∈ p),
  (a ≠ b) ∧ (c ≠ d) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧
  is_odd_pair_sum a b ∧ is_odd_pair_sum c d

theorem count_valid_permutations : 
  ∃ p, valid_permutation p ∧ p.card = 16 :=
sorry

end count_valid_permutations_l226_226633


namespace fish_population_estimate_l226_226757

theorem fish_population_estimate 
  (caught_marked : ℕ) 
  (caught_total : ℕ) 
  (marked_recaught : ℕ) 
  (est_population : ℕ) 
  (h1 : caught_marked = 10) 
  (h2 : caught_total = 100) 
  (h3 : marked_recaught = 2) 
  (h4 : est_population = caught_marked * caught_total / marked_recaught) :
  est_population = 500 :=
by
  -- Given conditions
  have h_caught_marked : caught_marked = 10 := h1,
  have h_caught_total : caught_total = 100 := h2,
  have h_marked_recaught : marked_recaught = 2 := h3,

  -- Use them in the formula
  have est_pop : est_population = 10 * 100 / 2 := h4,
  rw [h1, h2, h3] at est_pop,
  -- Calculate the expected population
  linarith,
  sorry

end fish_population_estimate_l226_226757


namespace P_neither_l226_226182

-- Definition of probabilities according to given conditions
def P_A : ℝ := 0.63      -- Probability of answering the first question correctly
def P_B : ℝ := 0.50      -- Probability of answering the second question correctly
def P_A_and_B : ℝ := 0.33  -- Probability of answering both questions correctly

-- Theorem to prove the probability of answering neither of the questions correctly
theorem P_neither : (1 - (P_A + P_B - P_A_and_B)) = 0.20 := by
  sorry

end P_neither_l226_226182


namespace system_solutions_l226_226855

theorem system_solutions {x y b : ℝ} (hx : x > 0) (hy : y > 0) :
  (sqrt (x * y) = 3 * b ^ (b + 1) ∧ log b (x ^ log b y) + log b (y ^ log b x) = 6 * b ^ 5) ↔
  b ∈ set.Icc 0 (real.cbrt (1 / 3)) :=
sorry

end system_solutions_l226_226855


namespace f_value_1987_l226_226647

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_1987 :
  (∀ x y v : ℝ, x > y → f(y) - y ≥ v → v ≥ f(x) - x → ∃ z : ℝ, x ≥ z ∧ z ≥ y ∧ f(z) = v + z) ∧
  (∃ x₀ : ℝ, f(x₀) = 0 ∧ ∀ x : ℝ, f(x) = 0 → x₀ ≥ x) ∧
  (f(0) = 1) ∧
  (f(1987) ≤ 1988) ∧
  (∀ x y : ℝ, f(x) * f(y) = f(x * f(y) + y * f(x) - x * y))
  → f(1987) = 1988 :=
by
  sorry

end f_value_1987_l226_226647


namespace cost_of_dinner_l226_226225

theorem cost_of_dinner (tax_rate tip_rate service_charge_rate total_cost : ℝ) 
  (tax_rate_eq : tax_rate = 0.095) 
  (tip_rate_eq : tip_rate = 0.18) 
  (service_charge_rate_eq : service_charge_rate = 0.05) 
  (total_cost_eq : total_cost = 34.50) : 
  ∃ (x : ℝ), 
    (total_cost = (1 + tax_rate + tip_rate + service_charge_rate) * x) ∧ 
    x = 26 :=
by
  have eq1 : 1 + tax_rate + tip_rate + service_charge_rate = 1.325 :=
    by rw [tax_rate_eq, tip_rate_eq, service_charge_rate_eq]; ring
  use 26
  rw [← total_cost_eq, eq1]
  norm_num
  ring
  sorry

end cost_of_dinner_l226_226225


namespace number_of_valid_pairing_ways_l226_226339

-- Define a natural number as a condition.
def is_natural (n : ℕ) : Prop := 0 < n

-- Define that 60 cards can be paired with the same modulus difference.
def pair_cards_same_modulus_difference (d : ℕ) (k : ℕ) : Prop :=
  60 = 2 * d * k

-- Define what it means for d to be a divisor of 30.
def is_divisor_of_30 (d : ℕ) : Prop :=
  ∃ k, 30 = d * k

theorem number_of_valid_pairing_ways :
  (finset.univ.filter is_divisor_of_30).card = 8 :=
begin
  sorry
end

end number_of_valid_pairing_ways_l226_226339


namespace int_solution_count_l226_226166

theorem int_solution_count :
  let count_solutions (eq : ℤ → ℤ → Bool) : Nat :=
    Finset.card (Finset.filter (λ ⟨(y, x)⟩, eq y x) Finset.univ.prod Finset.univ)
  count_solutions (λ y x, 6 * y^2 + 3 * y * x + x + 2 * y + 180 = 0) = 6 :=
sorry

end int_solution_count_l226_226166


namespace part_one_part_two_l226_226547

def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x + a * (1 / x - 1)

def a_n_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then 1 else f (a (n - 1)) 1 + 2

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, ⌊a (i + 1)⌋

theorem part_one : ∃ (a : ℝ), (∀ (x : ℝ), 0 < x → f x a ≥ 0) ∧ (∃ (x : ℝ), 0 < x ∧ f x a = 0) := 
  by sorry

theorem part_two (a : ℕ → ℝ) (n : ℕ) :
  (a 1 = 1) → 
  (∀ n : ℕ, a (n + 1) = f (a n) 1 + 2) →
  (S_n a n = 2 * n - 1) :=
  by sorry

end part_one_part_two_l226_226547


namespace triangle_perimeter_l226_226128

/-- Given the lengths of two sides of a triangle are 1 and 4,
    and the length of the third side is an integer, 
    prove that the perimeter of the triangle is 9 -/
theorem triangle_perimeter
  (a b : ℕ)
  (c : ℤ)
  (h₁ : a = 1)
  (h₂ : b = 4)
  (h₃ : 3 < c ∧ c < 5) :
  a + b + c = 9 :=
by sorry

end triangle_perimeter_l226_226128


namespace smallest_sum_l226_226652

-- First, we define the conditions as assumptions:
def is_arithmetic_sequence (x y z : ℕ) : Prop :=
  2 * y = x + z

def is_geometric_sequence (x y z : ℕ) : Prop :=
  y ^ 2 = x * z

-- Given conditions
variables (A B C D : ℕ)
variables (hABC : is_arithmetic_sequence A B C) (hBCD : is_geometric_sequence B C D)
variables (h_ratio : 4 * C = 7 * B)

-- The main theorem to prove
theorem smallest_sum : A + B + C + D = 97 :=
sorry

end smallest_sum_l226_226652


namespace maximize_winning_once_in_three_l226_226806

-- Define the problem conditions
variables (n : ℕ) (Hn : n ≥ 5)

-- Function to compute probability of winning in one draw
noncomputable def winning_probability (n : ℕ) : ℝ := (10 * n) / ((n + 5) * (n + 4))

-- Probability of winning exactly once in three draws (with replacement)
noncomputable def winning_once_in_three (n : ℕ) : ℝ := 
  let p := winning_probability n in
  3 * p * (1 - p) ^ 2

-- Define the proof problem
theorem maximize_winning_once_in_three (n : ℕ) (Hn : n ≥ 5) :
  n = 20 → ∀ m ≥ 5, winning_once_in_three n ≥ winning_once_in_three m := 
sorry

end maximize_winning_once_in_three_l226_226806


namespace no_common_points_perpendicular_intersection_l226_226909

-- Definitions for the given circle and line equations
def Circle_eq (x y m : ℝ) : Prop := x^2 + y^2 + x - 6y + m = 0
def Line_eq (x y : ℝ) : Prop := x + 2y - 3 = 0

-- Problem (I): If line l and circle C have no common points, find the range of m
theorem no_common_points (m : ℝ) : 
  (¬ ∃ (x y : ℝ), Circle_eq x y m ∧ Line_eq x y) ↔ m ∈ (set.Ioo 8 (37/4)) :=
sorry

-- Problem (II): If line l intersects circle C at P and Q such that OP ⊥ OQ, find m
theorem perpendicular_intersection (m : ℝ) :
  (∃ (P Q : ℝ × ℝ), Circle_eq P.1 P.2 m ∧ Line_eq P.1 P.2 ∧ Circle_eq Q.1 Q.2 m ∧ Line_eq Q.1 Q.2 ∧ 
  (P.1 * Q.1 + P.2 * Q.2 = 0)) ↔ m = 3 :=
sorry

end no_common_points_perpendicular_intersection_l226_226909


namespace arithmetic_sequence_identity_l226_226923

noncomputable def a (n: ℕ) : ℝ := sorry -- Assume this definition represents the arithmetic sequence

theorem arithmetic_sequence_identity :
  let a := a in
  let integral_value := ∫ x in 0..2, sqrt(4 - x^2) in
  a 4 + a 8 = integral_value →
  a 6 * (a 2 + 2 * a 6 + a 10) = π^2 := 
by
  sorry -- The proof step is not required

end arithmetic_sequence_identity_l226_226923


namespace volume_of_pyramid_l226_226484

noncomputable def pyramid_volume : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 0)
  let C : ℝ × ℝ := (12, 20)
  let D : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) -- Midpoint of BC
  let E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) -- Midpoint of AC
  let F : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) -- Midpoint of AB
  let height : ℝ := 8.42 -- Vertically above the orthocenter
  let base_area : ℝ := 110 -- Area of the midpoint triangle
  (1 / 3) * base_area * height

theorem volume_of_pyramid : pyramid_volume = 309.07 :=
  by
    sorry

end volume_of_pyramid_l226_226484


namespace minBeneluxPartition_l226_226431

def isBadSet (S : Finset ℤ) : Prop :=
  S.sum id = 2010

def isBeneluxSet (S : Finset ℤ) : Prop :=
  ∀ T : Finset ℤ, T ⊆ S → ¬isBadSet T

def candidateSet : Finset ℤ :=
  Finset.range (2009 - 502 + 1) |>.map (λ x => x + 502)

theorem minBeneluxPartition (n : ℕ) : 
  (∃ parts : Fin n (Finset ℤ), 
    (Finset.univ : Finset (Fin n)).bUnion parts = candidateSet ∧
    ∀ i, isBeneluxSet (parts i)) → 
  n = 2 :=
sorry

end minBeneluxPartition_l226_226431


namespace isosceles_triangle_condition_l226_226314

theorem isosceles_triangle_condition (a b c A B C : ℝ) (h1 : a * Real.cos B + b * Real.cos C + c * Real.cos A = (1/2) * (a + b + c)) (h2 : A + B + C = Real.pi) : 
  (A = B ∨ B = C ∨ C = A) ∨ (a = b ∨ b = c ∨ c = a) :=
begin
  sorry
end

end isosceles_triangle_condition_l226_226314


namespace Douglas_won_72_percent_of_votes_in_county_X_l226_226608

/-- Definition of the problem conditions and the goal -/
theorem Douglas_won_72_percent_of_votes_in_county_X
  (V : ℝ)
  (total_votes_ratio : ∀ county_X county_Y, county_X = 2 * county_Y)
  (total_votes_percentage_both_counties : 0.60 = (1.8 * V) / (2 * V + V))
  (votes_percentage_county_Y : 0.36 = (0.36 * V) / V) : 
  ∃ P : ℝ, P = 72 ∧ P = (1.44 * V) / (2 * V) * 100 :=
sorry

end Douglas_won_72_percent_of_votes_in_county_X_l226_226608


namespace part_I_part_II_part_III_l226_226138

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1 / 2) * a * x^2

-- Part (Ⅰ)
theorem part_I (x : ℝ) : (0 < x) → (f 1 x < f 1 (x+1)) := sorry

-- Part (Ⅱ)
theorem part_II (f_has_two_distinct_extreme_values : ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ (f a x = f a y))) : 0 < a ∧ a < 1 := sorry

-- Part (Ⅲ)
theorem part_III (f_has_two_distinct_zeros : ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0)) : 0 < a ∧ a < (2 / Real.exp 1) := sorry

end part_I_part_II_part_III_l226_226138


namespace matchsticks_20th_stage_l226_226753

theorem matchsticks_20th_stage :
  let a1 := 3
  let d := 3
  let a20 := a1 + 19 * d
  a20 = 60 := by
  sorry

end matchsticks_20th_stage_l226_226753


namespace integer_solutions_count_l226_226175

theorem integer_solutions_count :
  ∃ (s : Finset (ℤ × ℤ)), (∀ (x y : ℤ), (6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0) ↔ ((x, y) ∈ s)) ∧ s.card = 4 :=
begin
  sorry
end

end integer_solutions_count_l226_226175


namespace equal_sums_of_squares_l226_226665

-- Define the coordinates of a rectangle in a 3D space.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define vertices of the rectangle.
def A : Point3D := ⟨0, 0, 0⟩
def B (a : ℝ) : Point3D := ⟨a, 0, 0⟩
def C (a b : ℝ) : Point3D := ⟨a, b, 0⟩
def D (b : ℝ) : Point3D := ⟨0, b, 0⟩

-- Distance squared between two points in 3D space.
def distance_squared (M N : Point3D) : ℝ :=
  (M.x - N.x)^2 + (M.y - N.y)^2 + (M.z - N.z)^2

-- Prove that the sums of the squares of the distances between an arbitrary point M and opposite vertices of the rectangle are equal.
theorem equal_sums_of_squares (a b : ℝ) (M : Point3D) :
  distance_squared M A + distance_squared M (C a b) = distance_squared M (B a) + distance_squared M (D b) :=
by
  sorry

end equal_sums_of_squares_l226_226665


namespace seating_arrangements_l226_226440

def ten_people := { "Ábel", "Bendegúz", "Zsuzsi", "Anikó", "E", "F", "G", "H", "I", "J" }
def rows := { { "A1", "A2", "A3", "A4", "A5" }, { "B1", "B2", "B3", "B4", "B5" } }

-- Assume a function seating that returns the set of all possible seatings given constraints
def seating (people : Set String) (rows : Set (Set String)) (neighbors : (String × String)) (separate : (String × String)) : Nat := sorry

theorem seating_arrangements : seating ten_people rows ("Ábel", "Bendegúz") ("Zsuzsi", "Anikó") = 518400 := 
sorry

end seating_arrangements_l226_226440


namespace range_f_x_le_neg_five_l226_226636

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x then 2^x - 3 else
if h : x < 0 then 3 - 2^(-x) else 0

theorem range_f_x_le_neg_five :
  ∀ x : ℝ, f x ≤ -5 ↔ x ≤ -3 :=
by sorry

end range_f_x_le_neg_five_l226_226636


namespace total_students_l226_226979

theorem total_students (S : ℕ) (R : ℕ) :
  (2 * 0 + 12 * 1 + 13 * 2 + R * 3) / S = 2 →
  2 + 12 + 13 + R = S →
  S = 43 :=
by
  sorry

end total_students_l226_226979


namespace smaller_octagon_area_fraction_l226_226728

theorem smaller_octagon_area_fraction (A B C D E F G H : Point)
  (midpoints_joined : Boolean)
  (regular_octagon : RegularOctagon A B C D E F G H)
  (smaller_octagon : Octagon (midpoint (A, B)) (midpoint (B, C)) (midpoint (C, D)) 
                              (midpoint (D, E)) (midpoint (E, F)) (midpoint (F, G))
                              (midpoint (G, H)) (midpoint (H, A))) :
  midpoints_joined → regular_octagon → 
  (area smaller_octagon) = (3 / 4) * (area regular_octagon) :=
by
  sorry

end smaller_octagon_area_fraction_l226_226728


namespace sum_p_q_r_l226_226305

theorem sum_p_q_r :
  ∃ (p q r : ℤ), 
    (∀ x : ℤ, x ^ 2 + 20 * x + 96 = (x + p) * (x + q)) ∧ 
    (∀ x : ℤ, x ^ 2 - 22 * x + 120 = (x - q) * (x - r)) ∧ 
    p + q + r = 30 :=
by 
  sorry

end sum_p_q_r_l226_226305


namespace function_has_zero_in_interval_l226_226832

   theorem function_has_zero_in_interval (fA fB fC fD : ℝ → ℝ) (hA : ∀ x, fA x = x - 3)
       (hB : ∀ x, fB x = 2^x) (hC : ∀ x, fC x = x^2) (hD : ∀ x, fD x = Real.log x) :
       ∃ x, 0 < x ∧ x < 2 ∧ fD x = 0 :=
   by
       sorry
   
end function_has_zero_in_interval_l226_226832


namespace sqrt_simplify_l226_226036

theorem sqrt_simplify (p : ℝ) :
  (Real.sqrt (12 * p) * Real.sqrt (7 * p^3) * Real.sqrt (15 * p^5)) =
  6 * p^4 * Real.sqrt (35 * p) :=
by
  sorry

end sqrt_simplify_l226_226036


namespace binomial_expansion_value_calculation_result_final_result_l226_226043

theorem binomial_expansion_value :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = (7 + 1)^4 := 
sorry

theorem calculation_result :
  (7 + 1)^4 = 4096 := 
sorry

theorem final_result :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by
  calc
    7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = (7 + 1)^4 := binomial_expansion_value
    ... = 4096 := calculation_result

end binomial_expansion_value_calculation_result_final_result_l226_226043


namespace mario_savings_percentage_l226_226206

-- Define the price of one ticket
def ticket_price : ℝ := sorry

-- Define the conditions
-- Condition 1: 5 tickets can be purchased for the usual price of 3 tickets
def price_for_5_tickets := 3 * ticket_price

-- Condition 2: Mario bought 5 tickets
def mario_tickets := 5 * ticket_price

-- Condition 3: Usual price for 5 tickets
def usual_price_5_tickets := 5 * ticket_price

-- Calculate the amount saved
def amount_saved := usual_price_5_tickets - price_for_5_tickets

theorem mario_savings_percentage
  (ticket_price: ℝ)
  (h1 : price_for_5_tickets = 3 * ticket_price)
  (h2 : mario_tickets = 5 * ticket_price)
  (h3 : usual_price_5_tickets = 5 * ticket_price)
  (h4 : amount_saved = usual_price_5_tickets - price_for_5_tickets):
  (amount_saved / usual_price_5_tickets) * 100 = 40 := 
by {
    -- Placeholder
    sorry
}

end mario_savings_percentage_l226_226206


namespace cubic_has_real_root_l226_226664

open Real

-- Define the conditions
variables (a0 a1 a2 a3 : ℝ) (h : a0 ≠ 0)

-- Define the cubic polynomial function
def cubic (x : ℝ) : ℝ :=
  a0 * x^3 + a1 * x^2 + a2 * x + a3

-- State the theorem
theorem cubic_has_real_root : ∃ x : ℝ, cubic a0 a1 a2 a3 x = 0 :=
by
  sorry

end cubic_has_real_root_l226_226664


namespace no_meeting_at_O_l226_226911

theorem no_meeting_at_O
  (A B C D O : Type)
  [quadrilateral : ConvexQuadrilateral A B C D]
  (path_Petya : WalkPerimeter A B C D)
  (path_Vasya : WalkDiagonal A C)
  (path_Tolya : WalkDiagonal B D)
  (simultaneous_Petya_Vasya_at_C : ArriveSimultaneously path_Petya A B C path_Vasya A C)
  (simultaneous_Petya_Tolya_at_D : ArriveSimultaneously path_Petya A B C D path_Tolya B D) :
  ¬ ArriveSimultaneously path_Vasya A O path_Tolya B O := sorry

end no_meeting_at_O_l226_226911


namespace circle_properties_l226_226133

theorem circle_properties
    (h_circle : ∀ {x y : ℝ}, x^2 + y^2 - 2 * x = 0)
    (h_line : ∀ {x y : ℝ}, y = x) :
    (circle_center : (ℝ × ℝ)) ∧
    (chord_length : ℝ) :=
by
  let circle_center := (1, 0)
  let chord_length := Real.sqrt 2
  exact ⟨circle_center, chord_length⟩

end circle_properties_l226_226133


namespace a_2_a_3_a_4_a_general_b_min_b_max_l226_226824

variable (n : ℕ) (a : ℕ → ℚ)

-- Conditions
axiom a1 : a 1 = 5
axiom a2 : ∀ n, n ≥ 2 → (∑ i in Finset.range (n-1), 1 / a i.succ) = 2 / a n

-- Prove (1): individual terms
theorem a_2 : a 2 = 10 := by
  sorry

theorem a_3 : a 3 = 20 / 3 := by
  sorry

theorem a_4 : a 4 = 40 / 9 := by
  sorry

-- Prove (2): general term formula
theorem a_general : ∀ n, a n = if n = 1 then 5 else 10 * (2 / 3) ^ (n - 2) := by
  sorry

-- Prove (3): maximum and minimum of b_n
def b (n : ℕ) : ℚ := a n / (11 - 2 * a n)

theorem b_min : b 3 = -20 / 7 := by
  sorry

theorem b_max : b 1 = 5 := by
  sorry

end a_2_a_3_a_4_a_general_b_min_b_max_l226_226824


namespace at_least_binom_ways_l226_226798

noncomputable def num_ways_to_choose := Nat.choose (2 * n - 2) (n - 1)

theorem at_least_binom_ways (n : ℕ) (h : 1 ≤ n) 
  (numbers : Fin (2 * n - 1) → ℝ) (distinct_pos : ∀ i j, i ≠ j → numbers i ≠ numbers j) 
  (positive : ∀ i, 0 < numbers i) (sum_S : (∑ i, numbers i) = S) :
  ∃ (choices : Finset (Fin (2 * n - 1))), choices.card = n ∧ 
  ∑ i in choices, numbers i ≥ S / 2 :=
sorry -- skip the proof

end at_least_binom_ways_l226_226798


namespace problem1_problem2_l226_226570

variables {p x1 x2 y1 y2 : ℝ} (h₁ : p > 0) (h₂ : x1 * x2 ≠ 0) (h₃ : y1^2 = 2 * p * x1) (h₄ : y2^2 = 2 * p * x2)

theorem problem1 (h₅ : x1 * x2 + y1 * y2 = 0) :
    ∀ (x y : ℝ), (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0 → 
        x^2 + y^2 - (x1 + x2) * x - (y1 + y2) * y = 0 := sorry

theorem problem2 (h₀ : ∀ x y, x = (x1 + x2) / 2 → y = (y1 + y2) / 2 → 
    |((x1 + x2) / 2) - 2 * ((y1 + y2) / 2)| / (Real.sqrt 5) = 2 * (Real.sqrt 5) / 5) :
    p = 2 := sorry

end problem1_problem2_l226_226570


namespace root_is_in_interval_l226_226190

noncomputable def f (x : ℝ) : ℝ := log x + x - 3

theorem root_is_in_interval 
  (k : ℤ) 
  (h1 : (f 2 : ℝ) < 0) 
  (h2 : 0 < (f 3 : ℝ)) :
  (∃ x, f x = 0 ∧ (k : ℝ) < x ∧ x < (k + 1)) → k = 2 :=
by
  sorry

end root_is_in_interval_l226_226190


namespace percys_dish_cost_l226_226269

variable (P : ℝ)

def total_cost := 10 + 13 + P
def tip := 0.10 * total_cost
def given_tip := 4

theorem percys_dish_cost : tip = given_tip → P = 17 :=
by
  sorry

end percys_dish_cost_l226_226269


namespace NaOH_HCl_reaction_l226_226511

theorem NaOH_HCl_reaction (m : ℝ) (HCl : ℝ) (NaCl : ℝ) 
  (reaction_eq : NaOH + HCl = NaCl + H2O)
  (HCl_combined : HCl = 1)
  (NaCl_produced : NaCl = 1) :
  m = 1 := by
  sorry

end NaOH_HCl_reaction_l226_226511


namespace john_lewis_meeting_distance_l226_226224

def distance_between (A B : String) : ℝ := 240
def john_speed : ℝ := 40
def lewis_speed_initial : ℝ := 60
def john_break_time_hours : ℝ := 15 / 60
def lewis_break_time_hours : ℝ := 20 / 60
def john_break_interval_hours : ℝ := 2
def lewis_break_interval_hours : ℝ := 2.5
def lewis_speed_return : ℝ := lewis_speed_initial - 10

theorem john_lewis_meeting_distance :
  ∃ (d : ℝ), d = 240 - 50 * (13 / 3) :=
by
  exists 23.33
  sorry

end john_lewis_meeting_distance_l226_226224


namespace foldable_positions_in_cross_l226_226690

noncomputable def foldable_positions_count : ℕ := 6

theorem foldable_positions_in_cross (positions : set ℕ) (h : positions = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :
  ∃ allowed_positions : set ℕ, allowed_positions ⊆ positions ∧ count allowed_positions 6 :=
sorry

end foldable_positions_in_cross_l226_226690


namespace simplify_expr_l226_226672

theorem simplify_expr : 
  (576:ℝ)^(1/4) * (216:ℝ)^(1/2) = 72 := 
by 
  have h1 : 576 = (2^4 * 36 : ℝ) := by norm_num
  have h2 : 36 = (6^2 : ℝ) := by norm_num
  have h3 : 216 = (6^3 : ℝ) := by norm_num
  sorry

end simplify_expr_l226_226672


namespace even_function_expression_l226_226538

variable (f : ℝ → ℝ)

noncomputable def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

theorem even_function_expression (h1 : is_even_function f)
    (h2 : ∀ x : ℝ, 0 < x → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = -x^3 - x + 1 :=
by
  intro x hx
  have h_negx_pos : 0 < -x := by linarith
  specialize h2 (-x) h_negx_pos
  have h3 : f x = f (-x) := h1 x
  rw [←h3, h2]
  ring

end even_function_expression_l226_226538


namespace molecular_weight_of_NH4I_l226_226380

-- Define the conditions in Lean
def molecular_weight (moles grams: ℕ) : Prop :=
  grams / moles = 145

-- Statement of the proof problem
theorem molecular_weight_of_NH4I :
  molecular_weight 9 1305 :=
by
  -- Proof is omitted 
  sorry

end molecular_weight_of_NH4I_l226_226380


namespace min_value_l226_226259

open Real

noncomputable def C (n k : ℕ) : ℝ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem min_value (n : ℕ) (p : ℝ → ℝ)
    (h1 : n > 0)
    (h2 : p (n+1) = 1)
    (h3 : ∀ x, p x = p n) :
    (∑ i in finset.range (n+1) + 1, (i^2 * (p i)^2) / 
        (∏ j in finset.range (n+1) + 1, (i+j)^2)) ≥ ((nat.factorial n)^4 / (4 * (nat.factorial (2*n)))) :=
sorry

end min_value_l226_226259


namespace chessboard_cover_impossible_l226_226621

theorem chessboard_cover_impossible :
  let chessboard_size := 13
  let piece_size := (4, 1)
  let number_of_pieces := 42
  let central_square_uncovered := true
  let total_squares := chessboard_size * chessboard_size
  let squares_uncovered := if central_square_uncovered then total_squares - 1 else total_squares
  let black_squares := (total_squares + 1) / 2
  let white_squares := total_squares - black_squares
  let remaining_black_squares := black_squares - 1
  let remaining_white_squares := white_squares
  number_of_pieces * (fst piece_size) ≠ squares_uncovered →
  (remaining_black_squares + remaining_white_squares = squares_uncovered) →
  (remaining_black_squares ≠ remaining_white_squares) →
  False :=
by {
  sorry
}

end chessboard_cover_impossible_l226_226621


namespace vika_card_pairing_l226_226352

theorem vika_card_pairing :
  ∃ (d ∈ {1, 2, 3, 5, 6, 10, 15, 30}), ∃ (k : ℕ), 60 = 2 * d * k :=
by sorry

end vika_card_pairing_l226_226352


namespace area_ratio_of_smaller_octagon_l226_226715

theorem area_ratio_of_smaller_octagon (A B C D E F G H P Q R S T U V W : Point) 
  (h1 : is_regular_octagon A B C D E F G H)
  (h2 : midpoint A B = P) (h3 : midpoint B C = Q) (h4 : midpoint C D = R)
  (h5 : midpoint D E = S) (h6 : midpoint E F = T) (h7 : midpoint F G = U)
  (h8 : midpoint G H = V) (h9 : midpoint H A = W):
  area (octagon A B C D E F G H) / area (octagon P Q R S T U V W) = 4 := sorry

end area_ratio_of_smaller_octagon_l226_226715


namespace pascal_triangle_sum_row_20_l226_226103

theorem pascal_triangle_sum_row_20 :
  (Nat.binomial 20 4) + (Nat.binomial 20 5) = 20349 :=
by
  sorry

end pascal_triangle_sum_row_20_l226_226103


namespace term_1010_of_sequence_l226_226896

theorem term_1010_of_sequence (a : ℕ → ℕ) (h : ∀ n : ℕ, n > 0 → (a (1) + a (2) + a (...) + a (n)) / n = 2 * n) : a 1010 = 4038 :=
by sorry

end term_1010_of_sequence_l226_226896


namespace smallest_n_for_coloring_l226_226516

theorem smallest_n_for_coloring (n : ℕ) : 
  (∀ (c : ℕ → Prop), (∀ x y z w ∈ ({1, 2, ..., n} : set ℕ), x + y + z = w → c x = c y ∧ c y = c z ∧ c z = c w)) ↔ n ≥ 11 :=
sorry

end smallest_n_for_coloring_l226_226516


namespace business_proof_l226_226421

section Business_Problem

variables (investment cost_initial rubles production_capacity : ℕ)
variables (produced_July incomplete_July bottles_August bottles_September days_September : ℕ)
variables (total_depreciation residual_value sales_amount profit_target : ℕ)

def depreciation_per_bottle (cost_initial production_capacity : ℕ) : ℕ := 
    cost_initial / production_capacity

def calculate_total_depreciation (depreciation_per_bottle produced_July bottles_August bottles_September : ℕ) : ℕ :=
    (produced_July * depreciation_per_bottle) + (bottles_August * depreciation_per_bottle) + (bottles_September * depreciation_per_bottle)

def calculate_residual_value (cost_initial total_depreciation : ℕ) : ℕ :=
    cost_initial - total_depreciation

def calculate_sales_amount (residual_value profit_target : ℕ) : ℕ :=
    residual_value + profit_target

theorem business_proof
    (H1: investment = 1500000) 
    (H2: cost_initial = 500000)
    (H3: production_capacity = 100000)
    (H4: produced_July = 200)
    (H5: incomplete_July = 5)
    (H6: bottles_August = 15000)
    (H7: bottles_September = 12300)
    (H8: days_September = 20)
    (H9: total_depreciation = 137500)
    (H10: residual_value = 362500)
    (H11: profit_target = 10000)
    (H12: sales_amount = 372500): 

    total_depreciation = calculate_total_depreciation (depreciation_per_bottle cost_initial production_capacity) produced_July bottles_August bottles_September ∧
    residual_value = calculate_residual_value cost_initial total_depreciation ∧
    sales_amount = calculate_sales_amount residual_value profit_target := 
by 
  sorry

end Business_Problem

end business_proof_l226_226421


namespace num_three_digit_even_l226_226853

def isEven (n : ℕ) : Prop := n % 2 = 0

def three_digit_number_set := {0, 1, 2, 3, 4}

def isValidThreeDigitNumber (n : ℕ) : Prop :=
  let digits := List.ofDigits [n / 100, (n / 10) % 10, n % 10] in
  n >= 100 ∧ n < 1000 ∧ isEven n ∧ digits.nodup ∧ digits.all (λ x, x ∈ three_digit_number_set)

theorem num_three_digit_even (s : Finset ℕ) (h1 : s = three_digit_number_set) :
  (Finset.filter (λ n, isValidThreeDigitNumber n ∧ n ≥ 100 ∧ n < 1000) Finset.univ).card = 30 :=
sorry

end num_three_digit_even_l226_226853


namespace card_pairing_modulus_l226_226360

theorem card_pairing_modulus (cards : Finset ℕ) (h : cards = (Finset.range 60).image (λ n, n + 1)) :
  ∃ n, n = 8 ∧ ∀ (pairs : Finset (ℕ × ℕ)), (∀ (p ∈ pairs), (p.1 ∈ cards ∧ p.2 ∈ cards ∧ (|p.1 - p.2| = d))) → pairs.card = 30 :=
sorry

end card_pairing_modulus_l226_226360


namespace seq_fixed_point_l226_226110

theorem seq_fixed_point (a_0 b_0 : ℝ) (a b : ℕ → ℝ)
  (h1 : a 0 = a_0)
  (h2 : b 0 = b_0)
  (h3 : ∀ n, a (n + 1) = a n + b n)
  (h4 : ∀ n, b (n + 1) = a n * b n) :
  a 2022 = a_0 ∧ b 2022 = b_0 ↔ b_0 = 0 := sorry

end seq_fixed_point_l226_226110


namespace find_a_l226_226234

-- Definitions of the conditions
def S (n : ℕ) : ℕ :=
  ∑ k in finset.filter (λ i, nat.coprime i n) (finset.range n), k^2

def p : ℕ := 2^7 - 1

def q : ℕ := 2^5 - 1

-- The theorem to prove
theorem find_a :
  ∃ a b c : ℕ, b < c ∧ nat.coprime b c ∧ 
  S (p * q) = (p^2 * q^2 / 6 : ℚ) * (a - (b / c : ℚ)) ∧ a = 7561 :=
sorry

end find_a_l226_226234


namespace cos_tan_α_l226_226536

noncomputable def P (y : ℝ) : ℝ × ℝ := (-real.sqrt 3, y)
def α := ℝ
def sin_α (y : ℝ) : ℝ := (real.sqrt 2 / 4) * y

theorem cos_tan_α (y : ℝ) (h₁ : sin α = sin_α y) (h₂ : P y = (-real.sqrt 3, y)) :
  (cos α = -1 ∧ tan α = 0) ∨ 
  (cos α = -real.sqrt 6 / 4 ∧ tan α = -real.sqrt 15 / 3) ∨ 
  (cos α = -real.sqrt 6 / 4 ∧ tan α = real.sqrt 15 / 3) :=
sorry

end cos_tan_α_l226_226536


namespace calc_sequence_l226_226435

theorem calc_sequence (x : ℝ) (n : ℕ) (h : x ≠ 0) : 
  let seq_oper := λ (z : ℝ), (z^2)^2⁻¹
  let y := (seq_oper^[n] x)
  y = x ^ ((-4)^n) :=
by
  sorry

end calc_sequence_l226_226435


namespace disproves_proposition_l226_226096

theorem disproves_proposition (a b : ℤ) (h : a = -3 ∧ b = 2) : ¬ (a^2 > b^2 → a > b) :=
by {
  obtain ⟨ha, hb⟩ := h,
  sorry
}

end disproves_proposition_l226_226096


namespace calculate_fraction_l226_226842

theorem calculate_fraction (x : ℕ) (h : x = 3) : 
  (∏ i in Finset.range 10, x^(i+1)) / (∏ i in Finset.range 8, x^(2*(i+1))) = 3^(-17) by {
  sorry
}

end calculate_fraction_l226_226842


namespace total_depreciation_correct_residual_value_correct_sales_price_correct_l226_226419

-- Definitions and conditions
def initial_cost := 500000
def max_capacity := 100000
def jul_bottles := 200
def aug_bottles := 15000
def sep_bottles := 12300

def depreciation_per_bottle := initial_cost / max_capacity

-- Part (a)
def total_depreciation_jul := jul_bottles * depreciation_per_bottle
def total_depreciation_aug := aug_bottles * depreciation_per_bottle
def total_depreciation_sep := sep_bottles * depreciation_per_bottle
def total_depreciation := total_depreciation_jul + total_depreciation_aug + total_depreciation_sep

theorem total_depreciation_correct :
  total_depreciation = 137500 := 
by sorry

-- Part (b)
def residual_value := initial_cost - total_depreciation

theorem residual_value_correct :
  residual_value = 362500 := 
by sorry

-- Part (c)
def desired_profit := 10000
def sales_price := residual_value + desired_profit

theorem sales_price_correct :
  sales_price = 372500 := 
by sorry

end total_depreciation_correct_residual_value_correct_sales_price_correct_l226_226419


namespace rebecca_eggs_l226_226285

/-- Rebecca wants to split a collection of eggs into 4 groups. Each group will have 2 eggs. -/
def number_of_groups : Nat := 4

def eggs_per_group : Nat := 2

theorem rebecca_eggs : (number_of_groups * eggs_per_group) = 8 := by
  sorry

end rebecca_eggs_l226_226285


namespace find_line_eq_l226_226152

-- Define the centers of the circles
def P : ℝ × ℝ := (7, -4)
def Q : ℝ × ℝ := (-5, 6)

-- Midpoint of P and Q
def M (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Slope of the line through P and Q
def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Negative reciprocal of the slope
def neg_reciprocal (m : ℝ) : ℝ := -1 / m

-- Equation of line in point-slope form
def line_eq (m : ℝ) (P : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ Q, Q.2 - P.2 = m * (Q.1 - P.1)

-- Simplified equation of the line in standard form
def line_std_form (a b c : ℝ) : ℝ × ℝ → Prop :=
  λ Q, a * Q.1 + b * Q.2 + c = 0

-- Main theorem
theorem find_line_eq :
  let l := line_eq (neg_reciprocal (slope P Q)) (M P Q) in
  ∀ Q, l Q ↔ line_std_form 6 (-5) 1 Q :=
by
  intro l Q
  sorry

end find_line_eq_l226_226152


namespace vertex_angle_isosceles_triangle_l226_226117

theorem vertex_angle_isosceles_triangle (ABC : Triangle) (isosceles : is_isosceles ABC)
  (exterior_angle_adjacent_to_A : ∃ β, β = 130 ∧ exterior_angle_adjacent ABC A β) : 
  vertex_angle ABC = 50 ∨ vertex_angle ABC = 80 :=
sorry

end vertex_angle_isosceles_triangle_l226_226117


namespace find_a5_l226_226212

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a 1 + (n - 1) * d

theorem find_a5 (a : ℕ → ℤ) (d : ℤ)
  (h_seq : arithmetic_sequence a d)
  (h1 : a 1 + a 5 = 8)
  (h4 : a 4 = 7) : 
  a 5 = 10 := sorry

end find_a5_l226_226212


namespace translated_graph_matches_derivative_l226_226248

noncomputable def f (x : Real) : Real := Real.cos x - Real.sin x

theorem translated_graph_matches_derivative :
  ∃ (m : Real), (∀ (x : Real), f(x + m) = -Real.sin x - Real.cos x) ↔ m = Real.pi / 2 := sorry

end translated_graph_matches_derivative_l226_226248


namespace sum_of_prime_factors_of_1365_l226_226782

theorem sum_of_prime_factors_of_1365 : 
  let smallest_prime_factor := 3 in
  let largest_prime_factor := 13 in
  smallest_prime_factor + largest_prime_factor = 16 :=
by
  let smallest_prime_factor := 3
  let largest_prime_factor := 13
  have h : smallest_prime_factor + largest_prime_factor = 16 := by
    -- the proof will be provided here, but it is not required in this step
    sorry
  exact h

end sum_of_prime_factors_of_1365_l226_226782


namespace probability_of_both_white_l226_226615

namespace UrnProblem

-- Define the conditions
def firstUrnWhiteBalls : ℕ := 4
def firstUrnTotalBalls : ℕ := 10
def secondUrnWhiteBalls : ℕ := 7
def secondUrnTotalBalls : ℕ := 12

-- Define the probabilities of drawing a white ball from each urn
def P_A1 : ℚ := firstUrnWhiteBalls / firstUrnTotalBalls
def P_A2 : ℚ := secondUrnWhiteBalls / secondUrnTotalBalls

-- Define the combined probability of both events occurring
def P_A1_and_A2 : ℚ := P_A1 * P_A2

-- Theorem statement that checks the combined probability
theorem probability_of_both_white : P_A1_and_A2 = 7 / 30 := by
  sorry

end UrnProblem

end probability_of_both_white_l226_226615


namespace minor_premise_syllogism_l226_226566

variable (x A ω ϕ : ℝ)

theorem minor_premise_syllogism :
  (∀ A ω ϕ, (∃ A ω ϕ, ∀ x, 2 * sin (1/2 * x) + cos (1/2 * x) = A * cos (ω * x + ϕ)) → 
  (∀ x, A * cos (ω * x + ϕ) = A * cos (ω * x + ϕ)) → 
  (∃ A ω ϕ, ∀ x, 2 * sin (1/2 * x) + cos (1/2 * x) = A * cos (ω * x + ϕ))) :=
sorry

end minor_premise_syllogism_l226_226566


namespace smaller_octagon_area_half_l226_226705

theorem smaller_octagon_area_half
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (P Q R S T U V W : Point)
  (H1 : midpoint P A B)
  (H2 : midpoint Q B C)
  (H3 : midpoint R C D)
  (H4 : midpoint S D E)
  (H5 : midpoint T E F)
  (H6 : midpoint U F G)
  (H7 : midpoint V G H)
  (H8 : midpoint W H A):
  area (octagon P Q R S T U V W) = (1 / 2) * area (octagon ABCDEFGH) :=
sorry

end smaller_octagon_area_half_l226_226705


namespace probability_fourth_term_integer_l226_226074

-- Define the initial condition
def initial_term : ℕ := 10

-- Define the sequence generation rules
  
def next_term (x : ℝ) (outcome : Bool) : ℝ :=
  if outcome then 2 * x - 1 else x / 2 - 1

-- Define the process to generate the sequence up to the fourth term

def term_seq (n : ℕ) : ℝ :=
  if n = 0 then initial_term
  else if n = 1 then next_term initial_term sorry -- the first coin result
  else if n = 2 then next_term (next_term initial_term sorry) sorry
  else if n = 3 then next_term (next_term (next_term initial_term sorry) sorry) sorry
  else next_term (next_term (next_term (next_term initial_term sorry) sorry) sorry) sorry

-- Proving the probability of the fourth term being an integer is 1/2
theorem probability_fourth_term_integer : 
  (probability {w : ℝ | ∃ a, term_seq 3 = a ∧ a ∈ ℤ} = 1 / 2) :=
by 
  -- To be completed
  sorry

end probability_fourth_term_integer_l226_226074


namespace range_of_sqrt_expr_l226_226591

theorem range_of_sqrt_expr (x : ℝ) (h : ∃ y : ℝ, y = sqrt (x - 2)) : x ≥ 2 :=
by
  sorry

end range_of_sqrt_expr_l226_226591


namespace triangle_base_length_l226_226685

-- Given conditions
def area_triangle (base height : ℕ) : ℕ := (1 / 2 : ℚ) * base * height

-- Problem statement
theorem triangle_base_length (A h : ℕ) (A_eq : A = 24) (h_eq : h = 8) :
  ∃ b : ℕ, area_triangle b h = A ∧ b = 6 := 
by
  sorry

end triangle_base_length_l226_226685


namespace Tom_total_miles_per_week_l226_226333

theorem Tom_total_miles_per_week :
  ∀ (days_per_week : ℕ) (hours_per_day : ℝ) (speed : ℝ),
    days_per_week = 5 →
    hours_per_day = 1.5 →
    speed = 8 →
    (days_per_week * hours_per_day * speed) = 60 := 
by
  intros days_per_week hours_per_day speed h_days h_hours h_speed
  rw [h_days, h_hours, h_speed]
  norm_num
  sorry

end Tom_total_miles_per_week_l226_226333


namespace parabola_focus_proof_l226_226301

-- Define the parabola y^2 = 4x
def is_parabola (p : ℝ × ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), p (x, y) ↔ y^2 = 4 * x

-- Define the coordinates of the focus
def is_focus (q : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ focus, focus = (1, 0)

-- Combine the definitions to state the problem
theorem parabola_focus_proof (p : ℝ × ℝ → Prop) (q : ℝ × ℝ) :
  is_parabola p → is_focus q (1, 0) → (∀ (x y : ℝ), p (x, y) ↔ y^2 = 4 * x) → q = (1, 0) :=
by
  intros hp hf hparabola
  sorry

end parabola_focus_proof_l226_226301


namespace trader_made_profit_l226_226017

theorem trader_made_profit (P : ℝ) (hP : 0 < P) :
  let bought_price := 0.8 * P in
  let modified_price := 1.15 * bought_price in
  let pre_final_price := 0.9 * modified_price in
  let final_price := 1.45 * pre_final_price in
  100 * (final_price - P) / P = 20.06 := 
by 
  let bought_price := 0.8 * P
  let modified_price := 1.15 * bought_price
  let pre_final_price := 0.9 * modified_price
  let final_price := 1.45 * pre_final_price
  calc
    100 * (final_price - P) / P = 100 * ((1.45 * (0.9 * (1.15 * (0.8 * P))) - P) / P) : by rfl
    ... = 20.06 : sorry  -- Completing the calculation will yield 20.06

end trader_made_profit_l226_226017


namespace solve_arithmetic_sequence_sum_l226_226210

-- Assume an arithmetic sequence with the given conditions
variables (a : ℕ → ℝ) (d : ℝ) (a1 : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := a 5 + a 10 = 58
def condition2 : Prop := a 4 + a 9 = 50

-- The definition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) : Prop := 
  ∀ n, a n = a1 + (n - 1) * d

-- The sum of the first 10 terms of the arithmetic sequence
def sum_first_10_terms (a : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range 10, a (i + 1)

theorem solve_arithmetic_sequence_sum (a : ℕ → ℝ) (a1 d : ℝ)
  (h1 : condition1 a)
  (h2 : condition2 a)
  (ha : arithmetic_sequence a a1 d) :
  sum_first_10_terms a = 210 := 
sorry

end solve_arithmetic_sequence_sum_l226_226210


namespace magnitude_of_a_l226_226120

open Real

-- Assuming the standard inner product space for vectors in Euclidean space

variables (a b : ℝ) -- Vectors in R^n (could be general but simplified to real numbers for this example)
variable (θ : ℝ)    -- Angle between vectors
axiom angle_ab : θ = 60 -- Given angle between vectors

-- Conditions:
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0
axiom norm_b : abs b = 1
axiom norm_2a_minus_b : abs (2 * a - b) = 1

-- To prove:
theorem magnitude_of_a : abs a = 1 / 2 :=
sorry

end magnitude_of_a_l226_226120


namespace min_cubes_for_views_l226_226811

theorem min_cubes_for_views : 
  (∃ (C : ℕ → ℕ → ℕ → Prop), 
    (∀ (x y z : ℕ), C x y z → C (x+1) y z ∨ C x (y+1) z ∨ C x y (z+1)) ∧
    (∀ k, ∃ x y z, C x y z ∧ (x + y + z = k) ∧
      ((C 0 0 0 ∧ x=2 ∧ y=0 ∧ z=0) ∧ 
       (C 0 0 0 ∧ x=0 ∧ y=2 ∧ z=0) ∧ 
       (C 0 0 0 ∧ x=0 ∧ y=0 ∧ z=2))) ∧ 
    C 0 0 0) → 
  (minimum_cubes = 5) :=
begin
  sorry
end

end min_cubes_for_views_l226_226811


namespace initial_games_count_l226_226628

-- Definitions used in conditions
def games_given_away : ℕ := 99
def games_left : ℝ := 22.0

-- Theorem statement for the initial number of games
theorem initial_games_count : games_given_away + games_left = 121.0 := by
  sorry

end initial_games_count_l226_226628


namespace rational_root_of_p_l226_226098

noncomputable def p (n : ℕ) (x : ℚ) : ℚ :=
  x^n + (2 + x)^n + (2 - x)^n

theorem rational_root_of_p :
  ∀ n : ℕ, n > 0 → (∃ x : ℚ, p n x = 0) ↔ n = 1 := by
  sorry

end rational_root_of_p_l226_226098


namespace bn_pos_int_for_n_gt_1_l226_226744

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1
  else (1 / 2) * (a (n - 1)) + (1 / (4 * (a (n - 1))))

def b (n : ℕ) : ℝ :=
  sqrt (2 / (2 * (a n)^2 - 1))

theorem bn_pos_int_for_n_gt_1 (n : ℕ) (h : n > 1) : ∃ (k : ℤ), b n = k ∧ 0 < k :=
sorry

end bn_pos_int_for_n_gt_1_l226_226744


namespace EF_bisects_angle_CFD_l226_226035

open EuclideanGeometry

theorem EF_bisects_angle_CFD
  (O C D : Point)
  (semicircle : Semicircle)
  (is_centroid : is_center O semicircle)
  (tangent_C : Tangent)
  (tangent_D : Tangent)
  (B : Point)
  (A : Point)
  (tangent_at_C : tangent_AT tangent_C .at C)
  (tangent_at_D : tangent_AT tangent_D .at D)
  (opposite_sides : A and B on opposite sides of O)
  (E : Point := meet (line_TAC AC) (line_TAC BD))
  (F : Point := Foot (Perpendicular E to line_TAC AB)) :
  bisects_angle EF (angle CFD) :=
  sorry

end EF_bisects_angle_CFD_l226_226035


namespace inequality_sum_leq_three_l226_226908

theorem inequality_sum_leq_three
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (hxyz : x^2 + y^2 + z^2 ≥ 3) :
  (x^2 + y^2 + z^2) / (x^5 + y^2 + z^2) + 
  (x^2 + y^2 + z^2) / (y^5 + x^2 + z^2) + 
  (x^2 + y^2 + z^2) / (z^5 + x^2 + y^2 + z^2) ≤ 3 := 
sorry

end inequality_sum_leq_three_l226_226908


namespace tangent_line_eq_a_zero_monotonicity_of_f_l226_226651

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log x + (x - 1) / (x + 1)
noncomputable def f_deriv (a : ℝ) (x : ℝ) := a / x + 2 / ((x + 1)^2)

theorem tangent_line_eq_a_zero :
  let a := 0 in
  let x : ℝ := 1 in
  let tangent_slope := f_deriv a x in
  let tangent_point := (x, f a x) in
  tangent_slope = 1 / 2 ∧ tangent_point = (1, 0) ∧ (∀ y : ℝ, y = 1 / 2 * (x - 1)) :=
by 
  sorry

theorem monotonicity_of_f (a : ℝ) :
  (a > 0 → ∀ x > 0, f_deriv a x > 0) ∧
  (a = 0 → ∀ x > 0, f_deriv a x >= 0) ∧
  (a < 0 → (
    (∀ x > 0, f_deriv a x > 0) ∨ 
    (∀ x > (-(a+1) - Real.sqrt (2 * a + 1)) / a, f_deriv a x < 0))) :=
by 
  sorry

end tangent_line_eq_a_zero_monotonicity_of_f_l226_226651


namespace max_distance_PM_l226_226593

theorem max_distance_PM :
  ∀ (m : ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ),
  (∃ x y : ℝ, (P = (x, y) ∧ x + m * y - 2 = 0 ∧ m * x - y + 2 = 0)) ∧
  (∃ x y : ℝ, (M = (x, y) ∧ (x + 2)^2 + (y + 2)^2 = 1 ∧ ((x + 2)*(fst P + 2) + (y + 2)*(snd P + 2) = 1))) →
  max (λ PM : ℝ, PM = real.sqrt ((fst P - fst M) ^ 2 + (snd P - snd M) ^ 2)) = real.sqrt 31 :=
by
  sorry

end max_distance_PM_l226_226593


namespace least_length_xz_zero_l226_226414

-- Definition of the right-angled triangle and its properties
def triangle_pqr := ∃ (P Q R : Type) (p : P) (q : Q) (r : R)
  (angle_pqr : angle (p, q, r) = 90)
  (pq : dist p q = 7)
  (qr : dist q r = 8)

-- Definition of the variable point X and lines through X and Y
def point_on_pq (X Q : Type) (x : X) (q : Q) :=
  variable_point x

def line_through_x (X QR : Type) (x : X) (qr : QR) :=
  parallel_line x qr
  
def line_through_y (Y PQ : Type) (y : Y) (pq : PQ) :=
  parallel_line y pq

-- Definition for the least possible length of XZ
def least_possible_length_of_xz (P Q R X Y Z : Type) (p : P) (q : Q) (r : R) (x : X) (y : Y) (z : Z)
  [triangle_pqr] [point_on_pq x q]
  [line_through_x x qr]
  [line_through_y y pq] : ℝ :=
  XZ_least

-- Statement expressing the proof we need to show
theorem least_length_xz_zero :
  ∀ (P Q R : Type) (p : P) (q : Q) (r : R)
    (X : Type) (x : X)
    (Y : Type) (y : Y)
    (Z : Type) (z : Z),
    (triangle_pqr p q r) →
    (point_on_pq x q) →
    (line_through_x x qr) →
    (line_through_y y pq) →
    (least_possible_length_of_xz p q r x y z = 0) :=
by 
  intros,
  sorry

end least_length_xz_zero_l226_226414


namespace correct_assignment_statement_l226_226024

def is_assignment_statement (stmt : String) : Prop :=
  stmt = "a = 2a"

theorem correct_assignment_statement : is_assignment_statement "a = 2a" :=
by
  sorry

end correct_assignment_statement_l226_226024


namespace simplify_expression_l226_226076

variable (x y : ℤ)

theorem simplify_expression :
  (2 : ℤ)^(3*y + 2) / (4^(-1:ℤ) + 2^(-1:ℤ)) = (4 * 2^(3*y + 2)) / 3 := 
by
  sorry

end simplify_expression_l226_226076


namespace next_ring_together_l226_226814

def nextRingTime (libraryInterval : ℕ) (fireStationInterval : ℕ) (hospitalInterval : ℕ) (start : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm libraryInterval fireStationInterval) hospitalInterval + start

theorem next_ring_together : nextRingTime 18 24 30 (8 * 60) = 14 * 60 :=
by
  sorry

end next_ring_together_l226_226814


namespace circle_represents_circle_iff_a_nonzero_l226_226861

-- Define the equation given in the problem
def circleEquation (a x y : ℝ) : Prop :=
  a*x^2 + a*y^2 - 4*(a-1)*x + 4*y = 0

-- State the required theorem
theorem circle_represents_circle_iff_a_nonzero (a : ℝ) :
  (∃ c : ℝ, ∃ h k : ℝ, ∀ x y : ℝ, circleEquation a x y ↔ (x - h)^2 + (y - k)^2 = c)
  ↔ a ≠ 0 :=
by
  sorry

end circle_represents_circle_iff_a_nonzero_l226_226861


namespace original_number_l226_226938

theorem original_number (x : ℝ) (h1 : 268 * x = 19832) (h2 : 2.68 * x = 1.9832) : x = 74 :=
sorry

end original_number_l226_226938


namespace lateral_surface_area_pyramid_is_correct_l226_226822

-- Given conditions
def is_regular_pyramid (p : Type) [pyramid p] (base_length : ℝ) (height : ℝ) : Prop :=
  base_length = 2 ∧ height = 1

-- Define the lateral surface area calculation
def lateral_surface_area (base_length : ℝ) (height : ℝ) : ℝ :=
  let lateral_height := real.sqrt (height^2 + (base_length / 2)^2) in
  4 * (1 / 2) * base_length * lateral_height

-- Main theorem to be proved
theorem lateral_surface_area_pyramid_is_correct (p : Type) [pyramid p] :
  is_regular_pyramid p 2 1 → lateral_surface_area 2 1 = 4 * real.sqrt 2 :=
by
  intros h
  simp [is_regular_pyramid, lateral_surface_area] at h
  rw h.left
  rw h.right
  sorry -- Proof omitted

end lateral_surface_area_pyramid_is_correct_l226_226822


namespace intervals_of_monotonicity_and_zeros_l226_226554

-- Define the function
def f (x a : ℝ) := (1 / 3) * x^3 + (1 - a) / 2 * x^2 - a * x - a

-- Conditions
variable (x : ℝ)
variable (a : ℝ)
variable (h_a_pos : a > 0)

-- Statement of the problem in Lean
theorem intervals_of_monotonicity_and_zeros (h : 0 < a ∧ a < 1 / 3) : 
  (∀ x ∈ Ioo (-∞) (-1), 0 < (fun x => (x + 1) * (x - a)) x) ∧ 
  (∀ x ∈ Ioo (-1, a), (fun x => (x + 1) * (x - a)) x < 0) ∧ 
  (∀ x ∈ Ioo (a) (+∞), 0 < (fun x => (x + 1) * (x - a)) x) ∧ 
  0 > f (-2) a ∧ 
  0 < f (-1) a ∧ 
  0 > f 0 a :=
sorry

end intervals_of_monotonicity_and_zeros_l226_226554


namespace inequality_part_1_inequality_part_2_l226_226139

noncomputable def f (x : ℝ) := |x - 2| + 2
noncomputable def g (x : ℝ) (m : ℝ) := m * |x|

theorem inequality_part_1 (x : ℝ) : f x > 5 ↔ x < -1 ∨ x > 5 := by
  sorry

theorem inequality_part_2 (m : ℝ) : (∀ x, f x ≥ g x m) ↔ m ≤ 1 := by
  sorry

end inequality_part_1_inequality_part_2_l226_226139


namespace solution_l226_226143

def f (x : ℝ) : ℝ := 5 * x^3

theorem solution (x : ℝ) : f(x) + f(-x) = 0 := 
by
  sorry

end solution_l226_226143


namespace range_of_a_l226_226957

-- Define the function f
def f (x : ℝ) : ℝ :=
  if (0 ≤ x ∧ x ≤ 1/2) then x / 3
  else if (1/2 < x ∧ x ≤ 1) then 2 * x^3 / (x + 1)
  else 0  -- This else is to make it a total function, ensuring x in [0, 1]

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * x - a / 2 + 3

-- Statement of the proof problem
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ (x1 : ℝ), 0 ≤ x1 ∧ x1 ≤ 1 → ∃ (x2 : ℝ), 0 ≤ x2 ∧ x2 ≤ 1/2 ∧ f x1 = g a x2) → (6 ≤ a) :=
by
  sorry

end range_of_a_l226_226957


namespace sin_monotonic_omega_l226_226592

theorem sin_monotonic_omega (ω : ℝ) (h : ω > 0) 
  (mono_inc : ∀ x : ℝ, 0 ≤ x → x ≤ π / 6 → ∀ y : ℝ, 0 ≤ y → y ≤ x → ∂ (sin (ω * y)) = (ω * cos (ω * y)) ≥ 0)
  (mono_dec : ∀ x : ℝ, π / 6 ≤ x → x ≤ π / 2 → ∀ y : ℝ, π / 6 ≤ y → y ≤ x → ∂ (sin (ω * y)) = (ω * cos (ω * y)) ≤ 0) :
  ω = 3 := by
sorry

end sin_monotonic_omega_l226_226592


namespace num_ways_athletes_seated_together_l226_226603

-- Conditions from part a)
def team_A := 2
def team_B := 2
def team_C := 2

-- Objective: total number of ways to arrange six athletes given conditions
theorem num_ways_athletes_seated_together : 
    (fact 3) * (fact team_A) * (fact team_B) * (fact team_C) = 48 := by
  sorry

end num_ways_athletes_seated_together_l226_226603


namespace smaller_octagon_area_fraction_l226_226729

theorem smaller_octagon_area_fraction (A B C D E F G H : Point) (O : Point) :
  is_regular_octagon A B C D E F G H →
  is_center O A B C D E F G H →
  let A' := midpoint A B,
      B' := midpoint B C,
      C' := midpoint C D,
      D' := midpoint D E,
      E' := midpoint E F,
      F' := midpoint F G,
      G' := midpoint G H,
      H' := midpoint H A in
  is_octa_center O A' B' C' D' E' F' G' H' →
  (area_of_octagon A B C D E F G H) * (1 / 4) = area_of_octagon A' B' C' D' E' F' G' H' :=
by
  -- Sorry, proof is omitted.
  sorry

end smaller_octagon_area_fraction_l226_226729


namespace vika_card_pairing_l226_226353

theorem vika_card_pairing :
  ∃ (d ∈ {1, 2, 3, 5, 6, 10, 15, 30}), ∃ (k : ℕ), 60 = 2 * d * k :=
by sorry

end vika_card_pairing_l226_226353


namespace problem1_problem2_problem3_l226_226844

noncomputable def expr1 : ℝ :=
  sqrt 12 + sqrt 20 - 2 * sqrt 3 + sqrt 5

noncomputable def expr2 : ℝ :=
  (sqrt 8 + sqrt (1 / 2)) * sqrt 2 - (sqrt 3 + 1) * (sqrt 3 - 1)

noncomputable def expr3 (x : ℝ) : ℝ :=
  (x / (x - 1) - 1) / ((x^2 + 2 * x + 1) / (x^2 - 1))

theorem problem1 : expr1 = 3 * sqrt 5 := by
  sorry

theorem problem2 : expr2 = 3 := by
  sorry

theorem problem3 : expr3 (sqrt 2 - 1) = sqrt 2 / 2 := by
  sorry

end problem1_problem2_problem3_l226_226844


namespace expression_equals_k_times_10_pow_1007_l226_226047

theorem expression_equals_k_times_10_pow_1007 :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 588 * 10^1007 := by
  sorry

end expression_equals_k_times_10_pow_1007_l226_226047


namespace neg_p_l226_226561

theorem neg_p : ∀ x : ℝ, log 2 (3^x + 1) > 0 := 
by 
  sorry

end neg_p_l226_226561


namespace repeating_decimal_427_diff_l226_226239

theorem repeating_decimal_427_diff :
  let G := 0.427427427427
  let num := 427
  let denom := 999
  num.gcd denom = 1 →
  denom - num = 572 :=
by
  intros G num denom gcd_condition
  sorry

end repeating_decimal_427_diff_l226_226239


namespace angle_Q_of_extended_sides_l226_226288

-- Definitions from Conditions
structure RegularHexagon (A B C D E F : Type) :=
(angles : ∀ (i : Fin 6), ℝ)
(is_regular : ∀ (i : Fin 6), angles i = 120)

section
variables {A B C D E F Q : Type}

-- The specific theorem/proof statement
theorem angle_Q_of_extended_sides (h₁ : RegularHexagon A B C D E F)
    (h₂ : extended_eq (B : A → Q) .extended_eq (extended_eq (C : D → Q))) :
    ∠ Q = 60 :=
  sorry
end

end angle_Q_of_extended_sides_l226_226288


namespace triangle_side_lengths_l226_226746

/-- The side lengths of a triangle are consecutive integers with one of its medians perpendicular
to one of its angle bisectors, imply that the side lengths are 2, 3, and 4 --/
theorem triangle_side_lengths (a b c : ℕ) (h1 : a < b ∧ b < c) (h2 : b = a + 1) (h3 : c = b + 1)
  (median_perpendicular_bisector : ∃ (median : ℝ) (bisector : ℝ), median ⊥ bisector):
  a = 2 ∧ b = 3 ∧ c = 4 :=
sorry

end triangle_side_lengths_l226_226746


namespace sum_of_perfect_squares_lt_500_l226_226273

theorem sum_of_perfect_squares_lt_500 : 
  (∑ n in finset.range 23, n * n) = 3795 :=
  sorry

end sum_of_perfect_squares_lt_500_l226_226273


namespace volumes_comparison_l226_226328

variable (a : ℝ) (h_a : a ≠ 3)

def volume_A := 3 * 3 * 3
def volume_B := 3 * 3 * a
def volume_C := a * a * 3
def volume_D := a * a * a

theorem volumes_comparison (h_a : a ≠ 3) :
  (volume_A + volume_D) > (volume_B + volume_C) :=
by
  have volume_A : ℝ := 27
  have volume_B := 9 * a
  have volume_C := 3 * a * a
  have volume_D := a * a * a
  sorry

end volumes_comparison_l226_226328


namespace sum_min_max_prime_factors_of_1365_l226_226774

-- Define a function to check if a number is prime (for manual handling in proof)
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the number we're working with
def number := 1365

-- Define the list of known prime factors
def prime_factors : list ℕ := [3, 5, 7, 13]

-- Ensure all elements of prime_factors are prime
lemma all_primes : ∀ p ∈ prime_factors, is_prime p := by
  intros p h
  cases h
  case or.inl h_0 => exact nat.prime_three
  case or.inr h_0 => cases h_0
    case or.inl h_1 => exact nat.prime_five
    case or.inr h_1 => cases h_1
      case or.inl h_2 => exact nat.prime_seven
      case or.inr h_2 => exact nat.prime_thirteen

-- Define the sum of the largest and smallest prime factors
def sum_of_min_and_max : ℕ := list.minimum' prime_factors + list.maximum' prime_factors

-- Theorem stating that this sum is 16
theorem sum_min_max_prime_factors_of_1365 : sum_of_min_and_max = 16 := by
  sorry

end sum_min_max_prime_factors_of_1365_l226_226774


namespace sum_of_prime_factors_of_1365_l226_226781

theorem sum_of_prime_factors_of_1365 : 
  let smallest_prime_factor := 3 in
  let largest_prime_factor := 13 in
  smallest_prime_factor + largest_prime_factor = 16 :=
by
  let smallest_prime_factor := 3
  let largest_prime_factor := 13
  have h : smallest_prime_factor + largest_prime_factor = 16 := by
    -- the proof will be provided here, but it is not required in this step
    sorry
  exact h

end sum_of_prime_factors_of_1365_l226_226781


namespace shopkeeper_sold_200_metres_l226_226011

-- This definition encapsulates all given conditions
def shopkeeper_conditions (x : ℕ) : Prop :=
  let CP_per_metre := 72
  let Loss_per_metre := 12
  let SP_per_metre := CP_per_metre - Loss_per_metre
  SP_per_metre * x = 12000 ∧ SP_per_metre = 60

theorem shopkeeper_sold_200_metres : ∃ (x : ℕ), shopkeeper_conditions x ∧ x = 200 :=
by
  existsi 200
  unfold shopkeeper_conditions
  simp
  split
  { sorry } -- This simplifies to the equation 60 * 200 = 12000
  { refl }  -- This simplifies to the equation 60 = 72 - 12

end shopkeeper_sold_200_metres_l226_226011


namespace general_term_formula_sum_of_first_n_terms_l226_226948

noncomputable def a (n : ℕ) : ℝ := 3 ^ n - 1

def S (n : ℕ) : ℝ := (a (n + 1)) / 2 - n - 1

def terms (n : ℕ) : ℝ := 2 * 3 ^ n / (a n * a (n + 1))

def T (n : ℕ) : ℝ :=
  ∑ i in range n, terms i

theorem general_term_formula (n : ℕ) :
  a n = 3 ^ n - 1 :=
sorry

theorem sum_of_first_n_terms (n : ℕ) :
  T n = 1 / 2 - 1 / (3 ^ (n + 1) - 1) :=
sorry

end general_term_formula_sum_of_first_n_terms_l226_226948


namespace circle_equation_l226_226942

noncomputable def circle_through_points (x1 x2 a b : ℝ) : Prop :=
  let y1 := x1 - 1 in
  let y2 := x2 - 1 in
  let radius := abs (a + 1) in
  let midpoint_x := (x1 + x2) / 2 in
  let midpoint_y := midpoint_x - 1 in
  let ab_len := real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) in
  let distance := abs (a - b - 1) / real.sqrt 2 in
  let lhs := radius^2 in
  let rhs := 16 + 2 * (a - midpoint_x)^2 in
  (x1^2 - 6 * x1 + 1 = 0) ∧ (x2^2 - 6 * x2 + 1 = 0) ∧
  (a + b = 5) ∧
  (midpoint_x = 3) ∧
  (midpoint_y = 2) ∧
  (abs (a - midpoint_x) = distance) ∧
  (lhs = rhs) ∧
  ((a = 3 ∧ b = 2) ∨ (a = 11 ∧ b = -6))

theorem circle_equation (x1 x2 : ℝ) :
  circle_through_points x1 x2 3 2 ∨ circle_through_points x1 x2 11 (-6) :=
sorry

end circle_equation_l226_226942


namespace trig_expression_eval_l226_226866

theorem trig_expression_eval :
  sin (26 * Real.pi / 3) + cos (-17 * Real.pi / 4) = (Real.sqrt 3 + Real.sqrt 2) / 2 :=
sorry

end trig_expression_eval_l226_226866


namespace find_ratio_l226_226094

-- Definition of the system of equations with k = 5
def system_of_equations (x y z : ℝ) :=
  x + 10 * y + 5 * z = 0 ∧
  2 * x + 5 * y + 4 * z = 0 ∧
  3 * x + 6 * y + 5 * z = 0

-- Proof that if (x, y, z) solves the system, then yz / x^2 = -3 / 49
theorem find_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : system_of_equations x y z) :
  (y * z) / (x ^ 2) = -3 / 49 :=
by
  -- Substitute the system of equations and solve for the ratio.
  sorry

end find_ratio_l226_226094


namespace trigonometric_identity_l226_226100

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = 2) :
  (4 * Real.sin α ^ 3 - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 2 / 5 :=
by
  sorry

end trigonometric_identity_l226_226100


namespace bobs_sisters_mile_time_l226_226034

theorem bobs_sisters_mile_time (bobs_current_time_minutes : ℕ) (bobs_current_time_seconds : ℕ) (improvement_percentage : ℝ) :
  bobs_current_time_minutes = 10 → bobs_current_time_seconds = 40 → improvement_percentage = 9.062499999999996 →
  bobs_sisters_time_minutes = 9 ∧ bobs_sisters_time_seconds = 42 :=
by
  -- Definitions from conditions
  let bobs_time_in_seconds := bobs_current_time_minutes * 60 + bobs_current_time_seconds
  let improvement_in_seconds := bobs_time_in_seconds * improvement_percentage / 100
  let target_time_in_seconds := bobs_time_in_seconds - improvement_in_seconds
  let bobs_sisters_time_minutes := target_time_in_seconds / 60
  let bobs_sisters_time_seconds := target_time_in_seconds % 60
  
  sorry

end bobs_sisters_mile_time_l226_226034


namespace cakes_served_yesterday_l226_226447

theorem cakes_served_yesterday (lunch_cakes dinner_cakes total_cakes served_yesterday : ℕ)
  (h1 : lunch_cakes = 5)
  (h2 : dinner_cakes = 6)
  (h3 : total_cakes = 14)
  (h4 : total_cakes = lunch_cakes + dinner_cakes + served_yesterday) :
  served_yesterday = 3 := 
by 
  sorry

end cakes_served_yesterday_l226_226447


namespace bananas_left_l226_226496

theorem bananas_left (original_bananas : ℕ) (bananas_eaten : ℕ) 
  (h1 : original_bananas = 12) (h2 : bananas_eaten = 4) : 
  original_bananas - bananas_eaten = 8 := 
by
  sorry

end bananas_left_l226_226496


namespace recurring_decimal_to_rational_l226_226068

def recurring_decimal_rational (x : ℚ) : Prop :=
  x = 125634 / 999999

theorem recurring_decimal_to_rational : 
  recurring_decimal_rational 0.125634125634... := sorry

end recurring_decimal_to_rational_l226_226068


namespace total_books_of_gwen_l226_226799

theorem total_books_of_gwen 
  (mystery_shelves : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ)
  (h1 : mystery_shelves = 3) (h2 : picture_shelves = 5) (h3 : books_per_shelf = 9) : 
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 72 :=
by
  -- Given:
  -- 1. Gwen had 3 shelves of mystery books.
  -- 2. Each shelf had 9 books.
  -- 3. Gwen had 5 shelves of picture books.
  -- 4. Each shelf had 9 books.
  -- Prove:
  -- The total number of books Gwen had is 72.
  sorry

end total_books_of_gwen_l226_226799


namespace problem1_problem2_problem3_l226_226983

-- Problem 1
theorem problem1 (m : ℝ) (h : -m^2 = m) : m^2 + m + 1 = 1 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) (h : m - n = 2) : 2 * (n - m) - 4 * m + 4 * n - 3 = -15 :=
by sorry

-- Problem 3
theorem problem3 (m n : ℝ) (h1 : m^2 + 2 * m * n = -2) (h2 : m * n - n^2 = -4) : 
  3 * m^2 + (9 / 2) * m * n + (3 / 2) * n^2 = 0 :=
by sorry

end problem1_problem2_problem3_l226_226983


namespace prove_sequence_l226_226112

def seq (n : ℕ) : ℕ → ℕ
| 0     => 2
| (n+1) => if n = 0 then 2 else 1

def sum_seq (n : ℕ) : ℕ := (n + 1) * seq (n + 1)

theorem prove_sequence :
  (seq 1 1 = 2) ∧ (∀ n : ℕ, n ≥ 2 → seq 1 n = 1) ∧ 
  (∀ n : ℕ, sum_seq n = (n + 1) * seq (n + 1)) :=
by
  sorry

end prove_sequence_l226_226112


namespace smaller_octagon_area_fraction_l226_226723

theorem smaller_octagon_area_fraction (A B C D E F G H : Point)
  (midpoints_joined : Boolean)
  (regular_octagon : RegularOctagon A B C D E F G H)
  (smaller_octagon : Octagon (midpoint (A, B)) (midpoint (B, C)) (midpoint (C, D)) 
                              (midpoint (D, E)) (midpoint (E, F)) (midpoint (F, G))
                              (midpoint (G, H)) (midpoint (H, A))) :
  midpoints_joined → regular_octagon → 
  (area smaller_octagon) = (3 / 4) * (area regular_octagon) :=
by
  sorry

end smaller_octagon_area_fraction_l226_226723


namespace harry_lost_sea_creatures_l226_226964

def initial_sea_stars := 34
def initial_seashells := 21
def initial_snails := 29
def initial_crabs := 17

def sea_stars_reproduced := 5
def seashells_reproduced := 3
def snails_reproduced := 4

def final_items := 105

def sea_stars_after_reproduction := initial_sea_stars + (sea_stars_reproduced * 2 - sea_stars_reproduced)
def seashells_after_reproduction := initial_seashells + (seashells_reproduced * 2 - seashells_reproduced)
def snails_after_reproduction := initial_snails + (snails_reproduced * 2 - snails_reproduced)
def crabs_after_reproduction := initial_crabs

def total_after_reproduction := sea_stars_after_reproduction + seashells_after_reproduction + snails_after_reproduction + crabs_after_reproduction

theorem harry_lost_sea_creatures : total_after_reproduction - final_items = 8 :=
by
  sorry

end harry_lost_sea_creatures_l226_226964


namespace y_intercept_of_line_through_PQ_l226_226698

theorem y_intercept_of_line_through_PQ :
  ∀ (P Q : ℝ × ℝ), P = (0, 1) → Q = (2, 1) → ∃ b, b = 1 :=
by
  intros P Q hP hQ
  exists 1
  sorry

end y_intercept_of_line_through_PQ_l226_226698


namespace inequality_geq_27_l226_226257

theorem inequality_geq_27 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_eq : a + b + c + 2 = a * b * c) : (a + 1) * (b + 1) * (c + 1) ≥ 27 := 
    sorry

end inequality_geq_27_l226_226257


namespace tank_holds_gallons_l226_226439

noncomputable def tank_initial_fraction := (7 : ℚ) / 8
noncomputable def tank_partial_fraction := (2 : ℚ) / 3
def gallons_used := 15

theorem tank_holds_gallons
  (x : ℚ) -- number of gallons the tank holds when full
  (h_initial : tank_initial_fraction * x - gallons_used = tank_partial_fraction * x) :
  x = 72 := 
sorry

end tank_holds_gallons_l226_226439


namespace range_of_sqrt_expr_l226_226590

theorem range_of_sqrt_expr (x : ℝ) (h : ∃ y : ℝ, y = sqrt (x - 2)) : x ≥ 2 :=
by
  sorry

end range_of_sqrt_expr_l226_226590


namespace pairA_neq_pairB_neq_pairC_neq_pairD_eq_true_l226_226023

noncomputable def pairA_eq : Prop :=
  (-(-2)) = -|-2|

noncomputable def pairB_eq : Prop :=
  -(2^2) = (-2)^2

noncomputable def pairC_eq : Prop :=
  ((-1/3)^3) = -(1^3)/3

noncomputable def pairD_eq : Prop :=
  (| -8 |)^2 = -(-4)^3

theorem pairA_neq : not pairA_eq := by
  sorry

theorem pairB_neq : not pairB_eq := by
  sorry

theorem pairC_neq : not pairC_eq := by
  sorry

theorem pairD_eq_true : pairD_eq := by
  sorry

end pairA_neq_pairB_neq_pairC_neq_pairD_eq_true_l226_226023


namespace sum_100_consecutive_from_neg49_l226_226393

noncomputable def sum_of_consecutive_integers (n : ℕ) (first_term : ℤ) : ℤ :=
  n * ( first_term + (first_term + n - 1) ) / 2

theorem sum_100_consecutive_from_neg49 : sum_of_consecutive_integers 100 (-49) = 50 :=
by sorry

end sum_100_consecutive_from_neg49_l226_226393


namespace vector_projection_example_l226_226513

noncomputable def vector_projection (v d : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := v.1 * d.1 + v.2 * d.2 + v.3 * d.3
  let magnitude_sq := d.1^2 + d.2^2 + d.3^2
  let scale := dot_product / magnitude_sq
  (scale * d.1, scale * d.2, scale * d.3)

theorem vector_projection_example :
  vector_projection (4, -1, 3) (3, 1, -2) = (15 / 14, 5 / 14, -5 / 7) := sorry

end vector_projection_example_l226_226513


namespace united_telephone_additional_charge_l226_226766

theorem united_telephone_additional_charge :
  ∃ x : ℝ, 
    (11 + 20 * x = 16) ↔ (x = 0.25) := by
  sorry

end united_telephone_additional_charge_l226_226766


namespace molecular_weight_of_9_moles_l226_226771

theorem molecular_weight_of_9_moles (molecular_weight : ℕ) (moles : ℕ) (h₁ : molecular_weight = 1098) (h₂ : moles = 9) :
  molecular_weight * moles = 9882 :=
by {
  sorry
}

end molecular_weight_of_9_moles_l226_226771


namespace scale_division_remainder_l226_226618

theorem scale_division_remainder (a b c r : ℕ) (h1 : a = b * c + r) (h2 : 0 ≤ r) (h3 : r < b) :
  (3 * a) % (3 * b) = 3 * r :=
sorry

end scale_division_remainder_l226_226618


namespace vector_dot_product_self_eq_36_l226_226969

variable {R : Type*} [RealSpace R]

-- Assume u to be a vector in 3D space with coordinates (a, b, c)
variable (a b c : ℝ)
def u : EuclideanSpace (Fin 3) ℝ := ![a, b, c]

-- The magnitude of the vector u is given to be 6
axiom magnitude_u : ‖u‖ = 6

-- We need to prove that u · u = 36
theorem vector_dot_product_self_eq_36 : u ∙ u = 36 :=
by
  sorry

end vector_dot_product_self_eq_36_l226_226969


namespace integer_solutions_count_l226_226171

theorem integer_solutions_count :
  let eq : Int -> Int -> Int := fun x y => 6 * y ^ 2 + 3 * x * y + x + 2 * y - 72
  ∃ (sols : List (Int × Int)), 
    (∀ x y, eq x y = 0 → (x, y) ∈ sols) ∧
    (∀ p ∈ sols, ∃ x y, p = (x, y) ∧ eq x y = 0) ∧
    sols.length = 4 :=
by
  sorry

end integer_solutions_count_l226_226171


namespace seventh_term_in_geometric_sequence_l226_226885

-- Define the geometric sequence conditions
def first_term : ℝ := 3
def second_term : ℝ := -1/2
def common_ratio : ℝ := second_term / first_term

-- Define the formula for the nth term of the geometric sequence
def nth_term (a r : ℝ) (n : ℕ) : ℝ := a * r^(n-1)

-- The Lean statement for proving the seventh term in the geometric sequence
theorem seventh_term_in_geometric_sequence :
  nth_term first_term common_ratio 7 = 1 / 15552 :=
by
  -- The proof is to be filled in.
  sorry

end seventh_term_in_geometric_sequence_l226_226885


namespace alec_correct_problems_l226_226227

-- Definitions of conditions and proof problem
theorem alec_correct_problems (c w : ℕ) (s : ℕ) (H1 : s = 30 + 4 * c - w) (H2 : s > 90)
  (H3 : ∀ s', 90 < s' ∧ s' < s → ¬(∃ c', ∃ w', s' = 30 + 4 * c' - w')) :
  c = 16 :=
by
  sorry

end alec_correct_problems_l226_226227


namespace calculate_value_l226_226895

def f (x : ℝ) : ℝ := 9 - x
def g (x : ℝ) : ℝ := x - 9

theorem calculate_value : g (f 15) = -15 := by
  sorry

end calculate_value_l226_226895


namespace rectangle_breadth_l226_226696

/-- The breadth of the rectangle is 10 units given that
1. The length of the rectangle is two-fifths of the radius of a circle.
2. The radius of the circle is equal to the side of the square.
3. The area of the square is 1225 sq. units.
4. The area of the rectangle is 140 sq. units. -/
theorem rectangle_breadth (r l b : ℝ) (h_radius : r = 35) (h_length : l = (2 / 5) * r) (h_square : 35 * 35 = 1225) (h_area_rect : l * b = 140) : b = 10 :=
by
  sorry

end rectangle_breadth_l226_226696


namespace geometric_mean_45_80_l226_226692

theorem geometric_mean_45_80 : ∃ x : ℝ, x^2 = 45 * 80 ∧ (x = 60 ∨ x = -60) := 
by 
  sorry

end geometric_mean_45_80_l226_226692


namespace locus_of_midpoints_is_circle_l226_226240

open EuclideanGeometry

theorem locus_of_midpoints_is_circle
  (K : Circle)
  (P : Point)
  (r : ℝ)
  (hP_in_K : Point_in_circle P K)
  (O : Point := K.center)
  (h_distance_OP : dist P O = r / 3) :
  ∃ (C : Circle) (M : Point_in_circle P C), 
  ∀ (AB : Chord),
  chord_passes_through_point AB P →
  midpoint AB ∈ circle_locus C :=
sorry

end locus_of_midpoints_is_circle_l226_226240


namespace sum_first_100_terms_l226_226529

theorem sum_first_100_terms (a b : ℤ) (a_n : ℕ → ℤ) 
  (h1 : a_n 1 = a) 
  (h2 : a_n 2 = b) 
  (h3 : ∀ n ≥ 2, a_n (n + 1) = a_n n - a_n (n - 1)) :
  (∑ i in finset.range 100, a_n (i + 1)) = 2 * b - a :=
by
  sorry

end sum_first_100_terms_l226_226529


namespace f_at_neg1_eq_neg1_l226_226967

noncomputable def g (x : ℝ) : ℝ := 1 - 2 * x
noncomputable def f (x : ℝ) : ℝ := by
  have h := log 2 (1 / (g⁻¹ x + 1))
  exact h

theorem f_at_neg1_eq_neg1 : f (-1) = -1 := by
  sorry

end f_at_neg1_eq_neg1_l226_226967


namespace sqrt_4_minus_1_l226_226846

  theorem sqrt_4_minus_1 : sqrt 4 - 1 = 1 := by
    sorry
  
end sqrt_4_minus_1_l226_226846


namespace volume_of_spherical_layer_l226_226476

theorem volume_of_spherical_layer (r1 r2 h : ℝ) : 
  let V := (π * h * (3 * r1^2 + 3 * r2^2 + h^2)) / 6
  in V = (π * h / 6) * (3 * r1^2 + 3 * r2^2 + h^2) :=
by
  sorry

end volume_of_spherical_layer_l226_226476


namespace angle_Q_l226_226289

-- Definitions for the conditions given in the problem
def is_regular_hexagon (A B C D E F : Type) : Prop :=
  ∀ (angle : ℕ), angle ∈ { ∠A B C, ∠B C D, ∠C D E, ∠D E F, ∠E F A, ∠F A B } → angle = 120

def extended_to_meet_at (A B C D Q : Type) : Prop :=
  ∃ P : Type, line A B = line P Q ∧ line C D = line P Q

-- The theorem we need to prove
theorem angle_Q (A B C D E F Q : Type) (H_hex : is_regular_hexagon A B C D E F)
  (H_meet : extended_to_meet_at A B C D Q) : ∠A Q C = 120 :=
by sorry

end angle_Q_l226_226289


namespace subset_zero_in_A_l226_226186

def A := { x : ℝ | x > -1 }

theorem subset_zero_in_A : {0} ⊆ A :=
by sorry

end subset_zero_in_A_l226_226186


namespace vectors_relationship_l226_226156

def vector_a : (ℝ × ℝ × ℝ) := (-2, -3, 1)
def vector_b : (ℝ × ℝ × ℝ) := (2, 0, 4)
def vector_c : (ℝ × ℝ × ℝ) := (-4, -6, 2)

def dot_product (u v : (ℝ × ℝ × ℝ)) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def is_parallel (u v : (ℝ × ℝ × ℝ)) : Prop :=
  ∃ k : ℝ, v.1 = k * u.1 ∧ v.2 = k * u.2 ∧ v.3 = k * u.3

def is_orthogonal (u v : (ℝ × ℝ × ℝ)) : Prop :=
  dot_product u v = 0

theorem vectors_relationship :
  is_parallel vector_a vector_c ∧ is_orthogonal vector_a vector_b :=
by {
  -- The proof will be filled in here
  sorry,
}

end vectors_relationship_l226_226156


namespace rate_of_current_l226_226443

theorem rate_of_current
  (D U R : ℝ)
  (hD : D = 45)
  (hU : U = 23)
  (hR : R = 34)
  : (D - R = 11) ∧ (R - U = 11) :=
by
  sorry

end rate_of_current_l226_226443


namespace larger_number_is_38_l226_226749

theorem larger_number_is_38 (x y : ℕ) (h1 : x + y = 64) (h2 : y = x + 12) : y = 38 :=
by
  sorry

end larger_number_is_38_l226_226749


namespace hexagon_angle_in_arithmetic_progression_l226_226679

theorem hexagon_angle_in_arithmetic_progression :
  ∃ (a d : ℝ), (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) = 720) ∧ 
  (a = 120 ∨ a + d = 120 ∨ a + 2 * d = 120 ∨ a + 3 * d = 120 ∨ a + 4 * d = 120 ∨ a + 5 * d = 120) := by
  sorry

end hexagon_angle_in_arithmetic_progression_l226_226679


namespace base_length_of_triangle_l226_226683

theorem base_length_of_triangle (height area : ℕ) (h1 : height = 8) (h2 : area = 24) : 
  ∃ base : ℕ, (1/2 : ℚ) * base * height = area ∧ base = 6 := by
  sorry

end base_length_of_triangle_l226_226683


namespace range_of_a_l226_226968

theorem range_of_a (a : ℝ) : (a+1)⁻¹ < (3-2a)⁻¹ → (2/3 < a ∧ a < 3/2) :=
by
  sorry

end range_of_a_l226_226968


namespace smaller_octagon_half_area_l226_226700

-- Define what it means to be a regular octagon
def is_regular_octagon (O : Point) (ABCDEFGH : List Point) : Prop :=
  -- Definition capturing the properties of a regular octagon around center O
  sorry

-- Define the function that computes the area of an octagon
def area_of_octagon (ABCDEFGH : List Point) : Real :=
  sorry

-- Define the function to create the smaller octagon by joining midpoints
def smaller_octagon (ABCDEFGH : List Point) : List Point :=
  sorry

theorem smaller_octagon_half_area (O : Point) (ABCDEFGH : List Point) :
  is_regular_octagon O ABCDEFGH →
  area_of_octagon (smaller_octagon ABCDEFGH) = (1 / 2) * area_of_octagon ABCDEFGH :=
by
  sorry

end smaller_octagon_half_area_l226_226700


namespace simplify_complex_fraction_l226_226674

theorem simplify_complex_fraction :
  let numerator := (5 : ℂ) + 7 * I
  let denominator := (2 : ℂ) + 3 * I
  numerator / denominator = (31 / 13 : ℂ) - (1 / 13) * I :=
by
  let numerator := (5 : ℂ) + 7 * I
  let denominator := (2 : ℂ) + 3 * I
  sorry

end simplify_complex_fraction_l226_226674


namespace tank_fill_time_l226_226015

theorem tank_fill_time (R L : ℝ) (h1 : (R - L) * 8 = 1) (h2 : L * 56 = 1) :
  (1 / R) = 7 :=
by
  sorry

end tank_fill_time_l226_226015


namespace deepak_and_wife_meet_time_l226_226695

noncomputable def meet_time (circumference : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  circumference / (speed1 + speed2)

theorem deepak_and_wife_meet_time :
  meet_time 627 (4.5 * 1000 / 60) (3.75 * 1000 / 60) ≈ 4.56 :=
by
  sorry

end deepak_and_wife_meet_time_l226_226695


namespace Ann_keeps_total_cookies_l226_226835

theorem Ann_keeps_total_cookies :
  let baked_oatmeal_raisin := 3 * 12,
      baked_sugar := 2 * 12,
      baked_chocolate_chip := 4 * 12,
      given_away_oatmeal_raisin := 2 * 12,
      given_away_sugar := 1.5 * 12,
      given_away_chocolate_chip := 2.5 * 12,
      kept_oatmeal_raisin := baked_oatmeal_raisin - given_away_oatmeal_raisin,
      kept_sugar := baked_sugar - given_away_sugar,
      kept_chocolate_chip := baked_chocolate_chip - given_away_chocolate_chip,
      total_kept := kept_oatmeal_raisin + kept_sugar + kept_chocolate_chip
  in total_kept = 36 :=
by
  sorry

end Ann_keeps_total_cookies_l226_226835


namespace problem_solution_l226_226131

theorem problem_solution :
  (∃ x a : ℝ, 2 * (x - 6) = -16 ∧ a * (x + 3) = (1 / 2) * a + x) →
  (∃ a : ℝ, a = -4 ∧ a^2 - (1 / 2) * a + 1 = 19) :=
by
  intro h
  obtain ⟨x, a, h1, h2⟩ := h
  have x_val : x = -2 := sorry
  have a_val : a = -4 := sorry
  use a
  exact ⟨a_val, by simp [a_val]⟩

end problem_solution_l226_226131


namespace cone_base_to_lateral_area_ratio_l226_226973

theorem cone_base_to_lateral_area_ratio (r : ℝ) (h : ℝ) (h_eq_2r : h = 2 * r) :
  let base_area := real.pi * r^2,
      slant_height := real.sqrt (h^2 + r^2),
      lateral_area := real.pi * r * slant_height in
  base_area / lateral_area = 1 / real.sqrt 5 :=
by
  let base_area := real.pi * r^2,
      slant_height := real.sqrt (h^2 + r^2),
      lateral_area := real.pi * r * slant_height
  sorry

end cone_base_to_lateral_area_ratio_l226_226973


namespace area_ratio_of_smaller_octagon_l226_226711

theorem area_ratio_of_smaller_octagon (A B C D E F G H P Q R S T U V W : Point) 
  (h1 : is_regular_octagon A B C D E F G H)
  (h2 : midpoint A B = P) (h3 : midpoint B C = Q) (h4 : midpoint C D = R)
  (h5 : midpoint D E = S) (h6 : midpoint E F = T) (h7 : midpoint F G = U)
  (h8 : midpoint G H = V) (h9 : midpoint H A = W):
  area (octagon A B C D E F G H) / area (octagon P Q R S T U V W) = 4 := sorry

end area_ratio_of_smaller_octagon_l226_226711


namespace triangle_area_of_right_triangle_l226_226293

-- Definitions as per the conditions
variables {A B C P : Type} [inner_product_space ℝ A]
variables (a b c p : A)
variable (angle : A → A → ℝ)

-- Definition that triangle ABC is a scalene right triangle
def is_right_triangle (a b c : A) (angle : A → A → ℝ) : Prop :=
  ∃ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (angle x y = π / 2 ∨ angle y z = π / 2 ∨ angle z x = π / 2) ∧
  (x = a ∨ y = a ∨ z = a) ∧ (x = b ∨ y = b ∨ z = b) ∧ (x = c ∨ y = c ∨ z = c)
 
-- Definitions as you might reason
def is_point_on_hypotenuse (a c p : A) : Prop :=
  norm (a - c) = (norm (a - p) + norm (p - c)) ∧
   -1 <= (inner_product_space.angle a p) / real.pi <= 1

-- Defining angle condition, point p and side lengths
def angle_30_deg (a b p : A) (angle : A → A → ℝ) : Prop :=
  angle a b = real.pi / 6

def side_lengths (a c p : A) : Prop :=
  norm (a - p) = 1 ∧ norm (p - c) = 3

-- Prove the required area
theorem triangle_area_of_right_triangle (a b c p : A) (angle : A → A → ℝ) 
  (h1 : is_right_triangle a b c angle)
  (h2 : is_point_on_hypotenuse a c p)
  (h3 : angle_30_deg a b p angle)
  (h4 : side_lengths a c p) :
  let ab := norm (a - b)
  let bc := norm (b - c)
  (1/2 * ab * bc = 2 * real.sqrt 3) :=
sorry

end triangle_area_of_right_triangle_l226_226293


namespace pentagon_segments_l226_226056

variables {a b c d e : ℝ}

theorem pentagon_segments (h_segments_order : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
                          (h_sum_eq_one : a + b + c + d + e = 1) :
  (a < 1/2 ∧ b < 1/2 ∧ c < 1/2 ∧ d < 1/2 ∧ e < 1/2) ↔ 
  ((a + b + c + d > e) ∧ (a + b + c + e > d) ∧ (a + b + d + e > c) ∧ (a + c + d + e > b) ∧ (b + c + d + e > a)) :=
begin
  sorry
end

end pentagon_segments_l226_226056


namespace angle_Q_of_extended_sides_l226_226286

-- Definitions from Conditions
structure RegularHexagon (A B C D E F : Type) :=
(angles : ∀ (i : Fin 6), ℝ)
(is_regular : ∀ (i : Fin 6), angles i = 120)

section
variables {A B C D E F Q : Type}

-- The specific theorem/proof statement
theorem angle_Q_of_extended_sides (h₁ : RegularHexagon A B C D E F)
    (h₂ : extended_eq (B : A → Q) .extended_eq (extended_eq (C : D → Q))) :
    ∠ Q = 60 :=
  sorry
end

end angle_Q_of_extended_sides_l226_226286


namespace ninth_term_arithmetic_sequence_l226_226691

def first_term : ℚ := 3 / 4
def seventeenth_term : ℚ := 6 / 7

theorem ninth_term_arithmetic_sequence :
  let a1 := first_term
  let a17 := seventeenth_term
  (a1 + a17) / 2 = 45 / 56 := 
sorry

end ninth_term_arithmetic_sequence_l226_226691


namespace Ray_has_4_nickels_left_l226_226284

def Ray_initial_cents := 95
def Ray_cents_to_Peter := 25
def Ray_cents_to_Randi := 2 * Ray_cents_to_Peter

-- There are 5 cents in each nickel
def cents_per_nickel := 5

-- Nickels Ray originally has
def Ray_initial_nickels := Ray_initial_cents / cents_per_nickel
-- Nickels given to Peter
def Ray_nickels_to_Peter := Ray_cents_to_Peter / cents_per_nickel
-- Nickels given to Randi
def Ray_nickels_to_Randi := Ray_cents_to_Randi / cents_per_nickel
-- Total nickels given away
def Ray_nickels_given_away := Ray_nickels_to_Peter + Ray_nickels_to_Randi
-- Nickels left with Ray
def Ray_nickels_left := Ray_initial_nickels - Ray_nickels_given_away

theorem Ray_has_4_nickels_left :
  Ray_nickels_left = 4 :=
by
  sorry

end Ray_has_4_nickels_left_l226_226284


namespace prime_factors_correct_sum_of_largest_and_smallest_prime_factors_l226_226777

/-- The number we are considering is 1365. -/
def number := 1365

/-- Hypothesis: The prime factors of 1365 are exactly 3, 5, 7, and 13. -/
def prime_factors_of_1365 : List ℕ := [3, 5, 7, 13]

/-- Hypothesis: The prime factors are prime numbers. -/
theorem prime_factors_correct (n : ℕ) (hn : n ∈ prime_factors_of_1365) : Nat.Prime n :=
begin
  rw prime_factors_of_1365 at hn,
  fin_cases hn,
  { exact Nat.prime_of_nat 3 },
  { exact Nat.prime_of_nat 5 },
  { exact Nat.prime_of_nat 7 },
  { exact Nat.prime_of_nat 13 },
end

/-- We state the final proof we need to show that the sum is 16. -/
theorem sum_of_largest_and_smallest_prime_factors : 
  let min_prime := List.minimum prime_factors_of_1365
  let max_prime := List.maximum prime_factors_of_1365
  min_prime + max_prime = 16 :=
begin
  sorry
end

end prime_factors_correct_sum_of_largest_and_smallest_prime_factors_l226_226777


namespace collinearity_MHS_l226_226620

-- Setup the triangle ABC and orthocenter H
variables {α : Type*} [Euclidean_space α] (A B C H : α) 

-- Define altitudes AD, BE, CF intersecting at orthocenter H
variables {D E F : α} 

-- circumcircle Γ_0 of triangle ABC and M as intersection point of line EF with Γ_0
variables (Γ_0 : circle α) {M : α} (h1 : collinear [E, F, M]) (h2 : point_on_circle Γ_0 M)

-- define intersection points P and Q
variables {P Q : α} (h3 : intersection (line D F) (line B E) P) (h4 : intersection (line D E) (line C F) Q)

-- circumcircle Γ of triangle DEF and S as intersection of line PQ with Γ
variables (Γ : circle α) {S : α} (h5 : point_on_circle Γ S) (h6 : collinear [P, Q, S])

-- Prove collinearity of M, H, and S
theorem collinearity_MHS : collinear [M, H, S] := 
sorry

end collinearity_MHS_l226_226620


namespace tom_searching_days_l226_226332

variable (d : ℕ) (total_cost : ℕ)

theorem tom_searching_days :
  (∀ n, n ≤ 5 → total_cost = n * 100 + (d - n) * 60) →
  (∀ n, n > 5 → total_cost = 5 * 100 + (d - 5) * 60) →
  total_cost = 800 →
  d = 10 :=
by
  intros h1 h2 h3
  sorry

end tom_searching_days_l226_226332


namespace division_value_l226_226407

theorem division_value (x : ℝ) : (9 / x) * 12 = 18 → x = 6 :=
by
  intro h
  rw mul_eq_iff_eq_div at h
  rw div_eq_iff_eq_mul at h
  sorry

end division_value_l226_226407


namespace trig_identity_l226_226049

theorem trig_identity :
  let tan30 := (Real.sqrt 3) / 3
  let cos60 := 1 / 2
  let sin45 := (Real.sqrt 2) / 2
  sqrt 3 * tan30 * cos60 + (sin45) ^ 2 = 1 := by
  sorry

end trig_identity_l226_226049


namespace problem_1_problem_2_l226_226145

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem problem_1 : {x : ℝ | f x > 2} = {x : ℝ | x < -1 / 2 ∨ x > 3 / 2} := sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x + |2 * (x + 3)| - 4 > m * x) → m ≤ -11 := sorry

end problem_1_problem_2_l226_226145


namespace find_m_plus_b_l226_226860

-- Definitions for point coordinates
def x1 : ℝ := 2
def y1 : ℝ := -1
def x2 : ℝ := -1
def y2 : ℝ := 6

-- Definition of the slope
def m : ℝ := (y2 - y1) / (x2 - x1)

-- Definition of the y-intercept from point-slope form
def b : ℝ := y1 - m * x1

-- Theorem statement for m + b
theorem find_m_plus_b : m + b = 4 / 3 := by
  sorry

end find_m_plus_b_l226_226860


namespace final_concentration_after_procedure_l226_226438

open Real

def initial_salt_concentration : ℝ := 0.16
def final_salt_concentration : ℝ := 0.107

def volume_ratio_large : ℝ := 10
def volume_ratio_medium : ℝ := 4
def volume_ratio_small : ℝ := 3

def overflow_due_to_small_ball : ℝ := 0.1

theorem final_concentration_after_procedure :
  (initial_salt_concentration * (overflow_due_to_small_ball)) * volume_ratio_small / (volume_ratio_large + volume_ratio_medium + volume_ratio_small) =
  final_salt_concentration :=
sorry

end final_concentration_after_procedure_l226_226438


namespace slope_angle_comparison_l226_226151

-- Define the lines and their equations
def line1 (x y : ℝ) := x - 2 * y + 6 = 0
def line2 (x y : ℝ) := x - 3 * y + 6 = 0

-- Define the slope angles of the lines
def slope_angle (m : ℝ) := Real.atan m

-- Define the angles
def alpha := slope_angle (1 / 2)
def beta := slope_angle (1 / 3)

-- State the theorem
theorem slope_angle_comparison : alpha > beta :=
by { sorry }

end slope_angle_comparison_l226_226151


namespace integer_solutions_count_l226_226169

theorem integer_solutions_count :
  let eq : Int -> Int -> Int := fun x y => 6 * y ^ 2 + 3 * x * y + x + 2 * y - 72
  ∃ (sols : List (Int × Int)), 
    (∀ x y, eq x y = 0 → (x, y) ∈ sols) ∧
    (∀ p ∈ sols, ∃ x y, p = (x, y) ∧ eq x y = 0) ∧
    sols.length = 4 :=
by
  sorry

end integer_solutions_count_l226_226169


namespace prime_factors_correct_sum_of_largest_and_smallest_prime_factors_l226_226779

/-- The number we are considering is 1365. -/
def number := 1365

/-- Hypothesis: The prime factors of 1365 are exactly 3, 5, 7, and 13. -/
def prime_factors_of_1365 : List ℕ := [3, 5, 7, 13]

/-- Hypothesis: The prime factors are prime numbers. -/
theorem prime_factors_correct (n : ℕ) (hn : n ∈ prime_factors_of_1365) : Nat.Prime n :=
begin
  rw prime_factors_of_1365 at hn,
  fin_cases hn,
  { exact Nat.prime_of_nat 3 },
  { exact Nat.prime_of_nat 5 },
  { exact Nat.prime_of_nat 7 },
  { exact Nat.prime_of_nat 13 },
end

/-- We state the final proof we need to show that the sum is 16. -/
theorem sum_of_largest_and_smallest_prime_factors : 
  let min_prime := List.minimum prime_factors_of_1365
  let max_prime := List.maximum prime_factors_of_1365
  min_prime + max_prime = 16 :=
begin
  sorry
end

end prime_factors_correct_sum_of_largest_and_smallest_prime_factors_l226_226779


namespace rectangle_area_l226_226465

theorem rectangle_area (D : ℕ) (C A B : ℕ) (small_sq_area : ℕ) (small_sq_side : ℕ)
  (D_sq_side : ℕ) (D_sq_area : ℕ) :
  small_sq_area = 4 →
  small_sq_side = Int.sqrt small_sq_area →
  D_sq_side = 2 →
  D = D_sq_side^2 →
  C = D_sq_side + 2 →
  A = C + 2 →
  B = A + 2 →
  let length := 2 * D_sq_side + C in
  let width := C + A in
  (small_sq_area = 4 ∧ small_sq_side = 2 ∧ D_sq_side = 2 ∧ D = 4 ∧
    C = 4 + 2 ∧ A = 6 + 2 ∧ B = 8 + 2) →
  let area := length * width in
  area = 572 :=
begin
  sorry
end

end rectangle_area_l226_226465


namespace least_positive_integer_not_representable_as_fraction_l226_226880

theorem least_positive_integer_not_representable_as_fraction : 
  ¬ ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ (2^a - 2^b) / (2^c - 2^d) = 11 :=
sorry

end least_positive_integer_not_representable_as_fraction_l226_226880


namespace number_of_positive_integers_x_l226_226882

theorem number_of_positive_integers_x : 
  ∀ x : ℕ, 1 < x ∧ x < 27 → 
  (|3 + log x (1 / 3)| < 8 / 3) → 
  finset.card {i : ℕ | 2 ≤ i ∧ i ≤ 26} = 25 :=
by
  sorry

end number_of_positive_integers_x_l226_226882


namespace circle_bisection_relation_l226_226584

theorem circle_bisection_relation (a b : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = b^2 + 1 → (x + 1)^2 + (y + 1)^2 = 4) ↔ 
  a^2 + 2 * a + 2 * b + 5 = 0 :=
by sorry

end circle_bisection_relation_l226_226584


namespace problem_I_problem_II_problem_III_l226_226527

noncomputable def a : ℕ+ → ℚ
| ⟨1, _⟩ := 1 / 2
| ⟨2, _⟩ := 3 / 4
| ⟨3, _⟩ := 7 / 8
| ⟨n+1, h⟩ := (n-a ⟨n, h⟩ + 1) / 2

def b (n : ℕ+) : ℚ :=
  (n - 2) * (a n - 1)

theorem problem_I : a ⟨1, _⟩ = 1 / 2 ∧ a ⟨2, _⟩ = 3 / 4 ∧ a ⟨3, _⟩ = 7 / 8 :=
by {
  split,
  { refl },
  split,
  { refl },
  { refl }
}

theorem problem_II : ∃ r : ℚ, ∃ t : ℚ, r ≠ 0 ∧ t ≠ 0 ∧ ∀ n : ℕ+, a (n+1) - 1 = r * (a n - 1) := sorry

theorem problem_III : ∀ t : ℝ, (∀ n : ℕ+, b n + 1/4 * t ≤ t^2) ↔ t ∈ Set.Icc (-∞) (-1/4) ∪ Set.Icc (1/2) ∞ := sorry

end problem_I_problem_II_problem_III_l226_226527


namespace smaller_octagon_area_fraction_l226_226733

theorem smaller_octagon_area_fraction (A B C D E F G H : Point) (O : Point) :
  is_regular_octagon A B C D E F G H →
  is_center O A B C D E F G H →
  let A' := midpoint A B,
      B' := midpoint B C,
      C' := midpoint C D,
      D' := midpoint D E,
      E' := midpoint E F,
      F' := midpoint F G,
      G' := midpoint G H,
      H' := midpoint H A in
  is_octa_center O A' B' C' D' E' F' G' H' →
  (area_of_octagon A B C D E F G H) * (1 / 4) = area_of_octagon A' B' C' D' E' F' G' H' :=
by
  -- Sorry, proof is omitted.
  sorry

end smaller_octagon_area_fraction_l226_226733


namespace percentage_decrease_increase_l226_226668

theorem percentage_decrease_increase (S : ℝ) (x : ℝ) (h₀ : S > 0) (h₁ : S * (1 - x / 100) * (1 + x / 100) = 0.75 * S) : 
  x = 50 := 
begin
  sorry
end

end percentage_decrease_increase_l226_226668


namespace smallest_integer_cube_root_form_l226_226637

theorem smallest_integer_cube_root_form (m : ℕ) (h : ∃ (n : ℕ) (r : ℝ), m = (n + r)^3 ∧ 0 < r ∧ r < 1 / 500) : 
  let n := 13 in n = 13 :=
by
  sorry

end smallest_integer_cube_root_form_l226_226637


namespace f_f_neg2_eq_2_range_of_x_f_x_ge_2_l226_226140

def f (x : ℝ) : ℝ :=
  if 0 ≤ x then 2^x else log 2 (-x)

theorem f_f_neg2_eq_2 : f (f (-2)) = 2 :=
by
  sorry

theorem range_of_x_f_x_ge_2 (x : ℝ) : f(x) ≥ 2 ↔ x ≥ 1 ∨ x ≤ -4 :=
by
  sorry

end f_f_neg2_eq_2_range_of_x_f_x_ge_2_l226_226140


namespace mass_percentage_C_in_CaCO3_l226_226378

/-- Define the molecular formula for calcium carbonate (CaCO3) -/
def molar_mass_Ca : ℝ := 40.08
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O

/-- The mass percentage of carbon (C) in calcium carbonate (CaCO3) -/
def mass_percentage_C : ℝ := (molar_mass_C / molar_mass_CaCO3) * 100

/-- The mass percentage of carbon (C) in CaCO3 is 12.00% given the conditions -/
theorem mass_percentage_C_in_CaCO3 : mass_percentage_C = 12.00 := by
  sorry

end mass_percentage_C_in_CaCO3_l226_226378


namespace largest_common_term_l226_226464

-- Definitions for the first arithmetic sequence
def arithmetic_seq1 (n : ℕ) : ℕ := 2 + 5 * n

-- Definitions for the second arithmetic sequence
def arithmetic_seq2 (m : ℕ) : ℕ := 5 + 8 * m

-- Main statement of the problem
theorem largest_common_term (n m k : ℕ) (a : ℕ) :
  (a = arithmetic_seq1 n) ∧ (a = arithmetic_seq2 m) ∧ (1 ≤ a) ∧ (a ≤ 150) →
  a = 117 :=
by {
  sorry
}

end largest_common_term_l226_226464


namespace lcm_of_ratio_hcf_l226_226741

theorem lcm_of_ratio_hcf {a b : ℕ} (ratioCond : a = 14 * 28) (ratioCond2 : b = 21 * 28) (hcfCond : Nat.gcd a b = 28) : Nat.lcm a b = 1176 := by
  sorry

end lcm_of_ratio_hcf_l226_226741


namespace geometric_sum_first_eight_terms_l226_226841

theorem geometric_sum_first_eight_terms 
  (a : ℚ) (r : ℚ) (n : ℕ)
  (h_a : a = 1 / 4) (h_r : r = 2) (h_n : n = 8) :
  let S_n := a * (1 - r^n) / (1 - r) in
  S_n = 255 / 4 :=
by 
  sorry

end geometric_sum_first_eight_terms_l226_226841


namespace scientific_notation_of_9280000000_l226_226852

theorem scientific_notation_of_9280000000 :
  9280000000 = 9.28 * 10^9 :=
by
  sorry

end scientific_notation_of_9280000000_l226_226852


namespace area_of_DEF_l226_226604

variables {P : Type*} [inner_product_space ℝ P]
variables (A B C D E F: P)
variables (AB AC BC BE EF : ℝ)
variables (regular_tetrahedron : RegularTetrahedron A B C D)
variables (on_edges : OnEdges E A B F A C)
variables (be_dist: BE = 3)
variables (ef_dist: EF = 4)
variables (ef_parallel_faces: ParallelToFace EF (Face B C D))

noncomputable def area_of_triangle_DEF : ℝ :=
  sqrt 4 + sqrt 36

theorem area_of_DEF : area_of_triangle_DEF (A B C D E F) (AB AC BC BE EF)
  (regular_tetrahedron) (on_edges) be_dist ef_dist ef_parallel_faces
  = 10 :=
  sorry

end area_of_DEF_l226_226604


namespace n_pow_n_gt_prod_odd_l226_226663

theorem n_pow_n_gt_prod_odd (n : ℕ) (hn : 0 < n) : 
  n ^ n > ∏ i in Finset.range n, (2 * i + 1) :=
by 
  sorry

end n_pow_n_gt_prod_odd_l226_226663


namespace min_value_x_plus_2y_l226_226937

theorem min_value_x_plus_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = x * y) :
  x + 2 * y ≥ 8 :=
sorry

end min_value_x_plus_2y_l226_226937


namespace center_of_circle_in_polar_coordinates_l226_226737

-- Given condition: the polar coordinate equation
def polar_circle_eq (ρ θ : ℝ) : Prop :=
  ρ = 2 * (Real.cos θ + Real.sin θ)

-- Conversion functions
def to_rectangular_x (ρ θ : ℝ) : ℝ := ρ * Real.cos θ
def to_rectangular_y (ρ θ : ℝ) : ℝ := ρ * Real.sin θ

-- The goal is to prove that the center of the circle in polar coordinates is (sqrt(2), π/4)
theorem center_of_circle_in_polar_coordinates :
  ∃ ρ θ : ℝ, (polar_circle_eq ρ θ) ∧ (ρ = Real.sqrt 2) ∧ (θ = Real.pi / 4) :=
by 
  sorry

end center_of_circle_in_polar_coordinates_l226_226737


namespace equation_of_line_m_equation_of_line_n_l226_226121

-- Definitions for Problem 1
def Point : Type := ℝ × ℝ
def P : Point := (2, -1)

def line1 (m : Point → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ m = (λ (p : Point), p.1 + p.2 = a)

theorem equation_of_line_m (m : Point → Prop) :
  (m P) ∧ (∃ a : ℝ, m = (λ (p : Point), p.1 + p.2 = a) ∧ a ≠ 0) →
  (m = (λ (p : Point), p.2 = - (1 / 2) * p.1) ∨ m = (λ (p : Point), p.1 + p.2 - 1 = 0)) :=
sorry

-- Definitions for Problem 2
def distance_from_origin (n : Point → Prop) (d : ℝ) : Prop :=
  ∀ x y : ℝ, n (x, y) → d = abs ((- y - 0 ) / sqrt (x * x + y * y))

theorem equation_of_line_n (n : Point → Prop) :
  (n P) ∧ (distance_from_origin n 2) →
  (n = (λ (p : Point), p.1 = 2) ∨ n = (λ (p : Point), 3 * p.1 - 4 * p.2 - 10 = 0 )) :=
sorry

end equation_of_line_m_equation_of_line_n_l226_226121


namespace max_log_sum_l226_226264

noncomputable def log (x : ℝ) := real.log x / real.log 10

theorem max_log_sum (x y : ℝ) (h1 : x + 4 * y = 40) (h2 : 0 < x) (h3 : 0 < y) : log x + log y ≤ 2 :=
sorry

end max_log_sum_l226_226264


namespace bottles_remaining_l226_226793

theorem bottles_remaining (small_bottles_initial : ℕ) (big_bottles_initial : ℕ)
  (percent_sold_small : ℕ) (percent_sold_big : ℕ) :
  small_bottles_initial = 6000 →
  big_bottles_initial = 15000 →
  percent_sold_small = 11 →
  percent_sold_big = 12 →
  let small_sold := small_bottles_initial * percent_sold_small / 100 in
  let big_sold := big_bottles_initial * percent_sold_big / 100 in
  let remaining_small := small_bottles_initial - small_sold in
  let remaining_big := big_bottles_initial - big_sold in
  remaining_small + remaining_big = 18540 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end bottles_remaining_l226_226793


namespace b_plus_one_power_of_two_l226_226247

theorem b_plus_one_power_of_two 
  (b m n : ℕ) (hb : b ≠ 1) (hmn : m ≠ n) (hbmn : ∀ p : ℕ, prime p → (p ∣ b^m - 1 ↔ p ∣ b^n - 1)) :
  ∃ k : ℕ, b + 1 = 2^k :=
by
  sorry

end b_plus_one_power_of_two_l226_226247


namespace slope_of_given_line_l226_226386

theorem slope_of_given_line : ∀ (x y : ℝ), (4 / x + 5 / y = 0) → (y = (-5 / 4) * x) := 
by 
  intros x y h
  sorry

end slope_of_given_line_l226_226386


namespace water_level_rise_l226_226827

theorem water_level_rise (A : Real) (h : Real) (delta1 : Real) (delta2 : Real) :
  A = 6 → h = 0.75 → delta1 = 0.15 → delta2 = 0.225 →
  (h + delta1 + delta2 = 1.125) :=
by
  intros h h_eq delta1 delta1_eq delta2 delta2_eq
  have h_height := h + delta1 + delta2
  sorry

end water_level_rise_l226_226827


namespace remainder_two_when_divided_by_3_l226_226280

-- Define the main theorem stating that for any positive integer n,
-- n^3 + 3/2 * n^2 + 1/2 * n - 1 leaves a remainder of 2 when divided by 3.

theorem remainder_two_when_divided_by_3 (n : ℕ) (h : n > 0) : 
  (n^3 + (3 / 2) * n^2 + (1 / 2) * n - 1) % 3 = 2 := 
sorry

end remainder_two_when_divided_by_3_l226_226280


namespace solution_set_of_inequality_l226_226321

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l226_226321


namespace part1_part2_min_norm_part2_projection_l226_226963

variables {t α : ℝ}
def a := (1 : ℝ, 2 : ℝ)
def b := (Real.cos α, Real.sin α)
def c := (a.1 - t * b.1, a.2 - t * b.2)

theorem part1 (ht : t = 1) (hc_parallel : c.1 * b.2 - c.2 * b.1 = 0) :
  2 * Real.cos α^2 - Real.sin α * Real.cos α = -2 / 5 := sorry

theorem part2_min_norm (hα : α = Real.pi / 4) :
  ∃ t, ∥c∥ = Real.sqrt 2 / 2 := sorry

theorem part2_projection (hα : α = Real.pi / 4) (ht : t = 3 * Real.sqrt 2 / 2) :
  a.1 * (c .1) + a.2 * (c.2) / (Real.sqrt ((c.1)^2 + (c.2)^2) = Real.sqrt 2 / 2 := sorry

end part1_part2_min_norm_part2_projection_l226_226963


namespace diameter_increase_l226_226680

theorem diameter_increase (π : ℝ) (D : ℝ) (A A' D' : ℝ)
  (hA : A = (π / 4) * D^2)
  (hA' : A' = 4 * A)
  (hA'_def : A' = (π / 4) * D'^2) :
  D' = 2 * D :=
by
  sorry

end diameter_increase_l226_226680


namespace greatest_divisible_by_11_l226_226007

theorem greatest_divisible_by_11 :
  ∃ (A B C : ℕ), A ≠ C ∧ A ≠ B ∧ B ≠ C ∧ 
  (∀ n, n = 10000 * A + 1000 * B + 100 * C + 10 * B + A → n = 96569) ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 11 = 0 :=
sorry

end greatest_divisible_by_11_l226_226007


namespace tangent_line_eq_max_k_l226_226954

section
  open Real

  noncomputable def f (x : ℝ) (hx : 0 < x) : ℝ := exp x / (exp x - 1)

  noncomputable def g (k : ℕ) (hk : k > 0) (x : ℝ) (hx : 0 < x) : ℝ := k / (x + 1)

  theorem tangent_line_eq : ∀ (x : ℝ) (hx : x = ln 2) (h : 0 < x),
    y = -2 * x + 2 * ln 2 + 2 ->  -- replace y with the actual function, usually we would consider defining the tangent line function
    let fx := f x h in
    let dfx := Derivative (f x) in 
    y = dfx * (x - ln 2) + fx := sorry

  theorem max_k (k : ℕ) (hk : k > 0) : (∀ x : ℝ, 0 < x → f x (by linarith) > g k hk x (by linarith)) → 
    k ≤ 3 := sorry

end

end tangent_line_eq_max_k_l226_226954


namespace integer_solutions_count_l226_226173

theorem integer_solutions_count :
  ∃ (s : Finset (ℤ × ℤ)), (∀ (x y : ℤ), (6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0) ↔ ((x, y) ∈ s)) ∧ s.card = 4 :=
begin
  sorry
end

end integer_solutions_count_l226_226173


namespace monotonicity_of_f_range_of_a_l226_226550

-- Definition of functions f and g
def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + a * x
def g (x : ℝ) : ℝ := - (1/2) * x^(3/2)

-- The first part of the proof: monotonicity of f
theorem monotonicity_of_f (a : ℝ) :
  (∀ x, x ∈ ℝ → f' (f x) a ≤ 0) ∨ (a > 0 ∧ (∀ x, x ∈ ℝ → (f' (f x) a > 0 ↔ x < -√(a/3) ∨ x > √(a/3)) ∧ (f' (f x) a < 0 ↔ -√(a/3) < x ∧ x < √(a/3)))) :=
sorry

-- The second part of the proof: range of a
theorem range_of_a (a : ℝ) :
  (∀ x, 0 < x ∧ x ≤ 1 → f x a < g x) → a < -3/16 :=
sorry

end monotonicity_of_f_range_of_a_l226_226550


namespace area_ratio_sum_400_l226_226241

open Real

variables {P Q R S T : Type*} 

structure Pentangle := 
  (PQ ST QR PS PT QS : ℝ)
  (angle_PQR : ℝ)
  (parallel_PQ_ST : (PQ ∥ ST))
  (parallel_QR_PS : (QR ∥ PS))
  (parallel_PT_QS : (PT ∥ QS))

theorem area_ratio_sum_400 (pent : Pentangle)
  (hPQ : pent.PQ = 4)
  (hQR : pent.QR = 6)
  (hST : pent.ST = 18)
  (h_angle_PQR : pent.angle_PQR = 150) :
  ∃ (x y : ℕ), (gcd x y = 1) ∧ (ratio_area_trian_PQR_RST = x/y) ∧ (x + y = 400) :=
sorry

end area_ratio_sum_400_l226_226241


namespace geometric_sequence_seventh_term_l226_226886

theorem geometric_sequence_seventh_term (a₁ : ℤ) (a₂ : ℚ) (r : ℚ) (k : ℕ) (a₇ : ℚ)
  (h₁ : a₁ = 3) 
  (h₂ : a₂ = -1 / 2)
  (h₃ : r = a₂ / a₁)
  (h₄ : k = 7)
  (h₅ : a₇ = a₁ * r^(k-1)) : 
  a₇ = 1 / 15552 := 
by
  sorry

end geometric_sequence_seventh_term_l226_226886


namespace birch_tree_taller_than_pine_tree_l226_226836

theorem birch_tree_taller_than_pine_tree :
  let pine_tree_height := (49 : ℚ) / 4
  let birch_tree_height := (37 : ℚ) / 2
  birch_tree_height - pine_tree_height = 25 / 4 :=
by
  sorry

end birch_tree_taller_than_pine_tree_l226_226836


namespace Q_is_perfect_square_trinomial_l226_226409

def is_perfect_square_trinomial (p : ℤ → ℤ) :=
∃ (b : ℤ), ∀ a : ℤ, p a = (a + b) * (a + b)

def P (a b : ℤ) : ℤ := a^2 + 2 * a * b - b^2
def Q (a : ℤ) : ℤ := a^2 + 2 * a + 1
def R (a b : ℤ) : ℤ := a^2 + a * b + b^2
def S (a : ℤ) : ℤ := a^2 + 2 * a - 1

theorem Q_is_perfect_square_trinomial : is_perfect_square_trinomial Q :=
sorry -- Proof goes here

end Q_is_perfect_square_trinomial_l226_226409


namespace greatest_value_of_b_l226_226879

noncomputable def solution : ℝ :=
  (3 + Real.sqrt 21) / 2

theorem greatest_value_of_b :
  ∀ b : ℝ, b^2 - 4 * b + 3 < -b + 6 → b ≤ solution :=
by
  intro b
  intro h
  sorry

end greatest_value_of_b_l226_226879


namespace number_of_valid_pairing_ways_l226_226343

-- Define a natural number as a condition.
def is_natural (n : ℕ) : Prop := 0 < n

-- Define that 60 cards can be paired with the same modulus difference.
def pair_cards_same_modulus_difference (d : ℕ) (k : ℕ) : Prop :=
  60 = 2 * d * k

-- Define what it means for d to be a divisor of 30.
def is_divisor_of_30 (d : ℕ) : Prop :=
  ∃ k, 30 = d * k

theorem number_of_valid_pairing_ways :
  (finset.univ.filter is_divisor_of_30).card = 8 :=
begin
  sorry
end

end number_of_valid_pairing_ways_l226_226343


namespace sum_of_solutions_eq_l226_226773

theorem sum_of_solutions_eq :
  let A := 100
  let B := 3
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ = abs (B*x₁ - abs (A - B*x₁)) ∧ 
    x₂ = abs (B*x₂ - abs (A - B*x₂)) ∧ 
    x₃ = abs (B*x₃ - abs (A - B*x₃))) ∧ 
    (x₁ + x₂ + x₃ = (1900 : ℝ) / 7)) :=
by
  sorry

end sum_of_solutions_eq_l226_226773


namespace termites_ate_black_squares_l226_226676

def chessboard_black_squares_eaten : Nat :=
  12

theorem termites_ate_black_squares :
  let rows := 8;
  let cols := 8;
  let total_squares := rows * cols / 2; -- This simplistically assumes half the squares are black.
  (total_squares = 32) → 
  chessboard_black_squares_eaten = 12 :=
by
  intros h
  sorry

end termites_ate_black_squares_l226_226676


namespace smaller_octagon_area_half_l226_226707

theorem smaller_octagon_area_half
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (P Q R S T U V W : Point)
  (H1 : midpoint P A B)
  (H2 : midpoint Q B C)
  (H3 : midpoint R C D)
  (H4 : midpoint S D E)
  (H5 : midpoint T E F)
  (H6 : midpoint U F G)
  (H7 : midpoint V G H)
  (H8 : midpoint W H A):
  area (octagon P Q R S T U V W) = (1 / 2) * area (octagon ABCDEFGH) :=
sorry

end smaller_octagon_area_half_l226_226707


namespace sqrt_diff_inequality_l226_226666

theorem sqrt_diff_inequality : sqrt 2 - sqrt 6 < sqrt 3 - sqrt 7 := sorry

end sqrt_diff_inequality_l226_226666


namespace number_of_valid_pairing_ways_l226_226342

-- Define a natural number as a condition.
def is_natural (n : ℕ) : Prop := 0 < n

-- Define that 60 cards can be paired with the same modulus difference.
def pair_cards_same_modulus_difference (d : ℕ) (k : ℕ) : Prop :=
  60 = 2 * d * k

-- Define what it means for d to be a divisor of 30.
def is_divisor_of_30 (d : ℕ) : Prop :=
  ∃ k, 30 = d * k

theorem number_of_valid_pairing_ways :
  (finset.univ.filter is_divisor_of_30).card = 8 :=
begin
  sorry
end

end number_of_valid_pairing_ways_l226_226342


namespace sum_of_digits_of_unique_maximizer_l226_226635

-- Define the number of divisors function d(n)
def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).card

-- Define the function f(n) = d(n) / n^(1/4)
noncomputable def f (n : ℕ) : ℚ :=
  num_divisors n / (n : ℚ)^(1/4 : ℚ)

-- Define the statement to find the unique N such that f(N) > f(n) for all n ≠ N
def unique_maximizer (N : ℕ) : Prop :=
  ∃! N, ∀ n, n ≠ N → f N > f n

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Prove that the sum of digits of the unique maximizer is 9
theorem sum_of_digits_of_unique_maximizer :
  unique_maximizer 432 ∧ sum_of_digits 432 = 9 :=
by
  sorry

end sum_of_digits_of_unique_maximizer_l226_226635


namespace correct_statements_l226_226317

def Frequency := Type

def Probability := Type

axiom frequency_not_equal_probability : Frequency ≠ Probability
axiom frequency_not_independent_of_experiments : ∀ n : ℕ, Frequency -- depends on the number of experiments
axiom frequency_randomness : Frequency -> Prop
axiom law_of_large_numbers : ∀ (n : ℕ), Frequency → Probability -- as n tends to infinity, frequency approximates probability

theorem correct_statements (f : Frequency) (p : Probability) :
  (frequency_randomness f ∧ law_of_large_numbers nat.succ f p) :=
  sorry

end correct_statements_l226_226317


namespace angles_terminal_yaxis_l226_226319

theorem angles_terminal_yaxis :
  {θ : ℝ | ∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi / 2 ∨ θ = 2 * k * Real.pi + 3 * Real.pi / 2} =
  {θ : ℝ | ∃ n : ℤ, θ = n * Real.pi + Real.pi / 2} :=
by sorry

end angles_terminal_yaxis_l226_226319


namespace sum_inequality_l226_226261
-- Import necessary library

-- Define the theorem with the given conditions and the statement to prove
theorem sum_inequality {n : ℕ} (x : ℕ → ℝ) (hx : ∀ i, (i < n) → 0 ≤ x i ∧ x i ≤ 1) :
  (∑ i in finset.range n, x i + 1)^2 ≥ 4 * (∑ i in finset.range n, (x i)^2) :=
by
  sorry

end sum_inequality_l226_226261


namespace rectangle_area_l226_226697

theorem rectangle_area (radius side : ℝ) (breadth : ℝ) (h1 : side ^ 2 = 3600) (h2 : radius = side) (h3 : breadth = 10) :
  let length := (2 / 5) * radius in
  let area := length * breadth in
  area = 240 :=
by
  sorry

end rectangle_area_l226_226697


namespace x_plus_y_l226_226413

-- Define the sum of integers from 20 to 40
def sum_from_20_to_40 : ℕ := (finset.range (40 - 20 + 1)).sum (λ i, i + 20)

-- Define the count of even integers from 20 to 40
def count_of_even_integers_from_20_to_40 : ℕ := (finset.range (40 - 20 + 1)).count (λ i, ((i + 20) % 2 = 0))

-- Define the value of x
def x : ℕ := sum_from_20_to_40

-- Define the value of y
def y : ℕ := count_of_even_integers_from_20_to_40

theorem x_plus_y : x + y = 641 := 
by 
    -- Our definition of x and y must result in the correct value
    sorry

end x_plus_y_l226_226413


namespace find_salary_C_l226_226743
-- Import the entire Mathlib library

-- Additional options for noncomputable definitions
noncomputable def salary_C_is_11000 :=
  let A := 8000
  let B := 5000
  let D := 7000
  let E := 9000
  let avg_salary := 8000
  let x := (avg_salary * 5) - (A + B + D + E) in
  x = 11000

-- Statement in Lean 4 for the problem
theorem find_salary_C :
  salary_C_is_11000 = true := 
  sorry

end find_salary_C_l226_226743


namespace part_a_part_b_part_c_l226_226106

noncomputable def f (x : ℝ) : ℝ := sorry -- placeholder function definition

axiom f_condition1 : ∀ x y : ℝ, f(x - y) = f(x) - f(y) + 1
axiom f_condition2 : ∀ x : ℝ, x > 0 → f(x) > 1

-- Finding the value of f(0)
theorem part_a : f(0) = 1 :=
sorry

-- Prove that f(x) is increasing on ℝ
theorem part_b : ∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2) :=
sorry

-- Find the range of real numbers for a given the inequality
theorem part_c (a : ℝ) (h_a : a ≤ -3) 
  (h_ineq : ∀ x : ℝ, -1 ≤ x → f(a * x - 2) + f(x - x^2) < 2) : 
  -4 < a ∧ a ≤ -3 :=
sorry

end part_a_part_b_part_c_l226_226106


namespace students_no_problems_l226_226602

open Finset

theorem students_no_problems 
  (n : ℕ) (A B C : Finset ℕ)
  (h_total : n = 30)
  (hA : card A = 20)
  (hB : card B = 16)
  (hC : card C = 10)
  (hAB : card (A ∩ B) = 11)
  (hAC : card (A ∩ C) = 7)
  (hBC : card (B ∩ C) = 5)
  (hABC : card (A ∩ B ∩ C) = 4) :
  card (range n \ (A ∪ B ∪ C)) = 3 := 
by {
  have total_sol : card (A ∪ B ∪ C) =
    card A + card B + card C - card (A ∩ B) - card (A ∩ C) - card (B ∩ C) + card (A ∩ B ∩ C), from sorry,
  have h_total_sol : card (A ∪ B ∪ C) = 27, from sorry,
  show card (range n \ (A ∪ B ∪ C)) = n - 27, from sorry,
  show card (range n \ (A ∪ B ∪ C)) = 3, from sorry
}

end students_no_problems_l226_226602


namespace slope_of_given_line_l226_226385

theorem slope_of_given_line : ∀ (x y : ℝ), (4 / x + 5 / y = 0) → (y = (-5 / 4) * x) := 
by 
  intros x y h
  sorry

end slope_of_given_line_l226_226385


namespace probability_of_selecting_defective_l226_226907

theorem probability_of_selecting_defective :
  let total_products := 100
  let defective_products := 10
  let non_defective_products := total_products - defective_products
  let total_selections := (Finset.card (Finset.powersetLen 5 (Finset.range total_products))).toReal
  let defective_selections := (Finset.card (Finset.powersetLen 2 (Finset.range defective_products))).toReal
  let non_defective_selections := (Finset.card (Finset.powersetLen 3 (Finset.range non_defective_products))).toReal
  let favorable_outcomes := defective_selections * non_defective_selections
  let probability := favorable_outcomes / total_selections
  probability = (18 : Real) / (11 * 97 * 96) :=
by
  sorry -- Proof omitted

end probability_of_selecting_defective_l226_226907


namespace find_z2_l226_226949

theorem find_z2 (z1 z2 : ℂ) (h1 : z1 = 1 - I) (h2 : z1 * z2 = 1 + I) : z2 = I :=
sorry

end find_z2_l226_226949


namespace series_sum_zero_l226_226840

theorem series_sum_zero : 
  (∑ n in Finset.range 25, (-1)^(n - 12)) = 0 :=
by sorry

end series_sum_zero_l226_226840


namespace smallest_x_l226_226403

theorem smallest_x (x : ℕ) : (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) → x = 59 :=
by
  intro h
  sorry

end smallest_x_l226_226403


namespace driver_spending_increase_l226_226594

theorem driver_spending_increase (P Q : ℝ) (X : ℝ) (h1 : 1.20 * P = (1 + 20 / 100) * P) (h2 : 0.90 * Q = (1 - 10 / 100) * Q) :
  (1 + X / 100) * (P * Q) = 1.20 * P * 0.90 * Q → X = 8 := 
by
  sorry

end driver_spending_increase_l226_226594


namespace integer_solutions_count_l226_226172

theorem integer_solutions_count :
  ∃ (s : Finset (ℤ × ℤ)), (∀ (x y : ℤ), (6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0) ↔ ((x, y) ∈ s)) ∧ s.card = 4 :=
begin
  sorry
end

end integer_solutions_count_l226_226172


namespace sin_plus_cos_l226_226179

theorem sin_plus_cos (α : ℝ) (h : tan (α / 2) = 1 / 2) : sin α + cos α = 7 / 5 :=
  sorry

end sin_plus_cos_l226_226179


namespace find_a2_l226_226132

def arithmetic_sequence (a : ℕ → ℚ) := 
  (a 1 = 1) ∧ ∀ n, a (n + 2) - a n = 3

theorem find_a2 (a : ℕ → ℚ) (h : arithmetic_sequence a) : 
  a 2 = 5 / 2 := 
by
  -- Conditions
  have a1 : a 1 = 1 := h.1
  have h_diff : ∀ n, a (n + 2) - a n = 3 := h.2
  -- Proof steps can be written here
  sorry

end find_a2_l226_226132


namespace remainder_of_product_div_10_l226_226772

theorem remainder_of_product_div_10 : 
  (3251 * 7462 * 93419) % 10 = 8 := 
sorry

end remainder_of_product_div_10_l226_226772


namespace real_solutions_l226_226082

theorem real_solutions (x y z : ℝ) (h : 3^(x^2 - x - y) + 3^(y^2 - y - z) + 3^(z^2 - z - x) = 1) : 
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end real_solutions_l226_226082


namespace vasya_petya_notebooks_not_identical_l226_226906

theorem vasya_petya_notebooks_not_identical
  (numbers : Finset ℝ)  -- A set of 10 distinct real numbers
  (h_distinct : numbers.card = 10)  -- They are distinct.
  :
  let vasya_entries := (Finset.off_diag numbers).image (λ p : ℝ × ℝ, (p.fst - p.snd) ^ 2),
      petya_entries := (Finset.off_diag numbers).image (λ p : ℝ × ℝ, |p.fst^2 - p.snd^2|) in
  vasya_entries ≠ petya_entries :=
by
  sorry

end vasya_petya_notebooks_not_identical_l226_226906


namespace solve_quadratic_difference_l226_226067

theorem solve_quadratic_difference (x : ℝ) :
  (let equation := 2 * x^2 - 6 * x + 18 = 2 * x + 82 in
  let rearranged := 2 * x^2 - 8 * x - 64 in
  let simplified := x^2 - 4 * x - 32 in
  let roots := [8, -4] in
  let positive_diff := 8 - (-4) in
  positive_diff = 12) :=
sorry

end solve_quadratic_difference_l226_226067


namespace part_i_part_ii_part_iii_l226_226249

def lamp_state := ℕ → Bool -- Define state of lamps

def operation (n : ℕ) (L : lamp_state) (S : ℕ) : lamp_state :=
  λ j, if j = S then not (L j) else L j

def all_on (n : ℕ) (L : lamp_state) : Prop :=
  ∀ j, j < n → L j = true

theorem part_i (n : ℕ) (h₀ : n > 1) (L : lamp_state) (h₁ : all_on n L)
  : ∃ M, ∀ k, k ≥ M → all_on n (operation n L k) :=
sorry

theorem part_ii (n k : ℕ) (h₀ : n = 2^k) (L : lamp_state) (h₁ : all_on n L)
  : ∀ k, k = n^2 - 1 → all_on n (operation n L k) :=
sorry

theorem part_iii (n k : ℕ) (h₀ : n = 2^k + 1) (L : lamp_state) (h₁ : all_on n L)
  : ∀ k, k = n^2 - n + 1 → all_on n (operation n L k) :=
sorry

end part_i_part_ii_part_iii_l226_226249


namespace direction_vector_l226_226970

-- Defining what it means for two lines to be perpendicular in terms of their direction vectors
def perpendicular_lines (d1 d2 : ℕ × ℕ) : Prop :=
  let (a1, b1) := d1
  let (a2, b2) := d2
  a1 * a2 + b1 * b2 = 0

-- Given condition: Line l is perpendicular to the line with direction vector (2, 5)
def l_is_perpendicular_to_v := perpendicular_lines (5, -2) (2, 5)

-- The direction vector of line l
def direction_vector_of_l : ℕ × ℕ := (2, 5)

-- Proof statement: Line l (with direction vector (5, -2)) is perpendicular to the line with direction vector (2, 5)
-- Therefore, direction vector of line l is (2, 5)
theorem direction_vector (h : l_is_perpendicular_to_v) : direction_vector_of_l = (2, 5) :=
sorry

end direction_vector_l226_226970


namespace pencils_to_make_profit_l226_226451

theorem pencils_to_make_profit
  (total_pencils : ℕ)
  (cost_per_pencil : ℝ)
  (selling_price_per_pencil : ℝ)
  (desired_profit : ℝ)
  (pencils_to_be_sold : ℕ) :
  total_pencils = 2000 →
  cost_per_pencil = 0.08 →
  selling_price_per_pencil = 0.20 →
  desired_profit = 160 →
  pencils_to_be_sold = 1600 :=
sorry

end pencils_to_make_profit_l226_226451


namespace division_by_fraction_l226_226081

theorem division_by_fraction :
  (3 : ℚ) / (6 / 11) = 11 / 2 :=
by
  sorry

end division_by_fraction_l226_226081


namespace greatest_common_divisor_XYXY_pattern_l226_226375

theorem greatest_common_divisor_XYXY_pattern (X Y : ℕ) (hX : X ≥ 0 ∧ X ≤ 9) (hY : Y ≥ 0 ∧ Y ≤ 9) :
  ∃ k, 11 * k = 1001 * X + 10 * Y :=
by
  sorry

end greatest_common_divisor_XYXY_pattern_l226_226375


namespace fruit_basket_combinations_l226_226298

theorem fruit_basket_combinations (a b : ℕ) (ha : a = 7) (hb : b = 12) :
  (a * b) = 84 :=
by { rw [ha, hb], norm_num, }

end fruit_basket_combinations_l226_226298


namespace Callie_caught_frogs_l226_226283

-- Definitions based on the conditions provided.
def Alster_count : ℕ := 2
def Quinn_count : ℕ := 2 * Alster_count
def Bret_count : ℕ := 3 * Quinn_count
noncomputable def Callie_count : ℕ := Int.ofNat (Bret_count * 5 / 8)

-- Theorem statement: Proving Callie caught exactly 7 frogs.
theorem Callie_caught_frogs : Callie_count = 7 := by
  sorry

end Callie_caught_frogs_l226_226283


namespace distance_from_center_to_line_l226_226541

open Real

-- Definitions of the circle with given equation and the line
def circle_eqn (x y : ℝ) := x^2 - 4 * x + y^2 = 0
def line_eqn (x y : ℝ) := x - y - 1 = 0

-- Definition of the center of the circle
def center_of_circle : ℝ × ℝ := (2, 0)

-- Definition of the distance from a point to a line using the point-to-line distance formula
def point_to_line_distance (P : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  (abs (A * P.1 + B * P.2 + C)) / (sqrt (A^2 + B^2))

-- Given the circle and line equations, prove that the distance from the center to the line is √2 / 2
theorem distance_from_center_to_line :
  point_to_line_distance center_of_circle 1 (-1) (-1) = (sqrt 2) / 2 :=
by {
  -- The proof is omitted
  sorry
}

end distance_from_center_to_line_l226_226541


namespace redistribute_bonus_l226_226893

theorem redistribute_bonus :
  let bonuses := [30, 40, 50, 60, 70] in
  let total_bonus := bonuses.sum in
  let equal_share := total_bonus / 5 in
  let bonus_received := 70 in
  let amount_to_redistribute := bonus_received - equal_share in
  amount_to_redistribute = 20 :=
by
  let bonuses := [30, 40, 50, 60, 70]
  let total_bonus := bonuses.sum
  let equal_share := total_bonus / 5
  let bonus_received := 70
  let amount_to_redistribute := bonus_received - equal_share
  show amount_to_redistribute = 20 from sorry

end redistribute_bonus_l226_226893


namespace eve_apple_baskets_l226_226868

-- Defining the conditions
def apple_weight_bound (w : ℝ) : Prop := w ≤ 1 / 2

def total_apple_weight (W : ℝ) : Prop := W > 1 / 3

def ceiling (x : ℝ) : ℤ := if x - x.floor = 0 then x.floor else x.floor + 1

-- Main theorem statement
theorem eve_apple_baskets (W : ℝ) (apples : list ℝ) :
  (∀ w ∈ apples, apple_weight_bound w) → 
  total_apple_weight W → 
  W = apples.sum → 
  ∃ b : ℕ, (b:ℝ) ≤ ceiling ((3 * W - 1) / 2) ∧ 
           (∀ bs : list (list ℝ), 
            (∀ b ∈ bs, b.sum ≤ 1) → 
            bs.length = b → 
            apples.perm (bs.join)) :=
begin
  intros,
  sorry
end

end eve_apple_baskets_l226_226868


namespace no_valid_subset_of_2000_in_1_to_3000_l226_226059

def no_valid_subset_exists : Prop :=
  ¬ ∃ (A : Set ℕ), (∀ x, x ∈ A → x ∈ Finset.range 3000.succ) ∧ A.card = 2000 ∧ 
  (∀ x ∈ A, 2 * x ∉ A)

theorem no_valid_subset_of_2000_in_1_to_3000 :
  no_valid_subset_exists :=
sorry

end no_valid_subset_of_2000_in_1_to_3000_l226_226059


namespace positive_partial_sum_existence_l226_226526

variable {n : ℕ}
variable {a : Fin n → ℝ}

theorem positive_partial_sum_existence (h : (Finset.univ.sum a) > 0) :
  ∃ i : Fin n, ∀ j : Fin n, i ≤ j → (Finset.Icc i j).sum a > 0 := by
  sorry

end positive_partial_sum_existence_l226_226526


namespace problem1_problem2_problem3_problem4_l226_226849

open Rat

-- Problem 1
theorem problem1 : abs (-6) - 7 + (-3) = -4 := by
  sorry

-- Problem 2
theorem problem2 : (1/2 - 5/9 + 2/3) * (-18) = -11 := by
  sorry

-- Problem 3
theorem problem3 : 4 ÷ (-2) * -(3/2) - -4 = 7 := by
  sorry

-- Problem 4
theorem problem4 : - (5/7) * ((-3)^2 * -(4/3) - 2) = 10 := by
  sorry

end problem1_problem2_problem3_problem4_l226_226849


namespace smaller_octagon_area_fraction_l226_226730

theorem smaller_octagon_area_fraction (A B C D E F G H : Point) (O : Point) :
  is_regular_octagon A B C D E F G H →
  is_center O A B C D E F G H →
  let A' := midpoint A B,
      B' := midpoint B C,
      C' := midpoint C D,
      D' := midpoint D E,
      E' := midpoint E F,
      F' := midpoint F G,
      G' := midpoint G H,
      H' := midpoint H A in
  is_octa_center O A' B' C' D' E' F' G' H' →
  (area_of_octagon A B C D E F G H) * (1 / 4) = area_of_octagon A' B' C' D' E' F' G' H' :=
by
  -- Sorry, proof is omitted.
  sorry

end smaller_octagon_area_fraction_l226_226730


namespace trajectory_and_slope_l226_226209

-- Definition of the problem where T is a moving point whose trajectory needs to be found
def problem_trajectory (T : ℝ × ℝ) : Prop :=
  let A := (-4 : ℝ, 0 : ℝ) in
  let B := (-1 : ℝ, 0 : ℝ) in
  dist T A = 2 * dist T B

-- The proof goal statement
theorem trajectory_and_slope :
  (∀ T : ℝ × ℝ, problem_trajectory T → (T.1 ^ 2 + T.2 ^ 2 = 4)) ∧
  (let P := (real.sqrt 2, real.sqrt 2) in
   ∀ Q R : ℝ × ℝ,
     line_through P Q ∧ line_through P R ∧ complementary_inclination P Q P R →
     slope_of_line (Q, R) = 1) :=
by sorry

end trajectory_and_slope_l226_226209


namespace repeating_decimal_sum_l226_226483

theorem repeating_decimal_sum : (0.4444444... : ℚ) + (0.5656565... : ℚ) = 100 / 99 := by
    -- Defining the repeating decimal representations as rational numbers
    let x : ℚ := 4 / 9
    let y : ℚ := 56 / 99
    -- Showing their sum equals 100 / 99
    have h1 : (0.4444444... : ℚ) = x := by sorry
    have h2 : (0.5656565... : ℚ) = y := by sorry
    calc
    (0.4444444... : ℚ) + (0.5656565... : ℚ) = x + y : by rw [h1, h2]
    ... = 4 / 9 + 56 / 99 : by simp [x, y]
    ... = 44 / 99 + 56 / 99 : by norm_num
    ... = 100 / 99 : by norm_num

end repeating_decimal_sum_l226_226483


namespace tom_wins_with_smallest_n_l226_226111

def tom_and_jerry_game_proof_problem (n : ℕ) : Prop :=
  ∀ (pos : ℕ), pos ≥ 1 ∧ pos ≤ 2018 → 
  ∀ (move : ℕ), move ≥ 1 ∧ move ≤ n →
  (∃ n_min : ℕ, n_min ≤ n ∧ ∀ pos, (pos ≤ n_min ∨ pos > 2018 - n_min) → false)

theorem tom_wins_with_smallest_n : tom_and_jerry_game_proof_problem 1010 :=
sorry

end tom_wins_with_smallest_n_l226_226111


namespace total_fruits_divisible_by_31_l226_226422

theorem total_fruits_divisible_by_31
    (n : ℕ) (n2010: n = 2010) 
    (b : fin n → ℕ) (l : fin n → ℕ) (p : fin n → ℕ)
    (h1 : ∀ i, b i = 2008 * (finset.univ.sum p) + p i)
    (h2 : ∀ i, l i = (finset.univ.sum p) - p i) :
  ∃ k, (finset.univ.sum (λ i, b i + l i + p i)) = 31 * k := 
sorry

end total_fruits_divisible_by_31_l226_226422


namespace binomial_expansion_value_calculation_result_final_result_l226_226042

theorem binomial_expansion_value :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = (7 + 1)^4 := 
sorry

theorem calculation_result :
  (7 + 1)^4 = 4096 := 
sorry

theorem final_result :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by
  calc
    7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = (7 + 1)^4 := binomial_expansion_value
    ... = 4096 := calculation_result

end binomial_expansion_value_calculation_result_final_result_l226_226042


namespace area_of_rhombus_with_conditions_l226_226943

theorem area_of_rhombus_with_conditions (x : ℝ) :
  let d1 := 2 * x,
      d2 := 2 * (x + 4),
      side_length := sqrt 117 
  in (side_length = sqrt 117) →
     (abs (d2 - d1) = 8) →
     ((d1^2 + (d1 + 4)^2 = 117) →
     (2 * x * (x + 4) = 101)) :=
by
  sorry

end area_of_rhombus_with_conditions_l226_226943


namespace expected_participants_in_2005_l226_226213

open Nat

def initial_participants : ℕ := 500
def annual_increase_rate : ℚ := 1.2
def num_years : ℕ := 5
def expected_participants_2005 : ℚ := 1244

theorem expected_participants_in_2005 :
  (initial_participants : ℚ) * annual_increase_rate ^ num_years = expected_participants_2005 := by
  sorry

end expected_participants_in_2005_l226_226213


namespace sequence_proof_l226_226634

theorem sequence_proof 
    (a : ℕ → ℕ) 
    (h1 : ∀ n, (a n) % 2 = 0 → a (n + 1) = (a n + 2) / 2)
    (h2 : ∀ n, (a n) % 2 = 1 → a (n + 1) = 3 * (a n + 2) + 1) : 
    {x : ℕ // x ≤ 3012 ∧ (∀ y < 4, a 0 < a y)}.card = 753 := 
sorry

end sequence_proof_l226_226634


namespace angle_Q_l226_226290

-- Definitions for the conditions given in the problem
def is_regular_hexagon (A B C D E F : Type) : Prop :=
  ∀ (angle : ℕ), angle ∈ { ∠A B C, ∠B C D, ∠C D E, ∠D E F, ∠E F A, ∠F A B } → angle = 120

def extended_to_meet_at (A B C D Q : Type) : Prop :=
  ∃ P : Type, line A B = line P Q ∧ line C D = line P Q

-- The theorem we need to prove
theorem angle_Q (A B C D E F Q : Type) (H_hex : is_regular_hexagon A B C D E F)
  (H_meet : extended_to_meet_at A B C D Q) : ∠A Q C = 120 :=
by sorry

end angle_Q_l226_226290


namespace Bhupathi_amount_l226_226456

variable (A B : ℝ)

theorem Bhupathi_amount
  (h1 : A + B = 1210)
  (h2 : (4 / 15) * A = (2 / 5) * B) :
  B = 484 := by
  sorry

end Bhupathi_amount_l226_226456


namespace fraction_value_l226_226794

theorem fraction_value (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : (x + y) / 3 = 5 / 3 :=
by sorry

end fraction_value_l226_226794


namespace tan_sum_trig_identity_l226_226099

variable {α : ℝ}

-- Part (I)
theorem tan_sum (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
by
  sorry

-- Part (II)
theorem trig_identity (h : Real.tan α = 2) : 
  (Real.sin (2 * α) - Real.cos α ^ 2) / (1 + Real.cos (2 * α)) = 3 / 2 :=
by
  sorry

end tan_sum_trig_identity_l226_226099


namespace problem_inequality_sol1_problem_inequality_sol2_l226_226552

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x - (2 * a + 2)

theorem problem_inequality_sol1 (a x : ℝ) :
  (a > -3 / 2 ∧ (x > 2 * a + 2 ∨ x < -1)) ∨
  (a = -3 / 2 ∧ x ≠ -1) ∨
  (a < -3 / 2 ∧ (x > -1 ∨ x < 2 * a + 2)) ↔
  f x a > x :=
sorry

theorem problem_inequality_sol2 (a : ℝ) :
  (∀ x : ℝ, x > -1 → f x a + 3 ≥ 0) ↔
  a ≤ Real.sqrt 2 - 1 :=
sorry

end problem_inequality_sol1_problem_inequality_sol2_l226_226552


namespace variable_value_l226_226427

theorem variable_value 
  (x : ℝ)
  (a k some_variable : ℝ)
  (eqn1 : (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + some_variable)
  (eqn2 : a - some_variable + k = 3)
  (a_val : a = 6)
  (k_val : k = -17) :
  some_variable = -14 :=
by
  sorry

end variable_value_l226_226427


namespace price_of_brand_Y_pen_l226_226495

theorem price_of_brand_Y_pen (cost_X : ℝ) (num_X : ℕ) (total_pens : ℕ) (total_cost : ℝ) :
  cost_X = 4 ∧ num_X = 6 ∧ total_pens = 12 ∧ total_cost = 42 →
  (∃ (price_Y : ℝ), price_Y = 3) :=
by
  sorry

end price_of_brand_Y_pen_l226_226495


namespace max_value_x3y2z_l226_226642

theorem max_value_x3y2z
  (x y z : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (h_total : x + 2 * y + 3 * z = 1)
  : x^3 * y^2 * z ≤ 2048 / 11^6 := 
by
  sorry

end max_value_x3y2z_l226_226642


namespace example1_example2_l226_226085

-- Define the distance function
def distance_to_plane (x0 y0 z0 a b c d : ℝ) : ℝ :=
  (abs (a * x0 + b * y0 + c * z0 + d)) / (real.sqrt (a * a + b * b + c * c))

-- Example 1: Prove the distance from point (2, 3, -4) to plane 2x + 6y - 3z + 16 = 0 is 50/7
theorem example1 : distance_to_plane 2 3 (-4) 2 6 (-3) 16 = 50 / 7 :=
by
  sorry

-- Example 2: Prove the distance from point (2, -4, 1) to plane x - 8y + 4z = 0 is 38/9
theorem example2 : distance_to_plane 2 (-4) 1 1 (-8) 4 0 = 38 / 9 :=
by
  sorry

end example1_example2_l226_226085


namespace max_value_cos2_sin_l226_226312

noncomputable def max_cos2_sin (x : Real) : Real := 
  (Real.cos x) ^ 2 + Real.sin x

theorem max_value_cos2_sin : 
  ∃ x : Real, (-1 ≤ Real.sin x) ∧ (Real.sin x ≤ 1) ∧ 
    max_cos2_sin x = 5 / 4 :=
sorry

end max_value_cos2_sin_l226_226312


namespace linear_function_quadrant_l226_226521

theorem linear_function_quadrant (x y : ℝ): 
  (∀x y, y = x - 1 → y ≤ 0 → x ≤ 0) → ∀ x ≠ 0 → (x > 0 ∨ y < 0) → x - 1 ≠ y :=
by 
  sorry

end linear_function_quadrant_l226_226521


namespace reuleaux_triangle_area_l226_226678

def side_length (s : ℝ) := s = 1

def area_equilateral_triangle (h : ℝ) (A_triangle : ℝ) := 
  h = (Real.sqrt 3) / 2 ∧ 
  A_triangle = (Real.sqrt 3) / 4

def area_sector (r : ℝ) (A_sector : ℝ) := 
  r = 1 ∧ 
  A_sector = π / 6

def area_segment (A_segment : ℝ) := 
  A_segment = π / 6 - (Real.sqrt 3) / 4

def total_area_segments (A_total_segments : ℝ) := 
  A_total_segments = 3 * (π / 6 - (Real.sqrt 3) / 4)

def area_reuleaux (A_triangle : ℝ) (A_total_segments : ℝ) (A_reuleaux : ℝ) :=
  A_reuleaux = 3 * (π / 6 - (Real.sqrt 3) / 4) + A_triangle

theorem reuleaux_triangle_area : 
  ∀ (s h A_triangle r A_sector A_segment A_total_segments A_reuleaux : ℝ),
  side_length s →
  area_equilateral_triangle h A_triangle →
  area_sector r A_sector →
  area_segment A_segment →
  total_area_segments A_total_segments →
  area_reuleaux A_triangle A_total_segments A_reuleaux →
  A_reuleaux = π / 2 - (Real.sqrt 3) / 2 :=
by
  intros s h A_triangle r A_sector A_segment A_total_segments A_reuleaux 
  intro s_def 
  intro eq_triangle
  intro eq_sector
  intro eq_segment
  intro eq_total_segments
  intro eq_reuleaux
  rw [s_def] at eq_triangle eq_sector eq_reuleaux
  exact sorry

end reuleaux_triangle_area_l226_226678


namespace perpendicular_line_l226_226508

theorem perpendicular_line (x y : ℝ) (h : 2 * x + y - 10 = 0) : 
    (∃ k : ℝ, (x = 1 ∧ y = 2) → (k * (-2) = -1)) → 
    (∃ m b : ℝ, b = 3 ∧ m = 1/2) → 
    (x - 2 * y + 3 = 0) := 
sorry

end perpendicular_line_l226_226508


namespace tangents_meet_on_given_circle_l226_226236

open_locale classical

variables {A B C D E : Point}

-- Definitions for the given conditions
def is_rectangle (ABCD : quadrilateral) : Prop := 
  -- Define properties of being a rectangle
  sorry

def is_projection (E C BD : Point) : Prop := 
  -- Define E as the projection of C onto diagonal BD
  sorry

def circle (P Q R : Point) := 
  -- Define a circle passing through points P, Q, R
  sorry

def common_external_tangents_meet_on (circle₁ circle₂ tangents_meet_point : Point) : Prop := 
  -- Define the property of the common external tangents meeting on a specific circle
  sorry

-- The Lean 4 statement for the proof problem
theorem tangents_meet_on_given_circle :
  is_rectangle (quadrilateral.mk A B C D) →
  is_projection E C (line_segment.bd B D) →
  common_external_tangents_meet_on (circle A E B) (circle A E D) (circle A E C) :=
begin
  sorry
end

end tangents_meet_on_given_circle_l226_226236


namespace difference_intersection_l226_226255

variable {α : Type*} (A B : Set α)

-- Define the difference set
def difference_set (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

-- The main theorem to prove
theorem difference_intersection (hA : A.nonempty) (hB : B.nonempty) :
  A \ (difference_set A B) = A ∩ B :=
sorry

end difference_intersection_l226_226255


namespace total_blocks_walking_l226_226229

-- Define conditions
def vacation_cost : ℝ := 1200
def family_members : ℕ := 5
def contribution_per_member := vacation_cost / family_members

def small_dog_cost : ℝ := 2
def medium_dog_cost : ℝ := 3
def large_dog_cost : ℝ := 4

def small_dog_block_cost : ℝ := 1.25
def medium_dog_block_cost : ℝ := 1.50
def large_dog_block_cost : ℝ := 2

def total_dogs : ℕ := 25
def small_dogs : ℕ := 10
def medium_dogs : ℕ := 8
def large_dogs : ℕ := 7

def small_dog_speed : ℕ := 3  -- blocks per 10 minutes
def medium_dog_speed : ℕ := 4  -- blocks per 10 minutes
def large_dog_speed : ℕ := 2  -- blocks per 10 minutes

def total_hours_available : ℕ := 8
def break_minutes : ℕ := 30
def effective_walking_time : ℕ := (total_hours_available * 60) - break_minutes

def blocks_walked_by_large_dogs := (effective_walking_time / 10) * large_dog_speed

def total_blocks_walked :=
  (small_dogs * blocks_walked_by_large_dogs) +
  (medium_dogs * blocks_walked_by_large_dogs) +
  (large_dogs * blocks_walked_by_large_dogs)

-- Theorem statement
theorem total_blocks_walking : total_blocks_walked = 2250 :=
  by sorry

end total_blocks_walking_l226_226229


namespace max_m_exists_l226_226119

noncomputable def circle_C := {p : ℝ × ℤ | (p.1 - 3)^2 + (p.2 - 4)^2 = 1 }
def point_A (m : ℝ) := (-m, 0)
def point_B (m : ℝ) := (m, 0)

theorem max_m_exists (m : ℝ) (h : m > 0) :
  (∃ P ∈ circle_C, ∠ (point_A m) P (point_B m) = 90) → m ≤ 6 :=
sorry

end max_m_exists_l226_226119


namespace find_k_value_l226_226108

-- Definitions based on conditions
variables {k b x y : ℝ} -- k, b, x, and y are real numbers

-- Conditions given in the problem
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Proposition: Given the conditions, prove that k = 2
theorem find_k_value (h₁ : ∀ x y, y = linear_function k b x → y + 6 = linear_function k b (x + 3)) : k = 2 :=
by
  sorry

end find_k_value_l226_226108


namespace midpoint_plane_distance_l226_226125

noncomputable def midpoint_distance (A B : ℝ) (dA dB : ℝ) : ℝ :=
  (dA + dB) / 2

theorem midpoint_plane_distance (A B : ℝ) (dA dB : ℝ) (hA : dA = 1) (hB : dB = 3) :
  midpoint_distance A B dA dB = 1 ∨ midpoint_distance A B dA dB = 2 :=
by
  sorry

end midpoint_plane_distance_l226_226125


namespace find_number_l226_226428

theorem find_number (x : ℝ) (h₁ : 0.40 * x = 130 + 190) : x = 800 :=
sorry

end find_number_l226_226428


namespace investment_recovery_l226_226228

-- Define the conditions and the goal
theorem investment_recovery (c : ℕ) : 
  (15 * c - 5 * c) ≥ 8000 ↔ c ≥ 800 := 
sorry

end investment_recovery_l226_226228


namespace complement_A_in_U_l226_226567

def U := {x : ℝ | -4 < x ∧ x < 4}
def A := {x : ℝ | -3 ≤ x ∧ x < 2}

theorem complement_A_in_U :
  {x : ℝ | x ∈ U ∧ x ∉ A} = {x : ℝ | (-4 < x ∧ x < -3) ∨ (2 ≤ x ∧ x < 4)} :=
by {
  sorry
}

end complement_A_in_U_l226_226567


namespace fahrenheit_to_celsius_equiv_l226_226058

theorem fahrenheit_to_celsius_equiv :
  let F' (F : ℤ) := (9 * (5 * (F - 32) / 9).nat_abs / 5 + 32)
  (count_eq_F' 50 2000 (fun F => if F = F' F then 1 else 0)) = 1089 :=
by sorry

end fahrenheit_to_celsius_equiv_l226_226058


namespace cyrus_shots_percentage_l226_226196

theorem cyrus_shots_percentage (total_shots : ℕ) (missed_shots : ℕ) (made_shots : ℕ)
  (h_total : total_shots = 20)
  (h_missed : missed_shots = 4)
  (h_made : made_shots = total_shots - missed_shots) :
  (made_shots / total_shots : ℚ) * 100 = 80 := by
  sorry

end cyrus_shots_percentage_l226_226196


namespace find_lengths_of_DE_FA_l226_226574

-- Let ABCDEF be a hexagon with given sides
def Hexagon :=
{ ABCDEF : ℝ,
  AB BC CD DE EF FA : ℝ }

noncomputable def SideLengths (h : Hexagon) :=
h.AB = 9 ∧ h.BC = 6 ∧ h.CD = 2 ∧ h.EF = 4

-- Formal statement of the problem
theorem find_lengths_of_DE_FA (h : Hexagon) (sides : SideLengths h) :
  {DE FA : ℝ // DE = (9 + Real.sqrt 33) / 2 ∧ FA = (9 - Real.sqrt 33) / 2} :=
sorry

end find_lengths_of_DE_FA_l226_226574


namespace find_y_l226_226786

theorem find_y (x : ℤ) (y : ℤ) (h : x = 5) (h1 : 3 * x = (y - x) + 4) : y = 16 :=
by
  sorry

end find_y_l226_226786


namespace sum_of_possible_m_values_l226_226123

theorem sum_of_possible_m_values : 
  (∑ m in (Finset.filter (λ m : ℤ, 0 < 3 * m ∧ 3 * m < 27) (Finset.Icc 1 8)), m) = 36 := 
by
  sorry

end sum_of_possible_m_values_l226_226123


namespace random_variable_selection_l226_226458

theorem random_variable_selection :
  let products := 6 in
  let defective := 2 in
  let non_defective := 4 in
  (∃ (X : Type) (n : X → ℕ), random_variable X ∧
    X = (number of non-defective products selected)) :=
sorry

end random_variable_selection_l226_226458


namespace max_value_norm_expression_l226_226649

variables (p q r : ℝ^3)
variables (hp : ∥p∥ = 2) (hq : ∥q∥ = 3) (hr : ∥r∥ = 4)

theorem max_value_norm_expression : 
  ∥p - 3 • q∥^2 + ∥q - 3 • r∥^2 + ∥r - 3 • p∥^2 = 446 :=
by sorry

end max_value_norm_expression_l226_226649


namespace positive_difference_solutions_l226_226381

theorem positive_difference_solutions (r₁ r₂ : ℝ) (h_r₁ : (r₁^2 - 5 * r₁ - 22) / (r₁ + 4) = 3 * r₁ + 8) (h_r₂ : (r₂^2 - 5 * r₂ - 22) / (r₂ + 4) = 3 * r₂ + 8) (h_r₁_ne : r₁ ≠ -4) (h_r₂_ne : r₂ ≠ -4) :
  |r₁ - r₂| = 3 / 2 := 
sorry


end positive_difference_solutions_l226_226381


namespace lace_per_ruffle_l226_226478

theorem lace_per_ruffle 
  (cuff_length hem_length waist_fraction cost_per_meter total_spent ruffles : ℕ)
  (h_cuff_length : cuff_length = 50)
  (h_hem_length : hem_length = 300)
  (h_waist_fraction : waist_fraction = 1 / 3)
  (h_cost_per_meter : cost_per_meter = 6)
  (h_total_spent : total_spent = 36)
  (h_ruffles : ruffles = 5) :
  let total_cuffs := 2 * cuff_length in
  let waist_length := hem_length * waist_fraction in
  let total_fixed_lace := total_cuffs + waist_length + hem_length in
  let total_lace := (total_spent / cost_per_meter) * 100 in
  let lace_for_ruffles := total_lace - total_fixed_lace in
  lace_for_ruffles / ruffles = 20 :=
by
  sorry

end lace_per_ruffle_l226_226478


namespace number_of_factors_l226_226575

theorem number_of_factors (N : ℕ) (hN : N = 2^5 * 3^2 * 5^3 * 7^1 * 11^1) : 
  (finset.univ.filter (λ d, d ∣ N)).card = 288 := 
by {
  sorry
}

end number_of_factors_l226_226575


namespace number_of_students_in_class_l226_226980

variable (S total_students smoking_students not_hospitalized_students : ℝ)

-- Conditions
def condition1 : Prop := smoking_students = 0.40 * total_students
def condition2 : Prop := not_hospitalized_students = 0.30 * smoking_students
def condition3 : Prop := not_hospitalized_students = 36

-- Conclusion we want to prove
theorem number_of_students_in_class (h1 : condition1 S total_students smoking_students)
                                    (h2 : condition2 S total_students smoking_students not_hospitalized_students)
                                    (h3 : condition3 not_hospitalized_students) :
  total_students = 300 :=
by sorry

end number_of_students_in_class_l226_226980


namespace tangent_line_at_point_l226_226543

noncomputable def curve : ℝ → ℝ := λ x, 3 * x - Real.log x

noncomputable def tangent_slope (x : ℝ) : ℝ := (3 : ℝ) - (1 / x)

theorem tangent_line_at_point :
  let x := 1
  let y := 3
  tangent_slope x * (x - 1) = y - ((2 : ℝ) * x - 1) :=
by
  sorry

end tangent_line_at_point_l226_226543


namespace curves_intersection_l226_226767

theorem curves_intersection :
  (∃ (A B C D : ℕ), A ∈ {1, 2, 3, 4, 5, 6, 7} ∧ B ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                   C ∈ {1, 2, 3, 4, 5, 6, 7} ∧ D ∈ {1, 2, 3, 4, 5, 6, 7} ∧
                   A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
                   (∃ x : ℚ, x ≥ 0 ∧ A * x ^ 2 + B = C * x ^ 2 + D)) →
  (∃ ways : ℕ, ways = 210) :=
by sorry

end curves_intersection_l226_226767


namespace matchsticks_in_20th_stage_l226_226754

-- Define the first term and common difference
def first_term : ℕ := 4
def common_difference : ℕ := 3

-- Define the mathematical function for the n-th term of the arithmetic sequence
def num_matchsticks (n : ℕ) : ℕ :=
  first_term + (n - 1) * common_difference

-- State the theorem to prove the number of matchsticks in the 20th stage
theorem matchsticks_in_20th_stage : num_matchsticks 20 = 61 :=
by
  -- Proof skipped
  sorry

end matchsticks_in_20th_stage_l226_226754


namespace angle_XPQ_l226_226219

theorem angle_XPQ (X Y Z P O Q : Type*)
  (h_triangle : triangle X Y Z)
  (h_XZY : ∠ X Z Y = 60)
  (h_YZX : ∠ Y Z X = 80)
  (h_perpendicular : P ⊥ ZY ∧ lies_on P line YZ)
  (h_circumcircle : is_circumcenter O (triangle X Y Z))
  (h_diameter : diameter O Q X ∧ lies_on Q circle (circumcircle X Y Z)) :
  ∠ X P Q = 20 := 
sorry

end angle_XPQ_l226_226219


namespace find_integer_roots_l226_226875

open Int Polynomial

def P (x : ℤ) : ℤ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_integer_roots : {x : ℤ | P x = 0} = {-3, 1, 5} := by
  sorry

end find_integer_roots_l226_226875


namespace hilary_total_payment_l226_226468

theorem hilary_total_payment :
  let samosas_cost := 3 * 2
      pakoras_cost := 4 * 3
      mango_lassi_cost := 2
      biryanis_cost := 2 * 5.5
      garlic_naan_cost := 1.5
      total_before_discounts_service_fee := samosas_cost + pakoras_cost + mango_lassi_cost + biryanis_cost + garlic_naan_cost
      biryani_discount := 0.10 * biryanis_cost
      total_after_biryani_discount := total_before_discounts_service_fee - biryani_discount
      service_fee := 0.03 * total_before_discounts_service_fee
      total_after_service_fee := total_after_biryani_discount + service_fee
      tip := 0.20 * total_after_biryani_discount
      sales_tax := 0.08 * total_after_service_fee
      total_paid := total_after_service_fee + tip + sales_tax
  in total_paid = 41.25 := sorry

end hilary_total_payment_l226_226468


namespace intersection_A_B_l226_226242

def A := {x : ℝ | x^2 - ⌊x⌋ = 2}
def B := {x : ℝ | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, Real.sqrt 3} :=
sorry

end intersection_A_B_l226_226242


namespace find_sum_a1_to_a7_l226_226901

theorem find_sum_a1_to_a7 : 
  let f (x : ℝ) := (1 - 2 * x) ^ 7,
      a_0 := f 0,
      a_sum := f 1 in
  a_sum - a_0 = -2 :=
by
  sorry

end find_sum_a1_to_a7_l226_226901


namespace solution_set_of_inequality_l226_226894

theorem solution_set_of_inequality (a : ℝ) :
  ¬ (∀ x : ℝ, ¬ (a * (x - a) * (a * x + a) ≥ 0)) ∧
  ¬ (∀ x : ℝ, (a - x ≤ 0 ∧ x - (-1) ≤ 0 → a * (x - a) * (a * x + a) ≥ 0)) :=
by
  sorry

end solution_set_of_inequality_l226_226894


namespace area_ratio_of_smaller_octagon_l226_226721

theorem area_ratio_of_smaller_octagon
    (A B C D E F G H : ℝ × ℝ) -- Coordinates of vertices of the larger octagon
    (P Q R S T U V W : ℝ × ℝ) -- Coordinates of vertices of the smaller octagon
    (regular_octagon : ∀ (X Y Z W U V T S : ℝ × ℝ), regular_octo X Y Z W U V T S)  -- Predicate for regular octagon
    (midpoints_joined : ∀ (X Y : ℝ × ℝ), midpoint X Y) : -- Condition that midpoints form the smaller octagon
  area (smaller_octo P Q R S T U V W) = (3 : ℝ) / 4 * area (larger_octo A B C D E F G H) :=
sorry

end area_ratio_of_smaller_octagon_l226_226721


namespace rational_expression_l226_226862

theorem rational_expression (x : ℝ) : 
  (∃ q : ℚ, (q:ℝ) = x + sqrt (x^2 + 1) + 1 / (x + sqrt (x^2 + 1))) ↔ 
  ∃ r : ℚ, (r:ℝ) = sqrt (x^2 + 1) := 
sorry

end rational_expression_l226_226862


namespace train_probability_l226_226654

theorem train_probability
  (train_wait : ℝ := 15) -- train waits for 15 minutes
  (alex_arrival_start : ℝ := 0) -- Alex can start arriving at 1:00
  (alex_arrival_end : ℝ := 90) -- Alex can arrive until 2:30 (90 minutes interval)
  (train_arrival_start : ℝ := 0) -- Train can start arriving at 1:00
  (train_arrival_end : ℝ := 60) -- Train can arrive until 2:00 (60 minutes interval)
  : 
  let area_overlap := (train_arrival_end - train_arrival_start) * train_wait
  let total_area := (alex_arrival_end - alex_arrival_start) * (train_arrival_end - train_arrival_start)
  in (area_overlap / total_area) = 1 / 6 := 
by
  sorry

end train_probability_l226_226654


namespace largest_circle_radius_on_chessboard_l226_226329

theorem largest_circle_radius_on_chessboard :
  let radius := Real.sqrt 10 / 2 in
  ∀ (N : ℕ), (N > 0) → -- To handle general size of chessboard; in our case, N = 1.
  (∀ (x y : ℕ), x < N → y < N →
  let dx := Real.sqrt ((x - (x + 1)) ^ 2 + (y - (y + 1)) ^ 2) in
  dx > radius) →
  radius = Real.sqrt 10 / 2 :=
by
  sorry

end largest_circle_radius_on_chessboard_l226_226329


namespace required_percentage_to_pass_l226_226453

theorem required_percentage_to_pass
  (marks_obtained : ℝ)
  (marks_failed_by : ℝ)
  (max_marks : ℝ)
  (passing_marks := marks_obtained + marks_failed_by)
  (required_percentage : ℝ := (passing_marks / max_marks) * 100)
  (h : marks_obtained = 80)
  (h' : marks_failed_by = 40)
  (h'' : max_marks = 200) :
  required_percentage = 60 := 
by
  sorry

end required_percentage_to_pass_l226_226453


namespace vika_pairs_exactly_8_ways_l226_226349

theorem vika_pairs_exactly_8_ways :
  ∃ d : ℕ, (d ∣ 30) ∧ (Finset.card (Finset.filter (λ d, d ∣ 30) (Finset.range 31)) = 8) := 
sorry

end vika_pairs_exactly_8_ways_l226_226349


namespace integer_solutions_of_equation_l226_226162

theorem integer_solutions_of_equation:
  (number_of_int_solutions (λ x y, 6 * y^2 + 3 * x * y + x + 2 * y + 180) = 6) :=
begin
  sorry
end

end integer_solutions_of_equation_l226_226162


namespace smaller_octagon_half_area_l226_226704

-- Define what it means to be a regular octagon
def is_regular_octagon (O : Point) (ABCDEFGH : List Point) : Prop :=
  -- Definition capturing the properties of a regular octagon around center O
  sorry

-- Define the function that computes the area of an octagon
def area_of_octagon (ABCDEFGH : List Point) : Real :=
  sorry

-- Define the function to create the smaller octagon by joining midpoints
def smaller_octagon (ABCDEFGH : List Point) : List Point :=
  sorry

theorem smaller_octagon_half_area (O : Point) (ABCDEFGH : List Point) :
  is_regular_octagon O ABCDEFGH →
  area_of_octagon (smaller_octagon ABCDEFGH) = (1 / 2) * area_of_octagon ABCDEFGH :=
by
  sorry

end smaller_octagon_half_area_l226_226704


namespace washing_machines_total_pounds_l226_226830

theorem washing_machines_total_pounds (pounds_per_machine_per_day : ℕ) (number_of_machines : ℕ)
  (h1 : pounds_per_machine_per_day = 28) (h2 : number_of_machines = 8) :
  number_of_machines * pounds_per_machine_per_day = 224 :=
by
  sorry

end washing_machines_total_pounds_l226_226830


namespace john_votes_l226_226805

theorem john_votes (J : ℝ) (total_votes : ℝ) (third_candidate_votes : ℝ) (james_votes : ℝ) 
  (h1 : total_votes = 1150) 
  (h2 : third_candidate_votes = J + 150) 
  (h3 : james_votes = 0.70 * (total_votes - J - third_candidate_votes)) 
  (h4 : total_votes = J + james_votes + third_candidate_votes) : 
  J = 500 := 
by 
  rw [h1, h2, h3] at h4 
  sorry

end john_votes_l226_226805


namespace solve_parabola_l226_226559

theorem solve_parabola (a b c : ℝ) 
  (h1 : 1 = a * 1^2 + b * 1 + c)
  (h2 : 4 * a + b = 1)
  (h3 : -1 = a * 2^2 + b * 2 + c) :
  a = 3 ∧ b = -11 ∧ c = 9 :=
by {
  sorry
}

end solve_parabola_l226_226559


namespace average_trip_speed_l226_226270

theorem average_trip_speed 
  (total_distance : ℝ)
  (dist1 : ℝ) (speed1 : ℝ)
  (dist2 : ℝ) (speed2 : ℝ)
  (dist2_limit : ℝ) (speed2_limit : ℝ)
  (dist3 : ℝ) (time_factor3 : ℝ)
  (time1 : ℝ := dist1 / speed1)
  (time2_limit : ℝ := dist2_limit / speed2_limit)
  (time2_other : ℝ := (dist2 - dist2_limit) / speed2)
  (time3 : ℝ := time_factor3 * time1 + time1)
  (total_time : ℝ := time1 + time2_limit + time2_other + time3)
  (j : ℝ := total_distance / total_time) :
  total_distance = dist1 + dist2 + dist3 ∧
  time_factor3 = 1.5 ∧ 
  total_distance = 760 ∧
  dist1 = 320 ∧
  speed1 = 80 ∧
  dist2 = 240 ∧
  speed2 = 60 ∧
  dist2_limit = 100 ∧
  speed2_limit = 45 ∧
  dist3 = 200 ∧
  j ≈ 40.97 :=
by
  -- Proof steps would go here, but since only statement is needed, we skip
  sorry

end average_trip_speed_l226_226270


namespace Emma_investment_l226_226865

-- Define the necessary context and variables
variable (E : ℝ) -- Emma's investment
variable (B : ℝ := 500) -- Briana's investment which is a known constant
variable (ROI_Emma : ℝ := 0.30 * E) -- Emma's return on investment after 2 years
variable (ROI_Briana : ℝ := 0.20 * B) -- Briana's return on investment after 2 years
variable (ROI_difference : ℝ := ROI_Emma - ROI_Briana) -- The difference in their ROI

theorem Emma_investment :
  ROI_difference = 10 → E = 366.67 :=
by
  intros h
  sorry

end Emma_investment_l226_226865


namespace B_investment_amount_l226_226004

-- Definitions based on given conditions
variable (A_investment : ℕ := 300) -- A's investment in dollars
variable (B_investment : ℕ)        -- B's investment in dollars
variable (A_time : ℕ := 12)        -- Time A's investment was in the business in months
variable (B_time : ℕ := 6)         -- Time B's investment was in the business in months
variable (profit : ℕ := 100)       -- Total profit in dollars
variable (A_share : ℕ := 75)       -- A's share of the profit in dollars

-- The mathematically equivalent proof problem to prove that B invested $200
theorem B_investment_amount (h : A_share * (A_investment * A_time + B_investment * B_time) / profit = A_investment * A_time) : 
  B_investment = 200 := by
  sorry

end B_investment_amount_l226_226004


namespace floor_sum_sequence_l226_226653

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ a 2 = 6 ∧ (∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n + 2)

theorem floor_sum_sequence (a : ℕ → ℕ) (h : sequence a) :
  ⌊∑ i in range 2017, 2017 / a i⌋ = 2016 :=
  by sorry

end floor_sum_sequence_l226_226653


namespace total_rent_calculation_l226_226276

variables (x y : ℕ) -- x: number of rooms rented for $40, y: number of rooms rented for $60
variable (rent_total : ℕ)

-- Condition: Each room at the motel was rented for either $40 or $60
-- Condition: If 10 of the rooms that were rented for $60 had instead been rented for $40, the total rent would have been reduced by 50 percent

theorem total_rent_calculation 
  (h1 : 40 * (x + 10) + 60 * (y - 10) = (40 * x + 60 * y) / 2) :
  40 * x + 60 * y = 800 :=
sorry

end total_rent_calculation_l226_226276


namespace unique_two_scoop_sundaes_l226_226026

theorem unique_two_scoop_sundaes : ∀ (n : ℕ), n = 8 → (∑ i in (finset.range(n)).powerset.filter (λ s, s.card = 2), 1) = 28 :=
by intros n hn
   cases hn
   simp
   sorry

end unique_two_scoop_sundaes_l226_226026


namespace union_sets_l226_226267

-- Define the sets A and B
def A : Set ℤ := {0, 1}
def B : Set ℤ := {x : ℤ | (x + 2) * (x - 1) < 0}

-- The theorem to be proven
theorem union_sets : A ∪ B = {-1, 0, 1} :=
by
  sorry

end union_sets_l226_226267


namespace trig_tan_225_expression_l226_226739

theorem trig_tan_225_expression (a b c d : ℕ) 
  (h : a ≥ b ∧ b ≥ c ∧ c ≥ d)
  (ha : a = 8)
  (hb : b = 0)
  (hc : c = 0)
  (hd : d = 2) :
  \[ \tan 22.5^\circ = \sqrt{a} - \sqrt{b} + \sqrt{c} - d \]
  (h_sum : a + b + c + d = 10) :=
by sorry

end trig_tan_225_expression_l226_226739


namespace solution_correct_l226_226292

-- Define the conditions
def abs_inequality (x : ℝ) : Prop := abs (x - 3) + abs (x + 4) < 8
def quadratic_eq (x : ℝ) : Prop := x^2 - x - 12 = 0

-- Define the main statement to prove
theorem solution_correct : ∃ (x : ℝ), abs_inequality x ∧ quadratic_eq x ∧ x = -3 := sorry

end solution_correct_l226_226292


namespace number_multiplied_by_10_cubed_is_30_l226_226976

-- Definitions based on conditions
variables (x : ℤ) (N : ℤ)
def greatest_possible_value_for_x := (x = 3)
def inequality_condition := (N * 10^x < 31000)
def integer_condition := (N : ℤ)

-- Proof problem statement
theorem number_multiplied_by_10_cubed_is_30 :
  greatest_possible_value_for_x x → 
  inequality_condition x N → 
  integer_condition N → 
  N = 30 :=
by
  sorry

end number_multiplied_by_10_cubed_is_30_l226_226976


namespace card_pairing_modulus_l226_226357

theorem card_pairing_modulus (cards : Finset ℕ) (h : cards = (Finset.range 60).image (λ n, n + 1)) :
  ∃ n, n = 8 ∧ ∀ (pairs : Finset (ℕ × ℕ)), (∀ (p ∈ pairs), (p.1 ∈ cards ∧ p.2 ∈ cards ∧ (|p.1 - p.2| = d))) → pairs.card = 30 :=
sorry

end card_pairing_modulus_l226_226357


namespace intersection_ratios_l226_226514

-- Define the conditions
def y_sin (x : ℝ) : ℝ := Real.sin x
def y_sin_60 : ℝ := Real.sin (Real.pi / 3)

-- Define the proof problem
theorem intersection_ratios (p q : ℕ) (hpq : Nat.coprime p q) (h : p < q) :
  y_sin 60 = y_sin_60 ∧ y_sin 120 = y_sin_60 ∧ (p, q) = (1, 5) :=
by
  sorry

end intersection_ratios_l226_226514


namespace air_quality_probability_l226_226199

variable (p_good_day : ℝ) (p_good_two_days : ℝ)

theorem air_quality_probability
  (h1 : p_good_day = 0.75)
  (h2 : p_good_two_days = 0.6) :
  (p_good_two_days / p_good_day = 0.8) :=
by
  rw [h1, h2]
  norm_num

end air_quality_probability_l226_226199


namespace expression_value_l226_226038

theorem expression_value : 7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by 
  -- proof goes here 
  sorry

end expression_value_l226_226038


namespace count_real_values_z5_l226_226751

-- Define the problem conditions and the statement to prove
theorem count_real_values_z5 :
  let roots_of_unity := {z : ℂ | z^30 = 1}
  let z5_reals := {z ∈ roots_of_unity | z^5 = 1 ∨ z^5 = -1}
  #|z5_reals| = 12 :=
begin
  sorry
end

end count_real_values_z5_l226_226751


namespace john_saved_120_dollars_l226_226226

-- Defining the conditions
def num_machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def total_ball_bearings : ℕ := num_machines * ball_bearings_per_machine
def regular_price_per_bearing : ℝ := 1
def sale_price_per_bearing : ℝ := 0.75
def bulk_discount : ℝ := 0.20
def discounted_price_per_bearing : ℝ := sale_price_per_bearing - (bulk_discount * sale_price_per_bearing)

-- Calculate total costs
def total_cost_without_sale : ℝ := total_ball_bearings * regular_price_per_bearing
def total_cost_with_sale : ℝ := total_ball_bearings * discounted_price_per_bearing

-- Calculate the savings
def savings : ℝ := total_cost_without_sale - total_cost_with_sale

-- The theorem we want to prove
theorem john_saved_120_dollars : savings = 120 := by
  sorry

end john_saved_120_dollars_l226_226226


namespace distribute_balls_into_boxes_l226_226177

-- Problem Conditions
def balls : ℕ := 5
def boxes : ℕ := 3

-- Theorem: There are 147 ways to distribute 5 balls into 3 boxes such that each box gets at least one ball.
theorem distribute_balls_into_boxes (balls boxes : ℕ) (hballs : balls = 5) (hboxes : boxes = 3) :
  (number_of_ways_to_distribute_balls balls boxes) = 147 :=
sorry

end distribute_balls_into_boxes_l226_226177


namespace factorization_correct_l226_226460

theorem factorization_correct :
  (¬ (x^2 - 2 * x - 1 = x * (x - 2) - 1)) ∧
  (¬ (2 * x + 1 = x * (2 + 1 / x))) ∧
  (¬ ((x + 2) * (x - 2) = x^2 - 4)) ∧
  (x^2 - 1 = (x + 1) * (x - 1)) :=
by
  sorry

end factorization_correct_l226_226460


namespace proof_complement_intersection_l226_226268

open Set Real

def A : Set ℝ := { y | y > 0 }
def B : Set ℝ := -2 ≈ ∅{-2, -1, 1, 2} ≡∪～

theorem proof_complement_intersection :
  (compl A ∩ B) = {-2, -1} := 
begin
  sorry
end

end proof_complement_intersection_l226_226268


namespace integer_solutions_of_equation_l226_226163

theorem integer_solutions_of_equation:
  (number_of_int_solutions (λ x y, 6 * y^2 + 3 * x * y + x + 2 * y + 180) = 6) :=
begin
  sorry
end

end integer_solutions_of_equation_l226_226163


namespace sum_hex_digits_2010_l226_226657

theorem sum_hex_digits_2010 : 
  let hex_2010 := 0x7DA in 
  (7 + 13 + 10 = 30) :=
by 
  let hex_2010 := 0x7DA 
  let digit_sum := 0x7 + 0xD + 0xA
  have h : digit_sum = 30 := by sorry
  exact h

end sum_hex_digits_2010_l226_226657


namespace constant_term_of_second_eq_l226_226095

theorem constant_term_of_second_eq (x y : ℝ) 
  (h1 : 7*x + y = 19) 
  (h2 : 2*x + y = 5) : 
  ∃ k : ℝ, x + 3*y = k ∧ k = 15 := 
by
  sorry

end constant_term_of_second_eq_l226_226095


namespace circle_center_l226_226491

theorem circle_center (x y : ℝ) : 
    (∃ x y : ℝ, x^2 - 8*x + y^2 - 4*y = 16) → (x, y) = (4, 2) := by
  sorry

end circle_center_l226_226491


namespace sqrt_four_minus_one_eq_one_l226_226848

theorem sqrt_four_minus_one_eq_one : sqrt 4 - 1 = 1 :=
by
  -- The proof would go here, but we're using sorry to indicate it's omitted.
  sorry

end sqrt_four_minus_one_eq_one_l226_226848


namespace total_annual_car_maintenance_expenses_is_330_l226_226624

-- Define the conditions as constants
def annualMileage : ℕ := 12000
def milesPerOilChange : ℕ := 3000
def freeOilChangesPerYear : ℕ := 1
def costPerOilChange : ℕ := 50
def milesPerTireRotation : ℕ := 6000
def costPerTireRotation : ℕ := 40
def milesPerBrakePadReplacement : ℕ := 24000
def costPerBrakePadReplacement : ℕ := 200

-- Define the total annual car maintenance expenses calculation
def annualOilChangeExpenses (annualMileage : ℕ) (milesPerOilChange : ℕ) (freeOilChangesPerYear : ℕ) (costPerOilChange : ℕ) : ℕ :=
  let oilChangesNeeded := annualMileage / milesPerOilChange
  let paidOilChanges := oilChangesNeeded - freeOilChangesPerYear
  paidOilChanges * costPerOilChange

def annualTireRotationExpenses (annualMileage : ℕ) (milesPerTireRotation : ℕ) (costPerTireRotation : ℕ) : ℕ :=
  let tireRotationsNeeded := annualMileage / milesPerTireRotation
  tireRotationsNeeded * costPerTireRotation

def annualBrakePadReplacementExpenses (annualMileage : ℕ) (milesPerBrakePadReplacement : ℕ) (costPerBrakePadReplacement : ℕ) : ℕ :=
  let brakePadReplacementInterval := milesPerBrakePadReplacement / annualMileage
  costPerBrakePadReplacement / brakePadReplacementInterval

def totalAnnualCarMaintenanceExpenses : ℕ :=
  annualOilChangeExpenses annualMileage milesPerOilChange freeOilChangesPerYear costPerOilChange +
  annualTireRotationExpenses annualMileage milesPerTireRotation costPerTireRotation +
  annualBrakePadReplacementExpenses annualMileage milesPerBrakePadReplacement costPerBrakePadReplacement

-- Prove the total annual car maintenance expenses equals $330
theorem total_annual_car_maintenance_expenses_is_330 : totalAnnualCarMaintenanceExpenses = 330 := by
  sorry

end total_annual_car_maintenance_expenses_is_330_l226_226624


namespace percentage_left_due_to_fear_l226_226607

-- Define all the given constants from the initial problem
def initial_population : ℕ := 7145
def death_percentage : ℝ := 0.15
def reduced_population : ℕ := 4555

-- Define the derived constants based on conditions
def died_due_to_bombardment : ℕ := (death_percentage * initial_population).toNat
def remaining_after_bombardment : ℕ := initial_population - died_due_to_bombardment
def people_left_due_to_fear : ℕ := remaining_after_bombardment - reduced_population

-- Define the proof goal
theorem percentage_left_due_to_fear :
    (people_left_due_to_fear : ℝ) / (remaining_after_bombardment : ℝ) * 100 = 25 :=
by
  -- Skipping the proof, only the statement is required as per instruction
  sorry

end percentage_left_due_to_fear_l226_226607


namespace solution_set_of_inequality_l226_226324

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l226_226324


namespace trajectory_equation_of_circle_center_l226_226444

theorem trajectory_equation_of_circle_center
  (x y : ℝ)
  (h : Real.sqrt ((x - 3)^2 + (y - 2)^2) = Real.abs (y - 1)) :
  x^2 - 6*x + 2*y + 12 = 0 :=
by
  sorry

end trajectory_equation_of_circle_center_l226_226444


namespace all_positive_integers_occur_l226_226010

def seq : ℕ → ℕ
| 0 => 1
| 1 => 24
| n => Nat.find (λ m, 
                   m > 0 ∧ 
                   m ∉ (List.range (n+1)).map seq ∧ 
                   ¬ Nat.coprime m (seq n))

noncomputable def Sₙ (n : ℕ) : Set ℕ := 
  {m | ∃ k, k ≤ n ∧ seq k = m}

noncomputable def S_infty : Set ℕ := 
  {m | ∃ n, seq n = m}

theorem all_positive_integers_occur : S_infty = { m | m > 0 } :=
by
sory

end all_positive_integers_occur_l226_226010


namespace green_balls_to_remove_l226_226029

theorem green_balls_to_remove :
  ∀ (total_balls : ℕ) (initial_red_percentage : ℚ) (final_red_percentage : ℚ) (initial_total_red_balls : ℕ) (initial_total_green_balls : ℕ),
  total_balls = 150 →
  initial_red_percentage = 40 / 100 →
  final_red_percentage = 80 / 100 →
  initial_total_red_balls = (initial_red_percentage * total_balls) →
  initial_total_green_balls = (total_balls - initial_total_red_balls) →
  60 / (150 - 75) = 0.80 :=
by
  intros total_balls initial_red_percentage final_red_percentage initial_total_red_balls initial_total_green_balls
  assume h1 : total_balls = 150
  assume h2 : initial_red_percentage = 40 / 100
  assume h3 : final_red_percentage = 80 / 100
  assume h4 : initial_total_red_balls = (initial_red_percentage * total_balls)
  assume h5 : initial_total_green_balls = (total_balls - initial_total_red_balls)

  sorry  -- Proof of the theorem (not required per instructions)

end green_balls_to_remove_l226_226029


namespace a_values_in_terms_of_x_l226_226903

open Real

-- Definitions for conditions
variables (a b x y : ℝ)
variables (h1 : a^3 - b^3 = 27 * x^3)
variables (h2 : a - b = y)
variables (h3 : y = 2 * x)

-- Theorem to prove
theorem a_values_in_terms_of_x : 
  (a = x + 5 * x / sqrt 6) ∨ (a = x - 5 * x / sqrt 6) :=
sorry

end a_values_in_terms_of_x_l226_226903


namespace smaller_octagon_half_area_l226_226699

-- Define what it means to be a regular octagon
def is_regular_octagon (O : Point) (ABCDEFGH : List Point) : Prop :=
  -- Definition capturing the properties of a regular octagon around center O
  sorry

-- Define the function that computes the area of an octagon
def area_of_octagon (ABCDEFGH : List Point) : Real :=
  sorry

-- Define the function to create the smaller octagon by joining midpoints
def smaller_octagon (ABCDEFGH : List Point) : List Point :=
  sorry

theorem smaller_octagon_half_area (O : Point) (ABCDEFGH : List Point) :
  is_regular_octagon O ABCDEFGH →
  area_of_octagon (smaller_octagon ABCDEFGH) = (1 / 2) * area_of_octagon ABCDEFGH :=
by
  sorry

end smaller_octagon_half_area_l226_226699


namespace smallest_x_l226_226400

theorem smallest_x (x : ℕ) (h₁ : x % 3 = 2) (h₂ : x % 4 = 3) (h₃ : x % 5 = 4) : x = 59 :=
by
  sorry

end smallest_x_l226_226400


namespace smallest_b_for_factoring_l226_226091

theorem smallest_b_for_factoring :
  ∃ b : ℕ, b > 0 ∧
    (∀ r s : ℤ, r * s = 2016 → r + s ≠ b) ∧
    (∀ r s : ℤ, r * s = 2016 → r + s = b → b = 92) :=
sorry

end smallest_b_for_factoring_l226_226091


namespace jasper_saw_60_rabbits_l226_226971

def cage_initial : ℕ := 13
def added_rabbits : ℕ := 7
def total_cage_rabbits := cage_initial + added_rabbits
def park_rabbits : ℕ := 3 * total_cage_rabbits

theorem jasper_saw_60_rabbits : park_rabbits = 60 :=
by
  have h1 : cage_initial = 13 := rfl
  have h2 : added_rabbits = 7 := rfl
  have h3 : total_cage_rabbits = cage_initial + added_rabbits := rfl
  have h4 : total_cage_rabbits = 20 := by rw [h1, h2]; rfl
  have h5 : park_rabbits = 3 * total_cage_rabbits := rfl
  rw [h4] at h5
  have h6 : park_rabbits = 3 * 20 := h5
  exact h6
  sorry

end jasper_saw_60_rabbits_l226_226971


namespace quadratic_function_positive_l226_226935

theorem quadratic_function_positive {a b c : ℝ} (h : a^2 = b^2 + c^2 - 2 * b * c * real.cos A) :
  ∀ x : ℝ, b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 > 0 :=
begin
  intro x,
  rw [h],
  let f := λ x, b^2 * x^2 + 2 * b * c * real.cos A * x + c^2,
  have discriminant_neg : 4 * b^2 * c^2 * (real.cos A ^ 2 - 1) < 0,
  {
    calc 4 * b^2 * c^2 * (real.cos A ^ 2 - 1) < 0,
    { -- Use the condition that \(b^2 > 0\) and \(\cos^2 A - 1 < 0\)
      sorry }
  },
  by_cases hbc : b = 0,
  {
    -- Handle the case where \(b = 0\)
    sorry
  },
  {
    -- Handle the case where \(b ≠ 0\)
    sorry
  }
end

end quadratic_function_positive_l226_226935


namespace determine_female_athletes_count_l226_226455

theorem determine_female_athletes_count (m : ℕ) (n : ℕ) (x y : ℕ) (probability : ℚ)
  (h_team : 56 + m = 56 + m) -- redundant, but setting up context
  (h_sample_size : n = 28)
  (h_probability : probability = 1 / 28)
  (h_sample_diff : x - y = 4)
  (h_sample_sum : x + y = n)
  (h_ratio : 56 * y = m * x) : m = 42 :=
by
  sorry

end determine_female_athletes_count_l226_226455


namespace second_player_wins_l226_226327

/-- 
Given a pile of 2003 stones, two players alternately select a positive divisor 
of the number of stones currently in the pile and remove that number of stones. 
The player who removes the last stone loses. 
Prove that the second player has a winning strategy by always taking 1 stone. 
-/
theorem second_player_wins : ∃ strategy : (ℕ → ℕ) → ℕ → Prop, strategy ((λ n, 1)) 2003 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end second_player_wins_l226_226327


namespace coupon_savings_difference_l226_226826

theorem coupon_savings_difference {P : ℝ} (hP : P > 200)
  (couponA_savings : ℝ := 0.20 * P) 
  (couponB_savings : ℝ := 50)
  (couponC_savings : ℝ := 0.30 * (P - 200)) :
  (200 ≤ P - 200 + 50 → 200 ≤ P ∧ P ≤ 200 + 400 → 600 - 250 = 350) :=
by
  sorry

end coupon_savings_difference_l226_226826


namespace find_number_l226_226071

theorem find_number (x : ℝ) (h : ((x / 8) + 8 - 30) * 6 = 12) : x = 192 :=
sorry

end find_number_l226_226071


namespace area_ratio_of_smaller_octagon_l226_226718

theorem area_ratio_of_smaller_octagon
    (A B C D E F G H : ℝ × ℝ) -- Coordinates of vertices of the larger octagon
    (P Q R S T U V W : ℝ × ℝ) -- Coordinates of vertices of the smaller octagon
    (regular_octagon : ∀ (X Y Z W U V T S : ℝ × ℝ), regular_octo X Y Z W U V T S)  -- Predicate for regular octagon
    (midpoints_joined : ∀ (X Y : ℝ × ℝ), midpoint X Y) : -- Condition that midpoints form the smaller octagon
  area (smaller_octo P Q R S T U V W) = (3 : ℝ) / 4 * area (larger_octo A B C D E F G H) :=
sorry

end area_ratio_of_smaller_octagon_l226_226718


namespace part_a_part_b_part_c_l226_226856

-- Define a real polynomial p(x)
variable {R : Type*} [CommRing R] (p : R[X])
variable (r s : R)

-- Part (a)
theorem part_a (hp : p.degree > 2) : p.degree = 2 + (p.comp (X + 1) + p.comp (X - 1) - 2 * p).degree := sorry

-- Part (b)
theorem part_b (hp_eq : ∀ x, p.eval (x + 1) + p.eval (x - 1) - r * p.eval x - s = 0) : p.degree ≤ 2 := sorry

-- Part (c)
theorem part_c (hp_eq : ∀ x, p.eval (x + 1) + p.eval (x - 1) - r * p.eval x - s = 0) (hs : s = 0) : p.coeff 2 = 0 := sorry

end part_a_part_b_part_c_l226_226856


namespace ellipse_C1_equation_l226_226135

-- Definitions of the given ellipses
def ellipse_C (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def eccentricity_C : ℝ :=
  1 / 2

def passes_through (x y x0 y0 : ℝ) : Prop :=
  (x = x0) ∧ (y = y0)

-- Proof statement
theorem ellipse_C1_equation (e : ℝ) (x y : ℝ) (hx : passes_through 2 (-sqrt 3) x y) :
  e = eccentricity_C →
  ∃ m n : ℝ, m > n ∧ m > 0 ∧ n > 0 ∧ (
    (m^2 = 8) ∧ (n^2 = 6) ∧ ((x^2 / m^2) + (y^2 / n^2) = 1)) :=
begin
  sorry
end

end ellipse_C1_equation_l226_226135


namespace radius_of_spherical_molds_l226_226813

theorem radius_of_spherical_molds 
    (r_bowl : ℝ) (r_mold : ℝ)
    (volume_bowl : ℝ) (volume_mold : ℝ)
    (n_molds : ℕ)
    (hb : r_bowl = 2)
    (hm : n_molds = 18)
    (vb : volume_bowl = (2/3) * real.pi * r_bowl^3)
    (vm : volume_mold = (4/3) * real.pi * r_mold^3) :
    18 * volume_mold = volume_bowl → 
    r_mold = real.cbrt (2 / 3) :=
by {
    sorry
}

end radius_of_spherical_molds_l226_226813


namespace number_of_teams_in_league_l226_226195

theorem number_of_teams_in_league (n : ℕ) :
  (6 * n * (n - 1)) / 2 = 396 ↔ n = 12 :=
by
  sorry

end number_of_teams_in_league_l226_226195


namespace Patricia_read_21_books_l226_226053

theorem Patricia_read_21_books
  (Candice_books Amanda_books Kara_books Patricia_books : ℕ)
  (h1 : Candice_books = 18)
  (h2 : Candice_books = 3 * Amanda_books)
  (h3 : Kara_books = Amanda_books / 2)
  (h4 : Patricia_books = 7 * Kara_books) :
  Patricia_books = 21 :=
by
  sorry

end Patricia_read_21_books_l226_226053


namespace A_can_complete_work_in_15_days_l226_226807

-- Given assumptions
variables (W : ℝ) -- Total work
variables (A_work B_work : ℝ) -- Amount of work done per day by A and B respectively
variables (A_days : ℝ) -- Number of days A alone can do the work

-- Condition B alone can complete the work in 14.999999999999996 days
def B_work_per_day : ℝ := W / 14.999999999999996

-- Condition A works for 5 days
def A_work_5_days : ℝ := 5 * A_work

-- Condition B completes the remaining work in 10 days
def Remaining_work_by_B : ℝ := 10 * B_work_per_day 
def Total_work_done : Prop := W = (A_work_5_days + Remaining_work_by_B)

-- Defining the main theorem
theorem A_can_complete_work_in_15_days (W A_work B_work : ℝ): Total_work_done W A_work B_work → A_days = 15 := 
sorry

end A_can_complete_work_in_15_days_l226_226807


namespace sum_seq_100_l226_226944

-- Define the sequence terms
noncomputable def a (n : ℕ) : ℝ := 1/2 + n * (1/2)

-- Define the term of our target sequence
noncomputable def seq (n : ℕ) : ℝ := 1 / (a n * a (n + 1))

-- Sum of the first 100 terms of this sequence
theorem sum_seq_100 : 
  (∑ i in Finset.range 100, seq i) = 400 / 101 :=
by
  sorry

end sum_seq_100_l226_226944


namespace area_of_inequality_region_l226_226769

noncomputable def area_of_region : ℝ :=
  let region : set (ℝ × ℝ) := { p | let (x, y) := p in |4 * x - 24| + |3 * y + 9| ≤ 6 } in
  ∫ x in region, 1  -- Integral over the region to find the area

-- Theorem statement to prove
theorem area_of_inequality_region : area_of_region = 72 := sorry

end area_of_inequality_region_l226_226769


namespace graph_represents_two_intersecting_lines_l226_226857

theorem graph_represents_two_intersecting_lines (x y : ℝ) :
  (x - 1) * (x + y + 2) = (y - 1) * (x + y + 2) → 
  (x + y + 2 = 0 ∨ x = y) ∧ 
  (∃ (x y : ℝ), (x = -1 ∧ y = -1 ∧ x = y ∨ x = -y - 2) ∧ (y = x ∨ y = -x - 2)) :=
by
  sorry

end graph_represents_two_intersecting_lines_l226_226857


namespace point_I_lies_on_directrix_if_AB_perpendicular_point_I_lies_on_directrix_if_AB_not_perpendicular_l226_226558

-- Definitions of the given conditions
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1
def is_point_inside_triangle (I A B F₁ : ℝ × ℝ) : Prop := sorry -- Assuming this definition is provided elsewhere

def F₁ : ℝ × ℝ := (sorry, sorry) -- Coordinates of left focus
def F₂ : ℝ × ℝ := (sorry, sorry) -- Coordinates of right focus
def A : ℝ × ℝ := (sorry, sorry) -- Coordinates of point A on hyperbola
def B : ℝ × ℝ := (sorry, sorry) -- Coordinates of point B on hyperbola
def I : ℝ × ℝ := (sorry, sorry) -- Coordinates of point I inside ΔF₁AB

-- Given line equation through F₂ that intersects hyperbola at A and B, and conditions of the problem
axiom AB_perpendicular_x_axis : Prop
axiom distances_eq : (|F₁.1 - B.1| * sorry + |A.1 - B.1| * sorry + |A.1 - F₁.1| * sorry) = 0

-- Statement of the problem in Lean: when AB perpendicular to x-axis, prove I lies on l
theorem point_I_lies_on_directrix_if_AB_perpendicular
  (h1 : AB_perpendicular_x_axis)
  (h2 : is_point_inside_triangle I A B F₁) :
  I.1 = (sqrt 2) / 2 := sorry

-- Statement of the problem in Lean: when AB not perpendicular to x-axis, prove I still lies on l
theorem point_I_lies_on_directrix_if_AB_not_perpendicular
  (h1 : ¬AB_perpendicular_x_axis)
  (h2 : is_point_inside_triangle I A B F₁) :
  I.1 = (sqrt 2) / 2 := sorry

end point_I_lies_on_directrix_if_AB_perpendicular_point_I_lies_on_directrix_if_AB_not_perpendicular_l226_226558


namespace ways_to_select_3_people_l226_226986

theorem ways_to_select_3_people : 
  let ways := nat.choose 6 3 * nat.choose 5 3 * 6
  in ways = 1200 :=
by
  let ways := nat.choose 6 3 * nat.choose 5 3 * 6
  show ways = 1200
  sorry

end ways_to_select_3_people_l226_226986


namespace inscribed_quadrilateral_opposite_angles_sum_180_l226_226281

theorem inscribed_quadrilateral_opposite_angles_sum_180 
    (A B C D : Point)
    (h_inscribed : inscribed_quadrilateral A B C D)
    : ∠ABC + ∠CDA = 180 := 
sorry

end inscribed_quadrilateral_opposite_angles_sum_180_l226_226281


namespace distance_between_foci_hyperbola_l226_226506

theorem distance_between_foci_hyperbola :
  ∀ (x y : ℝ), 9 * x^2 - 27 * x - 16 * y^2 - 32 * y = 72 → dist_foci (9 * x^2 - 27 * x - 16 * y^2 - 32 * y = 72) = (sqrt 41775) / 12 :=
by
  sorry

end distance_between_foci_hyperbola_l226_226506


namespace card_pairing_modulus_l226_226361

theorem card_pairing_modulus (cards : Finset ℕ) (h : cards = (Finset.range 60).image (λ n, n + 1)) :
  ∃ n, n = 8 ∧ ∀ (pairs : Finset (ℕ × ℕ)), (∀ (p ∈ pairs), (p.1 ∈ cards ∧ p.2 ∈ cards ∧ (|p.1 - p.2| = d))) → pairs.card = 30 :=
sorry

end card_pairing_modulus_l226_226361


namespace x_squared_eq_1_iff_x_eq_1_l226_226102

theorem x_squared_eq_1_iff_x_eq_1 (x : ℝ) : (x^2 = 1 → x = 1) ↔ false ∧ (x = 1 → x^2 = 1) :=
by
  sorry

end x_squared_eq_1_iff_x_eq_1_l226_226102


namespace multiple_choice_options_l226_226463

-- Define the problem conditions
def num_true_false_combinations : ℕ := 14
def num_possible_keys (n : ℕ) : ℕ := num_true_false_combinations * n^2
def total_keys : ℕ := 224

-- The theorem problem
theorem multiple_choice_options : ∃ n : ℕ, num_possible_keys n = total_keys ∧ n = 4 := by
  -- We don't need to provide the proof, so we use sorry. 
  sorry

end multiple_choice_options_l226_226463


namespace time_after_2023_minutes_l226_226222

def start_time : Nat := 1 * 60 -- Start time is 1:00 a.m. in minutes from midnight, which is 60 minutes.
def elapsed_time : Nat := 2023 -- The elapsed time is 2023 minutes.

theorem time_after_2023_minutes : (start_time + elapsed_time) % 1440 = 643 := 
by
  -- 1440 represents the total minutes in a day (24 hours * 60 minutes).
  -- 643 represents the time 10:43 a.m. in minutes from midnight. This is obtained as 10 * 60 + 43 = 643.
  sorry

end time_after_2023_minutes_l226_226222


namespace slope_of_line_determined_by_any_two_solutions_l226_226387

theorem slope_of_line_determined_by_any_two_solutions 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 4 / x₁ + 5 / y₁ = 0) 
  (h₂ : 4 / x₂ + 5 / y₂ = 0) 
  (h_distinct : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -5 / 4 := 
sorry

end slope_of_line_determined_by_any_two_solutions_l226_226387


namespace A_cannot_prevent_B_from_winning_l226_226336

-- Define the game's state and transition functions
def game_state (n : ℕ) : Type := ℕ

-- Predicate to check if B wins
def B_wins (n : ℕ) : Prop := n = 0

-- Predicate to check if A can prevent B from winning
def A_can_prevent_B_from_winning (n : game_state n) : Prop := 
  ∀ k : ℕ, ∃ b : ℕ, n ^ k ≠ n - b^2

-- Main theorem: Prove that A cannot prevent B from winning
theorem A_cannot_prevent_B_from_winning (n : ℕ) (h_pos : n > 0) : ¬A_can_prevent_B_from_winning n :=
sorry


end A_cannot_prevent_B_from_winning_l226_226336


namespace infinite_sum_is_zero_l226_226867

-- Define the infinite sum in question
def infinite_sum := ∑' (n : ℕ), if n = 0 then 0 else (n * Real.sin (n : ℝ)) / ((n^2 + 4)^2)

-- State the theorem
theorem infinite_sum_is_zero : infinite_sum = 0 :=
by
  sorry

end infinite_sum_is_zero_l226_226867


namespace numberOfPeopleToWorkFirst_l226_226331

noncomputable def requiredPeople (workRate : ℕ → ℕ → ℕ) : ℕ :=
  let totalPeople := 40
  let time1 := 4
  let additionalPeople := 2
  let time2 := 8
  let work1 (x : ℕ) := workRate x time1
  let work2 (x : ℕ) := workRate (x + additionalPeople) time2
  let totalWork (x : ℕ) := work1 x + work2 x
  x

theorem numberOfPeopleToWorkFirst : 
  requiredPeople (λ x t => x * t / 40) 2 :=
by
  sorry

end numberOfPeopleToWorkFirst_l226_226331


namespace a_minus_b_perpendicular_b_l226_226572

variables (a b : (ℝ × ℝ))
def a : (ℝ × ℝ) := (1, 0)
def b : (ℝ × ℝ) := (1/2, 1/2)

theorem a_minus_b_perpendicular_b : ((a - b) • b = 0) :=
by
  simp [a, b, dot_product]
  sorry

end a_minus_b_perpendicular_b_l226_226572


namespace length_of_street_600_meters_l226_226815

theorem length_of_street_600_meters :
  ∀ (time_minutes : ℕ) (speed_kmph : ℕ), 
  time_minutes = 5 →
  speed_kmph = 7.2 →
  (speed_kmph * 1000 / 60 * time_minutes = 600) :=
by
  intro time_minutes speed_kmph ht hs
  rw [ht, hs]
  norm_num
  sorry

end length_of_street_600_meters_l226_226815


namespace find_angle_C_l226_226977

-- Definitions and conditions from the problem.
variable {A B C : ℝ} {a b c : ℝ} 

-- Conditions given in the problem.
axiom h1 : a^2 = 3 * b^2 + 3 * c^2 - 2 * real.sqrt 3 * b * c * real.sin A

-- Definition of a triangle.
noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = real.pi

-- Desired angle C.
theorem find_angle_C (hABC : triangle_ABC A B C a b c) : C = real.pi / 6 :=
by
  sorry

end find_angle_C_l226_226977


namespace find_n_l226_226745

noncomputable def seq (n : ℕ) : ℝ :=
  1 / (Real.sqrt (n + 1) + Real.sqrt n)

noncomputable def sum_seq (n : ℕ) : ℝ :=
  (List.range n).sum (λ i => seq i)

theorem find_n (n : ℕ) (h : sum_seq n = 5) : n = 35 :=
  sorry

end find_n_l226_226745


namespace carnival_ticket_count_l226_226423

theorem carnival_ticket_count (ferris_wheel_rides bumper_car_rides ride_cost : ℕ) 
  (h1 : ferris_wheel_rides = 7) 
  (h2 : bumper_car_rides = 3) 
  (h3 : ride_cost = 5) : 
  ferris_wheel_rides + bumper_car_rides * ride_cost = 50 := 
by {
  -- proof omitted
  sorry
}

end carnival_ticket_count_l226_226423


namespace isosceles_triangle_area_ratio_l226_226988

/--
In an isosceles triangle \( ABC \) (\(AB = BC\)), the altitudes \( AA_1 \), \( BB_1 \), and \( CC_1 \) are drawn.
Given \( \frac{AB}{A_{1} B_{1}} = \sqrt{3} \), prove that the ratio of the area of triangle \( A_{1} B_{1} C_{1} \) to the area of triangle \( A B C \) is \( \frac{2}{9} \).
-/
theorem isosceles_triangle_area_ratio
  (ABC : Type)
  [triangle ABC]
  (isosceles : is_isosceles ABC)
  (altitudes_drawn : altitudes_drawn ABC)
  (ratio_AB_A1B1 : (AB / A1B1) = sqrt 3) :
  let S := area ABC in
  let S_A1B1C1 := area A1B1C1 in
  (S_A1B1C1 / S) = 2 / 9 := sorry

end isosceles_triangle_area_ratio_l226_226988


namespace sector_angle_degree_measure_l226_226449

-- Define the variables and conditions
variables (θ r : ℝ)
axiom h1 : (1 / 2) * θ * r^2 = 1
axiom h2 : 2 * r + θ * r = 4

-- Define the theorem to be proved
theorem sector_angle_degree_measure (θ r : ℝ) (h1 : (1 / 2) * θ * r^2 = 1) (h2 : 2 * r + θ * r = 4) : θ = 2 :=
sorry

end sector_angle_degree_measure_l226_226449


namespace smallest_M_l226_226517

theorem smallest_M :
  ∃ (M : ℕ),
    M > 0 ∧
    (∃ i ∈ {0, 1, 2}, (M + i) % 8 = 0) ∧
    (∃ i ∈ {0, 1, 2}, (M + i) % 27 = 0) ∧
    (∃ i ∈ {0, 1, 2}, (M + i) % 125 = 0) ∧
    (∃ i ∈ {0, 1, 2}, (M + i) % 343 = 0) ∧
    M = 1029 :=
by sorry

end smallest_M_l226_226517


namespace max_no_real_solutions_l226_226116

variable {α : Type*} [LinearOrderedField α]

-- Define the arithmetic sequences
variable (a b : ℕ → α)

-- Define the sums of the sequences up to the n-th term
def S (n : ℕ) := ∑ i in Finset.range (n + 1), a i
def T (n : ℕ) := ∑ i in Finset.range (n + 1), b i

-- Given condition that the equation has real solutions
axiom real_solutions : 2023 * (a 1012)^2 - 4 * 2023 * T 2023 ≥ 0

-- Define the quadratic equation for each i
def quadratic_discriminant (i : ℕ) : α := (a i)^2 - 4 * (b i)

-- The Lean theorem we want to prove
theorem max_no_real_solutions {α : Type*} [LinearOrderedField α]:
  (∑ i in Finset.range 1012, if (a i)^2 - 4 * b i < 0 then 1 else 0) ≤ 1011 :=
sorry

end max_no_real_solutions_l226_226116


namespace area_ratio_of_smaller_octagon_l226_226719

theorem area_ratio_of_smaller_octagon
    (A B C D E F G H : ℝ × ℝ) -- Coordinates of vertices of the larger octagon
    (P Q R S T U V W : ℝ × ℝ) -- Coordinates of vertices of the smaller octagon
    (regular_octagon : ∀ (X Y Z W U V T S : ℝ × ℝ), regular_octo X Y Z W U V T S)  -- Predicate for regular octagon
    (midpoints_joined : ∀ (X Y : ℝ × ℝ), midpoint X Y) : -- Condition that midpoints form the smaller octagon
  area (smaller_octo P Q R S T U V W) = (3 : ℝ) / 4 * area (larger_octo A B C D E F G H) :=
sorry

end area_ratio_of_smaller_octagon_l226_226719


namespace smallest_perimeter_triangle_DEF_l226_226978

theorem smallest_perimeter_triangle_DEF : 
  ∃ (d e f : ℤ), 
  (cos_angle D = 15 / 17) →
  (cos_angle E = 3 / 5) →
  (cos_angle F = -1 / 8) → 
  let perimeter := d + e + f in
  perimeter = 504 :=
begin
  sorry
end

end smallest_perimeter_triangle_DEF_l226_226978


namespace binomial_expansion_a_values_l226_226585

theorem binomial_expansion_a_values (n a : ℤ) 
  (h1 : (2 : ℤ)^n = 64) 
  (h2 : (1 + a)^n = 729) :
  a = -4 ∨ a = 2 :=
by
  sorry

end binomial_expansion_a_values_l226_226585


namespace ariana_total_owed_l226_226838

noncomputable def ariana_owes (first_bill amount : ℝ) (first_interest : ℝ) (first_months : ℝ) 
  (second_bill amount : ℝ) (second_late_fee : ℝ) (second_months : ℝ)
  (third_bill amount : ℝ)(third_fee : ℝ) (third_months : ℕ) : ℝ :=
 let first_total := amount * (1 + first_interest/12)^(first_months) in
 let second_total := second_bill + second_late_fee * second_months in
 let rec third_total (fee : ℝ) (months : ℕ) : ℝ :=
   match months with
   | 0     => 0
   | (n+1) => fee + third_total (fee * 2) n
 in first_total + second_total + (third_bill + third_total third_fee third_months)

theorem ariana_total_owed :
  ariana_owes 200 0.10 3 130 50 8 444 40 4 = 1779.031 :=
begin
  sorry
end

end ariana_total_owed_l226_226838


namespace zachary_pushups_l226_226791

theorem zachary_pushups (C P : ℕ) (h1 : C = 14) (h2 : P + C = 67) : P = 53 :=
by
  rw [h1] at h2
  linarith

end zachary_pushups_l226_226791


namespace smallest_x_l226_226406

theorem smallest_x (x : ℕ) : (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) → x = 59 :=
by
  intro h
  sorry

end smallest_x_l226_226406


namespace card_pairing_modulus_l226_226359

theorem card_pairing_modulus (cards : Finset ℕ) (h : cards = (Finset.range 60).image (λ n, n + 1)) :
  ∃ n, n = 8 ∧ ∀ (pairs : Finset (ℕ × ℕ)), (∀ (p ∈ pairs), (p.1 ∈ cards ∧ p.2 ∈ cards ∧ (|p.1 - p.2| = d))) → pairs.card = 30 :=
sorry

end card_pairing_modulus_l226_226359


namespace geometric_subsequence_unique_geometric_sequence_with_infinite_arithmetic_subsequence_l226_226859

-- Problem (1): Geometric subsequences

def sequence := [1, (1/2), (1/3), (1/4), (1/5)]

def is_geometric_subsequence (seq : List ℚ) : Prop :=
  ∃ (r : ℚ), ∀ i j, i < j → seq.get? i * r = seq.get? j

theorem geometric_subsequence_unique : ∀ (seq : List ℚ),
  seq = sequence →
  ∃ (subseq : List ℚ), subseq = [1, (1/2), (1/4)] ∧ is_geometric_subsequence subseq :=
sorry

-- Problem (2)(ii): Infinite arithmetic subsequence in geometric sequence

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ a₁, ∀ n, a (n + 1) = a₁ * q ^ (n + 1)

def contains_infinite_arithmetic_subsequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, ∃ m > n, a m = a n + d

theorem geometric_sequence_with_infinite_arithmetic_subsequence
  (a : ℕ → ℝ) (q : ℝ) (h1 : is_geometric_sequence a q)
  (h2 : contains_infinite_arithmetic_subsequence a) : q = -1 :=
sorry

-- Problem 2(i) is considered trivial and does not involve a theorem that needs proof in Lean.

end geometric_subsequence_unique_geometric_sequence_with_infinite_arithmetic_subsequence_l226_226859


namespace polynomial_evaluation_l226_226045

theorem polynomial_evaluation :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 :=
by
  sorry

end polynomial_evaluation_l226_226045


namespace number_of_valid_pairing_ways_l226_226340

-- Define a natural number as a condition.
def is_natural (n : ℕ) : Prop := 0 < n

-- Define that 60 cards can be paired with the same modulus difference.
def pair_cards_same_modulus_difference (d : ℕ) (k : ℕ) : Prop :=
  60 = 2 * d * k

-- Define what it means for d to be a divisor of 30.
def is_divisor_of_30 (d : ℕ) : Prop :=
  ∃ k, 30 = d * k

theorem number_of_valid_pairing_ways :
  (finset.univ.filter is_divisor_of_30).card = 8 :=
begin
  sorry
end

end number_of_valid_pairing_ways_l226_226340


namespace int_solution_count_l226_226165

theorem int_solution_count :
  let count_solutions (eq : ℤ → ℤ → Bool) : Nat :=
    Finset.card (Finset.filter (λ ⟨(y, x)⟩, eq y x) Finset.univ.prod Finset.univ)
  count_solutions (λ y x, 6 * y^2 + 3 * y * x + x + 2 * y + 180 = 0) = 6 :=
sorry

end int_solution_count_l226_226165


namespace water_hyacinth_half_filled_in_9_days_l226_226018

theorem water_hyacinth_half_filled_in_9_days :
  ∀ (S : ℕ), (∃ S = 2^10, 2^9 = S / 2) → 9 = 10 - 1 :=
by
  sorry

end water_hyacinth_half_filled_in_9_days_l226_226018


namespace solve_rational_eq_l226_226083

theorem solve_rational_eq (x : ℝ) :
  (1 / (x^2 + 14*x - 36)) + (1 / (x^2 + 5*x - 14)) + (1 / (x^2 - 16*x - 36)) = 0 ↔ 
  x = 9 ∨ x = -4 ∨ x = 12 ∨ x = 3 :=
sorry

end solve_rational_eq_l226_226083


namespace smaller_octagon_area_fraction_l226_226727

theorem smaller_octagon_area_fraction (A B C D E F G H : Point)
  (midpoints_joined : Boolean)
  (regular_octagon : RegularOctagon A B C D E F G H)
  (smaller_octagon : Octagon (midpoint (A, B)) (midpoint (B, C)) (midpoint (C, D)) 
                              (midpoint (D, E)) (midpoint (E, F)) (midpoint (F, G))
                              (midpoint (G, H)) (midpoint (H, A))) :
  midpoints_joined → regular_octagon → 
  (area smaller_octagon) = (3 / 4) * (area regular_octagon) :=
by
  sorry

end smaller_octagon_area_fraction_l226_226727


namespace ratio_CD_PQ_l226_226218

variable {A B C D P Q: Type} {AD BC CD PQ : ℝ}
variable [IsTrapezoid A B C D]
variable (h1: AD = 4 * BC)
variable (h2: ∠BCD = 2 * ∠BAD)
variable (h3: PQ = (BC + AD) / 2)

theorem ratio_CD_PQ (h1 : AD = 4 * BC) (h2 : ∠BCD = 2 * ∠BAD) (h3 : PQ = (BC + AD) / 2)
: (CD / PQ) = 6 / 5 := 
sorry

end ratio_CD_PQ_l226_226218


namespace smallest_number_jungkook_l226_226231

theorem smallest_number_jungkook (jungkook yoongi yuna : ℕ) 
  (hj : jungkook = 6 - 3) (hy : yoongi = 4) (hu : yuna = 5) : 
  jungkook < yoongi ∧ jungkook < yuna :=
by
  sorry

end smallest_number_jungkook_l226_226231


namespace smallest_digit_for_divisibility_by_3_l226_226090

theorem smallest_digit_for_divisibility_by_3 : ∃ x : ℕ, x < 10 ∧ (5 + 2 + 6 + x + 1 + 8) % 3 = 0 ∧ ∀ y : ℕ, y < 10 ∧ (5 + 2 + 6 + y + 1 + 8) % 3 = 0 → x ≤ y := by
  sorry

end smallest_digit_for_divisibility_by_3_l226_226090


namespace B_n_eq_zero_iff_l226_226154

open Nat Set

def A : ℕ → Set ℕ
| 0     => ∅
| (n+1) => {x + 1 | x ∈ B n}

def B : ℕ → Set ℕ
| 0     => {0}
| (n+1) => (A n ∪ B n) \ (A n ∩ B n)

theorem B_n_eq_zero_iff (n : ℕ) : 
  B n = {0} ↔ ∃ α : ℕ, n = 2^α :=
sorry

end B_n_eq_zero_iff_l226_226154


namespace length_dc_eq_l1_plus_l2_l226_226996

/-- In a circle, if E is the midpoint of the arc ABEC, and ED is perpendicular to the chord BC
at D. Given AB = l₁ and BD = l₂, then DC = l₂ + l₁. -/
theorem length_dc_eq_l1_plus_l2 (E A B C D : Point) (l1 l2 : ℝ)
  (h_midpoint : E = midpoint_arc A B E C)
  (h_perpendicular : perpendicular (segment E D) (segment B C))
  (h_AB : length (segment A B) = l1)
  (h_BD : length (segment B D) = l2) :
  length (segment D C) = l2 + l1 :=
by
  sorry

end length_dc_eq_l1_plus_l2_l226_226996


namespace anthony_success_rate_increase_l226_226837

theorem anthony_success_rate_increase :
  ∀ (initial_success : ℕ) (initial_attempts : ℕ) (next_success_rate : ℚ) (next_attempts : ℕ),
  initial_success = 4 →
  initial_attempts = 10 →
  next_success_rate = 3 / 4 →
  next_attempts = 28 →
  let total_success := initial_success + (next_success_rate * next_attempts).to_nat in
  let total_attempts := initial_attempts + next_attempts in
  let initial_rate := (initial_success : ℚ) / initial_attempts in
  let new_rate := (total_success : ℚ) / total_attempts in
  ((new_rate - initial_rate) * 100).round = 26 := 
by {
  intros,
  sorry,
}

end anthony_success_rate_increase_l226_226837


namespace geometric_sum_ratio_l226_226596

theorem geometric_sum_ratio (a₁ q : ℝ) (h₁ : q ≠ 1) (h₂ : (1 - q^4) / (1 - q^2) = 5) :
  (1 - q^8) / (1 - q^4) = 17 := 
by
  sorry

end geometric_sum_ratio_l226_226596


namespace line_tangent_to_ellipse_l226_226863

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * (m * x + 2)^2 = 3) ∧ 
  (∀ x1 x2 : ℝ, (2 + 3 * m^2) * x1^2 + 12 * m * x1 + 9 = 0 ∧ 
                (2 + 3 * m^2) * x2^2 + 12 * m * x2 + 9 = 0 → x1 = x2) ↔ m^2 = 2 := 
sorry

end line_tangent_to_ellipse_l226_226863


namespace number_of_arrangements_l226_226525

theorem number_of_arrangements (n : ℕ) : 
  let total_arrangements := (nat.factorial (n^2)) in
  let invalid_arrangements := (nat.choose (n^2) (2n - 1)) * (n^2) * (nat.factorial (n - 1)) * (nat.factorial (n - 1)) * (nat.factorial (n^2 - 2n + 1)) in
  total_arrangements - invalid_arrangements = (nat.factorial (n^2)) - (nat.choose (n^2) (2n - 1)) * (n^2) * (nat.factorial (n - 1)) * (nat.factorial (n - 1)) * (nat.factorial (n^2 - 2n + 1)) := 
sorry

end number_of_arrangements_l226_226525


namespace minimum_area_OAB_l226_226217

-- Define the properties of the line (l_1) and the parabola (C).
variables {α : ℝ} (hα1 : 0 ≤ α) (hα2 : α < π) (hα3 : α ≠ π / 2)

-- Polar equation definitions based on the problem.
def polar_line_eq (θ : ℝ) : Prop := θ = α
def polar_parabola_eq (ρ θ : ℝ) : Prop := ρ * sin(θ) ^ 2 = 4 * cos(θ)

-- Area computation for triangle OAB
noncomputable def area_OAB (ρA ρB : ℝ) : ℝ := (1 / 2) * abs(ρA) * abs(ρB)

-- rho_A and rho_B based on the intersection conditions.
def rho_A (α : ℝ) : ℝ := 4 * cos(α) / (sin(α) ^ 2)
def rho_B (α : ℝ) : ℝ := -4 * sin(α) / (cos(α) ^ 2)

-- Prove the minimum area of triangle OAB is 16.
theorem minimum_area_OAB :
  ∀ {α : ℝ}, (0 ≤ α ∧ α < π ∧ α ≠ π / 2) → (area_OAB (rho_A α) (rho_B α) ≥ 16) :=
by
  intros α hα
  sorry

end minimum_area_OAB_l226_226217


namespace milk_days_calculation_l226_226181

-- Given conditions
variables (y : ℕ)

def milk_production_in_days (cows milk_days : ℕ) : ℕ := (cows * milk_days)

-- Prove the number of days required for (y + 5) cows to yield (y + 8) cans of milk 
-- given that y cows yield (y + 2) cans of milk in (y + 4) days.
theorem milk_days_calculation (y : ℕ) (h: milk_production_in_days y (y + 4) = y * (y + 2)) :
  let daily_production_per_cow := (y + 2) / (y * (y + 4)) in
  let total_daily_production_plus_5_cows := (y + 5) * daily_production_per_cow in
  let days_for_y_plus_5_cows := (y + 8) / total_daily_production_plus_5_cows in
  days_for_y_plus_5_cows = (y * (y + 4) * (y + 8)) / ((y + 2) * (y + 5)) :=
sorry

end milk_days_calculation_l226_226181


namespace proof_problem_l226_226930

noncomputable def ellipse_eqn (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (a^2 = 4 ∧ b^2 = 2)

noncomputable def area_triangle (P Q F1 F2 : ℝ × ℝ) (h1 : P ∈ ellipse_eqn 2 √2)
(h2 : ∠ F1 P F2 = π/2) : ℝ :=
  1/2 * dist P F1 * dist P F2

theorem proof_problem (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) (a b : ℝ)
(h1 : a > b) (h2 : b > 0) (h3 : Q = (- √2, 1)) (h4 : midpoint Q F2 = (0, some_y))
(h5 : F2 = (√2, 0)) (h6 : angle F1 P F2 = π/2) : 
  ellipse_eqn a b h1 h2 ∧ area_triangle P Q F1 F2 h1 h2 = 2 :=
by
  sorry

end proof_problem_l226_226930


namespace surveys_on_tuesday_l226_226033

theorem surveys_on_tuesday
  (num_surveys_monday: ℕ) -- number of surveys Bart completed on Monday
  (earnings_monday: ℕ) -- earning per survey on Monday
  (total_earnings: ℕ) -- total earnings over the two days
  (earnings_per_survey: ℕ) -- earnings Bart gets per survey
  (monday_earnings_eq : earnings_monday = num_surveys_monday * earnings_per_survey)
  (total_earnings_eq : total_earnings = earnings_monday + (8 : ℕ))
  (earnings_per_survey_eq : earnings_per_survey = 2)
  : ((8 : ℕ) / earnings_per_survey = 4) := sorry

end surveys_on_tuesday_l226_226033


namespace tangent_line_through_pointA_l226_226509

-- Define the equation of the circle
def circle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 4

-- Define the point (0, sqrt(3))
def pointA (x y : ℝ) : Prop :=
  x = 0 ∧ y = sqrt 3

-- Define the candidate tangent line equation
def tangent_line (x y : ℝ) : Prop :=
  y = (sqrt 3 / 3) * x + sqrt 3

-- The statement to prove
theorem tangent_line_through_pointA :
  ∃ x y, pointA x y ∧ ∃ m b, tangent_line x y ∧ 
  (∀ x y, circle x y → y = m * x + b → (x - 1) * m + y = b) :=
sorry

end tangent_line_through_pointA_l226_226509


namespace range_y_l226_226551

def f (x : ℝ) : ℝ := 2 + log x / log 3

theorem range_y (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 9) : 
  let y := (f x)^2 + f (x^2) in 
  6 ≤ y ∧ y ≤ 13 :=
sorry

end range_y_l226_226551


namespace slope_of_line_determined_by_any_two_solutions_l226_226389

theorem slope_of_line_determined_by_any_two_solutions 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 4 / x₁ + 5 / y₁ = 0) 
  (h₂ : 4 / x₂ + 5 / y₂ = 0) 
  (h_distinct : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -5 / 4 := 
sorry

end slope_of_line_determined_by_any_two_solutions_l226_226389


namespace Willy_Lucy_more_crayons_l226_226787

def Willy_initial : ℕ := 1400
def Lucy_initial : ℕ := 290
def Max_crayons : ℕ := 650
def Willy_giveaway_percent : ℚ := 25 / 100
def Lucy_giveaway_percent : ℚ := 10 / 100

theorem Willy_Lucy_more_crayons :
  let Willy_remaining := Willy_initial - Willy_initial * Willy_giveaway_percent
  let Lucy_remaining := Lucy_initial - Lucy_initial * Lucy_giveaway_percent
  Willy_remaining + Lucy_remaining - Max_crayons = 661 := by
  sorry

end Willy_Lucy_more_crayons_l226_226787


namespace triangle_base_length_l226_226686

-- Given conditions
def area_triangle (base height : ℕ) : ℕ := (1 / 2 : ℚ) * base * height

-- Problem statement
theorem triangle_base_length (A h : ℕ) (A_eq : A = 24) (h_eq : h = 8) :
  ∃ b : ℕ, area_triangle b h = A ∧ b = 6 := 
by
  sorry

end triangle_base_length_l226_226686


namespace sandy_total_cost_is_correct_l226_226669

-- Definitions for the areas
def wall_area := 8 * 6
def total_wall_area := 2 * wall_area

def roof_area := (1 / 2) * 8 * 5
def total_roof_area := 2 * roof_area

def total_area_to_cover := total_wall_area + total_roof_area

-- Definitions for siding sections
def section_area := 10 * 12
def number_of_sections := Int.ceil (total_area_to_cover / section_area)

def section_cost := 30
def total_cost := number_of_sections * section_cost

-- The theorem to prove
theorem sandy_total_cost_is_correct : total_cost = 60 :=
by
  -- We are not including the proof as per the requirements
  sorry

end sandy_total_cost_is_correct_l226_226669


namespace bug_converges_to_final_position_l226_226434

noncomputable def bug_final_position : ℝ × ℝ := 
  let horizontal_sum := ∑' n, if n % 4 = 0 then (1 / 4) ^ (n / 4) else 0
  let vertical_sum := ∑' n, if n % 4 = 1 then (1 / 4) ^ (n / 4) else 0
  (horizontal_sum, vertical_sum)

theorem bug_converges_to_final_position : bug_final_position = (4 / 5, 2 / 5) := 
  sorry

end bug_converges_to_final_position_l226_226434


namespace min_time_for_students_to_re_enter_check_disinfection_effectiveness_l226_226808

-- Define the conditions and relationships
def concentration_while_burning (x : ℝ) : ℝ := (3 / 4) * x
def concentration_after_burning (x : ℝ) : ℝ := 48 / x

-- Given conditions
def burns_out_at_8_min : Prop := concentration_while_burning 8 = 6
def students_can_enter (x : ℝ) : Prop := concentration_after_burning x < 1.6
def effective_disinfection (x : ℝ) : Prop := ∀ (t : ℝ), 8 ≤ t ∧ t ≤ x → concentration_after_burning t ≥ 3
def kills_bacteria (x : ℝ) : Prop := x - 8 ≥ 10

-- Proof goals
theorem min_time_for_students_to_re_enter : ∃ x, students_can_enter x ∧ x = 30 := by
  sorry

theorem check_disinfection_effectiveness : effective_disinfection 18 := by
  sorry

end min_time_for_students_to_re_enter_check_disinfection_effectiveness_l226_226808


namespace isosceles_triangle_grasshopper_angle_l226_226924

open Real

theorem isosceles_triangle_grasshopper_angle 
  (A B C : Point)
  (is_isosceles : ∀ (A B C : Point), isIsosceles A B C)
  (grasshopper_jumps : ∀ (n : ℕ) (P : Point → Point), P(22) = A)
  (odd_even_sides : ∀ (n : ℕ), odd_sides_alternation (C B A))
  (closer_to_A : ∀ (n : ℕ), closer_to A) :
  (angle A B C = 88) ∧ (angle A C B = 88) ∧ (angle B A C = 4) := 
sorry

end isosceles_triangle_grasshopper_angle_l226_226924


namespace electronics_store_sales_l226_226619

-- Definition of variables and conditions for microwaves
variable (m1 m2 : ℕ) (d1 d2 : ℕ)
variable (j : ℕ)

-- Definition of variables and conditions for toasters
variable (p1 p2 : ℕ) (c1 c2 : ℕ)
variable (k : ℕ)

def microwave_proportionality : Prop :=
  (m1 * d1 = j) → (m2 * d2 = j) → d1 = 400 → m1 = 10 → d2 = 800 → m2 = 5

def toaster_proportionality : Prop :=
  (p1 * c1 = k) → (p2 * c2 = k) → c1 = 600 → p1 = 6 → c2 = 1000 → p2 = 4

theorem electronics_store_sales :
  microwave_proportionality m1 m2 d1 d2 j ∧ toaster_proportionality p1 p2 c1 c2 k :=
begin
  sorry, -- Proof not required
end

end electronics_store_sales_l226_226619


namespace tribe_leadership_ways_l226_226025

noncomputable def calculate_ways (total_members supporting_chiefs_per_chief num_inferiors_per_chief : ℕ) : ℕ :=
  let choose := nat.choose
  total_members *
  (total_members - 1) *
  (total_members - 2) *
  (total_members - 3) *
  choose (total_members - 4) supporting_chiefs_per_chief *
  choose (total_members - 4 - supporting_chiefs_per_chief) supporting_chiefs_per_chief *
  choose (total_members - 4 - 2 * supporting_chiefs_per_chief) supporting_chiefs_per_chief

theorem tribe_leadership_ways :
  calculate_ways 13 3 2 = 12355200 := by
  sorry

end tribe_leadership_ways_l226_226025


namespace distance_from_center_to_plane_is_zero_l226_226013

noncomputable def sphere_center_distance_to_triangle_plane : ℝ :=
  let P := (0, 0, 0) in -- assuming an arbitrary point P which does not affect the distance calculation
  let r := 4 in         -- radius of the sphere
  let a := 13 in
  let b := 14 in
  let c := 15 in
  let s := (a + b + c) / 2 in -- semiperimeter
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in -- Heron's formula for triangle area
  let inradius := A / s in -- inradius of the triangle
  0 -- the distance from P to the plane determined by the triangle

theorem distance_from_center_to_plane_is_zero :
  sphere_center_distance_to_triangle_plane = 0 :=
by sorry

end distance_from_center_to_plane_is_zero_l226_226013


namespace least_unboxed_balls_l226_226000

theorem least_unboxed_balls (total_balls big_box_capacity small_box_capacity : ℕ) 
                            (h_total : total_balls = 104)
                            (h_big_box : big_box_capacity = 25)
                            (h_small_box : small_box_capacity = 20) :
  ∃ (unboxed_balls : ℕ), unboxed_balls = 4 :=
by
  use 4
  sorry

end least_unboxed_balls_l226_226000


namespace certain_positive_integer_value_l226_226193

theorem certain_positive_integer_value :
  ∃ (i m p : ℕ), (x = 2 ^ i * 3 ^ 2 * 5 ^ m * 7 ^ p) ∧ (i + 2 + m + p = 11) :=
by
  let x := 40320 -- 8!
  sorry

end certain_positive_integer_value_l226_226193


namespace smaller_octagon_area_fraction_l226_226724

theorem smaller_octagon_area_fraction (A B C D E F G H : Point)
  (midpoints_joined : Boolean)
  (regular_octagon : RegularOctagon A B C D E F G H)
  (smaller_octagon : Octagon (midpoint (A, B)) (midpoint (B, C)) (midpoint (C, D)) 
                              (midpoint (D, E)) (midpoint (E, F)) (midpoint (F, G))
                              (midpoint (G, H)) (midpoint (H, A))) :
  midpoints_joined → regular_octagon → 
  (area smaller_octagon) = (3 / 4) * (area regular_octagon) :=
by
  sorry

end smaller_octagon_area_fraction_l226_226724


namespace nth_term_arithmetic_sequence_sum_of_b_sequence_less_than_one_sixth_l226_226933

/- Definitions based on conditions -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d > 0, ∀ n, a n+1 = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range (n + 1), a i

def b_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  1 / (a n * a (n + 1))

def t_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range n, b_sequence a i

/- problem 1: Finding the nth term of the sequence -/
theorem nth_term_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (h₀: a 1 * a 4 = 22) (h₁: sum_of_first_n_terms a 4 = 26) (h₂: d > 0) :
  a = λ n, 3n - 1 :=
  sorry

/- problem 2: Proving that T_n < 1/6 -/
theorem sum_of_b_sequence_less_than_one_sixth (a : ℕ → ℝ) (d : ℝ) (h₀: a 1 * a 4 = 22) (h₁: sum_of_first_n_terms a 4 = 26) (h₂: d > 0) :
  ∀ n, t_sequence a n < 1 / 6 :=
  sorry

end nth_term_arithmetic_sequence_sum_of_b_sequence_less_than_one_sixth_l226_226933


namespace simplify_trig_expr_l226_226673

theorem simplify_trig_expr : 
  Real.cot (Real.pi / 9) + Real.tan (Real.pi / 18) = Real.csc (Real.pi / 9) :=
by
  -- proof to be filled
  sorry

end simplify_trig_expr_l226_226673


namespace problem_statement_l226_226580

theorem problem_statement (f : ℕ → ℤ) (a b : ℤ) 
  (h1 : f 1 = 7) 
  (h2 : f 2 = 11)
  (h3 : ∀ x, f x = a * x^2 + b * x + 3) :
  f 3 = 15 := 
sorry

end problem_statement_l226_226580


namespace sum_of_squares_500_l226_226804

theorem sum_of_squares_500 : (Finset.range 500).sum (λ x => (x + 1) ^ 2) = 41841791750 := by
  sorry

end sum_of_squares_500_l226_226804


namespace largest_5_digit_congruent_15_mod_24_l226_226376

theorem largest_5_digit_congruent_15_mod_24 : ∃ x, 10000 ≤ x ∧ x < 100000 ∧ x % 24 = 15 ∧ x = 99999 := by
  sorry

end largest_5_digit_congruent_15_mod_24_l226_226376


namespace sum_of_prime_factors_of_1365_l226_226780

theorem sum_of_prime_factors_of_1365 : 
  let smallest_prime_factor := 3 in
  let largest_prime_factor := 13 in
  smallest_prime_factor + largest_prime_factor = 16 :=
by
  let smallest_prime_factor := 3
  let largest_prime_factor := 13
  have h : smallest_prime_factor + largest_prime_factor = 16 := by
    -- the proof will be provided here, but it is not required in this step
    sorry
  exact h

end sum_of_prime_factors_of_1365_l226_226780


namespace patricia_books_read_l226_226051

noncomputable def calculate_books := 
  λ (Candice_read : ℕ) =>
    let Amanda_read := Candice_read / 3
    let Kara_read := Amanda_read / 2
    let Patricia_read := 7 * Kara_read
    Patricia_read

theorem patricia_books_read (Candice_books : ℕ) (hC : Candice_books = 18) :
  calculate_books Candice_books = 21 := by
  rw [hC]
  unfold calculate_books
  simp
  sorry

end patricia_books_read_l226_226051


namespace area_ratio_of_smaller_octagon_l226_226720

theorem area_ratio_of_smaller_octagon
    (A B C D E F G H : ℝ × ℝ) -- Coordinates of vertices of the larger octagon
    (P Q R S T U V W : ℝ × ℝ) -- Coordinates of vertices of the smaller octagon
    (regular_octagon : ∀ (X Y Z W U V T S : ℝ × ℝ), regular_octo X Y Z W U V T S)  -- Predicate for regular octagon
    (midpoints_joined : ∀ (X Y : ℝ × ℝ), midpoint X Y) : -- Condition that midpoints form the smaller octagon
  area (smaller_octo P Q R S T U V W) = (3 : ℝ) / 4 * area (larger_octo A B C D E F G H) :=
sorry

end area_ratio_of_smaller_octagon_l226_226720


namespace range_of_a_l226_226555

noncomputable def f (x a : ℝ) := (x^2 + a * x + 11) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x : ℕ, x > 0 → f x a ≥ 3) ↔ (a ≥ -8 / 3) :=
by sorry

end range_of_a_l226_226555


namespace collinear_midpoints_and_intersection_l226_226256

noncomputable def collinear_points (ABC : Triangle) (I D E P M N : Point) : Prop :=
  let A := ABC.A
  let B := ABC.B
  let C := ABC.C
  -- I is the incenter of triangle ABC
  incenter I ABC ∧
  -- D is the point where the incircle touches BC
  incircle_touches_side D BC I ABC ∧
  -- E is the point where the incircle touches AC
  incircle_touches_side E AC I ABC ∧
  -- P is the intersection of lines AI and DE
  intersection P (line_through A I) (line_through D E) ∧
  -- M is the midpoint of side BC
  midpoint M B C ∧
  -- N is the midpoint of side AB
  midpoint N A B ∧
  -- The points M, N, and P are collinear
  collinear M N P

-- Statement of the theorem
theorem collinear_midpoints_and_intersection
  (ABC : Triangle) (I D E P M N : Point) :
  collinear_points ABC I D E P M N := 
sorry

end collinear_midpoints_and_intersection_l226_226256


namespace problem_I_problem_II_l226_226959

def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + x
def g (m x : ℝ) : ℝ := (x - 2) * Real.exp x - x^2 + m
def x1 (a : ℝ) := (1 - Real.sqrt (1 + 8 * a)) / 2
def x2 (a : ℝ) := (1 + Real.sqrt (1 + 8 * a)) / 2

theorem problem_I (a : ℝ) (x : ℝ) (h₀ : 0 < x) (h₁ : a ≤ 0) :
  (a ≤ -1/8 → (∀ x, (0 < x) → (f a x).monotonically_decreasing_on (0, ∞))) ∧
  (-1/8 < a → (∀ x, (0 < x) → 
    ((0 < x ∧ x < x1 a) → (f a x).monotonically_decreasing_on (0, x1 a)) ∧ 
    ((x1 a < x ∧ x < x2 a) → (f a x).monotonically_increasing_on (x1 a, x2 a)) ∧ 
    ((x2 a < x ∧ x < ∞) → (f a x).monotonically_decreasing_on (x2 a, ∞)))) := sorry

theorem problem_II (m : ℝ) :
  (∀ x, (0 < x ∧ x ≤ 1) → f (-1) x > g m x) → m ≤ 3 := sorry

end problem_I_problem_II_l226_226959


namespace sin_of_acute_angle_l226_226124

theorem sin_of_acute_angle (α : ℝ) (hα1 : α > 0) (hα2 : α < π / 2)
  (h_cos : cos (α + π / 4) = 3 / 5) : sin α = sqrt 2 / 10 :=
sorry

end sin_of_acute_angle_l226_226124


namespace prob_B_occurs_l226_226795

noncomputable def prob_A := 0.4
noncomputable def prob_A_and_B := 0.25
noncomputable def prob_A_or_B := 0.8

theorem prob_B_occurs : prob_A_or_B = prob_A + (0.65 : ℝ) - prob_A_and_B :=
by
  sorry

end prob_B_occurs_l226_226795


namespace sum_of_symmetry_and_rotation_lines_l226_226485

theorem sum_of_symmetry_and_rotation_lines (L R : ℕ) (hL : L = 20) (hR : R = 18) : L + R = 38 := by
  rw [hL, hR]
  rfl

end sum_of_symmetry_and_rotation_lines_l226_226485


namespace length_of_QR_in_circle_l226_226600

theorem length_of_QR_in_circle 
  (A B C : Point) 
  (AB AC BC : ℝ)
  (hABC : Triangle A B C) 
  (hAB : AB = 12) 
  (hAC : AC = 9) 
  (hBC : BC = 15)
  (circleP : Circle C r)
  (hTangent : circleP.tangent A B)
  (Q R : Point) 
  (hQ : Q ∈ circleP)
  (hR : R ∈ circleP)
  (hQAC : Q ∈ LineSegment A C)
  (hRBC : R ∈ LineSegment B C)
  (hQR : Q ≠ C ∧ R ≠ C) :
  dist Q R = 6 * Real.sqrt 5 := 
sorry

end length_of_QR_in_circle_l226_226600


namespace museum_revenue_l226_226612

theorem museum_revenue (V : ℕ) (H : V = 500)
  (R : ℕ) (H_R : R = 60 * V / 100)
  (C_p : ℕ) (H_C_p : C_p = 40 * R / 100)
  (S_p : ℕ) (H_S_p : S_p = 30 * R / 100)
  (A_p : ℕ) (H_A_p : A_p = 30 * R / 100)
  (C_t S_t A_t : ℕ) (H_C_t : C_t = 4) (H_S_t : S_t = 6) (H_A_t : A_t = 12) :
  C_p * C_t + S_p * S_t + A_p * A_t = 2100 :=
by 
  sorry

end museum_revenue_l226_226612


namespace f_zero_f_decreasing_find_x_range_l226_226107

variable {R : Type*} [OrderedRing R] (f : R → R)

theorem f_zero (h : ∀ x y : R, f(x + y) = f(x) + f(y) - 3) (hx : ∀ x > 0, f x < 3) : f 0 = 3 :=
by
  sorry

theorem f_decreasing (h : ∀ x y : R, f(x + y) = f(x) + f(y) - 3) (hx : ∀ x > 0, f x < 3) : ∀ x1 x2 : R, x1 > x2 → f x1 < f x2 :=
by
  sorry

theorem find_x_range (h : ∀ x y : R, f(x + y) = f(x) + f(y) - 3) (hx : ∀ x > 0, f x < 3)
  (hineq : ∀ t : R, 2 < t → t < 4 → f((t-2)*|4 - x|) + 3 > f(t^2 + 8) + f(5 - 4 * t) ) :
  -5/2 <= x ∧ x <= 21/2 :=
by
  sorry

end f_zero_f_decreasing_find_x_range_l226_226107


namespace length_of_wall_l226_226474

-- Definitions based on the conditions given
def rate_for_2_boys := 40 / 4 -- 40 meters in 4 days for 2 boys
def rate_for_1_boy := rate_for_2_boys / 2 -- Rate for 1 boy
def rate_for_5_boys := rate_for_1_boy * 5 -- Rate for 5 boys
def time_for_5_boys := 2.6 -- 5 boys working for 2.6 days

-- Proof problem to prove the length of the wall
theorem length_of_wall : rate_for_5_boys * time_for_5_boys = 65 := by 
  sorry

end length_of_wall_l226_226474


namespace base_length_of_triangle_l226_226684

theorem base_length_of_triangle (height area : ℕ) (h1 : height = 8) (h2 : area = 24) : 
  ∃ base : ℕ, (1/2 : ℚ) * base * height = area ∧ base = 6 := by
  sorry

end base_length_of_triangle_l226_226684


namespace ordered_pair_proportional_l226_226742

theorem ordered_pair_proportional (p q : ℝ) (h : (3 : ℝ) • (-4 : ℝ) = (5 : ℝ) • p ∧ (3 : ℝ) • q = (5 : ℝ) • (-4 : ℝ)) :
  (p, q) = (5 / 2, -8) :=
by
  sorry

end ordered_pair_proportional_l226_226742


namespace club_comm_selection_count_l226_226810

theorem club_comm_selection_count :
  ∃ n : ℕ, n = nat.choose 20 3 ∧ n = 1140 :=
by {
  have h : nat.choose 20 3 = 1140 := by sorry,
  use 1140,
  exact ⟨h, rfl⟩,
}

end club_comm_selection_count_l226_226810


namespace symmetry_axes_condition_l226_226850

/-- Define the property of having axes of symmetry for a geometric figure -/
def has_symmetry_axes (bounded : Bool) (two_parallel_axes : Bool) : Prop :=
  if bounded then 
    ¬ two_parallel_axes 
  else 
    true

/-- Main theorem stating the condition on symmetry axes for bounded and unbounded geometric figures -/
theorem symmetry_axes_condition (bounded : Bool) : 
  ∃ two_parallel_axes : Bool, has_symmetry_axes bounded two_parallel_axes :=
by
  -- The proof itself is not necessary as per the problem statement
  sorry

end symmetry_axes_condition_l226_226850


namespace minimal_stages_l226_226430

theorem minimal_stages (n k : ℕ) : 
  (∃ m : ℕ, ∀ i : ℕ, (i ≤ m → (n - i * k ≤ k) ∧ (n - (i-m) * k) ) ) ↔ m = 2 ⌊n / k⌋ + 2 :=
by
  sorry

end minimal_stages_l226_226430


namespace product_of_geometric_progressions_is_geometric_general_function_form_geometric_l226_226783

variables {α β γ : Type*} [CommSemiring α] [CommSemiring β] [CommSemiring γ]

-- Define the terms of geometric progressions
def term (a r : α) (k : ℕ) : α := a * r ^ (k - 1)

-- Define a general function with respective powers
def general_term (a r : α) (k p : ℕ) : α := a ^ p * (r ^ p) ^ (k - 1)

theorem product_of_geometric_progressions_is_geometric
  {a b c : α} {r1 r2 r3 : α} (k : ℕ) :
  term a r1 k * term b r2 k * term c r3 k = 
  (a * b * c) * (r1 * r2 * r3) ^ (k - 1) := 
sorry

theorem general_function_form_geometric
  {a b c : α} {r1 r2 r3 : α} {p q r : ℕ} (k : ℕ) :
  general_term a r1 k p * general_term b r2 k q * general_term c r3 k r = 
  (a^p * b^q * c^r) * (r1^p * r2^q * r3^r) ^ (k - 1) := 
sorry

end product_of_geometric_progressions_is_geometric_general_function_form_geometric_l226_226783


namespace number_of_terms_is_10_l226_226530

-- Defining the arithmetic sequence with a common difference of 2
def arith_seq (a d : ℤ) (n : ℤ) := a + (n - 1) * d

-- Given conditions
def is_arith_seq (a : ℤ) (d : ℤ) : Prop := d = 2
def even_number_of_terms (k : ℤ) : Prop := ∃ n : ℤ, n = 2 * k
def sum_odd_terms (a k : ℤ) : Prop := (Finset.range k).sum (λ i, arith_seq a 2 (2 * i + 1)) = 15
def sum_even_terms (a k : ℤ) : Prop := (Finset.range k).sum (λ i, arith_seq a 2 (2 * (i + 1))) = 25

theorem number_of_terms_is_10 : ∃ k : ℤ, is_arith_seq a 2 ∧ even_number_of_terms k ∧ sum_odd_terms a k ∧ sum_even_terms a k → 2 * k = 10 :=
by
  sorry

end number_of_terms_is_10_l226_226530


namespace both_pumps_fill_time_l226_226028

theorem both_pumps_fill_time (old_pump_time new_pump_time : ℕ) : 
  old_pump_time = 600 → new_pump_time = 200 → 
  1 / ((1 / old_pump_time.to_rat) + (1 / new_pump_time.to_rat)) = 150 :=
by
  intros h_old h_new
  rw [h_old, h_new]
  sorry

end both_pumps_fill_time_l226_226028


namespace find_t_l226_226951

theorem find_t (t : ℝ) : 
  (∃ a b : ℝ, a^2 = t^2 ∧ b^2 = 5 * t ∧ (a - b = 2 * Real.sqrt 6 ∨ b - a = 2 * Real.sqrt 6)) → 
  (t = 2 ∨ t = 3 ∨ t = 6) := 
by
  sorry

end find_t_l226_226951


namespace seq_periodic_iff_mod3_zero_l226_226520

def sequence (a : ℕ) : ℕ × ℕ :=
if a.sqrt^2 = a then (a.sqrt, 0) else (a + 3, 0)

def seq_n (a0 : ℕ) (n : ℕ) : ℕ :=
nat.rec_on n a0 (λ n' an', (sequence an').fst)

theorem seq_periodic_iff_mod3_zero (a0 : ℕ) (h : a0 > 1) :
  (∃ A : ℕ, ∃ᶠ n in at_top, seq_n a0 n = A) ↔ a0 % 3 = 0 :=
sorry

end seq_periodic_iff_mod3_zero_l226_226520


namespace no_matrix_swaps_columns_l226_226510

theorem no_matrix_swaps_columns :
  ∀ (M : Matrix (Fin 2) (Fin 2) ℝ), ¬ (∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
  M ⬝ A = ⟨![⟨![A 0 1, A 0 0]⟩, ⟨![A 1 1, A 1 0]⟩]⟩) :=
by
  sorry

end no_matrix_swaps_columns_l226_226510


namespace harry_terry_difference_l226_226573

-- Define Harry's answer
def H : ℤ := 8 - (2 + 5)

-- Define Terry's answer
def T : ℤ := 8 - 2 + 5

-- State the theorem to prove H - T = -10
theorem harry_terry_difference : H - T = -10 := by
  sorry

end harry_terry_difference_l226_226573


namespace tangent_condition_l226_226800

theorem tangent_condition (a b : ℝ) : 
    a = b → 
    (∀ x y : ℝ, (y = x + 2 → (x - a)^2 + (y - b)^2 = 2 → y = x + 2)) :=
by
  sorry

end tangent_condition_l226_226800


namespace relationship_between_a_b_c_l226_226069

theorem relationship_between_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h₁ : a = (10 ^ 1988 + 1) / (10 ^ 1989 + 1))
  (h₂ : b = (10 ^ 1987 + 1) / (10 ^ 1988 + 1))
  (h₃ : c = (10 ^ 1987 + 9) / (10 ^ 1988 + 9)) :
  a < b ∧ b < c := 
sorry

end relationship_between_a_b_c_l226_226069


namespace vika_card_pairing_l226_226356

theorem vika_card_pairing :
  ∃ (d ∈ {1, 2, 3, 5, 6, 10, 15, 30}), ∃ (k : ℕ), 60 = 2 * d * k :=
by sorry

end vika_card_pairing_l226_226356


namespace find_area_of_quadrilateral_l226_226916

noncomputable def volume_tetrahedron (a : ℝ) : ℝ :=
  a^3 / (6 * real.sqrt 2)

noncomputable def area_quadrilateral (D E F G : ℝ × ℝ × ℝ) : ℝ :=
  -- Add appropriate formula for the area of quadrilateral here
  sorry

variables (a : ℝ) (P A B C D E F G : ℝ × ℝ × ℝ) (V : ℝ)
variable (theta : ℝ)

-- Given conditions
axiom h_volume : V = 9 * real.sqrt 3
axiom h_dihedral_angle : theta = real.pi / 3 -- 60 degrees in radians
axiom h_AD : dist A D = (1/6) * dist A B
axiom h_AE : dist A E = (1/6) * dist A C
axiom h_F_midpoint : F = midpoint ℝ P C
axiom h_G_plane_intersection : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ G = line_intersection_plane P B D E F

-- The theorem to prove
theorem find_area_of_quadrilateral : ∃ (area : ℝ), area = area_quadrilateral D E F G :=
sorry

end find_area_of_quadrilateral_l226_226916


namespace fruits_eaten_l226_226274

theorem fruits_eaten (initial_cherries initial_strawberries initial_blueberries left_cherries left_strawberries left_blueberries : ℕ)
  (h1 : initial_cherries = 16) (h2 : initial_strawberries = 10) (h3 : initial_blueberries = 20)
  (h4 : left_cherries = 6) (h5 : left_strawberries = 8) (h6 : left_blueberries = 15) :
  (initial_cherries - left_cherries) + (initial_strawberries - left_strawberries) + (initial_blueberries - left_blueberries) = 17 := 
by
  sorry

end fruits_eaten_l226_226274


namespace sqrt_four_minus_one_eq_one_l226_226847

theorem sqrt_four_minus_one_eq_one : sqrt 4 - 1 = 1 :=
by
  -- The proof would go here, but we're using sorry to indicate it's omitted.
  sorry

end sqrt_four_minus_one_eq_one_l226_226847


namespace power_sum_l226_226498

theorem power_sum : 1^234 + 4^6 / 4^4 = 17 :=
by
  sorry

end power_sum_l226_226498


namespace calculate_expression_l226_226477

theorem calculate_expression :
  3 * (1 / 2) ^ (-2 : ℤ) + |2 - Real.pi| + (-3) ^ (0 : ℤ) = 11 + Real.pi := by
  have h1: (1 / 2) ^ (-2 : ℤ) = 4 := by sorry,
  have h2: |2 - Real.pi| = Real.pi - 2 := by sorry,
  have h3: (-3) ^ (0 : ℤ) = 1 := by sorry,
  rw [h1, h2, h3],
  ring,
  sorry

end calculate_expression_l226_226477


namespace arithmetic_sequence_nine_l226_226532

variable (a : ℕ → ℝ)
variable (d : ℝ)
-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nine (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h_cond : a 4 + a 14 = 2) : 
  a 9 = 1 := 
sorry

end arithmetic_sequence_nine_l226_226532


namespace find_t_find_h_l226_226214

-- Given conditions
variables (AB BC AD : ℝ) (θ : ℝ)
-- Constants given in the problem
def AB_val : ℝ := 7
def BC_val : ℝ := 25
def periodic_100sin : ℝ := 96
def AD_val : ℝ := (168 / 25)

-- Required proof statements:
theorem find_t (AB BC : ℝ) (θ : ℝ) (h : ℝ) (hyp1 : AB = 7) (hyp2 : BC = 25) (hyp3 : 100 * real.sin θ = periodic_100sin) : 100 * (real.sin θ) = 96 := sorry

theorem find_h (AB BC h : ℝ) (θ : ℝ) (hyp1 : AB = 7) (hyp2 : BC = 25) (hyp4 : h = (168 / 25)) : h = (168 / 25) := sorry

end find_t_find_h_l226_226214


namespace boat_travel_distance_l226_226688

theorem boat_travel_distance (stream_speed boat_still_water_speed total_time : ℝ)
  (h_stream_speed : stream_speed = 4)
  (h_boat_still_water_speed : boat_still_water_speed = 8)
  (h_total_time : total_time = 2) :
  ∃ D : ℝ, D = 6 :=
by
  have downstream_speed := boat_still_water_speed + stream_speed
  have upstream_speed := boat_still_water_speed - stream_speed
  have h_downstream := downstream_speed = 12
  have h_upstream := upstream_speed = 4
  have time_downstream := D / downstream_speed
  have time_upstream := D / upstream_speed
  have h_time_eq := time_downstream + time_upstream = 2
  sorry

end boat_travel_distance_l226_226688


namespace integer_roots_of_polynomial_l226_226877

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, (x^3 - 3 * x^2 - 13 * x + 15 = 0) → (x = -3 ∨ x = 1 ∨ x = 5) :=
by
  sorry

end integer_roots_of_polynomial_l226_226877


namespace distances_not_possible_l226_226919

noncomputable def point := (ℝ × ℝ)
noncomputable def square := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distances_not_possible (P : point) (A B C D : point)
  (square_def : square := (A, B, C, D))
  (dPA : distance P A = 1)
  (dPB : distance P B = 1)
  (dPC : distance P C = 3)
  (dPD : distance P D = 2) :
  false :=
sorry

end distances_not_possible_l226_226919


namespace find_a_l226_226221

theorem find_a : (a : ℕ) = 103 * 97 * 10009 → a = 99999919 := by
  intro h
  sorry

end find_a_l226_226221


namespace suzanne_french_toast_slices_l226_226275

theorem suzanne_french_toast_slices :
  (∀ (weeks : ℕ) (days_in_week : ℕ) (weekly_days_of_french_toast : ℕ) 
     (loaves_needed_per_year : ℕ) (slices_per_loaf : ℕ) (total_slices_of_bread : ℕ)
     (split_slices : ℕ) (people : ℕ) (slices_per_day : ℕ),
    weeks = 52 →
    days_in_week = 7 →
    weekly_days_of_french_toast = 2 →
    loaves_needed_per_year = 26 →
    slices_per_loaf = 12 →
    total_slices_of_bread = loaves_needed_per_year * slices_per_loaf →
    total_slices_of_bread / (weeks * weekly_days_of_french_toast) = slices_per_day →
    slices_per_day - split_slices = people →
    people = 2 →
    Suzanne_and_husband_slice : slices_per_day - 1 = people) := 1
by
    sorry

end suzanne_french_toast_slices_l226_226275


namespace label_intersection_points_l226_226118

theorem label_intersection_points (k : ℕ) 
  (h1 : ∀ (i j : ℕ), i ≠ j → ∃ p, p ∈ l_i ∧ p ∈ l_j) -- No two lines are parallel
  (h2 : ∀ (i j m : ℕ), i ≠ j ∧ j ≠ m ∧ i ≠ m → ∃ ! p, p ∈ l_i ∧ p ∈ l_j ∧ p ∈ l_m) -- No three lines are concurrent
  : (∃ labeling : Finset ℕ → Finset (Finset ℕ), 
      (∀ i j, i ≠ j → labeling l_i ∩ labeling l_j = ∅) ∧ -- Each label appears exactly once on each line
      (∀ i, (labeling l_i).card = k-1)) ↔ 2 ∣ k := 
begin
  sorry
end

end label_intersection_points_l226_226118


namespace plane_through_A_perpendicular_to_BC_l226_226788

def point (ℝ : Type*) := ℝ × ℝ × ℝ

noncomputable def plane_equation (A B C : point ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := A
  let (x2, y2, z2) := B
  let (x3, y3, z3) := C
  let normal := ((x3 - x2), (y3 - y2), (z3 - z2))
  let (a, b, c) := normal
  let d := -(a * x1 + b * y1 + c * z1)
  (a, b, c, d)

theorem plane_through_A_perpendicular_to_BC :
  plane_equation (-1, 3, 4) (-1, 5, 0) (2, 6, 1) = (3, 1, 1, -4) :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l226_226788


namespace minimal_positive_period_f_max_value_f_min_value_f_l226_226549

noncomputable def f (x : ℝ) := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

theorem minimal_positive_period_f :
  ∃ T > 0, ∀ x, f (x + T) = f x := 
begin
  use π,
  split,
  { exact real.pi_pos },
  { intro x,
    -- The given function is periodic with period π
    sorry }
end

theorem max_value_f :
  ∃ x ∈ Icc (0 : ℝ) (π / 2), f x = 1 + Real.sqrt 2 :=
begin
  use π / 8,
  split,
  { split,
    { linarith [real.pi_pos], },
    { linarith [real.pi_div_two_pos], } },
  { -- Simplify f at x = π / 8 to show f x = 1 + √2
    sorry }
end

theorem min_value_f :
  ∃ x ∈ Icc (0 : ℝ) (π / 2), f x = 0 :=
begin
  use π / 2,
  split,
  { split,
    { linarith [real.pi_pos], },
    { linarith [real.pi_div_two_pos], } },
  { -- Simplify f at x = π / 2 to show f x = 0
    sorry }
end

end minimal_positive_period_f_max_value_f_min_value_f_l226_226549


namespace number_of_ways_to_form_11_digit_number_l226_226216

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

def valid_digits : Finset ℕ := {0, 2, 4, 7, 8, 9}

def ways_divisible_by_12 (n : ℕ) : ℕ :=
  if is_divisible_by_12 n then 1 else 0

theorem number_of_ways_to_form_11_digit_number : 
  (Finset.filter (λ n, is_divisible_by_12 (n * 12345602))
    (Finset.finRange (6^5))).card = 1296 := 
sorry

end number_of_ways_to_form_11_digit_number_l226_226216


namespace buffy_breath_time_l226_226232

theorem buffy_breath_time (k : ℕ) (b : ℕ) (f : ℕ) 
  (h1 : k = 3 * 60) 
  (h2 : b = k - 20) 
  (h3 : f = b - 40) :
  f = 120 :=
by {
  sorry
}

end buffy_breath_time_l226_226232


namespace average_birth_rate_is_10_l226_226606

-- The given conditions
def average_birth_rate (B : ℕ) : Prop :=
  let death_rate := 2 in
  let net_increase_per_day := 345600 in
  let seconds_per_day := 24 * 60 * 60 in
  let changes_per_day := seconds_per_day / 2 in
  let net_increase_per_change := net_increase_per_day / changes_per_day in
  B - death_rate = net_increase_per_change

theorem average_birth_rate_is_10 (B : ℕ) (h : average_birth_rate B) : B = 10 :=
by
  -- Proof goes here
  sorry

end average_birth_rate_is_10_l226_226606


namespace selection_ways_1200_l226_226985

/--
In an event, there are 30 people arranged in 6 rows and 5 columns. Now,
3 people are to be selected to perform a ceremony, with the requirement
that any two of these 3 people must not be in the same row or column.
The number of different ways to select these 3 people is 1200.
-/
theorem selection_ways_1200 :
  let rows := 6
  let columns := 5
  let select_3 := 3
  (Nat.choose rows select_3) * (Nat.choose columns select_3) * (Nat.factorial select_3) = 1200 :=
by
  let rows := 6
  let columns := 5
  let select_3 := 3
  have h1 : Nat.choose rows select_3 = 20 := by rw Nat.choose_eq_factorial_div_factorial
  have h2 : Nat.choose columns select_3 = 10 := by rw Nat.choose_eq_factorial_div_factorial
  have h3 : Nat.factorial select_3 = 6 := by rw Nat.factorial
  calc
  (Nat.choose rows select_3) * (Nat.choose columns select_3) * (Nat.factorial select_3)
      = 20 * 10 * 6 : by rw [h1, h2, h3]
  ... = 1200 : by norm_num

end selection_ways_1200_l226_226985


namespace card_pairing_modulus_l226_226362

theorem card_pairing_modulus (cards : Finset ℕ) (h : cards = (Finset.range 60).image (λ n, n + 1)) :
  ∃ n, n = 8 ∧ ∀ (pairs : Finset (ℕ × ℕ)), (∀ (p ∈ pairs), (p.1 ∈ cards ∧ p.2 ∈ cards ∧ (|p.1 - p.2| = d))) → pairs.card = 30 :=
sorry

end card_pairing_modulus_l226_226362


namespace range_of_m_l226_226562

-- Define the quadratic function
def f (a c x : ℝ) := a * x^2 - 2 * a * x + c

-- Define the condition that the function is monotonically decreasing on [0,1]
def is_monotonically_decreasing_in_interval (a : ℝ) : Prop :=
  ∀ x ∈ set.Icc (0 : ℝ) (1 : ℝ), (2 * a * (x - 1) < 0)

-- Define the interval condition for m: [0, 2]
def interval_condition (a c m : ℝ) : Prop := f a c m ≤ f a c 0

-- Final theorem combining the conditions and proving the range of m
theorem range_of_m (a c m : ℝ) (h : is_monotonically_decreasing_in_interval a) :
  interval_condition a c m → 0 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l226_226562


namespace card_pairing_modulus_l226_226358

theorem card_pairing_modulus (cards : Finset ℕ) (h : cards = (Finset.range 60).image (λ n, n + 1)) :
  ∃ n, n = 8 ∧ ∀ (pairs : Finset (ℕ × ℕ)), (∀ (p ∈ pairs), (p.1 ∈ cards ∧ p.2 ∈ cards ∧ (|p.1 - p.2| = d))) → pairs.card = 30 :=
sorry

end card_pairing_modulus_l226_226358


namespace equation_of_ellipse_exists_line_l_l226_226531

variable (C : Set (ℝ × ℝ))
variable (M P A B: ℝ × ℝ)
variable (e : ℝ)
variable (a b : ℝ)

-- Given conditions
def is_center_origin (C : Set (ℝ × ℝ)) : Prop := 
  ∀ p ∈ C, (0, 0) is the center of C

def has_foci_on_x_axis (C : Set (ℝ × ℝ)) : Prop := 
  ∃ F1 F2 : ℝ × ℝ, F1.2 = 0 ∧ F2.2 = 0 ∧ (∀ p ∈ C, (dist p F1 + dist p F2 = 2 * a))

def is_point_on_ellipse (p : ℝ × ℝ) (C : Set (ℝ × ℝ)) : Prop := 
  p ∈ C

def eccentricity (C : Set (ℝ × ℝ)) : ℝ :=
  e

def passes_through_M (C : Set (ℝ × ℝ)) : Prop :=
  (1, 3/2) ∈ C

def is_line_passing_through (l : ℝ × ℝ → Prop) :=
  ∀ t : ℝ, l (2 + t, 1 + t)

-- To prove:
theorem equation_of_ellipse : 
  is_center_origin C ∧ has_foci_on_x_axis C ∧ eccentricity C = 1/2 ∧ passes_through_M C → 
  (C = {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / 3) = 1}) := sorry

theorem exists_line_l : 
  ∃ l : ℝ × ℝ → Prop, (is_line_passing_through l) ∧ 
  (∀ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C → 
  ((2 - A.1) * (2 - B.1) + (1 - A.2) * (1 - B.2) = (2 - 1)^2 + (1 - 3/2)^2)) :=
  sorry

end equation_of_ellipse_exists_line_l_l226_226531


namespace elements_of_order_p_l226_226297

open FiniteGroup

theorem elements_of_order_p (G : Type _) [Group G] [Fintype G] 
  (p : ℕ) [Prime p] (n : ℕ) 
  (h_n : ∀ g : G,  orderOf g = p ↔ g = 1 ∨ g ∈ { x | orderOf x = p }) 
  (h_card : Fintype.card (subtype {x : G | orderOf x = p}) = n) :
  n = 0 ∨ p ∣ (n + 1) :=
sorry

end elements_of_order_p_l226_226297


namespace problem_statement_l226_226955

variable (a b : ℝ) (f : ℝ → ℝ)
variable (h1 : ∀ x > 0, f x = Real.log x / Real.log 3)
variable (h2 : b = 9 * a)

theorem problem_statement : f a - f b = -2 := by
  sorry

end problem_statement_l226_226955


namespace width_of_rectangular_plot_l226_226446

theorem width_of_rectangular_plot 
  (length : ℕ) (distance_between_poles : ℕ) (number_of_poles : ℕ)
  (h_length : length = 90)
  (h_distance_between_poles : distance_between_poles = 20)
  (h_number_of_poles : number_of_poles = 14) :
  let number_of_spacings := number_of_poles - 1 in
  let total_length_of_fencing := number_of_spacings * distance_between_poles in
  let width := (total_length_of_fencing / 2) - length
  width = 40 :=
by
  sorry

end width_of_rectangular_plot_l226_226446


namespace smaller_octagon_half_area_l226_226701

-- Define what it means to be a regular octagon
def is_regular_octagon (O : Point) (ABCDEFGH : List Point) : Prop :=
  -- Definition capturing the properties of a regular octagon around center O
  sorry

-- Define the function that computes the area of an octagon
def area_of_octagon (ABCDEFGH : List Point) : Real :=
  sorry

-- Define the function to create the smaller octagon by joining midpoints
def smaller_octagon (ABCDEFGH : List Point) : List Point :=
  sorry

theorem smaller_octagon_half_area (O : Point) (ABCDEFGH : List Point) :
  is_regular_octagon O ABCDEFGH →
  area_of_octagon (smaller_octagon ABCDEFGH) = (1 / 2) * area_of_octagon ABCDEFGH :=
by
  sorry

end smaller_octagon_half_area_l226_226701


namespace smaller_octagon_area_fraction_l226_226726

theorem smaller_octagon_area_fraction (A B C D E F G H : Point)
  (midpoints_joined : Boolean)
  (regular_octagon : RegularOctagon A B C D E F G H)
  (smaller_octagon : Octagon (midpoint (A, B)) (midpoint (B, C)) (midpoint (C, D)) 
                              (midpoint (D, E)) (midpoint (E, F)) (midpoint (F, G))
                              (midpoint (G, H)) (midpoint (H, A))) :
  midpoints_joined → regular_octagon → 
  (area smaller_octagon) = (3 / 4) * (area regular_octagon) :=
by
  sorry

end smaller_octagon_area_fraction_l226_226726


namespace calculation1_calculation2_l226_226050

-- Problem 1
theorem calculation1 :
  sqrt 27 * sqrt (2 / 3) - sqrt 40 / sqrt 5 + abs (2 - sqrt 2) = 2 :=
by 
  sorry

-- Problem 2
theorem calculation2 :
  (π - 1)^0 + (-1 / 3)^(-2) - sqrt ((-3)^2) + 1 / (sqrt 2 - 1) = 8 + sqrt 2 :=
by 
  sorry

end calculation1_calculation2_l226_226050


namespace polynomial_equality_l226_226952

theorem polynomial_equality (x y : ℝ) (h₁ : 3 * x + 2 * y = 6) (h₂ : 2 * x + 3 * y = 7) : 
  14 * x^2 + 25 * x * y + 14 * y^2 = 85 := 
by
  sorry

end polynomial_equality_l226_226952


namespace angle_G_in_heptagon_l226_226609

theorem angle_G_in_heptagon :
  let x := 133.33 in
  (∑ θ in [x, x, x, 160, 160, 180-x, 180-x], θ) = 900 :=
by
  sorry

end angle_G_in_heptagon_l226_226609


namespace smallest_x_l226_226397

open Classical
noncomputable theory

def conditions (x : ℕ) : Prop :=
  x % 3 = 2 ∧ x % 4 = 3 ∧ x % 5 = 4

theorem smallest_x : ∃ (x : ℕ), conditions x ∧ (∀ (y : ℕ), conditions y → x ≤ y) ∧ x = 59 :=
by {
  sorry
}

end smallest_x_l226_226397


namespace first_train_speed_eq_l226_226337

noncomputable def speed_of_first_train {length_of_each_train speed_second_train time_to_pass_each_other} : ℝ :=
  let distance := 0.38 -- 380 meters converted to kilometers
  let time_in_hours := 0.003304903381642512 -- 11.895652173913044 seconds converted to hours
  ((distance / time_in_hours) - speed_second_train)

theorem first_train_speed_eq
  (length_of_each_train : ℝ)
  (speed_second_train : ℝ)
  (time_to_pass_each_other : ℝ)
  (h_length : length_of_each_train = 0.19) -- 190 meters converted to kilometers
  (h_speed : speed_second_train = 50) 
  (h_time : time_to_pass_each_other = 11.895652173913044) : 
  speed_of_first_train length_of_each_train speed_second_train time_to_pass_each_other = 64.94252873563218 := sorry

end first_train_speed_eq_l226_226337


namespace comparison_stability_l226_226001

def data_A := [102, 101, 99, 98, 103, 98, 99]
def data_B := [110, 115, 90, 85, 75, 115, 110]

noncomputable def range (data : List ℝ) : ℝ := List.maximum data - List.minimum data

theorem comparison_stability :
  range data_A < range data_B :=
by
  sorry

end comparison_stability_l226_226001


namespace shaded_fraction_is_correct_l226_226310

-- Definitions of conditions
def large_square_area := 9  -- The large square quilt is composed of nine smaller squares
def small_square_area := 1  -- Each smaller square has an area of 1 unit square
def triangle_area := 0.5    -- Each triangle has an area of 0.5 unit square

-- Total shaded area: one whole square and three triangles
def total_shaded_area := small_square_area + 3 * triangle_area

-- The fraction of the square quilt that is shaded
def shaded_fraction := total_shaded_area / large_square_area

-- Now, we provide the theorem that the shaded fraction equals 5/18
theorem shaded_fraction_is_correct : shaded_fraction = 5 / 18 := 
by 
  -- Proof is omitted here
  sorry

end shaded_fraction_is_correct_l226_226310


namespace Events_A_and_B_are_mutually_exclusive_l226_226784

def fair_die := {1, 2, 3, 4, 5, 6}

def Event_A (a : ℕ) := a = 1
def Event_B (a : ℕ) := a = 2
def Event_C (a : ℕ) := a = 2 ∨ a = 4 ∨ a = 6

theorem Events_A_and_B_are_mutually_exclusive :
  ∀ a b ∈ fair_die, Event_A a ∧ Event_B b → a ≠ b :=
by
  sorry

end Events_A_and_B_are_mutually_exclusive_l226_226784


namespace solve_system_of_inequalities_l226_226675

theorem solve_system_of_inequalities (x : ℝ) (h1 : x + 2 > 3) (h2 : 2x - 1 < 5) : 1 < x ∧ x < 3 :=
by
  sorry

end solve_system_of_inequalities_l226_226675


namespace doughnuts_per_box_l226_226433

theorem doughnuts_per_box
  (total_doughnuts : ℕ)
  (boxes_sold : ℕ)
  (doughnuts_given_away : ℕ)
  (doughnuts_per_box : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : boxes_sold = 27)
  (h3 : doughnuts_given_away = 30) :
  doughnuts_per_box = (total_doughnuts - doughnuts_given_away) / boxes_sold := by
  -- proof goes here
  sorry

end doughnuts_per_box_l226_226433


namespace piece_exits_at_A2_l226_226601

-- Define initial state setup
def grid_initial_state : (ℕ × ℕ) := (3, 2) -- C2 in row 3 and column 2
constant arrow_initial_direction : (ℕ × ℕ) -> string
axiom initial_arrow : arrow_initial_direction (3, 2) = "right"

-- Function to invert an arrow direction
def invert_arrow (dir : string) : string :=
  match dir with
  | "right" => "left"
  | "left" => "right"
  | "up" => "down"
  | "down" => "up"
  | _ => dir
  end

-- Function to move to next cell based on direction
def move_to_next (position : (ℕ × ℕ)) (dir : string) : (ℕ × ℕ) :=
  match dir with
  | "right" => (position.1, position.2 + 1)
  | "left" => (position.1, position.2 - 1)
  | "up" => (position.1 - 1, position.2)
  | "down" => (position.1 + 1, position.2)
  | _ => position
  end

-- Define the final position we want to prove it reaches
def final_position : (ℕ × ℕ) := (1, 2) -- A2 in row 1, column 2

-- Main theorem statement
theorem piece_exits_at_A2 :
  ∃ p : (ℕ × ℕ), p = final_position :=
by
  -- Definitions of initial state, arrow direction changes and moves
  let initial_position := grid_initial_state
  -- Conduct moves based on arrow directions here
  -- ...
  sorry

end piece_exits_at_A2_l226_226601


namespace eccentricity_range_l226_226136

noncomputable def ellipse_condition 
  (a b c e : ℝ) (h1 : a > b > 0) (h2 : c^2 = a^2 - b^2) : Prop :=
  ∃ (P : ℝ × ℝ), 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ 
    (∃ θ : ℝ, a / Real.sin θ = c / Real.sin (θ - π))

theorem eccentricity_range
  (a b c e : ℝ) (h1 : a > b > 0) (h2 : c^2 = a^2 - b^2)
  (he : 0 < e ∧ e < 1)
  (hc : ellipse_condition a b c e h1 h2) : e ∈ Ioo (sqrt 2 - 1) 1 :=
sorry

end eccentricity_range_l226_226136


namespace odd_function_a_eq_1_l226_226189

-- Definitions of the conditions in the problem
def f (a x : ℝ) : ℝ := Real.log (x + Real.sqrt (a * x^2 + 1))

-- Statement of the problem in Lean 4
theorem odd_function_a_eq_1 (a : ℝ) :
  (∀ x : ℝ, f a x + f a (-x) = 0) → a = 1 :=
by
  sorry

end odd_function_a_eq_1_l226_226189


namespace goods_train_speed_l226_226812

theorem goods_train_speed
  (length_train : ℕ)
  (length_platform : ℕ)
  (time_seconds : ℕ)
  (h1 : length_train = 190)
  (h2 : length_platform = 250)
  (h3 : time_seconds = 22) :
  (length_train + length_platform) / time_seconds * 3.6 = 72 := by
  sorry

end goods_train_speed_l226_226812


namespace sqrt_domain_l226_226586

theorem sqrt_domain (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (x - 2)) ↔ (x ≥ 2) :=
by
  -- proof goes here
  sorry

end sqrt_domain_l226_226586


namespace smallest_positive_angle_l226_226127

theorem smallest_positive_angle (α : ℝ) (h : (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3)) = (Real.sin α, Real.cos α)) : 
  α = 11 * Real.pi / 6 := by
sorry

end smallest_positive_angle_l226_226127


namespace island_width_l226_226834

theorem island_width (area length width : ℕ) (h₁ : area = 50) (h₂ : length = 10) : width = area / length := by 
  sorry

end island_width_l226_226834


namespace problem_l226_226631

def vec3 : Type := ℝ × ℝ × ℝ

def a : vec3 := (-1, 6, 3)
def b (e : ℝ) : vec3 := (3, e, 2)
def c : vec3 := (-4, -1, 5)

def sub_vec (u v : vec3) : vec3 := (u.1 - v.1, u.2 - v.2, u.3 - v.3)
def cross (u v : vec3) : vec3 :=
((u.2 * v.3 - u.3 * v.2), (u.3 * v.1 - u.1 * v.3), (u.1 * v.2 - u.2 * v.1))
def dot (u v : vec3) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem problem (e : ℝ) : 
  dot (sub_vec a (b e)) (cross (sub_vec (b e) c) (sub_vec c a)) = 8 * e - 56 :=
by
  sorry

end problem_l226_226631


namespace find_r_l226_226648

noncomputable def P (x : ℝ) : ℝ := sorry  -- Definition of polynomial P(x) is skipped

theorem find_r (r : ℝ) (Q : ℝ → ℝ) :
  (P(r) = 2) ∧
  (∃ Q : ℝ → ℝ, ∀ x : ℝ, P(x) = (2 * x^2 + 7 * x - 4) * (x - r) * Q(x) + (-2 * x^2 - 3 * x + 4)) →
  (r = -2 ∨ r = 1 / 2) :=
by {
  -- Proof is skipped
  sorry
}

end find_r_l226_226648


namespace correct_judgment_is_B_l226_226410

/-- Definitions needed for the problem -/
def stem_and_leaf_plot_condition : Prop :=
  ∀ (data : List ℕ), ∀ (x : ℕ), x ∈ data → (∃ (count : ℕ), count > 1 ∧ List.count data x = count)

def systematic_sampling_condition : Prop :=
  ∀ (population : List ℕ), population ≠ [] → ∃ (sample : List ℕ), sample ≠ [] ∧ simple_random_sample population sample

def sum_event_condition : Prop :=
  ∀ (A B : Prop), (A ∨ B) = (A ∨ B)

def stratified_sampling_condition : Prop :=
  ∀ (population : List (ℕ × ℕ)), population ≠ [] →
  (∀ (ind : ℕ × ℕ), ind ∈ population → (finite population ∧ (∃ (sample : List (ℕ × ℕ)), sample ≠ [] ∧ equal_probability_sample population sample)))

/-- The main proposition to be proved -/
theorem correct_judgment_is_B :
  (systematic_sampling_condition) :=
begin
  sorry
end

end correct_judgment_is_B_l226_226410


namespace angle_Q_l226_226291

-- Definitions for the conditions given in the problem
def is_regular_hexagon (A B C D E F : Type) : Prop :=
  ∀ (angle : ℕ), angle ∈ { ∠A B C, ∠B C D, ∠C D E, ∠D E F, ∠E F A, ∠F A B } → angle = 120

def extended_to_meet_at (A B C D Q : Type) : Prop :=
  ∃ P : Type, line A B = line P Q ∧ line C D = line P Q

-- The theorem we need to prove
theorem angle_Q (A B C D E F Q : Type) (H_hex : is_regular_hexagon A B C D E F)
  (H_meet : extended_to_meet_at A B C D Q) : ∠A Q C = 120 :=
by sorry

end angle_Q_l226_226291


namespace sum_of_perfect_square_divisors_eq_21_l226_226890

theorem sum_of_perfect_square_divisors_eq_21 (n : ℕ) : 
  (∑ d in (finset.filter (λ x : ℕ, ∃ k : ℕ, k^2 = d) (finset.divisors n)), d) = 21 ↔ n = 64 := 
by
  sorry

end sum_of_perfect_square_divisors_eq_21_l226_226890


namespace number_of_subsets_of_set_3_l226_226315

def is_subset_3 (x : ℤ) : Prop := 0 < x ∧ x < 3

def set_3 : set ℤ := { x | is_subset_3 x }

def cardinality_of_set (s : set ℤ) : ℕ := set.card s

theorem number_of_subsets_of_set_3 : (2 ^ (cardinality_of_set set_3)) = 4 :=
by
  sorry

end number_of_subsets_of_set_3_l226_226315


namespace average_sales_is_correct_l226_226677

variable (January_sales February_sales March_sales April_sales May_sales : ℕ)
variable (total_sales number_of_months average_sales_per_month : ℕ)

-- Given conditions
def January_sales : ℕ := 110
def February_sales : ℕ := 90
def March_sales : ℕ := 70
def April_sales : ℕ := 130
def May_sales : ℕ := 50

-- Theorem statement
theorem average_sales_is_correct :
  January_sales + February_sales + March_sales + April_sales + May_sales = 450 ∧
  450 / 5 = 90 :=
by
  sorry

end average_sales_is_correct_l226_226677


namespace business_proof_l226_226420

section Business_Problem

variables (investment cost_initial rubles production_capacity : ℕ)
variables (produced_July incomplete_July bottles_August bottles_September days_September : ℕ)
variables (total_depreciation residual_value sales_amount profit_target : ℕ)

def depreciation_per_bottle (cost_initial production_capacity : ℕ) : ℕ := 
    cost_initial / production_capacity

def calculate_total_depreciation (depreciation_per_bottle produced_July bottles_August bottles_September : ℕ) : ℕ :=
    (produced_July * depreciation_per_bottle) + (bottles_August * depreciation_per_bottle) + (bottles_September * depreciation_per_bottle)

def calculate_residual_value (cost_initial total_depreciation : ℕ) : ℕ :=
    cost_initial - total_depreciation

def calculate_sales_amount (residual_value profit_target : ℕ) : ℕ :=
    residual_value + profit_target

theorem business_proof
    (H1: investment = 1500000) 
    (H2: cost_initial = 500000)
    (H3: production_capacity = 100000)
    (H4: produced_July = 200)
    (H5: incomplete_July = 5)
    (H6: bottles_August = 15000)
    (H7: bottles_September = 12300)
    (H8: days_September = 20)
    (H9: total_depreciation = 137500)
    (H10: residual_value = 362500)
    (H11: profit_target = 10000)
    (H12: sales_amount = 372500): 

    total_depreciation = calculate_total_depreciation (depreciation_per_bottle cost_initial production_capacity) produced_July bottles_August bottles_September ∧
    residual_value = calculate_residual_value cost_initial total_depreciation ∧
    sales_amount = calculate_sales_amount residual_value profit_target := 
by 
  sorry

end Business_Problem

end business_proof_l226_226420


namespace sum_of_four_digit_numbers_l226_226891

theorem sum_of_four_digit_numbers :
  let digits := [2, 4, 5, 3]
  let factorial := Nat.factorial (List.length digits)
  let each_appearance := factorial / (List.length digits)
  (each_appearance * (2 + 4 + 5 + 3) * (1000 + 100 + 10 + 1)) = 93324 :=
by
  let digits := [2, 4, 5, 3]
  let factorial := Nat.factorial (List.length digits)
  let each_appearance := factorial / (List.length digits)
  show (each_appearance * (2 + 4 + 5 + 3) * (1000 + 100 + 10 + 1)) = 93324
  sorry

end sum_of_four_digit_numbers_l226_226891


namespace smallest_x_satisfying_abs_eq_eight_l226_226889

theorem smallest_x_satisfying_abs_eq_eight : ∃ x, |x - 3| = 8 ∧ ∀ y, |y - 3| = 8 → x ≤ y :=
begin
  use -5,
  split,
  { rw abs_eq_iff,
    left,
    norm_num, },
  { intros y hy,
    rw abs_eq_iff at hy,
    cases hy,
    { linarith, },
    { exfalso,
      linarith, } }
end

end smallest_x_satisfying_abs_eq_eight_l226_226889


namespace max_value_x2y_l226_226185

theorem max_value_x2y : 
  ∃ (x y : ℕ), 
    7 * x + 4 * y = 140 ∧
    (∀ (x' y' : ℕ),
       7 * x' + 4 * y' = 140 → 
       x' ^ 2 * y' ≤ x ^ 2 * y) ∧
    x ^ 2 * y = 2016 :=
by {
  sorry
}

end max_value_x2y_l226_226185


namespace vika_card_pairs_l226_226372

theorem vika_card_pairs : 
  let numbers := finset.range 61 \ finset.singleton 0 in
  let divs := {d | d ∈ finset.divisors 30} in
  numbers.card = 60 →
  ∀ d ∈ divs, ∀ pair : finset (ℕ × ℕ),
    pair.card = 30 →
    finset.forall₂ pair (λ x y, |x.1 - x.2| % d = |y.1 - y.2| % d) → 
    ∃ (number_of_pairs : ℕ), number_of_pairs = 8 :=
by 
  intro numbers divs hc hd hp hpairs,
  sorry

end vika_card_pairs_l226_226372


namespace part1_part2_part3_l226_226113

noncomputable def sequence_an (t : ℝ) : ℕ → ℝ
| 0       := t
| (n + 1) := t * (sequence_an t n)

def sum_Sn (t : ℝ) (n : ℕ) : ℝ :=
(nat.rec_on n 0 (λ n S_n, S_n + sequence_an t (n + 1)))

def sequence_bn (t : ℝ) (n : ℕ) : ℝ :=
(sequence_an t n)^2 + (sum_Sn t n) * (sequence_an t n)

def geometric_sequence (seq : ℕ → ℝ) :=
∀ n, seq (n + 1) / seq n = seq 1 / seq 0

theorem part1 (t : ℝ) (h1 : t ≠ 0) (h2 : t ≠ 1) :
  (sequence_an t) = λ n, t^n :=
sorry

theorem part2 (h : geometric_sequence (sequence_bn 1/2)) :
  1/2 = 1/2 :=
sorry

theorem part3 (k : ℝ) (n : ℕ) (T_n : ℕ → ℝ)
  (h_Tn : T_n n = 4 + n - 4 / 2^n) (h_ineq : 12 * k / (4 + n - T_n n) ≥ 2 * n - 7) :
  k ≥ 1/32 :=
sorry

end part1_part2_part3_l226_226113


namespace remainder_sum_of_squares_mod_13_l226_226383

-- Define the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

-- Prove that the remainder when the sum of squares of the first 20 natural numbers
-- is divided by 13 is 10
theorem remainder_sum_of_squares_mod_13 : sum_of_squares 20 % 13 = 10 := 
by
  -- Here you can imagine the relevant steps or intermediate computations might go, if needed.
  sorry -- Placeholder for the proof.

end remainder_sum_of_squares_mod_13_l226_226383


namespace lemma_perpendicular_ap_oh_l226_226235

variables (A B C D E H O M N P : Point) -- Consider appropriate type for Point
variable  (triangle_abc : Triangle A B C)
variable  (scalene_triangle : scalene triangle_abc)
variable  (altitude_bd : Altitude B D A C)
variable  (altitude_ce : Altitude C E A B)
variable  (orthocenterH : orthocenter triangle_abc H)
variable  (circumcenterO : circumcenter triangle_abc O)
variable  (midpointM : midpoint M A B)
variable  (midpointN : midpoint N A C)
variable  (intersection_pointP : intersection_point P (line_through M N) (line_through D E))

theorem lemma_perpendicular_ap_oh : perpendicular (line_through A P) (line_through O H) :=
sorry

end lemma_perpendicular_ap_oh_l226_226235


namespace angle_Q_of_extended_sides_l226_226287

-- Definitions from Conditions
structure RegularHexagon (A B C D E F : Type) :=
(angles : ∀ (i : Fin 6), ℝ)
(is_regular : ∀ (i : Fin 6), angles i = 120)

section
variables {A B C D E F Q : Type}

-- The specific theorem/proof statement
theorem angle_Q_of_extended_sides (h₁ : RegularHexagon A B C D E F)
    (h₂ : extended_eq (B : A → Q) .extended_eq (extended_eq (C : D → Q))) :
    ∠ Q = 60 :=
  sorry
end

end angle_Q_of_extended_sides_l226_226287


namespace trees_per_square_meter_l226_226682

-- Definitions of the given conditions
def side_length : ℕ := 100
def total_trees : ℕ := 120000

def area_of_street : ℤ := side_length * side_length
def area_of_forest : ℤ := 3 * area_of_street

-- The question translated to Lean theorem statement
theorem trees_per_square_meter (h1: area_of_street = side_length * side_length)
    (h2: area_of_forest = 3 * area_of_street) 
    (h3: total_trees = 120000) : 
    (total_trees / area_of_forest) = 4 :=
sorry

end trees_per_square_meter_l226_226682


namespace negation_P1_is_false_negation_P2_is_false_l226_226426

-- Define the propositions
def isMultiDigitNumber (n : ℕ) : Prop := n >= 10
def lastDigitIsZero (n : ℕ) : Prop := n % 10 = 0
def isMultipleOfFive (n : ℕ) : Prop := n % 5 = 0
def isEven (n : ℕ) : Prop := n % 2 = 0

-- The propositions
def P1 (n : ℕ) : Prop := isMultiDigitNumber n → (lastDigitIsZero n → isMultipleOfFive n)
def P2 : Prop := ∀ n, isEven n → n % 2 = 0

-- The negations
def notP1 (n : ℕ) : Prop := isMultiDigitNumber n ∧ lastDigitIsZero n → ¬isMultipleOfFive n
def notP2 : Prop := ∃ n, isEven n ∧ ¬(n % 2 = 0)

-- The proof problems
theorem negation_P1_is_false (n : ℕ) : notP1 n → False := by
  sorry

theorem negation_P2_is_false : notP2 → False := by
  sorry

end negation_P1_is_false_negation_P2_is_false_l226_226426


namespace profit_percent_is_26_l226_226415

variables (P C : ℝ)
variables (h1 : (2/3) * P = 0.84 * C)

theorem profit_percent_is_26 :
  ((P - C) / C) * 100 = 26 :=
by
  sorry

end profit_percent_is_26_l226_226415


namespace find_a_n_l226_226114

noncomputable def is_arithmetic_seq (a b : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n = a + n * b

noncomputable def is_geometric_seq (b a : ℕ) (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n = b * a ^ n

theorem find_a_n (a b : ℕ) 
  (a_positive : a > 1)
  (b_positive : b > 1)
  (a_seq : ℕ → ℕ)
  (b_seq : ℕ → ℕ)
  (arith_seq : is_arithmetic_seq a b a_seq)
  (geom_seq : is_geometric_seq b a b_seq)
  (init_condition : a_seq 0 < b_seq 0)
  (next_condition : b_seq 1 < a_seq 2)
  (relation_condition : ∀ n, ∃ m, a_seq m + 3 = b_seq n) :
  ∀ n, a_seq n = 5 * n - 3 :=
sorry

end find_a_n_l226_226114


namespace min_value_x_plus_4y_l226_226523

theorem min_value_x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h_cond : (1 / x) + (1 / (2 * y)) = 1) : x + 4 * y = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_x_plus_4y_l226_226523


namespace factor_diff_of_squares_l226_226079

theorem factor_diff_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4 * y) * (5 + 4 * y) := 
sorry

end factor_diff_of_squares_l226_226079


namespace sufficient_but_not_necessary_sufficient_but_not_necessary_rel_l226_226934

theorem sufficient_but_not_necessary (a b : ℝ) (h : 0 < a ∧ a < b) : (1 / a) > (1 / b) :=
by
  sorry

theorem sufficient_but_not_necessary_rel (a b : ℝ) : 0 < a ∧ a < b ↔ (1 / a) > (1 / b) :=
by
  sorry

end sufficient_but_not_necessary_sufficient_but_not_necessary_rel_l226_226934


namespace sum_of_coefficients_l226_226994

noncomputable def coeff_sum (x y z : ℝ) : ℝ :=
  let p := (x + 2*y - z)^8  
  -- extract and sum coefficients where exponent of x is 2 and exponent of y is not 1
  sorry

theorem sum_of_coefficients (x y z : ℝ) :
  coeff_sum x y z = 364 := by
  sorry

end sum_of_coefficients_l226_226994


namespace matchsticks_in_20th_stage_l226_226755

-- Define the first term and common difference
def first_term : ℕ := 4
def common_difference : ℕ := 3

-- Define the mathematical function for the n-th term of the arithmetic sequence
def num_matchsticks (n : ℕ) : ℕ :=
  first_term + (n - 1) * common_difference

-- State the theorem to prove the number of matchsticks in the 20th stage
theorem matchsticks_in_20th_stage : num_matchsticks 20 = 61 :=
by
  -- Proof skipped
  sorry

end matchsticks_in_20th_stage_l226_226755


namespace number_of_blobs_of_glue_is_96_l226_226482

def pyramid_blobs_of_glue : Nat :=
  let layer1 := 4 * (4 - 1) * 2
  let layer2 := 3 * (3 - 1) * 2
  let layer3 := 2 * (2 - 1) * 2
  let between1_and_2 := 3 * 3 * 4
  let between2_and_3 := 2 * 2 * 4
  let between3_and_4 := 4
  layer1 + layer2 + layer3 + between1_and_2 + between2_and_3 + between3_and_4

theorem number_of_blobs_of_glue_is_96 :
  pyramid_blobs_of_glue = 96 :=
by
  sorry

end number_of_blobs_of_glue_is_96_l226_226482


namespace range_of_a_l226_226928

open Set

-- Define proposition p
def p (x : ℝ) : Prop := x^2 + 2 * x - 3 > 0

-- Define proposition q
def q (x a : ℝ) : Prop := (x - a) / (x - a - 1) > 0

-- Define negation of p
def not_p (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1

-- Define negation of q
def not_q (x a : ℝ) : Prop := a ≤ x ∧ x ≤ a + 1

-- Main theorem to prove the range of a
theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ x ∧ x ≤ a + 1 → -3 ≤ x ∧ x ≤ 1) → a ∈ Icc (-3 : ℝ) (0 : ℝ) :=
by
  intro h
  -- skipped detailed proof
  sorry

end range_of_a_l226_226928


namespace geometric_sequence_properties_l226_226616

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_properties :
  ∀ (a : ℕ → ℝ),
    geometric_sequence a q →
    a 2 = 6 →
    a 5 - 2 * a 4 - a 3 + 12 = 0 →
    ∀ n, a n = 6 ∨ a n = 6 * (-1)^(n-2) ∨ a n = 6 * 2^(n-2) :=
by
  sorry

end geometric_sequence_properties_l226_226616


namespace count_valid_six_digit_numbers_l226_226459

theorem count_valid_six_digit_numbers : 
  ∃ (n : ℕ), n = 6 ∧ 
    (∀ (digit : ℤ), digit ∈ {0, 1, 2, 3, 4, 5}) ∧ 
    (∀ (d1 d2 d3 d4 d5 d6 : ℤ), 
      {d1, d2, d3, d4, d5, d6} = {0, 1, 2, 3, 4, 5} ∧ 
      d6 < d5) → 
  n = 300 :=
by {
    sorry
}

end count_valid_six_digit_numbers_l226_226459


namespace num_satisfying_b_l226_226295

def is_factor (a b : ℕ) : Prop := ∃ k, b = a * k

def positive_divisors (n : ℕ) : list ℕ := (list.range (n+1)).filter (λ d, d > 0 ∧ is_factor d n)

theorem num_satisfying_b (b : ℕ) (hb1 : is_factor 4 b) (hb2 : is_factor b 16) (hb3 : b > 0) :
  {b' : ℕ | is_factor 4 b' ∧ is_factor b' 16 ∧ b' > 0}.to_finset.card = 3 :=
by sorry

end num_satisfying_b_l226_226295


namespace part1_part2_l226_226142

-- Step 1: Define necessary constants and the function
def f (x : ℝ) (m : ℝ) : ℝ := -sin x ^ 2 + m * cos x - 1

-- Step 2: Define the interval
def interval (x : ℝ) : Prop := -π / 3 ≤ x ∧ x ≤ 2 * π / 3

-- Step 3: Formulate the statement for Part (1)
theorem part1 (m : ℝ) (h1 : ∀ x, interval x → f x m ≥ -4) : m = 4.5 ∨ m = -3 := sorry

-- Define function for part (2)
def g (x : ℝ) : ℝ := (cos x + 1) ^ 2 - 3

-- Step 4: Formulate the statement for Part (2)
theorem part2 (a : ℝ) (h2 : ∀ x1 x2, interval x1 → interval x2 → |g x1 - g x2| ≤ 2 * a - 1 / 4) : a ≥ 2 := sorry

end part1_part2_l226_226142


namespace gold_opposite_face_is_silver_l226_226670

-- Definitions for the faces and colors involved
inductive Color
  | P  -- Purple
  | M  -- Magenta
  | C  -- Cyan
  | S  -- Silver
  | G  -- Gold
  | V  -- Violet
  | L  -- Lime

open Color

-- Definition of the condition problem as a Lean 4 statement
theorem gold_opposite_face_is_silver :
  ∃ (cube : Cube Color), 
    cube.top = P ∧
    cube.bottom = V ∧
    cube.has_faces [P, M, C, S, G, V, L] ∧
    cube.opposite_face G = S :=
  sorry

end gold_opposite_face_is_silver_l226_226670


namespace nylon_cord_length_approx_l226_226412

noncomputable def radius_approx (arc_length : ℝ) : ℝ := arc_length / Real.pi

theorem nylon_cord_length_approx (arc_length : ℝ) (h : arc_length ≈ 30) : 
  radius_approx arc_length ≈ 9.55 :=
by
  sorry

end nylon_cord_length_approx_l226_226412


namespace ellipse_locus_l226_226318
open Real EuclideanGeometry

noncomputable section

theorem ellipse_locus (A B : Point) (P : Point) :
  (dist P A + dist P B = 2 * dist A B) ↔
  ∃ c d : Point, is_ellipse c d (dist c d) P :=
sorry

end ellipse_locus_l226_226318


namespace ellipse_area_max_l226_226130

def ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity (e a : ℝ) : Prop :=
  e = sqrt (1 - (b^2 / a^2))

def max_area_OACB (a b c : ℝ) : ℝ :=
  (2 * sqrt (3) / 3)

theorem ellipse_area_max 
  (a b c : ℝ)
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ellipse_equation a b) 
  (h4 : eccentricity (sqrt 3 / 2) a) 
  (h5 : a = 2) 
  (h6 : b = 1) 
  (h7 : c = sqrt 3) :
  max_area_OACB a b c = 2 * sqrt 3 / 3 :=
by {
  sorry
}

end ellipse_area_max_l226_226130


namespace solve_cubic_equation_l226_226829

variable (k a b x : ℝ)

theorem solve_cubic_equation (h : k * x^3 + 3 * b^3 = (a - x)^3) :
  x = (a^2 + real.sqrt (a^4 + 4 * a^2 / 3 - 4 * b^3)) / 2 :=
sorry

end solve_cubic_equation_l226_226829


namespace smallest_no_1999_AP_contains_exactly_n_integers_l226_226888

theorem smallest_no_1999_AP_contains_exactly_n_integers :
  ∃ n : ℕ, n = 70 ∧ ∀ (a : ℝ) (d : ℝ), d = 1/n → a = 0 →
  (a + k * d = n → ∀ k ∈ (0..1998 : Set ℕ) → False) :=
sorry

end smallest_no_1999_AP_contains_exactly_n_integers_l226_226888


namespace expression_value_l226_226040

theorem expression_value : 7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by 
  -- proof goes here 
  sorry

end expression_value_l226_226040


namespace expression_value_l226_226039

theorem expression_value : 7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by 
  -- proof goes here 
  sorry

end expression_value_l226_226039


namespace malachi_selfies_l226_226610

theorem malachi_selfies (total_selfies : ℕ) (ratio1 ratio2 ratio3 ratio4 : ℕ) 
    (total_ratio : ℝ) (photo_per_ratio_unit : ℝ) 
    (ratio_first_year : ℕ) (ratio_fourth_year : ℕ) 
    (combined_photos : ℕ) :
    ratio1 = 10 → 
    ratio2 = 18 → 
    ratio3 = 23 → 
    ratio4 = 29 → 
    total_selfies = 2923 → 
    total_ratio = ratio1 + ratio2 + ratio3 + ratio4 → 
    photo_per_ratio_unit = total_selfies / total_ratio → 
    ratio_first_year = ratio1 → 
    ratio_fourth_year = ratio4 → 
    combined_photos = (ratio_first_year + ratio_fourth_year) * photo_per_ratio_unit →
    combined_photos ≈ 1434 := 
by 
  sorry

end malachi_selfies_l226_226610


namespace greatest_integer_value_l226_226770

theorem greatest_integer_value (x : ℤ) (h : x ∈ set.Icc (⌊ (-5 + sqrt 109) / 2 ⌋) (⌈ (5 + sqrt 109) / 2 ⌉)) :
  7 - 5 * x + x ^ 2 < 28 → x ≤ 7 :=
begin
  intro hx,
  have h1 := int.lt_ceil (5 + sqrt 109) / 2,
  have h2 := int.gt_floor (-5 + sqrt 109) / 2,
  sorry
end

end greatest_integer_value_l226_226770


namespace jump_rope_difference_l226_226481

noncomputable def cindy_jump_time : ℕ := 12
noncomputable def betsy_jump_time : ℕ := cindy_jump_time / 2
noncomputable def tina_jump_time : ℕ := 3 * betsy_jump_time

theorem jump_rope_difference : tina_jump_time - cindy_jump_time = 6 :=
by
  -- proof steps would go here
  sorry

end jump_rope_difference_l226_226481


namespace B_subscription_difference_l226_226831

noncomputable def subscription_difference (A B C P : ℕ) (delta : ℕ) (comb_sub: A + B + C = 50000) (c_profit: 8400 = 35000 * C / 50000) :=
  B - C

theorem B_subscription_difference (A B C : ℕ) (z: ℕ) 
  (h1 : A + B + C = 50000) 
  (h2 : A = B + 4000) 
  (h3 : (B - C) = z)
  (h4 :  8400 = 35000 * C / 50000):
  B - C = 10000 :=
by {
  sorry
}

end B_subscription_difference_l226_226831


namespace area_ratio_of_smaller_octagon_l226_226712

theorem area_ratio_of_smaller_octagon (A B C D E F G H P Q R S T U V W : Point) 
  (h1 : is_regular_octagon A B C D E F G H)
  (h2 : midpoint A B = P) (h3 : midpoint B C = Q) (h4 : midpoint C D = R)
  (h5 : midpoint D E = S) (h6 : midpoint E F = T) (h7 : midpoint F G = U)
  (h8 : midpoint G H = V) (h9 : midpoint H A = W):
  area (octagon A B C D E F G H) / area (octagon P Q R S T U V W) = 4 := sorry

end area_ratio_of_smaller_octagon_l226_226712


namespace functional_equation_solution_l226_226500

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 2 - x) :=
sorry

end functional_equation_solution_l226_226500


namespace division_problem_l226_226202

theorem division_problem (D : ℕ) (Quotient Dividend Remainder : ℕ) 
    (h1 : Quotient = 36) 
    (h2 : Dividend = 3086) 
    (h3 : Remainder = 26) 
    (h_div : Dividend = (D * Quotient) + Remainder) : 
    D = 85 := 
by 
  -- Steps to prove the theorem will go here
  sorry

end division_problem_l226_226202


namespace min_colors_pentagonal_tessellation_l226_226486

theorem min_colors_pentagonal_tessellation :
  ∀ (n : ℕ) (tessellation : list ℕ), (∀ i < n - 1, tessellation[i] ≠ tessellation[i + 1]) → (∃ colors : ℕ, colors = 2) :=
by
  assume n tessellation h,
  existsi 2,
  sorry

end min_colors_pentagonal_tessellation_l226_226486


namespace value_of_double_operation_l226_226519

def op1 (x : ℝ) : ℝ := 9 - x
def op2 (x : ℝ) : ℝ := x - 9

theorem value_of_double_operation :
  op2 (op1 10) = -10 := 
by 
  sorry

end value_of_double_operation_l226_226519


namespace arithmetic_seq_part1_arithmetic_seq_part2_l226_226533

variable {α : Type _} [linear_ordered_field α]

-- Let a be an arithmetic sequence with common difference d
def arithmetic_seq (a : ℕ → α) (d : α) := ∀ n, a (n + 1) = a n + d

-- The specific sequence part (Ⅰ)
theorem arithmetic_seq_part1 (a : ℕ → α) (d : α) (h1 : ∀ n, a n ≠ 0) (h2 : d ≠ 0) (h_arith: arithmetic_seq a d) :
  (1 / a 1) - (2 / a 2) + (1 / a 3) = 2 * d^2 / (a 1 * a 2 * a 3) := sorry

-- The generalized sequence part (Ⅱ)
theorem arithmetic_seq_part2 (a : ℕ → α) (d : α) (h1 : ∀ n, a n ≠ 0) (h2 : d ≠ 0) (h_arith: arithmetic_seq a d) :
  ∀ n ≥ 2, 
  (∑ i in finset.range n, (-1)^i * (nat.choose (n - 1) i) / a (i + 1)) = (n - 1)! * d^(n - 1) / (∏ i in finset.range n, a (i + 1)) := sorry

end arithmetic_seq_part1_arithmetic_seq_part2_l226_226533


namespace distance_preserved_by_f_l226_226003

structure Point :=
(x : ℝ) (y : ℝ)

def dist (p1 p2 : Point) : ℝ :=
real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

variable (f : Point → Point)

theorem distance_preserved_by_f (h : ∀ p1 p2 : Point, dist p1 p2 = 1 → dist (f p1) (f p2) = 1) :
  ∀ p q : Point, dist (f p) (f q) = dist p q :=
by sorry

end distance_preserved_by_f_l226_226003


namespace tan_add_identities_l226_226802

theorem tan_add_identities : tan 22 + tan 23 + (tan 22 * tan 23) = 1 :=
by
  sorry

end tan_add_identities_l226_226802


namespace find_number_l226_226006

noncomputable def question (x : ℝ) : Prop :=
  (2 * x^2 + Real.sqrt 6)^3 = 19683

theorem find_number : ∃ x : ℝ, question x ∧ (x = Real.sqrt ((27 - Real.sqrt 6) / 2) ∨ x = -Real.sqrt ((27 - Real.sqrt 6) / 2)) :=
  sorry

end find_number_l226_226006


namespace card_paiting_modulus_l226_226367

theorem card_paiting_modulus (cards : Finset ℕ) (H : cards = Finset.range 61 \ {0}) :
  ∃ d : ℕ, ∀ n ∈ cards, ∃! k, (∀ x ∈ cards, (x + n ≡ k [MOD d])) ∧ (d ∣ 30) ∧ (∃! n : ℕ, 1 ≤ n ∧ n ≤ 8) :=
sorry

end card_paiting_modulus_l226_226367


namespace sum_of_remainders_correct_l226_226201

-- Define the dividend
def dividend : ℕ := 12

-- Define the valid divisors, noting they must be natural numbers less than 12
def divisors : List ℕ := List.filter (λ x, x < 12) (List.range (dividend))

-- Define the remainders
def remainders : List ℕ := List.map (λ x, dividend % x) divisors

-- Define the distinct remainders
def distinct_remainders : List ℕ := List.eraseDup remainders

-- Define the sum of distinct remainders
def sum_of_distinct_remainders : ℕ := List.sum distinct_remainders

-- The proof problem statement
theorem sum_of_remainders_correct : sum_of_distinct_remainders = 15 :=
by
  sorry

end sum_of_remainders_correct_l226_226201


namespace paint_brush_sweep_ratio_l226_226817

variable (s w : ℝ)

theorem paint_brush_sweep_ratio (half_painted : s² / 2 = w² + (s - w)² / 2) :
  s / w = 2 * Real.sqrt 2 + 2 :=
sorry

end paint_brush_sweep_ratio_l226_226817


namespace area_ratio_of_smaller_octagon_l226_226713

theorem area_ratio_of_smaller_octagon (A B C D E F G H P Q R S T U V W : Point) 
  (h1 : is_regular_octagon A B C D E F G H)
  (h2 : midpoint A B = P) (h3 : midpoint B C = Q) (h4 : midpoint C D = R)
  (h5 : midpoint D E = S) (h6 : midpoint E F = T) (h7 : midpoint F G = U)
  (h8 : midpoint G H = V) (h9 : midpoint H A = W):
  area (octagon A B C D E F G H) / area (octagon P Q R S T U V W) = 4 := sorry

end area_ratio_of_smaller_octagon_l226_226713


namespace smaller_octagon_half_area_l226_226703

-- Define what it means to be a regular octagon
def is_regular_octagon (O : Point) (ABCDEFGH : List Point) : Prop :=
  -- Definition capturing the properties of a regular octagon around center O
  sorry

-- Define the function that computes the area of an octagon
def area_of_octagon (ABCDEFGH : List Point) : Real :=
  sorry

-- Define the function to create the smaller octagon by joining midpoints
def smaller_octagon (ABCDEFGH : List Point) : List Point :=
  sorry

theorem smaller_octagon_half_area (O : Point) (ABCDEFGH : List Point) :
  is_regular_octagon O ABCDEFGH →
  area_of_octagon (smaller_octagon ABCDEFGH) = (1 / 2) * area_of_octagon ABCDEFGH :=
by
  sorry

end smaller_octagon_half_area_l226_226703


namespace same_color_probability_l226_226178

/-- There are 7 red plates and 5 blue plates. We want to prove that the probability of
    selecting 3 plates, where all are of the same color, is 9/44. -/
theorem same_color_probability :
  let total_plates := 12
  let total_ways_to_choose := Nat.choose total_plates 3
  let red_plates := 7
  let blue_plates := 5
  let ways_to_choose_red := Nat.choose red_plates 3
  let ways_to_choose_blue := Nat.choose blue_plates 3
  let favorable_ways_to_choose := ways_to_choose_red + ways_to_choose_blue
  ∃ (prob : ℚ), prob = (favorable_ways_to_choose : ℚ) / (total_ways_to_choose : ℚ) ∧
                 prob = 9 / 44 :=
by
  sorry

end same_color_probability_l226_226178


namespace proof_problem_l226_226546

variables {a b c : Vec} -- Assuming a, b, and c are vectors of some vector space
variables {zero_vec : Vec} -- Assuming zero_vec is the zero vector in that vector space

-- Condition ①: |a| = |b| does not imply a = ±b
def prop_1 : Prop := ∀ (a b : Vec), (∥a∥ = ∥b∥) → (a ≠ b ∧ a ≠ -b)

-- Condition ②: a ⋅ b = 0 does not imply a = 0 or b = 0
def prop_2 : Prop := ∀ (a b : Vec), (a ⋅ b = 0) → (a ≠ zero_vec ∧ b ≠ zero_vec)

-- Condition ③: (a ∥ b) and (b ∥ c) does not imply a ∥ c if b = 0
def prop_3 : Prop := ∀ (a b c : Vec), (a ∥ b) → (b ∥ c) → (b = zero_vec) → ¬ (a ∥ c)

-- Condition ④: a ⋅ b = b ⋅ c does not imply a = c
def prop_4 : Prop := ∀ (a b c : Vec), (a ⋅ b = b ⋅ c) → (a ≠ c)

-- The full proof problem is to show that none of these propositions hold true
theorem proof_problem : prop_1 ∧ prop_2 ∧ prop_3 ∧ prop_4 := 
by 
  -- Proof steps would go here, but we are skipping the proof as instructed
  sorry

end proof_problem_l226_226546


namespace geom_seq_a12_value_l226_226617

-- Define the geometric sequence as a function from natural numbers to real numbers
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geom_seq_a12_value (a : ℕ → ℝ) 
  (H_geom : geom_seq a) 
  (H_7_9 : a 7 * a 9 = 4) 
  (H_4 : a 4 = 1) : 
  a 12 = 4 := 
by 
  sorry

end geom_seq_a12_value_l226_226617


namespace complex_solution_l226_226542

noncomputable def complex_z : ℂ := ((1-1*Complex.i)^2 - 3*(1+Complex.i)) / (2-Complex.i)
noncomputable def z_conj : ℂ := Complex.conj complex_z

theorem complex_solution :
  (complex_z = -1/5 - 13/5*Complex.i) ∧ 
  (z_conj = -1/5 + 13/5*Complex.i) ∧ 
  (∃ (a b : ℝ), a * complex_z + b = 1 - Complex.i ∧ a = 5/13 ∧ b = 14/13) :=
by sorry

end complex_solution_l226_226542


namespace find_a_and_solve_inequalities_l226_226595

-- Definitions as per conditions
def inequality1 (a : ℝ) (x : ℝ) : Prop := a*x^2 + 5*x - 2 > 0
def inequality2 (a : ℝ) (x : ℝ) : Prop := a*x^2 - 5*x + a^2 - 1 > 0

-- Statement of the theorem
theorem find_a_and_solve_inequalities :
  ∀ (a : ℝ),
    (∀ x, (1/2 < x ∧ x < 2) ↔ inequality1 a x) →
    a = -2 ∧
    (∀ x, (-1/2 < x ∧ x < 3) ↔ inequality2 (-2) x) :=
by
  intros a h
  sorry

end find_a_and_solve_inequalities_l226_226595


namespace cos_theta_l226_226245

def n1 := vector.ofFn (λ i : Fin 3, match i.val with
  | 0 => 1
  | 1 => -2
  | _ => 3)

def n2 := vector.ofFn (λ i : Fin 3, match i.val with
  | 0 => 4
  | 1 => 1
  | _ => -2)

def dot_product (v1 v2 : Vector ℝ 3) : ℝ :=
  (vector.nth v1 0 * vector.nth v2 0) +
  (vector.nth v1 1 * vector.nth v2 1) +
  (vector.nth v1 2 * vector.nth v2 2)

def magnitude (v : Vector ℝ 3) : ℝ :=
  Real.sqrt ((vector.nth v 0) ^ 2 + (vector.nth v 1) ^ 2 + (vector.nth v 2) ^ 2)

theorem cos_theta : 
  let cos_theta := (dot_product n1 n2) / ((magnitude n1) * (magnitude n2))
  in cos_theta = (-4 / Real.sqrt 294) :=
by sorry

end cos_theta_l226_226245


namespace smallest_hamburger_packages_l226_226063

theorem smallest_hamburger_packages (h_num : ℕ) (b_num : ℕ) (h_bag_num : h_num = 10) (b_bag_num : b_num = 15) :
  ∃ (n : ℕ), n = 3 ∧ (n * h_num) = (2 * b_num) := by
  sorry

end smallest_hamburger_packages_l226_226063


namespace sum_of_distinct_integers_l226_226251

-- Introduce the needed definitions and conditions
variables {p q r s t : ℤ} 

-- The main theorem to be proven
theorem sum_of_distinct_integers:
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -120 ∧
  (p, q, r, s, t).ToList.nodup → 
  p + q + r + s + t = 27 := 
by
  sorry

end sum_of_distinct_integers_l226_226251


namespace unique_function_l226_226501

noncomputable def f (x : ℚ) : ℚ :=
  x + 1

theorem unique_function (f : ℚ → ℚ) :
  (f 1 = 2) ∧ (∀ (x y : ℚ), f (x * y) = f x * f y - f (x + y) + 1)
  → (∀ (x : ℚ), f x = x + 1) := 
by
  intro h
  cases h with h1 h2
  have := funext h2
  sorry

end unique_function_l226_226501


namespace factorization_correct_l226_226306

theorem factorization_correct:
  (a: ℝ) → (a^2 - 4a + 4 = (a - 2)^2) :=
by
  intro a
  sorry

end factorization_correct_l226_226306


namespace expected_value_X_l226_226765

-- Define the event of selecting courses.
noncomputable def select_courses : Type := {
  A_courses : Finset (Fin 6) // Courses selected by individual A
  B_courses : Finset (Fin 6) // Courses selected by individual B
  hA : A_courses.card = 3
  hB : B_courses.card = 3
}

-- Define the random variable X representing the number of common courses
def X (s : select_courses) : ℕ := (s.A_courses ∩ s.B_courses).card

-- Theorem: The expected value of X is 1.5
theorem expected_value_X : (E[X] = 1.5 : ℝ) := sorry

end expected_value_X_l226_226765


namespace tom_beach_days_l226_226758

theorem tom_beach_days (total_seashells days_seashells : ℕ) (found_each_day total_found : ℕ) 
    (h1 : found_each_day = 7) (h2 : total_found = 35) : total_found / found_each_day = 5 := 
by 
  sorry

end tom_beach_days_l226_226758


namespace smallest_x_l226_226402

theorem smallest_x (x : ℕ) (h₁ : x % 3 = 2) (h₂ : x % 4 = 3) (h₃ : x % 5 = 4) : x = 59 :=
by
  sorry

end smallest_x_l226_226402


namespace sheep_to_cow_ratio_l226_226272

theorem sheep_to_cow_ratio : 
  ∀ (cows sheep : ℕ) (cow_water sheep_water : ℕ),
  cows = 40 →
  cow_water = 80 →
  sheep_water = cow_water / 4 →
  7 * (cows * cow_water + sheep * sheep_water) = 78400 →
  sheep / cows = 10 :=
by
  intros cows sheep cow_water sheep_water hcows hcow_water hsheep_water htotal
  sorry

end sheep_to_cow_ratio_l226_226272


namespace problem_I_problem_II_l226_226556

-- Problem I: 
variables {a m : ℝ}
def f (x : ℝ) : ℝ := |x - a|

theorem problem_I (h : ∀ x, f x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) : a = 2 ∧ m = 3 :=
by sorry

-- Problem II: 
variables {t x : ℝ}

def g (x : ℝ) : ℝ := |x - 2|
def p (x : ℝ) (t : ℝ) : Prop := g x + t ≥ |x|

theorem problem_II (h₁ : 0 ≤ t) (h₂ : t < 2) : p x t ↔ x ∈ Set.Iic ((t + 2) / 2) :=
by sorry

end problem_I_problem_II_l226_226556


namespace vika_card_pairs_l226_226373

theorem vika_card_pairs : 
  let numbers := finset.range 61 \ finset.singleton 0 in
  let divs := {d | d ∈ finset.divisors 30} in
  numbers.card = 60 →
  ∀ d ∈ divs, ∀ pair : finset (ℕ × ℕ),
    pair.card = 30 →
    finset.forall₂ pair (λ x y, |x.1 - x.2| % d = |y.1 - y.2| % d) → 
    ∃ (number_of_pairs : ℕ), number_of_pairs = 8 :=
by 
  intro numbers divs hc hd hp hpairs,
  sorry

end vika_card_pairs_l226_226373


namespace surface_area_between_paraboloids_l226_226092

def paraboloid (x y : ℝ) : ℝ := x^2 - y^2
def upper_paraboloid (x y : ℝ) : ℝ := 3 * x^2 + y^2 - 2
def lower_paraboloid (x y : ℝ) : ℝ := 3 * x^2 + y^2 - 4
def projection_boundary (x y : ℝ) : Prop := 1 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2

theorem surface_area_between_paraboloids :
  let S := { (x, y) : ℝ × ℝ | projection_boundary x y }
  ∃ (μ : ℝ), μ = ∫∫ S, sqrt (1 + 4 * x^2 + 4 * y^2) ∂(x, y) → μ = (π / 6) * (27 - 5 * sqrt 5) :=
by
  sorry

end surface_area_between_paraboloids_l226_226092


namespace fraction_of_girls_is_one_half_l226_226030

def fraction_of_girls (total_students_jasper : ℕ) (ratio_jasper : ℕ × ℕ) (total_students_brookstone : ℕ) (ratio_brookstone : ℕ × ℕ) : ℚ :=
  let (boys_ratio_jasper, girls_ratio_jasper) := ratio_jasper
  let (boys_ratio_brookstone, girls_ratio_brookstone) := ratio_brookstone
  let girls_jasper := (total_students_jasper * girls_ratio_jasper) / (boys_ratio_jasper + girls_ratio_jasper)
  let girls_brookstone := (total_students_brookstone * girls_ratio_brookstone) / (boys_ratio_brookstone + girls_ratio_brookstone)
  let total_girls := girls_jasper + girls_brookstone
  let total_students := total_students_jasper + total_students_brookstone
  total_girls / total_students

theorem fraction_of_girls_is_one_half :
  fraction_of_girls 360 (7, 5) 240 (3, 5) = 1 / 2 :=
  sorry

end fraction_of_girls_is_one_half_l226_226030


namespace value_of_x_l226_226313

noncomputable def mean (l : List ℝ) : ℝ := l.sum / l.length

def mode (l : List ℝ) : Option ℝ :=
l.groupBy (· = ·) 
|> List.sortBy (·.length) 
|> List.head

def median (l : List ℝ) : ℝ :=
let sorted := l.qsort (· < ·)
if h : sorted.length % 2 = 1 then
  sorted.get ⟨sorted.length / 2, Nat.div_lt_of_lt_mul (List.length_pos_of_mem (List.head_ne_nil l sorted))⟩
else
  (sorted.get ⟨sorted.length / 2 - 1, Nat.pred_lt (by exact_mod_cast sorted.length / 2)⟩ + 
   sorted.get ⟨sorted.length / 2, Nat.div_lt_of_lt_mul (List.length_pos_of_mem (List.head_ne_nil l sorted))⟩) / 2

theorem value_of_x {x : ℝ} :
  mean [65, 105, x, x, 45, 55, 205, 95, 85] = x ∧
  median [65, 105, x, x, 45, 55, 205, 95, 85] = x ∧
  mode [65, 105, x, x, 45, 55, 205, 95, 85] = some x →
  x ≈ 93.57 :=
by
  sorry

end value_of_x_l226_226313


namespace linear_function_value_at_neg_one_l226_226296

noncomputable def is_linear (f : ℝ → ℝ) : Prop :=
∃ (a b : ℝ), ∀ x, f(x) = a * x + b

theorem linear_function_value_at_neg_one
  (f : ℝ → ℝ)
  (h_linear : is_linear f)
  (h_inverse : ∀ x, f(x) = 3 * (f.symm x) + 5)
  (h_at_zero : f 0 = 3) :
  f (-1) = (2 * Real.sqrt 3) / 3 :=
  sorry

end linear_function_value_at_neg_one_l226_226296


namespace max_g_in_interval_0_sqrt5_max_value_g_in_interval_0_sqrt5_l226_226087

noncomputable def g (x : ℝ) : ℝ := 5 * x - x^5

theorem max_g_in_interval_0_sqrt5 :
  ∀ x ∈ set.Icc (0 : ℝ) (Real.sqrt 5), g x ≤ 4 :=
begin
  sorry
end

theorem max_value_g_in_interval_0_sqrt5 :
  ∃ x ∈ set.Icc (0 : ℝ) (Real.sqrt 5), g x = 4 :=
begin
  use 1,
  split,
  { norm_num, apply Real.sqrt_nonneg },
  { norm_num }
end

end max_g_in_interval_0_sqrt5_max_value_g_in_interval_0_sqrt5_l226_226087


namespace expression_undefined_at_y_l226_226864

noncomputable def undefined_value (y : ℝ) : Prop :=
  y^2 - 10*y + 25 = 0

theorem expression_undefined_at_y :
  ∃ y : ℝ, undefined_value y ∧ y = 5 :=
by
  use 5
  split
  · calc
      (5 : ℝ)^2 - 10*5 + 25 = 25 - 50 + 25 := by norm_num
                            ... = 0         := by norm_num
  · exact rfl

end expression_undefined_at_y_l226_226864


namespace vectors_subtraction_perpendicular_l226_226571

def vector_a : (ℝ × ℝ) := (1, 0)
def vector_b : (ℝ × ℝ) := (1 / 2, 1 / 2)

theorem vectors_subtraction_perpendicular (a b : ℝ × ℝ) (ha : a = vector_a) (hb : b = vector_b) :
  let diff := (a.1 - b.1, a.2 - b.2) in
  (diff.1 * b.1 + diff.2 * b.2) = 0 :=
sorry

end vectors_subtraction_perpendicular_l226_226571


namespace k_value_perpendicular_l226_226155

noncomputable def k_perpendicular_condition : Prop := 
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (0, 1)
  ∀ k : ℝ, let u := (k * a.1 + b.1, k * a.2 + b.2)
           let v := (3 * a.1 - b.1, 3 * a.2 - b.2)
           inner u v = 0 → k = 1 / 3

theorem k_value_perpendicular : k_perpendicular_condition :=
sorry

end k_value_perpendicular_l226_226155


namespace num_six_digit_numbers_with_conditions_l226_226899

open Finset

theorem num_six_digit_numbers_with_conditions :
  (∑ p in permutations (finset.range 6), 
    (∀ i in finset.range 5, parity (p i) ≠ parity (p (i + 1))) ∧
    ∃ i in finset.range 5, (p i = 0 ∧ p (i + 1) = 1 ∨ p i = 1 ∧ p (i + 1) = 0)) =
  40 :=
by sorry

end num_six_digit_numbers_with_conditions_l226_226899


namespace sum_of_squares_ne_sum_of_fourth_powers_l226_226671

theorem sum_of_squares_ne_sum_of_fourth_powers :
  ∀ (a b : ℤ), a^2 + (a + 1)^2 ≠ b^4 + (b + 1)^4 :=
by 
  sorry

end sum_of_squares_ne_sum_of_fourth_powers_l226_226671


namespace characteristic_values_eigenfunctions_l226_226416

-- Define the kernel function K
def K (x t : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ t then cos x * sin t
  else if t ≤ x ∧ x ≤ π then cos t * sin x
  else 0  -- Outside of the defined regions

-- Define the integral equation as a structure for reference and later use
structure IntegralEquation :=
  (φ : ℝ → ℝ)
  (λ : ℝ)
  (eqn : ∀ x : ℝ, φ x - λ * ∫ t in 0..π, K x t * φ t = 0)

-- Boundary conditions
def boundary_conditions (φ : ℝ → ℝ) :=
  (φ π = 0) ∧ (deriv φ 0 = 0)

-- The main theorem statement
theorem characteristic_values_eigenfunctions (λ : ℝ) (φ : ℝ → ℝ) (h_eq : IntegralEquation φ λ) (h_bc : boundary_conditions φ) :
  inf (λ_n : ℝ) (φ_n : ℝ → ℝ) (λ_n = 1 - (n + 1/2)^2 ∧ φ_n x = cos ((n + 1/2) * x)) :=
sorry

end characteristic_values_eigenfunctions_l226_226416


namespace regression_line_l226_226497

noncomputable def mean (s : List ℝ) : ℝ := (s.sum) / (s.length)

theorem regression_line (x y : List ℝ)
  (hx : x = [1, 2, 4, 5])
  (hy : y = [2, 5, 7, 10]) :
  let xbar := mean x
      ybar := mean y
      numerator := (List.zip x y).map (λ p, (p.1 - xbar) * (p.2 - ybar)).sum
      denominator := x.map (λ xi, (xi - xbar) ^ 2).sum
      b := numerator / denominator
      a := ybar - b * xbar
  in
  a = 0.6 ∧ b = 1.8 := 
by
  sorry

end regression_line_l226_226497


namespace cube_surface_area_correct_l226_226437

noncomputable def total_surface_area_of_reassembled_cube : ℝ :=
  let height_X := 1 / 4
  let height_Y := 1 / 6
  let height_Z := 1 - (height_X + height_Y)
  let top_bottom_area := 3 * 1 -- Each slab contributes 1 square foot for the top and bottom
  let side_area := 2 * 1 -- Each side slab contributes 1 square foot
  let front_back_area := 2 * 1 -- Each front and back contributes 1 square foot
  top_bottom_area + side_area + front_back_area

theorem cube_surface_area_correct :
  let height_X := 1 / 4
  let height_Y := 1 / 6
  let height_Z := 1 - (height_X + height_Y)
  let total_surface_area := total_surface_area_of_reassembled_cube
  total_surface_area = 10 :=
by
  sorry

end cube_surface_area_correct_l226_226437


namespace max_value_of_f_l226_226311

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem max_value_of_f : ∀ x > 0, f x ≤ 1 :=
begin
  sorry
end

end max_value_of_f_l226_226311


namespace hexagon_interior_angles_equal_l226_226105

variables {V : Type*} [inner_product_space ℝ V] [nontrivial V]

structure Hexagon :=
(A B C D E F : V)
(convex : convex_hull ℝ {A, B, C, D, E, F}.finite.to_set = convex_hull ℝ (finset.univ : finset V))

def opposite_pairs (hex : Hexagon) := [(hex.A, hex.D), (hex.B, hex.E), (hex.C, hex.F)]

def distance_between_midpoints (p₁ p₂ : V × V) : ℝ :=
∥(p₁.1 +ᵥ (p₁.2 -ᵥ p₁.1)/2 : V) - (p₂.1 +ᵥ (p₂.2 -ᵥ p₂.1)/2 : V)∥

theorem hexagon_interior_angles_equal (hex : Hexagon)
  (h : ∀ (p ∈ opposite_pairs hex), distance_between_midpoints p.1 p.2 = (√3 / 2) * (∥p.1.2 -ᵥ p.1.1∥ + ∥p.2.2 -ᵥ p.2.1∥)) :
  ∠A hex.A hex.F hex.B = 120 ∧
  ∠A hex.B hex.A hex.C = 120 ∧
  ∠A hex.C hex.B hex.D = 120 ∧
  ∠A hex.D hex.C hex.E = 120 ∧
  ∠A hex.E hex.D hex.F = 120 ∧
  ∠A hex.F hex.E hex.A = 120 :=
sorry

end hexagon_interior_angles_equal_l226_226105


namespace linear_transform_l226_226897

variable (a b x : ℝ)
def curve_eqn : ℝ := a * Real.exp(b / x)
def mu (y : ℝ) : ℝ := Real.log y
def c (a : ℝ) : ℝ := Real.log a
def v (x : ℝ) : ℝ := 1 / x

theorem linear_transform (y : ℝ) (h_y : y = curve_eqn a b x)
  (h_mu : mu y = Real.log y) (h_c : c a = Real.log a) (h_v : v x = 1 / x) :
  mu y = c a + b * v x :=
by 
  rw [h_mu, h_y, curve_eqn, Real.log_mul (Real.exp_pos (b / x)).ne', Real.log_exp, h_c, h_v]
  sorry

end linear_transform_l226_226897


namespace main_theorem_l226_226253

-- Declare nonzero complex numbers
variables {x y z : ℂ} 

-- State the conditions
def conditions (x y z : ℂ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  x + y + z = 30 ∧
  (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z

-- Prove the main statement given the conditions
theorem main_theorem (h : conditions x y z) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 33 :=
by
  sorry

end main_theorem_l226_226253


namespace find_m_l226_226147

variable (m : ℝ)

def hyperbola_eq (y x : ℝ) : Prop := m * y^2 - x^2 = 1

def ellipse_eq (y x : ℝ) : Prop := y^2 / 5 + x^2 = 1

def same_foci : Prop :=
  ∃ c : ℝ, c^2 = 5 - 1 ∧ 1 / m + 1 = 4

theorem find_m (h1 : ∀ y x, hyperbola_eq m y x) (h2 : ∀ y x, ellipse_eq y x) (h3 : same_foci) :
  m = 1 / 3 :=
sorry

end find_m_l226_226147


namespace simplify_expression_l226_226518

theorem simplify_expression : 
  let a := (3 + 2 : ℚ)
  let b := a⁻¹ + 2
  let c := b⁻¹ + 2
  let d := c⁻¹ + 2
  d = 65 / 27 := by
  sorry

end simplify_expression_l226_226518


namespace simplify_expression_l226_226425

variable (x y : ℝ)

theorem simplify_expression : 2 * x^2 * y - 4 * x * y^2 - (-3 * x * y^2 + x^2 * y) = x^2 * y - x * y^2 :=
by
  sorry

end simplify_expression_l226_226425


namespace fourth_metal_mass_approx_l226_226816

noncomputable def mass_of_fourth_metal 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 = 1.5 * x2)
  (h2 : x3 = 4 / 3 * x2)
  (h3 : x4 = 6 / 5 * x3)
  (h4 : x1 + x2 + x3 + x4 = 25) : ℝ :=
  x4

theorem fourth_metal_mass_approx 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 = 1.5 * x2)
  (h2 : x3 = 4 / 3 * x2)
  (h3 : x4 = 6 / 5 * x3)
  (h4 : x1 + x2 + x3 + x4 = 25) : 
  abs (mass_of_fourth_metal x1 x2 x3 x4 h1 h2 h3 h4 - 7.36) < 0.01 :=
by
  sorry

end fourth_metal_mass_approx_l226_226816


namespace sum_min_max_prime_factors_of_1365_l226_226776

-- Define a function to check if a number is prime (for manual handling in proof)
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the number we're working with
def number := 1365

-- Define the list of known prime factors
def prime_factors : list ℕ := [3, 5, 7, 13]

-- Ensure all elements of prime_factors are prime
lemma all_primes : ∀ p ∈ prime_factors, is_prime p := by
  intros p h
  cases h
  case or.inl h_0 => exact nat.prime_three
  case or.inr h_0 => cases h_0
    case or.inl h_1 => exact nat.prime_five
    case or.inr h_1 => cases h_1
      case or.inl h_2 => exact nat.prime_seven
      case or.inr h_2 => exact nat.prime_thirteen

-- Define the sum of the largest and smallest prime factors
def sum_of_min_and_max : ℕ := list.minimum' prime_factors + list.maximum' prime_factors

-- Theorem stating that this sum is 16
theorem sum_min_max_prime_factors_of_1365 : sum_of_min_and_max = 16 := by
  sorry

end sum_min_max_prime_factors_of_1365_l226_226776


namespace marbles_left_l226_226271

theorem marbles_left (total_marbles : ℕ) (large_boxes : ℕ) (marbles_per_large_box : ℕ) 
  (small_boxes : ℕ) (marbles_per_small_box : ℕ) : 
  total_marbles = 240 ∧ large_boxes = 4 ∧ marbles_per_large_box = 35 ∧ small_boxes = 3 ∧ marbles_per_small_box = 6 →
  total_marbles - (large_boxes * marbles_per_large_box + small_boxes * marbles_per_small_box) = 82 :=
by
  intros h,
  cases h with h_total_marbles h,
  cases h with h_large_boxes h,
  cases h with h_marbles_per_large_box h,
  cases h with h_small_boxes h_marbles_per_small_box,
  simp [h_total_marbles, h_large_boxes, h_marbles_per_large_box, h_small_boxes, h_marbles_per_small_box],
  apply eq.refl,
  sorry

end marbles_left_l226_226271


namespace three_digit_valid_count_l226_226966

theorem three_digit_valid_count : 
  let total_three_digit_numbers := 900 in
  let count_AAA := 9 in
  let count_AAB := 81 in
  let count_BAA := 81 in
  total_three_digit_numbers - (count_AAA + count_AAB + count_BAA) = 729 :=
by
  -- Define the total number of three-digit numbers
  let total_three_digit_numbers := 900
  -- Define the count for each excluded pattern
  let count_AAA := 9
  let count_AAB := 81
  let count_BAA := 81
  -- Calculate the remaining valid three-digit numbers
  have valid_three_digit_numbers := total_three_digit_numbers - (count_AAA + count_AAB + count_BAA)
  exact valid_three_digit_numbers = 729

end three_digit_valid_count_l226_226966


namespace find_angle_A_find_area_ABC_l226_226194

noncomputable def angle_A (B C : ℝ) : ℝ :=
if h : sin^2(2*π/3) = sin^2(B) + sin^2(C) + sin(B) * sin(C)
then 2*π/3 else 0

noncomputable def area_ABC (a b : ℝ) (c : ℝ) (A : ℝ) : ℝ :=
1/2 * b * c * sin (A)

theorem find_angle_A (B C : ℝ) :
  ∀ A, (sin^2 A = sin^2 B + sin^2 C + sin B * sin C) → A = 2*π/3 :=
by sorry

theorem find_area_ABC (a b : ℝ) (c : ℝ) :
  ∀ A, A = 2*π/3 → area_ABC 3 2 (sqrt 6 - 1) (2*π/3) = (3*sqrt 2 - sqrt 3)/2 :=
by sorry

end find_angle_A_find_area_ABC_l226_226194


namespace contradiction_divisibility_l226_226408

theorem contradiction_divisibility (a b : ℕ) 
(h1 : ¬ (a % 5 = 0)) 
(h2 : ¬ (b % 5 = 0)) 
(h3 : (a * b) % 5 = 0) : 
  false := 
begin
  sorry
end

end contradiction_divisibility_l226_226408


namespace hyperbola_triangle_area_l226_226914

theorem hyperbola_triangle_area 
(x y : ℝ) (P F1 F2 : ℝ × ℝ) 
(hyperbola_eq : P.1^2 - P.2^2 / 12 = 1)
(foci_def : (F1, F2) = ((√13, 0), (-√13, 0)))
(dist_ratio : dist P F1 / dist P F2 = 3 / 2) :
  ∃ S : ℝ, S = 12 :=
by
  sorry

end hyperbola_triangle_area_l226_226914


namespace large_circle_diameter_l226_226073

-- Definitions for the conditions
def small_circle_radius : ℝ := 4
def number_of_small_circles : ℕ := 8
def inner_layer_form : String := "regular hexagon"
def concentric_layers : Bool := true
def tangency_condition_inner_layer : Bool := true
def tangency_condition_outer_layer : Bool := true
def inner_circle_side_length := 2 * small_circle_radius

-- Main theorem statement
theorem large_circle_diameter :
  small_circle_radius = 4 →
  number_of_small_circles = 8 →
  inner_layer_form = "regular hexagon" →
  concentric_layers = true →
  tangency_condition_inner_layer = true →
  tangency_condition_outer_layer = true →
  inner_circle_side_length = 8 →
  2 * (8 * √3 / 2 + 4) = 32 :=
by
  intro h_radius h_num h_form h_concentric h_tangency_inner h_tangency_outer h_side_length
  sorry

end large_circle_diameter_l226_226073


namespace find_sequence_and_sum_l226_226115

variable {n : ℕ}

-- Conditions
def a_n : ℕ → ℕ := λ n => 2 * n + 1
def b_n : ℕ → ℕ := λ n => 2 ^ (n - 1)
def S_n : ℕ → ℕ := λ n => n * (n + 2)
def c_n : ℕ → ℝ :=
  λ n => if n % 2 = 1 then 2 / S_n n else (2 : ℝ)^(n - 1)

def T_n (n : ℕ) : ℝ :=
  if n % 2 = 1 then (2^n + 1) / 3 - (n / (n + 2)) else (2^(n + 1) + 1) / 3 - (1 / (n + 1))

-- Proof problem statement
theorem find_sequence_and_sum :
  (a_n 1 = 3) ∧ (b_n 1 = 1) ∧ (b_n 2 + S_n 2 = 10) ∧ (a_n 5 - 2 * b_n 2 = a_n 3) ∧
  (∀ n, S_n n = n * (n + 2)) ∧ 
  (forall n, T_n n = if n % 2 = 1 then (2^n + 1) / 3 - (n / (n + 2)) else (2^(n + 1) + 1) / 3 - (1 / n + 1))
:= sorry

end find_sequence_and_sum_l226_226115


namespace find_a_l226_226945

theorem find_a (a : ℝ) 
  (h1 : 0 ≤ a) 
  (p_X0 : 2 * a^2 = P(0)) 
  (p_X1 : a = P(1)) 
  (h2 : P(0) + P(1) = 1): 
  a = 1 / 2 := 
by
  sorry

end find_a_l226_226945


namespace line_circle_intersection_l226_226191

theorem line_circle_intersection (k : ℝ) :
    (∃ P Q : ℝ × ℝ, (P ≠ Q) ∧ (P.1 ^ 2 + P.2 ^ 2 = 1) ∧ (Q.1 ^ 2 + Q.2 ^ 2 = 1) ∧ 
                     (P.2 = k * P.1 - 1) ∧ (Q.2 = k * Q.1 - 1) ∧ 
                     ∃ O : ℝ × ℝ, O = (0, 0) ∧ (angle P O Q = real.pi * 5 / 6)) 
    → (k = 2 + real.sqrt 3 ∨ k = -(2 + real.sqrt 3)) := sorry

end line_circle_intersection_l226_226191


namespace value_set_real_l226_226650

noncomputable def f (P0 : ℝ → ℝ) (P : ℕ → (ℝ → ℝ)) (a : ℕ → ℝ) (n : ℕ) : ℝ → ℝ :=
  λ x, P0 x + ∑ k in Finset.range n, a k * |P k x|

theorem value_set_real (P0 : ℝ → ℝ) (P : ℕ → (ℝ → ℝ)) (a : ℕ → ℝ) (n : ℕ) :
  (∀ x₁ x₂ : ℝ, f P0 P a n x₁ = f P0 P a n x₂ → x₁ = x₂) →
  set.range (f P0 P a n) = set.univ :=
sorry

end value_set_real_l226_226650


namespace completely_factored_form_l226_226055

theorem completely_factored_form (x : ℤ) :
  (12 * x ^ 3 + 95 * x - 6) - (-3 * x ^ 3 + 5 * x - 6) = 15 * x * (x ^ 2 + 6) :=
by
  sorry

end completely_factored_form_l226_226055


namespace find_a_1000_l226_226203

noncomputable def a : ℕ → ℤ
| 0     := 5000
| 1     := 5001
| (n+2) := 2 * (n+1) - a n - a (n+1)

theorem find_a_1000 : a 999 = 5666 :=
by folding ⟨ λ n : ℕ, (a n + a (n+1) + a (n+2) = 2 n) sorry ⟩

end find_a_1000_l226_226203


namespace vika_pairs_exactly_8_ways_l226_226345

theorem vika_pairs_exactly_8_ways :
  ∃ d : ℕ, (d ∣ 30) ∧ (Finset.card (Finset.filter (λ d, d ∣ 30) (Finset.range 31)) = 8) := 
sorry

end vika_pairs_exactly_8_ways_l226_226345


namespace solution_set_of_inequality_l226_226101

theorem solution_set_of_inequality (a : ℝ) (h : 0 < a) :
  {x : ℝ | x ^ 2 - 4 * a * x - 5 * a ^ 2 < 0} = {x : ℝ | -a < x ∧ x < 5 * a} :=
sorry

end solution_set_of_inequality_l226_226101


namespace range_of_a_l226_226693

def f (x : ℝ) (a : ℝ) : ℝ := 2^x - a / 2^x
def p (x : ℝ) (a : ℝ) : ℝ := 2^(x - 2) - a / 2^(x - 2)
def g (x : ℝ) (a : ℝ) : ℝ := a / 2^(x - 2) - 2^(x - 2) + 2
def F (x : ℝ) (a : ℝ) : ℝ := f x a / a + g x a

theorem range_of_a (a : ℝ) (m : ℝ) :
  (∃ m, m = (F x a) ∧ m > 2 + real.sqrt 7) ↔ (1/2 < a ∧ a < 2) :=
sorry

end range_of_a_l226_226693


namespace smaller_octagon_area_half_l226_226710

theorem smaller_octagon_area_half
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (P Q R S T U V W : Point)
  (H1 : midpoint P A B)
  (H2 : midpoint Q B C)
  (H3 : midpoint R C D)
  (H4 : midpoint S D E)
  (H5 : midpoint T E F)
  (H6 : midpoint U F G)
  (H7 : midpoint V G H)
  (H8 : midpoint W H A):
  area (octagon P Q R S T U V W) = (1 / 2) * area (octagon ABCDEFGH) :=
sorry

end smaller_octagon_area_half_l226_226710


namespace certain_number_l226_226582

def bin_op (n : ℤ) : ℤ := n - (n * 5)

theorem certain_number : ∃ N : ℤ, (∀ n : ℤ, 0 < n ∧ n @ n < N) ∧ N = -11 :=
by
  sorry

end certain_number_l226_226582


namespace product_of_roots_l226_226883

-- Define the first polynomial f(x)
def f (x : ℝ) : ℝ := 3*x^4 + 2*x^3 - 5*x + 15

-- Define the second polynomial g(x)
def g (x : ℝ) : ℝ := 4*x^3 - 10*x^2 + 22*x - 7

-- Define the product polynomial h(x) as f(x) * g(x)
def h (x : ℝ) : ℝ := f x * g x

-- State the theorem about the product of the roots of h(x)
theorem product_of_roots : 
  (0 : ℝ) = h ∧ (∀ z : ℝ, h z = 0 → product_of_roots_of_polynomial h = -(35/4)) := 
sorry

end product_of_roots_l226_226883


namespace cupcakes_frosted_l226_226471

def rate_Cagney := 1 / 15
def rate_Lacey := 1 / 25
def rate_Hardy := 1 / 50
def combined_rate := rate_Cagney + rate_Lacey + rate_Hardy
def total_time := 6 * 60 (seconds)

theorem cupcakes_frosted : (combined_rate * total_time = 45) := by
  sorry

end cupcakes_frosted_l226_226471


namespace problem_bx_100a_plus_b_l226_226990

theorem problem_bx_100a_plus_b
  (A B C D X O : Type)
  [RightTriangle ABC B]
  (hD : OnHypotenuse D AC)
  (hBD_perp : BD ⊥ AC)
  (ω : Circle O)
  (hω_passes : PassesThrough ω C)
  (hω_passes_D : PassesThrough ω D)
  (hω_tangent : TangentTo ω AB)
  (hω_NOT_tangent_B : TangentNotAt ω AB B)
  (hX_on_BC : On X BC)
  (hAX_perp_BO : AX ⊥ BO)
  (hAB_eq : AB = 2)
  (hBC_eq : BC = 5)
  : ∃ (a b : ℕ), RelativelyPrime a b ∧ 100 * a + b = 8041 :=
begin
  sorry
end

end problem_bx_100a_plus_b_l226_226990


namespace vika_card_pairing_l226_226355

theorem vika_card_pairing :
  ∃ (d ∈ {1, 2, 3, 5, 6, 10, 15, 30}), ∃ (k : ℕ), 60 = 2 * d * k :=
by sorry

end vika_card_pairing_l226_226355


namespace marbles_before_purchase_l226_226851

-- Lean 4 statement for the problem
theorem marbles_before_purchase (bought : ℝ) (total_now : ℝ) (initial : ℝ) 
    (h1 : bought = 134.0) 
    (h2 : total_now = 321) 
    (h3 : total_now = initial + bought) : 
    initial = 187 :=
by 
    sorry

end marbles_before_purchase_l226_226851


namespace smallest_x_l226_226396

open Classical
noncomputable theory

def conditions (x : ℕ) : Prop :=
  x % 3 = 2 ∧ x % 4 = 3 ∧ x % 5 = 4

theorem smallest_x : ∃ (x : ℕ), conditions x ∧ (∀ (y : ℕ), conditions y → x ≤ y) ∧ x = 59 :=
by {
  sorry
}

end smallest_x_l226_226396


namespace seating_arrangement_l226_226611

theorem seating_arrangement (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 2) :
  (∃ (table : Fin n → ℕ), ∃ (a b : Fin n), a ≠ b ∧ table a = table b ∧
      ∀ (i j : Fin n), i ≠ j → table (Fin.mod (i + 1) n) = table (Fin.mod (j + 1) n) →
        table i = table j) = 80640 :=
by
  sorry

end seating_arrangement_l226_226611


namespace range_of_a_l226_226146

theorem range_of_a (a : ℝ) : 
  (∀ x1 x2 ∈ set.Icc (0 : ℝ) 2, x1 < x2 → (log a (4 - a * x1)) > (log a (4 - a * x2))) ↔ (1 < a ∧ a < 2) :=
  sorry

end range_of_a_l226_226146


namespace categorization_correct_l226_226870

def numbers : List ℚ := [-3, 5, 2023, 0, 7 / 4, -8.23, (-1 : ℚ) ^ 2023]

def negative_rationals (l : List ℚ) : List ℚ :=
  l.filter (λ x => x < 0)

def integers (l : List ℚ) : List ℚ :=
  l.filter (λ x => x.denominator = 1)

theorem categorization_correct :
  negative_rationals numbers = [-3, -8.23, -1] ∧
  integers numbers = [-3, 5, 2023, 0, -1] :=
by
  sorry

end categorization_correct_l226_226870


namespace polynomial_evaluation_l226_226046

theorem polynomial_evaluation :
  7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 :=
by
  sorry

end polynomial_evaluation_l226_226046


namespace isosceles_triangle_base_angle_l226_226921

theorem isosceles_triangle_base_angle
  (A B C A1 B1 C1: ℝ)
  (h_iso: A = B)
  (h_dual: (cos A / sin A1 = cos B / sin B1) ∧ 
           (cos B / sin B1 = cos C / sin C1) ∧ 
           (cos C / sin C1 = 1)) :
  A = B ∧ A = 3 * π / 8 := 
begin
  -- Proof is not required as per instruction
  sorry 
end

end isosceles_triangle_base_angle_l226_226921


namespace rectangle_area_l226_226972

theorem rectangle_area (x y : ℝ) (hx : x ≠ 0) (h : x * y = 10) : y = 10 / x :=
sorry

end rectangle_area_l226_226972


namespace probability_point_inside_circle_l226_226598

theorem probability_point_inside_circle :
  (∃ (m n : ℕ), 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) →
  (∃ (P : ℚ), P = 2/9) :=
by
  sorry

end probability_point_inside_circle_l226_226598


namespace magnitude_a_eq_3sqrt2_l226_226254

open Real

def a (x: ℝ) : ℝ × ℝ := (3, x)
def b : ℝ × ℝ := (-1, 1)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem magnitude_a_eq_3sqrt2 (x : ℝ) (h : perpendicular (a x) b) :
  ‖a 3‖ = 3 * sqrt 2 := by
  sorry

end magnitude_a_eq_3sqrt2_l226_226254


namespace recurring_decimal_product_l226_226088

theorem recurring_decimal_product (q : ℝ) (h : q = 1 / 3) : q * 12 = 4 :=
by {
  rw h,
  norm_num,
  sorry
}

end recurring_decimal_product_l226_226088


namespace system_solution_unique_l226_226564

theorem system_solution_unique
  (a b m n : ℝ)
  (h1 : a * 1 + b * 2 = 10)
  (h2 : m * 1 - n * 2 = 8) :
  (a / 2 * (4 + -2) + b / 3 * (4 - -2) = 10) ∧
  (m / 2 * (4 + -2) - n / 3 * (4 - -2) = 8) := 
  by
    sorry

end system_solution_unique_l226_226564


namespace smallest_x_l226_226405

theorem smallest_x (x : ℕ) : (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) → x = 59 :=
by
  intro h
  sorry

end smallest_x_l226_226405


namespace hyperbola_properties_l226_226126

noncomputable def hyperbola_asymptote_b (b : ℝ) (h_b_pos : b > 0) : Prop :=
  2 = b * 1

noncomputable def hyperbola_eccentricity (b : ℝ) (e : ℝ) (h_b_pos : b > 0) : Prop :=
  b = 2 ∧ e = Real.sqrt 5

theorem hyperbola_properties :
  ∃ (b e : ℝ) (h_b_pos : b > 0), hyperbola_asymptote_b b h_b_pos ∧ hyperbola_eccentricity b e h_b_pos :=
by
  use 2, Real.sqrt 5, by norm_num
  split
  · sorry
  · sorry

end hyperbola_properties_l226_226126


namespace value_of_expression_l226_226936

theorem value_of_expression (m : ℝ) (h : m^2 - m - 1 = 0) : m^2 - m + 5 = 6 :=
by
  sorry

end value_of_expression_l226_226936


namespace area_ratio_of_smaller_octagon_l226_226716

theorem area_ratio_of_smaller_octagon (A B C D E F G H P Q R S T U V W : Point) 
  (h1 : is_regular_octagon A B C D E F G H)
  (h2 : midpoint A B = P) (h3 : midpoint B C = Q) (h4 : midpoint C D = R)
  (h5 : midpoint D E = S) (h6 : midpoint E F = T) (h7 : midpoint F G = U)
  (h8 : midpoint G H = V) (h9 : midpoint H A = W):
  area (octagon A B C D E F G H) / area (octagon P Q R S T U V W) = 4 := sorry

end area_ratio_of_smaller_octagon_l226_226716


namespace constant_term_expansion_l226_226505

theorem constant_term_expansion (x : ℝ) : 
  (∃ r : ℕ, 2 * r - 6 = 0 ∧ binomial 6 r * (2^r : ℝ) = 160) :=
begin
  use 3,
  split,
  { norm_num },
  { norm_num, sorry }
end

end constant_term_expansion_l226_226505


namespace smallest_decimal_l226_226411

def binary_to_decimal (n : Nat) : Nat :=
  n.foldl (λ acc d => acc * 2 + d.toNat) 0

def base6_to_decimal (n : Nat) : Nat :=
  n.foldl (λ acc d => acc * 6 + d.toNat) 0

def base4_to_decimal (n : Nat) : Nat :=
  n.foldl (λ acc d => acc * 4 + d.toNat) 0

def base9_to_decimal (n : Nat) : Nat :=
  n.foldl (λ acc d => acc * 9 + d.toNat) 0

def n1 : Nat := binary_to_decimal [1, 1, 1, 1, 1, 1]
def n2 : Nat := base6_to_decimal [2, 1, 0]
def n3 : Nat := base4_to_decimal [1, 0, 0, 0]
def n4 : Nat := base9_to_decimal [8, 1]

theorem smallest_decimal :
  n1 < n2 ∧ n1 < n3 ∧ n1 < n4 :=
by
  sorry

end smallest_decimal_l226_226411


namespace Jenny_wants_to_read_three_books_l226_226622

noncomputable def books : Nat := 3

-- Definitions based on provided conditions
def reading_speed : Nat := 100 -- words per hour
def book1_words : Nat := 200 
def book2_words : Nat := 400
def book3_words : Nat := 300
def daily_reading_minutes : Nat := 54 
def days : Nat := 10

-- Derived definitions for the proof
def total_words : Nat := book1_words + book2_words + book3_words
def total_hours_needed : ℚ := total_words / reading_speed
def daily_reading_hours : ℚ := daily_reading_minutes / 60
def total_reading_hours : ℚ := daily_reading_hours * days

theorem Jenny_wants_to_read_three_books :
  total_reading_hours = total_hours_needed → books = 3 :=
by
  -- Proof goes here
  sorry

end Jenny_wants_to_read_three_books_l226_226622


namespace range_of_sqrt_expr_l226_226589

theorem range_of_sqrt_expr (x : ℝ) (h : ∃ y : ℝ, y = sqrt (x - 2)) : x ≥ 2 :=
by
  sorry

end range_of_sqrt_expr_l226_226589


namespace sum_to_product_l226_226759

theorem sum_to_product (x : ℝ) (n k : ℕ) : 
  x^(3*n + 2) + x^(3*k + 1) + 1 = 
  (x^2 + x + 1) * (x^2 * (x - 1) * (finset.range n).sum (λ i, x^(3*i)) + 
                   x * (x - 1) * (finset.range k).sum (λ i, x^(3*i)) + 1) :=
by sorry

end sum_to_product_l226_226759


namespace number_of_magic_numbers_lt_600_l226_226821

def is_magic_number (N : ℕ) : Prop :=
  ∀ (m : ℕ), N ∣ (m * 10^(Nat.log10 N + 1) + N)

def count_magic_numbers_below (limit : ℕ) : ℕ :=
  Nat.filter (λ N, is_magic_number N) (List.range' 1 limit).length

theorem number_of_magic_numbers_lt_600 : count_magic_numbers_below 600 = 13 :=
  sorry

end number_of_magic_numbers_lt_600_l226_226821


namespace smaller_octagon_area_half_l226_226706

theorem smaller_octagon_area_half
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (P Q R S T U V W : Point)
  (H1 : midpoint P A B)
  (H2 : midpoint Q B C)
  (H3 : midpoint R C D)
  (H4 : midpoint S D E)
  (H5 : midpoint T E F)
  (H6 : midpoint U F G)
  (H7 : midpoint V G H)
  (H8 : midpoint W H A):
  area (octagon P Q R S T U V W) = (1 / 2) * area (octagon ABCDEFGH) :=
sorry

end smaller_octagon_area_half_l226_226706


namespace cartesian_eq_C1_cartesian_eq_C2_range_distance_MN_min_value_function_min_value_expression_l226_226207

-- Part I
theorem cartesian_eq_C1 (φ : ℝ) : 
  (x = 2 * real.cos φ) ∧ (y = real.sin φ) → (x^2 / 4 + y^2 = 1) :=
by 
  sorry

theorem cartesian_eq_C2 : 
  (x^2 + (y - 3)^2 = 1) :=
by 
  sorry

theorem range_distance_MN (φ : ℝ): 
  (M = (2 * real.cos φ, real.sin φ)) ∧ (N ∈ set_of (λ q, (q.1^2 + (q.2 - 3)^2 = 1))) → (1 ≤ (dist M N) ∧ (dist M N) ≤ 5) :=
by 
  sorry

-- Part II
theorem min_value_function (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  (∀ x, (|x + a| + |x - b|) ≥ 4) → (a + b = 4) :=
by 
  sorry

theorem min_value_expression (a b : ℝ) (h : a + b = 4) : 
  (min (1 / 4 * a^2 + 1 / 4 * b^2) = 3 / 16) :=
by 
  sorry

end cartesian_eq_C1_cartesian_eq_C2_range_distance_MN_min_value_function_min_value_expression_l226_226207


namespace isosceles_triangle_length_l226_226925

variable (a b : ℝ)

theorem isosceles_triangle_length (h1 : 2 * a + 3 = 16) (h2 : a != 3) : a = 6.5 :=
sorry

end isosceles_triangle_length_l226_226925


namespace vika_pairs_exactly_8_ways_l226_226348

theorem vika_pairs_exactly_8_ways :
  ∃ d : ℕ, (d ∣ 30) ∧ (Finset.card (Finset.filter (λ d, d ∣ 30) (Finset.range 31)) = 8) := 
sorry

end vika_pairs_exactly_8_ways_l226_226348


namespace smaller_octagon_area_fraction_l226_226732

theorem smaller_octagon_area_fraction (A B C D E F G H : Point) (O : Point) :
  is_regular_octagon A B C D E F G H →
  is_center O A B C D E F G H →
  let A' := midpoint A B,
      B' := midpoint B C,
      C' := midpoint C D,
      D' := midpoint D E,
      E' := midpoint E F,
      F' := midpoint F G,
      G' := midpoint G H,
      H' := midpoint H A in
  is_octa_center O A' B' C' D' E' F' G' H' →
  (area_of_octagon A B C D E F G H) * (1 / 4) = area_of_octagon A' B' C' D' E' F' G' H' :=
by
  -- Sorry, proof is omitted.
  sorry

end smaller_octagon_area_fraction_l226_226732


namespace terminal_side_quadrant_l226_226932

-- Given conditions
variables {α : ℝ}
variable (h1 : Real.sin α > 0)
variable (h2 : Real.tan α < 0)

-- Conclusion to be proved
theorem terminal_side_quadrant (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  (∃ k : ℤ, (k % 2 = 0 ∧ Real.pi * k / 2 < α / 2 ∧ α / 2 < Real.pi / 2 + Real.pi * k) ∨ 
            (k % 2 = 1 ∧ Real.pi * (k - 1) < α / 2 ∧ α / 2 < Real.pi / 4 + Real.pi * (k - 0.5))) :=
by
  sorry

end terminal_side_quadrant_l226_226932


namespace length_RS_l226_226204

theorem length_RS
  (P Q R S : Type)
  (h_PQ_34 : dist P Q = 34)
  (h_PS_26 : dist P S = 26)
  (h_QR_40 : dist Q R = 40)
  (h_angle_bisector : ∠ P R S = ∠ Q R S):
  ∃ RS : ℝ, RS = 130 := 
sorry

end length_RS_l226_226204


namespace permutation_count_l226_226545

theorem permutation_count :
  {σ : List ℕ | σ = [a1, a2, a3, a4, a5] ∧
  Multiset.OfList σ = Multiset [1, 2, 3, 4, 5] ∧
  a1 < a2 ∧ a2 > a3 ∧ a3 < a4 ∧ a4 > a5 }.card = 16 := 
sorry

end permutation_count_l226_226545


namespace find_s_l226_226632

variable {a b n r s : ℝ}

theorem find_s (h1 : Polynomial.aeval a (Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 6) = 0)
              (h2 : Polynomial.aeval b (Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 6) = 0)
              (h_ab : a * b = 6)
              (h_roots : Polynomial.aeval (a + 2/b) (Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s) = 0)
              (h_roots2 : Polynomial.aeval (b + 2/a) (Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s) = 0) :
  s = 32/3 := 
sorry

end find_s_l226_226632


namespace card_paiting_modulus_l226_226365

theorem card_paiting_modulus (cards : Finset ℕ) (H : cards = Finset.range 61 \ {0}) :
  ∃ d : ℕ, ∀ n ∈ cards, ∃! k, (∀ x ∈ cards, (x + n ≡ k [MOD d])) ∧ (d ∣ 30) ∧ (∃! n : ℕ, 1 ≤ n ∧ n ≤ 8) :=
sorry

end card_paiting_modulus_l226_226365


namespace selection_ways_1200_l226_226984

/--
In an event, there are 30 people arranged in 6 rows and 5 columns. Now,
3 people are to be selected to perform a ceremony, with the requirement
that any two of these 3 people must not be in the same row or column.
The number of different ways to select these 3 people is 1200.
-/
theorem selection_ways_1200 :
  let rows := 6
  let columns := 5
  let select_3 := 3
  (Nat.choose rows select_3) * (Nat.choose columns select_3) * (Nat.factorial select_3) = 1200 :=
by
  let rows := 6
  let columns := 5
  let select_3 := 3
  have h1 : Nat.choose rows select_3 = 20 := by rw Nat.choose_eq_factorial_div_factorial
  have h2 : Nat.choose columns select_3 = 10 := by rw Nat.choose_eq_factorial_div_factorial
  have h3 : Nat.factorial select_3 = 6 := by rw Nat.factorial
  calc
  (Nat.choose rows select_3) * (Nat.choose columns select_3) * (Nat.factorial select_3)
      = 20 * 10 * 6 : by rw [h1, h2, h3]
  ... = 1200 : by norm_num

end selection_ways_1200_l226_226984


namespace log_base_identity_l226_226579

theorem log_base_identity (x : ℝ) (hx : log 8 (5 * x) = 3) : log x 125 = 3 * log 5 / (9 * log 2 - log 5) :=
by
  sorry

end log_base_identity_l226_226579


namespace problem_I_problem_II_problem_III_l226_226553

-- Definitions based on conditions
def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + log x

def g (x : ℝ) : ℝ := log (1 + x) - x

-- Problem (I): Prove \(a \geq 1\) given \( f \) is increasing on \( (1, +\infty) \)
theorem problem_I (a : ℝ) (h_pos : 0 < a) (h_inc : ∀ x : ℝ, 1 < x → 0 ≤ f a x) : 1 ≤ a := 
sorry

-- Problem (II): Prove the maximum value of \( g \) on \( [0, +\infty) \) is \( 0 \)
theorem problem_II : ∀ x : ℝ, 0 ≤ x → g x ≤ g 0 :=
sorry

-- Problem (III): Prove \( \frac{1}{a+b} \leq \ln \frac{a+b}{b} < \frac{a}{b} \)
theorem problem_III (a b : ℝ) (h_a : 1 < a) (h_b : 0 < b) : 
  1 / (a + b) ≤ log ((a + b) / b) ∧ log ((a + b) / b) < a / b :=
sorry

end problem_I_problem_II_problem_III_l226_226553


namespace solution_set_of_inequality_l226_226323

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / x ≤ 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x ≥ 1 / 2} :=
by
  sorry

end solution_set_of_inequality_l226_226323


namespace find_b_condition_l226_226263

variable (n : ℕ)
variable (a : Fin n → ℝ) -- Array a mapped to n elements which are real numbers
variable (q : ℝ) -- q as a real number

theorem find_b_condition
  (h_pos : ∀ k, 0 < a k)
  (h_q : 0 < q ∧ q < 1) :
  ∃ (b : Fin n → ℝ),
    (∀ k, a k < b k) ∧
    (∀ k : Fin (n-1), q < (b ⟨ k.1 + 1 , sorry ⟩ / b k) ∧ (b ⟨ k.1 + 1 , sorry ⟩ / b k) < 1 / q) ∧
    (∑ k, b k < (1 + q) / (1 - q) * ∑ k, a k) :=
begin
  sorry
end

end find_b_condition_l226_226263


namespace maximum_x_value_l226_226252

theorem maximum_x_value (x y z : ℝ) (h1 : x + y + z = 10) (h2 : x * y + x * z + y * z = 20) : 
  x ≤ 10 / 3 := sorry

end maximum_x_value_l226_226252


namespace log_geo_seq_l226_226528

theorem log_geo_seq (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 3 * a n) (h2 : a 2 + a 4 + a 6 = 9) :
  log (1/3) (a 5 + a 7 + a 9) = -5 :=
sorry

end log_geo_seq_l226_226528


namespace slope_of_line_l226_226070

theorem slope_of_line (x y : ℝ) :
  (∀ (x y : ℝ), (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5 / 4)) :=
by
  sorry

end slope_of_line_l226_226070


namespace hyperbola_same_foci_ellipse_l226_226507

-- Define the equation of the original ellipse
def ellipse_equation := ∀ x y : ℝ, (x^2 / 49) + (y^2 / 24) = 1

-- Define that c is the distance of the foci for the ellipse
def foci_distance (c : ℝ) := c = 5

-- Define the eccentricity of the hyperbola
def hyperbola_eccentricity (e : ℝ) := e = 5 / 4

-- Define that a is the denominator of x^2 for the derived hyperbola equation
def a_value (a : ℝ) := a = 4

-- Define that b^2 is the difference of c^2 and a^2 for the hyperbola
def b_squared (b : ℝ) := b^2 = 25 - a^2

-- The target: the equation of the hyperbola
def hyperbola_equation := (x^2 / 16) - (y^2 / 9) = 1

-- Main theorem statement
theorem hyperbola_same_foci_ellipse : ∀ (x y a b c : ℝ), 
    ellipse_equation x y → 
    foci_distance c → 
    hyperbola_eccentricity (c / a) → 
    a_value a → 
    b_squared b → 
    hyperbola_equation := 
by sorry

end hyperbola_same_foci_ellipse_l226_226507


namespace magic_sum_order_8_l226_226432

def is_magic_square (n : ℕ) (S : list (list ℕ)) : Prop :=
  let indices := list.fin_range n in
  let row_sums := indices.map (λ i, (indices.map (λ j, S.nth_le i (nat.lt_of_lt n n))).sum) in
  let col_sums := indices.map (λ j, (indices.map (λ i, S.nth_le i (nat.lt_of_lt n n))).sum) in
  let diag1_sum := (indices.map (λ i, S.nth_le i (nat.lt_of_lt n n))).sum in
  let diag2_sum := (indices.map (λ i, S.nth_le i (nat.lt_of_lt n n).reverse)).sum in
  ∀ x ∈ row_sums, x = diag1_sum ∧ diag1_sum = diag2_sum ∧ diag2_sum = col_sums.head

theorem magic_sum_order_8 : 
  let n := 8
  ∃ S, is_magic_square n S ∧ 
       (n * (n^2 + 1) / 2 = 260) := by
  let k := 64
  let sum := k * (k + 1) / 2
  let M := sum / 8
  calc M = 260 : by sorry

end magic_sum_order_8_l226_226432


namespace product_stu_eq_18_l226_226304

theorem product_stu_eq_18 {a x y c : ℤ} :
  (∃ s t u : ℤ, (a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1)) ∧
                 ((a^s * x - a^t) * (a^u * y - a^3) = a^5 * c^5) ∧
                 (s * t * u = 18)) :=
begin
  sorry
end

end product_stu_eq_18_l226_226304


namespace exists_multiple_of_ones_l226_226641

-- Problem statement in Lean 4
theorem exists_multiple_of_ones (n : ℕ) (hn : Nat.coprime n 10) : 
  ∃ k m : ℕ, m = (10^k - 1) / 9 ∧ n ∣ m := 
sorry

end exists_multiple_of_ones_l226_226641


namespace original_number_l226_226335

theorem original_number (x : ℝ) (h : 20 = 0.4 * (x - 5)) : x = 55 :=
sorry

end original_number_l226_226335


namespace last_digit_p_adic_l226_226645

theorem last_digit_p_adic (a : ℤ) (p : ℕ) (hp : Nat.Prime p) (h_last_digit_nonzero : a % p ≠ 0) : (a ^ (p - 1) - 1) % p = 0 :=
by
  sorry

end last_digit_p_adic_l226_226645


namespace inequality_proof_l226_226282

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) : 
  a^4 + b^4 + c^4 ≥ a * b * c * (a + b + c) := 
by 
  sorry

end inequality_proof_l226_226282


namespace volumes_equal_l226_226325

def region1 := { p : ℝ × ℝ // (p.1 ^ 2 = 4 * p.2 ∨ p.1 ^ 2 = -4 * p.2) ∧ -4 ≤ p.1 ∧ p.1 ≤ 4 }
def region2 := { p : ℝ × ℝ // p.1 ^ 2 * p.2 ^ 2 ≤ 16 ∧ (p.1 ^ 2 + (p.2 - 2) ^ 2 ≥ 4) ∧ (p.1 ^ 2 + (p.2 + 2) ^ 2 ≥ 4) }

noncomputable def volume_of_revolution (region : ℝ × ℝ → Prop) (axis : ℝ × ℝ) : ℝ := sorry

theorem volumes_equal :
  volume_of_revolution (λ p, p ∈ region1) (0, 1) = volume_of_revolution (λ p, p ∈ region2) (0, 1) :=
sorry

end volumes_equal_l226_226325


namespace find_a3_minus_b3_l226_226939

theorem find_a3_minus_b3 (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 47) : a^3 - b^3 = 322 :=
by
  sorry

end find_a3_minus_b3_l226_226939


namespace problem_l226_226122

theorem problem (
  h1: - real.pi / 2 < β ∧ β < 0 ∧ 0 < α ∧ α < real.pi / 2,
  h2: real.cos (real.pi / 4 + α) = 1 / 3,
  h3: real.cos (real.pi / 4 - β / 2) = real.sqrt 3 / 3
) : real.cos (α + β / 2) = 5 * real.sqrt 3 / 9 :=
sorry

end problem_l226_226122


namespace altitude_of_airplane_is_correct_l226_226021

-- Define the conditions
def distance_AB : ℝ := 12
def angle_elevation_alice : ℝ := Real.pi / 4  -- 45 degrees in radians
def angle_elevation_bob : ℝ := Real.pi / 6    -- 30 degrees in radians

-- The intermediate steps are implicitly assumed to be used in the proof
theorem altitude_of_airplane_is_correct : 
  ∃ h : ℝ, (tan angle_elevation_alice = 1) ∧ (tan angle_elevation_bob = 1 / Real.sqrt 3) ∧ 
  (12 - h) * Real.tan (Real.pi / 6) = h → h = 5.2 := 
by
  sorry

end altitude_of_airplane_is_correct_l226_226021


namespace good_carrots_l226_226157

theorem good_carrots (haley_picked : ℕ) (mom_picked : ℕ) (bad_carrots : ℕ) :
  haley_picked = 39 → mom_picked = 38 → bad_carrots = 13 →
  (haley_picked + mom_picked - bad_carrots) = 64 :=
by
  sorry  -- Proof is omitted.

end good_carrots_l226_226157


namespace min_cubes_l226_226065

-- Define the conditions as properties
structure FigureViews :=
  (front_view : ℕ)
  (side_view : ℕ)
  (top_view : ℕ)
  (adjacency_requirement : Bool)

-- Define the given views
def given_views : FigureViews := {
  front_view := 3,  -- as described: 2 cubes at bottom + 1 on top
  side_view := 3,   -- same as front view
  top_view := 3,    -- L-shape consists of 3 cubes
  adjacency_requirement := true
}

-- The theorem to state that the minimum number of cubes is 3
theorem min_cubes (views : FigureViews) : views.front_view = 3 ∧ views.side_view = 3 ∧ views.top_view = 3 ∧ views.adjacency_requirement = true → ∃ n, n = 3 :=
by {
  sorry
}

end min_cubes_l226_226065


namespace freshman_to_sophomore_ratio_l226_226031

variables (f s : ℕ)
hypothesis h1 : 3 / 7 * f = 5 / 7 * s

theorem freshman_to_sophomore_ratio (f s : ℕ) (h1 : 3 * f = 5 * s) : f = 5 * s / 3 :=
by
  sorry

end freshman_to_sophomore_ratio_l226_226031


namespace card_paiting_modulus_l226_226364

theorem card_paiting_modulus (cards : Finset ℕ) (H : cards = Finset.range 61 \ {0}) :
  ∃ d : ℕ, ∀ n ∈ cards, ∃! k, (∀ x ∈ cards, (x + n ≡ k [MOD d])) ∧ (d ∣ 30) ∧ (∃! n : ℕ, 1 ≤ n ∧ n ≤ 8) :=
sorry

end card_paiting_modulus_l226_226364


namespace fill_table_condition_l226_226467

theorem fill_table_condition (n : ℕ) (hpos : 0 < n) : 
  ∃ (f : ℕ → ℕ → ℕ), (∀ i j : ℕ, i < n ∧ j < n → 1 ≤ f i j ∧ f i j ≤ n^2) ∧ 
  (∀ i : ℕ, i < n → n ∣ (finset.sum (finset.range n) (λ j, f i j))) ∧ 
  (∀ j : ℕ, j < n → n ∣ (finset.sum (finset.range n) (λ i, f i j))) :=
sorry

end fill_table_condition_l226_226467


namespace smaller_octagon_area_half_l226_226708

theorem smaller_octagon_area_half
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (P Q R S T U V W : Point)
  (H1 : midpoint P A B)
  (H2 : midpoint Q B C)
  (H3 : midpoint R C D)
  (H4 : midpoint S D E)
  (H5 : midpoint T E F)
  (H6 : midpoint U F G)
  (H7 : midpoint V G H)
  (H8 : midpoint W H A):
  area (octagon P Q R S T U V W) = (1 / 2) * area (octagon ABCDEFGH) :=
sorry

end smaller_octagon_area_half_l226_226708


namespace geometric_sequence_seventh_term_l226_226887

theorem geometric_sequence_seventh_term (a₁ : ℤ) (a₂ : ℚ) (r : ℚ) (k : ℕ) (a₇ : ℚ)
  (h₁ : a₁ = 3) 
  (h₂ : a₂ = -1 / 2)
  (h₃ : r = a₂ / a₁)
  (h₄ : k = 7)
  (h₅ : a₇ = a₁ * r^(k-1)) : 
  a₇ = 1 / 15552 := 
by
  sorry

end geometric_sequence_seventh_term_l226_226887


namespace ways_to_select_3_people_l226_226987

theorem ways_to_select_3_people : 
  let ways := nat.choose 6 3 * nat.choose 5 3 * 6
  in ways = 1200 :=
by
  let ways := nat.choose 6 3 * nat.choose 5 3 * 6
  show ways = 1200
  sorry

end ways_to_select_3_people_l226_226987


namespace commodity_X_costs_70_cents_more_than_Y_in_2010_l226_226738

noncomputable def price_X (n : ℕ) : ℝ := 4.20 + 0.30 * n
noncomputable def price_Y (n : ℕ) : ℝ := 4.40 + 0.20 * n

theorem commodity_X_costs_70_cents_more_than_Y_in_2010 :
  ∃ n : ℕ, (price_X n = price_Y n + 0.70) ∧ (2001 + n = 2010) :=
by
  use 9
  unfold price_X price_Y
  norm_num
  sorry

end commodity_X_costs_70_cents_more_than_Y_in_2010_l226_226738


namespace number_of_true_propositions_l226_226929

open Classical

axiom real_numbers (a b : ℝ): Prop

noncomputable def original_proposition (a b : ℝ) : Prop := a > b → a * abs a > b * abs b
noncomputable def converse_proposition (a b : ℝ) : Prop := a * abs a > b * abs b → a > b
noncomputable def negation_proposition (a b : ℝ) : Prop := a ≤ b → a * abs a ≤ b * abs b
noncomputable def contrapositive_proposition (a b : ℝ) : Prop := a * abs a ≤ b * abs b → a ≤ b

theorem number_of_true_propositions (a b : ℝ) (h₁: original_proposition a b) 
  (h₂: converse_proposition a b) (h₃: negation_proposition a b)
  (h₄: contrapositive_proposition a b) : ∃ n, n = 4 := 
by
  -- The proof would go here, proving that ∃ n, n = 4 is true.
  sorry

end number_of_true_propositions_l226_226929


namespace find_principal_amount_l226_226027

-- Define the conditions as constants and assumptions
def monthly_interest_payment : ℝ := 216
def annual_interest_rate : ℝ := 0.09

-- Define the Lean statement to show that the amount of the investment is 28800
theorem find_principal_amount (monthly_payment : ℝ) (annual_rate : ℝ) (P : ℝ) :
  monthly_payment = 216 →
  annual_rate = 0.09 →
  P = 28800 :=
by
  intros 
  sorry

end find_principal_amount_l226_226027


namespace phi_cannot_be_5pi_over_4_l226_226309

noncomputable def f (x θ : ℝ) : ℝ := 3 * Real.sin (2 * x + θ)
noncomputable def g (x θ φ : ℝ) : ℝ := 3 * Real.sin (2 * x + θ - 2 * φ)
def point_P : ℝ × ℝ := (0, 3 * Real.sqrt 2 / 2)

theorem phi_cannot_be_5pi_over_4 
  (θ φ : ℝ) 
  (h1 : -Real.pi / 2 < θ ∧ θ < Real.pi / 2)
  (h2 : f 0 θ = point_P.2)
  (h3 : g 0 θ φ = point_P.2) :
  φ ≠ 5 * Real.pi / 4 :=
sorry

end phi_cannot_be_5pi_over_4_l226_226309


namespace polynomial_characterization_l226_226873
open Polynomial

noncomputable def satisfies_functional_eq (P : Polynomial ℝ) :=
  ∀ (a b c : ℝ), 
  P.eval (a + b - 2*c) + P.eval (b + c - 2*a) + P.eval (c + a - 2*b) = 
  3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)

theorem polynomial_characterization (P : Polynomial ℝ) :
  satisfies_functional_eq P ↔ 
  (∃ a b : ℝ, P = Polynomial.C a * Polynomial.X + Polynomial.C b) ∨
  (∃ a b : ℝ, P = Polynomial.C a * Polynomial.X^2 + Polynomial.C b * Polynomial.X) :=
sorry

end polynomial_characterization_l226_226873


namespace find_vec_c_find_cos_theta_l226_226569

-- Definition for our vectors
def vec_a : ℝ × ℝ := (1, -2)
axiom vec_b : ℝ × ℝ
axiom vec_c : ℝ × ℝ

-- Conditions for the vectors
axiom mag_vec_c : real.norm (vec_c.1, vec_c.2) = 2 * real.sqrt 5
axiom parallel_vec_c_a : ∃ k : ℝ, vec_c = (k * vec_a.1, k * vec_a.2)
axiom mag_vec_b : real.norm (vec_b.1, vec_b.2) = 1
axiom perpendicular_a_plus_b_a_minus_2b : 
  inner ((vec_a.1 + vec_b.1, vec_a.2 + vec_b.2)) ((vec_a.1 - 2 * vec_b.1, vec_a.2 - 2 * vec_b.2)) = 0

-- The proofs we need
theorem find_vec_c : vec_c = (-2, 4) ∨ vec_c = (2, -4) :=
sorry

theorem find_cos_theta : real.cos_angle vec_a vec_b = 3 / real.sqrt 5 :=
sorry

end find_vec_c_find_cos_theta_l226_226569


namespace ordered_triples_zero_satisfy_l226_226066

theorem ordered_triples_zero_satisfy (
    x y z : ℤ) 
    (h1 : x^2 - 2 * x * y + 3 * y^2 - 2 * z^2 = 25)
    (h2 : -x^2 + 4 * y * z + 3 * z^2 = 55)
    (h3 : x^2 + 3 * x * y - y^2 + 7 * z^2 = 130) :
    (x = 0 ∧ y = 0 ∧ z = 0) := 
begin
    sorry
end

end ordered_triples_zero_satisfy_l226_226066


namespace M_union_N_equals_M_l226_226644

def setM : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1 * p.2 = 1 ∧ p.1 > 0 }
def setN : Set (ℝ × ℝ) := { p : ℝ × ℝ | Real.arctan p.1 + Real.arccot p.2 = Real.pi }

theorem M_union_N_equals_M : setM ∪ setN = setM := by
  sorry

end M_union_N_equals_M_l226_226644


namespace function_correct_max_min_values_l226_226953

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

@[simp]
theorem function_correct : (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 4)) ∧ 
                           (f (3 * Real.pi / 8) = 0) ∧ 
                           (f (Real.pi / 8) = 2) :=
by
  sorry

theorem max_min_values : (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
                          f x = -2) ∧ 
                         (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
                          f x = 2) :=
by
  sorry

end function_correct_max_min_values_l226_226953


namespace triangle_ratio_l226_226220

theorem triangle_ratio 
  (A B C D E : Type) 
  [point A] [point B] [point C] [point D] [point E] 
  [triangle ABC : triangle A B C]
  (angle_A : ∠ A = 60)
  (angle_B : ∠ B = 45)
  (D_on_AB : is_on_line D A B)
  (angle_ADE : ∠ (D E) = 45)
  (equal_area : area (triangle D A E) = (1/2) * area (triangle A B C)) :
  ratio (length D A / length A B) = 1 / (2 + sqrt 3) :=
by sorry

end triangle_ratio_l226_226220


namespace flour_needed_for_bread_l226_226658

-- Definitions based on conditions
def flour_per_loaf : ℝ := 2.5
def number_of_loaves : ℕ := 2

-- Theorem statement
theorem flour_needed_for_bread : flour_per_loaf * number_of_loaves = 5 :=
by sorry

end flour_needed_for_bread_l226_226658


namespace smallest_x_l226_226401

theorem smallest_x (x : ℕ) (h₁ : x % 3 = 2) (h₂ : x % 4 = 3) (h₃ : x % 5 = 4) : x = 59 :=
by
  sorry

end smallest_x_l226_226401


namespace Lance_daily_earnings_l226_226072

theorem Lance_daily_earnings :
  ∀ (hours_per_week : ℕ) (workdays_per_week : ℕ) (hourly_rate : ℕ) (total_earnings : ℕ) (daily_earnings : ℕ),
  hours_per_week = 35 →
  workdays_per_week = 5 →
  hourly_rate = 9 →
  total_earnings = hours_per_week * hourly_rate →
  daily_earnings = total_earnings / workdays_per_week →
  daily_earnings = 63 := 
by
  intros hours_per_week workdays_per_week hourly_rate total_earnings daily_earnings
  intros H1 H2 H3 H4 H5
  sorry

end Lance_daily_earnings_l226_226072


namespace maximum_r_squared_sum_is_2194_l226_226762

noncomputable def maximum_possible_r_squared_cones_sphere : ℚ := 
  (45 / 13)^2

theorem maximum_r_squared_sum_is_2194 
  (radius_base_of_cones height_of_cones intersection_distance : ℕ)
  (h1 : radius_base_of_cones = 5)
  (h2 : height_of_cones = 12)
  (h3 : intersection_distance = 4) :
  let r_squared := maximum_possible_r_squared_cones_sphere in
  let m := r_squared.num in
  let n := r_squared.denom in
  (m + n : ℕ) = 2194 :=
sorry

end maximum_r_squared_sum_is_2194_l226_226762


namespace horizontal_asymptote_l226_226064

theorem horizontal_asymptote :
  ∀ x: ℝ, lim (λ x, (6*x^2 + 4) / (4*x^2 + 3*x + 1)) at_top = 1.5 :=
begin
  sorry
end

end horizontal_asymptote_l226_226064


namespace joe_total_time_l226_226623

theorem joe_total_time (t_w : ℕ) (r_r : ℕ → ℕ) (t_break : ℕ) (t_r : ℕ) (t_total : ℕ) :
  t_w = 8 ∧ r_r = (λ r_w, 4 * r_w) ∧ t_break = 1 ∧ t_r = t_w / 4 ∧ t_total = t_w + t_break + t_r →
  t_total = 11 :=
sorry

end joe_total_time_l226_226623


namespace Ava_watch_minutes_l226_226470

theorem Ava_watch_minutes (hours_watched : ℕ) (minutes_per_hour : ℕ) (h : hours_watched = 4) (m : minutes_per_hour = 60) : 
  hours_watched * minutes_per_hour = 240 :=
by
  sorry

end Ava_watch_minutes_l226_226470


namespace cost_price_of_article_l226_226012

-- Given that the selling price (SP) is 100 and the profit is 30% of CP, prove that CP is approximately 76.92
theorem cost_price_of_article (SP : ℝ) (hSP : SP = 100) (profit_percentage : ℝ) (hprofit_percentage : profit_percentage = 0.30) :
  ∃ CP : ℝ, SP = CP + profit_percentage * CP ∧ CP ≈ 76.92 :=
by
  sorry

end cost_price_of_article_l226_226012


namespace BCDE_is_parallelogram_l226_226910

-- Define the points of the pentagon
variables (A B C D E : Point)
-- Define angles
variables {α β : ℝ}
variables (BAC ABE DEA BCA ADE : ℝ)
-- Define lengths
variables (BC ED: ℝ)

-- Conditions from the problem
def angle_condition_1 := BAC = ABE ∧ ABE = DEA - 90
def angle_condition_2 := BCA = ADE
def length_condition := BC = ED

-- Prove that BCDE is a parallelogram
theorem BCDE_is_parallelogram
  (h1 : angle_condition_1)
  (h2 : angle_condition_2)
  (h3 : length_condition) :
  is_parallelogram B C D E :=
by
  sorry

end BCDE_is_parallelogram_l226_226910


namespace max_expression_value_l226_226940

theorem max_expression_value (n : ℕ) (hn : 2 ≤ n) (a : ℕ → ℝ) (ha : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ a i) :
  (∑ i in finset.range n, (i + 1) * a (i + 1)) * (∑ i in finset.range n, a (i + 1) / (i + 1)) / 
    (∑ i in finset.range n, a (i + 1))^2 ≤ 1 :=
by
  sorry

end max_expression_value_l226_226940


namespace price_of_each_bracelet_l226_226334

-- The conditions
def bike_cost : ℕ := 112
def days_in_two_weeks : ℕ := 14
def bracelets_per_day : ℕ := 8
def total_bracelets := days_in_two_weeks * bracelets_per_day

-- The question and the expected answer
def price_per_bracelet : ℕ := bike_cost / total_bracelets

theorem price_of_each_bracelet :
  price_per_bracelet = 1 := 
by
  sorry

end price_of_each_bracelet_l226_226334


namespace simplify_polynomial_value_of_A_if_l226_226560

def polynomial (x : ℝ) : ℝ := (x + 2)^2 + (1 - x) * (2 + x) - 3

theorem simplify_polynomial (x : ℝ):
  polynomial x = 3 * x + 3 :=
by
  -- Provide the simplification proof here
  sorry

theorem value_of_A_if (x : ℝ) (h : (x + 1)^2 = 6):
  polynomial x = 3 * real.sqrt 6 ∨ polynomial x = -3 * real.sqrt 6 :=
by
  -- Provide the proof for the given condition here
  sorry

end simplify_polynomial_value_of_A_if_l226_226560


namespace integer_solutions_count_l226_226174

theorem integer_solutions_count :
  ∃ (s : Finset (ℤ × ℤ)), (∀ (x y : ℤ), (6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0) ↔ ((x, y) ∈ s)) ∧ s.card = 4 :=
begin
  sorry
end

end integer_solutions_count_l226_226174


namespace product_numerator_denominator_of_0_027_l226_226382

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  have h : x = 27 / 999 := by
    have h1 : 1000 * x = 999 * x + 27 := sorry
    exact h
  h.symm ▸ (1 / 37 : ℚ)

theorem product_numerator_denominator_of_0_027 :
  ∀ x : ℚ, x = 0.027^(999) → (let f := repeating_decimal_to_fraction x in (f.num * f.denom) = 37) :=
by
  intro x h
  have frac_def := repeating_decimal_to_fraction x
  rw h at frac_def
  have simplified_frac : frac_def = 1 / 37 := sorry
  rw simplified_frac
  norm_num

end product_numerator_denominator_of_0_027_l226_226382


namespace angle_A_is_right_l226_226262

theorem angle_A_is_right 
  (A B C D : Type)
  [triangle ABC]
  (D_midpoint_BC : midpoint D B C)
  (l : line)
  (bisector_ADC : is_angle_bisector l (∠ADC))
  (tangent_circumcircle_ABD : is_tangent l (circumcircle ABD) D) :
  ∠A = 90 :=
by
  sorry

end angle_A_is_right_l226_226262


namespace find_older_friend_age_l226_226302

theorem find_older_friend_age (A B C : ℕ) 
  (h1 : A - B = 2) 
  (h2 : A - C = 5) 
  (h3 : A + B + C = 110) : 
  A = 39 := 
by 
  sorry

end find_older_friend_age_l226_226302


namespace find_angle_l226_226316

noncomputable def angle_between_generatrix_and_base (k : ℝ) (h : k > 3 / 2) : ℝ :=
  real.arctan ((2 : ℝ) / real.sqrt (2 * k - 3))

-- Theorem
theorem find_angle (k : ℝ) (h : k > 3 / 2) : 
  angle_between_generatrix_and_base k h = real.arctan ((2 : ℝ) / real.sqrt (2 * k - 3)) :=
sorry

end find_angle_l226_226316


namespace complement_union_complement_l226_226150

open Set

def A : Set ℝ := {x | 1 ≤ 2^(x - 3) ∧ 2^(x - 3) < 16}
def B : Set ℝ := {x | Real.log x / Real.log 2 - 2 < 3}

theorem complement_union_complement (x : ℝ) :
  (x ∈ compl (A ∪ B) ↔ x ∈ (-∞, 2] ∪ [10, +∞)) ∧
  (x ∈ compl (A ∩ B) ↔ x ∈ (-∞, 3) ∪ [7, +∞)) ∧
  (x ∈ (compl A) ∩ B ↔ x ∈ (2, 3) ∪ [7, 10)) :=
sorry

end complement_union_complement_l226_226150


namespace sum_yk_eq_2_pow_p_minus_1_l226_226250

noncomputable def y_seq (p : ℕ) : ℕ → ℕ 
| 0       => 1
| 1       => p + 1
| (k + 2) => ((p - 1) * y_seq p (k + 1) - (p - k - 1) * y_seq p k) / (k + 1)

theorem sum_yk_eq_2_pow_p_minus_1 (p : ℕ) (hp : 0 < p) : 
  ∑ k in Finset.range (p+1), y_seq p k = 2 ^ (p - 1) :=
begin
  sorry
end

end sum_yk_eq_2_pow_p_minus_1_l226_226250


namespace abs_ineq_no_solution_l226_226974

theorem abs_ineq_no_solution (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 1| - |x - 2| ≥ a^2 + a + 1)) ↔ a ∈ Iio (-1) ∪ Ioi 0 := by
  sorry

end abs_ineq_no_solution_l226_226974


namespace tetrahedron_volume_eq_three_l226_226843

noncomputable def volume_of_tetrahedron : ℝ :=
  let PQ := 3
  let PR := 4
  let PS := 5
  let QR := 5
  let QS := Real.sqrt 34
  let RS := Real.sqrt 41
  have := (PQ = 3) ∧ (PR = 4) ∧ (PS = 5) ∧ (QR = 5) ∧ (QS = Real.sqrt 34) ∧ (RS = Real.sqrt 41)
  3

theorem tetrahedron_volume_eq_three : volume_of_tetrahedron = 3 := 
by { sorry }

end tetrahedron_volume_eq_three_l226_226843


namespace cos_alpha_value_l226_226540
noncomputable theory

-- Define the angle and point where the terminal side intersects the unit circle
variables (α : ℝ) 

-- Points on the unit circle
def P : ℝ × ℝ := (- (real.sqrt 3) / 2, -1 / 2)

-- Define cos alpha based on the given point
def cos_alpha := P.1

-- The theorem to prove
theorem cos_alpha_value : cos_alpha = - (real.sqrt 3) / 2 :=
by sorry

end cos_alpha_value_l226_226540


namespace tangent_line_problem_l226_226975

-- Definitions of the functions involved
def f (x : ℝ) : ℝ := x^3
def g (a x : ℝ) : ℝ := a * x^2 + (15 / 4) * x - 9

-- Function to check if a line is tangent at a point on a curve
def is_tangent (l : ℝ → ℝ) (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ (m b : ℝ), l = λ x, m * x + b ∧ m = Real.deriv f x₀ ∧ l x₀ = f x₀

-- Statement of the math proof problem
theorem tangent_line_problem :
  ∃ a : ℝ, (a = -1 ∨ a = -25 / 64) ∧
  ∃ l : ℝ → ℝ, is_tangent l f 1 ∧ is_tangent l (g a) 1 :=
sorry

end tangent_line_problem_l226_226975


namespace smaller_octagon_area_half_l226_226709

theorem smaller_octagon_area_half
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (P Q R S T U V W : Point)
  (H1 : midpoint P A B)
  (H2 : midpoint Q B C)
  (H3 : midpoint R C D)
  (H4 : midpoint S D E)
  (H5 : midpoint T E F)
  (H6 : midpoint U F G)
  (H7 : midpoint V G H)
  (H8 : midpoint W H A):
  area (octagon P Q R S T U V W) = (1 / 2) * area (octagon ABCDEFGH) :=
sorry

end smaller_octagon_area_half_l226_226709


namespace inequality_solution_set_l226_226960

theorem inequality_solution_set {m n : ℝ} (h : ∀ x : ℝ, -3 < x ∧ x < 6 ↔ x^2 - m * x - 6 * n < 0) : m + n = 6 :=
by
  sorry

end inequality_solution_set_l226_226960


namespace problems_remaining_l226_226016

variables (M S E : ℕ)
variables (math_problems_per_worksheet science_problems_per_worksheet english_problems_per_worksheet : ℕ)
variables (math_worksheets science_worksheets english_worksheets : ℕ)
variables (graded_math_worksheets graded_science_worksheets graded_english_worksheets : ℕ)

theorem problems_remaining :
  let total_problems (worksheets problems_per_worksheet : ℕ) := worksheets * problems_per_worksheet in
  let graded_problems (graded_worksheets problems_per_worksheet : ℕ) := graded_worksheets * problems_per_worksheet in
  let remaining_problems (worksheets graded_worksheets problems_per_worksheet : ℕ) := 
    total_problems worksheets problems_per_worksheet - graded_problems graded_worksheets problems_per_worksheet in
  remaining_problems math_worksheets graded_math_worksheets math_problems_per_worksheet +
  remaining_problems science_worksheets graded_science_worksheets science_problems_per_worksheet +
  remaining_problems english_worksheets graded_english_worksheets english_problems_per_worksheet = 84 :=
begin
  -- Define the problem specific variables
  let math_problems_per_worksheet := 5,
  let science_problems_per_worksheet := 3,
  let english_problems_per_worksheet := 7,
  let math_worksheets := 10,
  let science_worksheets := 15,
  let english_worksheets := 12,
  let graded_math_worksheets := 6,
  let graded_science_worksheets := 10,
  let graded_english_worksheets := 5,
  
  -- Express the remaining problems for each subject and prove the result
  have total_math_problems := 50,
  have total_science_problems := 45,
  have total_english_problems := 84,
  have graded_math_problems := 30,
  have graded_science_problems := 30,
  have graded_english_problems := 35,
  show 20 + 15 + 49 = 84,
  
  -- Verify the calculation
  sorry
end

end problems_remaining_l226_226016


namespace tangent_line_at_a_zero_max_value_at_positive_a_sum_inequality_l226_226958

namespace TangentLine

def f (x : ℝ) (a : ℝ) := log x - (1 / 2) * a * x^2 + x

theorem tangent_line_at_a_zero : 
  (∀ x : ℝ, f x 0 = log x + x) → 
  (∀ x : ℝ, deriv (f x 0) x = (1 / x) + 1) → 
  (f 1 0 = 1) → 
  (deriv (f x 0) 1 = 2) → 
  ∃ y : ℝ, 2 * x - y - 1 = 0 :=
sorry

end TangentLine

namespace MaxValue

def g (x : ℝ) (a : ℝ) := log x - (1 / 2) * a * x^2 + (1 - a) * x + 1

theorem max_value_at_positive_a (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, g x a = log x - (1 / 2) * a * x^2 + (1 - a) * x + 1 → 
  ∃ y : ℝ, y = g (1 / a) a ∧ y = log (1 / a) - (1 / a) :=
sorry

end MaxValue

namespace SumInequality

def f (x : ℝ) := log x - x^2 + x

theorem sum_inequality (x1 x2 : ℝ) :
  (a = -2) → 
  (x1 > 0) → 
  (x2 > 0) → 
  (f x1 + f x2 + x1 * x2 = 0) → 
  x1 + x2 ≥ (sqrt 5 - 1) / 2 :=
sorry

end SumInequality

end tangent_line_at_a_zero_max_value_at_positive_a_sum_inequality_l226_226958


namespace pipe_fill_time_without_leak_l226_226819

theorem pipe_fill_time_without_leak :
  (∃ T : ℝ, 
    let pipe_fill_rate := 1 / T,
    let leak_rate := 1 / 18,
    let combined_rate := pipe_fill_rate - leak_rate,
    combined_rate = 1 / 9) → T = 6 :=
begin
  sorry
end

end pipe_fill_time_without_leak_l226_226819


namespace number_of_consistent_k_configurations_l226_226294

theorem number_of_consistent_k_configurations {A : Type*} (n k : ℕ) (h1 : fintype.card A = n) (h2 : k ∣ n) :
  ∃ (count : ℕ), count = n.factorial / ((n/k).factorial * (k.factorial)^(n/k)) :=
begin
  sorry
end

end number_of_consistent_k_configurations_l226_226294


namespace not_possible_distances_l226_226918

-- Define the square and its geometric properties
structure Square (α : Type*) :=
  (A B C D : α → α → Prop)
  (side_length : ℝ)
  (diag_length : ℝ := side_length * Real.sqrt 2)
  (P : α → α → Prop)

-- Conditions of the problem
variables {α : Type*} [MetricSpace α]
variables (S : Square α)
variables {PA PB PC PD : ℝ}
variables {P : α}

-- Define the distances from point P to vertices of the square
def dist_to_vertices := [PA, PB, PC, PD]

-- The main statement to prove
theorem not_possible_distances (h : set.dist P S.A = 1)
                              (hS1 : set.dist P S.B = 1)
                              (hS2 : set.dist P S.C = 2)
                              (hS3 : set.dist P S.D = 3) :
  false := 
begin
  sorry
end

end not_possible_distances_l226_226918


namespace sqrt_domain_l226_226588

theorem sqrt_domain (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (x - 2)) ↔ (x ≥ 2) :=
by
  -- proof goes here
  sorry

end sqrt_domain_l226_226588


namespace share_difference_l226_226833

theorem share_difference (p q r : ℕ) (x : ℕ) (h_ratio : p = 3 * x ∧ q = 7 * x ∧ r = 12 * x)
  (h_diff_qr : q - r = 5500) : q - p = 4400 :=
by
  sorry

end share_difference_l226_226833


namespace radius_increase_area_triple_l226_226681

theorem radius_increase_area_triple (r m : ℝ) (h : π * (r + m)^2 = 3 * π * r^2) : 
  r = (m * (Real.sqrt 3 - 1)) / 2 := 
sorry

end radius_increase_area_triple_l226_226681


namespace domain_f_2x_plus_1_eq_l226_226134

-- Conditions
def domain_fx_plus_1 : Set ℝ := {x : ℝ | -2 < x ∧ x < -1}

-- Question and Correct Answer
theorem domain_f_2x_plus_1_eq :
  (∃ (x : ℝ), x ∈ domain_fx_plus_1) →
  {x : ℝ | -1 < x ∧ x < -1/2} = {x : ℝ | (2*x + 1 ∈ domain_fx_plus_1)} :=
by
  sorry

end domain_f_2x_plus_1_eq_l226_226134


namespace correct_option_B_l226_226462

-- Definitions reflecting the geometric relationships
variables {line : Type} {plane : Type}
variables (l m : line) (α β : plane)
variables (parallel : plane → plane → Prop)
variables (perpendicular : line → plane → Prop)

-- Assumptions based on option B
def conditionB := (perpendicular l β) ∧ (parallel α β)

-- Conclusion to be proven
def conclusionB := perpendicular l α

-- The main statement
theorem correct_option_B : conditionB l α β → conclusionB l α :=
by sorry

end correct_option_B_l226_226462


namespace difference_in_speed_l226_226764

theorem difference_in_speed (d : ℕ) (tA tE : ℕ) (vA vE : ℕ) (h1 : d = 300) (h2 : tA = tE - 3) 
    (h3 : vE = 20) (h4 : vE = d / tE) (h5 : vA = d / tA) : vA - vE = 5 := 
    sorry

end difference_in_speed_l226_226764


namespace Patricia_read_21_books_l226_226054

theorem Patricia_read_21_books
  (Candice_books Amanda_books Kara_books Patricia_books : ℕ)
  (h1 : Candice_books = 18)
  (h2 : Candice_books = 3 * Amanda_books)
  (h3 : Kara_books = Amanda_books / 2)
  (h4 : Patricia_books = 7 * Kara_books) :
  Patricia_books = 21 :=
by
  sorry

end Patricia_read_21_books_l226_226054


namespace interest_rate_increase_60_percent_l226_226454

noncomputable def percentage_increase (A P A' t : ℝ) : ℝ :=
  let r₁ := (A - P) / (P * t)
  let r₂ := (A' - P) / (P * t)
  ((r₂ - r₁) / r₁) * 100

theorem interest_rate_increase_60_percent :
  percentage_increase 920 800 992 3 = 60 := by
  sorry

end interest_rate_increase_60_percent_l226_226454


namespace perimeter_of_shaded_area_l226_226466

theorem perimeter_of_shaded_area (AB AD : ℝ) (h1 : AB = 14) (h2 : AD = 12) : 
  2 * AB + 2 * AD = 52 := 
by
  sorry

end perimeter_of_shaded_area_l226_226466


namespace find_k_l226_226504

theorem find_k (k b : ℤ) (h1 : -x^2 - (k + 10) * x - b = -(x - 2) * (x - 4))
  (h2 : b = 8) : k = -16 :=
sorry

end find_k_l226_226504


namespace probability_even_ab_minus_a_minus_b_l226_226763

theorem probability_even_ab_minus_a_minus_b :
  ∀ (a b : ℕ), a ≠ b → a ∈ {1, 3, 5, 7, 9} → b ∈ {1, 3, 5, 7, 9} → Even (a * b - a - b) :=
by
  intros
  sorry

end probability_even_ab_minus_a_minus_b_l226_226763


namespace max_gold_coins_l226_226790

-- Define the conditions as predicates
def divides_with_remainder (n : ℕ) (d r : ℕ) : Prop := n % d = r
def less_than (n k : ℕ) : Prop := n < k

-- Main statement incorporating the conditions and the conclusion
theorem max_gold_coins (n : ℕ) :
  divides_with_remainder n 15 3 ∧ less_than n 120 → n ≤ 105 :=
by
  sorry

end max_gold_coins_l226_226790


namespace no_integer_root_permutation_l226_226524

theorem no_integer_root_permutation (n : ℕ) (k : Fin (2 * n + 1) → ℤ) 
  (h_nonzero : ∀ i, k i ≠ 0) (h_sum_nonzero : ∑ i, k i ≠ 0) :
  ∃ (a : Fin (2 * n + 1) → ℤ), 
  function.bijective (k ∘ a) ∧ (∀ x : ℤ, (eval x (polynomial.bind (a ∘ Fin.val)).to_polynomial) ≠ 0) := 
sorry

end no_integer_root_permutation_l226_226524


namespace value_of_a_plus_b_l226_226144

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem value_of_a_plus_b (a b : ℝ) (h1 : 3 * a + b = 4) (h2 : a + b + 1 = 3) : a + b = 2 :=
by
  sorry

end value_of_a_plus_b_l226_226144


namespace find_preimage_of_7_l226_226109

variable {A B : Type} (f : A → B) (h : ∀ x : A, f x = 3 * x + 1)

theorem find_preimage_of_7 {x : A} (hx : f x = 7) : x = 2 :=
by sorry

end find_preimage_of_7_l226_226109


namespace sum_of_odd_numbered_terms_arithmetic_seq_l226_226211

theorem sum_of_odd_numbered_terms_arithmetic_seq :
  ∀ (a : ℕ → ℚ) (d : ℚ) (S_100 : ℚ),
  (∀ n, a (n + 1) = a n + d) →
  d = 1/2 →
  S_100 = 45 →
  S_100 = 50 * (a 0 + a 99) →
  (∑ k in Finset.range 50, a (2*k + 1)) = -69 :=
by
  sorry

end sum_of_odd_numbered_terms_arithmetic_seq_l226_226211


namespace correctness_l226_226057

-- Define the propositions
def p (a : ℝ) : Prop := (0 < a ∧ a ≠ 1) → ∀ x, (f(x) = a^x - 2 ∧ f(0) = -2)
def q : Prop := ∀ x, x ≠ 0 → (log (abs x) = 0 → x = 1 ∨ x = -1)

-- Theorem to be proved
theorem correctness (a : ℝ) (h : 0 < a ∧ a ≠ 1) : (p a ∨ q) :=
sorry -- proof omitted

end correctness_l226_226057


namespace max_value_of_b_l226_226947

theorem max_value_of_b (a b c : ℝ) (q : ℝ) (hq : q ≠ 0) 
  (h_geom : a = b / q ∧ c = b * q) 
  (h_arith : 2 * b + 4 = a + 6 + (b + 2) + (c + 1) - (b + 2)) :
  b ≤ 3 / 4 :=
sorry

end max_value_of_b_l226_226947


namespace numbers_of_form_xy9z_div_by_132_l226_226872

theorem numbers_of_form_xy9z_div_by_132 (x y z : ℕ) :
  let N := 1000 * x + 100 * y + 90 + z
  (N % 4 = 0) ∧ ((x + y + 9 + z) % 3 = 0) ∧ ((x + 9 - y - z) % 11 = 0) ↔ 
  (N = 3696) ∨ (N = 4092) ∨ (N = 6996) ∨ (N = 7392) :=
by
  intros
  let N := 1000 * x + 100 * y + 90 + z
  sorry

end numbers_of_form_xy9z_div_by_132_l226_226872


namespace field_area_hectares_l226_226809

theorem field_area_hectares 
  (cost_per_meter : ℝ) 
  (total_cost : ℝ) 
  (h : cost_per_meter = 4.50) 
  (h1 : total_cost = 5938.80) : 
  ∃ (area_hectares : ℝ), abs (area_hectares - 13.854) < 0.001 :=
by 
  let circumference := total_cost / cost_per_meter
  have hc : circumference = 1319.7333 := by
    calc
      circumference = total_cost / cost_per_meter : by rfl
                  ... = 5938.80 / 4.50 : by rw [h, h1]
                  ... = 1319.7333 : by norm_num
  let radius := circumference / (2 * real.pi)
  have hr : radius = 210.024 := by
    calc
      radius = 1319.7333 / (2 * real.pi) : by rw [hc]
            ... = 210.024 : by norm_num
  let area := real.pi * radius^2
  have ha : area = 138544.548 := by
    calc
      area = real.pi * (210.024)^2 : by rw [hr]
          ... = 138544.548 : by norm_num
  let area_hectares := area / 10000
  have ha_hectares : abs (area_hectares - 13.854) < 0.001 := by
    calc
      abs (area_hectares - 13.854) 
        = abs (138544.548 / 10000 - 13.854) : by rw [ha]
        ... = abs (13.8544548 - 13.854) : by norm_num
        ... < 0.001 : by norm_num
  exact ⟨area_hectares, ha_hectares⟩

end field_area_hectares_l226_226809


namespace find_n_l226_226912

theorem find_n (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = Real.exp x + Real.log (x + 1)) ∧ 
  ∀ (y : ℝ), y = 2 * (x : ℝ : (y = (4 / n - x))) → 
  (n = -2) :=
by
  sorry

end find_n_l226_226912


namespace no_fruit_children_count_l226_226326

theorem no_fruit_children_count : 
  let n := 158 
  let apples := {i | i % 2 = 1, 1 ≤ i ∧ i ≤ n} 
  let bananas := {n - i + 1 | i % 3 = 1, 1 ≤ i ∧ i ≤ n} 
  let both := apples ∩ bananas
  let either_fruit := apples ∪ bananas 
  let no_fruit := finset.range (n + 1) \ either_fruit
  no_fruit.card = 52 := 
by
  sorry

end no_fruit_children_count_l226_226326


namespace range_of_b_l226_226493

theorem range_of_b (b : ℝ) : 
  (¬ (4 ≤ 3 * 3 + b) ∧ (4 ≤ 3 * 4 + b)) ↔ (-8 ≤ b ∧ b < -5) := 
by
  sorry

end range_of_b_l226_226493


namespace intersection_empty_implies_m_leq_neg1_l226_226630

theorem intersection_empty_implies_m_leq_neg1 (m : ℝ) :
  (∀ (x y: ℝ), (x < m) → (y = x^2 + 2*x) → y < -1) →
  m ≤ -1 :=
by
  intro h
  sorry

end intersection_empty_implies_m_leq_neg1_l226_226630


namespace relationship_among_a_b_c_l226_226902

noncomputable def a : ℝ := 2^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.cos (100 * Real.pi / 180)

theorem relationship_among_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_among_a_b_c_l226_226902


namespace david_marks_in_biology_l226_226489

theorem david_marks_in_biology
  (marks_english : ℕ := 96)
  (marks_math : ℕ := 95)
  (marks_physics : ℕ := 82)
  (marks_chemistry : ℕ := 87)
  (average_marks : ℝ := 90.4)
  (total_subjects : ℕ := 5):
  let total_marks_other_subjects := marks_english + marks_math + marks_physics + marks_chemistry in
  let total_marks_all_subjects := average_marks * total_subjects in
  let marks_biology := total_marks_all_subjects - total_marks_other_subjects in
  marks_biology = 92 :=
by
  -- Defining the total marks obtained in other subjects
  let total_marks_other_subjects := marks_english + marks_math + marks_physics + marks_chemistry

  -- Defining the total marks for all subjects based on average
  let total_marks_all_subjects := average_marks * total_subjects

  -- Calculating the marks in Biology
  let marks_biology := total_marks_all_subjects - total_marks_other_subjects
  
  -- The expected marks in Biology should be 92
  have h : marks_biology = 92, from sorry -- The proof steps go here but are skipped.

  exact h

end david_marks_in_biology_l226_226489


namespace probability_distance_greater_than_2_l226_226565

noncomputable def region (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3

noncomputable def regionArea : ℝ :=
  9

noncomputable def circle (x y : ℝ) : Prop :=
  x^2 + y^2 > 4

noncomputable def excludedCircleArea : ℝ :=
  9 - Real.pi

theorem probability_distance_greater_than_2 :
  ∫ (x y : ℝ) in {p | region p.1 p.2 ∧ circle p.1 p.2}, 1 ∂(MeasureTheory.Measure.prod MeasureTheory.measureSpace.volume MeasureTheory.measureSpace.volume) /
  regionArea = (9 - Real.pi) / 9 := by
sorry

end probability_distance_greater_than_2_l226_226565


namespace geometric_sequence_sum_l226_226997

theorem geometric_sequence_sum (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ n, a (n + 1) = a n * r)
  (h1 : a 0 + a 1 + a 2 = 8)
  (h2 : a 3 + a 4 + a 5 = -4) :
  a 6 + a 7 + a 8 = 2 := 
sorry

end geometric_sequence_sum_l226_226997


namespace avg_tickets_sold_by_male_members_l226_226792

variable (M : ℕ) -- Number of male members
variable (F : ℕ) -- Number of female members
variable (q : ℕ) -- Average number of tickets sold by the male members

-- Conditions
axiom avg_per_member : (M + F) > 0 → (M * q + 70 * F) / (M + F) = 66
axiom female_ratio : F = 2 * M

theorem avg_tickets_sold_by_male_members :
  (M + F) > 0 → (M + F) > 0 → q = 58 :=
by
  intro h1 h2
  rw [female_ratio] at avg_per_member
  have main_inequality := avg_per_member h1
  have equation := calc
    (M * q + 70 * (2 * M)) / (M + 2 * M) = 66 : main_inequality
    (M * q + 140 * M) / (3 * M) = 66 : by sorry -- Provide the full steps
    (M * q + 140 * M) = 198 * M : by sorry      -- in the proof.
    M * q + 140 * M = 198 * M : by sorry
    M * q = 58 * M : by sorry
  q = 58 : by sorry


end avg_tickets_sold_by_male_members_l226_226792


namespace wario_missed_fraction_l226_226768

-- Definitions based on conditions
def total_att := 60

def wide_right_percent := 0.20

def wide_right_missed := 3

noncomputable def total_missed : ℕ := wide_right_missed / wide_right_percent

def fraction_missed : ℚ := total_missed / total_att

-- Proof statement
theorem wario_missed_fraction :
  fraction_missed = 1 / 4 :=
sorry

end wario_missed_fraction_l226_226768


namespace notebooks_difference_l226_226223

theorem notebooks_difference :
  ∀ (Jac_left Jac_Paula Jac_Mike Ger_not Jac_init : ℕ),
  Ger_not = 8 →
  Jac_left = 10 →
  Jac_Paula = 5 →
  Jac_Mike = 6 →
  Jac_init = Jac_left + Jac_Paula + Jac_Mike →
  Jac_init - Ger_not = 13 := 
by
  intros Jac_left Jac_Paula Jac_Mike Ger_not Jac_init
  intros Ger_not_8 Jac_left_10 Jac_Paula_5 Jac_Mike_6 Jac_init_def
  sorry

end notebooks_difference_l226_226223


namespace distances_not_possible_l226_226920

noncomputable def point := (ℝ × ℝ)
noncomputable def square := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distances_not_possible (P : point) (A B C D : point)
  (square_def : square := (A, B, C, D))
  (dPA : distance P A = 1)
  (dPB : distance P B = 1)
  (dPC : distance P C = 3)
  (dPD : distance P D = 2) :
  false :=
sorry

end distances_not_possible_l226_226920


namespace count_4x4_increasing_matrices_l226_226303

theorem count_4x4_increasing_matrices : 
  ∃ C, C = (number of ways to arrange the digits from 1 to 16 in a 4x4 matrix such that each row and column are in increasing order) := 
sorry

end count_4x4_increasing_matrices_l226_226303


namespace sum_of_distinct_prime_factors_of_420_l226_226475

theorem sum_of_distinct_prime_factors_of_420 : 
  let prime_factors_420 := [2, 3, 5, 7] in
  prime_factors_420.sum = 17 :=
by
  let prime_factors_420 := [2, 3, 5, 7]
  have h : prime_factors_420.sum = 17 := sorry
  exact h

end sum_of_distinct_prime_factors_of_420_l226_226475


namespace card_paiting_modulus_l226_226368

theorem card_paiting_modulus (cards : Finset ℕ) (H : cards = Finset.range 61 \ {0}) :
  ∃ d : ℕ, ∀ n ∈ cards, ∃! k, (∀ x ∈ cards, (x + n ≡ k [MOD d])) ∧ (d ∣ 30) ∧ (∃! n : ℕ, 1 ≤ n ∧ n ≤ 8) :=
sorry

end card_paiting_modulus_l226_226368


namespace sqrt_4_minus_1_l226_226845

  theorem sqrt_4_minus_1 : sqrt 4 - 1 = 1 := by
    sorry
  
end sqrt_4_minus_1_l226_226845


namespace radius_square_find_sum_of_m_and_n_l226_226002

noncomputable def radius_of_spheres_problem : ℚ := 160 / 13

theorem radius_square (r: ℚ) (m n : ℕ) (h : Int.gcd m n = 1) 
  (hr_sq : r^2 = (m : ℚ) / (n : ℚ)) 
  (h_cond1: ∀ r, 
    let a := (8:ℚ)*r*(r-4:ℚ)+53 = 0 :=
    sorry)
  (h_cond2: ∀ r, 
    let b := r * (r -1 : ℚ) = 2 :=
    sorry)
  : radius_of_spheres_problem = (160: ℚ) / (13 : ℚ) :=
begin
  sorry
end

theorem find_sum_of_m_and_n
  (m n : ℕ) 
  (hmn_rel_prime : Int.gcd m n = 1)
  (h_fraction : 160 / 13 = (m : ℚ) / (n : ℚ)) :
  m + n = 173 :=
begin
  have h_val : m = 160,
  have h_val2 : n = 13,
  sorry -- more detailed proof needing iterations
end

end radius_square_find_sum_of_m_and_n_l226_226002


namespace not_possible_distances_l226_226917

-- Define the square and its geometric properties
structure Square (α : Type*) :=
  (A B C D : α → α → Prop)
  (side_length : ℝ)
  (diag_length : ℝ := side_length * Real.sqrt 2)
  (P : α → α → Prop)

-- Conditions of the problem
variables {α : Type*} [MetricSpace α]
variables (S : Square α)
variables {PA PB PC PD : ℝ}
variables {P : α}

-- Define the distances from point P to vertices of the square
def dist_to_vertices := [PA, PB, PC, PD]

-- The main statement to prove
theorem not_possible_distances (h : set.dist P S.A = 1)
                              (hS1 : set.dist P S.B = 1)
                              (hS2 : set.dist P S.C = 2)
                              (hS3 : set.dist P S.D = 3) :
  false := 
begin
  sorry
end

end not_possible_distances_l226_226917


namespace triangle_constructibility_l226_226320

noncomputable def constructible_triangle (a b w_c : ℝ) : Prop :=
  (2 * a * b) / (a + b) > w_c

theorem triangle_constructibility {a b w_c : ℝ} (h : (a > 0) ∧ (b > 0) ∧ (w_c > 0)) :
  constructible_triangle a b w_c ↔ True :=
by
  sorry

end triangle_constructibility_l226_226320


namespace complex_number_calculation_l226_226472

theorem complex_number_calculation : 
  (5 - 5 * Complex.i) + (-2 - Complex.i) - (3 + 4 * Complex.i) = -10 * Complex.i := 
by 
  sorry

end complex_number_calculation_l226_226472


namespace domain_sqrt_cos_l226_226689

theorem domain_sqrt_cos:
  ∀ x: ℝ, (∃ k : ℤ, -π/3 + 2*k*π ≤ x ∧ x ≤ π/3 + 2*k*π) ↔ ∃ y: ℝ, y = sqrt (2*(cos x) - 1) :=
begin
  sorry
end

end domain_sqrt_cos_l226_226689


namespace monotonically_increasing_condition_l226_226904

theorem monotonically_increasing_condition 
  (a b c d : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 ≤ 3 * a * x ^ 2 + 2 * b * x + c) ↔ (b^2 - 3 * a * c ≤ 0) :=
by {
  sorry
}

end monotonically_increasing_condition_l226_226904


namespace function_decreasing_on_interval_l226_226137

noncomputable def f (ω varphi : ℝ) (x : ℝ) : ℝ :=
  sin (ω * x + varphi) + cos (ω * x + varphi)

theorem function_decreasing_on_interval
  (ω varphi : ℝ)
  (h1 : ω > 0)
  (h2 : abs varphi < real.pi / 2)
  (h3 : (∀ x, f ω varphi (x + real.pi / ω) = f ω varphi x))
  (h4 : (∀ x, f ω varphi (-x) = f ω varphi x))
  : ∀ x x' : ℝ, 0 ≤ x ∧ x ≤ x' ∧ x' ≤ real.pi / 2 → f ω varphi x ≥ f ω varphi x' := by
  sorry

end function_decreasing_on_interval_l226_226137


namespace number_of_valid_pairing_ways_l226_226341

-- Define a natural number as a condition.
def is_natural (n : ℕ) : Prop := 0 < n

-- Define that 60 cards can be paired with the same modulus difference.
def pair_cards_same_modulus_difference (d : ℕ) (k : ℕ) : Prop :=
  60 = 2 * d * k

-- Define what it means for d to be a divisor of 30.
def is_divisor_of_30 (d : ℕ) : Prop :=
  ∃ k, 30 = d * k

theorem number_of_valid_pairing_ways :
  (finset.univ.filter is_divisor_of_30).card = 8 :=
begin
  sorry
end

end number_of_valid_pairing_ways_l226_226341


namespace exists_triangle_area_le_two_l226_226992

-- Definitions
def is_lattice_point (P : ℤ × ℤ) : Prop :=
  let (x, y) := P in abs x ≤ 2 ∧ abs y ≤ 2

def no_three_collinear (P : Set (ℤ × ℤ)) : Prop :=
  ∀ (A B C : ℤ × ℤ), A ∈ P → B ∈ P → C ∈ P → 
  (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) ≠ 0)

noncomputable def triangle_area (A B C : ℤ × ℤ) : ℚ :=
  (1 / 2 : ℚ) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Theorem Statement
theorem exists_triangle_area_le_two (P : Finset (ℤ × ℤ)) (h1 : ∀ p ∈ P, is_lattice_point p) 
  (h2 : no_three_collinear (P : Set (ℤ × ℤ))) (h3 : P.card = 6) :
  ∃ (A B C : ℤ × ℤ), A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ 
  triangle_area A B C ≤ (2 : ℚ) :=
sorry

end exists_triangle_area_le_two_l226_226992


namespace gauss_line_l226_226005

open Function

variable {A B C D E F M N P : Type}
variable [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ E] [AffineSpace ℝ F] 
variable [AffineSpace ℝ M] [AffineSpace ℝ N] [AffineSpace ℝ P]

-- Define the midpoints
def is_midpoint (M : A) (D C : A) : Prop :=
IsMidpoint M D C

def is_midpoint_n (N : A) (A E : A) : Prop :=
IsMidpoint N A E

def is_midpoint_p (P : A) (B F : A) : Prop :=
IsMidpoint P B F

-- Collinearity predicate
def are_collinear (M N P : A) : Prop :=
Collinear ℝ {M, N, P}

-- Main theorem statement
theorem gauss_line (hD_intersects: ∃ (line: A → ℝ), ∀ {p : A},
  p = D ∨ p = E ∨ p = F → line p = 0)
  (h_mid_M: is_midpoint M D C)
  (h_mid_N: is_midpoint_n N A E)
  (h_mid_P: is_midpoint_p P B F) :
  are_collinear M N P := 
by 
  sorry

end gauss_line_l226_226005


namespace percentage_decrease_l226_226995

theorem percentage_decrease (x y : ℝ) : 
  let original_value := x^2 * y^3 in
  let new_x := 0.8 * x in
  let new_y := 0.7 * y in
  let new_value := (new_x)^2 * (new_y)^3 in
  (original_value - new_value) / original_value * 100 = 78.048 :=
by 
  sorry

end percentage_decrease_l226_226995


namespace area_ratio_of_smaller_octagon_l226_226717

theorem area_ratio_of_smaller_octagon
    (A B C D E F G H : ℝ × ℝ) -- Coordinates of vertices of the larger octagon
    (P Q R S T U V W : ℝ × ℝ) -- Coordinates of vertices of the smaller octagon
    (regular_octagon : ∀ (X Y Z W U V T S : ℝ × ℝ), regular_octo X Y Z W U V T S)  -- Predicate for regular octagon
    (midpoints_joined : ∀ (X Y : ℝ × ℝ), midpoint X Y) : -- Condition that midpoints form the smaller octagon
  area (smaller_octo P Q R S T U V W) = (3 : ℝ) / 4 * area (larger_octo A B C D E F G H) :=
sorry

end area_ratio_of_smaller_octagon_l226_226717


namespace varphi_solution_l226_226548

noncomputable def varphi (x : ℝ) (m n : ℝ) : ℝ :=
  m * x + n / x

theorem varphi_solution :
  ∃ (m n : ℝ), (varphi 1 m n = 8) ∧ (varphi 16 m n = 16) ∧ (∀ x, varphi x m n = 3 * x + 5 / x) :=
sorry

end varphi_solution_l226_226548


namespace cows_and_sheep_bushels_l226_226060

theorem cows_and_sheep_bushels (bushels_per_chicken: Int) (total_bushels: Int) (num_chickens: Int) 
  (bushels_chickens: Int) (bushels_cows_sheep: Int) (num_cows: Int) (num_sheep: Int):
  bushels_per_chicken = 3 ∧ total_bushels = 35 ∧ num_chickens = 7 ∧
  bushels_chickens = num_chickens * bushels_per_chicken ∧ bushels_chickens = 21 ∧ bushels_cows_sheep = total_bushels - bushels_chickens → 
  bushels_cows_sheep = 14 := by
  sorry

end cows_and_sheep_bushels_l226_226060


namespace find_a_l226_226858

theorem find_a (a : ℝ) (h : ∀ x : ℝ, a * x = x) : a = 1 / 4 :=
by
  have h_ax : ∀ x : ℝ, 4 * a * x = x := by
    intro x
    have h_def := h x
    rw mul_comm at h_def
    exact h_def
  have a_eq : a * 1 = 1 := by
    specialize h_ax 1
    rw mul_one at h_ax
    exact h_ax
  sorry

end find_a_l226_226858


namespace juliet_card_probability_l226_226230

theorem juliet_card_probability :
  (∃ (C : Finset ℕ), (∀ c ∈ C, 1 ≤ c ∧ c ≤ 120) ∧ C.card = 120) →
  (∀ C, (∃ (c : ℕ), c ∈ C ∧ (c % 2 = 0 ∨ c % 4 = 0 ∨ c % 6 = 0)) →
  ∃ p, p = (5 : ℚ) / 6) :=
by
  intros
  sorry

end juliet_card_probability_l226_226230


namespace total_cost_l226_226662

theorem total_cost (p1 p2 p3 p4 p5 : ℝ) (t : ℝ) (d g : ℝ)
  (special_triple: ∀ n, n % 2 = 0 → n / 2)
  (special_meat: ∀ n, n % 3 = 0 → 2 * n / 3)
  (special_veggie: ∀ n, n % 5 = 0 → 3 * n / 5)
  (special_drink: ∀ n : ℕ, 2 * n)
  (num_triple : ℕ) (num_meat : ℕ) (num_veggie : ℕ) (num_garlic_bread : ℕ) (num_drink : ℕ)
  (large_price medium_price small_price topping_price drink_price garlic_bread_price : ℝ)
  (num_triple_toppings num_meat_toppings num_veggie_toppings: ℕ) :
  large_price = 10 →
  medium_price = 8 →
  small_price = 5 →
  topping_price = 2.5 →
  drink_price = 2 →
  garlic_bread_price = 4 →
  num_triple = 6 →
  num_meat = 4 →
  num_veggie = 10 →
  num_garlic_bread = 5 →
  num_drink = 8 →
  num_triple_toppings = 2 →
  num_meat_toppings = 3 →
  num_veggie_toppings = 1 →
  p1 = num_triple / 2 * large_price + num_triple * num_triple_toppings * topping_price →
  p2 = num_meat / 3 * 2 * medium_price + num_meat * num_meat_toppings * topping_price →
  p3 = num_veggie / 5 * 3 * small_price + num_veggie * num_veggie_toppings * topping_price →
  p4 = num_drink * drink_price →
  p5 = num_garlic_bread * garlic_bread_price →
  t = p1 + p2 + p3 + p4 - (num_garlic_bread * garlic_bread_price) →
  t = 185 := 
sorry

end total_cost_l226_226662


namespace distinct_pairs_l226_226490

theorem distinct_pairs (x y : ℝ) (h : x ≠ y) :
  x^100 - y^100 = 2^99 * (x - y) ∧ x^200 - y^200 = 2^199 * (x - y) ↔ (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) :=
by
  sorry

end distinct_pairs_l226_226490


namespace geometricSeqMinimumValue_l226_226998

noncomputable def isMinimumValue (a : ℕ → ℝ) (n m : ℕ) (value : ℝ) : Prop :=
  ∀ b : ℝ, (1 / a n + b / a m) ≥ value

theorem geometricSeqMinimumValue {a : ℕ → ℝ}
  (h1 : ∀ n, a n > 0)
  (h2 : a 7 = (Real.sqrt 2) / 2)
  (h3 : ∀ n, ∀ m, a n * a m = a (n + m)) :
  isMinimumValue a 3 11 4 :=
sorry

end geometricSeqMinimumValue_l226_226998


namespace evaluate_expression_l226_226077

theorem evaluate_expression (a b : ℝ) (h : a ≠ b) :
    (a⁻⁶ - b⁻⁶) / (a⁻³ - b⁻³) = a⁻⁶ + a⁻³ * b⁻³ + b⁻⁶ := 
by sorry

end evaluate_expression_l226_226077


namespace AQI_median_mode_l226_226299

-- Define the AQI data set
def AQIData : List Nat := [29, 24, 38, 27, 29, 27, 27]

-- Define the median function for lists of natural numbers
def median (l : List Nat) : Nat :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2) -- Get the middle element

-- Define the mode function for lists of natural numbers
def mode (l : List Nat) : Nat :=
  l.map (λ n => (n, l.count n)).maxBy fun a b => a.2 |>.1 -- Find the value with the highest count

-- Theorem stating that the median and mode of the AQIData are both 27
theorem AQI_median_mode :
  median AQIData = 27 ∧ mode AQIData = 27 :=
by
  sorry

end AQI_median_mode_l226_226299


namespace vika_pairs_exactly_8_ways_l226_226347

theorem vika_pairs_exactly_8_ways :
  ∃ d : ℕ, (d ∣ 30) ∧ (Finset.card (Finset.filter (λ d, d ∣ 30) (Finset.range 31)) = 8) := 
sorry

end vika_pairs_exactly_8_ways_l226_226347


namespace num_ordered_pairs_24_l226_226512

noncomputable def numSolutions : Nat :=
  let conditions (a b : ℂ) := a^4 * b^6 = 1 ∧ a^8 * b^3 = 1
  have h: { p : ℂ × ℂ | conditions p.1 p.2 }.to_finset.card = 24
  24

theorem num_ordered_pairs_24 :
  ∃ n : ℕ, n = numSolutions := by
  use 24
  sorry

end num_ordered_pairs_24_l226_226512


namespace single_intersection_l226_226785

noncomputable def Q_rotated_coordinates (t : ℝ) : ℝ × ℝ :=
  let s := (√2) * t^2 - 2 * t
  ( ((3 * √2)/2) * t - t^2, t^2 - (√2 / 2) * t )

theorem single_intersection (a : ℝ) : a = -1/8 → 
  ∀ t : ℝ, let Q := Q_rotated_coordinates t in
  let C := (Q.fst = ((3 * √2)/2) * t - t^2) ∧ (Q.snd = t^2 - (√2 / 2) * t) in
  ∃! p : ℝ × ℝ, p.snd = a ∧ C :=
sorry

#check single_intersection

end single_intersection_l226_226785


namespace total_area_of_squares_l226_226962

-- Define the coordinates of the vertices of the first square
def p1 := (0 : ℝ, 3 : ℝ)
def p2 := (4 : ℝ, 0 : ℝ)

-- The distance between p1 and p2 is the side length of the square
def side_length : ℝ := Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Each square has an area equal to the side length squared
def area_one_square : ℝ := side_length ^ 2

-- The total area of two identical squares
def total_area : ℝ := 2 * area_one_square

-- Prove that the total area is 50 when the vertices of the first square are given
theorem total_area_of_squares : total_area = 50 := by
  -- Define the coordinates of the points
  let p1 := (0 : ℝ, 3 : ℝ)
  let p2 := (4 : ℝ, 0 : ℝ)
  
  -- Calculate the side length
  let side_length := Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)
  
  -- Calculate the area of one square
  let area_one_square := side_length ^ 2
  
  -- Calculate the total area
  have h1 : total_area = 2 * area_one_square := rfl
  
  -- Check the side length computation
  have h2 : side_length = 5 := by
    rw [Real.sqrt_eq_iff_sq_eq (by norm_num : 0 ≤ (5 : ℝ))]
    simp [*, pow_two]
  
  -- Check the area computation
  have h3 : area_one_square = 25 := by
    rw [h2, pow_two]
  
  -- Substitute in total_area computation
  rw [h1, h3]
  norm_num -- this will reduce 2 * 25 to 50
  done

end total_area_of_squares_l226_226962


namespace range_of_k_for_four_distinct_real_solutions_l226_226188

open Real

theorem range_of_k_for_four_distinct_real_solutions :
  (∃ k : ℝ, (∀ x : ℝ, (k * x^2 + 6 * k * x - 1 = 0 ∨ k * x^2 + 6 * k * x + 1 = 0))
  → (∃ x1 x2 x3 x4 : ℝ,
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
    (k * x1^2 + 6 * k * x1 - 1 = 0 ∨ k * x1^2 + 6 * k * x1 + 1 = 0) ∧
    (k * x2^2 + 6 * k * x2 - 1 = 0 ∨ k * x2^2 + 6 * k * x2 + 1 = 0) ∧
    (k * x3^2 + 6 * k * x3 - 1 = 0 ∨ k * x3^2 + 6 * k * x3 + 1 = 0) ∧
    (k * x4^2 + 6 * k * x4 - 1 = 0 ∨ k * x4^2 + 6 * k * x4 + 1 = 0)) ↔ 
    ∃ k : ℝ, k > (1/9)) :=
begin
  sorry
end

end range_of_k_for_four_distinct_real_solutions_l226_226188


namespace irrational_number_is_neg_sqrt_3_l226_226461

-- Define the given conditions
def sqrt_16 := real.sqrt 16
def neg_sqrt_3 := -real.sqrt 3
def cbrt_8 := real.cbrt 8
def frac_7_3 := 7 / 3

-- The main theorem to prove that -√3 is the only irrational number
theorem irrational_number_is_neg_sqrt_3 : 
  (irrational neg_sqrt_3) ∧ 
  (¬ irrational sqrt_16) ∧ 
  (¬ irrational cbrt_8) ∧ 
  (¬ irrational frac_7_3) :=
by
  sorry

end irrational_number_is_neg_sqrt_3_l226_226461


namespace measure_angle_URS_l226_226614

-- Define angles and their relationships
def parallel_lines (TU PS : Type) := sorry   -- Parallelism condition, simplified
def lies_on (Q R PS: Type) := sorry          -- Point lies on line condition, simplified
def angle_PQT (x : ℝ) := sorry              -- Angle measure placeholder
def angle_RQT (x : ℝ) := sorry              -- Angle measure placeholder
def angle_TUR (x : ℝ) := sorry              -- Angle measure placeholder

theorem measure_angle_URS (TU PS : Type) (Q R : PS) (x : ℝ)
  (h1 : parallel_lines TU PS)
  (h2 : lies_on Q PS)
  (h3 : lies_on R PS)
  (h4 : angle_PQT x)
  (h5 : angle_RQT (x - 50))
  (h6 : angle_TUR (x + 25)) :
  (x = 115) -> ∠ URS = 140 := by sorry

end measure_angle_URS_l226_226614


namespace general_term_formula_a_n_exists_d_n_Sn_eq_10_l226_226922

-- Define the arithmetic sequence
def arithmetic_seq (a_1 d n : ℕ) : ℕ := a_1 + (n - 1) * d

-- (1) Prove the general term formula for the sequence when given a specific term
theorem general_term_formula_a_n {
  d : ℕ
  (h_d_pos : 0 < d)
  (h_a_5 : arithmetic_seq (-2) d 5 = 30) :
  ∃ a_n : ℕ → ℕ, ∀ n : ℕ, a_n n = 8 * n - 10 :=
by
  sorry

-- (2) Determine if there exist d and n such that Sn = 10
theorem exists_d_n_Sn_eq_10 {
  S_n : ℕ → ℕ → ℕ → ℕ := λ a_1 d n, n * a_1 + (n * (n - 1) * d) / 2
  h_S_n_eq_10 : ∃ (d n : ℕ), S_n (-2) d n = 10
} :
  (∃ d n, (d ∈ {14, 3, 2}) ∧ 
    ((d = 14 ∧ n = 2) → ∀ m, arithmetic_seq (-2) d m = 14 * m - 16) ∧
    ((d = 3 ∧ n = 4) → ∀ m, arithmetic_seq (-2) d m = 3 * m - 5) ∧
    ((d = 2 ∧ n = 5) → ∀ m, arithmetic_seq (-2) d m = 2 * m - 4)) :=
by
  sorry

end general_term_formula_a_n_exists_d_n_Sn_eq_10_l226_226922


namespace exterior_angle_DEF_l226_226667

noncomputable def angle_of_hexagon : ℝ := 120
noncomputable def angle_of_octagon : ℝ := 135

theorem exterior_angle_DEF 
    (angle_hexagon : angle_of_hexagon = 120)
    (angle_octagon : angle_of_octagon = 135) : 
    angle_hexagon + angle_octagon + 105 = 360 :=
by 
  sorry

end exterior_angle_DEF_l226_226667


namespace valerie_cookie_problem_l226_226338

-- Define variables and conditions for the problem
variable (x : ℕ) -- Original recipe makes x dozen cookies
variable (butter_original : ℕ) -- Pounds of butter in original recipe
variable (butter_needed : ℕ) -- Pounds of butter needed for 4 dozen cookies
variable (cookies_needed : ℕ) -- Dozen cookies needed for the weekend

-- Conditions given in the problem
def valerie_conditions :=
  butter_original = 4 ∧
  butter_needed = 1 ∧
  cookies_needed = 4 

-- Statement translating to the proof problem
theorem valerie_cookie_problem (h : valerie_conditions) : x = 16 :=
  by
  rcases h with ⟨h1, h2, h3⟩
  have proportion : 4 / x = 1 / 4 := by sorry
  sorry

end valerie_cookie_problem_l226_226338


namespace ratio_of_areas_l226_226761

theorem ratio_of_areas (C1 C2 : ℝ) (h1 : (60 : ℝ) / 360 * C1 = (48 : ℝ) / 360 * C2) : 
  (C1 / C2) ^ 2 = 16 / 25 := 
by
  sorry

end ratio_of_areas_l226_226761


namespace smaller_angle_between_clock_hands_at_7_35_l226_226379

theorem smaller_angle_between_clock_hands_at_7_35
  (minute_hand_degrees_per_minute : ℝ := 6)
  (hour_hand_degrees_per_minute : ℝ := 0.5)
  (initial_hour_hand_position : ℝ := 7 * 30)
  (time_in_minutes : ℝ := 35) :
  let minute_hand_position := 35 * minute_hand_degrees_per_minute,
      hour_hand_position := initial_hour_hand_position + 35 * hour_hand_degrees_per_minute,
      angle_between_hands := |hour_hand_position - minute_hand_position| in
  angle_between_hands = 17.5 := by
  sorry

end smaller_angle_between_clock_hands_at_7_35_l226_226379


namespace complex_number_properties_l226_226104

theorem complex_number_properties (a b : ℝ) (z : ℂ) (h_z : z = a + b * complex.I) (h_cond : a + b = 1) :
  ((∃ a ℝ, b = 0 ∧ z = a) ∧ ¬(∃ b ℝ, a = 0 ∧ z = b * complex.I) ∧ (∃ (conj_z : ℂ), conj_z = a - b * complex.I ∧ z * conj_z = a^2 + b^2)) → (count_correct_statements a b z conj_z = 2) :=
sorry

end complex_number_properties_l226_226104


namespace infinite_primes_of_form_2px_plus_1_l226_226260

theorem infinite_primes_of_form_2px_plus_1 (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) : 
  ∃ᶠ (n : ℕ) in at_top, Nat.Prime (2 * p * n + 1) :=
sorry

end infinite_primes_of_form_2px_plus_1_l226_226260


namespace monotonically_increasing_intervals_values_of_a_and_b_l226_226956

-- Problem 1: Monotonically increasing interval
theorem monotonically_increasing_intervals (a b : ℝ) (h : a > 0):
  ∃ (k : ℤ), 
  ∀ (x : ℝ), 
  x ∈ [k*ℝ.pi - 3*ℝ.pi / 8, k*ℝ.pi + ℝ.pi / 8] →
  monotonically_increasing (λ x, a * (real.cos x ^ 2 + real.sin x * real.cos x) + b) :=
sorry

-- Problem 2: Values of a and b
theorem values_of_a_and_b (a b : ℝ) (h : a < 0) (hx : ∀ x, x ∈ Icc 0 (ℝ.pi / 2)):
  (∀ (x : Icc 0 (ℝ.pi / 2)), a * (real.cos x ^ 2 + real.sin x * real.cos x) + b ∈ Icc 3 4) →
  a = 2 - 2 * real.sqrt 2 ∧ b = 4 :=
sorry

end monotonically_increasing_intervals_values_of_a_and_b_l226_226956


namespace angle_6_measure_l226_226655

-- Define the problem and the theorem
theorem angle_6_measure {p q : Line} (h1 : parallel p q) 
    (h2 : ∀ ⦃x⦄, degrees (angle_1) = x / 3) 
    (h3 : angle_5 = angle_1) 
    (h4 : angle_5 + angle_6 = 180) 
    (h5 : opposite angle_6 angle_5) 
    : degrees angle_6 = 135 := 
sorry

end angle_6_measure_l226_226655


namespace quadrilateral_area_l226_226205

theorem quadrilateral_area 
  {A B C D : Type} 
  (h1: AB = BC)
  (h2: ∠ABC = 90)
  (h3: ∠ADC = 90)
  (h4: dist B (line AD) = 10) 
  : area ABCD = 100 := 
sorry

end quadrilateral_area_l226_226205


namespace find_n_exists_l226_226639

noncomputable theory

open Real

def conditions (n : ℕ) (r : ℝ) (m : ℕ) :=
  m = (n + r)^3 ∧ 0 < r ∧ r < (1 / 500 : ℝ)

theorem find_n_exists (m : ℕ) (n : ℕ) (r : ℝ) :
  (∃ m, ∃ r, conditions n r m) → n = 13 :=
by
  sorry

end find_n_exists_l226_226639


namespace evaluate_expression_at_2_l226_226075

theorem evaluate_expression_at_2 : ∀ (x : ℕ), x = 2 → (x^x)^(x^(x^x)) = 4294967296 := by
  intros x h
  rw [h]
  sorry

end evaluate_expression_at_2_l226_226075


namespace free_endpoints_eq_1001_l226_226913

theorem free_endpoints_eq_1001 : 
  ∃ k : ℕ, 1 + 4 * k = 1001 :=
by {
  sorry
}

end free_endpoints_eq_1001_l226_226913


namespace triangle_angle_sine_l226_226931

theorem triangle_angle_sine (A B C : ℝ) 
  (h1 : A + B + C = real.pi) 
  (h2 : 0 < A) 
  (h3 : 0 < B) 
  (h4 : 0 < C) 
  (h5 : A < real.pi) 
  (h6 : B < real.pi) 
  (h7 : C < real.pi)
  (h8 : A + B > 0) 
  (h9 : B + C > 0) 
  (h10 : C + A > 0) 
  : real.sin (A / 2) * real.sin (B / 2) * real.sin (C / 2) ≤ 1 / 8 := 
sorry

end triangle_angle_sine_l226_226931


namespace vika_card_pairs_l226_226374

theorem vika_card_pairs : 
  let numbers := finset.range 61 \ finset.singleton 0 in
  let divs := {d | d ∈ finset.divisors 30} in
  numbers.card = 60 →
  ∀ d ∈ divs, ∀ pair : finset (ℕ × ℕ),
    pair.card = 30 →
    finset.forall₂ pair (λ x y, |x.1 - x.2| % d = |y.1 - y.2| % d) → 
    ∃ (number_of_pairs : ℕ), number_of_pairs = 8 :=
by 
  intro numbers divs hc hd hp hpairs,
  sorry

end vika_card_pairs_l226_226374


namespace limit_S_n_l226_226417

noncomputable def S (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (2 * (k + 1) + 1 : ℝ) / (((k + 1 : ℕ)^2) * ((k + 2 : ℕ)^2))

theorem limit_S_n : Tendsto (S) atTop (𝓝 1) :=
sorry

end limit_S_n_l226_226417


namespace waste_scientific_notation_correct_l226_226839

def total_waste_in_scientific : ℕ := 500000000000

theorem waste_scientific_notation_correct :
  total_waste_in_scientific = 5 * 10^10 :=
by
  sorry

end waste_scientific_notation_correct_l226_226839


namespace inclination_of_BC_l226_226009

/-- Prove that the angle of inclination of line BC is π/3 given certain points and conditions -/
theorem inclination_of_BC (A B C : ℝ × ℝ)
    (hA : A = (-2, sqrt 3))
    (hB : ∃ x : ℝ, B = (x, 0))
    (hC : C = (1, 2 * sqrt 3))
    (hA' : (-2, -sqrt 3) = (-2, -sqrt 3)) :
    ∃ θ : ℝ, θ = π / 3 ∧ 
             let kBC := (C.2 - (-sqrt 3)) / (C.1 - A.1)
             in kBC = sqrt 3 ∧
                tan θ = sqrt 3 := 
    sorry

end inclination_of_BC_l226_226009


namespace juice_water_ratio_l226_226796

theorem juice_water_ratio (V : ℝ) :
  let glass_juice_ratio := (2, 1)
  let mug_volume := 2 * V
  let mug_juice_ratio := (4, 1)
  let glass_juice_vol := (2 / 3) * V
  let glass_water_vol := (1 / 3) * V
  let mug_juice_vol := (8 / 5) * V
  let mug_water_vol := (2 / 5) * V
  let total_juice := glass_juice_vol + mug_juice_vol
  let total_water := glass_water_vol + mug_water_vol
  let ratio := total_juice / total_water
  ratio = 34 / 11 :=
by
  sorry

end juice_water_ratio_l226_226796


namespace circular_film_diameter_l226_226656

-- Definition of the problem conditions
def liquidVolume : ℝ := 576  -- volume of liquid Y in cm^3
def filmThickness : ℝ := 0.2  -- thickness of the film in cm

-- Statement of the theorem to prove the diameter of the film
theorem circular_film_diameter :
  2 * Real.sqrt (2880 / Real.pi) = 2 * Real.sqrt (liquidVolume / (filmThickness * Real.pi)) := by
  sorry

end circular_film_diameter_l226_226656


namespace sequence_value_at_2014_l226_226148

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = 1 / (a n - 1) + 1

theorem sequence_value_at_2014 :
  ∀ (a : ℕ → ℝ), sequence a → a 2014 = 3 / 2 :=
by
  intro a h
  sorry

end sequence_value_at_2014_l226_226148


namespace child_growth_l226_226818

-- Define variables for heights
def current_height : ℝ := 41.5
def previous_height : ℝ := 38.5

-- Define the problem statement in Lean 4
theorem child_growth :
  current_height - previous_height = 3 :=
by 
  sorry

end child_growth_l226_226818


namespace wolf_catches_hare_l226_226424

theorem wolf_catches_hare :
  ∀ (d_wolf_to_hare d_hare_to_hide spot_wolf_speed spot_hare_speed : ℝ),
  (d_wolf_to_hare = 30) →
  (d_hare_to_hide = 333) →
  (spot_wolf_speed = 600) →
  (spot_hare_speed = 550) →
  let d_initial := d_hare_to_hide - d_wolf_to_hare in
  let v_relative := spot_wolf_speed - spot_hare_speed in
  let t_catch := d_initial / v_relative in
  let d_hare_cover := spot_hare_speed * t_catch in
  d_hare_cover > d_hare_to_hide :=
sorry

end wolf_catches_hare_l226_226424


namespace max_members_in_band_l226_226750

noncomputable def max_band_members : ℕ :=
  let n := 100
  in 20 * n

theorem max_members_in_band (n : ℕ) :
  (20 * n ≡ 6 [MOD 28]) ∧ (20 * n ≡ 5 [MOD 19]) ∧ (20 * n < 1200) → (20 * n = 2000) :=
by
  intros h
  have h1 : 20 * n ≡ 6 [MOD 28] := h.left,
  have h2 : 20 * n ≡ 5 [MOD 19] := h.right.left,
  have h3 : 20 * n < 1200 := h.right.right,
  sorry

end max_members_in_band_l226_226750


namespace number_of_two_element_subsets_l226_226825

def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem number_of_two_element_subsets (S : Type*) [Fintype S] 
  (h : binomial_coeff (Fintype.card S) 7 = 36) :
  binomial_coeff (Fintype.card S) 2 = 36 :=
by
  sorry

end number_of_two_element_subsets_l226_226825


namespace common_root_for_equations_l226_226097

theorem common_root_for_equations : 
  ∃ p x : ℤ, 3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0 ∧ p = 3 ∧ x = 1 :=
by
  sorry

end common_root_for_equations_l226_226097


namespace find_n_exists_l226_226640

noncomputable theory

open Real

def conditions (n : ℕ) (r : ℝ) (m : ℕ) :=
  m = (n + r)^3 ∧ 0 < r ∧ r < (1 / 500 : ℝ)

theorem find_n_exists (m : ℕ) (n : ℕ) (r : ℝ) :
  (∃ m, ∃ r, conditions n r m) → n = 13 :=
by
  sorry

end find_n_exists_l226_226640


namespace vika_card_pairing_l226_226354

theorem vika_card_pairing :
  ∃ (d ∈ {1, 2, 3, 5, 6, 10, 15, 30}), ∃ (k : ℕ), 60 = 2 * d * k :=
by sorry

end vika_card_pairing_l226_226354


namespace juliet_older_than_maggie_l226_226627

-- Definitions from the given conditions
def Juliet_age : ℕ := 10
def Ralph_age (J : ℕ) : ℕ := J + 2
def Maggie_age (R : ℕ) : ℕ := 19 - R

-- Theorem statement
theorem juliet_older_than_maggie :
  Juliet_age - Maggie_age (Ralph_age Juliet_age) = 3 :=
by
  sorry

end juliet_older_than_maggie_l226_226627


namespace find_pairs_ab_product_10_abs_diff_l226_226502

theorem find_pairs_ab_product_10_abs_diff (a b : ℕ) (h : a * b = 10 * | a - b |) :
  (a, b) = (90, 9) ∨ (a, b) = (40, 8) ∨ (a, b) = (15, 6) ∨ (a, b) = (10, 5) ∨
  (a, b) = (9, 90) ∨ (a, b) = (8, 40) ∨ (a, b) = (6, 15) ∨ (a, b) = (5, 10) :=
sorry

end find_pairs_ab_product_10_abs_diff_l226_226502


namespace area_of_triangle_PQR_is_7_l226_226760

-- Define the vertices of the triangle
def P : (ℝ × ℝ) := (0, 2 : ℝ)
def Q : (ℝ × ℝ) := (3, 0 : ℝ)
def R : (ℝ × ℝ) := (1, 6 : ℝ)

-- Calculate the area of the triangle given the vertices
noncomputable def area (P Q R : (ℝ × ℝ)) : ℝ :=
  1 / 2 * |P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)|

theorem area_of_triangle_PQR_is_7 : area P Q R = 7 := by
  sorry

end area_of_triangle_PQR_is_7_l226_226760


namespace halfway_between_one_fourth_and_one_seventh_l226_226736

theorem halfway_between_one_fourth_and_one_seventh : (1 / 4 + 1 / 7) / 2 = 11 / 56 := by
  sorry

end halfway_between_one_fourth_and_one_seventh_l226_226736


namespace smallest_10_digit_number_with_largest_digit_sum_l226_226515

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |> List.sum

theorem smallest_10_digit_number_with_largest_digit_sum :
  let smallest_10_digit := 1999999999 in
  sum_of_digits smallest_10_digit = 82 ∧
  sum_of_digits smallest_10_digit > 81 :=
by
  sorry

end smallest_10_digit_number_with_largest_digit_sum_l226_226515


namespace fraction_product_equals_1_67_l226_226473

noncomputable def product_of_fractions : ℝ :=
  (2 / 1) * (2 / 3) * (4 / 3) * (4 / 5) * (6 / 5) * (6 / 7) * (8 / 7)

theorem fraction_product_equals_1_67 :
  round (product_of_fractions * 100) / 100 = 1.67 := by
  sorry

end fraction_product_equals_1_67_l226_226473


namespace ln_a2016_eq_2015_l226_226129

-- Definitions according to the conditions
def geom_seq (b : ℕ → ℝ) := ∃ r, ∀ n, b (n + 1) = b n * r

def seq_a (a : ℕ → ℝ) (b : ℕ → ℝ) := (a 0 = 1) ∧ (∀ n, a (n + 1) = a n * b n)

def e_constant := real.exp 1

-- Statement of the problem
theorem ln_a2016_eq_2015 :
  ∀ (b a : ℕ → ℝ),
  geom_seq b → 
  b 1008 = e_constant →
  seq_a a b →
  real.log (a 2016) = 2015 :=
by
  intros b a hb h1008 ha
  sorry

end ln_a2016_eq_2015_l226_226129


namespace slope_of_line_determined_by_any_two_solutions_l226_226388

theorem slope_of_line_determined_by_any_two_solutions 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 4 / x₁ + 5 / y₁ = 0) 
  (h₂ : 4 / x₂ + 5 / y₂ = 0) 
  (h_distinct : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -5 / 4 := 
sorry

end slope_of_line_determined_by_any_two_solutions_l226_226388


namespace solve_quadratic_inequality_l226_226748

theorem solve_quadratic_inequality : 
  {x : ℝ | -x^2 - 5x + 6 ≥ 0} = {x : ℝ | -6 ≤ x ∧ x ≤ 1} :=
sorry

end solve_quadratic_inequality_l226_226748


namespace least_k_for_convex_1001gon_diagonals_l226_226377

theorem least_k_for_convex_1001gon_diagonals :
  ∃ k : ℕ, (∃ S : ℕ, S = 499499) ∧ (∀ dList : List ℕ, (dList.length = 499499) →
  (∀ chosen_k : Finset ℕ, (chosen_k.card = k) → 
  chosen_k.sum (λ x, dList.nthLe x (by sorry)) ≥ (S / 2))) ∧ (k = 249750) := 
begin
  sorry
end

end least_k_for_convex_1001gon_diagonals_l226_226377


namespace smaller_octagon_half_area_l226_226702

-- Define what it means to be a regular octagon
def is_regular_octagon (O : Point) (ABCDEFGH : List Point) : Prop :=
  -- Definition capturing the properties of a regular octagon around center O
  sorry

-- Define the function that computes the area of an octagon
def area_of_octagon (ABCDEFGH : List Point) : Real :=
  sorry

-- Define the function to create the smaller octagon by joining midpoints
def smaller_octagon (ABCDEFGH : List Point) : List Point :=
  sorry

theorem smaller_octagon_half_area (O : Point) (ABCDEFGH : List Point) :
  is_regular_octagon O ABCDEFGH →
  area_of_octagon (smaller_octagon ABCDEFGH) = (1 / 2) * area_of_octagon ABCDEFGH :=
by
  sorry

end smaller_octagon_half_area_l226_226702


namespace four_digit_numbers_l226_226158

theorem four_digit_numbers (N : ℕ) : 
  (N ≥ 1000 ∧ N < 10000) ∧ 
  (N > 4999) ∧ 
  (odd ((N / 100 % 10) * (N / 10 % 10))) → 
  (1250 : ℕ) :=
sorry

end four_digit_numbers_l226_226158


namespace hyperbola_eccentricity_l226_226926

noncomputable theory

variables {a b c x y : ℝ} (h1 : a > 0) (h2 : b > 0)
def hyperbola_eq := (x*x) / (a*a) - (y*y) / (b*b) = 1
def focus := (c: ℝ) = (√(a*a + b*b))
def point_P := (x: ℝ), (y: ℝ)
def OF2 := (c, 0)
def F2 := (c, 0)
def M := ((x + c) / 2, y / 2)
def OF2_dist := (c)
def F2M_dist := (sqrt(((x - c) / 2)^2 + (y / 2)^2))
def dot_product := (c * ((x - c) / 2)) = c^2 / 2

theorem hyperbola_eccentricity (h_hyperbola : hyperbola_eq) (h_focus : focus) (h_point_P: point_P) (h_OF2_dist: OF2_dist = F2M_dist) (h_dot_product: dot_product):
  ∃ e : ℝ, e = (sqrt 3 + 1) / 2 :=
sorry

end hyperbola_eccentricity_l226_226926


namespace patricia_books_read_l226_226052

noncomputable def calculate_books := 
  λ (Candice_read : ℕ) =>
    let Amanda_read := Candice_read / 3
    let Kara_read := Amanda_read / 2
    let Patricia_read := 7 * Kara_read
    Patricia_read

theorem patricia_books_read (Candice_books : ℕ) (hC : Candice_books = 18) :
  calculate_books Candice_books = 21 := by
  rw [hC]
  unfold calculate_books
  simp
  sorry

end patricia_books_read_l226_226052


namespace julia_and_carla_items_l226_226626

noncomputable def max_items_purchased (total_money : ℝ) (cost_per_pastry : ℝ) (cost_per_premium_pastry : ℝ) (cost_per_coffee : ℝ) (discount_threshold : ℕ) : ℕ :=
  let max_pastries_without_discount := (total_money / cost_per_pastry).to_nat in
  if max_pastries_without_discount ≤ discount_threshold then
    let pastries := max_pastries_without_discount in
    let coffees := ((total_money - pastries * cost_per_pastry) / cost_per_coffee).to_nat in
    pastries + coffees
  else
    let pastries := (total_money / cost_per_premium_pastry).to_nat in
    let coffees := ((total_money - pastries * cost_per_premium_pastry) / cost_per_coffee).to_nat in
    pastries + coffees

theorem julia_and_carla_items :
  max_items_purchased 50 6 5.5 1.5 5 = 9 :=
  by
    sorry

end julia_and_carla_items_l226_226626


namespace num_zeros_in_expansion_l226_226577

-- Given conditions and definitions
def n : ℕ := 10^12 - 2

-- Proposition stating the main question and answer
theorem num_zeros_in_expansion : num_zeros_in_expansion (n^2) = 11 := by
  sorry

end num_zeros_in_expansion_l226_226577


namespace shaded_area_l226_226993

theorem shaded_area (area_large : ℝ) (area_small : ℝ) (n_small_squares : ℕ) 
  (n_triangles: ℕ) (area_total : ℝ) : 
  area_large = 16 → 
  area_small = 1 → 
  n_small_squares = 4 → 
  n_triangles = 4 → 
  area_total = 4 → 
  4 * area_small = 4 →
  area_large - (area_total + (n_small_squares * area_small)) = 4 :=
by
  intros
  sorry

end shaded_area_l226_226993


namespace jia_winning_strategy_l226_226008

variables {p q : ℝ}
def is_quadratic_real_roots (a b c : ℝ) : Prop := b ^ 2 - 4 * a * c > 0

def quadratic_with_roots (x1 x2 : ℝ) :=
  x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ is_quadratic_real_roots 1 (- (x1 + x2)) (x1 * x2)

def modify_jia (p q x1 : ℝ) : (ℝ × ℝ) := (p + 1, q - x1)

def modify_yi1 (p q : ℝ) : (ℝ × ℝ) := (p - 1, q)

def modify_yi2 (p q x2 : ℝ) : (ℝ × ℝ) := (p - 1, q + x2)

def winning_strategy_jia (x1 x2 : ℝ) : Prop :=
  ∃ n : ℕ, ∀ m ≥ n, ∀ p q, quadratic_with_roots x1 x2 → 
  (¬ is_quadratic_real_roots 1 p q) ∨ (q ≤ 0)

theorem jia_winning_strategy (x1 x2 : ℝ)
  (h: quadratic_with_roots x1 x2) : 
  winning_strategy_jia x1 x2 :=
sorry

end jia_winning_strategy_l226_226008


namespace unique_signals_l226_226429

-- Define the conditions in the problem.
def lamp_colors := {white, red, green}
def number_of_hooks := 6
def number_of_lamps := 3

-- We need to show that the number of unique signals is 471.
theorem unique_signals : 
  (number_of_unique_signals lamp_colors number_of_hooks number_of_lamps) = 471 :=
sorry

end unique_signals_l226_226429


namespace slope_of_line_determined_by_solutions_eq_l226_226392

theorem slope_of_line_determined_by_solutions_eq :
  ∀ (x y : ℝ), (4 / x + 5 / y = 0) → ∃ m : ℝ, m = -5 / 4 :=
by
  intro x y h
  use -5 / 4
  sorry

end slope_of_line_determined_by_solutions_eq_l226_226392
