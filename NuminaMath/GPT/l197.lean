import Mathlib

namespace distinct_prime_factors_2310_l197_197076

theorem distinct_prime_factors_2310 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ (S.card = 5) ∧ (S.prod id = 2310) := by
  sorry

end distinct_prime_factors_2310_l197_197076


namespace arithmetic_seq_a6_l197_197828

variable (a : ℕ → ℝ)

-- Conditions
axiom a3 : a 3 = 16
axiom a9 : a 9 = 80

-- Theorem to prove
theorem arithmetic_seq_a6 : a 6 = 48 :=
by
  sorry

end arithmetic_seq_a6_l197_197828


namespace length_of_24_l197_197184

def length_of_integer (k : ℕ) : ℕ :=
  k.factors.length

theorem length_of_24 : length_of_integer 24 = 4 :=
by
  sorry

end length_of_24_l197_197184


namespace emily_sixth_quiz_score_l197_197499

theorem emily_sixth_quiz_score (s1 s2 s3 s4 s5 : ℕ) (target_average : ℕ) : 
  s1 = 92 → s2 = 95 → s3 = 87 → s4 = 89 → s5 = 100 → target_average = 93 →
  ∃ s6 : ℕ, (s1 + s2 + s3 + s4 + s5 + s6) / 6 = target_average :=
begin
  intros h1 h2 h3 h4 h5 h6,
  use 95,
  have : (92 + 95 + 87 + 89 + 100 + 95) = 558,
  { trivial },
  simp [h1, h2, h3, h4, h5, h6, this],
  norm_num,
  exact rfl,
end

end emily_sixth_quiz_score_l197_197499


namespace solve_system_of_equations_l197_197415

theorem solve_system_of_equations (a b c x y z : ℝ)
  (h1 : a^3 + a^2 * x + a * y + z = 0)
  (h2 : b^3 + b^2 * x + b * y + z = 0)
  (h3 : c^3 + c^2 * x + c * y + z = 0) :
  x = -(a + b + c) ∧ y = ab + ac + bc ∧ z = -abc :=
by {
  sorry
}

end solve_system_of_equations_l197_197415


namespace alyssa_puppies_left_l197_197470

def initial_puppies : Nat := 7
def puppies_per_puppy : Nat := 4
def given_away : Nat := 15

theorem alyssa_puppies_left :
  (initial_puppies + initial_puppies * puppies_per_puppy) - given_away = 20 := 
  by
    sorry

end alyssa_puppies_left_l197_197470


namespace floor_x_floor_x_eq_20_l197_197505

theorem floor_x_floor_x_eq_20 (x : ℝ) : ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := 
sorry

end floor_x_floor_x_eq_20_l197_197505


namespace kenneth_fabric_amount_l197_197875

theorem kenneth_fabric_amount :
  ∃ K : ℤ, (∃ N : ℤ, N = 6 * K ∧ (K * 40 + 140000 = N * 40) ∧ K > 0) ∧ K = 700 :=
by
  sorry

end kenneth_fabric_amount_l197_197875


namespace positive_difference_median_mode_eq_nineteen_point_five_l197_197143

open Real

def data := [30, 31, 32, 33, 33, 33, 40, 41, 42, 43, 44, 45, 51, 51, 51, 52, 53, 55, 60, 61, 62, 64, 65, 67, 71, 72, 73, 74, 75, 76]
def mode := 33
def median := 52.5

def positive_difference (a b : ℝ) : ℝ := abs (a - b)

theorem positive_difference_median_mode_eq_nineteen_point_five :
  positive_difference median mode = 19.5 :=
by
  sorry

end positive_difference_median_mode_eq_nineteen_point_five_l197_197143


namespace total_marbles_l197_197906

theorem total_marbles (bowl2_capacity : ℕ) (h₁ : bowl2_capacity = 600)
    (h₂ : 3 / 4 * bowl2_capacity = 450) : 600 + (3 / 4 * 600) = 1050 := by
  sorry

end total_marbles_l197_197906


namespace hyperbola_condition_l197_197733

theorem hyperbola_condition (k : ℝ) : 
  (0 ≤ k ∧ k < 3) → (∃ a b : ℝ, a * b < 0 ∧ 
    (a = k + 1) ∧ (b = k - 5)) ∧ (∀ m : ℝ, -1 < m ∧ m < 5 → ∃ a b : ℝ, a * b < 0 ∧ 
    (a = m + 1) ∧ (b = m - 5)) :=
by
  sorry

end hyperbola_condition_l197_197733


namespace smallest_number_is_1111_in_binary_l197_197448

theorem smallest_number_is_1111_in_binary :
  let a := 15   -- Decimal equivalent of 1111 in binary
  let b := 78   -- Decimal equivalent of 210 in base 6
  let c := 64   -- Decimal equivalent of 1000 in base 4
  let d := 65   -- Decimal equivalent of 101 in base 8
  a < b ∧ a < c ∧ a < d := 
by
  let a := 15
  let b := 78
  let c := 64
  let d := 65
  show a < b ∧ a < c ∧ a < d
  sorry

end smallest_number_is_1111_in_binary_l197_197448


namespace coin_flips_probability_equal_heads_l197_197710

def fair_coin (p : ℚ) := p = 1 / 2
def second_coin (p : ℚ) := p = 3 / 5
def third_coin (p : ℚ) := p = 2 / 3

theorem coin_flips_probability_equal_heads :
  ∀ p1 p2 p3, fair_coin p1 → second_coin p2 → third_coin p3 →
  ∃ m n, m + n = 119 ∧ m / n = 29 / 90 :=
by
  sorry

end coin_flips_probability_equal_heads_l197_197710


namespace problem_solution_l197_197631

theorem problem_solution :
  ∃ f : ℝ → ℝ, (∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 < f x2) ∧ f = fun x ↦ 2^x :=
by
  use (fun x ↦ 2^x)
  sorry

end problem_solution_l197_197631


namespace num_license_plates_l197_197365

-- Let's state the number of letters in the alphabet, vowels, consonants, and digits.
def num_letters : ℕ := 26
def num_vowels : ℕ := 5  -- A, E, I, O, U and Y is not a vowel
def num_consonants : ℕ := 21  -- Remaining letters including Y
def num_digits : ℕ := 10  -- 0 through 9

-- Prove the number of five-character license plates
theorem num_license_plates : 
  (num_consonants * num_consonants * num_vowels * num_vowels * num_digits) = 110250 :=
  by 
  sorry

end num_license_plates_l197_197365


namespace sum_of_distinct_integers_l197_197792

theorem sum_of_distinct_integers 
  (p q r s : ℕ) 
  (h1 : p * q = 6) 
  (h2 : r * s = 8) 
  (h3 : p * r = 4) 
  (h4 : q * s = 12) 
  (hpqrs : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) : 
  p + q + r + s = 13 :=
sorry

end sum_of_distinct_integers_l197_197792


namespace range_of_a_l197_197210

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x - 1/2

noncomputable def g (x a : ℝ) : ℝ := x^2 + Real.log (x + a)

theorem range_of_a : 
  (∀ x ∈ Set.Iio 0, ∃ y, f x = g y a ∧ y = -x) →
  a < Real.sqrt (Real.exp 1) :=
  sorry

end range_of_a_l197_197210


namespace determine_value_of_m_l197_197494

theorem determine_value_of_m (m : ℤ) :
  2^2002 - 2^2000 - 2^1999 + 2^1998 = m * 2^1998 ↔ m = 11 := 
sorry

end determine_value_of_m_l197_197494


namespace plane_equation_l197_197114

variables {x y z : ℝ}
def w : ℝ^3 := ⟨3, -2, 3⟩
def v : ℝ^3 := ⟨x, y, z⟩
def proj_w_v : ℝ^3 := ⟨6, -4, 6⟩

theorem plane_equation (h : proj_w_v = (real_inner v w / real_inner w w) • w) :
  3 * x - 2 * y + 3 * z - 44 = 0 :=
sorry

end plane_equation_l197_197114


namespace unique_two_scoop_sundaes_l197_197163

-- Define the problem conditions
def eight_flavors : Finset String := 
  {"Vanilla", "Chocolate", "Strawberry", "Mint", "Coffee", "Pistachio", "Lemon", "Blueberry"}

def vanilla_and_chocolate_together (S : Finset (Finset String)) : Prop :=
  ∀ s ∈ S, ("Vanilla" ∈ s ∧ "Chocolate" ∈ s) ∨ ("Vanilla" ∉ s ∧ "Chocolate" ∉ s)

-- Define the unique two-scoop calculation
theorem unique_two_scoop_sundaes : 
  ∃ S : Finset (Finset String), vanilla_and_chocolate_together S ∧ S.card = 7 :=
begin
  sorry
end

end unique_two_scoop_sundaes_l197_197163


namespace parallelogram_sides_l197_197311

theorem parallelogram_sides (a b : ℝ)
  (h1 : 2 * (a + b) = 32)
  (h2 : b - a = 8) :
  a = 4 ∧ b = 12 :=
by
  -- Proof is to be provided
  sorry

end parallelogram_sides_l197_197311


namespace part_a_part_b_l197_197228

noncomputable def f_condition_a (f: ℝ → ℝ) : Prop :=
∀ (r: ℝ), ∇ (∇ f) (r : ℝ^3) = 0 → f = -C / r

noncomputable def f_condition_b (f: ℝ → ℝ) (C : ℝ) : Prop :=
∀ (r: ℝ), ∇ (λ r, f r * r) = 0 → f = C / r^3

theorem part_a {f: ℝ → ℝ} (h: f_condition_a f) : ∀ (r: ℝ), ∇ (∇ f) (r: ℝ^3) = 0 → f = -C / r := by
  sorry

theorem part_b {f: ℝ → ℝ} {C: ℝ} (h: f_condition_b f C) : ∀ (r: ℝ), ∇ (λ r, f r * r) = 0 → f = C / r^3 := by
  sorry

end part_a_part_b_l197_197228


namespace find_cheapest_option_l197_197360

variable (transportation_cost : ℕ) (berries_collected : ℕ)
          (cost_train_per_week : ℕ) (cost_berries_market : ℕ)
          (cost_sugar : ℕ) (jam_rate : ℚ) (cost_ready_made_jam : ℕ)
      
-- Define the cost of gathering 1.5 kg of jam
def option1_cost := (cost_train_per_week / berries_collected + cost_sugar) * jam_rate

-- Define the cost of buying berries and sugar to make 1.5 kg of jam
def option2_cost := (cost_berries_market + cost_sugar) * jam_rate

-- Define the cost of buying 1.5 kg of ready-made jam
def option3_cost := cost_ready_made_jam * jam_rate

theorem find_cheapest_option :
  option1_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam 
  < min (option2_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam)
        (option3_cost transportation_cost berries_collected cost_train_per_week cost_berries_market cost_sugar jam_rate cost_ready_made_jam) :=
by
  unfold option1_cost option2_cost option3_cost
  have hc1 : (40 : ℕ) + 54 = 94 := by norm_num
  have hc2 : (150 : ℕ) + 54 = 204 := by norm_num
  have hc3 : (220 : ℕ) * (3/2) = 330 := by norm_num
  linarith
  sorry

end find_cheapest_option_l197_197360


namespace find_a5_l197_197670

theorem find_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = 2 * n * (n + 1))
  (ha : ∀ n ≥ 2, a n = S n - S (n - 1)) : 
  a 5 = 20 := 
sorry

end find_a5_l197_197670


namespace isosceles_right_triangle_area_l197_197271

theorem isosceles_right_triangle_area (hypotenuse : ℝ) (leg_length : ℝ) (area : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  leg_length = hypotenuse / Real.sqrt 2 →
  area = (1 / 2) * leg_length * leg_length →
  area = 18 :=
by
  -- problem states hypotenuse is 6*sqrt(2)
  intro h₁
  -- calculus leg length from hypotenuse / sqrt(2)
  intro h₂
  -- area of the triangle from legs
  intro h₃
  -- state the desired result
  sorry

end isosceles_right_triangle_area_l197_197271


namespace line_intersects_ellipse_slopes_l197_197021

theorem line_intersects_ellipse_slopes :
  {m : ℝ | ∃ x, 4 * x^2 + 25 * (m * x + 8)^2 = 100} = 
  {m : ℝ | m ≤ -Real.sqrt 2.4 ∨ Real.sqrt 2.4 ≤ m} := 
by
  sorry

end line_intersects_ellipse_slopes_l197_197021


namespace abs_val_of_5_minus_e_l197_197033

theorem abs_val_of_5_minus_e : ∀ (e : ℝ), e = 2.718 → |5 - e| = 2.282 :=
by
  intros e he
  sorry

end abs_val_of_5_minus_e_l197_197033


namespace multiple_of_3_iff_has_odd_cycle_l197_197562

-- Define the undirected simple graph G
variable {V : Type} (G : SimpleGraph V)

-- Define the function f(G) which counts the number of acyclic orientations
def f (G : SimpleGraph V) : ℕ := sorry

-- Define what it means for a graph to have an odd-length cycle
def has_odd_cycle (G : SimpleGraph V) : Prop := sorry

-- The theorem statement
theorem multiple_of_3_iff_has_odd_cycle (G : SimpleGraph V) : 
  (f G) % 3 = 0 ↔ has_odd_cycle G := 
sorry

end multiple_of_3_iff_has_odd_cycle_l197_197562


namespace floor_e_eq_two_l197_197649

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 :=
by
  sorry

end floor_e_eq_two_l197_197649


namespace average_percent_score_is_65_point_25_l197_197762

theorem average_percent_score_is_65_point_25 :
  let percent_score : List (ℕ × ℕ) := [(95, 10), (85, 20), (75, 40), (65, 50), (55, 60), (45, 15), (35, 5)]
  let total_students : ℕ := 200
  let total_score : ℕ := percent_score.foldl (fun acc p => acc + p.1 * p.2) 0
  (total_score : ℚ) / (total_students : ℚ) = 65.25 := by
{
  sorry
}

end average_percent_score_is_65_point_25_l197_197762


namespace trigonometric_identity_l197_197086

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trigonometric_identity_l197_197086


namespace simon_number_of_legos_l197_197246

variable (Kent_legos : ℕ) (Bruce_legos : ℕ) (Simon_legos : ℕ)

def Kent_condition : Prop := Kent_legos = 40
def Bruce_condition : Prop := Bruce_legos = Kent_legos + 20 
def Simon_condition : Prop := Simon_legos = Bruce_legos + (Bruce_legos * 20 / 100)

theorem simon_number_of_legos : Kent_condition Kent_legos ∧ Bruce_condition Kent_legos Bruce_legos ∧ Simon_condition Bruce_legos Simon_legos → Simon_legos = 72 := by
  intros h
  -- proof steps would go here
  sorry

end simon_number_of_legos_l197_197246


namespace ratio_of_black_to_white_areas_l197_197974

theorem ratio_of_black_to_white_areas :
  let π := Real.pi
  let radii := [2, 4, 6, 8]
  let areas := [π * (radii[0])^2, π * (radii[1])^2, π * (radii[2])^2, π * (radii[3])^2]
  let black_areas := [areas[0], areas[2] - areas[1]]
  let white_areas := [areas[1] - areas[0], areas[3] - areas[2]]
  let total_black_area := black_areas.sum
  let total_white_area := white_areas.sum
  let ratio := total_black_area / total_white_area
  ratio = 3 / 5 := sorry

end ratio_of_black_to_white_areas_l197_197974


namespace evaluateExpression_correct_l197_197324

open Real

noncomputable def evaluateExpression : ℝ :=
  (-2)^2 + 2 * sin (π / 3) - tan (π / 3)

theorem evaluateExpression_correct : evaluateExpression = 4 :=
  sorry

end evaluateExpression_correct_l197_197324


namespace sum_digit_product_1001_to_2011_l197_197925

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).foldr (λ d acc => d * acc) 1

theorem sum_digit_product_1001_to_2011 :
  (Finset.range 1011).sum (λ k => digit_product (1001 + k)) = 91125 :=
by
  sorry

end sum_digit_product_1001_to_2011_l197_197925


namespace convert_500_to_base2_l197_197328

theorem convert_500_to_base2 :
  let n_base10 : ℕ := 500
  let n_base8 : ℕ := 7 * 64 + 6 * 8 + 4
  let n_base2 : ℕ := 1 * 256 + 1 * 128 + 1 * 64 + 1 * 32 + 1 * 16 + 0 * 8 + 1 * 4 + 0 * 2 + 0
  n_base10 = 500 ∧ n_base8 = 500 ∧ n_base2 = n_base8 :=
by
  sorry

end convert_500_to_base2_l197_197328


namespace first_train_speed_l197_197154

theorem first_train_speed:
  ∃ v : ℝ, 
    (∀ t : ℝ, t = 1 → (v * t) + (4 * v) = 200) ∧ 
    (∀ t : ℝ, t = 4 → 50 * t = 200) → 
    v = 40 :=
by {
 sorry
}

end first_train_speed_l197_197154


namespace negated_roots_quadratic_reciprocals_roots_quadratic_l197_197244

-- For (1)
theorem negated_roots_quadratic (x y : ℝ) : 
    (x^2 + 3 * x - 2 = 0) ↔ (y^2 - 3 * y - 2 = 0) :=
sorry

-- For (2)
theorem reciprocals_roots_quadratic (a b c x y : ℝ) (h : a ≠ 0) :
    (a * x^2 - b * x + c = 0) ↔ (c * y^2 - b * y + a = 0) :=
sorry

end negated_roots_quadratic_reciprocals_roots_quadratic_l197_197244


namespace mark_total_spending_l197_197568

variable (p_tomato_cost : ℕ) (p_apple_cost : ℕ) 
variable (pounds_tomato : ℕ) (pounds_apple : ℕ)

def total_cost (p_tomato_cost : ℕ) (pounds_tomato : ℕ) (p_apple_cost : ℕ) (pounds_apple : ℕ) : ℕ :=
  (p_tomato_cost * pounds_tomato) + (p_apple_cost * pounds_apple)

theorem mark_total_spending :
  total_cost 5 2 6 5 = 40 :=
by
  sorry

end mark_total_spending_l197_197568


namespace vertical_shirts_count_l197_197611

-- Definitions from conditions
def total_people : ℕ := 40
def checkered_shirts : ℕ := 7
def horizontal_shirts := 4 * checkered_shirts

-- Proof goal
theorem vertical_shirts_count :
  ∃ vertical_shirts : ℕ, vertical_shirts = total_people - (checkered_shirts + horizontal_shirts) ∧ vertical_shirts = 5 :=
sorry

end vertical_shirts_count_l197_197611


namespace factorization_of_x4_plus_81_l197_197421

theorem factorization_of_x4_plus_81 :
  ∀ x : ℝ, x^4 + 81 = (x^2 - 3 * x + 4.5) * (x^2 + 3 * x + 4.5) :=
by
  intros x
  sorry

end factorization_of_x4_plus_81_l197_197421


namespace solution_set_inequalities_l197_197820

theorem solution_set_inequalities (a b x : ℝ) (h1 : ∃ x, x > a ∧ x < b) :
  (x < 1 - a ∧ x < 1 - b) ↔ x < 1 - b :=
by
  sorry

end solution_set_inequalities_l197_197820


namespace percentage_discount_l197_197325

theorem percentage_discount (original_price sale_price : ℝ) (h1 : original_price = 25) (h2 : sale_price = 18.75) : 
  100 * (original_price - sale_price) / original_price = 25 := 
by
  -- Begin Proof
  sorry

end percentage_discount_l197_197325


namespace value_of_a2_b2_l197_197367

theorem value_of_a2_b2 (a b : ℝ) (i : ℂ) (hi : i^2 = -1) (h : (a - i) * i = b - i) : a^2 + b^2 = 2 :=
by sorry

end value_of_a2_b2_l197_197367


namespace volume_at_10_l197_197791

noncomputable def gas_volume (T : ℝ) : ℝ :=
  if T = 30 then 40 else 40 - (30 - T) / 5 * 5

theorem volume_at_10 :
  gas_volume 10 = 20 :=
by
  simp [gas_volume]
  sorry

end volume_at_10_l197_197791


namespace baseball_weight_l197_197144

theorem baseball_weight
  (weight_total : ℝ)
  (weight_soccer_ball : ℝ)
  (n_soccer_balls : ℕ)
  (n_baseballs : ℕ)
  (total_weight : ℝ)
  (B : ℝ) :
  n_soccer_balls * weight_soccer_ball + n_baseballs * B = total_weight →
  n_soccer_balls = 9 →
  weight_soccer_ball = 0.8 →
  n_baseballs = 7 →
  total_weight = 10.98 →
  B = 0.54 := sorry

end baseball_weight_l197_197144


namespace washer_total_cost_l197_197402

variable (C : ℝ)
variable (h : 0.25 * C = 200)

theorem washer_total_cost : C = 800 :=
by
  sorry

end washer_total_cost_l197_197402


namespace sophia_finished_more_pages_l197_197256

noncomputable def length_of_book : ℝ := 89.99999999999999

noncomputable def total_pages : ℕ := 90  -- Considering the practical purpose

noncomputable def finished_pages : ℕ := total_pages * 2 / 3

noncomputable def remaining_pages : ℕ := total_pages - finished_pages

theorem sophia_finished_more_pages :
  finished_pages - remaining_pages = 30 := 
  by
    -- Use sorry here as placeholder for the proof
    sorry

end sophia_finished_more_pages_l197_197256


namespace cos_2theta_plus_pi_l197_197063

-- Given condition
def tan_theta_eq_2 (θ : ℝ) : Prop := Real.tan θ = 2

-- The mathematical statement to prove
theorem cos_2theta_plus_pi (θ : ℝ) (h : tan_theta_eq_2 θ) : Real.cos (2 * θ + Real.pi) = 3 / 5 := 
sorry

end cos_2theta_plus_pi_l197_197063


namespace min_value_l197_197675

theorem min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : 
  ∃ min_val, min_val = 5 + 2 * Real.sqrt 6 ∧ (∀ x, (x = 5 + 2 * Real.sqrt 6) → x ≥ min_val) :=
by
  sorry

end min_value_l197_197675


namespace MN_intersection_correct_l197_197679

-- Define the sets M and N
def setM : Set ℝ := {y | ∃ x ∈ (Set.univ : Set ℝ), y = x^2 + 2*x - 3}
def setN : Set ℝ := {x | |x - 2| ≤ 3}

-- Reformulated sets
def setM_reformulated : Set ℝ := {y | y ≥ -4}
def setN_reformulated : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- The intersection set
def MN_intersection : Set ℝ := {y | -1 ≤ y ∧ y ≤ 5}

-- The theorem stating the intersection of M and N equals MN_intersection
theorem MN_intersection_correct :
  {y | ∃ x ∈ setN_reformulated, y = x^2 + 2*x - 3} = MN_intersection :=
sorry  -- Proof not required as per instruction

end MN_intersection_correct_l197_197679


namespace phone_price_in_october_l197_197275

variable (a : ℝ) (P_October : ℝ) (r : ℝ)

noncomputable def price_in_january := a
noncomputable def price_in_october (a : ℝ) (r : ℝ) := a * r^9

theorem phone_price_in_october :
  r = 0.97 →
  P_October = price_in_october a r →
  P_October = a * (0.97)^9 :=
by
  intros h1 h2
  rw [h1] at h2
  exact h2

end phone_price_in_october_l197_197275


namespace sin_cos_15_deg_l197_197943

noncomputable def sin_deg (deg : ℝ) : ℝ := Real.sin (deg * Real.pi / 180)
noncomputable def cos_deg (deg : ℝ) : ℝ := Real.cos (deg * Real.pi / 180)

theorem sin_cos_15_deg :
  (sin_deg 15 + cos_deg 15) * (sin_deg 15 - cos_deg 15) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_cos_15_deg_l197_197943


namespace area_of_rectangle_l197_197340

-- Define the problem statement and conditions
theorem area_of_rectangle (p d : ℝ) :
  ∃ A : ℝ, (∀ (x y : ℝ), 2 * x + 2 * y = p ∧ x^2 + y^2 = d^2 → A = x * y) →
  A = (p^2 - 4 * d^2) / 8 :=
by 
  sorry

end area_of_rectangle_l197_197340


namespace expand_product_polynomials_l197_197657

noncomputable def poly1 : Polynomial ℤ := 5 * Polynomial.X + 3
noncomputable def poly2 : Polynomial ℤ := 7 * Polynomial.X^2 + 2 * Polynomial.X + 4
noncomputable def expanded_form : Polynomial ℤ := 35 * Polynomial.X^3 + 31 * Polynomial.X^2 + 26 * Polynomial.X + 12

theorem expand_product_polynomials :
  poly1 * poly2 = expanded_form := 
by
  sorry

end expand_product_polynomials_l197_197657


namespace sin_double_angle_l197_197097

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ = 2) : Real.sin (2 * θ) = 4 / 5 :=
by
  sorry

end sin_double_angle_l197_197097


namespace expected_number_of_1s_rolling_two_dice_l197_197746

def prob_roll_1 := (1 : ℚ) / 6
def prob_roll_not_1 := (5 : ℚ) / 6
def prob_zero_1s := prob_roll_not_1 ^ 2
def prob_two_1s := prob_roll_1 ^ 2
def prob_one_1  := 1 - prob_zero_1s - prob_two_1s

theorem expected_number_of_1s_rolling_two_dice : 
  ∑ i in {0, 1, 2}, i * (if i = 0 then prob_zero_1s else if i = 1 then prob_one_1 else prob_two_1s) = 1 / 3 :=
by
  sorry

end expected_number_of_1s_rolling_two_dice_l197_197746


namespace proof_of_neg_p_or_neg_q_l197_197351

variables (p q : Prop)

theorem proof_of_neg_p_or_neg_q (h₁ : ¬ (p ∧ q)) (h₂ : p ∨ q) : ¬ p ∨ ¬ q :=
  sorry

end proof_of_neg_p_or_neg_q_l197_197351


namespace grandmother_cheapest_option_l197_197362

-- Conditions definition
def cost_of_transportation : Nat := 200
def berries_collected : Nat := 5
def market_price_berries : Nat := 150
def price_sugar : Nat := 54
def amount_jam_from_1kg_berries_sugar : ℚ := 1.5
def cost_ready_made_jam_per_kg : Nat := 220

-- Calculations
def cost_per_kg_berries : ℚ := cost_of_transportation / berries_collected
def cost_bought_berries : Nat := market_price_berries
def total_cost_1kg_self_picked : ℚ := cost_per_kg_berries + price_sugar
def total_cost_1kg_bought : Nat := cost_bought_berries + price_sugar
def total_cost_1_5kg_self_picked : ℚ := total_cost_1kg_self_picked
def total_cost_1_5kg_bought : ℚ := total_cost_1kg_bought
def total_cost_1_5kg_ready_made : ℚ := cost_ready_made_jam_per_kg * amount_jam_from_1kg_berries_sugar

theorem grandmother_cheapest_option :
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_bought ∧ 
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_ready_made :=
  by
    sorry

end grandmother_cheapest_option_l197_197362


namespace total_pages_l197_197726

def reading_rate (pages : ℕ) (minutes : ℕ) : ℝ :=
  pages / minutes

def total_pages_read (rate : ℝ) (minutes : ℕ) : ℝ :=
  rate * minutes

theorem total_pages (t : ℕ) (rene_pages : ℕ) (rene_minutes : ℕ) (lulu_pages : ℕ) (lulu_minutes : ℕ) (cherry_pages : ℕ) (cherry_minutes : ℕ) :
  t = 240 →
  rene_pages = 30 →
  rene_minutes = 60 →
  lulu_pages = 27 →
  lulu_minutes = 60 →
  cherry_pages = 25 →
  cherry_minutes = 60 →
  total_pages_read (reading_rate rene_pages rene_minutes) t +
  total_pages_read (reading_rate lulu_pages lulu_minutes) t +
  total_pages_read (reading_rate cherry_pages cherry_minutes) t = 328 :=
by
  intros t_val rene_p_val rene_m_val lulu_p_val lulu_m_val cherry_p_val cherry_m_val
  rw [t_val, rene_p_val, rene_m_val, lulu_p_val, lulu_m_val, cherry_p_val, cherry_m_val]
  simp [reading_rate, total_pages_read]
  norm_num
  sorry

end total_pages_l197_197726


namespace value_to_subtract_l197_197698

theorem value_to_subtract (N x : ℕ) (h1 : (N - x) / 7 = 7) (h2 : (N - 34) / 10 = 2) : x = 5 :=
by 
  sorry

end value_to_subtract_l197_197698


namespace divisibility_by_5_l197_197911

theorem divisibility_by_5 (B : ℕ) (hB : B < 10) : (476 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := 
by
  sorry

end divisibility_by_5_l197_197911


namespace mark_profit_l197_197868

variable (initial_cost tripling_factor new_value profit : ℕ)

-- Conditions
def initial_card_cost := 100
def card_tripling_factor := 3

-- Calculations based on conditions
def card_new_value := initial_card_cost * card_tripling_factor
def card_profit := card_new_value - initial_card_cost

-- Proof Statement
theorem mark_profit (initial_card_cost tripling_factor card_new_value card_profit : ℕ) 
  (h1: initial_card_cost = 100)
  (h2: tripling_factor = 3)
  (h3: card_new_value = initial_card_cost * tripling_factor)
  (h4: card_profit = card_new_value - initial_card_cost) :
  card_profit = 200 :=
  by sorry

end mark_profit_l197_197868


namespace problem1_problem2_l197_197944

-- Problem 1: Prove (-a^3)^2 * (-a^2)^3 / a = -a^11 given a is a real number.
theorem problem1 (a : ℝ) : (-a^3)^2 * (-a^2)^3 / a = -a^11 :=
  sorry

-- Problem 2: Prove (m - n)^3 * (n - m)^4 * (n - m)^5 = - (n - m)^12 given m, n are real numbers.
theorem problem2 (m n : ℝ) : (m - n)^3 * (n - m)^4 * (n - m)^5 = - (n - m)^12 :=
  sorry

end problem1_problem2_l197_197944


namespace smallest_number_is_28_l197_197285

theorem smallest_number_is_28 (a b c : ℕ) (h1 : (a + b + c) / 3 = 30) (h2 : b = 28) (h3 : b = c - 6) : a = 28 :=
by sorry

end smallest_number_is_28_l197_197285


namespace length_of_each_reel_l197_197217

theorem length_of_each_reel
  (reels : ℕ)
  (sections : ℕ)
  (length_per_section : ℕ)
  (total_sections : ℕ)
  (h1 : reels = 3)
  (h2 : length_per_section = 10)
  (h3 : total_sections = 30)
  : (total_sections * length_per_section) / reels = 100 := 
by
  sorry

end length_of_each_reel_l197_197217


namespace base_three_to_decimal_l197_197260

theorem base_three_to_decimal :
  let n := 20121 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 178 :=
by {
  sorry
}

end base_three_to_decimal_l197_197260


namespace ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth_l197_197229

theorem ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : 0 < a * b * c)
  : a * b + b * c + c * a < (Real.sqrt (a * b * c)) / 2 + 1 / 4 := 
sorry

end ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth_l197_197229


namespace remainder_is_three_l197_197234

def dividend : ℕ := 15
def divisor : ℕ := 3
def quotient : ℕ := 4

theorem remainder_is_three : dividend = (divisor * quotient) + Nat.mod dividend divisor := by
  sorry

end remainder_is_three_l197_197234


namespace solve_abs_equation_l197_197883

theorem solve_abs_equation (y : ℝ) (h : |y - 8| + 3 * y = 12) : y = 2 :=
sorry

end solve_abs_equation_l197_197883


namespace solve_for_x_l197_197413

theorem solve_for_x (x : ℝ) (h₁ : (7 * x) / (x + 4) - 4 / (x + 4) = 2 / (x + 4)) (h₂ : x ≠ -4) : x = 6 / 7 :=
by
  sorry

end solve_for_x_l197_197413


namespace find_b_value_l197_197705

-- Definitions based on given conditions
def original_line (x : ℝ) (b : ℝ) : ℝ := 2 * x + b
def shifted_line (x : ℝ) (b : ℝ) : ℝ := 2 * (x - 2) + b
def passes_through_origin (b : ℝ) := shifted_line 0 b = 0

-- Main proof statement
theorem find_b_value (b : ℝ) (h : passes_through_origin b) : b = 4 := by
  sorry

end find_b_value_l197_197705


namespace y_intercept_of_line_b_l197_197858

-- Define the conditions
def line_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def point_on_line (m b x y : ℝ) : Prop := y = m * x + b

-- Given conditions
variables (m b : ℝ)
variable (x₁ := 3)
variable (y₁ := -2)
axiom parallel_condition : line_parallel m (-3)
axiom point_condition : point_on_line m b x₁ y₁

-- Prove that the y-intercept b equals 7
theorem y_intercept_of_line_b : b = 7 :=
sorry

end y_intercept_of_line_b_l197_197858


namespace solve_y_from_expression_l197_197491

-- Define the conditions
def given_conditions := (784 = 28^2) ∧ (49 = 7^2)

-- Define the equivalency to prove based on the given conditions
theorem solve_y_from_expression (h : given_conditions) : 784 + 2 * 28 * 7 + 49 = 1225 := by
  sorry

end solve_y_from_expression_l197_197491


namespace sqrt_number_is_169_l197_197742

theorem sqrt_number_is_169 (a b : ℝ) 
  (h : a^2 + b^2 + (4 * a - 6 * b + 13) = 0) : 
  (a^2 + b^2)^2 = 169 :=
sorry

end sqrt_number_is_169_l197_197742


namespace arithmetic_progression_x_value_l197_197264

theorem arithmetic_progression_x_value (x: ℝ) (h1: 3*x - 1 - (2*x - 3) = 4*x + 1 - (3*x - 1)) : x = 3 :=
by
  sorry

end arithmetic_progression_x_value_l197_197264


namespace union_complement_eq_complement_intersection_eq_l197_197072

-- Define the universal set U and sets A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

-- Theorem 1: A ∪ (U \ B) = {2, 4, 5, 6}
theorem union_complement_eq : A ∪ (U \ B) = {2, 4, 5, 6} := by
  sorry

-- Theorem 2: U \ (A ∩ B) = {1, 2, 3, 4, 6, 7}
theorem complement_intersection_eq : U \ (A ∩ B) = {1, 2, 3, 4, 6, 7} := by
  sorry

end union_complement_eq_complement_intersection_eq_l197_197072


namespace determine_k_l197_197335

theorem determine_k (k : ℝ) : 
  (∀ x : ℝ, (x^2 = 2 * x + k) → (∃ x0 : ℝ, ∀ x : ℝ, (x - x0)^2 = 0)) ↔ k = -1 :=
by 
  sorry

end determine_k_l197_197335


namespace percentage_decrease_of_larger_angle_l197_197429

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end percentage_decrease_of_larger_angle_l197_197429


namespace connie_earbuds_tickets_l197_197326

theorem connie_earbuds_tickets (total_tickets : ℕ) (koala_fraction : ℕ) (bracelet_tickets : ℕ) (earbud_tickets : ℕ) :
  total_tickets = 50 →
  koala_fraction = 2 →
  bracelet_tickets = 15 →
  (total_tickets / koala_fraction) + bracelet_tickets + earbud_tickets = total_tickets →
  earbud_tickets = 10 :=
by
  intros h_total h_koala h_bracelets h_sum
  sorry

end connie_earbuds_tickets_l197_197326


namespace price_of_adult_ticket_l197_197462

theorem price_of_adult_ticket (total_payment : ℕ) (child_price : ℕ) (difference : ℕ) (children : ℕ) (adults : ℕ) (A : ℕ)
  (h1 : total_payment = 720) 
  (h2 : child_price = 8) 
  (h3 : difference = 25) 
  (h4 : children = 15)
  (h5 : adults = children + difference)
  (h6 : total_payment = children * child_price + adults * A) :
  A = 15 :=
by
  sorry

end price_of_adult_ticket_l197_197462


namespace car_dealer_bmw_sales_l197_197759

theorem car_dealer_bmw_sales (total_cars : ℕ)
  (vw_percentage : ℝ)
  (toyota_percentage : ℝ)
  (acura_percentage : ℝ)
  (bmw_count : ℕ) :
  total_cars = 300 →
  vw_percentage = 0.10 →
  toyota_percentage = 0.25 →
  acura_percentage = 0.20 →
  bmw_count = total_cars * (1 - (vw_percentage + toyota_percentage + acura_percentage)) →
  bmw_count = 135 :=
by
  intros
  sorry

end car_dealer_bmw_sales_l197_197759


namespace original_price_of_shoes_l197_197060

theorem original_price_of_shoes (x : ℝ) (h : 1/4 * x = 18) : x = 72 := by
  sorry

end original_price_of_shoes_l197_197060


namespace ratio_steel_iron_is_5_to_2_l197_197161

-- Definitions based on the given conditions
def amount_steel : ℕ := 35
def amount_iron : ℕ := 14

-- Main statement
theorem ratio_steel_iron_is_5_to_2 :
  (amount_steel / Nat.gcd amount_steel amount_iron) = 5 ∧
  (amount_iron / Nat.gcd amount_steel amount_iron) = 2 :=
by
  sorry

end ratio_steel_iron_is_5_to_2_l197_197161


namespace distance_from_P_to_AB_l197_197744

-- Definitions of conditions
def is_point_in_triangle (P A B C : ℝ×ℝ) : Prop := sorry
def parallel_to_base (P A B C : ℝ×ℝ) : Prop := sorry
def divides_area_in_ratio (P A B C : ℝ×ℝ) (r1 r2 : ℕ) : Prop := sorry

theorem distance_from_P_to_AB (P A B C : ℝ×ℝ) 
  (H_in_triangle : is_point_in_triangle P A B C)
  (H_parallel : parallel_to_base P A B C)
  (H_area_ratio : divides_area_in_ratio P A B C 1 3)
  (H_altitude : ∃ h : ℝ, h = 1) :
  ∃ d : ℝ, d = 3/4 :=
by
  sorry

end distance_from_P_to_AB_l197_197744


namespace solution_set_l197_197987

variable (f : ℝ → ℝ)

def cond1 := ∀ x, f x = f (-x)
def cond2 := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
def cond3 := f (1/3) = 0

theorem solution_set (hf1 : cond1 f) (hf2 : cond2 f) (hf3 : cond3 f) :
  { x : ℝ | f (Real.log x / Real.log (1/8)) > 0 } = { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | 2 < x } :=
sorry

end solution_set_l197_197987


namespace geom_series_sum_l197_197483

theorem geom_series_sum : 
  let a := 1 : ℝ
  let r := 1 / 4 : ℝ
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geom_series_sum_l197_197483


namespace quarter_more_than_whole_l197_197296

theorem quarter_more_than_whole (x : ℝ) (h : x / 4 = 9 + x) : x = -12 :=
by
  sorry

end quarter_more_than_whole_l197_197296


namespace monomials_exponents_l197_197699

theorem monomials_exponents (m n : ℕ) 
  (h₁ : 3 * x ^ 5 * y ^ m + -2 * x ^ n * y ^ 7 = 0) : m - n = 2 := 
by
  sorry

end monomials_exponents_l197_197699


namespace num_integers_between_700_and_900_with_sum_of_digits_18_l197_197684

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem num_integers_between_700_and_900_with_sum_of_digits_18 : 
  ∃ k, k = 17 ∧ ∀ n, 700 ≤ n ∧ n ≤ 900 ∧ sum_of_digits n = 18 ↔ (1 ≤ k) := 
sorry

end num_integers_between_700_and_900_with_sum_of_digits_18_l197_197684


namespace isosceles_triangle_area_l197_197269

theorem isosceles_triangle_area (h : ℝ) (a : ℝ) (A : ℝ) 
  (h_eq : h = 6 * sqrt 2) 
  (h_leg : h = a * sqrt 2) 
  (area_eq : A = 1 / 2 * a^2) : 
  A = 18 :=
by
  sorry

end isosceles_triangle_area_l197_197269


namespace trigonometric_identity_l197_197081

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end trigonometric_identity_l197_197081


namespace viewing_spot_coordinate_correct_l197_197595

-- Define the coordinates of the landmarks
def first_landmark := 150
def second_landmark := 450

-- The expected coordinate of the viewing spot
def expected_viewing_spot := 350

-- The theorem that formalizes the problem
theorem viewing_spot_coordinate_correct :
  let distance := second_landmark - first_landmark
  let fractional_distance := (2 / 3) * distance
  let viewing_spot := first_landmark + fractional_distance
  viewing_spot = expected_viewing_spot := 
by
  -- This is where the proof would go
  sorry

end viewing_spot_coordinate_correct_l197_197595


namespace find_x_l197_197853

variable (x : ℤ)
def A : Set ℤ := {x^2, x + 1, -3}
def B : Set ℤ := {x - 5, 2 * x - 1, x^2 + 1}

theorem find_x (h : A x ∩ B x = {-3}) : x = -1 :=
sorry

end find_x_l197_197853


namespace least_possible_area_l197_197313

variable (x y : ℝ) (n : ℤ)

-- Conditions
def is_integer (x : ℝ) := ∃ k : ℤ, x = k
def is_half_integer (y : ℝ) := ∃ n : ℤ, y = n + 0.5

-- Problem statement in Lean 4
theorem least_possible_area (h1 : is_integer x) (h2 : is_half_integer y)
(h3 : 2 * (x + y) = 150) : ∃ A, A = 0 :=
sorry

end least_possible_area_l197_197313


namespace tan_pi_div_a_of_point_on_cubed_function_l197_197800

theorem tan_pi_div_a_of_point_on_cubed_function (a : ℝ) (h : (a, 27) ∈ {p : ℝ × ℝ | p.snd = p.fst ^ 3}) : 
  Real.tan (Real.pi / a) = Real.sqrt 3 := sorry

end tan_pi_div_a_of_point_on_cubed_function_l197_197800


namespace shuttlecock_weight_probability_l197_197129

variable (p_lt_4_8 : ℝ) -- Probability that its weight is less than 4.8 g
variable (p_le_4_85 : ℝ) -- Probability that its weight is not greater than 4.85 g

theorem shuttlecock_weight_probability (h1 : p_lt_4_8 = 0.3) (h2 : p_le_4_85 = 0.32) :
  p_le_4_85 - p_lt_4_8 = 0.02 :=
by
  sorry

end shuttlecock_weight_probability_l197_197129


namespace count_integers_with_sum_of_digits_18_l197_197685

def sum_of_digits (n : ℕ) : ℕ := (n / 100) + (n / 10 % 10) + (n % 10)

def valid_integer_count : ℕ :=
  let range := List.range' 700 (900 - 700 + 1)
  List.length $ List.filter (λ n => sum_of_digits n = 18) range

theorem count_integers_with_sum_of_digits_18 :
  valid_integer_count = 17 :=
sorry

end count_integers_with_sum_of_digits_18_l197_197685


namespace parabola_difference_eq_l197_197619

variable (a b c : ℝ)

def original_parabola (x : ℝ) : ℝ := a * x^2 + b * x + c
def reflected_parabola (x : ℝ) : ℝ := -(a * x^2 + b * x + c)
def translated_original (x : ℝ) : ℝ := a * x^2 + b * x + c + 3
def translated_reflection (x : ℝ) : ℝ := -(a * x^2 + b * x + c) - 3

theorem parabola_difference_eq (x : ℝ) :
  (translated_original a b c x) - (translated_reflection a b c x) = 2 * a * x^2 + 2 * b * x + 2 * c + 6 :=
by 
  sorry

end parabola_difference_eq_l197_197619


namespace find_absolute_value_l197_197995

theorem find_absolute_value (h k : ℤ) (h1 : 3 * (-3)^3 - h * (-3) + k = 0) (h2 : 3 * 2^3 - h * 2 + k = 0) : |3 * h - 2 * k| = 27 :=
by
  sorry

end find_absolute_value_l197_197995


namespace isosceles_triangle_base_length_l197_197774

theorem isosceles_triangle_base_length
  (perimeter : ℝ)
  (side1 side2 base : ℝ)
  (h_perimeter : perimeter = 18)
  (h_side1 : side1 = 4)
  (h_isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base)
  (h_triangle : side1 + side2 + base = 18) :
  base = 7 := 
sorry

end isosceles_triangle_base_length_l197_197774


namespace arithmetic_square_root_16_l197_197732

theorem arithmetic_square_root_16 : ∀ x : ℝ, x ≥ 0 → x^2 = 16 → x = 4 :=
by
  intro x hx h
  sorry

end arithmetic_square_root_16_l197_197732


namespace monotonicity_and_no_x_intercept_l197_197389

noncomputable theory

def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

def is_monotonic (a : ℝ) (x : ℝ) : Prop := 
  if x < 1 / a then 
    f a x > f a (x + ε) -- ε is a small positive value
  else if x > 1 / a then 
    f a x < f a (x - ε)
  else
    true -- At x = 1/a, the function transits from decreasing to increasing

theorem monotonicity_and_no_x_intercept 
  (a : ℝ) (h1 : 0 < a) : 
  (∀ x : ℝ, 0 < x → is_monotonic a x) ∧ 
  (∀ x : ℝ, f a x ≠ 0) ↔ 
  (a ∈ Ioi (1 / real.exp 1)) := 
sorry

end monotonicity_and_no_x_intercept_l197_197389


namespace zoo_animal_count_l197_197156

def tiger_enclosures : ℕ := 4
def zebra_enclosures_per_tiger_enclosures : ℕ := 2
def zebra_enclosures : ℕ := tiger_enclosures * zebra_enclosures_per_tiger_enclosures
def giraffe_enclosures_per_zebra_enclosures : ℕ := 3
def giraffe_enclosures : ℕ := zebra_enclosures * giraffe_enclosures_per_zebra_enclosures
def tigers_per_enclosure : ℕ := 4
def zebras_per_enclosure : ℕ := 10
def giraffes_per_enclosure : ℕ := 2

def total_animals_in_zoo : ℕ := 
    (tiger_enclosures * tigers_per_enclosure) + 
    (zebra_enclosures * zebras_per_enclosure) + 
    (giraffe_enclosures * giraffes_per_enclosure)

theorem zoo_animal_count : total_animals_in_zoo = 144 := 
by
  -- proof would go here
  sorry

end zoo_animal_count_l197_197156


namespace coeff_x3_in_x_mul_1_add_x_pow_6_l197_197492

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem coeff_x3_in_x_mul_1_add_x_pow_6 :
  ∀ x : ℕ, (∃ c : ℕ, c * x^3 = x * (1 + x)^6 ∧ c = 15) :=
by
  sorry

end coeff_x3_in_x_mul_1_add_x_pow_6_l197_197492


namespace find_stock_face_value_l197_197291

theorem find_stock_face_value
  (cost_price : ℝ) -- Definition for the cost price
  (discount_rate : ℝ) -- Definition for the discount rate
  (brokerage_rate : ℝ) -- Definition for the brokerage rate
  (h1 : cost_price = 98.2) -- Condition: The cost price is 98.2
  (h2 : discount_rate = 0.02) -- Condition: The discount rate is 2%
  (h3 : brokerage_rate = 0.002) -- Condition: The brokerage rate is 1/5% (0.002)
  : ∃ X : ℝ, 0.982 * X = cost_price ∧ X = 100 := -- Theorem statement to prove
by
  -- Proof omitted
  sorry

end find_stock_face_value_l197_197291


namespace percent_decrease_l197_197010

theorem percent_decrease (P S : ℝ) (h₀ : P = 100) (h₁ : S = 70) :
  ((P - S) / P) * 100 = 30 :=
by
  sorry

end percent_decrease_l197_197010


namespace Fred_last_week_l197_197842

-- Definitions from conditions
def Fred_now := 40
def Fred_earned := 21

-- The theorem we need to prove
theorem Fred_last_week :
  Fred_now - Fred_earned = 19 :=
by
  sorry

end Fred_last_week_l197_197842


namespace redPoints_l197_197708

open Nat

def isRedPoint (x y : ℕ) : Prop :=
  (y = (x - 36) * (x - 144) - 1991) ∧ (∃ m : ℕ, y = m * m)

theorem redPoints :
  {p : ℕ × ℕ | isRedPoint p.1 p.2} = { (2544, 6017209), (444, 120409) } :=
by
  sorry

end redPoints_l197_197708


namespace count_valid_sequences_l197_197077

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_sequence (x : ℕ → ℕ) : Prop :=
  (x 7 % 2 = 0) ∧ (∀ i < 7, (x i % 2 = 0 → x (i + 1) % 2 = 1) ∧ (x i % 2 = 1 → x (i + 1) % 2 = 0))

theorem count_valid_sequences : ∃ n, 
  n = 78125 ∧ 
  ∃ x : ℕ → ℕ, 
    (∀ i < 8, 0 ≤ x i ∧ x i ≤ 9) ∧ valid_sequence x :=
sorry

end count_valid_sequences_l197_197077


namespace exponent_m_n_add_l197_197535

variable (a : ℝ) (m n : ℕ)

theorem exponent_m_n_add (h1 : a ^ m = 2) (h2 : a ^ n = 3) : a ^ (m + n) = 6 := by
  sorry

end exponent_m_n_add_l197_197535


namespace trig_expression_simplify_l197_197094

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trig_expression_simplify_l197_197094


namespace expression_undefined_at_x_l197_197174

theorem expression_undefined_at_x (x : ℝ) : (x^2 - 18 * x + 81 = 0) → x = 9 :=
by {
  sorry
}

end expression_undefined_at_x_l197_197174


namespace not_all_acute_angled_triangles_l197_197795

theorem not_all_acute_angled_triangles (A B C D : Point)
  (h1 : ¬(Collinear A B C))
  (h2 : ¬(Collinear A B D))
  (h3 : ¬(Collinear A C D))
  (h4 : ¬(Collinear B C D))
  : ∃ (P Q R : Point), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ Angle.is_not_acute (angle P Q R) := 
sorry

end not_all_acute_angled_triangles_l197_197795


namespace num_supermarkets_us_l197_197899

noncomputable def num_supermarkets_total : ℕ := 84

noncomputable def us_canada_relationship (C : ℕ) : Prop := C + (C + 10) = num_supermarkets_total

theorem num_supermarkets_us (C : ℕ) (h : us_canada_relationship C) : C + 10 = 47 :=
sorry

end num_supermarkets_us_l197_197899


namespace minimum_value_exists_l197_197182

noncomputable def minimized_function (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y + y^3

theorem minimum_value_exists :
  ∃ (x y : ℝ), minimized_function x y = minimized_function (4/3 - 2 * y/3) y :=
sorry

end minimum_value_exists_l197_197182


namespace smallest_mul_seven_perfect_square_l197_197749

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define the problem statement
theorem smallest_mul_seven_perfect_square :
  ∀ x : ℕ, x > 0 → (is_perfect_square (7 * x) ↔ x = 7) := 
by {
  sorry
}

end smallest_mul_seven_perfect_square_l197_197749


namespace break_25_ruble_bill_l197_197286

theorem break_25_ruble_bill (x y z : ℕ) :
  (x + y + z = 11 ∧ 1 * x + 3 * y + 5 * z = 25) ↔ 
    (x = 4 ∧ y = 7 ∧ z = 0) ∨ 
    (x = 5 ∧ y = 5 ∧ z = 1) ∨ 
    (x = 6 ∧ y = 3 ∧ z = 2) ∨ 
    (x = 7 ∧ y = 1 ∧ z = 3) :=
sorry

end break_25_ruble_bill_l197_197286


namespace factorize_polynomial_l197_197502

theorem factorize_polynomial (m : ℤ) : 4 * m^2 - 16 = 4 * (m + 2) * (m - 2) := by
  sorry

end factorize_polynomial_l197_197502


namespace time_to_reach_6400ft_is_200min_l197_197767

noncomputable def time_to_reach_ship (depth : ℕ) (rate : ℕ) : ℕ :=
  depth / rate

theorem time_to_reach_6400ft_is_200min :
  time_to_reach_ship 6400 32 = 200 := by
  sorry

end time_to_reach_6400ft_is_200min_l197_197767


namespace range_of_b_l197_197065

theorem range_of_b (b : ℝ) :
  (∀ x y : ℝ, (x ≠ y) → (y = 1/3 * x^3 + b * x^2 + (b + 2) * x + 3) → (y ≥ 1/3 * x^3 + b * x^2 + (b + 2) * x + 3))
  ↔ (-1 ≤ b ∧ b ≤ 2) :=
sorry

end range_of_b_l197_197065


namespace days_c_worked_l197_197007

theorem days_c_worked 
    (days_a : ℕ) (days_b : ℕ) (wage_ratio_a : ℚ) (wage_ratio_b : ℚ) (wage_ratio_c : ℚ)
    (total_earnings : ℚ) (wage_c : ℚ) :
    days_a = 16 →
    days_b = 9 →
    wage_ratio_a = 3 →
    wage_ratio_b = 4 →
    wage_ratio_c = 5 →
    wage_c = 71.15384615384615 →
    total_earnings = 1480 →
    ∃ days_c : ℕ, (total_earnings = (wage_ratio_a / wage_ratio_c * wage_c * days_a) + 
                                 (wage_ratio_b / wage_ratio_c * wage_c * days_b) + 
                                 (wage_c * days_c)) ∧ days_c = 4 :=
by
  intros
  sorry

end days_c_worked_l197_197007


namespace floor_x_floor_x_eq_20_l197_197506

theorem floor_x_floor_x_eq_20 (x : ℝ) : ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := 
sorry

end floor_x_floor_x_eq_20_l197_197506


namespace remainder_when_two_pow_thirty_three_div_nine_l197_197445

-- Define the base and the exponent
def base : ℕ := 2
def exp : ℕ := 33
def modulus : ℕ := 9

-- The main statement to prove
theorem remainder_when_two_pow_thirty_three_div_nine :
  (base ^ exp) % modulus = 8 :=
by
  sorry

end remainder_when_two_pow_thirty_three_div_nine_l197_197445


namespace ship_speed_in_still_water_l197_197316

theorem ship_speed_in_still_water 
  (distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (x : ℝ) 
  (h1 : distance = 36)
  (h2 : time = 6)
  (h3 : current_speed = 3) 
  (h4 : (18 / (x + 3) + 18 / (x - 3) = 6)) 
  : x = 3 + 3 * Real.sqrt 2 :=
sorry

end ship_speed_in_still_water_l197_197316


namespace photos_in_each_album_l197_197632

theorem photos_in_each_album (total_photos : ℕ) (number_of_albums : ℕ) (photos_per_album : ℕ) 
    (h1 : total_photos = 2560) 
    (h2 : number_of_albums = 32) 
    (h3 : total_photos = number_of_albums * photos_per_album) : 
    photos_per_album = 80 := 
by 
    sorry

end photos_in_each_album_l197_197632


namespace geometric_series_sum_l197_197486

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℚ) / 4
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geometric_series_sum_l197_197486


namespace zinc_to_copper_ratio_l197_197947

theorem zinc_to_copper_ratio (total_weight zinc_weight copper_weight : ℝ) 
  (h1 : total_weight = 64) 
  (h2 : zinc_weight = 28.8) 
  (h3 : copper_weight = total_weight - zinc_weight) : 
  (zinc_weight / 0.4) / (copper_weight / 0.4) = 9 / 11 :=
by
  sorry

end zinc_to_copper_ratio_l197_197947


namespace polynomial_non_negative_l197_197408

theorem polynomial_non_negative (x : ℝ) : x^8 + x^6 - 4*x^4 + x^2 + 1 ≥ 0 := 
sorry

end polynomial_non_negative_l197_197408


namespace cookie_ratio_l197_197145

theorem cookie_ratio (f : ℚ) (h_monday : 32 = 32) (h_tuesday : (f : ℚ) * 32 = 32 * (f : ℚ)) 
    (h_wednesday : 3 * (f : ℚ) * 32 - 4 + 32 + (f : ℚ) * 32 = 92) :
    f = 1/2 :=
by
  sorry

end cookie_ratio_l197_197145


namespace circle_m_condition_l197_197262

theorem circle_m_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + m = 0) → m < 5 :=
by
  sorry

end circle_m_condition_l197_197262


namespace max_value_of_a_l197_197798

theorem max_value_of_a (a b c d : ℤ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : d < 100) : a ≤ 2367 := by 
  sorry

end max_value_of_a_l197_197798


namespace coord_of_point_B_l197_197547
-- Necessary import for mathematical definitions and structures

-- Define the initial point A and the translation conditions
def point_A : ℝ × ℝ := (1, -2)
def translation_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ := (p.1, p.2 + units)

-- The target point B after translation
def point_B := translation_up point_A 1

-- The theorem to prove that the coordinates of B are (1, -1)
theorem coord_of_point_B : point_B = (1, -1) :=
by
  -- Placeholder for proof
  sorry

end coord_of_point_B_l197_197547


namespace kenneth_and_ellen_l197_197745

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 1000)

def kenneth_sum (a : ℂ) : ℂ :=
  ∑ k in Finset.range 1000, 1 / (omega^k - a)

def ellen_sum (a : ℂ) : ℂ :=
  ∑ k in Finset.range 1000, 1 / omega^k - 1000 * a

theorem kenneth_and_ellen (a : ℂ) (h : kenneth_sum a = ellen_sum a) :
  a = 0 ∨ (a^1000 - a^998 - 1 = 0) :=
sorry

end kenneth_and_ellen_l197_197745


namespace evaluate_expression_l197_197533

def operation_star (A B : ℕ) : ℕ := (A + B) / 2
def operation_ominus (A B : ℕ) : ℕ := A - B

theorem evaluate_expression :
  operation_ominus (operation_star 6 10) (operation_star 2 4) = 5 := 
by 
  sorry

end evaluate_expression_l197_197533


namespace compare_pi_314_compare_neg_sqrt3_neg_sqrt2_compare_2_sqrt5_l197_197489

theorem compare_pi_314 : Real.pi > 3.14 :=
by sorry

theorem compare_neg_sqrt3_neg_sqrt2 : -Real.sqrt 3 < -Real.sqrt 2 :=
by sorry

theorem compare_2_sqrt5 : 2 < Real.sqrt 5 :=
by sorry

end compare_pi_314_compare_neg_sqrt3_neg_sqrt2_compare_2_sqrt5_l197_197489


namespace find_x_l197_197756

theorem find_x (x : ℕ) (h1 : x ≥ 10) (h2 : x > 8) : x = 9 := by
  sorry

end find_x_l197_197756


namespace Elberta_has_23_dollars_l197_197363

theorem Elberta_has_23_dollars :
  let granny_smith_amount := 63
  let anjou_amount := 1 / 3 * granny_smith_amount
  let elberta_amount := anjou_amount + 2
  elberta_amount = 23 := by
  sorry

end Elberta_has_23_dollars_l197_197363


namespace project_completion_days_l197_197940

theorem project_completion_days (A_days : ℕ) (B_days : ℕ) (A_alone_days : ℕ) :
  A_days = 20 → B_days = 25 → A_alone_days = 2 → (A_alone_days : ℚ) * (1 / A_days) + (10 : ℚ) * (1 / (A_days * B_days / (A_days + B_days))) = 1 :=
by
  sorry

end project_completion_days_l197_197940


namespace sequence_value_l197_197805

theorem sequence_value (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) : a 5 = 17 :=
by
  -- The proof is not required, so we add sorry to indicate that
  sorry

end sequence_value_l197_197805


namespace original_amount_charged_l197_197843

variables (P : ℝ) (interest_rate : ℝ) (total_owed : ℝ)

theorem original_amount_charged :
  interest_rate = 0.09 →
  total_owed = 38.15 →
  (P + P * interest_rate = total_owed) →
  P = 35 :=
by
  intros h_interest_rate h_total_owed h_equation
  sorry

end original_amount_charged_l197_197843


namespace range_of_a_l197_197525

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x > 2 ↔ x < 2 / (a - 1)) → a < 1 :=
by
  sorry

end range_of_a_l197_197525


namespace converse_proposition_true_l197_197449

theorem converse_proposition_true (x y : ℝ) (h : x > abs y) : x > y := 
by
sorry

end converse_proposition_true_l197_197449


namespace sale_record_is_negative_five_l197_197629

-- Given that a purchase of 10 items is recorded as +10
def purchase_record (items : Int) : Int := items

-- Prove that the sale of 5 items should be recorded as -5
theorem sale_record_is_negative_five : purchase_record 10 = 10 → purchase_record (-5) = -5 :=
by
  intro h
  sorry

end sale_record_is_negative_five_l197_197629


namespace digit_equation_l197_197047

-- Define the digits for the letters L, O, V, E, and S in base 10.
def digit_L := 4
def digit_O := 3
def digit_V := 7
def digit_E := 8
def digit_S := 6

-- Define the numeral representations.
def LOVE := digit_L * 1000 + digit_O * 100 + digit_V * 10 + digit_E
def EVOL := digit_E * 1000 + digit_V * 100 + digit_O * 10 + digit_L
def SOLVES := digit_S * 100000 + digit_O * 10000 + digit_L * 1000 + digit_V * 100 + digit_E * 10 + digit_S

-- Prove that LOVE + EVOL + LOVE = SOLVES in base 10.
theorem digit_equation :
  LOVE + EVOL + LOVE = SOLVES :=
by
  -- Proof is omitted; include a proper proof in your verification process.
  sorry

end digit_equation_l197_197047


namespace percentage_of_earrings_l197_197475

theorem percentage_of_earrings (B M R : ℕ) (hB : B = 10) (hM : M = 2 * R) (hTotal : B + M + R = 70) : 
  (B * 100) / M = 25 := 
by
  sorry

end percentage_of_earrings_l197_197475


namespace sequence_term_number_l197_197709

theorem sequence_term_number (n : ℕ) (a_n : ℕ) (h : a_n = 2 * n ^ 2 - 3) : a_n = 125 → n = 8 :=
by
  sorry

end sequence_term_number_l197_197709


namespace coordinates_of_A_l197_197345

-- Define initial coordinates of point A
def A : ℝ × ℝ := (-2, 4)

-- Define the transformation of moving 2 units upwards
def move_up (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 + units)

-- Define the transformation of moving 3 units to the left
def move_left (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 - units, point.2)

-- Combine the transformations to get point A'
def A' : ℝ × ℝ :=
  move_left (move_up A 2) 3

-- The theorem stating that A' is (-5, 6)
theorem coordinates_of_A' : A' = (-5, 6) :=
by
  sorry

end coordinates_of_A_l197_197345


namespace row_3_seat_6_representation_l197_197105

-- Given Conditions
def seat_representation (r : ℕ) (s : ℕ) : (ℕ × ℕ) :=
  (r, s)

-- Proof Statement
theorem row_3_seat_6_representation :
  seat_representation 3 6 = (3, 6) :=
by
  sorry

end row_3_seat_6_representation_l197_197105


namespace common_root_value_l197_197511

theorem common_root_value (a : ℝ) : 
  (∃ x : ℝ, x^2 + a * x + 8 = 0 ∧ x^2 + x + a = 0) ↔ a = -6 :=
sorry

end common_root_value_l197_197511


namespace percent_decrease_internet_cost_l197_197580

theorem percent_decrease_internet_cost :
  ∀ (initial_cost final_cost : ℝ), initial_cost = 120 → final_cost = 45 → 
  ((initial_cost - final_cost) / initial_cost) * 100 = 62.5 :=
by
  intros initial_cost final_cost h_initial h_final
  sorry

end percent_decrease_internet_cost_l197_197580


namespace monotonicity_of_f_range_of_a_if_no_zeros_l197_197396

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l197_197396


namespace num_integers_between_700_and_900_with_sum_of_digits_18_l197_197683

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem num_integers_between_700_and_900_with_sum_of_digits_18 : 
  ∃ k, k = 17 ∧ ∀ n, 700 ≤ n ∧ n ≤ 900 ∧ sum_of_digits n = 18 ↔ (1 ≤ k) := 
sorry

end num_integers_between_700_and_900_with_sum_of_digits_18_l197_197683


namespace fill_sacks_times_l197_197488

-- Define the capacities of the sacks
def father_sack_capacity : ℕ := 20
def senior_ranger_sack_capacity : ℕ := 30
def volunteer_sack_capacity : ℕ := 25
def number_of_volunteers : ℕ := 2

-- Total wood gathered
def total_wood_gathered : ℕ := 200

-- Statement of the proof problem
theorem fill_sacks_times : (total_wood_gathered / (father_sack_capacity + senior_ranger_sack_capacity + (number_of_volunteers * volunteer_sack_capacity))) = 2 := by
  sorry

end fill_sacks_times_l197_197488


namespace Emily_sixth_quiz_score_l197_197500

theorem Emily_sixth_quiz_score :
  let scores := [92, 95, 87, 89, 100]
  ∃ s : ℕ, (s + scores.sum : ℚ) / 6 = 93 :=
  by
    sorry

end Emily_sixth_quiz_score_l197_197500


namespace diff_of_squares_635_615_l197_197918

theorem diff_of_squares_635_615 : 635^2 - 615^2 = 25000 :=
by
  sorry

end diff_of_squares_635_615_l197_197918


namespace sum_a_b_neg1_l197_197814

-- Define the problem using the given condition
theorem sum_a_b_neg1 (a b : ℝ) (h : |a + 3| + (b - 2) ^ 2 = 0) : a + b = -1 := 
by
  sorry

end sum_a_b_neg1_l197_197814


namespace max_value_of_expr_l197_197064

theorem max_value_of_expr 
  (x y z : ℝ) 
  (h₀ : 0 < x) 
  (h₁ : 0 < y) 
  (h₂ : 0 < z)
  (h : x^2 + y^2 + z^2 = 1) : 
  3 * x * y + y * z ≤ (Real.sqrt 10) / 2 := 
  sorry

end max_value_of_expr_l197_197064


namespace maximum_small_circles_l197_197878

-- Definitions for small circle radius, large circle radius, and the maximum number n.
def smallCircleRadius : ℝ := 1
def largeCircleRadius : ℝ := 11

-- Function to check if small circles can be placed without overlapping
def canPlaceCircles (n : ℕ) : Prop := n * 2 < 2 * Real.pi * (largeCircleRadius - smallCircleRadius)

theorem maximum_small_circles : ∀ n : ℕ, canPlaceCircles n → n ≤ 31 := by
  sorry

end maximum_small_circles_l197_197878


namespace monotonicity_and_no_x_intercept_l197_197390

noncomputable theory

def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

def is_monotonic (a : ℝ) (x : ℝ) : Prop := 
  if x < 1 / a then 
    f a x > f a (x + ε) -- ε is a small positive value
  else if x > 1 / a then 
    f a x < f a (x - ε)
  else
    true -- At x = 1/a, the function transits from decreasing to increasing

theorem monotonicity_and_no_x_intercept 
  (a : ℝ) (h1 : 0 < a) : 
  (∀ x : ℝ, 0 < x → is_monotonic a x) ∧ 
  (∀ x : ℝ, f a x ≠ 0) ↔ 
  (a ∈ Ioi (1 / real.exp 1)) := 
sorry

end monotonicity_and_no_x_intercept_l197_197390


namespace geometric_series_sum_eq_4_over_3_l197_197480

theorem geometric_series_sum_eq_4_over_3 : 
  let a := 1
  let r := 1/4
  inf_geometric_sum a r = 4/3 := by
begin
  intros,
  sorry
end

end geometric_series_sum_eq_4_over_3_l197_197480


namespace ratio_of_segments_l197_197724

theorem ratio_of_segments (E F G H : ℝ) (h_collinear : E < F ∧ F < G ∧ G < H)
  (hEF : F - E = 3) (hFG : G - F = 6) (hEH : H - E = 20) : (G - E) / (H - F) = 9 / 17 := by
  sorry

end ratio_of_segments_l197_197724


namespace regular_polygon_sides_l197_197826

-- Define the main theorem statement
theorem regular_polygon_sides (n : ℕ) : 
  (n > 2) ∧ 
  ((n - 2) * 180 / n - 360 / n = 90) → 
  n = 8 := by
  sorry

end regular_polygon_sides_l197_197826


namespace height_of_isosceles_triangle_l197_197028

variable (s : ℝ) (h : ℝ) (A : ℝ)
variable (triangle : ∀ (s : ℝ) (h : ℝ), A = 0.5 * (2 * s) * h)
variable (rectangle : ∀ (s : ℝ), A = s^2)

theorem height_of_isosceles_triangle (s : ℝ) (h : ℝ) (A : ℝ) (triangle : ∀ (s : ℝ) (h : ℝ), A = 0.5 * (2 * s) * h)
  (rectangle : ∀ (s : ℝ), A = s^2) : h = s := by
  sorry

end height_of_isosceles_triangle_l197_197028


namespace remainder_when_divided_by_13_is_11_l197_197003

theorem remainder_when_divided_by_13_is_11 
  (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : 
  349 % 13 = 11 := 
by 
  sorry

end remainder_when_divided_by_13_is_11_l197_197003


namespace geom_series_sum_l197_197484

theorem geom_series_sum : 
  let a := 1 : ℝ
  let r := 1 / 4 : ℝ
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geom_series_sum_l197_197484


namespace find_m_l197_197268

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 3*x + m
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem find_m (m : ℝ) : 3 * f 4 m = g 4 m → m = 4 :=
by 
  sorry

end find_m_l197_197268


namespace ConfuciusBirthYear_l197_197423

-- Definitions based on the conditions provided
def birthYearAD (year : Int) : Int := year

def birthYearBC (year : Int) : Int := -year

theorem ConfuciusBirthYear :
  birthYearBC 551 = -551 :=
by
  sorry

end ConfuciusBirthYear_l197_197423


namespace value_of_f_at_3_l197_197266

def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem value_of_f_at_3 : f 3 = 15 :=
by
  -- This proof needs to be filled in
  sorry

end value_of_f_at_3_l197_197266


namespace jessica_monthly_car_insurance_payment_l197_197712

theorem jessica_monthly_car_insurance_payment
  (rent_last_year : ℤ := 1000)
  (food_last_year : ℤ := 200)
  (car_insurance_last_year : ℤ)
  (rent_increase_rate : ℕ := 3 / 10)
  (food_increase_rate : ℕ := 1 / 2)
  (car_insurance_increase_rate : ℕ := 3)
  (additional_expenses_this_year : ℤ := 7200) :
  car_insurance_last_year = 300 :=
by
  sorry

end jessica_monthly_car_insurance_payment_l197_197712


namespace FGH_supermarkets_US_l197_197594

/-- There are 60 supermarkets in the FGH chain,
all of them are either in the US or Canada,
there are 14 more FGH supermarkets in the US than in Canada.
Prove that there are 37 FGH supermarkets in the US. -/
theorem FGH_supermarkets_US (C U : ℕ) (h1 : C + U = 60) (h2 : U = C + 14) : U = 37 := by
  sorry

end FGH_supermarkets_US_l197_197594


namespace depth_of_melted_ice_cream_l197_197464

theorem depth_of_melted_ice_cream
  (r_sphere : ℝ) (r_cylinder : ℝ) (V_sphere : ℝ)
  (h : ℝ)
  (sphere_volume_eq : V_sphere = (4 / 3) * Real.pi * r_sphere^3)
  (cylinder_volume_eq : V_sphere = Real.pi * r_cylinder^2 * h)
  (r_sphere_eq : r_sphere = 3)
  (r_cylinder_eq : r_cylinder = 9)
  : h = 4 / 9 :=
by
  -- Proof is omitted
  sorry

end depth_of_melted_ice_cream_l197_197464


namespace integer_root_sum_abs_l197_197664

theorem integer_root_sum_abs :
  ∃ a b c m : ℤ, 
    (a + b + c = 0 ∧ ab + bc + ca = -2023 ∧ |a| + |b| + |c| = 94) := sorry

end integer_root_sum_abs_l197_197664


namespace monotonicity_of_even_function_l197_197810

-- Define the function and its properties
def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + 2*m*x + 3

-- A function is even if f(x) = f(-x) for all x
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

-- The main theorem statement
theorem monotonicity_of_even_function :
  ∀ (m : ℝ), is_even (f m) → (f 0 = 3) ∧ (∀ x : ℝ, f 0 x = - x^2 + 3) →
  (∀ a b, -3 < a ∧ a < b ∧ b < 1 → f 0 a < f 0 b → f 0 b > f 0 a) :=
by
  intro m
  intro h
  intro H
  sorry

end monotonicity_of_even_function_l197_197810


namespace solve_for_x_l197_197539

theorem solve_for_x (x y : ℝ) (h₁ : y = 1 / (4 * x + 2)) (h₂ : y = 1 / 2) : x = 0 :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_l197_197539


namespace rational_solution_l197_197560

theorem rational_solution (a b c : ℚ) 
  (h : (3 * a - 2 * b + c - 4)^2 + (a + 2 * b - 3 * c + 6)^2 + (2 * a - b + 2 * c - 2)^2 ≤ 0) : 
  2 * a + b - 4 * c = -4 := 
by
  sorry

end rational_solution_l197_197560


namespace monotonicity_and_no_real_roots_l197_197388

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

theorem monotonicity_and_no_real_roots 
  (a : ℝ) (ha : 0 < a) : 
  (∀ x : ℝ, (0 < x ∧ x < (1 / a) → deriv (f a) x < 0) ∧ (x > (1 / a) → deriv (f a) x > 0)) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≠ 0 → a > (1 / real.exp 1))) :=
begin
  sorry
end

end monotonicity_and_no_real_roots_l197_197388


namespace average_comparison_l197_197454

theorem average_comparison (x : ℝ) : 
    (14 + 32 + 53) / 3 = 3 + (21 + 47 + x) / 3 → 
    x = 22 :=
by 
  sorry

end average_comparison_l197_197454


namespace relationship_between_M_and_N_l197_197668

variable (a : ℝ)

def M : ℝ := 2 * a * (a - 2) + 4
def N : ℝ := (a - 1) * (a - 3)

theorem relationship_between_M_and_N : M a > N a :=
by sorry

end relationship_between_M_and_N_l197_197668


namespace probability_of_product_form_l197_197103

open set classical function

noncomputable def expressions : set (ℚ → ℚ → ℚ) := {λ (x y : ℚ), x + y, λ (x y : ℚ), x + 5*y, λ (x y : ℚ), x - y, λ (x y : ℚ), 5*x + y}

def is_form_x_squared_minus_by_squared (f g : ℚ → ℚ → ℚ) : Prop :=
∃ (b : ℚ), ∀ (x y : ℚ), f x y * g x y = x^2 - (b * y)^2

theorem probability_of_product_form :
  (finset.univ.image (λ (p : finset (ℚ → ℚ → ℚ)), p.1))
    (finset.filter (λ (p : (ℚ → ℚ → ℚ) × (ℚ → ℚ → ℚ)), is_form_x_squared_minus_by_squared p.1 p.2)
      (finset.univ.image (λ (p : finset (ℚ → ℚ → ℚ)), p.1))).card =
  (1 / 6 : ℚ) :=
sorry

end probability_of_product_form_l197_197103


namespace scientific_notation_representation_l197_197300

theorem scientific_notation_representation :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_representation_l197_197300


namespace mass_of_cork_l197_197621

theorem mass_of_cork (ρ_p ρ_w ρ_s : ℝ) (m_p x : ℝ) :
  ρ_p = 2.15 * 10^4 → 
  ρ_w = 2.4 * 10^2 →
  ρ_s = 4.8 * 10^2 →
  m_p = 86.94 →
  x = 2.4 * 10^2 * (m_p / ρ_p) →
  x = 85 :=
by
  intros
  sorry

end mass_of_cork_l197_197621


namespace tangent_lines_to_circle_through_point_l197_197333

noncomputable def circle_center : ℝ × ℝ := (1, 2)
noncomputable def circle_radius : ℝ := 2
noncomputable def point_P : ℝ × ℝ := (-1, 5)

theorem tangent_lines_to_circle_through_point :
  ∃ m c : ℝ, (∀ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = 4 → (m * x + y + c = 0 → (y = -m * x - c))) ∧
  (m = 5/12 ∧ c = -55/12) ∨ (m = 0 ∧ ∀ x : ℝ, x = -1) :=
sorry

end tangent_lines_to_circle_through_point_l197_197333


namespace inequalities_always_hold_l197_197536

theorem inequalities_always_hold (x y a b : ℝ) (hxy : x > y) (hab : a > b) :
  (a + x > b + y) ∧ (x - b > y - a) :=
by
  sorry

end inequalities_always_hold_l197_197536


namespace quadratic_inequality_solutions_l197_197966

theorem quadratic_inequality_solutions {k : ℝ} (h1 : 0 < k) (h2 : k < 16) :
  ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end quadratic_inequality_solutions_l197_197966


namespace triangle_inequality_l197_197119

theorem triangle_inequality (a b c S : ℝ)
  (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)   -- a, b, c are sides of a non-isosceles triangle
  (S_def : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  (a^3) / ((a - b) * (a - c)) + (b^3) / ((b - c) * (b - a)) + (c^3) / ((c - a) * (c - b)) > 2 * 3^(3/4) * S :=
by
  sorry

end triangle_inequality_l197_197119


namespace mark_total_spending_l197_197570

theorem mark_total_spending:
  let cost_per_pound_tomatoes := 5
  let pounds_tomatoes := 2
  let cost_per_pound_apples := 6
  let pounds_apples := 5
  let cost_tomatoes := cost_per_pound_tomatoes * pounds_tomatoes
  let cost_apples := cost_per_pound_apples * pounds_apples
  let total_spending := cost_tomatoes + cost_apples
  total_spending = 40 :=
by
  sorry

end mark_total_spending_l197_197570


namespace arccos_one_half_l197_197171

theorem arccos_one_half :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_one_half_l197_197171


namespace machines_together_work_time_l197_197566

theorem machines_together_work_time :
  let rate_A := 1 / 4
  let rate_B := 1 / 12
  let rate_C := 1 / 6
  let rate_D := 1 / 8
  let rate_E := 1 / 18
  let total_rate := rate_A + rate_B + rate_C + rate_D + rate_E
  total_rate ≠ 0 → 
  let total_time := 1 / total_rate
  total_time = 72 / 49 :=
by
  sorry

end machines_together_work_time_l197_197566


namespace sufficient_but_not_necessary_condition_l197_197676

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|)
  ∧ (∃ y : ℝ, ¬ (y ≥ 1) ∧ |y + 1| + |y - 1| = 2 * |y|) :=
by
  sorry

end sufficient_but_not_necessary_condition_l197_197676


namespace length_of_platform_l197_197612

theorem length_of_platform {train_length platform_crossing_time signal_pole_crossing_time : ℚ}
  (h_train_length : train_length = 300)
  (h_platform_crossing_time : platform_crossing_time = 40)
  (h_signal_pole_crossing_time : signal_pole_crossing_time = 18) :
  ∃ L : ℚ, L = 1100 / 3 :=
by
  sorry

end length_of_platform_l197_197612


namespace linear_combination_value_l197_197637

theorem linear_combination_value (x y : ℝ) (h₁ : 2 * x + y = 8) (h₂ : x + 2 * y = 10) :
  8 * x ^ 2 + 10 * x * y + 8 * y ^ 2 = 164 :=
sorry

end linear_combination_value_l197_197637


namespace choose_athlete_B_l197_197030

variable (SA2 : ℝ) (SB2 : ℝ)
variable (num_shots : ℕ) (avg_rings : ℝ)

-- Conditions
def athlete_A_variance := SA2 = 3.5
def athlete_B_variance := SB2 = 2.8
def same_number_of_shots := true -- Implicit condition, doesn't need proof
def same_average_rings := true -- Implicit condition, doesn't need proof

-- Question: prove Athlete B should be chosen
theorem choose_athlete_B 
  (hA_var : athlete_A_variance SA2)
  (hB_var : athlete_B_variance SB2)
  (same_shots : same_number_of_shots)
  (same_avg : same_average_rings) :
  "B" = "B" :=
by 
  sorry

end choose_athlete_B_l197_197030


namespace divide_milk_in_half_l197_197451

theorem divide_milk_in_half (bucket : ℕ) (a : ℕ) (b : ℕ) (a_liters : a = 5) (b_liters : b = 7) (bucket_liters : bucket = 12) :
  ∃ x y : ℕ, x = 6 ∧ y = 6 ∧ x + y = bucket := by
  sorry

end divide_milk_in_half_l197_197451


namespace inequality_problem_l197_197516

theorem inequality_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (a + 1 / (2 * b^2)) * (b + 1 / (2 * a^2)) ≥ 25 / 4 := 
sorry

end inequality_problem_l197_197516


namespace percentage_of_knives_after_trade_l197_197781
open scoped Rat

theorem percentage_of_knives_after_trade :
  ∀ (initial_knives : ℕ) (initial_forks : ℕ) (initial_spoons : ℕ) (trade_knives : ℕ) (trade_spoons : ℕ),
    initial_knives = 6 →
    initial_forks = 12 →
    initial_spoons = 3 * initial_knives →
    trade_knives = 10 →
    trade_spoons = 6 →
    (16 : ℚ) / (40 : ℚ) * (100 : ℚ) = 40 :=
begin
  sorry
end

end percentage_of_knives_after_trade_l197_197781


namespace no_integer_solution_l197_197409

theorem no_integer_solution (x y z : ℤ) (h : x ≠ 0) : ¬(2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) :=
sorry

end no_integer_solution_l197_197409


namespace range_of_function_l197_197172

theorem range_of_function : 
  (∃ x : ℝ, |x + 5| - |x - 3| + 4 = 12) ∧ 
  (∃ x : ℝ, |x + 5| - |x - 3| + 4 = 18) ∧ 
  (∀ y : ℝ, (12 ≤ y ∧ y ≤ 18) → 
    ∃ x : ℝ, y = |x + 5| - |x - 3| + 4) :=
by
  sorry

end range_of_function_l197_197172


namespace merchant_problem_l197_197022

theorem merchant_problem (P C : ℝ) (h1 : P + C = 60) (h2 : 2.40 * P + 6.00 * C = 180) : C = 10 := 
by
  -- Proof goes here
  sorry

end merchant_problem_l197_197022


namespace find_number_mul_l197_197057

theorem find_number_mul (n : ℕ) (h : n * 9999 = 724777430) : n = 72483 :=
by
  sorry

end find_number_mul_l197_197057


namespace floor_e_eq_two_l197_197648

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 :=
by
  sorry

end floor_e_eq_two_l197_197648


namespace num_neg_values_of_x_l197_197058

theorem num_neg_values_of_x 
  (n : ℕ) 
  (xn_pos_int : ∃ k, n = k ∧ k > 0) 
  (sqrt_x_169_pos_int : ∀ x, ∃ m, x + 169 = m^2 ∧ m > 0) :
  ∃ count, count = 12 := 
by
  sorry

end num_neg_values_of_x_l197_197058


namespace sqrt_37_between_6_and_7_l197_197501

theorem sqrt_37_between_6_and_7 : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := 
by 
  have h₁ : Real.sqrt 36 = 6 := by sorry
  have h₂ : Real.sqrt 49 = 7 := by sorry
  sorry

end sqrt_37_between_6_and_7_l197_197501


namespace mark_profit_from_selling_magic_card_l197_197870

theorem mark_profit_from_selling_magic_card : 
    ∀ (purchase_price new_value profit : ℕ), 
        purchase_price = 100 ∧ 
        new_value = 3 * purchase_price ∧ 
        profit = new_value - purchase_price 
    → 
        profit = 200 := 
by 
  intros purchase_price new_value profit h,
  cases h with hp1 h,
  cases h with hv1 hp2,
  rw hp1 at hv1,
  rw hp1 at hp2,
  rw hv1 at hp2,
  rw hp2,
  rw hp1,
  norm_num,
  exact eq.refl 200

end mark_profit_from_selling_magic_card_l197_197870


namespace circumcircle_incircle_inequality_l197_197290

theorem circumcircle_incircle_inequality
  (a b : ℝ)
  (h_a : a = 16)
  (h_b : b = 11)
  (R r : ℝ)
  (triangle_inequality : ∀ c : ℝ, 5 < c ∧ c < 27) :
  R ≥ 2.2 * r := sorry

end circumcircle_incircle_inequality_l197_197290


namespace abs_neg2023_eq_2023_l197_197578

-- Define a function to represent the absolute value
def abs (x : ℤ) : ℤ :=
  if x < 0 then -x else x

-- Prove that abs (-2023) = 2023
theorem abs_neg2023_eq_2023 : abs (-2023) = 2023 :=
by
  -- In this theorem, all necessary definitions are already included
  sorry

end abs_neg2023_eq_2023_l197_197578


namespace largest_possible_value_b_l197_197561

theorem largest_possible_value_b : 
  ∃ b : ℚ, (3 * b + 7) * (b - 2) = 4 * b ∧ b = 40 / 15 := 
by 
  sorry

end largest_possible_value_b_l197_197561


namespace relationship_S_T_l197_197979

def S (n : ℕ) : ℤ := 2^n
def T (n : ℕ) : ℤ := 2^n - (-1)^n

theorem relationship_S_T (n : ℕ) (h : n > 0) : 
  (n % 2 = 1 → S n < T n) ∧ (n % 2 = 0 → S n > T n) :=
by
  sorry

end relationship_S_T_l197_197979


namespace each_player_gets_seven_l197_197833

-- Define the total number of dominoes and players
def total_dominoes : Nat := 28
def total_players : Nat := 4

-- Define the question for how many dominoes each player would receive
def dominoes_per_player (dominoes players : Nat) : Nat := dominoes / players

-- The theorem to prove each player gets 7 dominoes
theorem each_player_gets_seven : dominoes_per_player total_dominoes total_players = 7 :=
by
  sorry

end each_player_gets_seven_l197_197833


namespace find_min_value_l197_197196

theorem find_min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : a + b = 2) : 
  (1 / (2 * a)) + (2 / (b - 1)) ≥ 9 / 2 :=
by
  sorry

end find_min_value_l197_197196


namespace pandas_increase_l197_197453

theorem pandas_increase 
  (C P : ℕ) -- C: Number of cheetahs 5 years ago, P: Number of pandas 5 years ago
  (h_ratio_5_years_ago : C / P = 1 / 3)
  (h_cheetahs_increase : ∃ z : ℕ, z = 2)
  (h_ratio_now : ∃ k : ℕ, (C + k) / (P + x) = 1 / 3) :
  x = 6 :=
by
  sorry

end pandas_increase_l197_197453


namespace vertices_divisible_by_three_l197_197549

namespace PolygonDivisibility

theorem vertices_divisible_by_three (v : Fin 2018 → ℤ) 
  (h_initial : (Finset.univ.sum v) = 1) 
  (h_move : ∀ i : Fin 2018, ∃ j : Fin 2018, abs (v i - v j) = 1) :
  ¬ ∃ (k : Fin 2018 → ℤ), (∀ n : Fin 2018, k n % 3 = 0) :=
by {
  sorry
}

end PolygonDivisibility

end vertices_divisible_by_three_l197_197549


namespace union_complement_correct_l197_197997

open Set

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

theorem union_complement_correct : A ∪ (compl B) = Ioo (-1 : ℝ) 3 ∪ {3} := by
  sorry

end union_complement_correct_l197_197997


namespace admission_price_for_adults_l197_197468

theorem admission_price_for_adults (A : ℕ) (ticket_price_children : ℕ) (total_children_tickets : ℕ) 
    (total_amount : ℕ) (total_tickets : ℕ) (children_ticket_costs : ℕ) 
    (adult_tickets : ℕ) (adult_ticket_costs : ℕ) :
    ticket_price_children = 5 → 
    total_children_tickets = 21 → 
    total_amount = 201 → 
    total_tickets = 33 → 
    children_ticket_costs = 21 * 5 → 
    adult_tickets = 33 - 21 → 
    adult_ticket_costs = 201 - 21 * 5 → 
    A = (201 - 21 * 5) / (33 - 21) → 
    A = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end admission_price_for_adults_l197_197468


namespace geometric_series_sum_eq_4_div_3_l197_197477

theorem geometric_series_sum_eq_4_div_3 (a : ℝ) (r : ℝ) (h₀ : a = 1) (h₁ : r = 1 / 4) :
  ∑' n : ℕ, a * r^n = 4 / 3 := by
  sorry

end geometric_series_sum_eq_4_div_3_l197_197477


namespace jessica_dice_problem_l197_197838

noncomputable def probability_of_third_six (p q : ℕ) (h : Nat.coprime p q) :
  p = 109 ∧ q = 148 := by
  sorry

theorem jessica_dice_problem :
  probability_of_third_six 109 148 (Nat.coprime_intro 1 109 148) :=
  by sorry

end jessica_dice_problem_l197_197838


namespace speed_of_second_person_l197_197024

-- Definitions based on the conditions
def speed_person1 := 70 -- km/hr
def distance_AB := 600 -- km

def time_traveled := 4 -- hours (from 10 am to 2 pm)

-- The goal is to prove that the speed of the second person is 80 km/hr
theorem speed_of_second_person :
  (distance_AB - speed_person1 * time_traveled) / time_traveled = 80 := 
by 
  sorry

end speed_of_second_person_l197_197024


namespace min_value_expression_l197_197564

noncomputable def min_expression (a b c : ℝ) : ℝ :=
  (9 / a) + (16 / b) + (25 / c)

theorem min_value_expression :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 6 →
  min_expression a b c ≥ 18 :=
by
  intro a b c ha hb hc habc
  sorry

end min_value_expression_l197_197564


namespace housewife_spent_fraction_l197_197020

theorem housewife_spent_fraction
  (initial_amount : ℝ)
  (amount_left : ℝ)
  (initial_amount_eq : initial_amount = 150)
  (amount_left_eq : amount_left = 50) :
  (initial_amount - amount_left) / initial_amount = 2/3 :=
by 
  sorry

end housewife_spent_fraction_l197_197020


namespace lele_has_enough_money_and_remaining_19_yuan_l197_197136

def price_A : ℝ := 46.5
def price_B : ℝ := 54.5
def total_money : ℝ := 120

theorem lele_has_enough_money_and_remaining_19_yuan : 
  (price_A + price_B ≤ total_money) ∧ (total_money - (price_A + price_B) = 19) :=
by
  sorry

end lele_has_enough_money_and_remaining_19_yuan_l197_197136


namespace event_B_more_likely_than_event_A_l197_197240

-- Definitions based on given conditions
def total_possible_outcomes := 6^3
def favorable_outcomes_B := (Nat.choose 6 3) * (Nat.factorial 3)
def prob_B := favorable_outcomes_B / total_possible_outcomes
def prob_A := 1 - prob_B

-- The theorem to be proved:
theorem event_B_more_likely_than_event_A (total_possible_outcomes = 216) 
    (favorable_outcomes_B = 120) 
    (prob_B = 5 / 9) 
    (prob_A = 4 / 9) :
    prob_B > prob_A := 
by {
    sorry
}

end event_B_more_likely_than_event_A_l197_197240


namespace random_walk_expected_distance_l197_197153

noncomputable def expected_distance_after_random_walk (n : ℕ) : ℚ :=
(sorry : ℚ) -- We'll define this in the proof

-- Proof problem statement in Lean 4
theorem random_walk_expected_distance :
  expected_distance_after_random_walk 6 = 15 / 8 :=
by 
  sorry

end random_walk_expected_distance_l197_197153


namespace solve_quadratic_1_solve_quadratic_2_l197_197414

theorem solve_quadratic_1 : ∀ x : ℝ, x^2 - 5 * x + 4 = 0 ↔ x = 4 ∨ x = 1 :=
by sorry

theorem solve_quadratic_2 : ∀ x : ℝ, x^2 = 4 - 2 * x ↔ x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5 :=
by sorry

end solve_quadratic_1_solve_quadratic_2_l197_197414


namespace select_best_athlete_l197_197139

theorem select_best_athlete
  (avg_A avg_B avg_C avg_D: ℝ)
  (var_A var_B var_C var_D: ℝ)
  (h_avg_A: avg_A = 185)
  (h_avg_B: avg_B = 180)
  (h_avg_C: avg_C = 185)
  (h_avg_D: avg_D = 180)
  (h_var_A: var_A = 3.6)
  (h_var_B: var_B = 3.6)
  (h_var_C: var_C = 7.4)
  (h_var_D: var_D = 8.1) :
  (avg_A > avg_B ∧ avg_A > avg_D ∧ var_A < var_C) →
  (avg_A = 185 ∧ var_A = 3.6) :=
by
  sorry

end select_best_athlete_l197_197139


namespace problem1_l197_197166

theorem problem1 (a b : ℝ) : 
  ((-2 * a) ^ 3 * (- (a * b^2)) ^ 3 - 4 * a * b^2 * (2 * a^5 * b^4 + (1 / 2) * a * b^3 - 5)) / (-2 * a * b) = a * b^4 - 10 * b :=
sorry

end problem1_l197_197166


namespace total_days_off_l197_197469

-- Definitions for the problem conditions
def days_off_personal (months_in_year : ℕ) (days_per_month : ℕ) : ℕ :=
  days_per_month * months_in_year

def days_off_professional (months_in_year : ℕ) (days_per_month : ℕ) : ℕ :=
  days_per_month * months_in_year

def days_off_teambuilding (quarters_in_year : ℕ) (days_per_quarter : ℕ) : ℕ :=
  days_per_quarter * quarters_in_year

-- Main theorem to prove
theorem total_days_off
  (months_in_year : ℕ) (quarters_in_year : ℕ)
  (days_per_month_personal : ℕ) (days_per_month_professional : ℕ) (days_per_quarter_teambuilding: ℕ)
  (h_months : months_in_year = 12) (h_quarters : quarters_in_year = 4) 
  (h_days_personal : days_per_month_personal = 4) (h_days_professional : days_per_month_professional = 2) (h_days_teambuilding : days_per_quarter_teambuilding = 1) :
  days_off_personal months_in_year days_per_month_personal
  + days_off_professional months_in_year days_per_month_professional
  + days_off_teambuilding quarters_in_year days_per_quarter_teambuilding
  = 76 := 
by {
  -- Calculation
  sorry
}

end total_days_off_l197_197469


namespace next_birthday_monday_l197_197410
open Nat

-- Define the basic structure and parameters of our problem
def is_leap_year (year : ℕ) : Prop := 
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def day_of_week (start_day : ℕ) (year_diff : ℕ) (is_leap : ℕ → Prop) : ℕ :=
  (start_day + year_diff + (year_diff / 4) - (year_diff / 100) + (year_diff / 400)) % 7

-- Specify problem conditions
def initial_year := 2009
def initial_day := 5 -- 2009-06-18 is Friday, which is 5 if we start counting from Sunday as 0
def end_day := 1 -- target day is Monday, which is 1

-- Main theorem
theorem next_birthday_monday : ∃ year, year > initial_year ∧
  day_of_week initial_day (year - initial_year) is_leap_year = end_day := by
  use 2017
  -- The proof would go here, skipping with sorry
  sorry

end next_birthday_monday_l197_197410


namespace train_speed_is_60_l197_197772

noncomputable def train_speed_proof : Prop :=
  let train_length := 550 -- in meters
  let time_to_pass := 29.997600191984645 -- in seconds
  let man_speed_kmhr := 6 -- in km/hr
  let man_speed_ms := man_speed_kmhr * (1000 / 3600) -- converting km/hr to m/s
  let relative_speed_ms := train_length / time_to_pass -- relative speed in m/s
  let train_speed_ms := relative_speed_ms - man_speed_ms -- speed of the train in m/s
  let train_speed_kmhr := train_speed_ms * (3600 / 1000) -- converting m/s to km/hr
  train_speed_kmhr = 60 -- the speed of the train in km/hr

theorem train_speed_is_60 : train_speed_proof := by
  sorry

end train_speed_is_60_l197_197772


namespace distance_to_SFL_l197_197159

def distance_per_hour : ℕ := 27
def hours_travelled : ℕ := 3

theorem distance_to_SFL :
  (distance_per_hour * hours_travelled) = 81 := 
by
  sorry

end distance_to_SFL_l197_197159


namespace no_real_x_satisfying_quadratic_inequality_l197_197818

theorem no_real_x_satisfying_quadratic_inequality (a : ℝ) :
  ¬(∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end no_real_x_satisfying_quadratic_inequality_l197_197818


namespace complementary_angles_ratio_decrease_l197_197428

theorem complementary_angles_ratio_decrease 
  (a b : ℝ) (h_ratio : a / b = 3 / 7) (h_comp : a + b = 90) : 
  let a' := a * 1.20 in
  let b' := 90 - a' in
  ((b' / b) * 100 = 91.43) ∧ (100 - (b' / b) * 100 = 8.57) :=
by
  have : a / b = 3 / 7 := h_ratio
  have : a + b = 90 := h_comp
  let a' := a * 1.20
  let b' := 90 - a'
  have : b' / b = 57.6 / 63 := sorry -- Calculations omitted for brevity
  have : (b' / b) * 100 = 91.43 := by sorry
  have : 100 - (b' / b) * 100 = 8.57 := by sorry
  split;
  assumption

end complementary_angles_ratio_decrease_l197_197428


namespace students_only_one_language_l197_197546

-- Define the sets of students for each language class
def French := {1, 2, 3, ..., 30} -- Assume we have numbers representing students
def Spanish := {31, 32, 33, ..., 55} -- Continued numbering for Spanish class
def German := {56, 57, 58, ..., 75} -- Continued numbering for German class

-- Define the intersection sizes for given overlaps
def French_Spanish := 10
def French_German := 7
def Spanish_German := 5
def All_three := 4

-- Given total students in language classes considering overlaps
def total_students := 75

-- Given total number of students actually enrolled in exactly one language
def only_one_language_students := 45

-- Lean 4 statement to express the proof
theorem students_only_one_language :
  ∀ (French Spanish German : Finset ℕ)
    (French_Spanish French_German Spanish_German All_three total_students only_one_language_students : ℕ),
  French.card + Spanish.card + German.card - 
  (French.card ∩ Spanish.card + French.card ∩ German.card + Spanish.card ∩ German.card) +
  All_three = total_students →
  total_students - (French_Spanish + French_German + Spanish_German - 2 * All_three) - All_three = only_one_language_students :=
begin
  sorry
end

end students_only_one_language_l197_197546


namespace matrix_det_zero_l197_197130

variables {α β γ : ℝ}

theorem matrix_det_zero (h : α + β + γ = π) :
  Matrix.det ![
    ![Real.cos β, Real.cos α, -1],
    ![Real.cos γ, -1, Real.cos α],
    ![-1, Real.cos γ, Real.cos β]
  ] = 0 :=
sorry

end matrix_det_zero_l197_197130


namespace associates_hired_l197_197616

variable (partners : ℕ) (associates initial_associates hired_associates : ℕ)
variable (initial_ratio : partners / initial_associates = 2 / 63)
variable (final_ratio : partners / (initial_associates + hired_associates) = 1 / 34)
variable (partners_count : partners = 18)

theorem associates_hired : hired_associates = 45 :=
by
  -- Insert solution steps here...
  sorry

end associates_hired_l197_197616


namespace line_contains_point_l197_197509

theorem line_contains_point (k : ℝ) (x : ℝ) (y : ℝ) (H : 2 - 2 * k * x = -4 * y) : k = -1 ↔ (x = 3 ∧ y = -2) :=
by
  sorry

end line_contains_point_l197_197509


namespace proof_max_ρ_sq_l197_197116

noncomputable def max_ρ_sq (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≥ b) 
    (x y : ℝ) (h₃ : 0 ≤ x) (h₄ : x < a) (h₅ : 0 ≤ y) (h₆ : y < b)
    (h_xy : a^2 + y^2 = b^2 + x^2)
    (h_eq : a^2 + y^2 = (a - x)^2 + (b - y)^2)
    (h_x_le : x ≤ 2 * a / 3) : ℝ :=
  (a / b) ^ 2

theorem proof_max_ρ_sq (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≥ b)
    (x y : ℝ) (h₃ : 0 ≤ x) (h₄ : x < a) (h₅ : 0 ≤ y) (h₆ : y < b)
    (h_xy : a^2 + y^2 = b^2 + x^2)
    (h_eq : a^2 + y^2 = (a - x)^2 + (b - y)^2)
    (h_x_le : x ≤ 2 * a / 3) : (max_ρ_sq a b h₀ h₁ h₂ x y h₃ h₄ h₅ h₆ h_xy h_eq h_x_le) ≤ 9 / 5 := by
  sorry

end proof_max_ρ_sq_l197_197116


namespace rectangle_sides_l197_197435

theorem rectangle_sides (x y : ℝ) (h1 : 4 * x = 3 * y) (h2 : x * y = 2 * (x + y)) :
  (x = 7 / 2 ∧ y = 14 / 3) ∨ (x = 14 / 3 ∧ y = 7 / 2) :=
by {
  sorry
}

end rectangle_sides_l197_197435


namespace part1_part2_l197_197678

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a - Real.log x

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → a ≤ 1 := sorry

theorem part2 (a : ℝ) (x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) :
  (f x₁ a - f x₂ a) / (x₂ - x₁) < 1 / (x₁ * (x₁ + 1)) := sorry

end part1_part2_l197_197678


namespace trig_expression_simplify_l197_197096

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trig_expression_simplify_l197_197096


namespace intersection_A_B_l197_197807

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}
def intersection_of_A_and_B : Set ℕ := {0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = intersection_of_A_and_B :=
by
  sorry

end intersection_A_B_l197_197807


namespace maximum_students_per_dentist_l197_197592

theorem maximum_students_per_dentist (dentists students : ℕ) (min_students : ℕ) (attended_students : ℕ)
  (h_dentists : dentists = 12)
  (h_students : students = 29)
  (h_min_students : min_students = 2)
  (h_total_students : attended_students = students) :
  ∃ max_students, 
    (∀ d, d < dentists → min_students ≤ attended_students / dentists) ∧
    (∀ d, d < dentists → attended_students = students - (dentists * min_students) + min_students) ∧
    max_students = 7 :=
by
  sorry

end maximum_students_per_dentist_l197_197592


namespace find_value_of_f_f_neg1_l197_197891

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then -2 / x else 3 + Real.log x / Real.log 2

theorem find_value_of_f_f_neg1 :
  f (f (-1)) = 4 := by
  -- proof omitted
  sorry

end find_value_of_f_f_neg1_l197_197891


namespace probability_of_odd_sum_rows_columns_l197_197273

open BigOperators

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def probability_odd_sums : ℚ :=
  let even_arrangements := factorial 4
  let odd_positions := factorial 12
  let total_arrangements := factorial 16
  (even_arrangements * odd_positions : ℚ) / total_arrangements

theorem probability_of_odd_sum_rows_columns :
  probability_odd_sums = 1 / 1814400 :=
by
  sorry

end probability_of_odd_sum_rows_columns_l197_197273


namespace modified_goldbach_2024_l197_197948

def is_prime (p : ℕ) : Prop := ∀ n : ℕ, n > 1 → n < p → ¬ (p % n = 0)

theorem modified_goldbach_2024 :
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ is_prime p1 ∧ is_prime p2 ∧ p1 + p2 = 2024 := 
sorry

end modified_goldbach_2024_l197_197948


namespace combined_area_of_four_removed_triangles_l197_197769

noncomputable def combined_area_of_removed_triangles (s x y: ℝ) : Prop :=
  x + y = s ∧ s - 2 * x = 15 ∧ s - 2 * y = 9 ∧
  4 * (1 / 2 * x * y) = 67.5

-- Statement of the problem
theorem combined_area_of_four_removed_triangles (s x y: ℝ) :
  combined_area_of_removed_triangles s x y :=
  by
    sorry

end combined_area_of_four_removed_triangles_l197_197769


namespace ensemble_average_age_l197_197212

theorem ensemble_average_age (female_avg_age : ℝ) (num_females : ℕ) (male_avg_age : ℝ) (num_males : ℕ)
  (h1 : female_avg_age = 32) (h2 : num_females = 12) (h3 : male_avg_age = 40) (h4 : num_males = 18) :
  (num_females * female_avg_age + num_males * male_avg_age) / (num_females + num_males) =  36.8 :=
by sorry

end ensemble_average_age_l197_197212


namespace area_of_garden_l197_197046

-- Define the garden properties
variables {l w : ℕ}

-- Calculate length from the condition of walking length 30 times
def length_of_garden (total_distance : ℕ) (times : ℕ) := total_distance / times

-- Calculate perimeter from the condition of walking perimeter 12 times
def perimeter_of_garden (total_distance : ℕ) (times : ℕ) := total_distance / times

-- Define the proof statement
theorem area_of_garden (total_distance : ℕ) (times_length_walk : ℕ) (times_perimeter_walk : ℕ)
  (h1 : length_of_garden total_distance times_length_walk = l)
  (h2 : perimeter_of_garden total_distance (2 * times_perimeter_walk) = 2 * (l + w)) :
  l * w = 400 := 
sorry

end area_of_garden_l197_197046


namespace factorize_expression_l197_197658

theorem factorize_expression (x y : ℝ) : x^3 * y - 4 * x * y = x * y * (x - 2) * (x + 2) :=
sorry

end factorize_expression_l197_197658


namespace exist_100_noncoverable_triangles_l197_197497

theorem exist_100_noncoverable_triangles :
  ∃ (T : Fin 100 → Triangle), (∀ i j : Fin 100, i ≠ j → ¬ (T i ⊆ T j)) ∧
  (∀ i : Fin 99, height (T (i + 1)) = 200 * diameter (T i) ∧ area (T (i + 1)) = area (T i) / 20000) :=
sorry

end exist_100_noncoverable_triangles_l197_197497


namespace factors_2310_l197_197073

theorem factors_2310 : ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ S.card = 5 ∧ (2310 = S.prod id) :=
by
  sorry

end factors_2310_l197_197073


namespace even_function_zero_coefficient_l197_197696

theorem even_function_zero_coefficient: ∀ a : ℝ, (∀ x : ℝ, (x^2 + a * x + 1) = ((-x)^2 + a * (-x) + 1)) → a = 0 :=
by
  intros a h
  sorry

end even_function_zero_coefficient_l197_197696


namespace macaroon_problem_l197_197187

def total_macaroons_remaining (red_baked green_baked red_ate green_ate : ℕ) : ℕ :=
  (red_baked - red_ate) + (green_baked - green_ate)

theorem macaroon_problem :
  let red_baked := 50 in
  let green_baked := 40 in
  let green_ate := 15 in
  let red_ate := 2 * green_ate in
  total_macaroons_remaining red_baked green_baked red_ate green_ate = 45 :=
by
  sorry

end macaroon_problem_l197_197187


namespace train_speed_faster_l197_197910

-- The Lean statement of the problem
theorem train_speed_faster (Vs : ℝ) (L : ℝ) (T : ℝ) (Vf : ℝ) :
  Vs = 36 ∧ L = 340 ∧ T = 17 ∧ (Vf - Vs) * (5 / 18) = L / T → Vf = 108 :=
by 
  intros 
  sorry

end train_speed_faster_l197_197910


namespace ivy_collectors_edition_dolls_l197_197644

-- Definitions from the conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def collectors_edition_dolls : ℕ := (2 * ivy_dolls) / 3

-- Assertion
theorem ivy_collectors_edition_dolls : collectors_edition_dolls = 20 := by
  sorry

end ivy_collectors_edition_dolls_l197_197644


namespace diameter_of_circle_l197_197315

theorem diameter_of_circle (a b : ℕ) (r : ℝ) (h_a : a = 6) (h_b : b = 8) (h_triangle : a^2 + b^2 = r^2) : r = 10 :=
by 
  rw [h_a, h_b] at h_triangle
  sorry

end diameter_of_circle_l197_197315


namespace find_range_of_a_l197_197062

def have_real_roots (a : ℝ) : Prop := a^2 - 16 ≥ 0

def is_increasing_on_interval (a : ℝ) : Prop := a ≥ -12

theorem find_range_of_a (a : ℝ) : ((have_real_roots a ∨ is_increasing_on_interval a) ∧ ¬(have_real_roots a ∧ is_increasing_on_interval a)) → (a < -12 ∨ (-4 < a ∧ a < 4)) :=
by 
  sorry

end find_range_of_a_l197_197062


namespace event_B_more_likely_than_event_A_l197_197239

-- Definitions based on given conditions
def total_possible_outcomes := 6^3
def favorable_outcomes_B := (Nat.choose 6 3) * (Nat.factorial 3)
def prob_B := favorable_outcomes_B / total_possible_outcomes
def prob_A := 1 - prob_B

-- The theorem to be proved:
theorem event_B_more_likely_than_event_A (total_possible_outcomes = 216) 
    (favorable_outcomes_B = 120) 
    (prob_B = 5 / 9) 
    (prob_A = 4 / 9) :
    prob_B > prob_A := 
by {
    sorry
}

end event_B_more_likely_than_event_A_l197_197239


namespace binomial_product_l197_197656

theorem binomial_product (x : ℝ) : 
  (2 - x^4) * (3 + x^5) = -x^9 - 3 * x^4 + 2 * x^5 + 6 :=
by 
  sorry

end binomial_product_l197_197656


namespace imaginary_part_of_z_l197_197338

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : I * z = 1 + I) : z.im = -1 := 
sorry

end imaginary_part_of_z_l197_197338


namespace butterflies_left_correct_l197_197960

-- Define the total number of butterflies and the fraction that flies away
def butterflies_total : ℕ := 9
def fraction_fly_away : ℚ := 1 / 3

-- Define the number of butterflies left in the garden
def butterflies_left (t : ℕ) (f : ℚ) : ℚ := t - (t : ℚ) * f

-- State the theorem
theorem butterflies_left_correct : butterflies_left butterflies_total fraction_fly_away = 6 := by
  sorry

end butterflies_left_correct_l197_197960


namespace arccos_one_half_l197_197170

theorem arccos_one_half :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_one_half_l197_197170


namespace distribute_7_balls_into_4_boxes_l197_197207

-- Define the problem conditions
def number_of_ways_to_distribute_balls (balls boxes : ℕ) : ℕ :=
  if balls < boxes then 0 else Nat.choose (balls - 1) (boxes - 1)

-- Prove the specific case
theorem distribute_7_balls_into_4_boxes : number_of_ways_to_distribute_balls 7 4 = 20 :=
by
  -- Definition and proof to be filled
  sorry

end distribute_7_balls_into_4_boxes_l197_197207


namespace valid_interval_for_k_l197_197968

theorem valid_interval_for_k :
  ∀ k : ℝ, (∀ x : ℝ, x^2 - 8*x + k < 0 → 0 < k ∧ k < 16) :=
by
  sorry

end valid_interval_for_k_l197_197968


namespace combined_speed_in_still_water_l197_197289

theorem combined_speed_in_still_water 
  (U1 D1 U2 D2 : ℝ) 
  (hU1 : U1 = 30) 
  (hD1 : D1 = 60) 
  (hU2 : U2 = 40) 
  (hD2 : D2 = 80) 
  : (U1 + D1) / 2 + (U2 + D2) / 2 = 105 := 
by 
  sorry

end combined_speed_in_still_water_l197_197289


namespace insurance_covers_80_percent_l197_197550

def total_cost : ℝ := 300
def out_of_pocket_cost : ℝ := 60
def insurance_coverage : ℝ := 0.8  -- Representing 80%

theorem insurance_covers_80_percent :
  (total_cost - out_of_pocket_cost) / total_cost = insurance_coverage := by
  sorry

end insurance_covers_80_percent_l197_197550


namespace problem_inequality_l197_197850

variable {a b c : ℝ}

-- Assuming a, b, c are positive real numbers
variables (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Assuming abc = 1
variable (h_abc : a * b * c = 1)

theorem problem_inequality :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by sorry

end problem_inequality_l197_197850


namespace simon_number_of_legos_l197_197248

variable (Kent_legos : ℕ) (Bruce_legos : ℕ) (Simon_legos : ℕ)

def Kent_condition : Prop := Kent_legos = 40
def Bruce_condition : Prop := Bruce_legos = Kent_legos + 20 
def Simon_condition : Prop := Simon_legos = Bruce_legos + (Bruce_legos * 20 / 100)

theorem simon_number_of_legos : Kent_condition Kent_legos ∧ Bruce_condition Kent_legos Bruce_legos ∧ Simon_condition Bruce_legos Simon_legos → Simon_legos = 72 := by
  intros h
  -- proof steps would go here
  sorry

end simon_number_of_legos_l197_197248


namespace higher_selling_price_is_463_l197_197939

-- Definitions and conditions
def cost_price : ℝ := 400
def selling_price_340 : ℝ := 340
def loss_340 : ℝ := selling_price_340 - cost_price
def gain_percent : ℝ := 0.05
def additional_gain : ℝ := gain_percent * -loss_340
def expected_gain := -loss_340 + additional_gain

-- Theorem to prove that the higher selling price is 463
theorem higher_selling_price_is_463 : ∃ P : ℝ, P = cost_price + expected_gain ∧ P = 463 :=
by
  sorry

end higher_selling_price_is_463_l197_197939


namespace calc_problem1_calc_problem2_calc_problem3_calc_problem4_l197_197636

theorem calc_problem1 : (-3 + 8 - 15 - 6 = -16) :=
by
  sorry

theorem calc_problem2 : (-4/13 - (-4/17) + 4/13 + (-13/17) = -9/17) :=
by
  sorry

theorem calc_problem3 : (-25 - (5/4 * 4/5) - (-16) = -10) :=
by
  sorry

theorem calc_problem4 : (-2^4 - (1/2 * (5 - (-3)^2)) = -14) :=
by
  sorry

end calc_problem1_calc_problem2_calc_problem3_calc_problem4_l197_197636


namespace arccos_one_half_eq_pi_div_three_l197_197169

theorem arccos_one_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 :=
sorry

end arccos_one_half_eq_pi_div_three_l197_197169


namespace people_with_uncool_parents_l197_197439

theorem people_with_uncool_parents :
  ∀ (total cool_dads cool_moms cool_both : ℕ),
    total = 50 →
    cool_dads = 25 →
    cool_moms = 30 →
    cool_both = 15 →
    (total - (cool_dads + cool_moms - cool_both)) = 10 := 
by
  intros total cool_dads cool_moms cool_both h1 h2 h3 h4
  sorry

end people_with_uncool_parents_l197_197439


namespace range_of_a_l197_197204

noncomputable def f (x : ℝ) : ℝ := (1 / (1 + x^2)) - Real.log (abs x)

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) ↔
  (1 / Real.exp 1 ≤ a ∧ a ≤ (2 + Real.log 3) / 3) :=
sorry

end range_of_a_l197_197204


namespace find_S13_l197_197984

-- Define the arithmetic sequence
variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- The sequence is arithmetic, i.e., there exists a common difference d
variable (d : ℤ)
axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d

-- The sum of the first n terms is given by S_n
axiom sum_of_terms : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Given condition
axiom given_condition : a 1 + a 8 + a 12 = 12

-- We need to prove that S_{13} = 52
theorem find_S13 : S 13 = 52 :=
sorry

end find_S13_l197_197984


namespace exp_f_f_increasing_inequality_l197_197799

noncomputable def f (a b : ℝ) (x : ℝ) :=
  (a * x + b) / (x^2 + 1)

-- Conditions
variable (a b : ℝ)
axiom h_odd : ∀ x : ℝ, f a b (-x) = - f a b x
axiom h_value : f a b (1/2) = 2/5

-- Proof statements
theorem exp_f : f a b x = x / (x^2 + 1) := sorry

theorem f_increasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : 
  f a b x1 < f a b x2 := sorry

theorem inequality (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  f a b (2 * x - 1) + f a b x < 0 := sorry

end exp_f_f_increasing_inequality_l197_197799


namespace mark_total_spending_l197_197567

variable (p_tomato_cost : ℕ) (p_apple_cost : ℕ) 
variable (pounds_tomato : ℕ) (pounds_apple : ℕ)

def total_cost (p_tomato_cost : ℕ) (pounds_tomato : ℕ) (p_apple_cost : ℕ) (pounds_apple : ℕ) : ℕ :=
  (p_tomato_cost * pounds_tomato) + (p_apple_cost * pounds_apple)

theorem mark_total_spending :
  total_cost 5 2 6 5 = 40 :=
by
  sorry

end mark_total_spending_l197_197567


namespace power_mod_congruence_l197_197602

theorem power_mod_congruence (h : 3^400 ≡ 1 [MOD 500]) : 3^800 ≡ 1 [MOD 500] :=
by {
  sorry
}

end power_mod_congruence_l197_197602


namespace line_b_y_intercept_l197_197860

variable (b : ℝ → ℝ)
variable (x y : ℝ)

-- Line b is parallel to y = -3x + 6
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c, (∀ x, b x = m * x + c) ∧ m = -3

-- Line b passes through the point (3, -2)
def passes_through_point (b : ℝ → ℝ) : Prop :=
  b 3 = -2

-- The y-intercept of line b
def y_intercept (b : ℝ → ℝ) : ℝ :=
  b 0

theorem line_b_y_intercept (h1 : is_parallel b) (h2 : passes_through_point b) : y_intercept b = 7 :=
sorry

end line_b_y_intercept_l197_197860


namespace factory_fills_boxes_per_hour_l197_197461

theorem factory_fills_boxes_per_hour
  (colors_per_box : ℕ)
  (crayons_per_color : ℕ)
  (total_crayons : ℕ)
  (hours : ℕ)
  (crayons_per_hour := total_crayons / hours)
  (crayons_per_box := colors_per_box * crayons_per_color)
  (boxes_per_hour := crayons_per_hour / crayons_per_box) :
  colors_per_box = 4 →
  crayons_per_color = 2 →
  total_crayons = 160 →
  hours = 4 →
  boxes_per_hour = 5 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end factory_fills_boxes_per_hour_l197_197461


namespace mean_of_all_students_l197_197721

theorem mean_of_all_students (M A : ℕ) (m a : ℕ) (hM : M = 88) (hA : A = 68) (hRatio : m * 5 = 2 * a) : 
  (176 * a + 340 * a) / (7 * a) = 74 :=
by sorry

end mean_of_all_students_l197_197721


namespace cost_of_two_pans_is_20_l197_197840

variable (cost_of_pan : ℕ)

-- Conditions
def pots_cost := 3 * 20
def total_cost := 100
def pans_eq_cost := total_cost - pots_cost
def cost_of_pan_per_pans := pans_eq_cost / 4

-- Proof statement
theorem cost_of_two_pans_is_20 
  (h1 : pots_cost = 60)
  (h2 : total_cost = 100)
  (h3 : pans_eq_cost = total_cost - pots_cost)
  (h4 : cost_of_pan_per_pans = pans_eq_cost / 4)
  : 2 * cost_of_pan_per_pans = 20 :=
by sorry

end cost_of_two_pans_is_20_l197_197840


namespace diameter_correct_l197_197133

noncomputable def diameter_of_circle (C : ℝ) (hC : C = 36) : ℝ :=
  let r := C / (2 * Real.pi)
  2 * r

theorem diameter_correct (C : ℝ) (hC : C = 36) : diameter_of_circle C hC = 36 / Real.pi := by
  sorry

end diameter_correct_l197_197133


namespace butterflies_left_l197_197957

theorem butterflies_left (initial_butterflies : ℕ) (one_third_left : ℕ)
  (h1 : initial_butterflies = 9) (h2 : one_third_left = initial_butterflies / 3) :
  initial_butterflies - one_third_left = 6 :=
by
  sorry

end butterflies_left_l197_197957


namespace arithmetic_sequence_sum_S9_l197_197061

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable {S : ℕ → ℝ} -- Define the sum sequence

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a (n + 1) = a n + d
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = n * (a 1 + a n) / 2

-- Problem statement in Lean
theorem arithmetic_sequence_sum_S9 (h_seq : ∃ d, arithmetic_sequence a d) (h_a2 : a 2 = -2) (h_a8 : a 8 = 6) (h_S_def : sum_of_first_n_terms a S) : S 9 = 18 := 
by {
  sorry
}

end arithmetic_sequence_sum_S9_l197_197061


namespace fiftieth_term_arithmetic_seq_l197_197376

theorem fiftieth_term_arithmetic_seq : 
  (∀ (n : ℕ), (2 + (n - 1) * 5) = 247) := by
  sorry

end fiftieth_term_arithmetic_seq_l197_197376


namespace part1_part2_l197_197354

open Real

noncomputable def f (x : ℝ) (a : ℝ) := |2 * x - 1| - |x - a|

theorem part1 (a : ℝ) (h : a = 0) :
  {x : ℝ | f x a < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a < 1 → |(1 - 2 * a)^2 / 6| > 3 / 2) 
  : a < -1 :=
by
  sorry

end part1_part2_l197_197354


namespace a3_value_l197_197983

variable {a : ℕ → ℤ} -- Arithmetic sequence as a function from natural numbers to integers
variable {S : ℕ → ℤ} -- Sum of the first n terms

-- Conditions
axiom a1_eq : a 1 = -11
axiom a4_plus_a6_eq : a 4 + a 6 = -6
-- Common difference d
variable {d : ℤ}
axiom d_def : ∀ n, a (n + 1) = a n + d

theorem a3_value : a 3 = -7 := by
  sorry -- Proof not required as per the instructions

end a3_value_l197_197983


namespace problem1_problem2_l197_197755

-- Define condition p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

-- Define the negation of p
def neg_p (x a : ℝ) : Prop := ¬ p x a
-- Define the negation of q
def neg_q (x : ℝ) : Prop := ¬ q x

-- Question 1: Prove that if a = 1 and p ∧ q is true, then 2 < x < 3
theorem problem1 (x : ℝ) (h1 : p x 1 ∧ q x) : 2 < x ∧ x < 3 := 
by sorry

-- Question 2: Prove that if ¬ p is a sufficient but not necessary condition for ¬ q, then 1 < a ≤ 2
theorem problem2 (a : ℝ) (h2 : ∀ x : ℝ, neg_p x a → neg_q x) : 1 < a ∧ a ≤ 2 := 
by sorry

end problem1_problem2_l197_197755


namespace smallest_number_proof_l197_197284

noncomputable def smallest_number (a b c : ℕ) :=
  let sum := a + b + c
  let mean := sum / 3
  let sorted := (list.sort (≤) [a, b, c])
  (sorted.head!, sorted.nth! 1, sorted.nth! 2)

theorem smallest_number_proof (a b c : ℕ) (h_mean : (a + b + c) / 3 = 30)
  (h_med : (list.sort (≤) [a, b, c]).nth! 1 = 28)
  (h_largest : (list.sort (≤) [a, b, c]).nth! 2 = 28 + 6) :
  (list.sort (≤) [a, b, c]).head! = 28 :=
by
  sorry

end smallest_number_proof_l197_197284


namespace spend_on_laundry_detergent_l197_197157

def budget : ℕ := 60
def price_shower_gel : ℕ := 4
def num_shower_gels : ℕ := 4
def price_toothpaste : ℕ := 3
def remaining_budget : ℕ := 30

theorem spend_on_laundry_detergent : 
  (budget - remaining_budget) = (num_shower_gels * price_shower_gel + price_toothpaste) + 11 := 
by
  sorry

end spend_on_laundry_detergent_l197_197157


namespace floor_e_eq_two_l197_197650

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 := 
sorry

end floor_e_eq_two_l197_197650


namespace total_spent_snacks_l197_197882

-- Define the costs and discounts
def cost_pizza : ℕ := 10
def boxes_robert_orders : ℕ := 5
def pizza_discount : ℝ := 0.15
def cost_soft_drink : ℝ := 1.50
def soft_drinks_robert : ℕ := 10
def cost_hamburger : ℕ := 3
def hamburgers_teddy_orders : ℕ := 6
def hamburger_discount : ℝ := 0.10
def soft_drinks_teddy : ℕ := 10

-- Calculate total costs
def total_cost_robert : ℝ := 
  let cost_pizza_total := (boxes_robert_orders * cost_pizza) * (1 - pizza_discount)
  let cost_soft_drinks_total := soft_drinks_robert * cost_soft_drink
  cost_pizza_total + cost_soft_drinks_total

def total_cost_teddy : ℝ :=
  let cost_hamburger_total := (hamburgers_teddy_orders * cost_hamburger) * (1 - hamburger_discount)
  let cost_soft_drinks_total := soft_drinks_teddy * cost_soft_drink
  cost_hamburger_total + cost_soft_drinks_total

-- The final theorem to prove the total spending
theorem total_spent_snacks : 
  total_cost_robert + total_cost_teddy = 88.70 := by
  sorry

end total_spent_snacks_l197_197882


namespace cube_surface_area_equals_353_l197_197765

noncomputable def volume_of_prism : ℝ := 5 * 3 * 30
noncomputable def edge_length_of_cube (volume : ℝ) : ℝ := (volume)^(1/3)
noncomputable def surface_area_of_cube (edge_length : ℝ) : ℝ := 6 * edge_length^2

theorem cube_surface_area_equals_353 :
  surface_area_of_cube (edge_length_of_cube volume_of_prism) = 353 := by
sorry

end cube_surface_area_equals_353_l197_197765


namespace line_product_l197_197307

theorem line_product (b m : Int) (h_b : b = -2) (h_m : m = 3) : m * b = -6 :=
by
  rw [h_b, h_m]
  norm_num

end line_product_l197_197307


namespace solve_for_x_l197_197079

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 15 / (x / 3)) : x = 15 * Real.sqrt 3 ∨ x = -15 * Real.sqrt 3 :=
by
  sorry

end solve_for_x_l197_197079


namespace solution_l197_197118

noncomputable def problem_statement (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5)) : ℝ :=
  (x^2 * y^2)

theorem solution : ∀ x y : ℝ, x > 1 → y > 1 → (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5) →
  (x^2 * y^2) = 225^(Real.sqrt 2) :=
by
  intros x y hx hy h
  sorry

end solution_l197_197118


namespace simon_legos_l197_197251

theorem simon_legos (k b s : ℕ) 
  (h_kent : k = 40)
  (h_bruce : b = k + 20)
  (h_simon : s = b + b / 5) : 
  s = 72 := by
  -- sorry, proof not required.
  sorry

end simon_legos_l197_197251


namespace average_visitors_other_days_l197_197151

theorem average_visitors_other_days 
  (avg_sunday : ℕ) (avg_day : ℕ)
  (num_days : ℕ) (sunday_offset : ℕ)
  (other_days_count : ℕ) (total_days : ℕ) 
  (total_avg_visitors : ℕ)
  (sunday_avg_visitors : ℕ) :
  avg_sunday = 150 →
  avg_day = 125 →
  num_days = 30 →
  sunday_offset = 5 →
  total_days = 30 →
  total_avg_visitors * total_days =
    (sunday_offset * sunday_avg_visitors) + (other_days_count * avg_sunday) →
  125 = total_avg_visitors →
  150 = sunday_avg_visitors →
  other_days_count = num_days - sunday_offset →
  (125 * 30 = (5 * 150) + (other_days_count * avg_sunday)) →
  avg_sunday = 120 :=
by
  sorry

end average_visitors_other_days_l197_197151


namespace smallest_hiding_number_l197_197855

/-- Define the concept of "hides" -/
def hides (A B : ℕ) : Prop :=
  ∃ (remove : ℕ → ℕ), remove A = B

/-- The smallest natural number that hides all numbers from 2000 to 2021 is 20012013456789 -/
theorem smallest_hiding_number : hides 20012013456789 2000 ∧ hides 20012013456789 2001 ∧ hides 20012013456789 2002 ∧
    hides 20012013456789 2003 ∧ hides 20012013456789 2004 ∧ hides 20012013456789 2005 ∧ hides 20012013456789 2006 ∧
    hides 20012013456789 2007 ∧ hides 20012013456789 2008 ∧ hides 20012013456789 2009 ∧ hides 20012013456789 2010 ∧
    hides 20012013456789 2011 ∧ hides 20012013456789 2012 ∧ hides 20012013456789 2013 ∧ hides 20012013456789 2014 ∧
    hides 20012013456789 2015 ∧ hides 20012013456789 2016 ∧ hides 20012013456789 2017 ∧ hides 20012013456789 2018 ∧
    hides 20012013456789 2019 ∧ hides 20012013456789 2020 ∧ hides 20012013456789 2021 :=
by
  sorry

end smallest_hiding_number_l197_197855


namespace cost_of_airplane_l197_197467

theorem cost_of_airplane (amount : ℝ) (change : ℝ) (h_amount : amount = 5) (h_change : change = 0.72) : 
  amount - change = 4.28 := 
by
  sorry

end cost_of_airplane_l197_197467


namespace odd_numbers_not_dividing_each_other_l197_197513

theorem odd_numbers_not_dividing_each_other (n : ℕ) (hn : n ≥ 4) :
  ∃ (a b : ℕ), a ≠ b ∧ (2 ^ (2 * n) < a ∧ a < 2 ^ (3 * n)) ∧ 
  (2 ^ (2 * n) < b ∧ b < 2 ^ (3 * n)) ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ 
  ¬ (a ∣ b * b) ∧ ¬ (b ∣ a * a) := by
sorry

end odd_numbers_not_dividing_each_other_l197_197513


namespace count_integers_with_digit_sum_9_l197_197660

theorem count_integers_with_digit_sum_9 :
  {n : ℕ | 1 ≤ n ∧ n ≤ 2013 ∧ (nat.digits 10 n).sum = 9}.card = 101 := by
  sorry

end count_integers_with_digit_sum_9_l197_197660


namespace max_value_of_expression_l197_197337

noncomputable def max_expression_value (x y : ℝ) : ℝ :=
  let expr := x^2 + 6 * y + 2
  14

theorem max_value_of_expression 
  (x y : ℝ) (h : x^2 + y^2 = 4) : ∃ (M : ℝ), M = 14 ∧ ∀ x y, x^2 + y^2 = 4 → x^2 + 6 * y + 2 ≤ M :=
  by
    use 14
    sorry

end max_value_of_expression_l197_197337


namespace sin_721_eq_sin_1_l197_197490

theorem sin_721_eq_sin_1 : Real.sin (721 * Real.pi / 180) = Real.sin (1 * Real.pi / 180) := 
by
  sorry

end sin_721_eq_sin_1_l197_197490


namespace find_k_l197_197977

theorem find_k (k : ℤ)
  (h : ∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ ∀ x, ((k^2 - 1) * x^2 - 3 * (3 * k - 1) * x + 18 = 0) ↔ (x = x₁ ∨ x = x₂)
       ∧ x₁ > 0 ∧ x₂ > 0) : k = 2 :=
by
  sorry

end find_k_l197_197977


namespace number_of_pictures_l197_197752

theorem number_of_pictures (x : ℕ) (h : x - (x / 2 - 1) = 25) : x = 48 :=
sorry

end number_of_pictures_l197_197752


namespace numberOfBigBoats_l197_197610

-- Conditions
variable (students : Nat) (bigBoatCapacity : Nat) (smallBoatCapacity : Nat) (totalBoats : Nat)
variable (students_eq : students = 52)
variable (bigBoatCapacity_eq : bigBoatCapacity = 8)
variable (smallBoatCapacity_eq : smallBoatCapacity = 4)
variable (totalBoats_eq : totalBoats = 9)

theorem numberOfBigBoats : bigBoats + smallBoats = totalBoats → 
                         bigBoatCapacity * bigBoats + smallBoatCapacity * smallBoats = students → 
                         bigBoats = 4 := 
by
  intros h1 h2
  -- Proof steps
  sorry


end numberOfBigBoats_l197_197610


namespace original_denominator_is_18_l197_197771

variable (d : ℕ)

theorem original_denominator_is_18
  (h1 : ∃ (d : ℕ), (3 + 7) / (d + 7) = 2 / 5) :
  d = 18 := 
sorry

end original_denominator_is_18_l197_197771


namespace trigonometric_identity_l197_197090

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end trigonometric_identity_l197_197090


namespace find_number_l197_197876

theorem find_number : ∃ n : ℕ, n = (15 * 6) + 5 := 
by sorry

end find_number_l197_197876


namespace maximal_regions_convex_quadrilaterals_l197_197443

theorem maximal_regions_convex_quadrilaterals (n : ℕ) (hn : n ≥ 1) : 
  ∃ a_n : ℕ, a_n = 4*n^2 - 4*n + 2 :=
by
  sorry

end maximal_regions_convex_quadrilaterals_l197_197443


namespace mod_add_l197_197002

theorem mod_add (n : ℕ) (h : n % 5 = 3) : (n + 2025) % 5 = 3 := by
  sorry

end mod_add_l197_197002


namespace mark_total_spending_l197_197569

theorem mark_total_spending:
  let cost_per_pound_tomatoes := 5
  let pounds_tomatoes := 2
  let cost_per_pound_apples := 6
  let pounds_apples := 5
  let cost_tomatoes := cost_per_pound_tomatoes * pounds_tomatoes
  let cost_apples := cost_per_pound_apples * pounds_apples
  let total_spending := cost_tomatoes + cost_apples
  total_spending = 40 :=
by
  sorry

end mark_total_spending_l197_197569


namespace trigonometric_identity_l197_197991

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  7 * (Real.sin α)^2 + 3 * (Real.cos α)^2 = 31 / 5 := by
  sorry

end trigonometric_identity_l197_197991


namespace distinct_prime_factors_2310_l197_197075

theorem distinct_prime_factors_2310 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ (S.card = 5) ∧ (S.prod id = 2310) := by
  sorry

end distinct_prime_factors_2310_l197_197075


namespace birds_in_sanctuary_l197_197474

theorem birds_in_sanctuary (x y : ℕ) 
    (h1 : x + y = 200)
    (h2 : 2 * x + 4 * y = 590) : 
    x = 105 :=
by
  sorry

end birds_in_sanctuary_l197_197474


namespace sequence_increasing_or_decreasing_l197_197014

theorem sequence_increasing_or_decreasing (x : ℕ → ℝ) (h1 : x 1 > 0) (h2 : x 1 ≠ 1) 
  (hrec : ∀ n, x (n + 1) = (x n * (x n ^ 2 + 3)) / (3 * x n ^ 2 + 1)) :
  ∀ n, x n < x (n + 1) ∨ x n > x (n + 1) :=
by
  sorry

end sequence_increasing_or_decreasing_l197_197014


namespace sum_and_divide_repeating_decimals_l197_197635

noncomputable def repeating_decimal_83 : ℚ := 83 / 99
noncomputable def repeating_decimal_18 : ℚ := 18 / 99

theorem sum_and_divide_repeating_decimals :
  (repeating_decimal_83 + repeating_decimal_18) / (1 / 5) = 505 / 99 :=
by
  sorry

end sum_and_divide_repeating_decimals_l197_197635


namespace binomial_expansion_product_l197_197195

theorem binomial_expansion_product (a a1 a2 a3 a4 a5 : ℤ)
  (h1 : (1 - 1)^5 = a + a1 + a2 + a3 + a4 + a5)
  (h2 : (1 - (-1))^5 = a - a1 + a2 - a3 + a4 - a5) :
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := by
  sorry

end binomial_expansion_product_l197_197195


namespace latus_rectum_of_parabola_l197_197788

theorem latus_rectum_of_parabola (x : ℝ) :
  (∀ x, y = (-1 / 4 : ℝ) * x^2) → y = (-1 / 2 : ℝ) :=
sorry

end latus_rectum_of_parabola_l197_197788


namespace numbers_divisible_by_three_l197_197687

theorem numbers_divisible_by_three (a b : ℕ) (h1 : a = 150) (h2 : b = 450) :
  ∃ n : ℕ, ∀ x : ℕ, (a < x) → (x < b) → (x % 3 = 0) → (x = 153 + 3 * (n - 1)) :=
by
  sorry

end numbers_divisible_by_three_l197_197687


namespace wall_length_proof_l197_197926

-- Define the initial conditions
def men1 : ℕ := 20
def days1 : ℕ := 8
def men2 : ℕ := 86
def days2 : ℕ := 8
def wall_length2 : ℝ := 283.8

-- Define the expected length of the wall for the first condition
def expected_length : ℝ := 65.7

-- The proof statement.
theorem wall_length_proof : ((men1 * days1) / (men2 * days2)) * wall_length2 = expected_length :=
sorry

end wall_length_proof_l197_197926


namespace complementary_angles_decrease_percentage_l197_197434

theorem complementary_angles_decrease_percentage :
  ∀ (x : ℝ), (3 * x + 7 * x = 90) →
  (3 * x * 1.2 + 7 * x = 90) →
  (3 * x > 0) →
  (7 * x > 0) →
  let original_larger_angle : ℝ := 7 * x in
  let new_smaller_angle : ℝ := 3 * x * 1.2 in
  let new_larger_angle : ℝ := 90 - new_smaller_angle in
  let decrease : ℝ := original_larger_angle - new_larger_angle in
  let percentage_decrease : ℝ := (decrease / original_larger_angle) * 100 in
  percentage_decrease = 8.57 := 
sorry

end complementary_angles_decrease_percentage_l197_197434


namespace simplify_fraction_l197_197265

theorem simplify_fraction : 
  ((2^12)^2 - (2^10)^2) / ((2^11)^2 - (2^9)^2) = 4 := 
by sorry

end simplify_fraction_l197_197265


namespace cost_of_four_pencils_and_three_pens_l197_197889

variable {p q : ℝ}

theorem cost_of_four_pencils_and_three_pens (h1 : 3 * p + 2 * q = 4.30) (h2 : 2 * p + 3 * q = 4.05) : 4 * p + 3 * q = 5.97 := by
  sorry

end cost_of_four_pencils_and_three_pens_l197_197889


namespace profit_calculation_l197_197209

theorem profit_calculation
  (P : ℝ)
  (h1 : 9 > 0)  -- condition that there are 9 employees
  (h2 : 0 < 0.10 ∧ 0.10 < 1) -- 10 percent profit is between 0 and 100%
  (h3 : 5 > 0)  -- condition that each employee gets $5
  (h4 : 9 * 5 = 45) -- total amount distributed among employees
  (h5 : 0.90 * P = 45) -- remaining profit to be distributed
  : P = 50 :=
sorry

end profit_calculation_l197_197209


namespace tan_alpha_val_expr_value_first_quadrant_expr_value_third_quadrant_l197_197401

variable (α β : ℝ)

-- Definitions derived from the conditions
def vec_a := (Real.cos (α + β), Real.sin (α + β))
def vec_b := (Real.cos (α - β), Real.sin (α - β))
def vec_sum := (4 / 5, 3 / 5)

-- The leaned statement corresponding to the mathematical proof problem
theorem tan_alpha_val (h : vec_a + vec_b = vec_sum) : Real.tan α = 3 / 4 := sorry

theorem expr_value_first_quadrant (h : vec_a + vec_b = vec_sum) (h_tan : Real.tan α = 3 / 4) (h_first_quadrant : α ∈ Set.Ioo 0 (Real.pi / 2)) :
  (2 * Real.cos α ^ 2 - 4 * Real.sin α - 1) / (Real.sqrt 2 * Real.sin (α - Real.pi / 4)) = 53 / 5 := sorry

theorem expr_value_third_quadrant (h : vec_a + vec_b = vec_sum) (h_tan : Real.tan α = 3 / 4) (h_third_quadrant : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  (2 * Real.cos α ^ 2 - 4 * Real.sin α - 1) / (Real.sqrt 2 * Real.sin (α - Real.pi / 4)) = 67 / 5 := sorry

end tan_alpha_val_expr_value_first_quadrant_expr_value_third_quadrant_l197_197401


namespace possible_values_of_a_plus_b_l197_197521

/-- Given that a and b are constants and that the sum of the three monomials
    4xy², axy^b, -5xy is still a monomial -/
theorem possible_values_of_a_plus_b (a b : ℤ) (h : ∃ (F : ℝ → ℝ → ℝ), 
    ∀ (x y : ℝ), F (4 * x * y^2) (a * x * y^b + -5 * x * y) = 
    f (4 * x * y^2 + a * x * y^b + -5 * x * y)) :
    a + b = -2 ∨ a + b = 6 := sorry

end possible_values_of_a_plus_b_l197_197521


namespace chess_pieces_missing_l197_197230

theorem chess_pieces_missing 
  (total_pieces : ℕ) (pieces_present : ℕ) (h1 : total_pieces = 32) (h2 : pieces_present = 28) : 
  total_pieces - pieces_present = 4 := 
by
  -- Sorry proof
  sorry

end chess_pieces_missing_l197_197230


namespace smallest_N_proof_l197_197543

theorem smallest_N_proof (N c1 c2 c3 c4 : ℕ)
  (h1 : N + c1 = 4 * c3 - 2)
  (h2 : N + c2 = 4 * c1 - 3)
  (h3 : 2 * N + c3 = 4 * c4 - 1)
  (h4 : 3 * N + c4 = 4 * c2) :
  N = 12 :=
sorry

end smallest_N_proof_l197_197543


namespace geometric_series_sum_l197_197487

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℚ) / 4
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geometric_series_sum_l197_197487


namespace smallest_number_of_students_l197_197701

theorem smallest_number_of_students (n : ℕ) : 
  (6 * n + 2 > 40) → (∃ n, 4 * n + 2 * (n + 1) = 44) :=
 by
  intro h
  exact sorry

end smallest_number_of_students_l197_197701


namespace event_B_more_likely_than_event_A_l197_197238

/-- Define the outcomes when rolling a die three times --/
def total_outcomes : ℕ := 6 ^ 3

/-- Define the number of ways to choose 3 different numbers from 6 --/
def choose_3_from_6 : ℕ := Nat.choose 6 3

/-- Define the number of ways to arrange 3 different numbers --/
def arrangements_3 : ℕ := 3.factorial

/-- Calculate the number of favorable outcomes for event B --/
def favorable_B : ℕ := choose_3_from_6 * arrangements_3

/-- Define the probability of event B --/
noncomputable def prob_B : ℝ := favorable_B / total_outcomes

/-- Define the probability of event A as the complement of event B --/
noncomputable def prob_A : ℝ := 1 - prob_B

/-- The theorem to prove that event B is more likely than event A --/
theorem event_B_more_likely_than_event_A : prob_B > prob_A :=
by
  sorry

end event_B_more_likely_than_event_A_l197_197238


namespace part1_part2_l197_197851

-- Definitions for part 1
def prop_p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def prop_q (x : ℝ) : Prop := (x - 3) / (x + 2) < 0

-- Definitions for part 2
def neg_prop_q (x : ℝ) : Prop := ¬((x - 3) / (x + 2) < 0)
def neg_prop_p (a x : ℝ) : Prop := ¬(x^2 - 4*a*x + 3*a^2 < 0)

-- Proof problems
theorem part1 (a : ℝ) (x : ℝ) (h : a = 1) (hpq : prop_p a x ∧ prop_q x) : 1 < x ∧ x < 3 := 
by
  sorry

theorem part2 (a : ℝ) (h : ∀ x, neg_prop_q x → neg_prop_p a x) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end part1_part2_l197_197851


namespace travel_time_l197_197023

-- Definitions: 
def speed := 20 -- speed in km/hr
def distance := 160 -- distance in km

-- Proof statement: 
theorem travel_time (s : ℕ) (d : ℕ) (h1 : s = speed) (h2 : d = distance) : 
  d / s = 8 :=
by {
  sorry
}

end travel_time_l197_197023


namespace inequality_proof_l197_197688

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 9 * x * y * z) :
    x / Real.sqrt (x^2 + 2 * y * z + 2) + y / Real.sqrt (y^2 + 2 * z * x + 2) + z / Real.sqrt (z^2 + 2 * x * y + 2) ≥ 1 :=
by
  sorry

end inequality_proof_l197_197688


namespace convert_C_to_F_l197_197691

theorem convert_C_to_F (C F : ℝ) (h1 : C = 40) (h2 : C = 5 / 9 * (F - 32)) : F = 104 := 
by
  -- Proof goes here
  sorry

end convert_C_to_F_l197_197691


namespace power_function_inverse_l197_197540

theorem power_function_inverse (f : ℝ → ℝ) (h₁ : f 2 = (Real.sqrt 2) / 2) : f⁻¹ 2 = 1 / 4 :=
by
  -- Lean proof will be filled here
  sorry

end power_function_inverse_l197_197540


namespace jane_payment_per_bulb_l197_197551

theorem jane_payment_per_bulb :
  let tulip_bulbs := 20
  let iris_bulbs := tulip_bulbs / 2
  let daffodil_bulbs := 30
  let crocus_bulbs := 3 * daffodil_bulbs
  let total_bulbs := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  let total_earned := 75
  let payment_per_bulb := total_earned / total_bulbs
  payment_per_bulb = 0.50 := 
by
  sorry

end jane_payment_per_bulb_l197_197551


namespace first_batch_price_is_50_max_number_of_type_a_tools_l197_197440

-- Define the conditions
def first_batch_cost : Nat := 2000
def second_batch_cost : Nat := 2200
def price_increase : Nat := 5
def max_total_cost : Nat := 2500
def type_b_cost : Nat := 40
def total_third_batch : Nat := 50

-- First batch price per tool
theorem first_batch_price_is_50 (x : Nat) (h1 : first_batch_cost * (x + price_increase) = second_batch_cost * x) :
  x = 50 :=
sorry

-- Second batch price per tool & maximum type A tools in third batch
theorem max_number_of_type_a_tools (y : Nat)
  (h2 : 55 * y + type_b_cost * (total_third_batch - y) ≤ max_total_cost) :
  y ≤ 33 :=
sorry

end first_batch_price_is_50_max_number_of_type_a_tools_l197_197440


namespace linda_savings_fraction_l197_197856

theorem linda_savings_fraction (savings tv_cost : ℝ) (h1 : savings = 960) (h2 : tv_cost = 240) : (savings - tv_cost) / savings = 3 / 4 :=
by
  intros
  sorry

end linda_savings_fraction_l197_197856


namespace function_solution_l197_197965

theorem function_solution (f : ℝ → ℝ) (α : ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) →
  (∀ x : ℝ, f x = x ^ 3 + α * x) :=
by
  sorry

end function_solution_l197_197965


namespace brass_to_band_ratio_l197_197776

theorem brass_to_band_ratio
  (total_students : ℕ)
  (marching_band_fraction brass_saxophone_fraction saxophone_alto_fraction : ℚ)
  (alto_saxophone_students : ℕ)
  (h1 : total_students = 600)
  (h2 : marching_band_fraction = 1 / 5)
  (h3 : brass_saxophone_fraction = 1 / 5)
  (h4 : saxophone_alto_fraction = 1 / 3)
  (h5 : alto_saxophone_students = 4) :
  ((brass_saxophone_fraction * saxophone_alto_fraction) * total_students * marching_band_fraction = 4) →
  ((brass_saxophone_fraction * 3 * marching_band_fraction * total_students) / (marching_band_fraction * total_students) = 1 / 2) :=
by {
  -- Here we state the proof but leave it as a sorry placeholder.
  sorry
}

end brass_to_band_ratio_l197_197776


namespace fraction_of_a_eq_1_fifth_of_b_l197_197920

theorem fraction_of_a_eq_1_fifth_of_b (a b : ℝ) (x : ℝ) 
  (h1 : a + b = 100) 
  (h2 : (1/5) * b = 12)
  (h3 : b = 60) : x = 3/10 := by
  sorry

end fraction_of_a_eq_1_fifth_of_b_l197_197920


namespace minimum_area_of_square_on_parabola_l197_197320

theorem minimum_area_of_square_on_parabola :
  ∃ (A B C : ℝ × ℝ), 
  (∃ (x₁ x₂ x₃ : ℝ), (A = (x₁, x₁^2)) ∧ (B = (x₂, x₂^2)) ∧ (C = (x₃, x₃^2)) 
  ∧ x₁ < x₂ ∧ x₂ < x₃ 
  ∧ ∀ S : ℝ, (S = (1 + (x₃ + x₂)^2) * ((x₂ - x₃) - (x₃ - x₂))^2) → S ≥ 2) :=
sorry

end minimum_area_of_square_on_parabola_l197_197320


namespace floor_e_is_two_l197_197655

noncomputable def e : ℝ := Real.exp 1

theorem floor_e_is_two : ⌊e⌋ = 2 := by
  sorry

end floor_e_is_two_l197_197655


namespace discount_equation_l197_197460

variable (P₀ P_f x : ℝ)
variable (h₀ : P₀ = 200)
variable (h₁ : P_f = 164)

theorem discount_equation :
  P₀ * (1 - x)^2 = P_f := by
  sorry

end discount_equation_l197_197460


namespace total_marbles_l197_197905

theorem total_marbles (bowl2_capacity : ℕ) (h₁ : bowl2_capacity = 600)
    (h₂ : 3 / 4 * bowl2_capacity = 450) : 600 + (3 / 4 * 600) = 1050 := by
  sorry

end total_marbles_l197_197905


namespace fewest_handshakes_is_zero_l197_197378

noncomputable def fewest_handshakes (n k : ℕ) : ℕ :=
  if h : (n * (n - 1)) / 2 + k = 325 then k else 325

theorem fewest_handshakes_is_zero :
  ∃ n k : ℕ, (n * (n - 1)) / 2 + k = 325 ∧ 0 = fewest_handshakes n k :=
by
  sorry

end fewest_handshakes_is_zero_l197_197378


namespace trisha_total_distance_walked_l197_197124

def d1 : ℝ := 0.1111111111111111
def d2 : ℝ := 0.1111111111111111
def d3 : ℝ := 0.6666666666666666

theorem trisha_total_distance_walked :
  d1 + d2 + d3 = 0.8888888888888888 := 
sorry

end trisha_total_distance_walked_l197_197124


namespace pears_remaining_l197_197220

theorem pears_remaining (K_picked : ℕ) (M_picked : ℕ) (S_picked : ℕ)
                        (K_gave : ℕ) (M_gave : ℕ) (S_gave : ℕ)
                        (hK_pick : K_picked = 47)
                        (hM_pick : M_picked = 12)
                        (hS_pick : S_picked = 22)
                        (hK_give : K_gave = 46)
                        (hM_give : M_gave = 5)
                        (hS_give : S_gave = 15) :
  (K_picked - K_gave) + (M_picked - M_gave) + (S_picked - S_gave) = 15 :=
by
  sorry

end pears_remaining_l197_197220


namespace exists_equilateral_triangle_l197_197343

-- Defining a function to create a type for the color
inductive Color 
| black
| white

-- Defining the function to color the plane
def plane_coloring (c : ℝ × ℝ → Color) : Prop :=
  ∃ x y z : ℝ × ℝ, 
    (dist x y = 1 ∨ dist x y = real.sqrt 3) ∧
    (dist y z = 1 ∨ dist y z = real.sqrt 3) ∧
    (dist z x = 1 ∨ dist z x = real.sqrt 3) ∧
    (c x = c y ∧ c y = c z)

-- The theorem that corresponds to the proof problem
theorem exists_equilateral_triangle (c : ℝ × ℝ → Color) : plane_coloring c :=
sorry

end exists_equilateral_triangle_l197_197343


namespace ten_factorial_minus_nine_factorial_l197_197034

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem ten_factorial_minus_nine_factorial :
  factorial 10 - factorial 9 = 3265920 := 
by 
  sorry

end ten_factorial_minus_nine_factorial_l197_197034


namespace ripe_oranges_count_l197_197282

/-- They harvest 52 sacks of unripe oranges per day. -/
def unripe_oranges_per_day : ℕ := 52

/-- After 26 days of harvest, they will have 2080 sacks of oranges. -/
def total_oranges_after_26_days : ℕ := 2080

/-- Define the number of sacks of ripe oranges harvested per day. -/
def ripe_oranges_per_day (R : ℕ) : Prop :=
  26 * (R + unripe_oranges_per_day) = total_oranges_after_26_days

/-- Prove that they harvest 28 sacks of ripe oranges per day. -/
theorem ripe_oranges_count : ripe_oranges_per_day 28 :=
by {
  -- This is where the proof would go
  sorry
}

end ripe_oranges_count_l197_197282


namespace min_value_l197_197185

theorem min_value (x : ℝ) (h : x > 1) : ∃ m : ℝ, m = 2 * Real.sqrt 5 ∧ ∀ y : ℝ, y = Real.sqrt (x - 1) → (x = y^2 + 1) → (x + 4) / y = m :=
by
  sorry

end min_value_l197_197185


namespace compute_fraction_equation_l197_197167

theorem compute_fraction_equation :
  (8 * (2 / 3: ℚ)^4 + 2 = 290 / 81) :=
sorry

end compute_fraction_equation_l197_197167


namespace line_relation_with_plane_l197_197358

variables {P : Type} [Infinite P] [MetricSpace P]

variables (a b : Line P) (α : Plane P)

-- Conditions
axiom intersecting_lines : ∃ p : P, p ∈ a ∧ p ∈ b
axiom line_parallel_plane : ∀ p : P, p ∈ a → p ∈ α

-- Theorem statement for the proof problem
theorem line_relation_with_plane : (∀ p : P, p ∈ b → p ∈ α) ∨ (∃ q : P, q ∈ α ∧ q ∈ b) :=
sorry

end line_relation_with_plane_l197_197358


namespace geometric_sequence_seventh_term_l197_197914

theorem geometric_sequence_seventh_term :
  let a := 6
  let r := -2
  (a * r^(7 - 1)) = 384 := 
by
  sorry

end geometric_sequence_seventh_term_l197_197914


namespace range_of_a_l197_197436

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 5 * a) ↔ (4 ≤ a ∨ a ≤ 1) :=
by
  sorry

end range_of_a_l197_197436


namespace taxi_range_l197_197823

noncomputable def fare (x : ℝ) : ℝ :=
  if x <= 3 then 
    5
  else if x <= 10 then
    5 + (x - 3) * 2
  else
    5 + 7 * 2 + (x - 10) * 3

theorem taxi_range (x : ℝ) (h : fare x + 1 = 38) : 15 < x ∧ x ≤ 16 := 
  sorry

end taxi_range_l197_197823


namespace sum_of_consecutive_page_numbers_l197_197276

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end sum_of_consecutive_page_numbers_l197_197276


namespace mark_profit_l197_197869

variable (initial_cost tripling_factor new_value profit : ℕ)

-- Conditions
def initial_card_cost := 100
def card_tripling_factor := 3

-- Calculations based on conditions
def card_new_value := initial_card_cost * card_tripling_factor
def card_profit := card_new_value - initial_card_cost

-- Proof Statement
theorem mark_profit (initial_card_cost tripling_factor card_new_value card_profit : ℕ) 
  (h1: initial_card_cost = 100)
  (h2: tripling_factor = 3)
  (h3: card_new_value = initial_card_cost * tripling_factor)
  (h4: card_profit = card_new_value - initial_card_cost) :
  card_profit = 200 :=
  by sorry

end mark_profit_l197_197869


namespace sharpener_difference_l197_197935

/-- A hand-crank pencil sharpener can sharpen one pencil every 45 seconds.
An electric pencil sharpener can sharpen one pencil every 20 seconds.
The total available time is 360 seconds (i.e., 6 minutes).
Prove that the difference in the number of pencils sharpened 
by the electric sharpener and the hand-crank sharpener in 360 seconds is 10 pencils. -/
theorem sharpener_difference (time : ℕ) (hand_crank_rate : ℕ) (electric_rate : ℕ) 
(h_time : time = 360) (h_hand_crank : hand_crank_rate = 45) (h_electric : electric_rate = 20) :
  (time / electric_rate) - (time / hand_crank_rate) = 10 := by
  sorry

end sharpener_difference_l197_197935


namespace butterflies_left_l197_197956

theorem butterflies_left (initial_butterflies : ℕ) (one_third_left : ℕ)
  (h1 : initial_butterflies = 9) (h2 : one_third_left = initial_butterflies / 3) :
  initial_butterflies - one_third_left = 6 :=
by
  sorry

end butterflies_left_l197_197956


namespace length_of_AB_l197_197126

theorem length_of_AB
  (AP PB AQ QB : ℝ) 
  (h_ratioP : 5 * AP = 3 * PB)
  (h_ratioQ : 3 * AQ = 2 * QB)
  (h_PQ : AQ = AP + 3 ∧ QB = PB - 3)
  (h_PQ_length : AQ - AP = 3)
  : AP + PB = 120 :=
by {
  sorry
}

end length_of_AB_l197_197126


namespace power_subtraction_l197_197897

theorem power_subtraction : 2^4 - 2^3 = 2^3 := by
  sorry

end power_subtraction_l197_197897


namespace inequality_positive_reals_l197_197879

open Real

variable (x y : ℝ)

theorem inequality_positive_reals (hx : 0 < x) (hy : 0 < y) : x^2 + (8 / (x * y)) + y^2 ≥ 8 :=
by
  sorry

end inequality_positive_reals_l197_197879


namespace keys_per_lock_l197_197900

-- Define the given conditions
def num_complexes := 2
def apartments_per_complex := 12
def total_keys := 72

-- Calculate the total number of apartments
def total_apartments := num_complexes * apartments_per_complex

-- The theorem statement to prove
theorem keys_per_lock : total_keys / total_apartments = 3 := 
by
  sorry

end keys_per_lock_l197_197900


namespace tangent_integer_values_l197_197976

/-- From point P outside a circle with circumference 12π units, a tangent and a secant are drawn.
      The secant divides the circle into arcs with lengths m and n. Given that the length of the
      tangent t is the geometric mean between m and n, and that m is three times n, there are zero
      possible integer values for t. -/
theorem tangent_integer_values
  (circumference : ℝ) (m n t : ℝ)
  (h_circumference : circumference = 12 * Real.pi)
  (h_sum : m + n = 12 * Real.pi)
  (h_ratio : m = 3 * n)
  (h_tangent : t = Real.sqrt (m * n)) :
  ¬(∃ k : ℤ, t = k) := 
sorry

end tangent_integer_values_l197_197976


namespace find_first_month_sale_l197_197305

/-- Given the sales for months two to six and the average sales over six months,
    prove the sale in the first month. -/
theorem find_first_month_sale
  (sales_2 : ℤ) (sales_3 : ℤ) (sales_4 : ℤ) (sales_5 : ℤ) (sales_6 : ℤ)
  (avg_sales : ℤ)
  (h2 : sales_2 = 5468) (h3 : sales_3 = 5568) (h4 : sales_4 = 6088)
  (h5 : sales_5 = 6433) (h6 : sales_6 = 5922) (h_avg : avg_sales = 5900) : 
  ∃ (sale_1 : ℤ), sale_1 = 5921 := 
by
  have total_sales : ℤ := avg_sales * 6
  have known_sales_sum : ℤ := sales_2 + sales_3 + sales_4 + sales_5
  use total_sales - known_sales_sum - sales_6
  sorry

end find_first_month_sale_l197_197305


namespace pipe_fill_time_without_leak_l197_197312

theorem pipe_fill_time_without_leak (T : ℝ) (h1 : T > 0) 
  (h2 : 1/T - 1/8 = 1/8) :
  T = 4 := 
sorry

end pipe_fill_time_without_leak_l197_197312


namespace monotonicity_no_zeros_range_of_a_l197_197393

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * (Real.log x) + 1

theorem monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1/a → ∀ t : ℝ, t ∈ Ioo x y → f' a t < 0 ) ∧ 
  (∀ x y : ℝ, 1/a < x ∧ x < y → ∀ t : ℝ, t ∈ Ioo x y → f' a t > 0 ) :=
sorry

theorem no_zeros_range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → f a x > 0) : 
  a > 1/Real.exp 1 :=
sorry

end monotonicity_no_zeros_range_of_a_l197_197393


namespace sum_of_constants_l197_197949

theorem sum_of_constants (x a b : ℤ) (h : x^2 - 10 * x + 15 = 0) 
    (h1 : (x + a)^2 = b) : a + b = 5 := 
sorry

end sum_of_constants_l197_197949


namespace parabola_directrix_l197_197736

theorem parabola_directrix (x y : ℝ) (h : y = x^2) : 4 * y + 1 = 0 := 
sorry

end parabola_directrix_l197_197736


namespace minimum_value_of_expression_l197_197146

theorem minimum_value_of_expression (x y z w : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) (h : 5 * w = 3 * x ∧ 5 * w = 4 * y ∧ 5 * w = 7 * z) : x - y + z - w = 11 :=
sorry

end minimum_value_of_expression_l197_197146


namespace profit_calculation_l197_197872

def Initial_Value : ℕ := 100
def Multiplier : ℕ := 3
def New_Value : ℕ := Initial_Value * Multiplier
def Profit : ℕ := New_Value - Initial_Value

theorem profit_calculation : Profit = 200 := by
  sorry

end profit_calculation_l197_197872


namespace isosceles_right_triangle_area_l197_197270

-- Define the conditions as given in the problem statement
variables (h l : ℝ)
hypothesis (hypotenuse_rel : h = l * Real.sqrt 2)
hypothesis (hypotenuse_val : h = 6 * Real.sqrt 2)

-- Define the formula for the area of an isosceles right triangle
def area_of_isosceles_right_triangle (l : ℝ) : ℝ := (1 / 2) * l * l

-- Define the proof problem statement
theorem isosceles_right_triangle_area : 
  area_of_isosceles_right_triangle l = 18 :=
  sorry

end isosceles_right_triangle_area_l197_197270


namespace abs_neg_value_l197_197579

-- Definition of absolute value using the conditions given.
def abs (x : Int) : Int :=
  if x < 0 then -x else x

-- Theorem statement that |-2023| = 2023
theorem abs_neg_value : abs (-2023) = 2023 :=
  sorry

end abs_neg_value_l197_197579


namespace min_likes_both_l197_197288

-- Definitions corresponding to the conditions
def total_people : ℕ := 200
def likes_beethoven : ℕ := 160
def likes_chopin : ℕ := 150

-- Problem statement to prove
theorem min_likes_both : ∃ x : ℕ, x = 110 ∧ x = likes_beethoven - (total_people - likes_chopin) := by
  sorry

end min_likes_both_l197_197288


namespace sin_C_in_right_triangle_l197_197704

theorem sin_C_in_right_triangle 
  (A B C : ℝ)
  (h1 : A + B + C = π)
  (h2 : B = π / 2)
  (h3 : sin A = 8 / 17) :
  sin C = 15 / 17 :=
by
  sorry

end sin_C_in_right_triangle_l197_197704


namespace find_second_x_intercept_l197_197973

theorem find_second_x_intercept (a b c : ℝ)
  (h_vertex : ∀ x, y = a * x^2 + b * x + c → x = 5 → y = -3)
  (h_intercept1 : ∀ y, y = a * 1^2 + b * 1 + c → y = 0) :
  ∃ x, y = a * x^2 + b * x + c ∧ y = 0 ∧ x = 9 :=
sorry

end find_second_x_intercept_l197_197973


namespace find_n_in_arithmetic_sequence_l197_197794

noncomputable def arithmetic_sequence (n : ℕ) (a_n S_n d : ℕ) :=
  ∀ (a₁ : ℕ), 
    a₁ + d * (n - 1) = a_n →
    n * a₁ + d * n * (n - 1) / 2 = S_n

theorem find_n_in_arithmetic_sequence 
   (a_n S_n d n : ℕ) 
   (h_a_n : a_n = 44) 
   (h_S_n : S_n = 158) 
   (h_d : d = 3) :
   arithmetic_sequence n a_n S_n d → 
   n = 4 := 
by 
  sorry

end find_n_in_arithmetic_sequence_l197_197794


namespace find_a8_l197_197548

def seq (a : Nat → Int) := a 1 = -1 ∧ ∀ n, a (n + 1) = a n - 3

theorem find_a8 (a : Nat → Int) (h : seq a) : a 8 = -22 :=
by {
  sorry
}

end find_a8_l197_197548


namespace no_intersection_points_l197_197330

def intersection_points_eq_zero : Prop :=
∀ x y : ℝ, (y = abs (3 * x + 6)) ∧ (y = -abs (4 * x - 3)) → false

theorem no_intersection_points :
  intersection_points_eq_zero :=
by
  intro x y h
  cases h
  sorry

end no_intersection_points_l197_197330


namespace marble_problem_l197_197764

theorem marble_problem
  (M : ℕ)
  (X : ℕ)
  (h1 : M = 18 * X)
  (h2 : M = 20 * (X - 1)) :
  M = 180 :=
by
  sorry

end marble_problem_l197_197764


namespace sum_a_b_neg1_l197_197813

-- Define the problem using the given condition
theorem sum_a_b_neg1 (a b : ℝ) (h : |a + 3| + (b - 2) ^ 2 = 0) : a + b = -1 := 
by
  sorry

end sum_a_b_neg1_l197_197813


namespace geometric_series_sum_l197_197485

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℚ) / 4
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geometric_series_sum_l197_197485


namespace line_b_y_intercept_l197_197861

variable (b : ℝ → ℝ)
variable (x y : ℝ)

-- Line b is parallel to y = -3x + 6
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c, (∀ x, b x = m * x + c) ∧ m = -3

-- Line b passes through the point (3, -2)
def passes_through_point (b : ℝ → ℝ) : Prop :=
  b 3 = -2

-- The y-intercept of line b
def y_intercept (b : ℝ → ℝ) : ℝ :=
  b 0

theorem line_b_y_intercept (h1 : is_parallel b) (h2 : passes_through_point b) : y_intercept b = 7 :=
sorry

end line_b_y_intercept_l197_197861


namespace basketball_not_table_tennis_l197_197700

-- Definitions and conditions
def total_students := 30
def like_basketball := 15
def like_table_tennis := 10
def do_not_like_either := 8
def like_both (x : ℕ) := x

-- Theorem statement
theorem basketball_not_table_tennis (x : ℕ) (H : (like_basketball - x) + (like_table_tennis - x) + x + do_not_like_either = total_students) : (like_basketball - x) = 12 :=
by
  sorry

end basketball_not_table_tennis_l197_197700


namespace sequences_zero_at_2_l197_197672

theorem sequences_zero_at_2
  (a b c d : ℕ → ℝ)
  (h1 : ∀ n, a (n+1) = a n + b n)
  (h2 : ∀ n, b (n+1) = b n + c n)
  (h3 : ∀ n, c (n+1) = c n + d n)
  (h4 : ∀ n, d (n+1) = d n + a n)
  (k m : ℕ)
  (hk : 1 ≤ k)
  (hm : 1 ≤ m)
  (h5 : a (k + m) = a m)
  (h6 : b (k + m) = b m)
  (h7 : c (k + m) = c m)
  (h8 : d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 :=
by sorry

end sequences_zero_at_2_l197_197672


namespace five_times_number_equals_hundred_l197_197661

theorem five_times_number_equals_hundred (x : ℝ) (h : 5 * x = 100) : x = 20 :=
sorry

end five_times_number_equals_hundred_l197_197661


namespace factorial_subtraction_l197_197039

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_subtraction : factorial 10 - factorial 9 = 3265920 := by
  sorry

end factorial_subtraction_l197_197039


namespace monotonicity_no_zeros_range_of_a_l197_197394

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * (Real.log x) + 1

theorem monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1/a → ∀ t : ℝ, t ∈ Ioo x y → f' a t < 0 ) ∧ 
  (∀ x y : ℝ, 1/a < x ∧ x < y → ∀ t : ℝ, t ∈ Ioo x y → f' a t > 0 ) :=
sorry

theorem no_zeros_range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → f a x > 0) : 
  a > 1/Real.exp 1 :=
sorry

end monotonicity_no_zeros_range_of_a_l197_197394


namespace cos_double_angle_l197_197692

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2*n) = 8) :
  Real.cos (2 * θ) = 3 / 4 :=
sorry

end cos_double_angle_l197_197692


namespace tan_600_eq_neg_sqrt_3_l197_197183

theorem tan_600_eq_neg_sqrt_3 : Real.tan (600 * Real.pi / 180) = -Real.sqrt 3 := by
  sorry

end tan_600_eq_neg_sqrt_3_l197_197183


namespace totalTaxIsCorrect_l197_197013

-- Define the different income sources
def dividends : ℝ := 50000
def couponIncomeOFZ : ℝ := 40000
def couponIncomeCorporate : ℝ := 30000
def capitalGain : ℝ := (100 * 200) - (100 * 150)

-- Define the tax rates
def taxRateDividends : ℝ := 0.13
def taxRateCorporateBond : ℝ := 0.13
def taxRateCapitalGain : ℝ := 0.13

-- Calculate the tax for each type of income
def taxOnDividends : ℝ := dividends * taxRateDividends
def taxOnCorporateCoupon : ℝ := couponIncomeCorporate * taxRateCorporateBond
def taxOnCapitalGain : ℝ := capitalGain * taxRateCapitalGain

-- Sum of all tax amounts
def totalTax : ℝ := taxOnDividends + taxOnCorporateCoupon + taxOnCapitalGain

-- Prove that total tax equals the calculated figure
theorem totalTaxIsCorrect : totalTax = 11050 := by
  sorry

end totalTaxIsCorrect_l197_197013


namespace time_spent_answering_questions_l197_197809

theorem time_spent_answering_questions (total_questions answered_per_question_minutes unanswered_questions : ℕ) (minutes_per_hour : ℕ) :
  total_questions = 100 → unanswered_questions = 40 → answered_per_question_minutes = 2 → minutes_per_hour = 60 → 
  ((total_questions - unanswered_questions) * answered_per_question_minutes) / minutes_per_hour = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end time_spent_answering_questions_l197_197809


namespace f_2_equals_12_l197_197067

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^3 + x^2 else - (2 * (-x)^3 + (-x)^2)

theorem f_2_equals_12 : f 2 = 12 := by
  sorry

end f_2_equals_12_l197_197067


namespace unique_solution_for_quadratic_l197_197677

theorem unique_solution_for_quadratic (a : ℝ) : 
  ∃! (x : ℝ), x^2 - 2 * a * x + a^2 = 0 := 
by
  sorry

end unique_solution_for_quadratic_l197_197677


namespace find_original_number_l197_197241

/-- Given that one less than the reciprocal of a number is 5/2, the original number must be -2/3. -/
theorem find_original_number (y : ℚ) (h : 1 - 1 / y = 5 / 2) : y = -2 / 3 :=
sorry

end find_original_number_l197_197241


namespace butterfly_count_l197_197961

theorem butterfly_count (total_butterflies : ℕ) (one_third_flew_away : ℕ) (initial_butterflies : total_butterflies = 9) (flew_away : one_third_flew_away = total_butterflies / 3) : 
(total_butterflies - one_third_flew_away) = 6 := by
  sorry

end butterfly_count_l197_197961


namespace log_sum_correct_l197_197916

noncomputable def log_sum : ℝ := 
  Real.log 8 / Real.log 10 + 
  3 * Real.log 4 / Real.log 10 + 
  4 * Real.log 2 / Real.log 10 +
  2 * Real.log 5 / Real.log 10 +
  5 * Real.log 25 / Real.log 10

theorem log_sum_correct : abs (log_sum - 12.301) < 0.001 :=
by sorry

end log_sum_correct_l197_197916


namespace two_positive_roots_condition_l197_197201

theorem two_positive_roots_condition (a : ℝ) :
  (1 < a ∧ a ≤ 2) ∨ (a ≥ 10) ↔
  ∃ x1 x2 : ℝ, (1-a) * x1^2 + (a+2) * x1 - 4 = 0 ∧ 
               (1-a) * x2^2 + (a+2) * x2 - 4 = 0 ∧ 
               x1 > 0 ∧ x2 > 0 :=
sorry

end two_positive_roots_condition_l197_197201


namespace part_a_part_b_l197_197844

variable 
  {A B C D E P : Type}
  [ConvexPentagon ABCDE Set]
  (circ_abe : Circle ABCD)
  (intersects_AC : ∃ P, P ∈ circ_abe ∧ is_on_line P AC)
  (bisects_BAE : bisects AC ∠BAE)
  (bisects_DCB : bisects AC ∠DCB)
  (angle_AEB_90 : ∠AEB = 90)
  (angle_BDC_90 : ∠BDC = 90)

namespace Geometry

theorem part_a :
  P = circumcenter (Triangle BDE) :=
sorry

theorem part_b :
  cyclic_quadrilateral A C D E :=
sorry

end Geometry

end part_a_part_b_l197_197844


namespace factorize_expression_l197_197050

theorem factorize_expression (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l197_197050


namespace floor_e_eq_two_l197_197647

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 :=
by
  sorry

end floor_e_eq_two_l197_197647


namespace car_mpg_city_l197_197760

theorem car_mpg_city
  (h c T : ℝ)
  (h1 : h * T = 480)
  (h2 : c * T = 336)
  (h3 : c = h - 6) :
  c = 14 :=
by
  sorry

end car_mpg_city_l197_197760


namespace ratio_of_perimeters_l197_197078

-- Define lengths of the rectangular patch
def length_rect : ℝ := 400
def width_rect : ℝ := 300

-- Define the length of the side of the square patch
def side_square : ℝ := 700

-- Define the perimeters of both patches
def P_square : ℝ := 4 * side_square
def P_rectangle : ℝ := 2 * (length_rect + width_rect)

-- Theorem stating the ratio of the perimeters
theorem ratio_of_perimeters : P_square / P_rectangle = 2 :=
by sorry

end ratio_of_perimeters_l197_197078


namespace max_a_l197_197797

theorem max_a {a b c d : ℤ} (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : d < 100) : a ≤ 2367 :=
by {
  have h_b : b ≤ 3 * c - 1 := by linarith,
  have h_c : c ≤ 4 * d - 1 := by linarith,
  have h_d : d ≤ 99 := by linarith,
  have h_max_a := calc
    a ≤ 2 * b - 1 : by linarith
    ... ≤ 2 * (3 * c - 1) - 1 : by linarith
    ... ≤ 6 * c - 3 : by linarith
    ... ≤ 6 * (4 * d - 1) - 3 : by linarith
    ... ≤ 24 * d - 9 : by linarith
    ... ≤ 24 * 99 - 9 : by linarith
    ... = 2367 : by norm_num,
  exact h_max_a,
  sorry
}

end max_a_l197_197797


namespace cubic_roots_l197_197111

theorem cubic_roots (a b x₃ : ℤ)
  (h1 : (2^3 + a * 2^2 + b * 2 + 6 = 0))
  (h2 : (3^3 + a * 3^2 + b * 3 + 6 = 0))
  (h3 : 2 * 3 * x₃ = -6) :
  a = -4 ∧ b = 1 ∧ x₃ = -1 :=
by {
  sorry
}

end cubic_roots_l197_197111


namespace total_number_of_marbles_is_1050_l197_197909

def total_marbles : Nat :=
  let marbles_in_second_bowl := 600
  let marbles_in_first_bowl := (3 * marbles_in_second_bowl) / 4
  marbles_in_first_bowl + marbles_in_second_bowl

theorem total_number_of_marbles_is_1050 : total_marbles = 1050 := by
  sorry

end total_number_of_marbles_is_1050_l197_197909


namespace distinct_letters_permutations_count_l197_197206

-- Define a list of distinct letters
def letters : List Char := ['T', 'E₁', 'E₂', 'E₃', 'N₁', 'N₂', 'S₁', 'S₂']

-- we need to prove the number of permutations of these letters
theorem distinct_letters_permutations_count : 
  list.permutations letters |>.length = 40320 := 
by {
  sorry
}

end distinct_letters_permutations_count_l197_197206


namespace necessary_not_sufficient_condition_l197_197346

variable (a : ℝ) (D : Set ℝ)

def p : Prop := a ∈ D
def q : Prop := ∃ x₀ : ℝ, x₀^2 - a * x₀ - a ≤ -3

theorem necessary_not_sufficient_condition (h : p a D → q a) : D = {x : ℝ | x < -4 ∨ x > 0} :=
sorry

end necessary_not_sufficient_condition_l197_197346


namespace periodic_sequence_not_constant_l197_197310

theorem periodic_sequence_not_constant :
  ∃ (x : ℕ → ℤ), (∀ n : ℕ, x (n+1) = 2 * x n + 3 * x (n-1)) ∧ (∃ T > 0, ∀ n : ℕ, x (n+T) = x n) ∧ (∃ n m : ℕ, n ≠ m ∧ x n ≠ x m) :=
sorry

end periodic_sequence_not_constant_l197_197310


namespace solution_set_of_x_squared_lt_one_l197_197056

theorem solution_set_of_x_squared_lt_one : {x : ℝ | x^2 < 1} = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end solution_set_of_x_squared_lt_one_l197_197056


namespace least_pos_int_satisfies_conditions_l197_197001

theorem least_pos_int_satisfies_conditions :
  ∃ x : ℕ, x > 0 ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (x % 7 = 6) ∧ 
  x = 419 :=
by
  sorry

end least_pos_int_satisfies_conditions_l197_197001


namespace min_y_ellipse_l197_197041

-- Defining the ellipse equation
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 49) + ((y - 3)^2 / 25) = 1

-- Problem statement: Prove that the smallest y-coordinate is -2
theorem min_y_ellipse : 
  ∀ x y, ellipse x y → y ≥ -2 :=
sorry

end min_y_ellipse_l197_197041


namespace Julie_simple_interest_l197_197383

variable (S : ℝ) (r : ℝ) (A : ℝ) (C : ℝ)

def initially_savings (S : ℝ) := S = 784
def half_savings_in_each_account (S A : ℝ) := A = S / 2
def compound_interest_after_two_years (A r : ℝ) := A * (1 + r)^2 - A = 120

theorem Julie_simple_interest
  (S : ℝ) (r : ℝ) (A : ℝ)
  (h1 : initially_savings S)
  (h2 : half_savings_in_each_account S A)
  (h3 : compound_interest_after_two_years A r) :
  A * r * 2 = 112 :=
by 
  sorry

end Julie_simple_interest_l197_197383


namespace cos_seven_pi_over_four_l197_197964

theorem cos_seven_pi_over_four : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end cos_seven_pi_over_four_l197_197964


namespace Adam_total_cost_l197_197158

theorem Adam_total_cost :
  let laptop1_cost := 500
  let laptop2_base_cost := 3 * laptop1_cost
  let discount := 0.15 * laptop2_base_cost
  let laptop2_cost := laptop2_base_cost - discount
  let external_hard_drive := 80
  let mouse := 20
  let software1 := 120
  let software2 := 2 * 120
  let insurance1 := 0.10 * laptop1_cost
  let insurance2 := 0.10 * laptop2_cost
  let total_cost1 := laptop1_cost + external_hard_drive + mouse + software1 + insurance1
  let total_cost2 := laptop2_cost + external_hard_drive + mouse + software2 + insurance2
  total_cost1 + total_cost2 = 2512.5 :=
by
  sorry

end Adam_total_cost_l197_197158


namespace ivy_has_20_collectors_dolls_l197_197641

theorem ivy_has_20_collectors_dolls
  (D : ℕ) (I : ℕ) (C : ℕ)
  (h1 : D = 60)
  (h2 : D = 2 * I)
  (h3 : C = 2 * I / 3) 
  : C = 20 :=
by sorry

end ivy_has_20_collectors_dolls_l197_197641


namespace perfect_square_after_dividing_l197_197747

theorem perfect_square_after_dividing (n : ℕ) (h : n = 16800) : ∃ m : ℕ, (n / 21) = m * m :=
by {
  sorry
}

end perfect_square_after_dividing_l197_197747


namespace vasya_made_mistake_l197_197630

theorem vasya_made_mistake : 
  ∀ (total_digits : ℕ), 
    total_digits = 301 → 
    ¬∃ (n : ℕ), 
      (n ≤ 9 ∧ total_digits = (n * 1)) ∨ 
      (10 ≤ n ∧ n ≤ 99 ∧ total_digits = (9 * 1) + ((n - 9) * 2)) ∨ 
      (100 ≤ n ∧ total_digits = (9 * 1) + (90 * 2) + ((n - 99) * 3)) := 
by 
  sorry

end vasya_made_mistake_l197_197630


namespace selection_methods_eq_total_students_l197_197211

def num_boys := 36
def num_girls := 28
def total_students : ℕ := num_boys + num_girls

theorem selection_methods_eq_total_students :
    total_students = 64 :=
by
  -- Placeholder for the proof
  sorry

end selection_methods_eq_total_students_l197_197211


namespace expression_simplifies_to_zero_l197_197825

theorem expression_simplifies_to_zero (x y : ℝ) (h : x = 2024) :
    5 * (x ^ 3 - 3 * x ^ 2 * y - 2 * x * y ^ 2) -
    3 * (x ^ 3 - 5 * x ^ 2 * y + 2 * y ^ 3) +
    2 * (-x ^ 3 + 5 * x * y ^ 2 + 3 * y ^ 3) = 0 :=
by {
    sorry
}

end expression_simplifies_to_zero_l197_197825


namespace two_f_eq_eight_over_four_plus_x_l197_197694

noncomputable def f : ℝ → ℝ := sorry

theorem two_f_eq_eight_over_four_plus_x (f_def : ∀ x > 0, f (2 * x) = 2 / (2 + x)) :
  ∀ x > 0, 2 * f x = 8 / (4 + x) :=
by
  sorry

end two_f_eq_eight_over_four_plus_x_l197_197694


namespace trapezoid_circumcircle_radius_l197_197272

theorem trapezoid_circumcircle_radius :
  ∀ (BC AD height midline R : ℝ), 
  (BC / AD = (5 / 12)) →
  (height = 17) →
  (midline = height) →
  (midline = (BC + AD) / 2) →
  (BC = 10) →
  (AD = 24) →
  R = 13 :=
by
  intro BC AD height midline R
  intros h_ratio h_height h_midline_eq_height h_midline_eq_avg_bases h_BC h_AD
  -- Proof would go here, but it's skipped for now.
  sorry

end trapezoid_circumcircle_radius_l197_197272


namespace emma_bank_account_balance_l197_197176

theorem emma_bank_account_balance
  (initial_balance : ℕ)
  (daily_spend : ℕ)
  (days_in_week : ℕ)
  (unit_bill : ℕ) :
  initial_balance = 100 → daily_spend = 8 → days_in_week = 7 → unit_bill = 5 →
  (initial_balance - daily_spend * days_in_week) % unit_bill = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end emma_bank_account_balance_l197_197176


namespace problem_statement_l197_197522

variable {R : Type*} [LinearOrderedField R]

def is_even_function (f : R → R) : Prop := ∀ x : R, f x = f (-x)

theorem problem_statement (f : R → R)
  (h1 : is_even_function f)
  (h2 : ∀ x1 x2 : R, x1 ≤ -1 → x2 ≤ -1 → (x2 - x1) * (f x2 - f x1) < 0) :
  f (-1) < f (-3 / 2) ∧ f (-3 / 2) < f 2 :=
sorry

end problem_statement_l197_197522


namespace negation_of_p_l197_197070

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 := by
  sorry

end negation_of_p_l197_197070


namespace profit_without_discount_l197_197768

theorem profit_without_discount (CP SP_original SP_discount : ℝ) (h1 : CP > 0) (h2 : SP_discount = CP * 1.14) (h3 : SP_discount = SP_original * 0.95) :
  (SP_original - CP) / CP * 100 = 20 :=
by
  have h4 : SP_original = SP_discount / 0.95 := by sorry
  have h5 : SP_original = CP * 1.2 := by sorry
  have h6 : (SP_original - CP) / CP * 100 = (CP * 1.2 - CP) / CP * 100 := by sorry
  have h7 : (SP_original - CP) / CP * 100 = 20 := by sorry
  exact h7

end profit_without_discount_l197_197768


namespace find_n_l197_197372

theorem find_n (x n : ℝ) (h_x : x = 0.5) : (9 / (1 + n / x) = 1) → n = 4 := 
by
  intro h
  have h_x_eq : x = 0.5 := h_x
  -- Proof content here covering the intermediary steps
  sorry

end find_n_l197_197372


namespace find_f_of_half_l197_197996

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_half : (∀ x : ℝ, f (Real.logb 4 x) = x) → f (1 / 2) = 2 :=
by
  intros h
  have h1 := h (4 ^ (1 / 2))
  sorry

end find_f_of_half_l197_197996


namespace employee_payment_l197_197923

theorem employee_payment (X Y : ℝ) 
  (h1 : X + Y = 880) 
  (h2 : X = 1.2 * Y) : Y = 400 := by
  sorry

end employee_payment_l197_197923


namespace find_d_l197_197274

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_d (d e : ℝ) (h1 : -(-6) / 3 = 2) (h2 : 3 + d + e - 6 = 9) (h3 : -d / 3 = 6) : d = -18 :=
by
  sorry

end find_d_l197_197274


namespace sum_of_abcd_l197_197117

theorem sum_of_abcd (a b c d: ℝ) (h₁: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h₂: c + d = 10 * a) (h₃: c * d = -11 * b) (h₄: a + b = 10 * c) (h₅: a * b = -11 * d)
  : a + b + c + d = 1210 := by
  sorry

end sum_of_abcd_l197_197117


namespace paco_cookie_problem_l197_197877

theorem paco_cookie_problem (x : ℕ) (hx : x + 9 = 18) : x = 9 :=
by sorry

end paco_cookie_problem_l197_197877


namespace lance_hourly_earnings_l197_197045

theorem lance_hourly_earnings
  (hours_per_week : ℕ)
  (workdays_per_week : ℕ)
  (daily_earnings : ℕ)
  (total_weekly_earnings : ℕ)
  (hourly_wage : ℕ)
  (h1 : hours_per_week = 35)
  (h2 : workdays_per_week = 5)
  (h3 : daily_earnings = 63)
  (h4 : total_weekly_earnings = daily_earnings * workdays_per_week)
  (h5 : total_weekly_earnings = hourly_wage * hours_per_week)
  : hourly_wage = 9 :=
sorry

end lance_hourly_earnings_l197_197045


namespace total_students_l197_197106

variable (T : ℕ)

-- Conditions
def is_girls_percentage (T : ℕ) := 60 / 100 * T
def is_boys_percentage (T : ℕ) := 40 / 100 * T
def boys_not_in_clubs (number_of_boys : ℕ) := 2 / 3 * number_of_boys

theorem total_students (h1 : is_girls_percentage T + is_boys_percentage T = T)
  (h2 : boys_not_in_clubs (is_boys_percentage T) = 40) : T = 150 :=
by
  sorry

end total_students_l197_197106


namespace road_completion_days_l197_197027

variable (L : ℕ) (M_1 : ℕ) (W_1 : ℕ) (t1 : ℕ) (M_2 : ℕ)

theorem road_completion_days : L = 10 ∧ M_1 = 30 ∧ W_1 = 2 ∧ t1 = 5 ∧ M_2 = 60 → D = 15 :=
by
  sorry

end road_completion_days_l197_197027


namespace james_nickels_count_l197_197711

-- Definitions
def total_cents : ℕ := 685
def more_nickels_than_quarters := 11

-- Variables representing the number of nickels and quarters
variables (n q : ℕ)

-- Conditions
axiom h1 : 5 * n + 25 * q = total_cents
axiom h2 : n = q + more_nickels_than_quarters

-- Theorem stating the number of nickels
theorem james_nickels_count : n = 32 := 
by
  -- Proof will go here, marked as "sorry" to complete the statement
  sorry

end james_nickels_count_l197_197711


namespace bsnt_value_l197_197225

theorem bsnt_value (B S N T : ℝ) (hB : 0 < B) (hS : 0 < S) (hN : 0 < N) (hT : 0 < T)
    (h1 : Real.log (B * S) / Real.log 10 + Real.log (B * N) / Real.log 10 = 3)
    (h2 : Real.log (N * T) / Real.log 10 + Real.log (N * S) / Real.log 10 = 4)
    (h3 : Real.log (S * T) / Real.log 10 + Real.log (S * B) / Real.log 10 = 5) :
    B * S * N * T = 10000 :=
sorry

end bsnt_value_l197_197225


namespace electric_sharpens_more_l197_197934

noncomputable def number_of_pencils_hand_crank : ℕ := 360 / 45
noncomputable def number_of_pencils_electric : ℕ := 360 / 20

theorem electric_sharpens_more : number_of_pencils_electric - number_of_pencils_hand_crank = 10 := by
  sorry

end electric_sharpens_more_l197_197934


namespace blue_whale_tongue_weight_l197_197134

theorem blue_whale_tongue_weight (ton_in_pounds : ℕ) (tons : ℕ) (blue_whale_tongue_weight : ℕ) :
  ton_in_pounds = 2000 → tons = 3 → blue_whale_tongue_weight = tons * ton_in_pounds → blue_whale_tongue_weight = 6000 :=
  by
  intros h1 h2 h3
  rw [h2] at h3
  rw [h1] at h3
  exact h3

end blue_whale_tongue_weight_l197_197134


namespace algae_difference_l197_197874

-- Define the original number of algae plants.
def original_algae := 809

-- Define the current number of algae plants.
def current_algae := 3263

-- Statement to prove: The difference between the current number of algae plants and the original number of algae plants is 2454.
theorem algae_difference : current_algae - original_algae = 2454 := by
  sorry

end algae_difference_l197_197874


namespace min_AB_DE_l197_197069

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_focus (k x y : ℝ) : Prop := y = k * (x - 1)

theorem min_AB_DE 
(F : (ℝ × ℝ)) 
(A B D E : ℝ × ℝ) 
(k1 k2 : ℝ) 
(hF : F = (1, 0)) 
(hk : k1^2 + k2^2 = 1) 
(hAB : ∀ x y, parabola x y → line_through_focus k1 x y → A = (x, y) ∨ B = (x, y)) 
(hDE : ∀ x y, parabola x y → line_through_focus k2 x y → D = (x, y) ∨ E = (x, y)) 
: |(A.1 - B.1)| + |(D.1 - E.1)| ≥ 24 := 
sorry

end min_AB_DE_l197_197069


namespace find_x_l197_197180

theorem find_x :
  ∀ x : ℝ, (7 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 3) +
  8 / (Real.sqrt (x - 5) + 3) + 13 / (Real.sqrt (x - 5) + 10) = 0) →
  x = 1486 / 225 :=
by
  sorry

end find_x_l197_197180


namespace common_divisors_4n_7n_l197_197101

theorem common_divisors_4n_7n (n : ℕ) (h1 : n < 50) 
    (h2 : (Nat.gcd (4 * n + 5) (7 * n + 6) > 1)) :
    n = 7 ∨ n = 18 ∨ n = 29 ∨ n = 40 := 
  sorry

end common_divisors_4n_7n_l197_197101


namespace perfect_cubes_in_range_l197_197495

theorem perfect_cubes_in_range (K : ℤ) (hK_pos : K > 1) (Z : ℤ) 
  (hZ_eq : Z = K ^ 3) (hZ_range: 600 < Z ∧ Z < 2000) :
  K = 9 ∨ K = 10 ∨ K = 11 ∨ K = 12 :=
by
  sorry

end perfect_cubes_in_range_l197_197495


namespace marys_mother_bought_3_pounds_of_beef_l197_197573

-- Define the variables and constants
def total_paid : ℝ := 16
def cost_of_chicken : ℝ := 2 * 1  -- 2 pounds of chicken
def cost_per_pound_beef : ℝ := 4
def cost_of_oil : ℝ := 1
def shares : ℝ := 3  -- Mary and her two friends

theorem marys_mother_bought_3_pounds_of_beef:
  total_paid - (cost_of_chicken / shares) - cost_of_oil = 3 * cost_per_pound_beef :=
by
  -- the proof goes here
  sorry

end marys_mother_bought_3_pounds_of_beef_l197_197573


namespace correlation_height_weight_l197_197545

def is_functional_relationship (pair: String) : Prop :=
  pair = "The area of a square and its side length" ∨
  pair = "The distance traveled by a vehicle moving at a constant speed and time"

def has_no_correlation (pair: String) : Prop :=
  pair = "A person's height and eyesight"

def is_correlation (pair: String) : Prop :=
  ¬ is_functional_relationship pair ∧ ¬ has_no_correlation pair

theorem correlation_height_weight :
  is_correlation "A person's height and weight" :=
by sorry

end correlation_height_weight_l197_197545


namespace mass_percentage_O_in_CaO_l197_197789

theorem mass_percentage_O_in_CaO :
  let molar_mass_Ca := 40.08
  let molar_mass_O := 16.00
  let molar_mass_CaO := molar_mass_Ca + molar_mass_O
  let mass_percentage_O := (molar_mass_O / molar_mass_CaO) * 100
  mass_percentage_O = 28.53 :=
by
  sorry

end mass_percentage_O_in_CaO_l197_197789


namespace problem1_problem2_l197_197778

-- Let's define the first problem statement in Lean
theorem problem1 : 2 - 7 * (-3) + 10 + (-2) = 31 := sorry

-- Let's define the second problem statement in Lean
theorem problem2 : -1^2022 + 24 + (-2)^3 - 3^2 * (-1/3)^2 = 14 := sorry

end problem1_problem2_l197_197778


namespace multiples_of_three_l197_197115

theorem multiples_of_three (a b : ℤ) (h : 9 ∣ (a^2 + a * b + b^2)) : 3 ∣ a ∧ 3 ∣ b :=
by {
  sorry
}

end multiples_of_three_l197_197115


namespace equal_share_expense_l197_197713

theorem equal_share_expense (L B C X : ℝ) : 
  let T := L + B + C - X
  let share := T / 3 
  L + (share - L) == (B + C - X - 2 * L) / 3 := 
by
  sorry

end equal_share_expense_l197_197713


namespace percentage_decrease_of_larger_angle_l197_197430

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end percentage_decrease_of_larger_angle_l197_197430


namespace not_prime_257_1092_1092_l197_197005

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_257_1092_1092 :
  is_prime 1093 →
  ¬ is_prime (257 ^ 1092 + 1092) :=
by
  intro h_prime_1093
  -- Detailed steps are omitted, proof goes here
  sorry

end not_prime_257_1092_1092_l197_197005


namespace number_of_kiwis_l197_197280

/-
There are 500 pieces of fruit in a crate. One fourth of the fruits are apples,
20% are oranges, one fifth are strawberries, and the rest are kiwis.
Prove that the number of kiwis is 175.
-/

theorem number_of_kiwis (total_fruits apples oranges strawberries kiwis : ℕ)
  (h1 : total_fruits = 500)
  (h2 : apples = total_fruits / 4)
  (h3 : oranges = 20 * total_fruits / 100)
  (h4 : strawberries = total_fruits / 5)
  (h5 : kiwis = total_fruits - (apples + oranges + strawberries)) :
  kiwis = 175 :=
sorry

end number_of_kiwis_l197_197280


namespace compare_neg_fractions_l197_197783

theorem compare_neg_fractions : (-2 / 3 : ℚ) < -3 / 5 :=
by
  sorry

end compare_neg_fractions_l197_197783


namespace find_l_in_triangle_l197_197380

/-- In triangle XYZ, if XY = 5, YZ = 12, XZ = 13, and YM is the angle bisector from vertex Y with YM = l * sqrt 2, then l equals 60/17. -/
theorem find_l_in_triangle (XY YZ XZ : ℝ) (YM l : ℝ) (hXY : XY = 5) (hYZ : YZ = 12) (hXZ : XZ = 13) (hYM : YM = l * Real.sqrt 2) : 
    l = 60 / 17 :=
sorry

end find_l_in_triangle_l197_197380


namespace find_a_l197_197638

theorem find_a (a : ℚ) :
  let p1 := (3, 4)
  let p2 := (-4, 1)
  let direction_vector := (a, -2)
  let vector_between_points := (p2.1 - p1.1, p2.2 - p1.2)
  ∃ k : ℚ, direction_vector = (k * vector_between_points.1, k * vector_between_points.2) →
  a = -14 / 3 := by
    sorry

end find_a_l197_197638


namespace cube_side_length_of_paint_cost_l197_197261

theorem cube_side_length_of_paint_cost (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  cost_per_kg = 20 ∧ coverage_per_kg = 15 ∧ total_cost = 200 →
  6 * side_length ^ 2 = (total_cost / cost_per_kg) * coverage_per_kg →
  side_length = 5 :=
by
  intros h1 h2
  sorry

end cube_side_length_of_paint_cost_l197_197261


namespace evaluate_using_horners_method_l197_197441

def f (x : ℝ) : ℝ := 3 * x^6 + 12 * x^5 + 8 * x^4 - 3.5 * x^3 + 7.2 * x^2 + 5 * x - 13

theorem evaluate_using_horners_method :
  f 6 = 243168.2 :=
by
  sorry

end evaluate_using_horners_method_l197_197441


namespace speed_difference_l197_197164

def anna_time_min := 15
def ben_time_min := 25
def distance_miles := 8

def anna_speed_mph := (distance_miles : ℚ) / (anna_time_min / 60 : ℚ)
def ben_speed_mph := (distance_miles : ℚ) / (ben_time_min / 60 : ℚ)

theorem speed_difference : (anna_speed_mph - ben_speed_mph : ℚ) = 12.8 := by {
  sorry
}

end speed_difference_l197_197164


namespace cannot_cover_chessboard_with_one_corner_removed_l197_197327

theorem cannot_cover_chessboard_with_one_corner_removed :
  ¬ (∃ (f : Fin (8*8 - 1) → Fin (64-1) × Fin (64-1)), 
        (∀ (i j : Fin (64-1)), 
          i ≠ j → f i ≠ f j) ∧ 
        (∀ (i : Fin (8 * 8 - 1)), 
          (f i).fst + (f i).snd = 2)) :=
by
  sorry

end cannot_cover_chessboard_with_one_corner_removed_l197_197327


namespace value_of_f_f_3_l197_197563

def f (x : ℝ) := 3 * x^2 + 3 * x - 2

theorem value_of_f_f_3 : f (f 3) = 3568 :=
by {
  -- Definition of f is already given in the conditions
  sorry
}

end value_of_f_f_3_l197_197563


namespace event_B_more_likely_l197_197236

theorem event_B_more_likely (A B : Set (ℕ → ℕ)) 
  (hA : ∀ ω, ω ∈ A ↔ ∃ i j, i ≠ j ∧ ω i = ω j)
  (hB : ∀ ω, ω ∈ B ↔ ∀ i j, i ≠ j → ω i ≠ ω j) :
  ∃ prob_A prob_B : ℚ, prob_A = 4 / 9 ∧ prob_B = 5 / 9 ∧ prob_B > prob_A :=
by
  sorry

end event_B_more_likely_l197_197236


namespace find_radius_of_sphere_l197_197317

def radius_of_sphere (width : ℝ) (depth : ℝ) (r : ℝ) : Prop :=
  (width / 2) ^ 2 + (r - depth) ^ 2 = r ^ 2

theorem find_radius_of_sphere (r : ℝ) : radius_of_sphere 30 10 r → r = 16.25 :=
by
  intros h1
  -- sorry is a placeholder for the actual proof
  sorry

end find_radius_of_sphere_l197_197317


namespace possible_roots_l197_197138

theorem possible_roots (a b p q : ℤ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : a ≠ b)
  (h4 : p = -(a + b))
  (h5 : q = ab)
  (h6 : (a + p) % (q - 2 * b) = 0) :
  a = 1 ∨ a = 3 :=
  sorry

end possible_roots_l197_197138


namespace trig_expression_simplify_l197_197093

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trig_expression_simplify_l197_197093


namespace age_difference_l197_197937

theorem age_difference (A B : ℕ) (h1 : B = 34) (h2 : A + 10 = 2 * (B - 10)) : A - B = 4 :=
by
  sorry

end age_difference_l197_197937


namespace negation_of_statement_l197_197586

theorem negation_of_statement :
  ¬ (∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 := by
  sorry

end negation_of_statement_l197_197586


namespace value_of_expression_l197_197368

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 5 = 23 :=
by
  -- proof goes here
  sorry

end value_of_expression_l197_197368


namespace john_total_expense_l197_197219

-- Define variables
variables (M D : ℝ)

-- Define the conditions
axiom cond1 : M = 20 * D
axiom cond2 : M = 24 * (D - 3)

-- State the theorem to prove
theorem john_total_expense : M = 360 :=
by
  -- Add the proof steps here
  sorry

end john_total_expense_l197_197219


namespace y_intercept_of_line_b_l197_197864

noncomputable def line_b_y_intercept (b : Type) [HasElem ℝ b] : Prop :=
  ∃ (m : ℝ) (c : ℝ), (m = -3) ∧ (c = 7) ∧ ∀ (x : ℝ) (y : ℝ), (x, y) ∈ b → y = -3 * x + c

theorem y_intercept_of_line_b (b : Type) [HasElem (ℝ × ℝ) b] :
  (∃ (p : ℝ × ℝ), p = (3, -2) ∧ ∃ (q : line_b_y_intercept b), q) →
  ∃ (c : ℝ), c = 7 :=
by
  intro h
  sorry

end y_intercept_of_line_b_l197_197864


namespace last_digit_of_sum_of_powers_l197_197181

theorem last_digit_of_sum_of_powers {a b c d : ℕ} 
  (h1 : a = 2311) (h2 : b = 5731) (h3 : c = 3467) (h4 : d = 6563) 
  : (a^b + c^d) % 10 = 4 := by
  sorry

end last_digit_of_sum_of_powers_l197_197181


namespace proof_minimum_value_l197_197350

noncomputable def minimum_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : Prop :=
  (1 / a + a / b) ≥ 1 + 2 * Real.sqrt 2

theorem proof_minimum_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : minimum_value_inequality a b h1 h2 h3 :=
  by
    sorry

end proof_minimum_value_l197_197350


namespace termite_ridden_fraction_l197_197122

theorem termite_ridden_fraction (T : ℝ) 
    (h1 : 5/8 * T > 0)
    (h2 : 3/8 * T = 0.125) : T = 1/8 :=
by
  sorry

end termite_ridden_fraction_l197_197122


namespace find_a1_l197_197223

-- Definitions stemming from the conditions in the problem
def arithmetic_seq (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

def is_geometric (a₁ a₃ a₆ : ℕ) : Prop :=
  ∃ r : ℕ, a₃ = r * a₁ ∧ a₆ = r^2 * a₁

theorem find_a1 :
  ∀ a₁ : ℕ,
    (arithmetic_seq a₁ 3 1 = a₁) ∧
    (arithmetic_seq a₁ 3 3 = a₁ + 6) ∧
    (arithmetic_seq a₁ 3 6 = a₁ + 15) ∧
    is_geometric a₁ (a₁ + 6) (a₁ + 15) →
    a₁ = 12 :=
by
  intros
  sorry

end find_a1_l197_197223


namespace complex_exp_power_cos_angle_l197_197993

theorem complex_exp_power_cos_angle (z : ℂ) (h : z + 1/z = 2 * Complex.cos (Real.pi / 36)) :
    z^1000 + 1/(z^1000) = 2 * Complex.cos (Real.pi * 2 / 9) :=
by
  sorry

end complex_exp_power_cos_angle_l197_197993


namespace total_pages_read_l197_197725

-- Define the reading rates
def ReneReadingRate : ℕ := 30  -- pages in 60 minutes
def LuluReadingRate : ℕ := 27  -- pages in 60 minutes
def CherryReadingRate : ℕ := 25  -- pages in 60 minutes

-- Total time in minutes
def totalTime : ℕ := 240  -- minutes

-- Define a function to calculate pages read in given time
def pagesRead (rate : ℕ) (time : ℕ) : ℕ :=
  rate * (time / 60)

-- Theorem to prove the total number of pages read
theorem total_pages_read :
  pagesRead ReneReadingRate totalTime +
  pagesRead LuluReadingRate totalTime +
  pagesRead CherryReadingRate totalTime = 328 :=
by
  -- Proof is not required, hence replaced with sorry
  sorry

end total_pages_read_l197_197725


namespace simon_legos_l197_197249

theorem simon_legos (k b s : ℕ) 
  (h_kent : k = 40)
  (h_bruce : b = k + 20)
  (h_simon : s = b + b / 5) : 
  s = 72 := by
  -- sorry, proof not required.
  sorry

end simon_legos_l197_197249


namespace pendulum_faster_17_seconds_winter_l197_197281

noncomputable def pendulum_period (l g : ℝ) : ℝ :=
  2 * Real.pi * Real.sqrt (l / g)

noncomputable def pendulum_seconds_faster_in_winter (T : ℝ) (l : ℝ) (g : ℝ) (shorten : ℝ) (hours : ℝ) : ℝ :=
  let summer_period := T
  let winter_length := l - shorten
  let winter_period := pendulum_period winter_length g
  let summer_cycles := (hours * 60 * 60) / summer_period
  let winter_cycles := (hours * 60 * 60) / winter_period
  winter_cycles - summer_cycles

theorem pendulum_faster_17_seconds_winter :
  let T := 1
  let l := 980 * (1 / (4 * Real.pi ^ 2))
  let g := 980
  let shorten := 0.01 / 100
  let hours := 24
  pendulum_seconds_faster_in_winter T l g shorten hours = 17 :=
by
  sorry

end pendulum_faster_17_seconds_winter_l197_197281


namespace problem_1_problem_2_l197_197203

noncomputable def f (x a : ℝ) : ℝ := abs (x + a) + abs (x - 2)

-- (1) Prove that, given f(x) and a = -3, the solution set for f(x) ≥ 3 is (-∞, 1] ∪ [4, +∞)
theorem problem_1 (x : ℝ) : 
  (∃ (a : ℝ), a = -3 ∧ f x a ≥ 3) ↔ (x ≤ 1 ∨ x ≥ 4) :=
sorry

-- (2) Prove that for f(x) to be ≥ 3 for all x, the range of a is a ≥ 1 or a ≤ -5
theorem problem_2 : 
  (∀ (x : ℝ), f x a ≥ 3) ↔ (a ≥ 1 ∨ a ≤ -5) :=
sorry

end problem_1_problem_2_l197_197203


namespace shirt_wallet_ratio_l197_197403

theorem shirt_wallet_ratio
  (F W S : ℕ)
  (hF : F = 30)
  (hW : W = F + 60)
  (h_total : S + W + F = 150) :
  S / W = 1 / 3 := by
  sorry

end shirt_wallet_ratio_l197_197403


namespace pyramid_surface_area_l197_197938

noncomputable def total_surface_area : Real :=
  let ab := 14
  let bc := 8
  let pf := 15
  let base_area := ab * bc
  let fm := ab / 2
  let pm_ab := Real.sqrt (pf^2 + fm^2)
  let pm_bc := Real.sqrt (pf^2 + (bc / 2)^2)
  base_area + 2 * (ab / 2 * pm_ab) + 2 * (bc / 2 * pm_bc)

theorem pyramid_surface_area :
  total_surface_area = 112 + 14 * Real.sqrt 274 + 8 * Real.sqrt 241 := by
  sorry

end pyramid_surface_area_l197_197938


namespace pieces_left_to_place_l197_197720

noncomputable def total_pieces : ℕ := 300
noncomputable def reyn_pieces : ℕ := 25
noncomputable def rhys_pieces : ℕ := 2 * reyn_pieces
noncomputable def rory_pieces : ℕ := 3 * reyn_pieces
noncomputable def placed_pieces : ℕ := reyn_pieces + rhys_pieces + rory_pieces
noncomputable def remaining_pieces : ℕ := total_pieces - placed_pieces

theorem pieces_left_to_place : remaining_pieces = 150 :=
by sorry

end pieces_left_to_place_l197_197720


namespace find_real_x_l197_197504

noncomputable def solution_set (x : ℝ) := (5 ≤ x) ∧ (x < 5.25)

theorem find_real_x (x : ℝ) :
  (⌊x * ⌊x⌋⌋ = 20) ↔ solution_set x :=
by
  sorry

end find_real_x_l197_197504


namespace tangent_slope_at_one_l197_197334

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_slope_at_one : deriv f 1 = 2 * Real.exp 1 := sorry

end tangent_slope_at_one_l197_197334


namespace total_number_of_marbles_is_1050_l197_197908

def total_marbles : Nat :=
  let marbles_in_second_bowl := 600
  let marbles_in_first_bowl := (3 * marbles_in_second_bowl) / 4
  marbles_in_first_bowl + marbles_in_second_bowl

theorem total_number_of_marbles_is_1050 : total_marbles = 1050 := by
  sorry

end total_number_of_marbles_is_1050_l197_197908


namespace total_amount_spent_is_40_l197_197571

-- Definitions based on conditions
def tomatoes_pounds : ℕ := 2
def tomatoes_price_per_pound : ℕ := 5
def apples_pounds : ℕ := 5
def apples_price_per_pound : ℕ := 6

-- Total amount spent computed
def total_spent : ℕ :=
  (tomatoes_pounds * tomatoes_price_per_pound) +
  (apples_pounds * apples_price_per_pound)

-- The Lean theorem statement
theorem total_amount_spent_is_40 : total_spent = 40 := by
  unfold total_spent
  unfold tomatoes_pounds tomatoes_price_per_pound apples_pounds apples_price_per_pound
  calc
    2 * 5 + 5 * 6 = 10 + 30 : by rfl
    ... = 40 : by rfl

end total_amount_spent_is_40_l197_197571


namespace opposite_of_neg_three_fifths_l197_197588

theorem opposite_of_neg_three_fifths :
  -(-3 / 5) = 3 / 5 :=
by
  sorry

end opposite_of_neg_three_fifths_l197_197588


namespace integer_solutions_count_l197_197682

theorem integer_solutions_count : ∃ (s : Finset ℤ), (∀ x ∈ s, x^2 - x - 2 ≤ 0) ∧ (Finset.card s = 4) :=
by
  sorry

end integer_solutions_count_l197_197682


namespace possible_values_of_f2001_l197_197894

noncomputable def f : ℕ → ℝ := sorry

theorem possible_values_of_f2001 (f : ℕ → ℝ)
    (H : ∀ a b : ℕ, a > 1 → b > 1 → ∀ d : ℕ, d = Nat.gcd a b → 
           f (a * b) = f d * (f (a / d) + f (b / d))) :
    f 2001 = 0 ∨ f 2001 = 1/2 :=
sorry

end possible_values_of_f2001_l197_197894


namespace solve_inequality_l197_197515

theorem solve_inequality (a b x : ℝ) (h : a ≠ b) :
  a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2 ↔ 0 ≤ x ∧ x ≤ 1 :=
sorry

end solve_inequality_l197_197515


namespace floor_e_is_two_l197_197653

noncomputable def e : ℝ := Real.exp 1

theorem floor_e_is_two : ⌊e⌋ = 2 := by
  sorry

end floor_e_is_two_l197_197653


namespace factorial_difference_l197_197036

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem factorial_difference : fact 10 - fact 9 = 3265920 := by
  have h1 : fact 9 = 362880 := sorry
  have h2 : fact 10 = 10 * fact 9 := by rw [fact]
  calc
    fact 10 - fact 9
        = 10 * fact 9 - fact 9 := by rw [h2]
    ... = 9 * fact 9 := by ring
    ... = 9 * 362880 := by rw [h1]
    ... = 3265920 := by norm_num

end factorial_difference_l197_197036


namespace find_range_m_l197_197359

def p (m : ℝ) : Prop := m > 2 ∨ m < -2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem find_range_m (h₁ : ¬ p m) (h₂ : q m) : (1 : ℝ) < m ∧ m ≤ 2 :=
by sorry

end find_range_m_l197_197359


namespace remaining_distance_l197_197215

-- Definitions based on the conditions
def total_distance : ℕ := 78
def first_leg : ℕ := 35
def second_leg : ℕ := 18

-- The theorem we want to prove
theorem remaining_distance : total_distance - (first_leg + second_leg) = 25 := by
  sorry

end remaining_distance_l197_197215


namespace volume_of_prism_l197_197194

noncomputable def volume_of_triangular_prism
  (area_lateral_face : ℝ)
  (distance_cc1_to_lateral_face : ℝ) : ℝ :=
  area_lateral_face * distance_cc1_to_lateral_face

theorem volume_of_prism (area_lateral_face : ℝ) 
    (distance_cc1_to_lateral_face : ℝ)
    (h_area : area_lateral_face = 4)
    (h_distance : distance_cc1_to_lateral_face = 2):
  volume_of_triangular_prism area_lateral_face distance_cc1_to_lateral_face = 4 := by
  sorry

end volume_of_prism_l197_197194


namespace max_value_trig_l197_197790

theorem max_value_trig (x : ℝ) : ∃ y : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ y := by
  use 5
  sorry

end max_value_trig_l197_197790


namespace trigonometric_identity_l197_197088

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trigonometric_identity_l197_197088


namespace cars_in_north_america_correct_l197_197458

def total_cars_produced : ℕ := 6755
def cars_produced_in_europe : ℕ := 2871

def cars_produced_in_north_america : ℕ := total_cars_produced - cars_produced_in_europe

theorem cars_in_north_america_correct : cars_produced_in_north_america = 3884 :=
by sorry

end cars_in_north_america_correct_l197_197458


namespace distance_focus_directrix_l197_197890

theorem distance_focus_directrix (y x p : ℝ) (h : y^2 = 4 * x) (hp : 2 * p = 4) : p = 2 :=
by sorry

end distance_focus_directrix_l197_197890


namespace find_a_range_l197_197267

noncomputable def monotonic_func_a_range : Set ℝ :=
  {a : ℝ | ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → (3 * x^2 + a ≥ 0 ∨ 3 * x^2 + a ≤ 0)}

theorem find_a_range :
  monotonic_func_a_range = {a | a ≤ -27} ∪ {a | a ≥ 0} :=
by
  sorry

end find_a_range_l197_197267


namespace ratio_of_rises_l197_197598

noncomputable def radius_narrower_cone : ℝ := 4
noncomputable def radius_wider_cone : ℝ := 8
noncomputable def sphere_radius : ℝ := 2

noncomputable def height_ratio (h1 h2 : ℝ) : Prop := h1 = 4 * h2

noncomputable def volume_displacement := (4 / 3) * Real.pi * (sphere_radius^3)

noncomputable def new_height_narrower (h1 : ℝ) : ℝ := h1 + (volume_displacement / ((Real.pi * (radius_narrower_cone^2))))

noncomputable def new_height_wider (h2 : ℝ) : ℝ := h2 + (volume_displacement / ((Real.pi * (radius_wider_cone^2))))

theorem ratio_of_rises (h1 h2 : ℝ) (hr : height_ratio h1 h2) :
  (new_height_narrower h1 - h1) / (new_height_wider h2 - h2) = 4 :=
sorry

end ratio_of_rises_l197_197598


namespace trigonometric_identity_l197_197085

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trigonometric_identity_l197_197085


namespace intersection_complement_l197_197356

open Finset

variable (U A B : Finset ℕ)
variable [DecidableEq ℕ]

def U := {2, 3, 4, 5, 6}
def A := {2, 3, 4}
def B := {2, 3, 5}

theorem intersection_complement :
    A ∩ (U \ B) = {4} :=
by
  sorry

end intersection_complement_l197_197356


namespace butterfly_count_l197_197963

theorem butterfly_count (total_butterflies : ℕ) (one_third_flew_away : ℕ) (initial_butterflies : total_butterflies = 9) (flew_away : one_third_flew_away = total_butterflies / 3) : 
(total_butterflies - one_third_flew_away) = 6 := by
  sorry

end butterfly_count_l197_197963


namespace geometric_series_sum_eq_4_div_3_l197_197478

theorem geometric_series_sum_eq_4_div_3 (a : ℝ) (r : ℝ) (h₀ : a = 1) (h₁ : r = 1 / 4) :
  ∑' n : ℕ, a * r^n = 4 / 3 := by
  sorry

end geometric_series_sum_eq_4_div_3_l197_197478


namespace race_distance_l197_197703

/-
In a race, the ratio of the speeds of two contestants A and B is 3 : 4.
A has a start of 140 m.
A wins by 20 m.
Prove that the total distance of the race is 360 times the common speed factor.
-/
theorem race_distance (x D : ℕ)
  (ratio_A_B : ∀ (speed_A speed_B : ℕ), speed_A / speed_B = 3 / 4)
  (start_A : ∀ (start : ℕ), start = 140) 
  (win_A : ∀ (margin : ℕ), margin = 20) :
  D = 360 * x := 
sorry

end race_distance_l197_197703


namespace number_of_female_athletes_l197_197465

theorem number_of_female_athletes (male_athletes female_athletes male_selected female_selected : ℕ)
  (h1 : male_athletes = 56)
  (h2 : female_athletes = 42)
  (h3 : male_selected = 8)
  (ratio : male_athletes / female_athletes = 4 / 3)
  (stratified_sampling : female_selected = (3 / 4) * male_selected)
  : female_selected = 6 := by
  sorry

end number_of_female_athletes_l197_197465


namespace outer_boundary_diameter_l197_197304

def width_jogging_path : ℝ := 4
def width_garden_ring : ℝ := 10
def diameter_pond : ℝ := 12

theorem outer_boundary_diameter : 2 * (diameter_pond / 2 + width_garden_ring + width_jogging_path) = 40 := by
  sorry

end outer_boundary_diameter_l197_197304


namespace reduced_price_is_25_l197_197314

noncomputable def original_price (P : ℝ) := P
noncomputable def reduced_price (P : ℝ) := P * 0.85
noncomputable def amount_of_wheat_original (P : ℝ) := 500 / P
noncomputable def amount_of_wheat_reduced (P : ℝ) := 500 / (P * 0.85)

theorem reduced_price_is_25 : 
  ∃ (P : ℝ), reduced_price P = 25 ∧ (amount_of_wheat_reduced P = amount_of_wheat_original P + 3) :=
sorry

end reduced_price_is_25_l197_197314


namespace domain_sqrt_3_plus_2x_domain_1_plus_sqrt_9_minus_x2_domain_sqrt_log_5x_minus_x2_over_4_domain_sqrt_3_minus_x_plus_arccos_l197_197972

-- For the function y = sqrt(3 + 2x)
theorem domain_sqrt_3_plus_2x (x : ℝ) : 3 + 2 * x ≥ 0 -> x ∈ Set.Ici (-3 / 2) :=
by
  sorry

-- For the function f(x) = 1 + sqrt(9 - x^2)
theorem domain_1_plus_sqrt_9_minus_x2 (x : ℝ) : 9 - x^2 ≥ 0 -> x ∈ Set.Icc (-3) 3 :=
by
  sorry

-- For the function φ(x) = sqrt(log((5x - x^2) / 4))
theorem domain_sqrt_log_5x_minus_x2_over_4 (x : ℝ) : (5 * x - x^2) / 4 > 0 ∧ (5 * x - x^2) / 4 ≥ 1 -> x ∈ Set.Icc 1 4 :=
by
  sorry

-- For the function y = sqrt(3 - x) + arccos((x - 2) / 3)
theorem domain_sqrt_3_minus_x_plus_arccos (x : ℝ) : 3 - x ≥ 0 ∧ -1 ≤ (x - 2) / 3 ∧ (x - 2) / 3 ≤ 1 -> x ∈ Set.Icc (-1) 3 :=
by
  sorry

end domain_sqrt_3_plus_2x_domain_1_plus_sqrt_9_minus_x2_domain_sqrt_log_5x_minus_x2_over_4_domain_sqrt_3_minus_x_plus_arccos_l197_197972


namespace savings_percentage_l197_197605

theorem savings_percentage
  (S : ℝ)
  (last_year_saved : ℝ := 0.06 * S)
  (this_year_salary : ℝ := 1.10 * S)
  (this_year_saved : ℝ := 0.10 * this_year_salary)
  (ratio := this_year_saved / last_year_saved * 100):
  ratio = 183.33 := 
sorry

end savings_percentage_l197_197605


namespace cyclist_is_jean_l197_197832

theorem cyclist_is_jean (x x' y y' : ℝ) (hx : x' = 4 * x) (hy : y = 4 * y') : x < y :=
by
  sorry

end cyclist_is_jean_l197_197832


namespace pages_read_in_a_year_l197_197099

-- Definition of the problem conditions
def novels_per_month := 4
def pages_per_novel := 200
def months_per_year := 12

-- Theorem statement corresponding to the problem
theorem pages_read_in_a_year (h1 : novels_per_month = 4) (h2 : pages_per_novel = 200) (h3 : months_per_year = 12) : 
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by
  sorry

end pages_read_in_a_year_l197_197099


namespace possible_values_for_a_l197_197999

def setM : Set ℝ := {x | x^2 + x - 6 = 0}
def setN (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem possible_values_for_a (a : ℝ) : (∀ x, x ∈ setN a → x ∈ setM) ↔ (a = -1 ∨ a = 0 ∨ a = 2 / 3) := 
by
  sorry

end possible_values_for_a_l197_197999


namespace sufficient_but_not_necessary_condition_l197_197344

def condition_p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0
def condition_q (x : ℝ) : Prop := |x - 2| < 1

theorem sufficient_but_not_necessary_condition : 
  (∀ x : ℝ, condition_p x → condition_q x) ∧ ¬(∀ x : ℝ, condition_q x → condition_p x) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l197_197344


namespace David_is_8_years_older_than_Scott_l197_197149

noncomputable def DavidAge : ℕ := 14 -- Since David was 8 years old, 6 years ago
noncomputable def RichardAge : ℕ := DavidAge + 6
noncomputable def ScottAge : ℕ := (RichardAge + 8) / 2 - 8
noncomputable def AgeDifference : ℕ := DavidAge - ScottAge

theorem David_is_8_years_older_than_Scott :
  AgeDifference = 8 :=
by
  sorry

end David_is_8_years_older_than_Scott_l197_197149


namespace coeff_x14_in_quotient_l197_197446

open Polynomial

noncomputable def P : Polynomial ℤ := X ^ 1051 - 1
noncomputable def D : Polynomial ℤ := X ^ 4 + X ^ 3 + 2 * X ^ 2 + X + 1

-- Define the quotient of P by D
noncomputable def Q : Polynomial ℤ := P / D

-- The statement we need to prove
theorem coeff_x14_in_quotient : coeff Q 14 = -1 := 
sorry

end coeff_x14_in_quotient_l197_197446


namespace circle_diameter_l197_197615

theorem circle_diameter (r : ℝ) (h : π * r^2 = 16 * π) : 2 * r = 8 :=
by
  sorry

end circle_diameter_l197_197615


namespace ramesh_installation_cost_l197_197128

noncomputable def labelled_price (discounted_price : ℝ) (discount_rate : ℝ) : ℝ :=
  discounted_price / (1 - discount_rate)

noncomputable def selling_price (labelled_price : ℝ) (profit_rate : ℝ) : ℝ :=
  labelled_price * (1 + profit_rate)

def ramesh_total_cost (purchase_price transport_cost : ℝ) (installation_cost : ℝ) : ℝ :=
  purchase_price + transport_cost + installation_cost

theorem ramesh_installation_cost :
  ∀ (purchase_price discounted_price transport_cost labelled_price profit_rate selling_price installation_cost : ℝ),
  discounted_price = 12500 → transport_cost = 125 → profit_rate = 0.18 → selling_price = 18880 →
  labelled_price = discounted_price / (1 - 0.20) →
  selling_price = labelled_price * (1 + profit_rate) →
  ramesh_total_cost purchase_price transport_cost installation_cost = selling_price →
  installation_cost = 6255 :=
by
  intros
  sorry

end ramesh_installation_cost_l197_197128


namespace least_integer_remainder_l197_197748

theorem least_integer_remainder (n : ℕ) 
  (h₁ : n > 1)
  (h₂ : n % 5 = 2)
  (h₃ : n % 6 = 2)
  (h₄ : n % 7 = 2)
  (h₅ : n % 8 = 2)
  (h₆ : n % 10 = 2): 
  n = 842 := 
by
  sorry

end least_integer_remainder_l197_197748


namespace each_player_gets_seven_l197_197834

-- Define the total number of dominoes and players
def total_dominoes : Nat := 28
def total_players : Nat := 4

-- Define the question for how many dominoes each player would receive
def dominoes_per_player (dominoes players : Nat) : Nat := dominoes / players

-- The theorem to prove each player gets 7 dominoes
theorem each_player_gets_seven : dominoes_per_player total_dominoes total_players = 7 :=
by
  sorry

end each_player_gets_seven_l197_197834


namespace find_k_l197_197452

theorem find_k (k : ℕ) (h : (64 : ℕ) / k = 4) : k = 16 := by
  sorry

end find_k_l197_197452


namespace find_daily_wage_c_l197_197604

noncomputable def daily_wage_c (total_earning : ℕ) (days_a : ℕ) (days_b : ℕ) (days_c : ℕ) (days_d : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) (ratio_d : ℕ) : ℝ :=
  let total_ratio := days_a * ratio_a + days_b * ratio_b + days_c * ratio_c + days_d * ratio_d
  let x := total_earning / total_ratio
  ratio_c * x

theorem find_daily_wage_c :
  daily_wage_c 3780 6 9 4 12 3 4 5 7 = 119.60 :=
by
  sorry

end find_daily_wage_c_l197_197604


namespace merchant_articles_l197_197617

theorem merchant_articles 
   (CP SP : ℝ)
   (N : ℝ)
   (h1 : SP = 1.25 * CP)
   (h2 : N * CP = 16 * SP) : 
   N = 20 := by
   sorry

end merchant_articles_l197_197617


namespace find_M_at_x_eq_3_l197_197565

noncomputable def M (a b c d x : ℝ) := a * x^5 + b * x^3 + c * x + d

theorem find_M_at_x_eq_3
  (a b c d M : ℝ)
  (h₀ : d = -5)
  (h₁ : 243 * a + 27 * b + 3 * c = -12) :
  M = -17 :=
by
  sorry

end find_M_at_x_eq_3_l197_197565


namespace blue_bordered_area_on_outer_sphere_l197_197626

theorem blue_bordered_area_on_outer_sphere :
  let r := 1 -- cm
  let r1 := 4 -- cm
  let r2 := 6 -- cm
  let A_inner := 27 -- cm^2
  let h := A_inner / (2 * π * r1)
  let A_outer := 2 * π * r2 * h
  A_outer = 60.75 := sorry

end blue_bordered_area_on_outer_sphere_l197_197626


namespace find_m_l197_197808

open Set

def A (m: ℝ) := {x : ℝ | x^2 - m * x + m^2 - 19 = 0}

def B := {x : ℝ | x^2 - 5 * x + 6 = 0}

def C := ({2, -4} : Set ℝ)

theorem find_m (m : ℝ) (ha : A m ∩ B ≠ ∅) (hb : A m ∩ C = ∅) : m = -2 :=
  sorry

end find_m_l197_197808


namespace find_real_x_l197_197503

noncomputable def solution_set (x : ℝ) := (5 ≤ x) ∧ (x < 5.25)

theorem find_real_x (x : ℝ) :
  (⌊x * ⌊x⌋⌋ = 20) ↔ solution_set x :=
by
  sorry

end find_real_x_l197_197503


namespace sum_of_products_of_two_at_a_time_l197_197278

theorem sum_of_products_of_two_at_a_time (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : a + b + c = 21) : 
  a * b + b * c + a * c = 100 := 
  sorry

end sum_of_products_of_two_at_a_time_l197_197278


namespace nursing_home_beds_l197_197931

/-- A community plans to build a nursing home with 100 rooms, consisting of single, double, and triple rooms.
    Let t be the number of single rooms (1 nursing bed), double rooms (2 nursing beds) is twice the single rooms,
    and the rest are triple rooms (3 nursing beds).
    The equations are:
    - number of double rooms: 2 * t
    - number of single rooms: t
    - number of triple rooms: 100 - 3 * t
    - total number of nursing beds: t + 2 * (2 * t) + 3 * (100 - 3 * t) 
    Prove the following:
    1. If the total number of nursing beds is 200, then t = 25.
    2. The maximum number of nursing beds is 260.
    3. The minimum number of nursing beds is 180.
-/
theorem nursing_home_beds (t : ℕ) (h1 : 10 ≤ t ∧ t ≤ 30) (total_rooms : ℕ := 100) :
  (∀ total_beds, (total_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → total_beds = 200 → t = 25) ∧
  (∀ max_beds, (max_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → t = 10 → max_beds = 260) ∧
  (∀ min_beds, (min_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → t = 30 → min_beds = 180) := 
by
  sorry

end nursing_home_beds_l197_197931


namespace total_marbles_l197_197902

theorem total_marbles :
  let marbles_second_bowl := 600
  let marbles_first_bowl := (3/4) * marbles_second_bowl
  let total_marbles := marbles_first_bowl + marbles_second_bowl
  total_marbles = 1050 := by
  sorry -- proof skipped

end total_marbles_l197_197902


namespace factorize_expression_l197_197049

variable {a b : ℝ} -- define a and b as real numbers

theorem factorize_expression : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l197_197049


namespace card_sorting_moves_upper_bound_l197_197141

theorem card_sorting_moves_upper_bound (n : ℕ) (cells : Fin (n+1) → Fin (n+1)) (cards : Fin (n+1) → Fin (n+1)) : 
  (∃ (moves : (Fin (n+1) × Fin (n+1)) → ℕ),
    (∀ (i : Fin (n+1)), moves (i, cards i) ≤ 2 * n - 1) ∧ 
    (cards 0 = 0 → moves (0, 0) = 2 * n - 1) ∧ 
    (∃! start_pos : Fin (n+1) → Fin (n+1), 
      moves (start_pos (n), start_pos (0)) = 2 * n - 1)) := sorry

end card_sorting_moves_upper_bound_l197_197141


namespace range_of_a_l197_197998

theorem range_of_a (a : ℝ) :
  (∃ A : Finset ℝ, 
    (∀ x, x ∈ A ↔ x^3 - 2 * x^2 + a * x = 0) ∧ A.card = 3) ↔ (a < 0 ∨ (0 < a ∧ a < 1)) :=
by
  sorry

end range_of_a_l197_197998


namespace infinite_solutions_implies_d_eq_five_l197_197043

theorem infinite_solutions_implies_d_eq_five (d : ℝ) :
  (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ (d = 5) := by
sorry

end infinite_solutions_implies_d_eq_five_l197_197043


namespace students_voted_both_issues_l197_197029

-- Define the total number of students.
def total_students : ℕ := 150

-- Define the number of students who voted in favor of the first issue.
def voted_first_issue : ℕ := 110

-- Define the number of students who voted in favor of the second issue.
def voted_second_issue : ℕ := 95

-- Define the number of students who voted against both issues.
def voted_against_both : ℕ := 15

-- Theorem: Number of students who voted in favor of both issues is 70.
theorem students_voted_both_issues : 
  ((voted_first_issue + voted_second_issue) - (total_students - voted_against_both)) = 70 :=
by
  sorry

end students_voted_both_issues_l197_197029


namespace mittens_per_box_l197_197881

theorem mittens_per_box (boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) (h_boxes : boxes = 7) (h_scarves : scarves_per_box = 3) (h_total : total_clothing = 49) : 
  let total_scarves := boxes * scarves_per_box
  let total_mittens := total_clothing - total_scarves
  total_mittens / boxes = 4 :=
by
  sorry

end mittens_per_box_l197_197881


namespace trapezoid_PQRS_PQ_squared_l197_197379

theorem trapezoid_PQRS_PQ_squared
  (PR PS PQ : ℝ)
  (cond1 : PR = 13)
  (cond2 : PS = 17)
  (h : PQ^2 + PR^2 = PS^2) :
  PQ^2 = 120 :=
by
  rw [cond1, cond2] at h
  sorry

end trapezoid_PQRS_PQ_squared_l197_197379


namespace sequence_contains_perfect_square_l197_197735

noncomputable def f (n : ℕ) : ℕ := n + Nat.floor (Real.sqrt n)

theorem sequence_contains_perfect_square (m : ℕ) : ∃ k : ℕ, ∃ p : ℕ, f^[k] m = p * p := by
  sorry

end sequence_contains_perfect_square_l197_197735


namespace greatest_integer_less_than_or_equal_to_l197_197507

theorem greatest_integer_less_than_or_equal_to (x : ℝ) (h : x = 2 + Real.sqrt 3) : 
  ⌊x^3⌋ = 51 :=
by
  have h' : x ^ 3 = (2 + Real.sqrt 3) ^ 3 := by rw [h]
  sorry

end greatest_integer_less_than_or_equal_to_l197_197507


namespace kiwis_to_apples_l197_197417

theorem kiwis_to_apples :
  (1 / 4) * 20 = 10 → (3 / 4) * 12 * (2 / 5) = 18 :=
by
  sorry

end kiwis_to_apples_l197_197417


namespace pages_read_in_a_year_l197_197100

theorem pages_read_in_a_year (novels_per_month : ℕ) (pages_per_novel : ℕ) (months_per_year : ℕ)
  (h1 : novels_per_month = 4) (h2 : pages_per_novel = 200) (h3 : months_per_year = 12) :
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by
  -- Using the given conditions
  rw [h1, h2, h3]
  -- Simplifying the expression
  simp
  sorry

end pages_read_in_a_year_l197_197100


namespace cost_of_27_pounds_l197_197243

def rate_per_pound : ℝ := 1
def weight_pounds : ℝ := 27

theorem cost_of_27_pounds :
  weight_pounds * rate_per_pound = 27 := 
by 
  -- sorry placeholder indicates that the proof is not provided
  sorry

end cost_of_27_pounds_l197_197243


namespace transformed_polynomial_l197_197690

theorem transformed_polynomial (x y : ℝ) (h : y = x + 1 / x) :
  (x^4 - 2*x^3 - 3*x^2 + 2*x + 1 = 0) → (x^2 * (y^2 - y - 3) = 0) :=
by
  sorry

end transformed_polynomial_l197_197690


namespace repaired_shoes_last_correct_l197_197019

noncomputable def repaired_shoes_last := 
  let repair_cost: ℝ := 10.50
  let new_shoes_cost: ℝ := 30.00
  let new_shoes_years: ℝ := 2.0
  let percentage_increase: ℝ := 42.857142857142854 / 100
  (T : ℝ) -> 15.00 = (repair_cost / T) * (1 + percentage_increase) → T = 1

theorem repaired_shoes_last_correct : repaired_shoes_last :=
by
  sorry

end repaired_shoes_last_correct_l197_197019


namespace min_needed_framing_l197_197017

-- Define the original dimensions of the picture
def original_width_inch : ℕ := 5
def original_height_inch : ℕ := 7

-- Define the factor by which the dimensions are doubled
def doubling_factor : ℕ := 2

-- Define the width of the border
def border_width_inch : ℕ := 3

-- Define the function to calculate the new dimensions after doubling
def new_width_inch : ℕ := original_width_inch * doubling_factor
def new_height_inch : ℕ := original_height_inch * doubling_factor

-- Define the function to calculate dimensions including the border
def total_width_inch : ℕ := new_width_inch + 2 * border_width_inch
def total_height_inch : ℕ := new_height_inch + 2 * border_width_inch

-- Define the function to calculate the perimeter of the picture with border
def perimeter_inch : ℕ := 2 * (total_width_inch + total_height_inch)

-- Conversision from inches to feet (1 foot = 12 inches)
def inch_to_foot_conversion_factor : ℕ := 12

-- Define the function to calculate the minimum linear feet of framing needed
noncomputable def min_linear_feet_of_framing : ℕ := (perimeter_inch + inch_to_foot_conversion_factor - 1) / inch_to_foot_conversion_factor

-- The main theorem statement
theorem min_needed_framing : min_linear_feet_of_framing = 6 := by
  -- Proof construction is omitted as per the instructions
  sorry

end min_needed_framing_l197_197017


namespace factorize_expression_l197_197048

variable {a b : ℝ} -- define a and b as real numbers

theorem factorize_expression : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l197_197048


namespace intersection_result_l197_197680

def A : Set ℝ := {x | |x - 2| ≤ 2}

def B : Set ℝ := {y | ∃ x ∈ A, y = -2 * x + 2}

def intersection : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_result : (A ∩ B) = intersection :=
by
  sorry

end intersection_result_l197_197680


namespace tan_alpha_value_l197_197369

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin (Real.pi + α) = 3 / 5) 
  (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.tan α = 3 / 4 := 
sorry

end tan_alpha_value_l197_197369


namespace systematic_sampling_method_l197_197766

-- Define the problem conditions
def total_rows : Nat := 40
def seats_per_row : Nat := 25
def attendees_left (row : Nat) : Nat := if row < total_rows then 18 else 0

-- Problem statement to be proved: The method used is systematic sampling.
theorem systematic_sampling_method :
  (∀ r : Nat, r < total_rows → attendees_left r = 18) →
  (seats_per_row = 25) →
  (∃ k, k > 0 ∧ ∀ r, r < total_rows → attendees_left r = 18 + k * r) →
  True :=
by
  intro h1 h2 h3
  sorry

end systematic_sampling_method_l197_197766


namespace divides_five_iff_l197_197715

theorem divides_five_iff (a : ℤ) : (5 ∣ a^2) ↔ (5 ∣ a) := sorry

end divides_five_iff_l197_197715


namespace shift_upwards_by_2_l197_197584

-- Define the original function
def f (x : ℝ) : ℝ := - x^2

-- Define the shift transformation
def shift_upwards (g : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x, g x + k

-- Define the expected result after shifting f by 2 units upwards
def shifted_f (x : ℝ) : ℝ := - x^2 + 2

-- The proof statement itself
theorem shift_upwards_by_2 :
  shift_upwards f 2 = shifted_f :=
by sorry

end shift_upwards_by_2_l197_197584


namespace find_f_of_7_6_l197_197846

-- Definitions from conditions
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x k : ℤ, f (x + T * (k : ℝ)) = f x

def f_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x = x

-- The periodic function f with period 4
def f : ℝ → ℝ := sorry

-- Hypothesis
axiom f_periodic : periodic_function f 4
axiom f_on_interval : f_in_interval f

-- Theorem to prove
theorem find_f_of_7_6 : f 7.6 = 3.6 :=
by
  sorry

end find_f_of_7_6_l197_197846


namespace smallest_geometric_number_l197_197779

noncomputable def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

def is_smallest_geometric_number (n : ℕ) : Prop :=
  n = 261

theorem smallest_geometric_number :
  ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10)) ∧
  (n / 100 = 2) ∧ (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧
  is_smallest_geometric_number n :=
by
  sorry

end smallest_geometric_number_l197_197779


namespace common_element_exists_l197_197455

theorem common_element_exists {S : Fin 2011 → Set ℤ}
  (h_nonempty : ∀ (i : Fin 2011), (S i).Nonempty)
  (h_consecutive : ∀ (i : Fin 2011), ∃ a b : ℤ, S i = Set.Icc a b)
  (h_common : ∀ (i j : Fin 2011), (S i ∩ S j).Nonempty) :
  ∃ a : ℤ, 0 < a ∧ ∀ (i : Fin 2011), a ∈ S i := sorry

end common_element_exists_l197_197455


namespace cloth_meters_sold_l197_197628

-- Conditions as definitions
def total_selling_price : ℝ := 4500
def profit_per_meter : ℝ := 14
def cost_price_per_meter : ℝ := 86

-- The statement of the problem
theorem cloth_meters_sold (SP : ℝ := cost_price_per_meter + profit_per_meter) :
  total_selling_price / SP = 45 := by
  sorry

end cloth_meters_sold_l197_197628


namespace real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l197_197665

variables {m : ℝ}

-- (1) For z to be a real number
theorem real_number_condition : (m^2 - 3 * m = 0) ↔ (m = 0 ∨ m = 3) :=
by sorry

-- (2) For z to be an imaginary number
theorem imaginary_number_condition : (m^2 - 3 * m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 3) :=
by sorry

-- (3) For z to be a purely imaginary number
theorem pure_imaginary_number_condition : (m^2 - 5 * m + 6 = 0 ∧ m^2 - 3 * m ≠ 0) ↔ (m = 2) :=
by sorry

end real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l197_197665


namespace find_current_l197_197854

open Complex

noncomputable def V : ℂ := 2 + I
noncomputable def Z : ℂ := 2 - 4 * I

theorem find_current :
  V / Z = (1 / 2) * I := 
sorry

end find_current_l197_197854


namespace water_tank_capacity_l197_197466

theorem water_tank_capacity (x : ℝ)
  (h1 : (2 / 3) * x - (1 / 3) * x = 20) : x = 60 := 
  sorry

end water_tank_capacity_l197_197466


namespace butterflies_left_correct_l197_197959

-- Define the total number of butterflies and the fraction that flies away
def butterflies_total : ℕ := 9
def fraction_fly_away : ℚ := 1 / 3

-- Define the number of butterflies left in the garden
def butterflies_left (t : ℕ) (f : ℚ) : ℚ := t - (t : ℚ) * f

-- State the theorem
theorem butterflies_left_correct : butterflies_left butterflies_total fraction_fly_away = 6 := by
  sorry

end butterflies_left_correct_l197_197959


namespace contrapositive_eq_inverse_l197_197893

variable (p q : Prop)

theorem contrapositive_eq_inverse (h1 : p → q) :
  (¬ p → ¬ q) ↔ (q → p) := by
  sorry

end contrapositive_eq_inverse_l197_197893


namespace chessboard_max_squares_l197_197366

def max_squares (m n : ℕ) : ℕ :=
  if m = 1 then n else m + n - 2

theorem chessboard_max_squares (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) : max_squares 1000 1000 = 1998 := 
by
  -- This is the theorem statement representing the maximum number of squares chosen
  -- in a 1000 x 1000 chessboard without having exactly three of them with two in the same row
  -- and two in the same column.
  sorry

end chessboard_max_squares_l197_197366


namespace xyz_stock_final_price_l197_197498

theorem xyz_stock_final_price :
  let s0 := 120
  let s1 := s0 + s0 * 1.5
  let s2 := s1 - s1 * 0.3
  let s3 := s2 + s2 * 0.2
  s3 = 252 := by
  sorry

end xyz_stock_final_price_l197_197498


namespace Morio_age_when_Michiko_was_born_l197_197728

theorem Morio_age_when_Michiko_was_born (Teresa_age_now : ℕ) (Teresa_age_when_Michiko_born : ℕ) (Morio_age_now : ℕ)
  (hTeresa : Teresa_age_now = 59) (hTeresa_born : Teresa_age_when_Michiko_born = 26) (hMorio : Morio_age_now = 71) :
  Morio_age_now - (Teresa_age_now - Teresa_age_when_Michiko_born) = 38 :=
by
  sorry

end Morio_age_when_Michiko_was_born_l197_197728


namespace scientific_notation_361000000_l197_197123

theorem scientific_notation_361000000 :
  361000000 = 3.61 * 10^8 :=
sorry

end scientific_notation_361000000_l197_197123


namespace change_from_fifteen_dollars_l197_197952

theorem change_from_fifteen_dollars : 
  ∀ (cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid : ℕ),
  cost_eggs = 3 →
  cost_pancakes = 2 →
  cost_mug_cocoa = 2 →
  num_mugs = 2 →
  tax = 1 →
  additional_pancakes = 2 →
  additional_mug = 2 →
  paid = 15 →
  paid - (cost_eggs + cost_pancakes + (num_mugs * cost_mug_cocoa) + tax + additional_pancakes + additional_mug) = 1 :=
by
  intros cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid
  sorry

end change_from_fifteen_dollars_l197_197952


namespace time_to_complete_job_l197_197921

-- Define the conditions
variables {A B : ℕ} -- Efficiencies of A and B

-- Assume B's efficiency is 100 units, and A is 130 units.
def efficiency_A : ℕ := 130
def efficiency_B : ℕ := 100

-- Given: A can complete the job in 23 days
def days_A : ℕ := 23

-- Compute total work W. Since A can complete the job in 23 days and its efficiency is 130 units/day:
def total_work : ℕ := efficiency_A * days_A

-- Combined efficiency of A and B
def combined_efficiency : ℕ := efficiency_A + efficiency_B

-- Determine the time taken by A and B working together
def time_A_B_together : ℕ := total_work / combined_efficiency

-- Prove that the time A and B working together is 13 days
theorem time_to_complete_job : time_A_B_together = 13 :=
by
  sorry -- Proof is omitted as per instructions

end time_to_complete_job_l197_197921


namespace complementary_angles_decrease_86_percent_l197_197426

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end complementary_angles_decrease_86_percent_l197_197426


namespace tileable_contains_domino_l197_197880

theorem tileable_contains_domino {m n a b : ℕ} (h_m : m ≥ a) (h_n : n ≥ b) :
  (∀ (x : ℕ) (y : ℕ), x + a ≤ m → y + b ≤ n → ∃ (p : ℕ) (q : ℕ), p = x ∧ q = y) :=
sorry

end tileable_contains_domino_l197_197880


namespace trajectory_eq_l197_197529

-- Define the conditions provided in the problem
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * (m + 3) * x + 2 * (1 - 4 * m^2) + 16 * m^4 + 9 = 0

-- Define the required range for m based on the derivation
def m_valid (m : ℝ) : Prop :=
  -1/7 < m ∧ m < 1

-- Prove that the equation of the trajectory of the circle's center is y = 4(x-3)^2 -1 
-- and it's valid in the required range for x
theorem trajectory_eq (x y : ℝ) :
  (∃ m : ℝ, m_valid m ∧ y = 4 * (x - 3)^2 - 1 ∧ (x = m + 3) ∧ (y = 4 * m^2 - 1)) →
  y = 4 * (x - 3)^2 - 1 ∧ (20/7 < x) ∧ (x < 4) :=
by
  intro h
  cases' h with m hm
  sorry

end trajectory_eq_l197_197529


namespace Jeanine_has_more_pencils_than_Clare_l197_197552

def number_pencils_Jeanine_bought : Nat := 18
def number_pencils_Clare_bought := number_pencils_Jeanine_bought / 2
def number_pencils_given_to_Abby := number_pencils_Jeanine_bought / 3
def number_pencils_Jeanine_now := number_pencils_Jeanine_bought - number_pencils_given_to_Abby 

theorem Jeanine_has_more_pencils_than_Clare :
  number_pencils_Jeanine_now - number_pencils_Clare_bought = 3 := by
  sorry

end Jeanine_has_more_pencils_than_Clare_l197_197552


namespace saved_percentage_this_year_l197_197222

variable (S : ℝ) -- Annual salary last year

-- Conditions
def saved_last_year := 0.06 * S
def salary_this_year := 1.20 * S
def saved_this_year := saved_last_year

-- The goal is to prove that the percentage saved this year is 5%
theorem saved_percentage_this_year :
  (saved_this_year / salary_this_year) * 100 = 5 :=
by sorry

end saved_percentage_this_year_l197_197222


namespace trigonometric_identity_l197_197087

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trigonometric_identity_l197_197087


namespace third_player_game_count_l197_197283

theorem third_player_game_count (fp_games : ℕ) (sp_games : ℕ) (tp_games : ℕ) (total_games : ℕ) 
  (h1 : fp_games = 10) (h2 : sp_games = 21) (h3 : total_games = sp_games) 
  (h4 : total_games = fp_games + tp_games + 1): tp_games = 11 := 
  sorry

end third_player_game_count_l197_197283


namespace Carol_width_eq_24_l197_197780

-- Given conditions
def Carol_length : ℕ := 5
def Jordan_length : ℕ := 2
def Jordan_width : ℕ := 60

-- Required proof: Carol's width is 24 considering equal areas of both rectangles
theorem Carol_width_eq_24 (w : ℕ) (h : Carol_length * w = Jordan_length * Jordan_width) : w = 24 := 
by sorry

end Carol_width_eq_24_l197_197780


namespace part_I_part_II_l197_197518

def sequence_sn (n : ℕ) : ℚ := (3 / 2 : ℚ) * n^2 + (1 / 2 : ℚ) * n

def sequence_a (n : ℕ) : ℕ := 3 * n - 1

def sequence_b (n : ℕ) : ℚ := (1 / 2 : ℚ)^n

def sequence_C (n : ℕ) : ℚ := sequence_a (sequence_a n) + sequence_b (sequence_a n)

def sum_of_first_n_terms (f : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum f

theorem part_I (n : ℕ) : sequence_a n = 3 * n - 1 ∧ sequence_b n = (1 / 2)^n :=
by {
  sorry
}

theorem part_II (n : ℕ) : sum_of_first_n_terms sequence_C n =
  (n * (9 * n + 1) / 2) - (2 / 7) * (1 / 8)^n + (2 / 7) :=
by {
  sorry
}

end part_I_part_II_l197_197518


namespace total_revenue_calculation_l197_197162

-- Define the total number of etchings sold
def total_etchings : ℕ := 16

-- Define the number of etchings sold at $35 each
def etchings_sold_35 : ℕ := 9

-- Define the price per etching sold at $35
def price_per_etching_35 : ℕ := 35

-- Define the price per etching sold at $45
def price_per_etching_45 : ℕ := 45

-- Define the total revenue calculation
def total_revenue : ℕ :=
  let revenue_35 := etchings_sold_35 * price_per_etching_35
  let etchings_sold_45 := total_etchings - etchings_sold_35
  let revenue_45 := etchings_sold_45 * price_per_etching_45
  revenue_35 + revenue_45

-- Theorem stating the total revenue is $630
theorem total_revenue_calculation : total_revenue = 630 := by
  sorry

end total_revenue_calculation_l197_197162


namespace c_in_terms_of_t_l197_197633

theorem c_in_terms_of_t (t a b c : ℝ) (h_t_ne_zero : t ≠ 0)
    (h1 : t^3 + a * t = 0)
    (h2 : b * t^2 + c = 0)
    (h3 : 3 * t^2 + a = 2 * b * t) :
    c = -t^3 :=
by
sorry

end c_in_terms_of_t_l197_197633


namespace general_formula_correct_sequence_T_max_term_l197_197339

open Classical

noncomputable def geometric_sequence_term (n : ℕ) : ℝ :=
  if h : n > 0 then (-1)^(n-1) * (3 / 2^n)
  else 0

noncomputable def geometric_sequence_sum (n : ℕ) : ℝ :=
  if h : n > 0 then 1 - (-1 / 2)^n
  else 0

noncomputable def sequence_T (n : ℕ) : ℝ :=
  geometric_sequence_sum n + 1 / geometric_sequence_sum n

theorem general_formula_correct :
  ∀ n : ℕ, n > 0 → geometric_sequence_term n = (-1)^(n-1) * (3 / 2^n) :=
sorry

theorem sequence_T_max_term :
  ∀ n : ℕ, n > 0 → sequence_T n ≤ sequence_T 1 ∧ sequence_T 1 = 13 / 6 :=
sorry

end general_formula_correct_sequence_T_max_term_l197_197339


namespace quadratic_bounds_l197_197804

variable (a b c: ℝ)

-- Conditions
def quadratic_function (x: ℝ) : ℝ := a * x^2 + b * x + c

def within_range_neg_1_to_1 (h : ∀ x: ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 1) : Prop :=
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -7 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 7

-- Main statement
theorem quadratic_bounds
  (h : ∀ x: ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 1) :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -7 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 7 := sorry

end quadratic_bounds_l197_197804


namespace magnitude_a_minus_2b_l197_197681

noncomputable def magnitude_of_vector_difference : ℝ :=
  let a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
  let b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2)

theorem magnitude_a_minus_2b :
  let a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
  let b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 3 :=
by
  sorry

end magnitude_a_minus_2b_l197_197681


namespace rectangle_volume_l197_197025

theorem rectangle_volume {a b c : ℕ} (h1 : a * b - c * a - b * c = 1) (h2 : c * a = b * c + 1) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a * b * c = 6 :=
sorry

end rectangle_volume_l197_197025


namespace mark_profit_from_selling_magic_card_l197_197871

theorem mark_profit_from_selling_magic_card : 
    ∀ (purchase_price new_value profit : ℕ), 
        purchase_price = 100 ∧ 
        new_value = 3 * purchase_price ∧ 
        profit = new_value - purchase_price 
    → 
        profit = 200 := 
by 
  intros purchase_price new_value profit h,
  cases h with hp1 h,
  cases h with hv1 hp2,
  rw hp1 at hv1,
  rw hp1 at hp2,
  rw hv1 at hp2,
  rw hp2,
  rw hp1,
  norm_num,
  exact eq.refl 200

end mark_profit_from_selling_magic_card_l197_197871


namespace benjamin_decade_expense_l197_197777

-- Define the constants
def yearly_expense : ℕ := 3000
def years : ℕ := 10

-- Formalize the statement
theorem benjamin_decade_expense : yearly_expense * years = 30000 := 
by
  sorry

end benjamin_decade_expense_l197_197777


namespace coffee_ratio_l197_197381

/-- Define the conditions -/
def initial_coffees_per_day := 4
def initial_price_per_coffee := 2
def price_increase_percentage := 50 / 100
def savings_per_day := 2

/-- Define the price calculations -/
def new_price_per_coffee := initial_price_per_coffee + (initial_price_per_coffee * price_increase_percentage)
def initial_daily_cost := initial_coffees_per_day * initial_price_per_coffee
def new_daily_cost := initial_daily_cost - savings_per_day
def new_coffees_per_day := new_daily_cost / new_price_per_coffee

/-- Prove the ratio -/
theorem coffee_ratio : (new_coffees_per_day / initial_coffees_per_day) = (1 : ℝ) / (2 : ℝ) :=
  by sorry

end coffee_ratio_l197_197381


namespace complex_ratio_max_min_diff_l197_197848

noncomputable def max_minus_min_complex_ratio (z w : ℂ) : ℝ :=
max (1 : ℝ) (0 : ℝ) - min (1 : ℝ) (0 : ℝ)

theorem complex_ratio_max_min_diff (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) : 
  max_minus_min_complex_ratio z w = 1 :=
by sorry

end complex_ratio_max_min_diff_l197_197848


namespace count_integers_with_sum_of_digits_18_l197_197686

def sum_of_digits (n : ℕ) : ℕ := (n / 100) + (n / 10 % 10) + (n % 10)

def valid_integer_count : ℕ :=
  let range := List.range' 700 (900 - 700 + 1)
  List.length $ List.filter (λ n => sum_of_digits n = 18) range

theorem count_integers_with_sum_of_digits_18 :
  valid_integer_count = 17 :=
sorry

end count_integers_with_sum_of_digits_18_l197_197686


namespace cosine_of_angle_between_tangents_l197_197192

-- Definitions based on the conditions given in a)
def circle_eq (x y : ℝ) : Prop := x^2 - 2 * x + y^2 - 2 * y + 1 = 0
def P : ℝ × ℝ := (3, 2)

-- The main theorem to be proved
theorem cosine_of_angle_between_tangents (x y : ℝ)
  (hx : circle_eq x y) : 
  cos_angle_between_tangents := 
  sorry

end cosine_of_angle_between_tangents_l197_197192


namespace determine_angle_A_l197_197520

-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
def sin_rule_condition (a b c A B C : ℝ) : Prop :=
  (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C

-- The proof statement
theorem determine_angle_A (a b c A B C : ℝ) (h : sin_rule_condition a b c A B C) : A = π / 3 :=
  sorry

end determine_angle_A_l197_197520


namespace range_of_2alpha_minus_beta_l197_197815

def condition_range_alpha_beta (α β : ℝ) : Prop := 
  - (Real.pi / 2) < α ∧ α < β ∧ β < (Real.pi / 2)

theorem range_of_2alpha_minus_beta (α β : ℝ) (h : condition_range_alpha_beta α β) : 
  - Real.pi < 2 * α - β ∧ 2 * α - β < Real.pi / 2 :=
sorry

end range_of_2alpha_minus_beta_l197_197815


namespace simon_legos_l197_197250

theorem simon_legos (k b s : ℕ) 
  (h_kent : k = 40)
  (h_bruce : b = k + 20)
  (h_simon : s = b + b / 5) : 
  s = 72 := by
  -- sorry, proof not required.
  sorry

end simon_legos_l197_197250


namespace match_foci_of_parabola_and_hyperbola_l197_197349

noncomputable def focus_of_parabola (a : ℝ) : ℝ :=
a / 4

noncomputable def foci_of_hyperbola : Set ℝ :=
{2, -2}

theorem match_foci_of_parabola_and_hyperbola (a : ℝ) :
  focus_of_parabola a ∈ foci_of_hyperbola ↔ a = 8 ∨ a = -8 :=
by
  -- This is the placeholder for the proof.
  sorry

end match_foci_of_parabola_and_hyperbola_l197_197349


namespace sum_of_coordinates_l197_197257

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 2 = 4) :
  let x := 4
  let y := (f⁻¹ x) / 4
  x + y = 9 / 2 :=
by
  sorry

end sum_of_coordinates_l197_197257


namespace min_blocks_for_wall_l197_197613

noncomputable def min_blocks_needed (length height : ℕ) (block_sizes : List (ℕ × ℕ)) : ℕ :=
  sorry

theorem min_blocks_for_wall :
  min_blocks_needed 120 8 [(1, 3), (1, 2), (1, 1)] = 404 := by
  sorry

end min_blocks_for_wall_l197_197613


namespace probability_zero_after_2017_days_l197_197866

-- Define the people involved
inductive Person
| Lunasa | Merlin | Lyrica
deriving DecidableEq, Inhabited

open Person

-- Define the initial state with each person having their own distinct hat
def initial_state : Person → Person
| Lunasa => Lunasa
| Merlin => Merlin
| Lyrica => Lyrica

-- Define a function that represents switching hats between two people
def switch_hats (p1 p2 : Person) (state : Person → Person) : Person → Person :=
  λ p => if p = p1 then state p2 else if p = p2 then state p1 else state p

-- Define a function to represent the state after n days (iterations)
def iter_switch_hats (n : ℕ) : Person → Person :=
  sorry -- This would involve implementing the iterative random switching

-- Proposition: The probability that after 2017 days, every person has their own hat back is 0
theorem probability_zero_after_2017_days :
  iter_switch_hats 2017 = initial_state → false :=
by
  sorry

end probability_zero_after_2017_days_l197_197866


namespace profit_calculation_l197_197873

def Initial_Value : ℕ := 100
def Multiplier : ℕ := 3
def New_Value : ℕ := Initial_Value * Multiplier
def Profit : ℕ := New_Value - Initial_Value

theorem profit_calculation : Profit = 200 := by
  sorry

end profit_calculation_l197_197873


namespace initial_avg_weight_l197_197593

theorem initial_avg_weight (A : ℝ) (h : 6 * A + 121 = 7 * 151) : A = 156 :=
by
sorry

end initial_avg_weight_l197_197593


namespace adjacent_students_permutations_l197_197898

open Nat

/--
Given 6 students standing in a row, with students labeled A, B, and 4 others, 
prove that the number of permutations in which A and B are adjacent is 240.
-/
theorem adjacent_students_permutations (students : Fin 6 → ℕ) (A B : Fin 6) :
  A ≠ B → 
  ∃ n : ℕ, n = 240 ∧
  (count_adjacent_perms students A B) = n := 
sorry

-- A function to count the permutations where A and B are adjacent
noncomputable def count_adjacent_perms (students : Fin 6 → ℕ) (A B : Fin 6) : ℕ := 
  ((6 - 1)! * 2!)


end adjacent_students_permutations_l197_197898


namespace unique_ordered_triple_lcm_l197_197559

theorem unique_ordered_triple_lcm:
  ∃! (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c), 
    Nat.lcm a b = 2100 ∧ Nat.lcm b c = 3150 ∧ Nat.lcm c a = 4200 :=
by
  sorry

end unique_ordered_triple_lcm_l197_197559


namespace cost_of_two_books_and_one_magazine_l197_197734

-- Definitions of the conditions
def condition1 (x y : ℝ) : Prop := 3 * x + 2 * y = 18.40
def condition2 (x y : ℝ) : Prop := 2 * x + 3 * y = 17.60

-- Proof problem
theorem cost_of_two_books_and_one_magazine (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  2 * x + y = 11.20 :=
sorry

end cost_of_two_books_and_one_magazine_l197_197734


namespace arithmetic_mean_of_smallest_twin_prime_pair_l197_197155

open Nat

/-- Definition of twin prime pair -/
def is_twin_prime_pair (p q : ℕ) : Prop :=
  Prime p ∧ Prime q ∧ q = p + 2

/-- The smallest twin prime pair is (3, 5) and the arithmetic mean of 3 and 5 is 4. -/
theorem arithmetic_mean_of_smallest_twin_prime_pair :
  ∃ (p q : ℕ), is_twin_prime_pair p q ∧ p = 3 ∧ q = 5 ∧ (p + q) / 2 = 4 :=
by
  sorry

end arithmetic_mean_of_smallest_twin_prime_pair_l197_197155


namespace find_x_l197_197673

-- Define the vectors a and b
def a : ℝ × ℝ := (-2, 3)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

-- Define the parallel condition between (b - a) and b
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u  = (k * v.1, k * v.2)

-- The problem statement in Lean 4
theorem find_x (x : ℝ) (h : parallel (b x - a) (b x)) : x = 2 := 
  sorry

end find_x_l197_197673


namespace train_cross_bridge_time_l197_197532

theorem train_cross_bridge_time
  (length_train : ℕ) (speed_train_kmph : ℕ) (length_bridge : ℕ) 
  (km_to_m : ℕ) (hour_to_s : ℕ)
  (h1 : length_train = 165) 
  (h2 : speed_train_kmph = 54) 
  (h3 : length_bridge = 720) 
  (h4 : km_to_m = 1000) 
  (h5 : hour_to_s = 3600) 
  : (length_train + length_bridge) / ((speed_train_kmph * km_to_m) / hour_to_s) = 59 := 
sorry

end train_cross_bridge_time_l197_197532


namespace brooke_sidney_ratio_l197_197727

-- Definitions for the conditions
def sidney_monday : ℕ := 20
def sidney_tuesday : ℕ := 36
def sidney_wednesday : ℕ := 40
def sidney_thursday : ℕ := 50
def brooke_total : ℕ := 438

-- Total jumping jacks by Sidney
def sidney_total : ℕ := sidney_monday + sidney_tuesday + sidney_wednesday + sidney_thursday

-- The ratio of Brooke’s jumping jacks to Sidney's total jumping jacks
def ratio := brooke_total / sidney_total

-- The proof goal
theorem brooke_sidney_ratio : ratio = 3 :=
by
  sorry

end brooke_sidney_ratio_l197_197727


namespace min_value_expression_l197_197989

open Real

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x^2 + y^2 + z^2 = 1) : 
  (∃ (c : ℝ), c = 3 * sqrt 3 / 2 ∧ c ≤ (x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2))) :=
by
  sorry

end min_value_expression_l197_197989


namespace area_larger_sphere_l197_197624

variables {r1 r2 r : ℝ}
variables {A1 A2 : ℝ}

-- Declare constants for the problem
def radius_smaller_sphere : ℝ := 4 -- r1
def radius_larger_sphere : ℝ := 6  -- r2
def radius_ball : ℝ := 1           -- r
def area_smaller_sphere : ℝ := 27  -- A1

-- Given conditions
axiom radius_smaller_sphere_condition : r1 = radius_smaller_sphere
axiom radius_larger_sphere_condition : r2 = radius_larger_sphere
axiom radius_ball_condition : r = radius_ball
axiom area_smaller_sphere_condition : A1 = area_smaller_sphere

-- Statement to be proved
theorem area_larger_sphere :
  r1 = radius_smaller_sphere → r2 = radius_larger_sphere → r = radius_ball → A1 = area_smaller_sphere → A2 = 60.75 :=
by
  intros
  sorry

end area_larger_sphere_l197_197624


namespace correct_option_l197_197224

variable {S : Type*} (A B : Set S) (a : S)

-- Conditions
variable (hA : A.Nonempty) (hB : B.Nonempty) (hAprop : A ⊂ Set.univ) (hBprop : B ⊂ Set.univ)
variable (haA : a ∈ A) (haB : a ∉ B)

-- Proof Goal
theorem correct_option : a ∈ (A ∩ Bᶜ) :=
sorry

end correct_option_l197_197224


namespace expand_polynomial_l197_197786

theorem expand_polynomial : 
  ∀ (x : ℝ), (5 * x - 3) * (2 * x^2 + 4 * x + 1) = 10 * x^3 + 14 * x^2 - 7 * x - 3 :=
by
  intro x
  sorry

end expand_polynomial_l197_197786


namespace find_C_l197_197763

theorem find_C (A B C : ℕ)
  (hA : A = 348)
  (hB : B = A + 173)
  (hC : C = B + 299) :
  C = 820 :=
sorry

end find_C_l197_197763


namespace probability_point_outside_circle_l197_197618

open Classical

noncomputable def prob_point_outside_circle : ℚ :=
  let outcomes := (Σ m : fin 6, fin 6)
  let P := {p : outcomes × outcomes | p.1.1.val + 1 = p.1.2.val + 1 ∧ 
                          p.2.1.val + 1 = p.2.2.val + 1 ∧
                          (p.1.1.val + p.2.1.val + 2)^2 + (p.1.2.val + p.2.2.val + 2)^2 > 17}
  (P.to_finset.card : ℚ) / (6 * 6 * 6 * 6)

theorem probability_point_outside_circle :
  prob_point_outside_circle = 13 / 18 := by
  sorry

end probability_point_outside_circle_l197_197618


namespace permutations_sum_divisible_by_37_l197_197258

theorem permutations_sum_divisible_by_37 (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
    ∃ k, (100 * a + 10 * b + c) + (100 * a + 10 * c + b) + (100 * b + 10 * a + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a) = 37 * k := 
by
  sorry

end permutations_sum_divisible_by_37_l197_197258


namespace change_from_15_dollars_l197_197950

theorem change_from_15_dollars :
  let cost_eggs := 3
  let cost_pancakes := 2
  let cost_mugs_of_cocoa := 2 * 2
  let tax := 1
  let initial_cost := cost_eggs + cost_pancakes + cost_mugs_of_cocoa + tax
  let additional_pancakes := 2
  let additional_mug_of_cocoa := 2
  let additional_cost := additional_pancakes + additional_mug_of_cocoa
  let new_total_cost := initial_cost + additional_cost
  let payment := 15
  let change := payment - new_total_cost
  change = 1 :=
by
  sorry

end change_from_15_dollars_l197_197950


namespace cheapest_option_l197_197361

/-
  Problem: Prove that gathering berries in the forest to make jam is
  the cheapest option for Grandmother Vasya.
-/

def gathering_berries_cost (transportation_cost_per_kg sugar_cost_per_kg : ℕ) := (40 + sugar_cost_per_kg : ℕ)
def buying_berries_cost (berries_cost_per_kg sugar_cost_per_kg : ℕ) := (150 + sugar_cost_per_kg : ℕ)
def buying_ready_made_jam_cost (ready_made_jam_cost_per_kg : ℕ) := (220 * 1.5 : ℕ)

theorem cheapest_option (transportation_cost_per_kg sugar_cost_per_kg berries_cost_per_kg ready_made_jam_cost_per_kg : ℕ) : 
  gathering_berries_cost transportation_cost_per_kg sugar_cost_per_kg < buying_berries_cost berries_cost_per_kg sugar_cost_per_kg ∧
  gathering_berries_cost transportation_cost_per_kg sugar_cost_per_kg < buying_ready_made_jam_cost ready_made_jam_cost_per_kg := 
by
  sorry

end cheapest_option_l197_197361


namespace division_neg4_by_2_l197_197941

theorem division_neg4_by_2 : (-4) / 2 = -2 := sorry

end division_neg4_by_2_l197_197941


namespace prime_root_range_l197_197812

-- Let's define our conditions first
def is_prime (p : ℕ) : Prop := Nat.Prime p

def has_integer_roots (p : ℕ) : Prop :=
  ∃ (x y : ℤ), x ≠ y ∧ x + y = p ∧ x * y = -156 * p

-- Now state the theorem
theorem prime_root_range (p : ℕ) (hp : is_prime p) (hr : has_integer_roots p) : 11 < p ∧ p ≤ 21 :=
by
  sorry

end prime_root_range_l197_197812


namespace who_drank_most_l197_197646

theorem who_drank_most (eunji yujeong yuna : ℝ) 
    (h1 : eunji = 0.5) 
    (h2 : yujeong = 7 / 10) 
    (h3 : yuna = 6 / 10) :
    max (max eunji yujeong) yuna = yujeong :=
by {
    sorry
}

end who_drank_most_l197_197646


namespace sequence_a500_l197_197107

theorem sequence_a500 (a : ℕ → ℤ)
  (h1 : a 1 = 2010)
  (h2 : a 2 = 2011)
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n) :
  a 500 = 2177 :=
sorry

end sequence_a500_l197_197107


namespace sequence_value_l197_197982

theorem sequence_value (a : ℕ → ℤ) (h1 : ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q)
                       (h2 : a 2 = -6) : a 10 = -30 :=
by
  sorry

end sequence_value_l197_197982


namespace total_marbles_l197_197901

theorem total_marbles :
  let marbles_second_bowl := 600
  let marbles_first_bowl := (3/4) * marbles_second_bowl
  let total_marbles := marbles_first_bowl + marbles_second_bowl
  total_marbles = 1050 := by
  sorry -- proof skipped

end total_marbles_l197_197901


namespace LeRoy_should_pay_Bernardo_l197_197044

theorem LeRoy_should_pay_Bernardo 
    (initial_loan : ℕ := 100)
    (LeRoy_gas_expense : ℕ := 300)
    (LeRoy_food_expense : ℕ := 200)
    (Bernardo_accommodation_expense : ℕ := 500)
    (total_expense := LeRoy_gas_expense + LeRoy_food_expense + Bernardo_accommodation_expense)
    (shared_expense := total_expense / 2)
    (LeRoy_total_responsibility := shared_expense + initial_loan)
    (LeRoy_needs_to_pay := LeRoy_total_responsibility - (LeRoy_gas_expense + LeRoy_food_expense)) :
    LeRoy_needs_to_pay = 100 := 
by
    sorry

end LeRoy_should_pay_Bernardo_l197_197044


namespace club_members_neither_subject_l197_197824

theorem club_members_neither_subject (total members_cs members_bio members_both : ℕ)
  (h_total : total = 150)
  (h_cs : members_cs = 80)
  (h_bio : members_bio = 50)
  (h_both : members_both = 15) :
  total - ((members_cs - members_both) + (members_bio - members_both) + members_both) = 35 := by
  sorry

end club_members_neither_subject_l197_197824


namespace correct_factorization_l197_197447

theorem correct_factorization : 
  (¬ (6 * x^2 * y^3 = 2 * x^2 * 3 * y^3)) ∧ 
  (¬ (x^2 + 2 * x + 1 = x * (x^2 + 2) + 1)) ∧ 
  (¬ ((x + 2) * (x - 3) = x^2 - x - 6)) ∧ 
  (x^2 - 9 = (x - 3) * (x + 3)) :=
by 
  sorry

end correct_factorization_l197_197447


namespace trigonometric_identity_l197_197092

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end trigonometric_identity_l197_197092


namespace cookie_radius_l197_197418

theorem cookie_radius (x y : ℝ) (h : x^2 + y^2 + 2 * x - 4 * y = 4) : 
  ∃ r : ℝ, (x + 1)^2 + (y - 2)^2 = r^2 ∧ r = 3 := by
  sorry

end cookie_radius_l197_197418


namespace longest_sticks_triangle_shortest_sticks_not_triangle_l197_197761

-- Define the lengths of the six sticks in descending order
variables {a1 a2 a3 a4 a5 a6 : ℝ}

-- Assuming the conditions
axiom h1 : a1 ≥ a2
axiom h2 : a2 ≥ a3
axiom h3 : a3 ≥ a4
axiom h4 : a4 ≥ a5
axiom h5 : a5 ≥ a6
axiom h6 : a1 + a2 > a3

-- Proof problem 1: It is always possible to form a triangle from the three longest sticks.
theorem longest_sticks_triangle : a1 < a2 + a3 := by sorry

-- Assuming an additional condition for proof problem 2
axiom two_triangles_formed : ∃ b1 b2 b3 b4 b5 b6: ℝ, 
  ((b1 + b2 > b3 ∧ b1 + b3 > b2 ∧ b2 + b3 > b1) ∧
   (b4 + b5 > b6 ∧ b4 + b6 > b5 ∧ b5 + b6 > b4 ∧ 
    a1 = b1 ∧ a2 = b2 ∧ a3 = b3 ∧ a4 = b4 ∧ a5 = b5 ∧ a6 = b6))

-- Proof problem 2: It is not always possible to form a triangle from the three shortest sticks.
theorem shortest_sticks_not_triangle : ¬(a4 < a5 + a6 ∧ a5 < a4 + a6 ∧ a6 < a4 + a5) := by sorry

end longest_sticks_triangle_shortest_sticks_not_triangle_l197_197761


namespace eval_expression_correct_l197_197323

def eval_expression : ℤ :=
  -(-1) + abs (-1)

theorem eval_expression_correct : eval_expression = 2 :=
  by
    sorry

end eval_expression_correct_l197_197323


namespace cost_effective_bus_choice_l197_197623

theorem cost_effective_bus_choice (x y : ℕ) (h1 : y = x - 1) (h2 : 32 < 48 * x - 64 * y ∧ 48 * x - 64 * y < 64) : 
  64 * 300 < x * 2600 → True :=
by {
  sorry
}

end cost_effective_bus_choice_l197_197623


namespace orange_balls_count_l197_197377

theorem orange_balls_count (P_black : ℚ) (O : ℕ) (total_balls : ℕ) 
  (condition1 : total_balls = O + 7 + 6) 
  (condition2 : P_black = 7 / total_balls) 
  (condition3 : P_black = 0.38095238095238093) :
  O = 5 := 
by
  sorry

end orange_balls_count_l197_197377


namespace length_of_rod_l197_197371

theorem length_of_rod (w1 w2 l1 l2 : ℝ) (h_uniform : ∀ m n, m * w1 = n * w2) (h1 : w1 = 42.75) (h2 : l1 = 11.25) : 
  l2 = 6 := 
  by
  have wpm := w1 / l1
  have h3 : 22.8 / wpm = l2 := by sorry
  rw [h1, h2] at *
  simp at *
  sorry

end length_of_rod_l197_197371


namespace sequence_general_term_l197_197519

theorem sequence_general_term (a : ℕ → ℤ) (h₁ : a 1 = 2)
  (h₂ : ∀ n, 1 < n → a n = 2 * (n + a (n - 1))) :
  ∀ n, 1 ≤ n → a n = 2 ^ (n + 2) - 2 * n - 4 :=
by
  sorry

end sequence_general_term_l197_197519


namespace locus_of_intersection_l197_197493

-- Define the conditions
def line_e (m_e x y : ℝ) : Prop := y = m_e * (x - 1) + 1
def line_f (m_f x y : ℝ) : Prop := y = m_f * (x + 1) + 1
def slope_diff_cond (m_e m_f : ℝ) : Prop := (m_e - m_f = 2 ∨ m_f - m_e = 2)
def not_at_points (x y : ℝ) : Prop := (x, y) ≠ (1, 1) ∧ (x, y) ≠ (-1, 1)

-- Define the proof problem
theorem locus_of_intersection (x y m_e m_f : ℝ) :
  line_e m_e x y → line_f m_f x y → slope_diff_cond m_e m_f → not_at_points x y →
  (y = x^2 ∨ y = 2 - x^2) :=
by
  intros he hf h_diff h_not_at
  sorry

end locus_of_intersection_l197_197493


namespace integer_k_values_l197_197819

noncomputable def is_integer_solution (k x : ℤ) : Prop :=
  ((k - 2013) * x = 2015 - 2014 * x)

theorem integer_k_values (k : ℤ) (h : ∃ x : ℤ, is_integer_solution k x) :
  ∃ n : ℕ, n = 16 :=
by
  sorry

end integer_k_values_l197_197819


namespace find_other_number_l197_197384

theorem find_other_number (x y : ℤ) (h1 : 3 * x + 2 * y = 145) (h2 : x = 35 ∨ y = 35) : y = 20 :=
sorry

end find_other_number_l197_197384


namespace factorial_difference_l197_197038

-- Define factorial function for natural numbers
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- The theorem we want to prove
theorem factorial_difference : factorial 10 - factorial 9 = 3265920 :=
by
  rw [factorial, factorial]
  sorry

end factorial_difference_l197_197038


namespace ten_percent_of_x_l197_197927

theorem ten_percent_of_x
  (x : ℝ)
  (h : 3 - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = 27) :
  0.10 * x = 17.85 :=
by
  -- theorem proof goes here
  sorry

end ten_percent_of_x_l197_197927


namespace fraction_addition_equivalence_l197_197295

theorem fraction_addition_equivalence : 
  ∃ (n : ℚ), ((4 + n) / (7 + n) = 7 / 9) ∧ n = 13 / 2 :=
begin
  sorry
end

end fraction_addition_equivalence_l197_197295


namespace dow_jones_morning_value_l197_197730

theorem dow_jones_morning_value 
  (end_of_day_value : ℝ) 
  (percentage_fall : ℝ)
  (expected_morning_value : ℝ) 
  (h1 : end_of_day_value = 8722) 
  (h2 : percentage_fall = 0.02) 
  (h3 : expected_morning_value = 8900) :
  expected_morning_value = end_of_day_value / (1 - percentage_fall) :=
sorry

end dow_jones_morning_value_l197_197730


namespace power_function_odd_l197_197068

-- Define the conditions
def f : ℝ → ℝ := sorry
def condition1 (f : ℝ → ℝ) : Prop := f 1 = 3

-- Define the statement of the problem as a Lean theorem
theorem power_function_odd (f : ℝ → ℝ) (h : condition1 f) : ∀ x, f (-x) = -f x := sorry

end power_function_odd_l197_197068


namespace smallest_sum_of_digits_l197_197945

theorem smallest_sum_of_digits :
  ∃ (a b S : ℕ), 
    (100 ≤ a ∧ a < 1000) ∧ 
    (10 ≤ b ∧ b < 100) ∧ 
    (∃ (d1 d2 d3 d4 d5 : ℕ), 
      (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧ 
      (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧ 
      (d3 ≠ d4) ∧ (d3 ≠ d5) ∧ 
      (d4 ≠ d5) ∧ 
      S = a + b ∧ 100 ≤ S ∧ S < 1000 ∧ 
      (∃ (s : ℕ), 
        s = (S / 100) + ((S % 100) / 10) + (S % 10) ∧ 
        s = 3)) :=
sorry

end smallest_sum_of_digits_l197_197945


namespace volume_and_surface_area_implies_sum_of_edges_l197_197140

-- Define the problem conditions and prove the required statement
theorem volume_and_surface_area_implies_sum_of_edges :
  ∃ (a r : ℝ), 
    (a / r) * a * (a * r) = 216 ∧ 
    2 * ((a^2 / r) + a^2 * r + a^2) = 288 →
    4 * ((a / r) + a * r + a) = 96 :=
by
  sorry

end volume_and_surface_area_implies_sum_of_edges_l197_197140


namespace median_in_interval_65_69_l197_197785

-- Definitions for student counts in each interval
def count_50_54 := 5
def count_55_59 := 7
def count_60_64 := 22
def count_65_69 := 19
def count_70_74 := 15
def count_75_79 := 10
def count_80_84 := 18
def count_85_89 := 5

-- Total number of students
def total_students := 101

-- Calculation of the position of the median
def median_position := (total_students + 1) / 2

-- Cumulative counts
def cumulative_up_to_59 := count_50_54 + count_55_59
def cumulative_up_to_64 := cumulative_up_to_59 + count_60_64
def cumulative_up_to_69 := cumulative_up_to_64 + count_65_69

-- Proof statement
theorem median_in_interval_65_69 :
  34 < median_position ∧ median_position ≤ cumulative_up_to_69 :=
by
  sorry

end median_in_interval_65_69_l197_197785


namespace tangent_line_at_point_l197_197994

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then (Real.exp (-(x - 1)) - x) else (Real.exp (x - 1) + x)

theorem tangent_line_at_point (f_even : ∀ x : ℝ, f x = f (-x)) :
    ∀ (x y : ℝ), x = 1 → y = 2 → (∃ m b : ℝ, y = m * x + b ∧ m = 2 ∧ b = 0) := by
  sorry

end tangent_line_at_point_l197_197994


namespace trigonometric_identity_l197_197082

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end trigonometric_identity_l197_197082


namespace problem_find_f_l197_197053

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_find_f {k : ℝ} :
  (∀ x : ℝ, x * (f (x + 1) - f x) = f x) →
  (∀ x y : ℝ, |f x - f y| ≤ |x - y|) →
  (∀ x : ℝ, 0 < x → f x = k * x) :=
by
  intro h1 h2
  apply sorry

end problem_find_f_l197_197053


namespace true_proposition_l197_197375

-- Define propositions p and q
variable (p q : Prop)

-- Assume p is true and q is false
axiom h1 : p
axiom h2 : ¬q

-- Prove that p ∧ ¬q is true
theorem true_proposition (p q : Prop) (h1 : p) (h2 : ¬q) : p ∧ ¬q :=
by
  sorry

end true_proposition_l197_197375


namespace new_circle_radius_shaded_region_l197_197287

theorem new_circle_radius_shaded_region {r1 r2 : ℝ} 
    (h1 : r1 = 35) 
    (h2 : r2 = 24) : 
    ∃ r : ℝ, π * r^2 = π * (r1^2 - r2^2) ∧ r = Real.sqrt 649 := 
by
  sorry

end new_circle_radius_shaded_region_l197_197287


namespace flower_team_participation_l197_197597

-- Definitions based on the conditions in the problem
def num_rows : ℕ := 60
def first_row_people : ℕ := 40
def people_increment : ℕ := 1

-- Statement to be proved in Lean
theorem flower_team_participation (x : ℕ) (hx : 1 ≤ x ∧ x ≤ num_rows) : 
  ∃ y : ℕ, y = first_row_people - people_increment + x :=
by
  -- Placeholder for the proof
  sorry

end flower_team_participation_l197_197597


namespace scientific_notation_347000_l197_197583

theorem scientific_notation_347000 :
  347000 = 3.47 * 10^5 :=
by 
  -- Proof will go here
  sorry

end scientific_notation_347000_l197_197583


namespace parallelogram_area_l197_197888

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) (hb : b = 15) (hh : h = 2 * b) (hA : A = b * h) : A = 450 := 
by
  rw [hb, hh] at hA
  rw [hA]
  sorry

end parallelogram_area_l197_197888


namespace sufficient_not_necessary_condition_l197_197608

theorem sufficient_not_necessary_condition (x : ℝ) : (x ≥ 3 → (x - 2) ≥ 0) ∧ ((x - 2) ≥ 0 → x ≥ 3) = false :=
by
  sorry

end sufficient_not_necessary_condition_l197_197608


namespace baker_cakes_l197_197018

theorem baker_cakes (P x : ℝ) (h1 : P * x = 320) (h2 : 0.80 * P * (x + 2) = 320) : x = 8 :=
by
  sorry

end baker_cakes_l197_197018


namespace michelle_phone_bill_l197_197930

def base_cost : ℝ := 20
def text_cost_per_message : ℝ := 0.05
def minute_cost_over_20h : ℝ := 0.20
def messages_sent : ℝ := 150
def hours_talked : ℝ := 22
def allowed_hours : ℝ := 20

theorem michelle_phone_bill :
  base_cost + (messages_sent * text_cost_per_message) +
  ((hours_talked - allowed_hours) * 60 * minute_cost_over_20h) = 51.50 := by
  sorry

end michelle_phone_bill_l197_197930


namespace num_positive_integers_condition_l197_197662

theorem num_positive_integers_condition : 
  ∃! n : ℤ, 0 < n ∧ n < 50 ∧ (n + 2) % (50 - n) = 0 :=
by
  sorry

end num_positive_integers_condition_l197_197662


namespace sum_of_leading_digits_l197_197558

def leading_digit (n : ℕ) (x : ℝ) : ℕ := sorry

def M := 10^500 - 1

def g (r : ℕ) : ℕ := leading_digit r (M^(1 / r))

theorem sum_of_leading_digits :
  g 3 + g 4 + g 5 + g 7 + g 8 = 10 := sorry

end sum_of_leading_digits_l197_197558


namespace trig_expression_simplify_l197_197095

theorem trig_expression_simplify (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trig_expression_simplify_l197_197095


namespace butterflies_left_l197_197955

theorem butterflies_left (initial_butterflies : ℕ) (one_third_left : ℕ)
  (h1 : initial_butterflies = 9) (h2 : one_third_left = initial_butterflies / 3) :
  initial_butterflies - one_third_left = 6 :=
by
  sorry

end butterflies_left_l197_197955


namespace polynomial_sum_of_squares_l197_197849

theorem polynomial_sum_of_squares (P : Polynomial ℝ) (hP : ∀ x : ℝ, 0 < P.eval x) :
  ∃ (U V : Polynomial ℝ), P = U^2 + V^2 := 
by
  sorry

end polynomial_sum_of_squares_l197_197849


namespace circle_condition_l197_197420

-- Define the given equation
def equation (m x y : ℝ) : Prop := x^2 + y^2 + 4 * m * x - 2 * y + 5 * m = 0

-- Define the condition for the equation to represent a circle
def represents_circle (m x y : ℝ) : Prop :=
  (x + 2 * m)^2 + (y - 1)^2 = 4 * m^2 - 5 * m + 1 ∧ 4 * m^2 - 5 * m + 1 > 0

-- The main theorem to be proven
theorem circle_condition (m : ℝ) : represents_circle m x y → (m < 1/4 ∨ m > 1) := 
sorry

end circle_condition_l197_197420


namespace floor_e_is_two_l197_197654

noncomputable def e : ℝ := Real.exp 1

theorem floor_e_is_two : ⌊e⌋ = 2 := by
  sorry

end floor_e_is_two_l197_197654


namespace correct_computation_l197_197032

theorem correct_computation (x : ℕ) (h : x - 20 = 52) : x / 4 = 18 :=
  sorry

end correct_computation_l197_197032


namespace grandmother_mistaken_l197_197406

-- Definitions of the given conditions:
variables (N : ℕ) (x n : ℕ)
variable (initial_split : N % 4 = 0)

-- Conditions
axiom cows_survived : 4 * (N / 4) / 5 = N / 5
axiom horses_pigs : x = N / 4 - N / 5
axiom rabbit_ratio : (N / 4 - n) = 5 / 14 * (N / 5 + N / 4 + N / 4 - n)

-- Goal: Prove the grandmother is mistaken, i.e., some species avoided casualties
theorem grandmother_mistaken : n = 0 :=
sorry

end grandmother_mistaken_l197_197406


namespace y_intercept_of_line_b_l197_197863

noncomputable def line_b_y_intercept (b : Type) [HasElem ℝ b] : Prop :=
  ∃ (m : ℝ) (c : ℝ), (m = -3) ∧ (c = 7) ∧ ∀ (x : ℝ) (y : ℝ), (x, y) ∈ b → y = -3 * x + c

theorem y_intercept_of_line_b (b : Type) [HasElem (ℝ × ℝ) b] :
  (∃ (p : ℝ × ℝ), p = (3, -2) ∧ ∃ (q : line_b_y_intercept b), q) →
  ∃ (c : ℝ), c = 7 :=
by
  intro h
  sorry

end y_intercept_of_line_b_l197_197863


namespace radio_price_rank_l197_197321

theorem radio_price_rank (total_items : ℕ) (radio_position_highest : ℕ) (radio_position_lowest : ℕ) 
  (h1 : total_items = 40) (h2 : radio_position_highest = 17) : 
  radio_position_lowest = total_items - radio_position_highest + 1 :=
by
  sorry

end radio_price_rank_l197_197321


namespace logarithmic_inequality_l197_197336

noncomputable def log_a_b (a b : ℝ) := Real.log b / Real.log a

theorem logarithmic_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  log_a_b a b + log_a_b b c + log_a_b a c ≥ 3 :=
by
  sorry

end logarithmic_inequality_l197_197336


namespace last_digit_of_2_pow_2010_l197_197722

-- Define the pattern of last digits of powers of 2
def last_digit_of_power_of_2 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 0 -- This case is redundant as n % 4 ∈ {0, 1, 2, 3}

-- Main theorem stating the problem's assertion
theorem last_digit_of_2_pow_2010 : last_digit_of_power_of_2 2010 = 4 :=
by
  -- The proof is omitted
  sorry

end last_digit_of_2_pow_2010_l197_197722


namespace change_from_fifteen_dollars_l197_197953

theorem change_from_fifteen_dollars : 
  ∀ (cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid : ℕ),
  cost_eggs = 3 →
  cost_pancakes = 2 →
  cost_mug_cocoa = 2 →
  num_mugs = 2 →
  tax = 1 →
  additional_pancakes = 2 →
  additional_mug = 2 →
  paid = 15 →
  paid - (cost_eggs + cost_pancakes + (num_mugs * cost_mug_cocoa) + tax + additional_pancakes + additional_mug) = 1 :=
by
  intros cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid
  sorry

end change_from_fifteen_dollars_l197_197953


namespace find_y_l197_197419

theorem find_y (y : ℤ) (h : (15 + 26 + y) / 3 = 23) : y = 28 :=
by sorry

end find_y_l197_197419


namespace total_cost_l197_197557

def num_professionals := 2
def hours_per_professional_per_day := 6
def days_worked := 7
def hourly_rate := 15

theorem total_cost : 
  (num_professionals * hours_per_professional_per_day * days_worked * hourly_rate) = 1260 := by
  sorry

end total_cost_l197_197557


namespace distance_on_third_day_is_36_difference_between_longest_and_shortest_is_57_average_daily_distance_is_50_l197_197450

-- Definitions for each day's recorded distance deviation
def day_1_distance := -8
def day_2_distance := -11
def day_3_distance := -14
def day_4_distance := 0
def day_5_distance := 8
def day_6_distance := 41
def day_7_distance := -16

-- Parameters and conditions
def actual_distance (recorded: Int) : Int := 50 + recorded

noncomputable def distance_3rd_day : Int := actual_distance day_3_distance
noncomputable def longest_distance : Int :=
    max (max (max (day_1_distance) (day_2_distance)) (max (day_3_distance) (day_4_distance)))
        (max (max (day_5_distance) (day_6_distance)) (day_7_distance))
noncomputable def shortest_distance : Int :=
    min (min (min (day_1_distance) (day_2_distance)) (min (day_3_distance) (day_4_distance)))
        (min (min (day_5_distance) (day_6_distance)) (day_7_distance))
noncomputable def average_distance : Int :=
    50 + (day_1_distance + day_2_distance + day_3_distance + day_4_distance +
          day_5_distance + day_6_distance + day_7_distance) / 7

-- Theorems to prove each part of the problem
theorem distance_on_third_day_is_36 : distance_3rd_day = 36 := by
  sorry

theorem difference_between_longest_and_shortest_is_57 : 
  (actual_distance longest_distance - actual_distance shortest_distance) = 57 := by
  sorry

theorem average_daily_distance_is_50 : average_distance = 50 := by
  sorry

end distance_on_third_day_is_36_difference_between_longest_and_shortest_is_57_average_daily_distance_is_50_l197_197450


namespace find_digit_l197_197374

theorem find_digit (p q r : ℕ) (hq : p ≠ q) (hr : p ≠ r) (hq' : q ≠ r) 
    (hp_pos : 0 < p ∧ p < 10)
    (hq_pos : 0 < q ∧ q < 10)
    (hr_pos : 0 < r ∧ r < 10)
    (h1 : 10 * p + q = 17)
    (h2 : 10 * p + r = 13)
    (h3 : p + q + r = 11) : 
    q = 7 :=
sorry

end find_digit_l197_197374


namespace average_transformation_l197_197817

theorem average_transformation (a b c : ℝ) (h : (a + b + c) / 3 = 12) : ((2 * a + 1) + (2 * b + 2) + (2 * c + 3) + 2) / 4 = 20 :=
by
  sorry

end average_transformation_l197_197817


namespace strawberries_weight_before_l197_197867

variables (M D E B : ℝ)

noncomputable def total_weight_before (M D E : ℝ) := M + D - E

theorem strawberries_weight_before :
  ∀ (M D E : ℝ), M = 36 ∧ D = 16 ∧ E = 30 → total_weight_before M D E = 22 :=
by
  intros M D E h
  simp [total_weight_before, h]
  sorry

end strawberries_weight_before_l197_197867


namespace initial_amount_of_liquid_A_l197_197150

-- Definitions for liquids A and B and their ratios in the initial and modified mixtures
def initial_ratio_A_over_B : ℚ := 4 / 1
def final_ratio_A_over_B_after_replacement : ℚ := 2 / 3
def mixture_replacement_volume : ℚ := 30

-- Proof of the initial amount of liquid A
theorem initial_amount_of_liquid_A (x : ℚ) (A B : ℚ) (initial_mixture : ℚ) :
  (initial_ratio_A_over_B = 4 / 1) →
  (final_ratio_A_over_B_after_replacement = 2 / 3) →
  (mixture_replacement_volume = 30) →
  (A + B = 5 * x) →
  (A / B = 4 / 1) →
  ((A - 24) / (B - 6 + 30) = 2 / 3) →
  A = 48 :=
by {
  sorry
}

end initial_amount_of_liquid_A_l197_197150


namespace prove_fraction_identity_l197_197398

theorem prove_fraction_identity (x y : ℂ) (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) : 
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 41 / 20 := 
by 
  sorry

end prove_fraction_identity_l197_197398


namespace max_tunnel_construction_mileage_find_value_of_a_l197_197782

-- Define variables and conditions
variables (x a : ℝ)

def total_mileage := 56
def ordinary_road_mileage := 32
def elevated_road_mileage_in_q1 := total_mileage - ordinary_road_mileage - x
def minimum_elevated_road_mileage := 7 * x

-- Problem 1: Maximum tunnel construction mileage
theorem max_tunnel_construction_mileage :
  elevated_road_mileage_in_q1 >= minimum_elevated_road_mileage → x <= 3 :=
by intros h; sorry

-- Define cost variables and conditions for Q2
def ordinary_road_cost_per_km_q1 := 1
def elevated_road_cost_per_km_q1 := 2
def tunnel_road_cost_per_km_q1 := 4
def ordinary_road_mileage_q2 := ordinary_road_mileage - 9 * a
def elevated_road_mileage_q2 := elevated_road_mileage_in_q1 - 2 * a
def tunnel_road_mileage_q2 := x + a
def elevated_road_cost_per_km_q2 := elevated_road_cost_per_km_q1 + 0.5 * a

-- Problem 2: Value of a
theorem find_value_of_a (h1: elevated_road_mileage_in_q1 >= minimum_elevated_road_mileage) :
  (ordinary_road_cost_per_km_q1 * ordinary_road_mileage +
   elevated_road_cost_per_km_q1 * elevated_road_mileage_in_q1 +
   tunnel_road_cost_per_km_q1 * x = 
   (ordinary_road_cost_per_km_q1 * ordinary_road_mileage_q2 +
   elevated_road_cost_per_km_q2 * elevated_road_mileage_q2 +
   tunnel_road_cost_per_km_q1 * tunnel_road_mileage_q2)) → 
   a = 3 / 2 :=
by intros h2; sorry

end max_tunnel_construction_mileage_find_value_of_a_l197_197782


namespace ivy_has_20_collectors_dolls_l197_197642

theorem ivy_has_20_collectors_dolls
  (D : ℕ) (I : ℕ) (C : ℕ)
  (h1 : D = 60)
  (h2 : D = 2 * I)
  (h3 : C = 2 * I / 3) 
  : C = 20 :=
by sorry

end ivy_has_20_collectors_dolls_l197_197642


namespace ratios_of_square_areas_l197_197416

variable (x : ℝ)

def square_area (side_length : ℝ) : ℝ := side_length^2

theorem ratios_of_square_areas (hA : square_area x = x^2)
                               (hB : square_area (5 * x) = 25 * x^2)
                               (hC : square_area (2 * x) = 4 * x^2) :
  (square_area x / square_area (5 * x) = 1 / 25 ∧
   square_area (2 * x) / square_area (5 * x) = 4 / 25) := 
by {
  sorry
}

end ratios_of_square_areas_l197_197416


namespace scientific_notation_representation_l197_197299

theorem scientific_notation_representation :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_representation_l197_197299


namespace invitations_sent_out_l197_197216

-- Define the conditions
def RSVPed (I : ℝ) : ℝ := 0.9 * I
def Showed_up (I : ℝ) : ℝ := 0.8 * RSVPed I
def No_gift : ℝ := 10
def Thank_you_cards : ℝ := 134

-- Prove the number of invitations
theorem invitations_sent_out : ∃ I : ℝ, Showed_up I - No_gift = Thank_you_cards ∧ I = 200 :=
by
  sorry

end invitations_sent_out_l197_197216


namespace subset_implies_a_ge_2_l197_197852

theorem subset_implies_a_ge_2 (a : ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 2 → x ≤ a) → a ≥ 2 :=
by sorry

end subset_implies_a_ge_2_l197_197852


namespace sharpener_difference_l197_197936

/-- A hand-crank pencil sharpener can sharpen one pencil every 45 seconds.
An electric pencil sharpener can sharpen one pencil every 20 seconds.
The total available time is 360 seconds (i.e., 6 minutes).
Prove that the difference in the number of pencils sharpened 
by the electric sharpener and the hand-crank sharpener in 360 seconds is 10 pencils. -/
theorem sharpener_difference (time : ℕ) (hand_crank_rate : ℕ) (electric_rate : ℕ) 
(h_time : time = 360) (h_hand_crank : hand_crank_rate = 45) (h_electric : electric_rate = 20) :
  (time / electric_rate) - (time / hand_crank_rate) = 10 := by
  sorry

end sharpener_difference_l197_197936


namespace find_x_l197_197510

theorem find_x :
  let a := 5^3
  let b := 6^2
  a - 7 = b + 82 := 
by
  sorry

end find_x_l197_197510


namespace find_a_l197_197098

theorem find_a (a : ℚ) : (∃ b : ℚ, 4 * (x : ℚ)^2 + 14 * x + a = (2 * x + b)^2) → a = 49 / 4 :=
by
  sorry

end find_a_l197_197098


namespace ram_work_rate_l197_197606

-- Definitions as given in the problem
variable (W : ℕ) -- Total work can be represented by some natural number W
variable (R M : ℕ) -- Raja's work rate and Ram's work rate, respectively

-- Given conditions
variable (combined_work_rate : R + M = W / 4)
variable (raja_work_rate : R = W / 12)

-- Theorem to be proven
theorem ram_work_rate (combined_work_rate : R + M = W / 4) (raja_work_rate : R = W / 12) : M = W / 6 := 
  sorry

end ram_work_rate_l197_197606


namespace line_within_plane_l197_197152

variable (a : Set Point) (α : Set Point)

theorem line_within_plane : a ⊆ α :=
by
  sorry

end line_within_plane_l197_197152


namespace sum_of_100th_group_is_1010100_l197_197806

theorem sum_of_100th_group_is_1010100 : (100 + 100^2 + 100^3) = 1010100 :=
by
  sorry

end sum_of_100th_group_is_1010100_l197_197806


namespace complementary_angle_decrease_l197_197431

theorem complementary_angle_decrease (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3 / 7) : 
  (∃ new_α : ℝ, new_α = α * 1.2) →
  ∃ new_β : ℝ, new_β = (1 - 0.0857) * β :=
by
  intro h3
  cases h3 with new_α h_newα
  use 90 - new_α
  sorry

end complementary_angle_decrease_l197_197431


namespace dominoes_per_player_l197_197835

-- Define the conditions
def total_dominoes : ℕ := 28
def number_of_players : ℕ := 4

-- The theorem
theorem dominoes_per_player : total_dominoes / number_of_players = 7 :=
by sorry

end dominoes_per_player_l197_197835


namespace max_x_plus_2y_l197_197523

theorem max_x_plus_2y {x y : ℝ} (h : x^2 - x * y + y^2 = 1) :
  x + 2 * y ≤ (2 * Real.sqrt 21) / 3 :=
sorry

end max_x_plus_2y_l197_197523


namespace volleyball_team_geography_l197_197473

theorem volleyball_team_geography (total_players history_players both_subjects : ℕ) 
  (H1 : total_players = 15) 
  (H2 : history_players = 9) 
  (H3 : both_subjects = 4) : 
  ∃ (geography_players : ℕ), geography_players = 10 :=
by
  -- Definitions / Calculations
  -- Using conditions to derive the number of geography players
  let only_geography_players : ℕ := total_players - history_players
  let geography_players : ℕ := only_geography_players + both_subjects

  -- Prove the statement
  use geography_players
  sorry

end volleyball_team_geography_l197_197473


namespace monotonicity_of_f_range_of_a_if_no_zeros_l197_197395

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l197_197395


namespace inscribed_circle_radius_l197_197913

variable (AB AC BC : ℝ) (r : ℝ)

theorem inscribed_circle_radius 
  (h1 : AB = 9) 
  (h2 : AC = 9) 
  (h3 : BC = 8) : r = (4 * Real.sqrt 65) / 13 := 
sorry

end inscribed_circle_radius_l197_197913


namespace monotonicity_and_range_l197_197385

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l197_197385


namespace upward_shift_of_parabola_l197_197585

variable (k : ℝ) -- Define k as a real number representing the vertical shift

def original_function (x : ℝ) : ℝ := -x^2 -- Define the original function

def shifted_function (x : ℝ) : ℝ := original_function x + 2 -- Define the shifted function by 2 units upwards

theorem upward_shift_of_parabola (x : ℝ) : shifted_function x = -x^2 + k :=
by
  sorry

end upward_shift_of_parabola_l197_197585


namespace find_t_of_quadratic_root_l197_197793

variable (a t : ℝ)

def quadratic_root_condition (a : ℝ) : Prop :=
  ∃ t : ℝ, Complex.ofReal a + Complex.I * 3 = Complex.ofReal a - Complex.I * 3 ∧
           (Complex.ofReal a + Complex.I * 3).re * (Complex.ofReal a - Complex.I * 3).re = t

theorem find_t_of_quadratic_root (h : quadratic_root_condition a) : t = 13 :=
sorry

end find_t_of_quadratic_root_l197_197793


namespace complementary_angle_decrease_l197_197432

theorem complementary_angle_decrease (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3 / 7) : 
  (∃ new_α : ℝ, new_α = α * 1.2) →
  ∃ new_β : ℝ, new_β = (1 - 0.0857) * β :=
by
  intro h3
  cases h3 with new_α h_newα
  use 90 - new_α
  sorry

end complementary_angle_decrease_l197_197432


namespace find_abc_l197_197135

theorem find_abc (a b c : ℚ) 
  (h1 : a + b + c = 24)
  (h2 : a + 2 * b = 2 * c)
  (h3 : a = b / 2) : 
  a = 16 / 3 ∧ b = 32 / 3 ∧ c = 8 := 
by 
  sorry

end find_abc_l197_197135


namespace students_in_both_clubs_l197_197472

theorem students_in_both_clubs
  (T R B total_club_students : ℕ)
  (hT : T = 85) (hR : R = 120)
  (hTotal : T + R - B = total_club_students)
  (hTotalVal : total_club_students = 180) :
  B = 25 :=
by
  -- Placeholder for proof
  sorry

end students_in_both_clubs_l197_197472


namespace bingo_first_column_possibilities_l197_197702

theorem bingo_first_column_possibilities :
  (∏ i in (Finset.range 5), (15 - i)) = 360360 :=
by
  sorry

end bingo_first_column_possibilities_l197_197702


namespace y_intercept_of_line_b_l197_197859

-- Define the conditions
def line_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def point_on_line (m b x y : ℝ) : Prop := y = m * x + b

-- Given conditions
variables (m b : ℝ)
variable (x₁ := 3)
variable (y₁ := -2)
axiom parallel_condition : line_parallel m (-3)
axiom point_condition : point_on_line m b x₁ y₁

-- Prove that the y-intercept b equals 7
theorem y_intercept_of_line_b : b = 7 :=
sorry

end y_intercept_of_line_b_l197_197859


namespace rhombus_area_l197_197341

theorem rhombus_area (s d1 d2 : ℝ)
  (h1 : s = Real.sqrt 113)
  (h2 : abs (d1 - d2) = 8)
  (h3 : s^2 = (d1 / 2)^2 + (d2 / 2)^2) :
  (d1 * d2) / 2 = 194 := by
  sorry

end rhombus_area_l197_197341


namespace chairs_in_fifth_row_l197_197707

theorem chairs_in_fifth_row : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 14 ∧ 
    a 2 = 23 ∧ 
    a 3 = 32 ∧ 
    a 4 = 41 ∧ 
    a 6 = 59 ∧ 
    (∀ n, a (n + 1) = a n + 9) → 
  a 5 = 50 :=
by
  sorry

end chairs_in_fifth_row_l197_197707


namespace minimum_value_proof_l197_197191

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  (1 / (x + 1)) + (1 / y)

theorem minimum_value_proof (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) :
  minimum_value x y = (3 + 2 * real.sqrt 2) / 2 :=
sorry

end minimum_value_proof_l197_197191


namespace fraction_to_decimal_l197_197332

theorem fraction_to_decimal :
  (58 / 200 : ℝ) = 1.16 := by
  sorry

end fraction_to_decimal_l197_197332


namespace sum_of_angles_l197_197178

theorem sum_of_angles (a b : ℝ) (ha : a = 45) (hb : b = 225) : a + b = 270 :=
by
  rw [ha, hb]
  norm_num -- Lean's built-in tactic to normalize numerical expressions

end sum_of_angles_l197_197178


namespace proof_problem_l197_197331

-- Conditions
def op1 := (15 + 3) / (8 - 2) = 3
def op2 := (9 + 4) / (14 - 7)

-- Statement
theorem proof_problem : op1 → op2 = 13 / 7 :=
by 
  intro h
  unfold op2
  sorry

end proof_problem_l197_197331


namespace butterflies_left_correct_l197_197958

-- Define the total number of butterflies and the fraction that flies away
def butterflies_total : ℕ := 9
def fraction_fly_away : ℚ := 1 / 3

-- Define the number of butterflies left in the garden
def butterflies_left (t : ℕ) (f : ℚ) : ℚ := t - (t : ℚ) * f

-- State the theorem
theorem butterflies_left_correct : butterflies_left butterflies_total fraction_fly_away = 6 := by
  sorry

end butterflies_left_correct_l197_197958


namespace friend_reading_time_l197_197404

-- Define the conditions
def my_reading_time : ℝ := 1.5 * 60 -- 1.5 hours converted to minutes
def friend_speed_multiplier : ℝ := 5 -- Friend reads 5 times faster than I do
def distraction_time : ℝ := 15 -- Friend is distracted for 15 minutes

-- Define the time taken for my friend to read the book accounting for distraction
theorem friend_reading_time :
  (my_reading_time / friend_speed_multiplier) + distraction_time = 33 := by
  sorry

end friend_reading_time_l197_197404


namespace geometric_arithmetic_sum_l197_197193

theorem geometric_arithmetic_sum {a : Nat → ℝ} {b : Nat → ℝ} 
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d)
  (h_condition : a 3 * a 11 = 4 * a 7)
  (h_equal : a 7 = b 7) :
  b 5 + b 9 = 8 :=
sorry

end geometric_arithmetic_sum_l197_197193


namespace david_twice_as_old_in_Y_years_l197_197329

variable (R D Y : ℕ)

-- Conditions
def rosy_current_age := R = 8
def david_is_older := D = R + 12
def twice_as_old_in_Y_years := D + Y = 2 * (R + Y)

-- Proof statement
theorem david_twice_as_old_in_Y_years
  (h1 : rosy_current_age R)
  (h2 : david_is_older R D)
  (h3 : twice_as_old_in_Y_years R D Y) :
  Y = 4 := sorry

end david_twice_as_old_in_Y_years_l197_197329


namespace school_spent_on_grass_seeds_bottle_capacity_insufficient_l197_197303

-- Problem 1: Cost Calculation
theorem school_spent_on_grass_seeds (kg_seeds : ℝ) (cost_per_kg : ℝ) (total_cost : ℝ) 
  (h1 : kg_seeds = 3.3) (h2 : cost_per_kg = 9.48) :
  total_cost = 31.284 :=
  by
    sorry

-- Problem 2: Bottle Capacity
theorem bottle_capacity_insufficient (total_seeds : ℝ) (max_capacity_per_bottle : ℝ) (num_bottles : ℕ)
  (h1 : total_seeds = 3.3) (h2 : max_capacity_per_bottle = 0.35) (h3 : num_bottles = 9) :
  3.3 > 0.35 * 9 :=
  by
    sorry

end school_spent_on_grass_seeds_bottle_capacity_insufficient_l197_197303


namespace range_of_x_l197_197697

theorem range_of_x (x : ℝ) :
  (∀ y : ℝ, 0 < y → y^2 + (2*x - 5)*y - x^2 * (Real.log x - Real.log y) ≤ 0) ↔ x = 5 / 2 :=
by 
  sorry

end range_of_x_l197_197697


namespace circle_radius_l197_197932

theorem circle_radius (M N r : ℝ) (h1 : M = Real.pi * r^2) (h2 : N = 2 * Real.pi * r) (h3 : M / N = 25) : r = 50 :=
by
  sorry

end circle_radius_l197_197932


namespace volume_and_surface_area_of_convex_body_l197_197737

noncomputable def volume_of_convex_body (a b c : ℝ) : ℝ := 
  (a^2 + b^2 + c^2)^3 / (6 * a * b * c)

noncomputable def surface_area_of_convex_body (a b c : ℝ) : ℝ :=
  (a^2 + b^2 + c^2)^(5/2) / (a * b * c)

theorem volume_and_surface_area_of_convex_body (a b c d : ℝ)
  (h : d^2 = a^2 + b^2 + c^2) :
  volume_of_convex_body a b c = (a^2 + b^2 + c^2)^3 / (6 * a * b * c) ∧
  surface_area_of_convex_body a b c = (a^2 + b^2 + c^2)^(5/2) / (a * b * c) :=
by
  sorry

end volume_and_surface_area_of_convex_body_l197_197737


namespace geometric_seq_a7_l197_197706

-- Definitions for the geometric sequence and conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ}
axiom a1 : a 1 = 2
axiom a3 : a 3 = 4
axiom geom_seq : geometric_sequence a

-- Statement to prove
theorem geometric_seq_a7 : a 7 = 16 :=
by
  -- proof will be filled in here
  sorry

end geometric_seq_a7_l197_197706


namespace average_first_14_even_numbers_l197_197011

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun x => 2 * (x + 1))

theorem average_first_14_even_numbers :
  let even_nums := first_n_even_numbers 14
  (even_nums.sum / even_nums.length = 15) :=
by
  sorry

end average_first_14_even_numbers_l197_197011


namespace kilometers_to_meters_kilograms_to_grams_l197_197928

def km_to_meters (km: ℕ) : ℕ := km * 1000
def kg_to_grams (kg: ℕ) : ℕ := kg * 1000

theorem kilometers_to_meters (h: 3 = 3): km_to_meters 3 = 3000 := by {
 sorry
}

theorem kilograms_to_grams (h: 4 = 4): kg_to_grams 4 = 4000 := by {
 sorry
}

end kilometers_to_meters_kilograms_to_grams_l197_197928


namespace cost_of_fencing_per_meter_in_cents_l197_197739

-- Definitions for the conditions
def ratio_length_width : ℕ := 3
def ratio_width_length : ℕ := 2
def total_area : ℕ := 3750
def total_fencing_cost : ℕ := 175

-- Main theorem statement with proof omitted
theorem cost_of_fencing_per_meter_in_cents :
  (ratio_length_width = 3) →
  (ratio_width_length = 2) →
  (total_area = 3750) →
  (total_fencing_cost = 175) →
  ∃ (cost_per_meter_in_cents : ℕ), cost_per_meter_in_cents = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_of_fencing_per_meter_in_cents_l197_197739


namespace trigonometric_identity_l197_197089

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end trigonometric_identity_l197_197089


namespace power_of_i_2016_l197_197302
-- Importing necessary libraries to handle complex numbers

theorem power_of_i_2016 (i : ℂ) (h1 : i^2 = -1) (h2 : i^4 = 1) : 
  (i^2016 = 1) :=
sorry

end power_of_i_2016_l197_197302


namespace sum_of_digits_of_d_l197_197645

theorem sum_of_digits_of_d (d : ℕ) (h₁ : ∃ d_ca : ℕ, d_ca = (8 * d) / 5) (h₂ : d_ca - 75 = d) :
  (1 + 2 + 5 = 8) :=
by
  sorry

end sum_of_digits_of_d_l197_197645


namespace fraction_simplification_l197_197787

-- We define the given fractions
def a := 3 / 7
def b := 2 / 9
def c := 5 / 12
def d := 1 / 4

-- We state the main theorem
theorem fraction_simplification : (a - b) / (c + d) = 13 / 42 := by
  -- Skipping proof for the equivalence problem
  sorry

end fraction_simplification_l197_197787


namespace system_has_three_solutions_l197_197971

theorem system_has_three_solutions (a : ℝ) :
  (a = 4 ∨ a = 64 ∨ a = 51 + 10 * Real.sqrt 2) ↔
  ∃ (x y : ℝ), 
    (x = abs (y - Real.sqrt a) + Real.sqrt a - 4 
    ∧ (abs x - 6)^2 + (abs y - 8)^2 = 100) 
        ∧ (∃! x1 y1 : ℝ, (x1 = abs (y1 - Real.sqrt a) + Real.sqrt a - 4 
        ∧ (abs x1 - 6)^2 + (abs y1 - 8)^2 = 100)) :=
by
  sorry

end system_has_three_solutions_l197_197971


namespace find_a1_l197_197671

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = a 0 + n * d

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

theorem find_a1 (d : ℝ) (h1 : a 13 = 13) (h2 : S 13 = 13) : a 0 = -11 :=
by
  sorry

end find_a1_l197_197671


namespace Jeanine_more_pencils_than_Clare_l197_197555

variables (Jeanine_pencils : ℕ) (Clare_pencils : ℕ)

def Jeanine_initial_pencils := 18
def Clare_initial_pencils := Jeanine_initial_pencils / 2
def Jeanine_pencils_given_to_Abby := Jeanine_initial_pencils / 3
def Jeanine_remaining_pencils := Jeanine_initial_pencils - Jeanine_pencils_given_to_Abby

theorem Jeanine_more_pencils_than_Clare :
  Jeanine_remaining_pencils - Clare_initial_pencils = 3 :=
by
  -- This is just the statement, the proof is not provided as instructed.
  sorry

end Jeanine_more_pencils_than_Clare_l197_197555


namespace wire_cut_problem_l197_197753

variable (x : ℝ)

theorem wire_cut_problem 
  (h₁ : x + (5 / 2) * x = 49) : x = 14 :=
by
  sorry

end wire_cut_problem_l197_197753


namespace ones_digit_of_sum_is_0_l197_197444

-- Define the integer n
def n : ℕ := 2012

-- Define the ones digit function
def ones_digit (x : ℕ) : ℕ := x % 10

-- Define the power function mod 10
def power_mod_10 (d a : ℕ) : ℕ := (d^a) % 10

-- Define the sequence sum for ones digits
def seq_sum_mod_10 (m : ℕ) : ℕ :=
  Finset.sum (Finset.range m) (λ k => power_mod_10 (k+1) n)

-- Define the final sum mod 10 considering the repeating cycle and sum
def total_ones_digit_sum (a b : ℕ) : ℕ :=
  let cycle_sum := Finset.sum (Finset.range 10) (λ k => power_mod_10 (k+1) n)
  let s := cycle_sum * (a / 10) + Finset.sum (Finset.range b) (λ k => power_mod_10 (k+1) n)
  s % 10

-- Prove that the ones digit of the sum is 0
theorem ones_digit_of_sum_is_0 : total_ones_digit_sum n (n % 10) = 0 :=
sorry

end ones_digit_of_sum_is_0_l197_197444


namespace tan_of_trig_eq_l197_197669

theorem tan_of_trig_eq (x : Real) (h : (1 - Real.cos x + Real.sin x) / (1 + Real.cos x + Real.sin x) = -2) : Real.tan x = 4 / 3 :=
by sorry

end tan_of_trig_eq_l197_197669


namespace value_of_a_b_c_l197_197917

theorem value_of_a_b_c 
    (a b c : Int)
    (h1 : ∀ x : Int, x^2 + 10*x + 21 = (x + a) * (x + b))
    (h2 : ∀ x : Int, x^2 + 3*x - 88 = (x + b) * (x - c))
    :
    a + b + c = 18 := 
sorry

end value_of_a_b_c_l197_197917


namespace greatest_power_of_2_divides_l197_197142

-- Define the conditions as Lean definitions.
def a : ℕ := 15
def b : ℕ := 3
def n : ℕ := 600

-- Define the theorem statement based on the conditions and correct answer.
theorem greatest_power_of_2_divides (x : ℕ) (y : ℕ) (k : ℕ) (h₁ : x = a) (h₂ : y = b) (h₃ : k = n) :
  ∃ m : ℕ, (x^k - y^k) % (2^1200) = 0 ∧ ¬ ∃ m' : ℕ, m' > m ∧ (x^k - y^k) % (2^m') = 0 := sorry

end greatest_power_of_2_divides_l197_197142


namespace problem_proof_l197_197534

-- Define the conditions
def a (n : ℕ) : Real := sorry  -- a is some real number, so it's non-deterministic here

def a_squared (n : ℕ) : Real := a n ^ (2 * n)  -- a^(2n)

-- Main theorem to prove
theorem problem_proof (n : ℕ) (h : a_squared n = 3) : 2 * (a n ^ (6 * n)) - 1 = 53 :=
by
  sorry  -- Proof to be completed

end problem_proof_l197_197534


namespace remainder_of_fractions_l197_197601

theorem remainder_of_fractions : 
  ∀ (x y : ℚ), x = 5/7 → y = 3/4 → (x - y * ⌊x / y⌋) = 5/7 :=
by
  intros x y hx hy
  rw [hx, hy]
  -- Additional steps can be filled in here, if continuing with the proof.
  sorry

end remainder_of_fractions_l197_197601


namespace total_marbles_l197_197903

theorem total_marbles :
  let marbles_second_bowl := 600
  let marbles_first_bowl := (3/4) * marbles_second_bowl
  let total_marbles := marbles_first_bowl + marbles_second_bowl
  total_marbles = 1050 := by
  sorry -- proof skipped

end total_marbles_l197_197903


namespace quad_completion_l197_197137

theorem quad_completion (a b c : ℤ) 
    (h : ∀ x : ℤ, 8 * x^2 - 48 * x - 128 = a * (x + b)^2 + c) : 
    a + b + c = -195 := 
by
  sorry

end quad_completion_l197_197137


namespace sum_of_perimeters_l197_197896

theorem sum_of_perimeters (A1 A2 : ℝ) (h1 : A1 + A2 = 145) (h2 : A1 - A2 = 25) :
  4 * Real.sqrt 85 + 4 * Real.sqrt 60 = 4 * Real.sqrt A1 + 4 * Real.sqrt A2 :=
by
  sorry

end sum_of_perimeters_l197_197896


namespace even_gt_one_square_gt_l197_197208

theorem even_gt_one_square_gt (m : ℕ) (h_even : ∃ k : ℕ, m = 2 * k) (h_gt_one : m > 1) : m < m * m :=
by
  sorry

end even_gt_one_square_gt_l197_197208


namespace area_triangle_ABC_l197_197829

noncomputable def area_trapezoid (AB CD height : ℝ) : ℝ :=
  (AB + CD) * height / 2

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  base * height / 2

variable (AB CD height area_ABCD : ℝ)
variables (h0 : CD = 3 * AB) (h1 : area_trapezoid AB CD height = 24)

theorem area_triangle_ABC : area_triangle AB height = 6 :=
by
  sorry

end area_triangle_ABC_l197_197829


namespace cost_per_board_game_is_15_l197_197837

-- Definitions of the conditions
def number_of_board_games : ℕ := 6
def bill_paid : ℕ := 100
def bill_value : ℕ := 5
def bills_received : ℕ := 2

def total_change := bills_received * bill_value
def total_cost := bill_paid - total_change
def cost_per_board_game := total_cost / number_of_board_games

-- The theorem stating that the cost of each board game is $15
theorem cost_per_board_game_is_15 : cost_per_board_game = 15 := 
by
  -- Omitted proof steps
  sorry

end cost_per_board_game_is_15_l197_197837


namespace total_amount_spent_is_40_l197_197572

-- Definitions based on conditions
def tomatoes_pounds : ℕ := 2
def tomatoes_price_per_pound : ℕ := 5
def apples_pounds : ℕ := 5
def apples_price_per_pound : ℕ := 6

-- Total amount spent computed
def total_spent : ℕ :=
  (tomatoes_pounds * tomatoes_price_per_pound) +
  (apples_pounds * apples_price_per_pound)

-- The Lean theorem statement
theorem total_amount_spent_is_40 : total_spent = 40 := by
  unfold total_spent
  unfold tomatoes_pounds tomatoes_price_per_pound apples_pounds apples_price_per_pound
  calc
    2 * 5 + 5 * 6 = 10 + 30 : by rfl
    ... = 40 : by rfl

end total_amount_spent_is_40_l197_197572


namespace normal_price_of_article_l197_197924

theorem normal_price_of_article 
  (final_price : ℝ)
  (discount1 : ℝ) 
  (discount2 : ℝ) 
  (P : ℝ)
  (h : final_price = 108) 
  (h1 : discount1 = 0.10) 
  (h2 : discount2 = 0.20)
  (h_eq : (1 - discount1) * (1 - discount2) * P = final_price) :
  P = 150 := by
  sorry

end normal_price_of_article_l197_197924


namespace f_le_one_l197_197202

open Real

theorem f_le_one (x : ℝ) (hx : 0 < x) : (1 + log x) / x ≤ 1 := 
sorry

end f_le_one_l197_197202


namespace complementary_angles_decrease_percentage_l197_197433

theorem complementary_angles_decrease_percentage :
  ∀ (x : ℝ), (3 * x + 7 * x = 90) →
  (3 * x * 1.2 + 7 * x = 90) →
  (3 * x > 0) →
  (7 * x > 0) →
  let original_larger_angle : ℝ := 7 * x in
  let new_smaller_angle : ℝ := 3 * x * 1.2 in
  let new_larger_angle : ℝ := 90 - new_smaller_angle in
  let decrease : ℝ := original_larger_angle - new_larger_angle in
  let percentage_decrease : ℝ := (decrease / original_larger_angle) * 100 in
  percentage_decrease = 8.57 := 
sorry

end complementary_angles_decrease_percentage_l197_197433


namespace even_function_is_a_4_l197_197370

def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_is_a_4 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 4 := by
  sorry

end even_function_is_a_4_l197_197370


namespace elves_closed_eyes_l197_197723

theorem elves_closed_eyes :
  ∃ (age: ℕ → ℕ), -- Function assigning each position an age
  (∀ n, 1 ≤ n ∧ n ≤ 100 → (age n < age ((n % 100) + 1) ∧ age n < age (n - 1 % 100 + 1)) ∨
                          (age n > age ((n % 100) + 1) ∧ age n > age (n - 1 % 100 + 1))) :=
by
  sorry

end elves_closed_eyes_l197_197723


namespace line_intersects_ellipse_two_points_l197_197524

theorem line_intersects_ellipse_two_points {m n : ℝ} (h1 : ¬∃ x y : ℝ, m*x + n*y = 4 ∧ x^2 + y^2 = 4)
  (h2 : m^2 + n^2 < 4) : 
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ (m * p1.1 + n * p1.2 = 4) ∧ (m * p2.1 + n * p2.2 = 4) ∧ 
  (p1.1^2 / 9 + p1.2^2 / 4 = 1) ∧ (p2.1^2 / 9 + p2.2^2 / 4 = 1) :=
sorry

end line_intersects_ellipse_two_points_l197_197524


namespace remainder_of_sum_mod_18_l197_197055

theorem remainder_of_sum_mod_18 :
  let nums := [85, 86, 87, 88, 89, 90, 91, 92, 93]
  let sum_nums := nums.sum
  let product := 90 * sum_nums
  product % 18 = 10 :=
by
  sorry

end remainder_of_sum_mod_18_l197_197055


namespace Jeanine_has_more_pencils_than_Clare_l197_197553

def number_pencils_Jeanine_bought : Nat := 18
def number_pencils_Clare_bought := number_pencils_Jeanine_bought / 2
def number_pencils_given_to_Abby := number_pencils_Jeanine_bought / 3
def number_pencils_Jeanine_now := number_pencils_Jeanine_bought - number_pencils_given_to_Abby 

theorem Jeanine_has_more_pencils_than_Clare :
  number_pencils_Jeanine_now - number_pencils_Clare_bought = 3 := by
  sorry

end Jeanine_has_more_pencils_than_Clare_l197_197553


namespace complementary_angles_ratio_decrease_l197_197427

theorem complementary_angles_ratio_decrease 
  (a b : ℝ) (h_ratio : a / b = 3 / 7) (h_comp : a + b = 90) : 
  let a' := a * 1.20 in
  let b' := 90 - a' in
  ((b' / b) * 100 = 91.43) ∧ (100 - (b' / b) * 100 = 8.57) :=
by
  have : a / b = 3 / 7 := h_ratio
  have : a + b = 90 := h_comp
  let a' := a * 1.20
  let b' := 90 - a'
  have : b' / b = 57.6 / 63 := sorry -- Calculations omitted for brevity
  have : (b' / b) * 100 = 91.43 := by sorry
  have : 100 - (b' / b) * 100 = 8.57 := by sorry
  split;
  assumption

end complementary_angles_ratio_decrease_l197_197427


namespace arithmetic_sequence_general_formula_l197_197990

theorem arithmetic_sequence_general_formula :
  (∀ n:ℕ, ∃ (a_n : ℕ), ∀ k:ℕ, a_n = 2 * k → k = n)
  ∧ ( 2 * n + 2 * (n + 2) = 8 → 2 * n + 2 * (n + 3) = 12 → a_n = 2 * n )
  ∧ (S_n = (n * (n + 1)) / 2 → S_n = 420 → n = 20) :=
by { sorry }

end arithmetic_sequence_general_formula_l197_197990


namespace brother_catch_up_in_3_minutes_l197_197581

variables (v_s v_b : ℝ) (t t_new : ℝ)

-- Conditions
def brother_speed_later_leaves_catch (v_b : ℝ) (v_s : ℝ) : Prop :=
18 * v_s = 12 * v_b

def new_speed_of_brother (v_b v_s : ℝ) : ℝ :=
2 * v_b

def time_to_catch_up (v_s : ℝ) (t_new : ℝ) : Prop :=
6 + t_new = 3 * t_new

-- Goal: prove that t_new = 3
theorem brother_catch_up_in_3_minutes (v_s v_b : ℝ) (t_new : ℝ) :
  (brother_speed_later_leaves_catch v_b v_s) → 
  (new_speed_of_brother v_b v_s) = 3 * v_s → 
  time_to_catch_up v_s t_new → 
  t_new = 3 :=
by sorry

end brother_catch_up_in_3_minutes_l197_197581


namespace hyperbola_foci_l197_197422

theorem hyperbola_foci :
  (∀ x y : ℝ, x^2 - 2 * y^2 = 1) →
  (∃ c : ℝ, c = (Real.sqrt 6) / 2 ∧ (x = c ∨ x = -c) ∧ y = 0) :=
by
  sorry

end hyperbola_foci_l197_197422


namespace measure_angle_ADC_l197_197214

variable (A B C D : Type)
variable [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Definitions for the angles
variable (angle_ABC angle_BCD angle_ADC : ℝ)

-- Conditions for the problem
axiom Angle_ABC_is_4_times_Angle_BCD : angle_ABC = 4 * angle_BCD
axiom Angle_BCD_ADC_sum_to_180 : angle_BCD + angle_ADC = 180

-- The theorem that we want to prove
theorem measure_angle_ADC (Angle_ABC_is_4_times_Angle_BCD: angle_ABC = 4 * angle_BCD)
    (Angle_BCD_ADC_sum_to_180: angle_BCD + angle_ADC = 180) : 
    angle_ADC = 144 :=
by
  sorry

end measure_angle_ADC_l197_197214


namespace Jeanine_more_pencils_than_Clare_l197_197554

variables (Jeanine_pencils : ℕ) (Clare_pencils : ℕ)

def Jeanine_initial_pencils := 18
def Clare_initial_pencils := Jeanine_initial_pencils / 2
def Jeanine_pencils_given_to_Abby := Jeanine_initial_pencils / 3
def Jeanine_remaining_pencils := Jeanine_initial_pencils - Jeanine_pencils_given_to_Abby

theorem Jeanine_more_pencils_than_Clare :
  Jeanine_remaining_pencils - Clare_initial_pencils = 3 :=
by
  -- This is just the statement, the proof is not provided as instructed.
  sorry

end Jeanine_more_pencils_than_Clare_l197_197554


namespace decode_CLUE_is_8671_l197_197437

def BEST_OF_LUCK_code : List (Char × Nat) :=
  [('B', 0), ('E', 1), ('S', 2), ('T', 3), ('O', 4), ('F', 5),
   ('L', 6), ('U', 7), ('C', 8), ('K', 9)]

def decode (code : List (Char × Nat)) (word : String) : Option Nat :=
  word.toList.mapM (λ c => List.lookup c code) >>= (λ digits => 
  Option.some (Nat.ofDigits 10 digits))

theorem decode_CLUE_is_8671 :
  decode BEST_OF_LUCK_code "CLUE" = some 8671 :=
by
  -- Proof omitted
  sorry

end decode_CLUE_is_8671_l197_197437


namespace profit_percentage_l197_197620

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 550) (hSP : SP = 715) : 
  ((SP - CP) / CP) * 100 = 30 := sorry

end profit_percentage_l197_197620


namespace mandy_yoga_time_l197_197121

theorem mandy_yoga_time (G B Y : ℕ) (h1 : 2 * B = 3 * G) (h2 : 3 * Y = 2 * (G + B)) (h3 : Y = 30) : Y = 30 := by
  sorry

end mandy_yoga_time_l197_197121


namespace min_k_valid_l197_197531

def S : Set ℕ := {1, 2, 3, 4}

def valid_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ b : Fin 4 → ℕ,
    (∀ i : Fin 4, b i ∈ S) ∧ b 3 ≠ 1 →
    ∃ i1 i2 i3 i4 : Fin (k + 1), i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧
      (a i1 = b 0 ∧ a i2 = b 1 ∧ a i3 = b 2 ∧ a i4 = b 3)

def min_k := 11

theorem min_k_valid : ∀ a : ℕ → ℕ,
  valid_sequence a min_k → 
  min_k = 11 :=
sorry

end min_k_valid_l197_197531


namespace total_money_l197_197839

theorem total_money (John Alice Bob : ℝ) (hJohn : John = 5 / 8) (hAlice : Alice = 7 / 20) (hBob : Bob = 1 / 4) :
  John + Alice + Bob = 1.225 := 
by 
  sorry

end total_money_l197_197839


namespace quadratic_root_zero_l197_197059

theorem quadratic_root_zero (k : ℝ) :
  (∃ x : ℝ, (k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0) →
  (∃ x : ℝ, x = 0 ∧ ((k + 2) * x^2 + 6 * x + k^2 + k - 2 = 0)) →
  k = 1 :=
by
  sorry

end quadratic_root_zero_l197_197059


namespace event_B_more_likely_l197_197235

theorem event_B_more_likely (A B : Set (ℕ → ℕ)) 
  (hA : ∀ ω, ω ∈ A ↔ ∃ i j, i ≠ j ∧ ω i = ω j)
  (hB : ∀ ω, ω ∈ B ↔ ∀ i j, i ≠ j → ω i ≠ ω j) :
  ∃ prob_A prob_B : ℚ, prob_A = 4 / 9 ∧ prob_B = 5 / 9 ∧ prob_B > prob_A :=
by
  sorry

end event_B_more_likely_l197_197235


namespace exists_solution_l197_197716

noncomputable def smallest_c0 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 1) : ℕ :=
  a * b - a - b + 1

theorem exists_solution (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 1) :
  ∃ c0, (c0 = smallest_c0 a b ha hb h) ∧ ∀ c : ℕ, c ≥ c0 → ∃ x y : ℕ, a * x + b * y = c :=
sorry

end exists_solution_l197_197716


namespace distance_between_trees_l197_197009

theorem distance_between_trees (num_trees : ℕ) (length_yard : ℝ)
  (h1 : num_trees = 26) (h2 : length_yard = 800) : 
  (length_yard / (num_trees - 1)) = 32 :=
by
  sorry

end distance_between_trees_l197_197009


namespace count_zeros_in_decimal_rep_l197_197173

theorem count_zeros_in_decimal_rep (n : ℕ) (h : n = 2^3 * 5^7) : 
  ∀ (a b : ℕ), (∃ (a : ℕ) (b : ℕ), n = 10^b ∧ a < 10^b) → 
  6 = b - 1 := by
  sorry

end count_zeros_in_decimal_rep_l197_197173


namespace original_number_l197_197912

theorem original_number (x : ℤ) (h : (x + 4) % 23 = 0) : x = 19 :=
sorry

end original_number_l197_197912


namespace sum_of_squares_of_coefficients_l197_197293

theorem sum_of_squares_of_coefficients :
  let p := 5 * (Polynomial.C (1 : ℤ) * Polynomial.X^4 + Polynomial.C (4 : ℤ) * Polynomial.X^3 + Polynomial.C (2 : ℤ) * Polynomial.X^2 + Polynomial.C (1 : ℤ))
  (Polynomial.coeff p 4)^2 + (Polynomial.coeff p 3)^2 + (Polynomial.coeff p 2)^2 + (Polynomial.coeff p 1)^2 = 550 :=
by
  let p := 5 * (Polynomial.C (1 : ℤ) * Polynomial.X^4 + Polynomial.C (4 : ℤ) * Polynomial.X^3 + Polynomial.C (2 : ℤ) * Polynomial.X^2 + Polynomial.C (1 : ℤ))
  have hc4 : Polynomial.coeff p 4 = 5 := sorry
  have hc3 : Polynomial.coeff p 3 = 20 := sorry
  have hc2 : Polynomial.coeff p 2 = 10 := sorry
  have hc1 : Polynomial.coeff p 1 = 5 := sorry
  calc
    (Polynomial.coeff p 4)^2 + (Polynomial.coeff p 3)^2 + (Polynomial.coeff p 2)^2 + (Polynomial.coeff p 1)^2
      = 5^2 + 20^2 + 10^2 + 5^2 : by rw [hc4, hc3, hc2, hc1]
      = 25 + 400 + 100 + 25 : by norm_num
      = 550 : by norm_num


end sum_of_squares_of_coefficients_l197_197293


namespace ratio_of_radii_l197_197110

-- Given conditions
variables {b a c : ℝ}
variables (h1 : π * b^2 - π * c^2 = 2 * π * a^2)
variables (h2 : c = 1.5 * a)

-- Define and prove the ratio
theorem ratio_of_radii (h1: π * b^2 - π * c^2 = 2 * π * a^2) (h2: c = 1.5 * a) :
  a / b = 2 / Real.sqrt 17 :=
sorry

end ratio_of_radii_l197_197110


namespace inequality_holds_l197_197127

-- Define the function for the inequality condition
def inequality (n : ℕ) (x : ℝ) : Prop :=
  (1 - x + x^2 / 2) ^ n - (1 - x) ^ n ≤ x / 2

theorem inequality_holds :
  ∀ (n : ℕ) (x : ℝ), 0 < n → (0 ≤ x ∧ x ≤ 1) → inequality n x :=
begin
  intros n x hn hx,
  sorry -- Proof goes here
end

end inequality_holds_l197_197127


namespace minimum_toothpicks_removal_l197_197512

theorem minimum_toothpicks_removal
  (total_toothpicks : ℕ)
  (grid_size : ℕ)
  (toothpicks_per_square : ℕ)
  (shared_sides : ℕ)
  (interior_toothpicks : ℕ) 
  (diagonal_toothpicks : ℕ)
  (min_removal : ℕ) 
  (no_squares_or_triangles : Bool)
  (h1 : total_toothpicks = 40)
  (h2 : grid_size = 3)
  (h3 : toothpicks_per_square = 4)
  (h4 : shared_sides = 16)
  (h5 : interior_toothpicks = 16) 
  (h6 : diagonal_toothpicks = 12)
  (h7 : min_removal = 16)
: no_squares_or_triangles := 
sorry

end minimum_toothpicks_removal_l197_197512


namespace dow_original_value_l197_197729

-- Given conditions
def Dow_end := 8722
def percentage_fall := 0.02
def final_percentage := 1 - percentage_fall -- 98% of the original value

-- To prove: the original value of the Dow Jones Industrial Average equals 8900
theorem dow_original_value :
  (Dow_end: ℝ) = (final_percentage * 8900) := 
by sorry

end dow_original_value_l197_197729


namespace percentage_neither_language_l197_197233

noncomputable def total_diplomats : ℝ := 120
noncomputable def latin_speakers : ℝ := 20
noncomputable def russian_non_speakers : ℝ := 32
noncomputable def both_languages : ℝ := 0.10 * total_diplomats

theorem percentage_neither_language :
  let D := total_diplomats
  let L := latin_speakers
  let R := D - russian_non_speakers
  let LR := both_languages
  ∃ P, P = 100 * (D - (L + R - LR)) / D :=
by
  existsi ((total_diplomats - (latin_speakers + (total_diplomats - russian_non_speakers) - both_languages)) / total_diplomats * 100)
  sorry

end percentage_neither_language_l197_197233


namespace probability_of_quitters_from_10_member_tribe_is_correct_l197_197590

noncomputable def probability_quitters_from_10_member_tribe : ℚ :=
  let total_contestants := 18
  let ten_member_tribe := 10
  let total_quitters := 2
  let comb (n k : ℕ) : ℕ := Nat.choose n k
  
  let total_combinations := comb total_contestants total_quitters
  let ten_tribe_combinations := comb ten_member_tribe total_quitters
  
  ten_tribe_combinations / total_combinations

theorem probability_of_quitters_from_10_member_tribe_is_correct :
  probability_quitters_from_10_member_tribe = 5 / 17 :=
  by
    sorry

end probability_of_quitters_from_10_member_tribe_is_correct_l197_197590


namespace log_domain_inequality_l197_197102

theorem log_domain_inequality {a : ℝ} : 
  (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ a > 1 :=
sorry

end log_domain_inequality_l197_197102


namespace value_set_l197_197227

open Real Set

noncomputable def possible_values (a b c : ℝ) : Set ℝ :=
  {x | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ x = c / a + c / b}

theorem value_set (c : ℝ) (hc : c > 0) : possible_values a b c = Ici (2 * c) := by
  sorry

end value_set_l197_197227


namespace find_number_l197_197821

theorem find_number (x y : ℝ) (h1 : x = y + 0.25 * y) (h2 : x = 110) : y = 88 := 
by
  sorry

end find_number_l197_197821


namespace molly_age_is_63_l197_197411

variable (Sandy_age Molly_age : ℕ)

theorem molly_age_is_63 (h1 : Sandy_age = 49) (h2 : Sandy_age / Molly_age = 7 / 9) : Molly_age = 63 :=
by
  sorry

end molly_age_is_63_l197_197411


namespace michael_initial_fish_l197_197232

-- Define the conditions
def benGave : ℝ := 18.0
def totalFish : ℝ := 67

-- Define the statement to be proved
theorem michael_initial_fish :
  (totalFish - benGave) = 49 := by
  sorry

end michael_initial_fish_l197_197232


namespace exam_problem_solution_l197_197175

theorem exam_problem_solution :
  let Pa := 1/3
  let Pb := 1/4
  let Pc := 1/5
  let Pnone := (1 - Pa) * (1 - Pb) * (1 - Pc)
  let Pat_least_one := 1 - Pnone
  Pat_least_one = 3/5 :=
by
  -- formal proof would go here
  sorry

end exam_problem_solution_l197_197175


namespace manufacturing_section_degrees_l197_197132

theorem manufacturing_section_degrees (percentage : ℝ) (total_degrees : ℝ) (h1 : total_degrees = 360) (h2 : percentage = 35) : 
  ((percentage / 100) * total_degrees) = 126 :=
by
  sorry

end manufacturing_section_degrees_l197_197132


namespace max_unique_dance_counts_l197_197148

theorem max_unique_dance_counts (boys girls : ℕ) (positive_boys : boys = 29) (positive_girls : girls = 15) 
  (dances : ∀ b g, b ≤ boys → g ≤ girls → ℕ) :
  ∃ num_dances, num_dances = 29 := 
by
  sorry

end max_unique_dance_counts_l197_197148


namespace expected_balls_in_original_position_l197_197412

/-- 
  The expected number of balls that are in their original positions after three successive transpositions,
  given seven balls arranged in a circle and three people (Chris, Silva, and Alex) each randomly interchanging two adjacent balls, is 3.2.
-/
theorem expected_balls_in_original_position : 
  let n := 7 in
  let transpositions := 3 in
  let expected_position (n : ℕ) (transpositions : ℕ) : ℚ := sorry in
  expected_position n transpositions = 3.2 := 
sorry

end expected_balls_in_original_position_l197_197412


namespace arithmetic_sequence_sum_l197_197199

theorem arithmetic_sequence_sum :
  ∀(a_n : ℕ → ℕ) (S : ℕ → ℕ) (a_1 d : ℕ),
    (∀ n, a_n n = a_1 + (n - 1) * d) →
    (∀ n, S n = n * (a_1 + (n - 1) * d) / 2) →
    a_1 = 2 →
    S 4 = 20 →
    S 6 = 42 :=
by
  sorry

end arithmetic_sequence_sum_l197_197199


namespace irrational_product_rational_l197_197751

-- Definitions of irrational and rational for clarity
def irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q
def rational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Statement of the problem in Lean 4
theorem irrational_product_rational (a b : ℕ) (ha : irrational (Real.sqrt a)) (hb : irrational (Real.sqrt b)) :
  rational ((Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b)) :=
by
  sorry

end irrational_product_rational_l197_197751


namespace quadratic_common_root_l197_197066

theorem quadratic_common_root (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : ∃ x, x^2 + a * x + b = 0 ∧ x^2 + c * x + a = 0)
  (h2 : ∃ x, x^2 + a * x + b = 0 ∧ x^2 + b * x + c = 0)
  (h3 : ∃ x, x^2 + b * x + c = 0 ∧ x^2 + c * x + a = 0) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end quadratic_common_root_l197_197066


namespace problem_statement_l197_197577

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)

theorem problem_statement : ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by 
  sorry

end problem_statement_l197_197577


namespace second_number_value_l197_197016

theorem second_number_value (x y : ℝ) (h1 : (1/5) * x = (5/8) * y) 
                                      (h2 : x + 35 = 4 * y) : y = 40 := 
by 
  sorry

end second_number_value_l197_197016


namespace range_of_m_l197_197353

noncomputable def f (x : ℝ) (m : ℝ) :=
if x ≤ 2 then x^2 - m * (2 * x - 1) + m^2 else 2^(x + 1)

theorem range_of_m {m : ℝ} :
  (∀ x, f x m ≥ f 2 m) → (2 ≤ m ∧ m ≤ 4) :=
by
  sorry

end range_of_m_l197_197353


namespace quadratic_factors_l197_197438

-- Define the quadratic polynomial
def quadratic (b c x : ℝ) : ℝ := x^2 + b * x + c

-- Define the roots
def root1 : ℝ := -2
def root2 : ℝ := 3

-- Theorem: If the quadratic equation has roots -2 and 3, then it factors as (x + 2)(x - 3)
theorem quadratic_factors (b c : ℝ) (h1 : quadratic b c root1 = 0) (h2 : quadratic b c root2 = 0) :
  ∀ x : ℝ, quadratic b c x = (x + 2) * (x - 3) :=
by
  sorry

end quadratic_factors_l197_197438


namespace event_B_more_likely_than_event_A_l197_197237

/-- Define the outcomes when rolling a die three times --/
def total_outcomes : ℕ := 6 ^ 3

/-- Define the number of ways to choose 3 different numbers from 6 --/
def choose_3_from_6 : ℕ := Nat.choose 6 3

/-- Define the number of ways to arrange 3 different numbers --/
def arrangements_3 : ℕ := 3.factorial

/-- Calculate the number of favorable outcomes for event B --/
def favorable_B : ℕ := choose_3_from_6 * arrangements_3

/-- Define the probability of event B --/
noncomputable def prob_B : ℝ := favorable_B / total_outcomes

/-- Define the probability of event A as the complement of event B --/
noncomputable def prob_A : ℝ := 1 - prob_B

/-- The theorem to prove that event B is more likely than event A --/
theorem event_B_more_likely_than_event_A : prob_B > prob_A :=
by
  sorry

end event_B_more_likely_than_event_A_l197_197237


namespace total_birds_in_pet_store_l197_197456

theorem total_birds_in_pet_store
  (number_of_cages : ℕ)
  (parrots_per_cage : ℕ)
  (parakeets_per_cage : ℕ)
  (total_birds_in_cage : ℕ)
  (total_birds : ℕ) :
  number_of_cages = 8 →
  parrots_per_cage = 2 →
  parakeets_per_cage = 7 →
  total_birds_in_cage = parrots_per_cage + parakeets_per_cage →
  total_birds = number_of_cages * total_birds_in_cage →
  total_birds = 72 := by
  intros h1 h2 h3 h4 h5
  sorry

end total_birds_in_pet_store_l197_197456


namespace geometric_series_sum_eq_4_div_3_l197_197476

theorem geometric_series_sum_eq_4_div_3 (a : ℝ) (r : ℝ) (h₀ : a = 1) (h₁ : r = 1 / 4) :
  ∑' n : ℕ, a * r^n = 4 / 3 := by
  sorry

end geometric_series_sum_eq_4_div_3_l197_197476


namespace spadesuit_calculation_l197_197663

def spadesuit (x y : ℝ) : ℝ := (x + 2 * y) ^ 2 * (x - y)

theorem spadesuit_calculation :
  spadesuit 3 (spadesuit 2 3) = 1046875 :=
by
  sorry

end spadesuit_calculation_l197_197663


namespace problem_statement_l197_197113

theorem problem_statement (A B C : Real)
  (h1 : A + B + C = 180)
  (h2 : C > 90) : cos B > sin A := by
  sorry

end problem_statement_l197_197113


namespace divisibility_equivalence_distinct_positive_l197_197400

variable (a b c : ℕ)

theorem divisibility_equivalence_distinct_positive (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c) ∣ (a^3 * b + b^3 * c + c^3 * a)) ↔ ((a + b + c) ∣ (a * b^3 + b * c^3 + c * a^3)) :=
by sorry

end divisibility_equivalence_distinct_positive_l197_197400


namespace tan_alpha_plus_pi_div_four_l197_197189

theorem tan_alpha_plus_pi_div_four (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 := 
by
  sorry

end tan_alpha_plus_pi_div_four_l197_197189


namespace total_marbles_l197_197904

theorem total_marbles (bowl2_capacity : ℕ) (h₁ : bowl2_capacity = 600)
    (h₂ : 3 / 4 * bowl2_capacity = 450) : 600 + (3 / 4 * 600) = 1050 := by
  sorry

end total_marbles_l197_197904


namespace ellipse_semi_minor_axis_l197_197200

theorem ellipse_semi_minor_axis (b : ℝ) 
    (h1 : 0 < b) 
    (h2 : b < 5)
    (h_ellipse : ∀ x y : ℝ, x^2 / 25 + y^2 / b^2 = 1) 
    (h_eccentricity : 4 / 5 = 4 / 5) : b = 3 := 
sorry

end ellipse_semi_minor_axis_l197_197200


namespace suitable_for_systematic_sampling_l197_197319

-- Define the given conditions as a structure
structure SamplingProblem where
  option_A : String
  option_B : String
  option_C : String
  option_D : String

-- Define the equivalence theorem to prove Option C is the most suitable
theorem suitable_for_systematic_sampling (p : SamplingProblem) 
(hA: p.option_A = "Randomly selecting 8 students from a class of 48 students to participate in an activity")
(hB: p.option_B = "A city has 210 department stores, including 20 large stores, 40 medium stores, and 150 small stores. To understand the business situation of each store, a sample of 21 stores needs to be drawn")
(hC: p.option_C = "Randomly selecting 100 candidates from 1200 exam participants to analyze the answer situation of the questions")
(hD: p.option_D = "Randomly selecting 10 students from 1200 high school students participating in a mock exam to understand the situation") :
  p.option_C = "Randomly selecting 100 candidates from 1200 exam participants to analyze the answer situation of the questions" := 
sorry

end suitable_for_systematic_sampling_l197_197319


namespace total_number_of_marbles_is_1050_l197_197907

def total_marbles : Nat :=
  let marbles_in_second_bowl := 600
  let marbles_in_first_bowl := (3 * marbles_in_second_bowl) / 4
  marbles_in_first_bowl + marbles_in_second_bowl

theorem total_number_of_marbles_is_1050 : total_marbles = 1050 := by
  sorry

end total_number_of_marbles_is_1050_l197_197907


namespace product_of_469111_and_9999_l197_197942

theorem product_of_469111_and_9999 : 469111 * 9999 = 4690418889 := 
by 
  sorry

end product_of_469111_and_9999_l197_197942


namespace cayli_combinations_l197_197946

theorem cayli_combinations (art_choices sports_choices music_choices : ℕ)
  (h1 : art_choices = 2)
  (h2 : sports_choices = 3)
  (h3 : music_choices = 4) :
  art_choices * sports_choices * music_choices = 24 := by
  sorry

end cayli_combinations_l197_197946


namespace line_b_y_intercept_l197_197862

variable (b : ℝ → ℝ)
variable (x y : ℝ)

-- Line b is parallel to y = -3x + 6
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c, (∀ x, b x = m * x + c) ∧ m = -3

-- Line b passes through the point (3, -2)
def passes_through_point (b : ℝ → ℝ) : Prop :=
  b 3 = -2

-- The y-intercept of line b
def y_intercept (b : ℝ → ℝ) : ℝ :=
  b 0

theorem line_b_y_intercept (h1 : is_parallel b) (h2 : passes_through_point b) : y_intercept b = 7 :=
sorry

end line_b_y_intercept_l197_197862


namespace find_m_n_l197_197054

theorem find_m_n (m n : ℕ) (positive_m : 0 < m) (positive_n : 0 < n)
  (h1 : m = 3) (h2 : n = 4) :
    Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / m) + Real.arctan (1 / n) = π / 2 :=
  by 
    -- Placeholder for the proof
    sorry

end find_m_n_l197_197054


namespace pen_price_l197_197322

theorem pen_price (x y : ℝ) (h1 : 2 * x + 3 * y = 49) (h2 : 3 * x + y = 49) : x = 14 :=
by
  -- Proof required here
  sorry

end pen_price_l197_197322


namespace problem_statement_l197_197397

noncomputable def alpha : ℝ := 3 + Real.sqrt 8
noncomputable def x : ℝ := alpha ^ 1000
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := by
  sorry

end problem_statement_l197_197397


namespace quadratic_inequality_solutions_l197_197967

theorem quadratic_inequality_solutions {k : ℝ} (h1 : 0 < k) (h2 : k < 16) :
  ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end quadratic_inequality_solutions_l197_197967


namespace blue_bordered_area_on_outer_sphere_l197_197627

theorem blue_bordered_area_on_outer_sphere :
  let r := 1 -- cm
  let r1 := 4 -- cm
  let r2 := 6 -- cm
  let A_inner := 27 -- cm^2
  let h := A_inner / (2 * π * r1)
  let A_outer := 2 * π * r2 * h
  A_outer = 60.75 := sorry

end blue_bordered_area_on_outer_sphere_l197_197627


namespace Homer_first_try_points_l197_197364

variable (x : ℕ)
variable (h1 : x + (x - 70) + 2 * (x - 70) = 1390)

theorem Homer_first_try_points : x = 400 := by
  sorry

end Homer_first_try_points_l197_197364


namespace find_f1_l197_197784

theorem find_f1 (f : ℝ → ℝ)
  (h : ∀ x, x ≠ 1 / 2 → f x + f ((x + 2) / (1 - 2 * x)) = x) :
  f 1 = 7 / 6 :=
sorry

end find_f1_l197_197784


namespace tan_alpha_is_neg_5_over_12_l197_197347

variables (α : ℝ) (h1 : Real.sin α = 5/13) (h2 : π/2 < α ∧ α < π)

theorem tan_alpha_is_neg_5_over_12 : Real.tan α = -5/12 :=
by
  sorry

end tan_alpha_is_neg_5_over_12_l197_197347


namespace electric_sharpens_more_l197_197933

noncomputable def number_of_pencils_hand_crank : ℕ := 360 / 45
noncomputable def number_of_pencils_electric : ℕ := 360 / 20

theorem electric_sharpens_more : number_of_pencils_electric - number_of_pencils_hand_crank = 10 := by
  sorry

end electric_sharpens_more_l197_197933


namespace problem_provable_l197_197576

noncomputable def given_expression (a : ℝ) : ℝ :=
  (1 / (a + 2)) / ((a^2 - 4 * a + 4) / (a^2 - 4)) - (2 / (a - 2))

theorem problem_provable : given_expression (Real.sqrt 5 + 2) = - (Real.sqrt 5 / 5) :=
by
  sorry

end problem_provable_l197_197576


namespace cement_tesss_street_l197_197245

-- Definitions of the given conditions
def cement_lexis_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Proof statement to show the amount of cement used to pave Tess's street
theorem cement_tesss_street : total_cement_used - cement_lexis_street = 5.1 :=
by 
  -- Add proof steps to show the theorem is valid.
  sorry

end cement_tesss_street_l197_197245


namespace probability_different_colors_l197_197104

theorem probability_different_colors :
  let total_chips := 16
  let prob_blue := (7 : ℚ) / total_chips
  let prob_yellow := (5 : ℚ) / total_chips
  let prob_red := (4 : ℚ) / total_chips
  let prob_blue_then_nonblue := prob_blue * ((prob_yellow + prob_red) : ℚ)
  let prob_yellow_then_non_yellow := prob_yellow * ((prob_blue + prob_red) : ℚ)
  let prob_red_then_non_red := prob_red * ((prob_blue + prob_yellow) : ℚ)
  let total_prob := prob_blue_then_nonblue + prob_yellow_then_non_yellow + prob_red_then_non_red
  total_prob = (83 : ℚ) / 128 := 
by
  sorry

end probability_different_colors_l197_197104


namespace mike_baseball_cards_l197_197574

theorem mike_baseball_cards :
  let InitialCards : ℕ := 87
  let BoughtCards : ℕ := 13
  (InitialCards - BoughtCards = 74)
:= by
  sorry

end mike_baseball_cards_l197_197574


namespace real_and_imag_parts_of_z_l197_197197

noncomputable def real_part (z : ℂ) : ℝ := z.re
noncomputable def imag_part (z : ℂ) : ℝ := z.im

theorem real_and_imag_parts_of_z :
  ∀ (i : ℂ), i * i = -1 → 
  ∀ (z : ℂ), z = i * (-1 + 2 * i) → real_part z = -2 ∧ imag_part z = -1 :=
by 
  intros i hi z hz
  sorry

end real_and_imag_parts_of_z_l197_197197


namespace paint_ratio_l197_197514

theorem paint_ratio
  (blue yellow white : ℕ)
  (ratio_b : ℕ := 4)
  (ratio_y : ℕ := 3)
  (ratio_w : ℕ := 5)
  (total_white : ℕ := 15)
  : yellow = 9 := by
  have ratio := ratio_b + ratio_y + ratio_w
  have white_parts := total_white * ratio_w / ratio_w
  have yellow_parts := white_parts * ratio_y / ratio_w
  exact sorry

end paint_ratio_l197_197514


namespace arccos_one_half_eq_pi_div_three_l197_197168

theorem arccos_one_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 :=
sorry

end arccos_one_half_eq_pi_div_three_l197_197168


namespace sum_of_squares_of_coefficients_l197_197292

theorem sum_of_squares_of_coefficients :
  let p := 5 * (Polynomial.C (1 : ℤ) * Polynomial.X^4 + Polynomial.C (4 : ℤ) * Polynomial.X^3 + Polynomial.C (2 : ℤ) * Polynomial.X^2 + Polynomial.C (1 : ℤ))
  (Polynomial.coeff p 4)^2 + (Polynomial.coeff p 3)^2 + (Polynomial.coeff p 2)^2 + (Polynomial.coeff p 1)^2 = 550 :=
by
  let p := 5 * (Polynomial.C (1 : ℤ) * Polynomial.X^4 + Polynomial.C (4 : ℤ) * Polynomial.X^3 + Polynomial.C (2 : ℤ) * Polynomial.X^2 + Polynomial.C (1 : ℤ))
  have hc4 : Polynomial.coeff p 4 = 5 := sorry
  have hc3 : Polynomial.coeff p 3 = 20 := sorry
  have hc2 : Polynomial.coeff p 2 = 10 := sorry
  have hc1 : Polynomial.coeff p 1 = 5 := sorry
  calc
    (Polynomial.coeff p 4)^2 + (Polynomial.coeff p 3)^2 + (Polynomial.coeff p 2)^2 + (Polynomial.coeff p 1)^2
      = 5^2 + 20^2 + 10^2 + 5^2 : by rw [hc4, hc3, hc2, hc1]
      = 25 + 400 + 100 + 25 : by norm_num
      = 550 : by norm_num


end sum_of_squares_of_coefficients_l197_197292


namespace geometric_series_sum_eq_4_over_3_l197_197479

theorem geometric_series_sum_eq_4_over_3 : 
  let a := 1
  let r := 1/4
  inf_geometric_sum a r = 4/3 := by
begin
  intros,
  sorry
end

end geometric_series_sum_eq_4_over_3_l197_197479


namespace area_of_triangle_ABC_l197_197357

theorem area_of_triangle_ABC : 
  let A := (1, 1)
  let B := (4, 1)
  let C := (1, 5)
  let area := 6
  (1:ℝ) * abs (1 * (1 - 5) + 4 * (5 - 1) + 1 * (1 - 1)) / 2 = area := 
by
  sorry

end area_of_triangle_ABC_l197_197357


namespace base_three_to_base_ten_l197_197259

theorem base_three_to_base_ten (n : ℕ) (h : n = 20121) : 
  let convert := 2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0 in
  convert = 178 :=
by
  have : convert = 162 + 0 + 9 + 6 + 1, by sorry
  show convert = 178, by sorry

end base_three_to_base_ten_l197_197259


namespace no_solutions_l197_197886

theorem no_solutions (x : ℝ) (h : x ≠ 0) : 4 * Real.sin x - 3 * Real.cos x ≠ 5 + 1 / |x| := 
by
  sorry

end no_solutions_l197_197886


namespace factorize_expression_l197_197051

theorem factorize_expression (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l197_197051


namespace calculate_parallel_segment_length_l197_197822

theorem calculate_parallel_segment_length :
  ∀ (d : ℝ), 
    ∃ (X Y Z P : Type) 
    (XY YZ XZ : ℝ), 
    XY = 490 ∧ 
    YZ = 520 ∧ 
    XZ = 560 ∧ 
    ∃ (D D' E E' F F' : Type),
      (D ≠ E ∧ E ≠ F ∧ F ≠ D') ∧  
      (XZ - (d * (520/490) + d * (520/560))) = d → d = 268.148148 :=
by
  sorry

end calculate_parallel_segment_length_l197_197822


namespace probability_at_least_two_defective_probability_at_most_one_defective_l197_197004

variable (P_no_defective : ℝ)
variable (P_one_defective : ℝ)
variable (P_two_defective : ℝ)
variable (P_all_defective : ℝ)

theorem probability_at_least_two_defective (hP_no_defective : P_no_defective = 0.18)
                                          (hP_one_defective : P_one_defective = 0.53)
                                          (hP_two_defective : P_two_defective = 0.27)
                                          (hP_all_defective : P_all_defective = 0.02) :
  P_two_defective + P_all_defective = 0.29 :=
  by sorry

theorem probability_at_most_one_defective (hP_no_defective : P_no_defective = 0.18)
                                          (hP_one_defective : P_one_defective = 0.53)
                                          (hP_two_defective : P_two_defective = 0.27)
                                          (hP_all_defective : P_all_defective = 0.02) :
  P_no_defective + P_one_defective = 0.71 :=
  by sorry

end probability_at_least_two_defective_probability_at_most_one_defective_l197_197004


namespace total_games_played_l197_197012

theorem total_games_played (n : ℕ) (h : n = 8) : (n.choose 2) = 28 := by
  sorry

end total_games_played_l197_197012


namespace find_x_l197_197773

theorem find_x (x : ℕ) : 
  (∃ (students : ℕ), students = 10) ∧ 
  (∃ (selected : ℕ), selected = 6) ∧ 
  (¬ (∃ (k : ℕ), k = 5 ∧ k = x) ) ∧ 
  (1 ≤ 10 - x) ∧
  (3 ≤ x ∧ x ≤ 4) :=
by
  sorry

end find_x_l197_197773


namespace arithmetic_sequence_a20_l197_197348

theorem arithmetic_sequence_a20 (a : Nat → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 3 + a 5 = 18)
  (h3 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 :=
sorry

end arithmetic_sequence_a20_l197_197348


namespace pass_in_both_subjects_l197_197544

variable (F_H F_E F_HE : ℝ)

theorem pass_in_both_subjects (h1 : F_H = 20) (h2 : F_E = 70) (h3 : F_HE = 10) :
  100 - ((F_H + F_E) - F_HE) = 20 :=
by
  sorry

end pass_in_both_subjects_l197_197544


namespace units_digit_7_pow_451_l197_197915

theorem units_digit_7_pow_451 : (7^451 % 10) = 3 := by
  sorry

end units_digit_7_pow_451_l197_197915


namespace valid_interval_for_k_l197_197969

theorem valid_interval_for_k :
  ∀ k : ℝ, (∀ x : ℝ, x^2 - 8*x + k < 0 → 0 < k ∧ k < 16) :=
by
  sorry

end valid_interval_for_k_l197_197969


namespace original_number_l197_197309

theorem original_number (x : ℝ) (h : x + 0.5 * x = 90) : x = 60 :=
by
  sorry

end original_number_l197_197309


namespace proof_problem_l197_197919

/- Define relevant concepts -/
def is_factor (a b : Nat) := ∃ k, b = a * k
def is_divisor := is_factor

/- Given conditions with their translations -/
def condition_A : Prop := is_factor 5 35
def condition_B : Prop := is_divisor 21 252 ∧ ¬ is_divisor 21 48
def condition_C : Prop := ¬ (is_divisor 15 90 ∨ is_divisor 15 74)
def condition_D : Prop := is_divisor 18 36 ∧ ¬ is_divisor 18 72
def condition_E : Prop := is_factor 9 180

/- The main proof problem statement -/
theorem proof_problem : condition_A ∧ condition_B ∧ ¬ condition_C ∧ ¬ condition_D ∧ condition_E :=
by
  sorry

end proof_problem_l197_197919


namespace dominoes_per_player_l197_197836

-- Define the conditions
def total_dominoes : ℕ := 28
def number_of_players : ℕ := 4

-- The theorem
theorem dominoes_per_player : total_dominoes / number_of_players = 7 :=
by sorry

end dominoes_per_player_l197_197836


namespace arc_length_EF_l197_197582

-- Definitions based on the conditions
def angle_DEF_degrees : ℝ := 45
def circumference_D : ℝ := 80
def total_circle_degrees : ℝ := 360

-- Theorems/lemmata needed to prove the required statement
theorem arc_length_EF :
  let proportion := angle_DEF_degrees / total_circle_degrees
  let arc_length := proportion * circumference_D
  arc_length = 10 :=
by
  -- Placeholder for the proof
  sorry

end arc_length_EF_l197_197582


namespace simon_legos_l197_197253

theorem simon_legos (Kent_legos : ℕ) (hk : Kent_legos = 40)
                    (Bruce_legos : ℕ) (hb : Bruce_legos = Kent_legos + 20)
                    (Simon_legos : ℕ) (hs : Simon_legos = Bruce_legos + Bruce_legos / 5) :
    Simon_legos = 72 := 
sorry

end simon_legos_l197_197253


namespace simplify_sqrt_l197_197517

theorem simplify_sqrt (x : ℝ) (h : x < 2) : Real.sqrt (x^2 - 4*x + 4) = 2 - x :=
by
  sorry

end simplify_sqrt_l197_197517


namespace inequality_and_equality_conditions_l197_197978

theorem inequality_and_equality_conditions
    {a b c d : ℝ}
    (ha : 0 < a)
    (hb : 0 < b)
    (hc : 0 < c)
    (hd : 0 < d) :
  (a ^ (1/3) * b ^ (1/3) + c ^ (1/3) * d ^ (1/3) ≤ (a + b + c) ^ (1/3) * (a + c + d) ^ (1/3)) ↔ 
  (b = (a / c) * (a + c) ∧ d = (c / a) * (a + c)) :=
  sorry

end inequality_and_equality_conditions_l197_197978


namespace eval_nabla_l197_197639

def nabla (a b : ℕ) : ℕ := 3 + b^(a-1)

theorem eval_nabla : nabla (nabla 2 3) 4 = 1027 := by
  -- proof goes here
  sorry

end eval_nabla_l197_197639


namespace factor_expression_l197_197693

-- Define variables s and m
variables (s m : ℤ)

-- State the theorem to be proven: If s = 5, then m^2 - sm - 24 can be factored as (m - 8)(m + 3)
theorem factor_expression (hs : s = 5) : m^2 - s * m - 24 = (m - 8) * (m + 3) :=
by {
  sorry
}

end factor_expression_l197_197693


namespace smallest_mn_sum_l197_197892

theorem smallest_mn_sum {n m : ℕ} (h1 : n > m) (h2 : 1978 ^ n % 1000 = 1978 ^ m % 1000) (h3 : m ≥ 1) : m + n = 106 := 
sorry

end smallest_mn_sum_l197_197892


namespace krishan_nandan_investment_l197_197221

def investment_ratio (k r₁ r₂ : ℕ) (N T Gn : ℕ) : Prop :=
  k = r₁ ∧ r₂ = 1 ∧ Gn = N * T ∧ k * N * 3 * T + Gn = 26000 ∧ Gn = 2000

/-- Given the conditions, the ratio of Krishan's investment to Nandan's investment is 4:1. -/
theorem krishan_nandan_investment :
  ∃ k N T Gn Gn_total : ℕ, 
    investment_ratio k 4 1 N T Gn  ∧ k * N * 3 * T = 24000 :=
by
  sorry

end krishan_nandan_investment_l197_197221


namespace monotonicity_and_range_l197_197386

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l197_197386


namespace y_intercept_of_line_b_l197_197865

noncomputable def line_b_y_intercept (b : Type) [HasElem ℝ b] : Prop :=
  ∃ (m : ℝ) (c : ℝ), (m = -3) ∧ (c = 7) ∧ ∀ (x : ℝ) (y : ℝ), (x, y) ∈ b → y = -3 * x + c

theorem y_intercept_of_line_b (b : Type) [HasElem (ℝ × ℝ) b] :
  (∃ (p : ℝ × ℝ), p = (3, -2) ∧ ∃ (q : line_b_y_intercept b), q) →
  ∃ (c : ℝ), c = 7 :=
by
  intro h
  sorry

end y_intercept_of_line_b_l197_197865


namespace ellipse_eq_standard_line_intersect_ellipse_l197_197985

noncomputable theory

-- Given conditions for the ellipse
def ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

-- Conditions from the problem
variables (a b : ℝ)
variables (ha : a > b) (hb : b > 0)
variables (e : ℝ) (he : e = sqrt 3 / 2)
variables (c : ℝ) (hc : c = sqrt 3)
variables (focus_dist : ℝ) (hdist : focus_dist = sqrt 6)

-- Standard equation of the ellipse
theorem ellipse_eq_standard : a = 2 → b = 1 → ∀ x y : ℝ, x^2 / 4 + y^2 = 1 :=
sorry

-- Conditions for part II
variables (F : ℝ) (hF : F = sqrt 3)
variables (θ : ℝ) (hθ : θ = 45 * (π / 180))
variables (width : ℝ) (height : ℝ)
variables (intersect_1 intersect_2 : ℝ)

-- Line passing through right focus with angle 45, intersecting with the ellipse
theorem line_intersect_ellipse (h_eq : F = sqrt 3) (a_eq : a = 2) (b_eq : b = 1) :
  let y1 := intersect_1 in
  let y2 := intersect_2 in
  5 * y1^2 + 2 * sqrt 3 * y1 - 1 = 0 ∧
  5 * y2^2 + 2 * sqrt 3 * y2 - 1 = 0 ∧
  (|y1 - y2| = 4 * sqrt 2 / 5) ∧
  ((sqrt 3 / 2) * (4 * sqrt 2 / 5) = 2 * sqrt 6 / 5) :=
sorry

end ellipse_eq_standard_line_intersect_ellipse_l197_197985


namespace solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism_l197_197695

-- Definitions as conditions
def is_cone (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_cylinder (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_triangular_pyramid (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_rectangular_prism (solid : Type) : Prop := -- Definition placeholder
sorry 

-- Predicate to check if the front view of a solid is a quadrilateral
def front_view_is_quadrilateral (solid : Type) : Prop :=
  (is_cylinder solid ∨ is_rectangular_prism solid)

-- Theorem stating the problem
theorem solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism
    (s : Type) :
  front_view_is_quadrilateral s ↔ is_cylinder s ∨ is_rectangular_prism s :=
by
  sorry

end solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism_l197_197695


namespace tree_count_l197_197596

theorem tree_count (m N : ℕ) 
  (h1 : 12 ≡ (33 - m) [MOD N])
  (h2 : (105 - m) ≡ 8 [MOD N]) :
  N = 76 := 
sorry

end tree_count_l197_197596


namespace net_profit_is_90_l197_197031

theorem net_profit_is_90
    (cost_seeds cost_soil : ℝ)
    (num_plants : ℕ)
    (price_per_plant : ℝ)
    (h0 : cost_seeds = 2)
    (h1 : cost_soil = 8)
    (h2 : num_plants = 20)
    (h3 : price_per_plant = 5) :
    (num_plants * price_per_plant - (cost_seeds + cost_soil)) = 90 := by
  sorry

end net_profit_is_90_l197_197031


namespace car_speed_in_first_hour_l197_197741

theorem car_speed_in_first_hour (x : ℝ) 
  (second_hour_speed : ℝ := 40)
  (average_speed : ℝ := 60)
  (h : (x + second_hour_speed) / 2 = average_speed) :
  x = 80 := 
by
  -- Additional steps needed to solve this theorem
  sorry

end car_speed_in_first_hour_l197_197741


namespace inequality_solution_set_l197_197528

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : (a - 1) * x > 2) : x < 2 / (a - 1) ↔ a < 1 :=
by
  sorry

end inequality_solution_set_l197_197528


namespace sum_inequality_l197_197399

variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {m n p k : ℕ}

-- Definitions for the conditions given in the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i j, a (i + 1) - a i = a (j + 1) - a j

def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 1 + a (n - 1)) / 2

def non_negative_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≥ 0

-- The theorem to prove
theorem sum_inequality (arith_seq : is_arithmetic_sequence a)
  (S_eq : sum_of_arithmetic_sequence S a)
  (nn_seq : non_negative_sequence a)
  (h1 : m + n = 2 * p) (m_pos : 0 < m) (n_pos : 0 < n) (p_pos : 0 < p) :
  1 / (S m) ^ k + 1 / (S n) ^ k ≥ 2 / (S p) ^ k :=
by sorry

end sum_inequality_l197_197399


namespace triangles_exist_l197_197496

def exists_triangles : Prop :=
  ∃ (T : Fin 100 → Type) 
    (h : (i : Fin 100) → ℝ) 
    (A : (i : Fin 100) → ℝ)
    (is_isosceles : (i : Fin 100) → Prop),
    (∀ i : Fin 100, is_isosceles i) ∧
    (∀ i : Fin 99, h (i + 1) = 200 * h i) ∧
    (∀ i : Fin 99, A (i + 1) = A i / 20000) ∧
    (∀ i : Fin 100, 
      ¬(∃ (cover : (Fin 99) → Type),
        (∀ j : Fin 99, cover j = T j) ∧
        (∀ j : Fin 99, ∀ k : Fin 100, k ≠ i → ¬(cover j = T k))))

theorem triangles_exist : exists_triangles :=
sorry

end triangles_exist_l197_197496


namespace butterfly_count_l197_197962

theorem butterfly_count (total_butterflies : ℕ) (one_third_flew_away : ℕ) (initial_butterflies : total_butterflies = 9) (flew_away : one_third_flew_away = total_butterflies / 3) : 
(total_butterflies - one_third_flew_away) = 6 := by
  sorry

end butterfly_count_l197_197962


namespace value_of_x_l197_197080

theorem value_of_x (x y : ℝ) :
  x / (x + 1) = (y^2 + 3*y + 1) / (y^2 + 3*y + 2) → x = y^2 + 3*y + 1 :=
by
  intro h
  sorry

end value_of_x_l197_197080


namespace angle_BAD_37_5_degrees_l197_197301
  
noncomputable def ∆ (A B C : Type) : triangle A B C := sorry

variables {A B C D : Point}
variables (h₁ : isosceles_triangle A B C)
variables (h₂ : ∠ACB = 30)
variables (h₃ : midpoint_segment B C D)
variables (h₄ : angle_bisector A B C A D)

theorem angle_BAD_37_5_degrees (h₁ : isosceles_triangle AC BC)
    (h₂ : ∠ACB = 30) (h₃ : midpoint_segment B C D) (h₄ : angle_bisector A B C A D) :
  ∠BAD = 37.5 := sorry

end angle_BAD_37_5_degrees_l197_197301


namespace probability_divisible_by_three_l197_197537

noncomputable def prob_divisible_by_three : ℚ :=
  1 - (4/6)^6

theorem probability_divisible_by_three :
  prob_divisible_by_three = 665 / 729 :=
by
  sorry

end probability_divisible_by_three_l197_197537


namespace quadratic_roots_always_implies_l197_197802

variable {k x1 x2 : ℝ}

theorem quadratic_roots_always_implies (h1 : k^2 > 16) 
  (h2 : x1 + x2 = -k)
  (h3 : x1 * x2 = 4) : x1^2 + x2^2 > 8 :=
by
  sorry

end quadratic_roots_always_implies_l197_197802


namespace probability_of_first_four_cards_each_suit_l197_197373

noncomputable def probability_first_four_different_suits : ℚ := 3 / 32

theorem probability_of_first_four_cards_each_suit :
  let n := 52
  let k := 5
  let suits := 4
  (probability_first_four_different_suits = (3 / 32)) :=
by
  sorry

end probability_of_first_four_cards_each_suit_l197_197373


namespace max_elements_of_valid_set_l197_197714

def valid_set (M : Finset ℤ) : Prop :=
  ∀ (a b c : ℤ), a ∈ M → b ∈ M → c ∈ M → (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (a + b ∈ M ∨ a + c ∈ M ∨ b + c ∈ M)

theorem max_elements_of_valid_set (M : Finset ℤ) (h : valid_set M) : M.card ≤ 7 :=
sorry

end max_elements_of_valid_set_l197_197714


namespace arithmetic_mean_of_remaining_numbers_l197_197108

-- Definitions and conditions
def initial_set_size : ℕ := 60
def initial_arithmetic_mean : ℕ := 45
def numbers_to_remove : List ℕ := [50, 55, 60]

-- Calculation of the total sum
def total_sum : ℕ := initial_arithmetic_mean * initial_set_size

-- Calculation of the sum of the numbers to remove
def sum_of_removed_numbers : ℕ := numbers_to_remove.sum

-- Sum of the remaining numbers
def new_sum : ℕ := total_sum - sum_of_removed_numbers

-- Size of the remaining set
def remaining_set_size : ℕ := initial_set_size - numbers_to_remove.length

-- The arithmetic mean of the remaining numbers
def new_arithmetic_mean : ℚ := new_sum / remaining_set_size

-- The proof statement
theorem arithmetic_mean_of_remaining_numbers :
  new_arithmetic_mean = 2535 / 57 :=
by
  sorry

end arithmetic_mean_of_remaining_numbers_l197_197108


namespace g_of_1986_l197_197847

-- Define the function g and its properties
noncomputable def g : ℕ → ℤ :=
sorry  -- Placeholder for the actual definition according to the conditions

axiom g_is_defined (x : ℕ) : x ≥ 0 → ∃ y : ℤ, g x = y
axiom g_at_1 : g 1 = 1
axiom g_add (a b : ℕ) (h_a : a ≥ 0) (h_b : b ≥ 0) : g (a + b) = g a + g b - 3 * g (a * b) + 1

-- Lean statement for the proof problem
theorem g_of_1986 : g 1986 = 0 :=
sorry

end g_of_1986_l197_197847


namespace no_positive_integer_solutions_l197_197186

theorem no_positive_integer_solutions (m : ℕ) (h_pos : m > 0) :
  ¬ ∃ x : ℚ, m * x^2 + 40 * x + m = 0 :=
by {
  -- the proof goes here
  sorry
}

end no_positive_integer_solutions_l197_197186


namespace num_new_terms_in_sequence_l197_197599

theorem num_new_terms_in_sequence (k : ℕ) (h : k ≥ 2) : 
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end num_new_terms_in_sequence_l197_197599


namespace power_sum_l197_197634

theorem power_sum : 2^4 + 2^4 + 2^5 + 2^5 = 96 := 
by
  sorry

end power_sum_l197_197634


namespace pyramid_lateral_edge_ratio_l197_197622

variable (h x : ℝ)

-- We state the conditions as hypotheses
axiom pyramid_intersected_by_plane_parallel_to_base (h : ℝ) (S S' : ℝ) :
  S' = S / 2 → (S' / S = (x / h) ^ 2) → (x = h / Real.sqrt 2)

-- The theorem we need to prove
theorem pyramid_lateral_edge_ratio (h x : ℝ) (S S' : ℝ)
  (cond1 : S' = S / 2)
  (cond2 : S' / S = (x / h) ^ 2) :
  x / h = 1 / Real.sqrt 2 :=
by
  -- skip the proof
  sorry

end pyramid_lateral_edge_ratio_l197_197622


namespace exists_valid_board_configuration_l197_197125

open Matrix

def board (m n : Nat) := Matrix (Fin m) (Fin n) Bool

def is_adjacent {m n : Nat} (i j : Fin m) (k l : Fin n) : Prop :=
  (i = k ∧ abs (j.val - l.val) = 1) ∨ (j = l ∧ abs (i.val - k.val) = 1)

def valid_configuration {m n : Nat} (b : board m n) : Prop :=
  ∀ (i : Fin m) (j : Fin n), b i j = false → 
  ∃ (k : Fin m) (l : Fin n), is_adjacent i j k l ∧ b k l = true

def example_board : board 4 6
| ⟨0,_⟩, ⟨1,_⟩ := true
| ⟨0,_⟩, ⟨3,_⟩ := true
| ⟨1,_⟩, ⟨1,_⟩ := true
| ⟨1,_⟩, ⟨3,_⟩ := true
| ⟨2,_⟩, ⟨1,_⟩ := true
| ⟨2,_⟩, ⟨3,_⟩ := true
| ⟨3,_⟩, ⟨1,_⟩ := true
| _, _ := false

theorem exists_valid_board_configuration : 
  valid_configuration example_board := sorry

end exists_valid_board_configuration_l197_197125


namespace digit_a_for_divisibility_l197_197052

theorem digit_a_for_divisibility (a : ℕ) (h1 : (8 * 10^3 + 7 * 10^2 + 5 * 10 + a) % 6 = 0) : a = 4 :=
sorry

end digit_a_for_divisibility_l197_197052


namespace arrangement_count_l197_197609

def arrangements_with_conditions 
  (boys girls : Nat) 
  (cannot_be_next_to_each_other : Bool) : Nat :=
if cannot_be_next_to_each_other then
  sorry -- The proof will go here
else
  sorry

theorem arrangement_count :
  arrangements_with_conditions 3 2 true = 72 :=
sorry

end arrangement_count_l197_197609


namespace factorial_difference_l197_197035

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l197_197035


namespace total_value_l197_197754

/-- 
The total value of the item V can be determined based on the given conditions.
- The merchant paid an import tax of $109.90.
- The tax rate is 7%.
- The tax is only on the portion of the value above $1000.

Given these conditions, prove that the total value V is 2567.
-/
theorem total_value {V : ℝ} (h1 : 0.07 * (V - 1000) = 109.90) : V = 2567 :=
by
  sorry

end total_value_l197_197754


namespace A_and_B_finish_together_in_11_25_days_l197_197758

theorem A_and_B_finish_together_in_11_25_days (A_rate B_rate : ℝ)
    (hA : A_rate = 1/18) (hB : B_rate = 1/30) :
    1 / (A_rate + B_rate) = 11.25 := by
  sorry

end A_and_B_finish_together_in_11_25_days_l197_197758


namespace solve_abs_equation_l197_197884

theorem solve_abs_equation (y : ℝ) (h : |y - 8| + 3 * y = 12) : y = 2 :=
sorry

end solve_abs_equation_l197_197884


namespace person_a_catch_up_person_b_5_times_l197_197743

theorem person_a_catch_up_person_b_5_times :
  ∀ (num_flags laps_a laps_b : ℕ),
  num_flags = 2015 →
  laps_a = 23 →
  laps_b = 13 →
  (∃ t : ℕ, ∃ n : ℕ, 10 * t = num_flags * n ∧
             23 * t / 10 = k * num_flags ∧
             n % 2 = 0) →
  n = 10 →
  10 / (2 * 1) = 5 :=
by sorry

end person_a_catch_up_person_b_5_times_l197_197743


namespace sequence_less_than_inverse_l197_197717

-- Define the sequence and conditions given in the problem
variables {a : ℕ → ℝ}
axiom positive_sequence (n : ℕ) : 0 < a n
axiom sequence_inequality (n : ℕ) : a n ^ 2 ≤ a n - a (n + 1)

theorem sequence_less_than_inverse (n : ℕ) : a n < 1 / n := 
sorry

end sequence_less_than_inverse_l197_197717


namespace disjoint_subsets_with_same_sum_l197_197000

theorem disjoint_subsets_with_same_sum :
  ∀ (S : Finset ℕ), S.card = 10 ∧ (∀ x ∈ S, x ∈ Finset.range 101) →
  ∃ A B : Finset ℕ, A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end disjoint_subsets_with_same_sum_l197_197000


namespace a_sufficient_not_necessary_for_a_squared_eq_b_squared_l197_197689

theorem a_sufficient_not_necessary_for_a_squared_eq_b_squared
  (a b : ℝ) :
  (a = b) → (a^2 = b^2) ∧ ¬ ((a^2 = b^2) → (a = b)) :=
  sorry

end a_sufficient_not_necessary_for_a_squared_eq_b_squared_l197_197689


namespace ram_krish_task_completion_l197_197922

/-!
  Given:
  1. Ram's efficiency (R) is half of Krish's efficiency (K).
  2. Ram can complete the task alone in 24 days.

  To Prove:
  Ram and Krish will complete the task together in 8 days.
-/

theorem ram_krish_task_completion {R K : ℝ} (hR : R = 1 / 2 * K)
  (hRAMalone : R ≠ 0) (hRAMtime : 24 * R = 1) :
  1 / (R + K) = 8 := by
  sorry

end ram_krish_task_completion_l197_197922


namespace monomial_same_type_l197_197026

-- Define a structure for monomials
structure Monomial where
  coeff : ℕ
  vars : List String

-- Monomials definitions based on the given conditions
def m1 := Monomial.mk 3 ["a"]
def m2 := Monomial.mk 2 ["b"]
def m3 := Monomial.mk 1 ["a", "b"]
def m4 := Monomial.mk 3 ["a", "c"]
def target := Monomial.mk 2 ["a", "b"]

-- Define a predicate to check if two monomials are of the same type
def sameType (m n : Monomial) : Prop :=
  m.vars = n.vars

theorem monomial_same_type :
  sameType m3 target := sorry

end monomial_same_type_l197_197026


namespace cube_difference_l197_197796

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 26) : a^3 - b^3 = 124 :=
by sorry

end cube_difference_l197_197796


namespace factors_2310_l197_197074

theorem factors_2310 : ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ S.card = 5 ∧ (2310 = S.prod id) :=
by
  sorry

end factors_2310_l197_197074


namespace josh_total_payment_with_tax_and_discount_l197_197382

-- Definitions
def total_string_cheeses (pack1 : ℕ) (pack2 : ℕ) (pack3 : ℕ) : ℕ :=
  pack1 + pack2 + pack3

def total_cost_before_tax_and_discount (n : ℕ) (cost_per_cheese : ℚ) : ℚ :=
  n * cost_per_cheese

def discount_amount (cost : ℚ) (discount_rate : ℚ) : ℚ :=
  cost * discount_rate

def discounted_cost (cost : ℚ) (discount : ℚ) : ℚ :=
  cost - discount

def sales_tax_amount (cost : ℚ) (tax_rate : ℚ) : ℚ :=
  cost * tax_rate

def total_cost (cost : ℚ) (tax : ℚ) : ℚ :=
  cost + tax

-- The statement
theorem josh_total_payment_with_tax_and_discount :
  let cost_per_cheese := 0.10
  let discount_rate := 0.05
  let tax_rate := 0.12
  total_cost (discounted_cost (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese)
                              (discount_amount (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese) discount_rate))
             (sales_tax_amount (discounted_cost (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese)
                                               (discount_amount (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese) discount_rate)) tax_rate) = 6.81 := 
  sorry

end josh_total_payment_with_tax_and_discount_l197_197382


namespace find_base_k_l197_197109

-- Define the conversion condition as a polynomial equation.
def base_conversion (k : ℤ) : Prop := k^2 + 3*k + 2 = 42

-- State the theorem to be proven: given the conversion condition, k = 5.
theorem find_base_k (k : ℤ) (h : base_conversion k) : k = 5 :=
by
  sorry

end find_base_k_l197_197109


namespace trigonometric_identity_l197_197084

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end trigonometric_identity_l197_197084


namespace hearing_aid_cost_l197_197218

theorem hearing_aid_cost
  (cost : ℝ)
  (insurance_coverage : ℝ)
  (personal_payment : ℝ)
  (total_aid_count : ℕ)
  (h : total_aid_count = 2)
  (h_insurance : insurance_coverage = 0.80)
  (h_personal_payment : personal_payment = 1000)
  (h_equation : personal_payment = (1 - insurance_coverage) * (total_aid_count * cost)) :
  cost = 2500 :=
by
  sorry

end hearing_aid_cost_l197_197218


namespace probability_product_divisible_by_3_l197_197538

theorem probability_product_divisible_by_3 :
  let p := (1 : ℚ) - (2 / 3) ^ 6 in
  p = 665 / 729 :=
by
  sorry

end probability_product_divisible_by_3_l197_197538


namespace problem_statement_l197_197179

/-- Definition of the function f that relates the input n with floor functions -/
def f (n : ℕ) : ℤ :=
  n + ⌊(n : ℤ) / 6⌋ - ⌊(n : ℤ) / 2⌋ - ⌊2 * (n : ℤ) / 3⌋

/-- Prove the main statement -/
theorem problem_statement (n : ℕ) (hpos : 0 < n) :
  f n = 0 ↔ ∃ k : ℕ, n = 6 * k + 1 :=
sorry -- Proof goes here.

end problem_statement_l197_197179


namespace min_rungs_l197_197471

theorem min_rungs (a b : ℕ) (h_a : a > 0) (h_b : b > 0) :
  ∃ (n : ℕ), (∀ x y : ℤ, a*x - b*y = n) ∧ n = a + b - Nat.gcd a b := 
sorry

end min_rungs_l197_197471


namespace floor_e_eq_two_l197_197651

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 := 
sorry

end floor_e_eq_two_l197_197651


namespace original_number_is_13_l197_197308

theorem original_number_is_13 (x : ℝ) (h : 3 * (2 * x + 7) = 99) : x = 13 :=
sorry

end original_number_is_13_l197_197308


namespace complementary_angles_decrease_86_percent_l197_197425

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end complementary_angles_decrease_86_percent_l197_197425


namespace number_of_valid_integers_l197_197992

theorem number_of_valid_integers (n : ℕ) (h1 : n ≤ 2021) (h2 : ∀ m : ℕ, m^2 ≤ n → n < (m + 1)^2 → ((m^2 + 1) ∣ (n^2 + 1))) : 
  ∃ k, k = 47 :=
by
  sorry

end number_of_valid_integers_l197_197992


namespace solve_for_d_l197_197575

theorem solve_for_d (r s t d c : ℝ)
  (h1 : (t = -r - s))
  (h2 : (c = rs + rt + st))
  (h3 : (t - 1 = -(r + 5) - (s - 4)))
  (h4 : (c = (r + 5) * (s - 4) + (r + 5) * (t - 1) + (s - 4) * (t - 1)))
  (h5 : (d = -r * s * t))
  (h6 : (d + 210 = -(r + 5) * (s - 4) * (t - 1))) :
  d = 240 ∨ d = 420 :=
by
  sorry

end solve_for_d_l197_197575


namespace greatest_divisor_same_remainder_l197_197659

theorem greatest_divisor_same_remainder (a b c : ℕ) (h₁ : a = 54) (h₂ : b = 87) (h₃ : c = 172) : 
  ∃ d, (d ∣ (b - a)) ∧ (d ∣ (c - b)) ∧ (d ∣ (c - a)) ∧ (∀ e, (e ∣ (b - a)) ∧ (e ∣ (c - b)) ∧ (e ∣ (c - a)) → e ≤ d) ∧ d = 1 := 
by 
  sorry

end greatest_divisor_same_remainder_l197_197659


namespace inequality_solution_set_l197_197527

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : (a - 1) * x > 2) : x < 2 / (a - 1) ↔ a < 1 :=
by
  sorry

end inequality_solution_set_l197_197527


namespace simon_legos_l197_197254

theorem simon_legos (Kent_legos : ℕ) (hk : Kent_legos = 40)
                    (Bruce_legos : ℕ) (hb : Bruce_legos = Kent_legos + 20)
                    (Simon_legos : ℕ) (hs : Simon_legos = Bruce_legos + Bruce_legos / 5) :
    Simon_legos = 72 := 
sorry

end simon_legos_l197_197254


namespace marbles_in_jar_l197_197306

theorem marbles_in_jar (M : ℕ) (h1 : ∀ n : ℕ, n = 20 → ∀ m : ℕ, m = M / n → ∀ a b : ℕ, a = n + 2 → b = m - 1 → ∀ k : ℕ, k = M / a → k = b) : M = 220 :=
by 
  sorry

end marbles_in_jar_l197_197306


namespace simon_legos_l197_197252

theorem simon_legos (Kent_legos : ℕ) (hk : Kent_legos = 40)
                    (Bruce_legos : ℕ) (hb : Bruce_legos = Kent_legos + 20)
                    (Simon_legos : ℕ) (hs : Simon_legos = Bruce_legos + Bruce_legos / 5) :
    Simon_legos = 72 := 
sorry

end simon_legos_l197_197252


namespace find_pairs_l197_197970

theorem find_pairs (x y p : ℕ)
  (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : x ≤ y) (h4 : Prime p) :
  (x = 3 ∧ y = 5 ∧ p = 7) ∨ (x = 1 ∧ ∃ q, Prime q ∧ y = q + 1 ∧ p = q ∧ q ≠ 7) ↔
  (x + y) * (x * y - 1) / (x * y + 1) = p := 
sorry

end find_pairs_l197_197970


namespace find_remainder_l197_197198

theorem find_remainder (x y P Q : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x^4 + y^4 = (P + 13) * (x + y) + Q) : Q = 8 :=
sorry

end find_remainder_l197_197198


namespace value_of_abc_l197_197263

-- Conditions
def cond1 (a b : ℤ) : Prop := ∀ x : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b)
def cond2 (b c : ℤ) : Prop := ∀ x : ℤ, x^2 - 23 * x + 132 = (x - b) * (x - c)

-- Theorem statement
theorem value_of_abc (a b c : ℤ) (h₁ : cond1 a b) (h₂ : cond2 b c) : a + b + c = 31 :=
sorry

end value_of_abc_l197_197263


namespace distance_center_of_ball_travels_l197_197929

noncomputable def radius_of_ball : ℝ := 2
noncomputable def R1 : ℝ := 100
noncomputable def R2 : ℝ := 60
noncomputable def R3 : ℝ := 80

noncomputable def adjusted_R1 : ℝ := R1 - radius_of_ball
noncomputable def adjusted_R2 : ℝ := R2 + radius_of_ball
noncomputable def adjusted_R3 : ℝ := R3 - radius_of_ball

noncomputable def distance_travelled : ℝ :=
  (Real.pi * adjusted_R1) +
  (Real.pi * adjusted_R2) +
  (Real.pi * adjusted_R3)

theorem distance_center_of_ball_travels : distance_travelled = 238 * Real.pi :=
by
  sorry

end distance_center_of_ball_travels_l197_197929


namespace geometric_sequence_sum_l197_197542

theorem geometric_sequence_sum 
  (a r : ℝ) 
  (h1 : a + a * r = 8)
  (h2 : a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 = 120) :
  a * (1 + r + r^2 + r^3) = 30 := 
by
  sorry

end geometric_sequence_sum_l197_197542


namespace simon_number_of_legos_l197_197247

variable (Kent_legos : ℕ) (Bruce_legos : ℕ) (Simon_legos : ℕ)

def Kent_condition : Prop := Kent_legos = 40
def Bruce_condition : Prop := Bruce_legos = Kent_legos + 20 
def Simon_condition : Prop := Simon_legos = Bruce_legos + (Bruce_legos * 20 / 100)

theorem simon_number_of_legos : Kent_condition Kent_legos ∧ Bruce_condition Kent_legos Bruce_legos ∧ Simon_condition Bruce_legos Simon_legos → Simon_legos = 72 := by
  intros h
  -- proof steps would go here
  sorry

end simon_number_of_legos_l197_197247


namespace height_of_square_pyramid_is_13_l197_197770

noncomputable def square_pyramid_height (base_edge : ℝ) (adjacent_face_angle : ℝ) : ℝ :=
  let half_diagonal := base_edge * (Real.sqrt 2) / 2
  let sin_angle := Real.sin (adjacent_face_angle / 2 : ℝ)
  let opp_side := half_diagonal * sin_angle
  let height := half_diagonal * sin_angle / (Real.sqrt 3)
  height

theorem height_of_square_pyramid_is_13 :
  ∀ (base_edge : ℝ) (adjacent_face_angle : ℝ), 
  base_edge = 26 → 
  adjacent_face_angle = 120 → 
  square_pyramid_height base_edge adjacent_face_angle = 13 :=
by
  intros base_edge adjacent_face_angle h_base_edge h_adj_face_angle
  rw [h_base_edge, h_adj_face_angle]
  have half_diagonal := 26 * (Real.sqrt 2) / 2
  have sin_angle := Real.sin (120 / 2 : ℝ) -- sin 60 degrees
  have sqrt_three := Real.sqrt 3
  have height := (half_diagonal * sin_angle) / sqrt_three
  sorry

end height_of_square_pyramid_is_13_l197_197770


namespace probability_team_B_wins_third_game_l197_197887

theorem probability_team_B_wins_third_game :
  ∀ (A B : ℕ → Prop),
    (∀ n, A n ∨ B n) ∧ -- Each game is won by either A or B
    (∀ n, A n ↔ ¬ B n) ∧ -- No ties, outcomes are independent
    (A 0) ∧ -- Team A wins the first game
    (B 1) ∧ -- Team B wins the second game
    (∃ n1 n2 n3, A n1 ∧ A n2 ∧ A n3 ∧ n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3) -- Team A wins three games
    → (∃ S, ((A 0) ∧ (B 1) ∧ (B 2)) ↔ (S = 1/3)) := sorry

end probability_team_B_wins_third_game_l197_197887


namespace simplify_expression_l197_197131

theorem simplify_expression : 
  (1 / ((1 / (1 / 3)^1) + (1 / (1 / 3)^2) + (1 / (1 / 3)^3))) = 1 / 39 :=
by
  sorry

end simplify_expression_l197_197131


namespace geom_series_sum_l197_197482

theorem geom_series_sum : 
  let a := 1 : ℝ
  let r := 1 / 4 : ℝ
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geom_series_sum_l197_197482


namespace pythagorean_triple_example_l197_197160

noncomputable def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_example :
  is_pythagorean_triple 5 12 13 :=
by
  sorry

end pythagorean_triple_example_l197_197160


namespace ivy_collectors_edition_dolls_l197_197643

-- Definitions from the conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def collectors_edition_dolls : ℕ := (2 * ivy_dolls) / 3

-- Assertion
theorem ivy_collectors_edition_dolls : collectors_edition_dolls = 20 := by
  sorry

end ivy_collectors_edition_dolls_l197_197643


namespace number_of_positive_expressions_l197_197803

-- Define the conditions
variable (a b c : ℝ)
variable (h_a : a < 0)
variable (h_b : b > 0)
variable (h_c : c < 0)

-- Define the expressions
def ab := a * b
def ac := a * c
def a_b_c := a + b + c
def a_minus_b_c := a - b + c
def two_a_plus_b := 2 * a + b
def two_a_minus_b := 2 * a - b

-- Problem statement
theorem number_of_positive_expressions :
  (ab < 0) → (ac > 0) → (a_b_c > 0) → (a_minus_b_c < 0) → (two_a_plus_b < 0) → (two_a_minus_b < 0)
  → (2 = 2) :=
by
  sorry

end number_of_positive_expressions_l197_197803


namespace geometric_series_sum_eq_4_over_3_l197_197481

theorem geometric_series_sum_eq_4_over_3 : 
  let a := 1
  let r := 1/4
  inf_geometric_sum a r = 4/3 := by
begin
  intros,
  sorry
end

end geometric_series_sum_eq_4_over_3_l197_197481


namespace change_from_15_dollars_l197_197951

theorem change_from_15_dollars :
  let cost_eggs := 3
  let cost_pancakes := 2
  let cost_mugs_of_cocoa := 2 * 2
  let tax := 1
  let initial_cost := cost_eggs + cost_pancakes + cost_mugs_of_cocoa + tax
  let additional_pancakes := 2
  let additional_mug_of_cocoa := 2
  let additional_cost := additional_pancakes + additional_mug_of_cocoa
  let new_total_cost := initial_cost + additional_cost
  let payment := 15
  let change := payment - new_total_cost
  change = 1 :=
by
  sorry

end change_from_15_dollars_l197_197951


namespace trigonometric_identity_l197_197083

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
by 
  sorry

end trigonometric_identity_l197_197083


namespace pythagorean_theorem_l197_197827

theorem pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by
  sorry

end pythagorean_theorem_l197_197827


namespace non_negative_combined_quadratic_l197_197589

theorem non_negative_combined_quadratic (a b c A B C : ℝ) (h1 : a ≥ 0) (h2 : b^2 ≤ a * c) (h3 : A ≥ 0) (h4 : B^2 ≤ A * C) :
  ∀ x : ℝ, a * A * x^2 + 2 * b * B * x + c * C ≥ 0 :=
by
  sorry

end non_negative_combined_quadratic_l197_197589


namespace yvonnes_probability_l197_197297

open Classical

variables (P_X P_Y P_Z : ℝ)

theorem yvonnes_probability
  (h1 : P_X = 1/5)
  (h2 : P_Z = 5/8)
  (h3 : P_X * P_Y * (1 - P_Z) = 0.0375) :
  P_Y = 0.5 :=
by
  sorry

end yvonnes_probability_l197_197297


namespace fraction_of_even_integers_divisible_by_4_l197_197640

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem fraction_of_even_integers_divisible_by_4 :
  let candidates := (multiset.range (151 - 50)).map (λ x, x + 50)
  let evens := candidates.filter (λ n, n % 2 = 0 ∧ sum_of_digits n = 12)
  let divisible_by_4 := evens.filter (λ n, n % 4 = 0)
  (divisible_by_4.card : ℚ) / (evens.card : ℚ) = 2 / 5 :=
by
  sorry

end fraction_of_even_integers_divisible_by_4_l197_197640


namespace find_a3_l197_197981

noncomputable def S (n : ℕ) (a₁ q : ℚ) : ℚ :=
  a₁ * (1 - q ^ n) / (1 - q)

noncomputable def a (n : ℕ) (a₁ q : ℚ) : ℚ :=
  a₁ * q ^ (n - 1)

theorem find_a3 (a₁ q : ℚ) (h1 : S 6 a₁ q / S 3 a₁ q = -19 / 8)
  (h2 : a 4 a₁ q - a 2 a₁ q = -15 / 8) :
  a 3 a₁ q = 9 / 4 :=
by sorry

end find_a3_l197_197981


namespace g_1987_l197_197811

def g (x : ℕ) : ℚ := sorry

axiom g_defined_for_all (x : ℕ) : true

axiom g1 : g 1 = 1

axiom g_rec (a b : ℕ) : g (a + b) = g a + g b - 3 * g (a * b) + 1

theorem g_1987 : g 1987 = 2 := sorry

end g_1987_l197_197811


namespace framing_feet_required_l197_197757

noncomputable def original_width := 5
noncomputable def original_height := 7
noncomputable def enlargement_factor := 4
noncomputable def border_width := 3
noncomputable def inches_per_foot := 12

theorem framing_feet_required :
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (final_width + final_height)
  let framing_feet := perimeter / inches_per_foot
  framing_feet = 10 :=
by
  sorry

end framing_feet_required_l197_197757


namespace definite_integral_sin8_l197_197147

-- Define the definite integral problem and the expected result in Lean.
theorem definite_integral_sin8:
  ∫ x in (Real.pi / 2)..Real.pi, (2^8 * (Real.sin x)^8) = 32 * Real.pi :=
  sorry

end definite_integral_sin8_l197_197147


namespace calculate_subtraction_l197_197165

def base9_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 81 + ((n / 10) % 10) * 9 + (n % 10)

def base6_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

theorem calculate_subtraction : base9_to_base10 324 - base6_to_base10 231 = 174 :=
  by sorry

end calculate_subtraction_l197_197165


namespace wilson_theorem_non_prime_divisibility_l197_197407

theorem wilson_theorem (p : ℕ) (h : Nat.Prime p) : p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

theorem non_prime_divisibility (p : ℕ) (h : ¬ Nat.Prime p) : ¬ p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

end wilson_theorem_non_prime_divisibility_l197_197407


namespace alyssa_games_next_year_l197_197318

/-- Alyssa went to 11 games this year -/
def games_this_year : ℕ := 11

/-- Alyssa went to 13 games last year -/
def games_last_year : ℕ := 13

/-- Alyssa will go to a total of 39 games -/
def total_games : ℕ := 39

/-- Alyssa plans to go to 15 games next year -/
theorem alyssa_games_next_year : 
  games_this_year + games_last_year <= total_games ∧
  total_games - (games_this_year + games_last_year) = 15 := by {
  sorry
}

end alyssa_games_next_year_l197_197318


namespace geometric_seq_ratio_l197_197980

theorem geometric_seq_ratio : 
  ∀ (a : ℕ → ℝ) (q : ℝ), 
    (∀ n, a (n+1) = a n * q) → 
    q > 1 → 
    a 1 + a 6 = 8 → 
    a 3 * a 4 = 12 → 
    a 2018 / a 2013 = 3 :=
by
  intros a q h_geom h_q_pos h_sum_eq h_product_eq
  sorry

end geometric_seq_ratio_l197_197980


namespace area_larger_sphere_l197_197625

variables {r1 r2 r : ℝ}
variables {A1 A2 : ℝ}

-- Declare constants for the problem
def radius_smaller_sphere : ℝ := 4 -- r1
def radius_larger_sphere : ℝ := 6  -- r2
def radius_ball : ℝ := 1           -- r
def area_smaller_sphere : ℝ := 27  -- A1

-- Given conditions
axiom radius_smaller_sphere_condition : r1 = radius_smaller_sphere
axiom radius_larger_sphere_condition : r2 = radius_larger_sphere
axiom radius_ball_condition : r = radius_ball
axiom area_smaller_sphere_condition : A1 = area_smaller_sphere

-- Statement to be proved
theorem area_larger_sphere :
  r1 = radius_smaller_sphere → r2 = radius_larger_sphere → r = radius_ball → A1 = area_smaller_sphere → A2 = 60.75 :=
by
  intros
  sorry

end area_larger_sphere_l197_197625


namespace binary_to_decimal_and_septal_l197_197042

theorem binary_to_decimal_and_septal :
  let bin : ℕ := 110101
  let dec : ℕ := 53
  let septal : ℕ := 104
  let convert_to_decimal (b : ℕ) : ℕ := 
    (b % 10) * 2^0 + ((b / 10) % 10) * 2^1 + ((b / 100) % 10) * 2^2 + 
    ((b / 1000) % 10) * 2^3 + ((b / 10000) % 10) * 2^4 + ((b / 100000) % 10) * 2^5
  let convert_to_septal (n : ℕ) : ℕ :=
    let rec aux (n : ℕ) (acc : ℕ) (place : ℕ) : ℕ :=
      if n = 0 then acc
      else aux (n / 7) (acc + (n % 7) * place) (place * 10)
    aux n 0 1
  convert_to_decimal bin = dec ∧ convert_to_septal dec = septal :=
by
  sorry

end binary_to_decimal_and_septal_l197_197042


namespace daily_wage_of_c_l197_197008

theorem daily_wage_of_c 
  (a_days : ℕ) (b_days : ℕ) (c_days : ℕ) 
  (wage_ratio_a_b : ℚ) (wage_ratio_b_c : ℚ) 
  (total_earnings : ℚ) 
  (A : ℚ) (C : ℚ) :
  a_days = 6 →
  b_days = 9 →
  c_days = 4 →
  wage_ratio_a_b = 3 / 4 →
  wage_ratio_b_c = 4 / 5 →
  total_earnings = 1850 →
  A = 75 →
  C = 208.33 := 
sorry

end daily_wage_of_c_l197_197008


namespace unique_root_iff_l197_197895

def has_unique_solution (a : ℝ) : Prop :=
  ∃ (x : ℝ), ∀ (y : ℝ), (a * y^2 + 2 * y - 1 = 0 ↔ y = x)

theorem unique_root_iff (a : ℝ) : has_unique_solution a ↔ (a = 0 ∨ a = 1) := 
sorry

end unique_root_iff_l197_197895


namespace prime_cube_difference_l197_197885

theorem prime_cube_difference (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (eqn : p^3 - q^3 = 11 * r) : 
  (p = 13 ∧ q = 2 ∧ r = 199) :=
sorry

end prime_cube_difference_l197_197885


namespace cost_of_two_pans_is_20_l197_197841

variable (cost_of_pan : ℕ)

-- Conditions
def pots_cost := 3 * 20
def total_cost := 100
def pans_eq_cost := total_cost - pots_cost
def cost_of_pan_per_pans := pans_eq_cost / 4

-- Proof statement
theorem cost_of_two_pans_is_20 
  (h1 : pots_cost = 60)
  (h2 : total_cost = 100)
  (h3 : pans_eq_cost = total_cost - pots_cost)
  (h4 : cost_of_pan_per_pans = pans_eq_cost / 4)
  : 2 * cost_of_pan_per_pans = 20 :=
by sorry

end cost_of_two_pans_is_20_l197_197841


namespace factorial_difference_l197_197037

theorem factorial_difference : 10! - 9! = 3265920 := by
  sorry

end factorial_difference_l197_197037


namespace sum_of_consecutive_2022_l197_197667

theorem sum_of_consecutive_2022 (m n : ℕ) (h : m ≤ n - 1) (sum_eq : (n - m + 1) * (m + n) = 4044) :
  (m = 163 ∧ n = 174) ∨ (m = 504 ∧ n = 507) ∨ (m = 673 ∧ n = 675) :=
sorry

end sum_of_consecutive_2022_l197_197667


namespace lionel_initial_boxes_crackers_l197_197718

/--
Lionel went to the grocery store and bought some boxes of Graham crackers and 15 packets of Oreos. 
To make an Oreo cheesecake, Lionel needs 2 boxes of Graham crackers and 3 packets of Oreos. 
After making the maximum number of Oreo cheesecakes he can with the ingredients he bought, 
he had 4 boxes of Graham crackers left over. 

The number of boxes of Graham crackers Lionel initially bought is 14.
-/
theorem lionel_initial_boxes_crackers (G : ℕ) (h1 : G - 4 = 10) : G = 14 := 
by sorry

end lionel_initial_boxes_crackers_l197_197718


namespace geometric_sum_S12_l197_197120

theorem geometric_sum_S12 
  (S : ℕ → ℝ)
  (h_S4 : S 4 = 2) 
  (h_S8 : S 8 = 6) 
  (geom_property : ∀ n, (S (2 * n + 4) - S n) ^ 2 = S n * (S (3 * n + 4) - S (2 * n + 4))) 
  : S 12 = 14 := 
by sorry

end geometric_sum_S12_l197_197120


namespace hall_length_l197_197463

theorem hall_length
  (breadth : ℝ) (stone_length_dm stone_width_dm : ℝ) (num_stones : ℕ) (L : ℝ)
  (h_breadth : breadth = 15)
  (h_stone_length : stone_length_dm = 6)
  (h_stone_width : stone_width_dm = 5)
  (h_num_stones : num_stones = 1800)
  (h_length : L = 36) :
  let stone_length := stone_length_dm / 10
  let stone_width := stone_width_dm / 10
  let stone_area := stone_length * stone_width
  let total_area := num_stones * stone_area
  total_area / breadth = L :=
by {
  sorry
}

end hall_length_l197_197463


namespace sum_of_squares_positive_l197_197277

theorem sum_of_squares_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2 > 0) ∧ (b^2 + c^2 > 0) ∧ (c^2 + a^2 > 0) :=
by
  sorry

end sum_of_squares_positive_l197_197277


namespace part1_part2_l197_197342

noncomputable def a_seq : ℕ → ℝ
| 0       => 3
| (n + 1) => (5 * a_seq n - 4) / (2 * a_seq n - 1)

noncomputable def b_seq : ℕ → ℝ
| n => (a_seq n - 1) / (a_seq n - 2)

def geom_seq (b : ℕ → ℝ) (r : ℝ) :=
∀ n, b (n + 1) = r * b n

def sum_first_n (f : ℕ → ℝ) (n : ℕ) :=
∑ i in finset.range n, f i

theorem part1 :
  geom_seq b_seq 3 :=
by
  unfold geom_seq
  intros n
  have h := a_seq (n + 1)
  norm_num
  sorry

theorem part2 (n : ℕ) :
  sum_first_n (λ k, k * b_seq k) n = 
  (1/2 : ℝ) + (↑n - 1/2) * (3 : ℝ)^n :=
by
  sorry

end part1_part2_l197_197342


namespace point_P_distance_l197_197508

variable (a b c d x : ℝ)

-- Define the points on the line
def O := 0
def A := a
def B := b
def C := c
def D := d

-- Define the conditions for point P
def AP_PDRatio := (|a - x| / |x - d| = 2 * |b - x| / |x - c|)

theorem point_P_distance : AP_PDRatio a b c d x → b + c - a = x :=
by
  sorry

end point_P_distance_l197_197508


namespace scheduling_schemes_l197_197975

theorem scheduling_schemes (days : Finset ℕ) (A_schedule B_schedule C_schedule : Finset ℕ) 
  (A_not_mon : ¬ 0 ∈ A_schedule) (B_not_sat : ¬ 5 ∈ B_schedule)
  (A_days : A_schedule.card = 2) (B_days : B_schedule.card = 2) (C_days : C_schedule.card = 2) :
  42 = (A_schedule.product B_schedule.product C_schedule).count sorry :=
sorry

end scheduling_schemes_l197_197975


namespace radius_of_cone_l197_197801

theorem radius_of_cone (S : ℝ) (h_S: S = 9 * Real.pi) (h_net: net_is_semi_circle) :
  ∃ (r : ℝ), r = Real.sqrt 3 :=
by
  sorry

end radius_of_cone_l197_197801


namespace fred_baseball_cards_l197_197666

theorem fred_baseball_cards :
  ∀ (fred_cards_initial melanie_bought : ℕ), fred_cards_initial = 5 → melanie_bought = 3 → fred_cards_initial - melanie_bought = 2 :=
by
  intros fred_cards_initial melanie_bought h1 h2
  sorry

end fred_baseball_cards_l197_197666


namespace stickers_initial_count_l197_197405

theorem stickers_initial_count (S : ℕ) 
  (h1 : (3 / 5 : ℝ) * (2 / 3 : ℝ) * S = 54) : S = 135 := 
by
  sorry

end stickers_initial_count_l197_197405


namespace percent_relation_l197_197298

theorem percent_relation (x y z w : ℝ) (h1 : x = 1.25 * y) (h2 : y = 0.40 * z) (h3 : z = 1.10 * w) :
  (x / w) * 100 = 55 := by sorry

end percent_relation_l197_197298


namespace monotonicity_and_no_real_roots_l197_197387

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

theorem monotonicity_and_no_real_roots 
  (a : ℝ) (ha : 0 < a) : 
  (∀ x : ℝ, (0 < x ∧ x < (1 / a) → deriv (f a) x < 0) ∧ (x > (1 / a) → deriv (f a) x > 0)) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≠ 0 → a > (1 / real.exp 1))) :=
begin
  sorry
end

end monotonicity_and_no_real_roots_l197_197387


namespace monotonicity_and_range_of_a_l197_197391

noncomputable def f (a x : ℝ) : ℝ :=
  a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, x ∈ (set.Ioo 0 (1 / a)) → deriv (λ x, f a x) x < 0) ∧
  (∀ x, x ∈ (set.Ioi (1 / a)) → deriv (λ x, f a x) x > 0) ∧
  (∀ a, a > Real.exp (-1) → ∃ x : ℝ, ∀ x, f a x > 0) :=
by
  sorry

end monotonicity_and_range_of_a_l197_197391


namespace find_cost_price_of_clock_l197_197603

namespace ClockCost

variable (C : ℝ)

def cost_price_each_clock (n : ℝ) (gain1 : ℝ) (gain2 : ℝ) (uniform_gain : ℝ) (price_difference : ℝ) :=
  let selling_price1 := 40 * C * (1 + gain1)
  let selling_price2 := 50 * C * (1 + gain2)
  let uniform_selling_price := n * C * (1 + uniform_gain)
  selling_price1 + selling_price2 - uniform_selling_price = price_difference

theorem find_cost_price_of_clock (C : ℝ) (h : cost_price_each_clock C 90 0.10 0.20 0.15 40) : C = 80 :=
  sorry

end ClockCost

end find_cost_price_of_clock_l197_197603


namespace no_real_solutions_for_inequality_l197_197954

theorem no_real_solutions_for_inequality (a : ℝ) :
  ¬∃ x : ℝ, ∀ y : ℝ, |(x^2 + a*x + 2*a)| ≤ 5 → y = x :=
sorry

end no_real_solutions_for_inequality_l197_197954


namespace difference_of_lines_in_cm_l197_197177

def W : ℝ := 7.666666666666667
def B : ℝ := 3.3333333333333335
def inch_to_cm : ℝ := 2.54

theorem difference_of_lines_in_cm :
  (W * inch_to_cm) - (B * inch_to_cm) = 11.005555555555553 := 
sorry

end difference_of_lines_in_cm_l197_197177


namespace raghu_investment_l197_197442

theorem raghu_investment
  (R trishul vishal : ℝ)
  (h1 : trishul = 0.90 * R)
  (h2 : vishal = 0.99 * R)
  (h3 : R + trishul + vishal = 6647) :
  R = 2299.65 :=
by
  sorry

end raghu_investment_l197_197442


namespace number_of_factors_and_perfect_square_factors_l197_197205

open Nat

-- Define the number 1320 and its prime factorization.
def n : ℕ := 1320
def prime_factors : List (ℕ × ℕ) := [(2, 2), (3, 1), (5, 1), (11, 1)]

-- Define a function to count factors.
def count_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

-- Define a function to count perfect square factors.
def count_perfect_square_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨prime, exp⟩ => acc * (if exp % 2 == 0 then exp / 2 + 1 else 1)) 1

theorem number_of_factors_and_perfect_square_factors :
  count_factors prime_factors = 24 ∧ count_perfect_square_factors prime_factors = 2 :=
by
  sorry

end number_of_factors_and_perfect_square_factors_l197_197205


namespace sphere_volume_l197_197015

theorem sphere_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 36 * Real.pi) : (4/3) * Real.pi * r^3 = 36 * Real.pi := 
sorry

end sphere_volume_l197_197015


namespace four_by_four_increasing_matrices_l197_197040

noncomputable def count_increasing_matrices (n : ℕ) : ℕ := sorry

theorem four_by_four_increasing_matrices :
  count_increasing_matrices 4 = 320 :=
sorry

end four_by_four_increasing_matrices_l197_197040


namespace range_of_m_empty_solution_set_inequality_l197_197352

theorem range_of_m_empty_solution_set_inequality (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 ≥ 0 → false) ↔ -4 < m ∧ m < 0 := 
sorry

end range_of_m_empty_solution_set_inequality_l197_197352


namespace find_sin_value_l197_197674

variable (x : ℝ)

theorem find_sin_value (h : Real.sin (x + Real.pi / 3) = Real.sqrt 3 / 3) : 
  Real.sin (2 * Real.pi / 3 - x) = Real.sqrt 3 / 3 :=
by 
  sorry

end find_sin_value_l197_197674


namespace num_selected_from_each_teacher_probability_at_least_one_from_wang_l197_197459

-- Given conditions
def wu_questions : ℕ := 350
def wang_questions : ℕ := 700
def zhang_questions : ℕ := 1050
def total_questions : ℕ := wu_questions + wang_questions + zhang_questions
def sample_size : ℕ := 6
def sampling_ratio : ℚ := sample_size / total_questions
def wu_sample : ℚ := wu_questions * sampling_ratio
def wang_sample : ℚ := wang_questions * sampling_ratio
def zhang_sample : ℚ := zhang_questions * sampling_ratio

-- Prove that the number of selected test questions from Wu, Wang, and Zhang is 1, 2, and 3 respectively.
theorem num_selected_from_each_teacher : 
  wu_sample = 1 ∧ wang_sample = 2 ∧ zhang_sample = 3 := 
sorry

-- Possible combinations selected
def total_combinations : ℕ := 15
def favorable_combinations : ℕ := 9

-- Prove the probability that at least one of the 2 selected questions is from Wang is 3/5.
theorem probability_at_least_one_from_wang : 
  (favorable_combinations / total_combinations : ℚ) = 3/5 := 
sorry

end num_selected_from_each_teacher_probability_at_least_one_from_wang_l197_197459


namespace max_sequence_term_value_l197_197112

def a_n (n : ℕ) : ℤ := -2 * n^2 + 29 * n + 3

theorem max_sequence_term_value : ∃ n : ℕ, a_n n = 108 := 
sorry

end max_sequence_term_value_l197_197112


namespace polar_equation_C1_intersection_C2_C1_distance_l197_197830

noncomputable def parametric_to_cartesian (α : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 2 + 2 * Real.cos α ∧ y = 4 + 2 * Real.sin α

noncomputable def cartesian_to_polar (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 4)^2 = 4

noncomputable def polar_equation_of_C1 (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.cos θ - 8 * ρ * Real.sin θ + 16 = 0

noncomputable def C2_line_polar (θ : ℝ) : Prop :=
  θ = Real.pi / 4

theorem polar_equation_C1 (α : ℝ) (ρ θ : ℝ) :
  parametric_to_cartesian α →
  cartesian_to_polar (2 + 2 * Real.cos α) (4 + 2 * Real.sin α) →
  polar_equation_of_C1 ρ θ :=
by
  sorry

theorem intersection_C2_C1_distance (ρ θ : ℝ) (t1 t2 : ℝ) :
  C2_line_polar θ →
  polar_equation_of_C1 ρ θ →
  (t1 + t2 = 6 * Real.sqrt 2) ∧ (t1 * t2 = 16) →
  |t1 - t2| = 2 * Real.sqrt 2 :=
by
  sorry

end polar_equation_C1_intersection_C2_C1_distance_l197_197830


namespace sum_of_ages_is_55_l197_197591

def sum_of_ages (Y : ℕ) (interval : ℕ) (number_of_children : ℕ) : ℕ :=
  let ages := List.range number_of_children |>.map (λ i => Y + i * interval)
  ages.sum

theorem sum_of_ages_is_55 :
  sum_of_ages 7 2 5 = 55 :=
by
  sorry

end sum_of_ages_is_55_l197_197591


namespace incenter_sum_equals_one_l197_197775

noncomputable def incenter (A B C : Point) : Point := sorry -- Definition goes here

def side_length (A B C : Point) (a b c : ℝ) : Prop :=
  -- Definitions relating to side lengths go here
  sorry

theorem incenter_sum_equals_one (A B C I : Point) (a b c IA IB IC : ℝ) (h_incenter : I = incenter A B C)
    (h_sides : side_length A B C a b c) :
    (IA ^ 2 / (b * c)) + (IB ^ 2 / (a * c)) + (IC ^ 2 / (a * b)) = 1 :=
  sorry

end incenter_sum_equals_one_l197_197775


namespace min_value_fraction_l197_197190

theorem min_value_fraction (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  ∃ m, (∀ z, z = (1 / (x + 1) + 1 / y) → z ≥ m) ∧ m = (3 + 2 * Real.sqrt 2) / 2 :=
by
  sorry

end min_value_fraction_l197_197190


namespace complement_of_M_l197_197226

open Set

def U : Set ℝ := univ
def M : Set ℝ := { x | x^2 - 2 * x > 0 }
def comp_M_Real := compl M

theorem complement_of_M :
  comp_M_Real = { x : ℝ | 0 ≤ x ∧ x ≤ 2 } :=
sorry

end complement_of_M_l197_197226


namespace determine_a1_a2_a3_l197_197816

theorem determine_a1_a2_a3 (a a1 a2 a3 : ℝ)
  (h : ∀ x : ℝ, x^3 = a + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3) :
  a1 + a2 + a3 = 19 :=
by
  sorry

end determine_a1_a2_a3_l197_197816


namespace least_number_subtracted_l197_197750

theorem least_number_subtracted (x : ℤ) (N : ℤ) :
  N = 2590 - x →
  (N % 9 = 6) →
  (N % 11 = 6) →
  (N % 13 = 6) →
  x = 10 :=
by
  sorry

end least_number_subtracted_l197_197750


namespace puzzle_pieces_left_l197_197719

theorem puzzle_pieces_left (total_pieces : ℕ) (pieces_each : ℕ) (R_pieces : ℕ) (Rhys_pieces : ℕ) (Rory_pieces : ℕ)
  (h1 : total_pieces = 300)
  (h2 : pieces_each = total_pieces / 3)
  (h3 : R_pieces = 25)
  (h4 : Rhys_pieces = 2 * R_pieces)
  (h5 : Rory_pieces = 3 * R_pieces) :
  total_pieces - (R_pieces + Rhys_pieces + Rory_pieces) = 150 :=
begin
  sorry
end

end puzzle_pieces_left_l197_197719


namespace tower_height_proof_l197_197731

-- Definitions corresponding to given conditions
def elev_angle_A : ℝ := 45
def distance_AD : ℝ := 129
def elev_angle_D : ℝ := 60
def tower_height : ℝ := 305

-- Proving the height of Liaoning Broadcasting and Television Tower
theorem tower_height_proof (h : ℝ) (AC CD : ℝ) (h_eq_AC : h = AC) (h_eq_CD_sqrt3 : h = CD * (Real.sqrt 3)) (AC_CD_sum : AC + CD = 129) :
  h = 305 :=
by
  sorry

end tower_height_proof_l197_197731


namespace monotonicity_and_range_of_a_l197_197392

noncomputable def f (a x : ℝ) : ℝ :=
  a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, x ∈ (set.Ioo 0 (1 / a)) → deriv (λ x, f a x) x < 0) ∧
  (∀ x, x ∈ (set.Ioi (1 / a)) → deriv (λ x, f a x) x > 0) ∧
  (∀ a, a > Real.exp (-1) → ∃ x : ℝ, ∀ x, f a x > 0) :=
by
  sorry

end monotonicity_and_range_of_a_l197_197392


namespace sqrt_subtraction_l197_197294

theorem sqrt_subtraction : 
  let a := 49 + 81,
      b := 36 - 25
  in (Real.sqrt a - Real.sqrt b = Real.sqrt 130 - Real.sqrt 11) := by
  sorry

end sqrt_subtraction_l197_197294


namespace given_even_function_and_monotonic_increasing_l197_197986

-- Define f as an even function on ℝ
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

-- Define that f is monotonically increasing on (-∞, 0)
def is_monotonically_increasing_on_negatives (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < 0 → f (x) < f (y)

-- Theorem statement
theorem given_even_function_and_monotonic_increasing {
  f : ℝ → ℝ
} (h_even : is_even_function f)
  (h_monotonic : is_monotonically_increasing_on_negatives f) :
  f (1) > f (-2) :=
sorry

end given_even_function_and_monotonic_increasing_l197_197986


namespace probability_sum_8_twice_l197_197600

-- Define a structure for the scenario: a 7-sided die.
structure Die7 :=
(sides : Fin 7)

-- Define a function to check if the sum of two dice equals 8.
def is_sum_8 (d1 d2 : Die7) : Prop :=
  (d1.sides.val + 1) + (d2.sides.val + 1) = 8

-- Define the probability of the event given the conditions.
def probability_event_twice (successes total_outcomes : ℕ) : ℚ :=
  (successes / total_outcomes) * (successes / total_outcomes)

-- The total number of outcomes when rolling two 7-sided dice.
def total_outcomes : ℕ := 7 * 7

-- The number of successful outcomes that yield a sum of 8 with two rolls.
def successful_outcomes : ℕ := 7

-- Main theorem statement to be proved.
theorem probability_sum_8_twice :
  probability_event_twice successful_outcomes total_outcomes = 1 / 49 :=
by
  -- Sorry to indicate that the proof is omitted.
  sorry

end probability_sum_8_twice_l197_197600


namespace scallops_per_person_l197_197457

theorem scallops_per_person 
    (scallops_per_pound : ℕ)
    (cost_per_pound : ℝ)
    (total_cost : ℝ)
    (people : ℕ)
    (total_pounds : ℝ)
    (total_scallops : ℕ)
    (scallops_per_person : ℕ)
    (h1 : scallops_per_pound = 8)
    (h2 : cost_per_pound = 24)
    (h3 : total_cost = 48)
    (h4 : people = 8)
    (h5 : total_pounds = total_cost / cost_per_pound)
    (h6 : total_scallops = scallops_per_pound * total_pounds)
    (h7 : scallops_per_person = total_scallops / people) : 
    scallops_per_person = 2 := 
by {
    sorry
}

end scallops_per_person_l197_197457


namespace tournament_teams_l197_197255

theorem tournament_teams (n : ℕ) (H : 240 = 2 * n * (n - 1)) : n = 12 := 
by sorry

end tournament_teams_l197_197255


namespace area_inequality_l197_197831

noncomputable theory

variables {A B C D K L M N : Point}
variables {ABC AKL : Triangle}
variables {S T : ℝ}

-- Conditions
axiom h₁ : is_right_triangle A B C
axiom h₂ : altitude_from A D B C
axiom h₃ : incenter_intersects ABD AB K
axiom h₄ : incenter_intersects ACD AC L
axiom h₅ : triangle_area ABC = S
axiom h₆ : triangle_area AKL = T

-- Question
theorem area_inequality : S ≥ 2 * T := sorry

end area_inequality_l197_197831


namespace range_of_a_l197_197541

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a*x^2 - a*x - 2 ≤ 0) → (-8 ≤ a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l197_197541


namespace lowest_number_of_students_l197_197006

theorem lowest_number_of_students (n : ℕ) (h1 : n % 18 = 0) (h2 : n % 24 = 0) : n = 72 := by
  sorry

end lowest_number_of_students_l197_197006


namespace range_of_m_for_roots_greater_than_1_l197_197355

theorem range_of_m_for_roots_greater_than_1:
  ∀ m : ℝ, 
  (∀ x : ℝ, 8 * x^2 - (m - 1) * x + (m - 7) = 0 → 1 < x) ↔ 25 ≤ m :=
by
  sorry

end range_of_m_for_roots_greater_than_1_l197_197355


namespace number_of_packets_l197_197242

def ounces_in_packet : ℕ := 16 * 16 + 4
def ounces_in_ton : ℕ := 2500 * 16
def gunny_bag_capacity_in_ounces : ℕ := 13 * ounces_in_ton

theorem number_of_packets : gunny_bag_capacity_in_ounces / ounces_in_packet = 2000 :=
by
  sorry

end number_of_packets_l197_197242


namespace trigonometric_identity_l197_197091

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = (2 / 5) :=
by
  sorry

end trigonometric_identity_l197_197091


namespace max_f_on_interval_l197_197738

noncomputable def f : ℝ → ℝ := λ x, (Real.sqrt 3) * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

theorem max_f_on_interval : 
  ∃ x ∈ set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 :=
begin
  -- Proof omitted
  sorry
end

end max_f_on_interval_l197_197738


namespace incorrect_parallel_m_n_l197_197988

variables {l m n : Type} [LinearOrder m] [LinearOrder n] {α β : Type}

-- Assumptions for parallelism and orthogonality
def parallel (x y : Type) : Prop := sorry
def orthogonal (x y : Type) : Prop := sorry

-- Conditions
axiom parallel_m_l : parallel m l
axiom parallel_n_l : parallel n l
axiom orthogonal_m_α : orthogonal m α
axiom parallel_m_β : parallel m β
axiom parallel_m_α : parallel m α
axiom parallel_n_α : parallel n α
axiom orthogonal_m_β : orthogonal m β
axiom orthogonal_α_β : orthogonal α β

-- The theorem to prove
theorem incorrect_parallel_m_n : parallel m α ∧ parallel n α → ¬ parallel m n := sorry

end incorrect_parallel_m_n_l197_197988


namespace range_of_a_l197_197526

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x > 2 ↔ x < 2 / (a - 1)) → a < 1 :=
by
  sorry

end range_of_a_l197_197526


namespace floor_e_eq_two_l197_197652

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 := 
sorry

end floor_e_eq_two_l197_197652


namespace inequality_proof_l197_197845

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 3 * b^3) / (5 * a + b) + (b^3 + 3 * c^3) / (5 * b + c) + (c^3 + 3 * a^3) / (5 * c + a) >= (2 / 3) * (a^2 + b^2 + c^2) := 
sorry

end inequality_proof_l197_197845


namespace total_remaining_macaroons_l197_197188

-- Define initial macaroons count
def initial_red_macaroons : ℕ := 50
def initial_green_macaroons : ℕ := 40

-- Define macaroons eaten
def eaten_green_macaroons : ℕ := 15
def eaten_red_macaroons : ℕ := 2 * eaten_green_macaroons

-- Define remaining macaroons
def remaining_red_macaroons : ℕ := initial_red_macaroons - eaten_red_macaroons
def remaining_green_macaroons : ℕ := initial_green_macaroons - eaten_green_macaroons

-- Prove the total remaining macaroons
theorem total_remaining_macaroons : remaining_red_macaroons + remaining_green_macaroons = 45 := 
by
  -- Proof omitted
  sorry

end total_remaining_macaroons_l197_197188


namespace range_of_a_for_monotonicity_l197_197424

open Real

def quadratic_function (a x : ℝ) : ℝ :=
  x^2 - 2*a*x + 1

-- Statement of the problem
theorem range_of_a_for_monotonicity :
  ∀ (a : ℝ), monotone_on (quadratic_function a) (Icc (-2) 2) ↔ a ≤ -2 :=
sorry

end range_of_a_for_monotonicity_l197_197424


namespace quadratic_value_at_point_l197_197071

theorem quadratic_value_at_point :
  ∃ a b c, 
    (∃ y, y = a * 2^2 + b * 2 + c ∧ y = 7) ∧
    (∃ y, y = a * 0^2 + b * 0 + c ∧ y = -7) ∧
    (∃ y, y = a * 5^2 + b * 5 + c ∧ y = -24.5) := 
sorry

end quadratic_value_at_point_l197_197071


namespace theo_cookies_l197_197279

theorem theo_cookies (cookies_per_time times_per_day total_cookies total_months : ℕ) (h1 : cookies_per_time = 13) (h2 : times_per_day = 3) (h3 : total_cookies = 2340) (h4 : total_months = 3) : (total_cookies / total_months) / (cookies_per_time * times_per_day) = 20 := 
by
  -- Placeholder for the proof
  sorry

end theo_cookies_l197_197279


namespace equation_satisfying_solution_l197_197740

theorem equation_satisfying_solution (x y : ℤ) :
  (x = 1 ∧ y = 4 → x + 3 * y ≠ 7) ∧
  (x = 2 ∧ y = 1 → x + 3 * y ≠ 7) ∧
  (x = -2 ∧ y = 3 → x + 3 * y = 7) ∧
  (x = 4 ∧ y = 2 → x + 3 * y ≠ 7) :=
by
  sorry

end equation_satisfying_solution_l197_197740


namespace max_probability_of_binomial_l197_197213

open ProbabilityTheory

def P (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem max_probability_of_binomial :
  ∀ k : ℕ, k ≤ 5 →
    P 5 k (1/4) ≤ P 5 1 (1/4) :=
by
  -- Proof is omitted
  sorry

end max_probability_of_binomial_l197_197213


namespace y_intercept_of_line_b_l197_197857

-- Define the conditions
def line_parallel (m1 m2 : ℝ) : Prop := m1 = m2

def point_on_line (m b x y : ℝ) : Prop := y = m * x + b

-- Given conditions
variables (m b : ℝ)
variable (x₁ := 3)
variable (y₁ := -2)
axiom parallel_condition : line_parallel m (-3)
axiom point_condition : point_on_line m b x₁ y₁

-- Prove that the y-intercept b equals 7
theorem y_intercept_of_line_b : b = 7 :=
sorry

end y_intercept_of_line_b_l197_197857


namespace orthogonal_planes_k_value_l197_197587

theorem orthogonal_planes_k_value
  (k : ℝ)
  (h : 3 * (-1) + 1 * 1 + (-2) * k = 0) : 
  k = -1 :=
sorry

end orthogonal_planes_k_value_l197_197587


namespace john_age_is_24_l197_197556

noncomputable def john_age_condition (j d b : ℕ) : Prop :=
  j = d - 28 ∧
  j + d = 76 ∧
  j + 5 = 2 * (b + 5)

theorem john_age_is_24 (d b : ℕ) : ∃ j, john_age_condition j d b ∧ j = 24 :=
by
  use 24
  unfold john_age_condition
  sorry

end john_age_is_24_l197_197556


namespace tangent_line_at_one_f_gt_one_l197_197530

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x * Real.log x + (2 * Real.exp (x - 1)) / x

theorem tangent_line_at_one : 
  let y := f 1 + (Real.exp 1) * (x - 1)
  y = Real.exp (1 : ℝ) * (x - 1) + 2 := 
sorry

theorem f_gt_one (x : ℝ) (hx : 0 < x) : f x > 1 := 
sorry

end tangent_line_at_one_f_gt_one_l197_197530


namespace total_strawberry_weight_l197_197231

def MarcosStrawberries : ℕ := 3
def DadsStrawberries : ℕ := 17

theorem total_strawberry_weight : MarcosStrawberries + DadsStrawberries = 20 := by
  sorry

end total_strawberry_weight_l197_197231


namespace ratio_of_m_div_x_l197_197607

theorem ratio_of_m_div_x (a b : ℝ) (h1 : a / b = 4 / 5) (h2 : a > 0) (h3 : b > 0) :
  let x := a + 0.25 * a
  let m := b - 0.60 * b
  (m / x) = 2 / 5 :=
by
  -- Define x and m
  let x := a + 0.25 * a
  let m := b - 0.60 * b
  -- Include the steps or assumptions here if necessary
  sorry

end ratio_of_m_div_x_l197_197607


namespace work_done_in_one_day_by_A_and_B_l197_197614

noncomputable def A_days : ℕ := 12
noncomputable def B_days : ℕ := A_days / 2

theorem work_done_in_one_day_by_A_and_B : 1 / (A_days : ℚ) + 1 / (B_days : ℚ) = 1 / 4 := by
  sorry

end work_done_in_one_day_by_A_and_B_l197_197614
