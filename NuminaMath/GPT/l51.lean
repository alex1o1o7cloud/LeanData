import Mathlib

namespace tie_rate_correct_l51_51321

-- Define the fractions indicating win rates for Amy, Lily, and John
def AmyWinRate : ℚ := 4 / 9
def LilyWinRate : ℚ := 1 / 3
def JohnWinRate : ℚ := 1 / 6

-- Define the fraction they tie
def TieRate : ℚ := 1 / 18

-- The theorem for proving the tie rate
theorem tie_rate_correct : AmyWinRate + LilyWinRate + JohnWinRate = 17 / 18 → (1 : ℚ) - (17 / 18) = TieRate :=
by
  sorry -- Proof is omitted

-- Define the win rate sums and tie rate equivalence
example : (AmyWinRate + LilyWinRate + JohnWinRate = 17 / 18) ∧ (TieRate = 1 - 17 / 18) :=
by
  sorry -- Proof is omitted

end tie_rate_correct_l51_51321


namespace merry_go_round_cost_per_child_l51_51530

-- Definitions
def num_children := 5
def ferris_wheel_cost_per_child := 5
def num_children_on_ferris_wheel := 3
def ice_cream_cost_per_cone := 8
def ice_cream_cones_per_child := 2
def total_spent := 110

-- Totals
def ferris_wheel_total_cost := num_children_on_ferris_wheel * ferris_wheel_cost_per_child
def ice_cream_total_cost := num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone
def merry_go_round_total_cost := total_spent - ferris_wheel_total_cost - ice_cream_total_cost

-- Final proof statement
theorem merry_go_round_cost_per_child : 
  merry_go_round_total_cost / num_children = 3 :=
by
  -- We skip the actual proof here
  sorry

end merry_go_round_cost_per_child_l51_51530


namespace problem_l51_51215

def a := 1 / 4
def b := 1 / 2
def c := -3 / 4

def a_n (n : ℕ) : ℚ := 2 * n + 1
def S_n (n : ℕ) : ℚ := (n + 2) * n
def f (n : ℕ) : ℚ := 4 * a * n^2 + (4 * a + 2 * b) * n + (a + b + c)

theorem problem : ∀ n : ℕ, f n = S_n n := by
  sorry

end problem_l51_51215


namespace cost_price_marked_price_ratio_l51_51689

theorem cost_price_marked_price_ratio (x : ℝ) (hx : x > 0) :
  let selling_price := (2 / 3) * x
  let cost_price := (3 / 4) * selling_price 
  cost_price / x = 1 / 2 := 
by
  let selling_price := (2 / 3) * x 
  let cost_price := (3 / 4) * selling_price 
  have hs : selling_price = (2 / 3) * x := rfl 
  have hc : cost_price = (3 / 4) * selling_price := rfl 
  have ratio := hc.symm 
  simp [ratio, hs]
  sorry

end cost_price_marked_price_ratio_l51_51689


namespace water_consumption_correct_l51_51022

theorem water_consumption_correct (w n r : ℝ) 
  (hw : w = 21428) 
  (hn : n = 26848.55) 
  (hr : r = 302790.13) :
  w = 21428 ∧ n = 26848.55 ∧ r = 302790.13 :=
by 
  sorry

end water_consumption_correct_l51_51022


namespace water_consumption_comparison_l51_51020

-- Define the given conditions
def waterConsumptionWest : ℝ := 21428
def waterConsumptionNonWest : ℝ := 26848.55
def waterConsumptionRussia : ℝ := 302790.13

-- Theorem statement to prove that the water consumption per person matches the given values
theorem water_consumption_comparison :
  waterConsumptionWest = 21428 ∧
  waterConsumptionNonWest = 26848.55 ∧
  waterConsumptionRussia = 302790.13 :=
by
  -- Sorry to skip the proof
  sorry

end water_consumption_comparison_l51_51020


namespace exists_xyz_prime_expression_l51_51250

theorem exists_xyz_prime_expression (a b c p : ℤ) (h_prime : Prime p)
    (h_div : p ∣ (a^2 + b^2 + c^2 - ab - bc - ca))
    (h_gcd : Int.gcd p ((a^2 + b^2 + c^2 - ab - bc - ca) / p) = 1) :
    ∃ x y z : ℤ, p = x^2 + y^2 + z^2 - xy - yz - zx := by
  sorry

end exists_xyz_prime_expression_l51_51250


namespace john_gym_hours_l51_51778

theorem john_gym_hours :
  (2 * (1 + 1/3)) + (2 * (1 + 1/2)) + (1.5 + 3/4) = 7.92 :=
by
  sorry

end john_gym_hours_l51_51778


namespace sum_of_squares_of_first_10_primes_l51_51715

theorem sum_of_squares_of_first_10_primes :
  ((2^2) + (3^2) + (5^2) + (7^2) + (11^2) + (13^2) + (17^2) + (19^2) + (23^2) + (29^2)) = 2397 :=
by
  sorry

end sum_of_squares_of_first_10_primes_l51_51715


namespace length_of_AB_l51_51608

noncomputable def line (t : ℝ) : ℝ × ℝ :=
  (1/2 * t, (sqrt 2) / 2 + (sqrt 3) / 2 * t)

noncomputable def curve (α : ℝ) : ℝ × ℝ :=
  (sqrt 2 / 2 + cos α, sqrt 2 / 2 + sin α)

def length_AB (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  (sqrt ((x2 - x1)^2 + (y2 - y1)^2))

theorem length_of_AB : 
  ∃ t1 t2 α1 α2, 
  line t1 = curve α1 ∧ line t2 = curve α2 ∧ 
  length_AB (line t1) (line t2) = sqrt 10 / 2 := 
sorry

end length_of_AB_l51_51608


namespace smallest_positive_integer_a_l51_51523

theorem smallest_positive_integer_a (a : ℕ) (h1 : 0 < a) (h2 : ∃ b : ℕ, 3150 * a = b^2) : a = 14 :=
by
  sorry

end smallest_positive_integer_a_l51_51523


namespace true_propositions_among_converse_inverse_contrapositive_l51_51821

theorem true_propositions_among_converse_inverse_contrapositive
  (x : ℝ)
  (h1 : x^2 ≥ 1 → x ≥ 1) :
  (if x ≥ 1 then x^2 ≥ 1 else true) ∧ 
  (if x^2 < 1 then x < 1 else true) ∧ 
  (if x < 1 then x^2 < 1 else true) → 
  ∃ n, n = 2 :=
by sorry

end true_propositions_among_converse_inverse_contrapositive_l51_51821


namespace weight_shifted_count_l51_51406

def is_weight_shifted (a b x y : ℕ) : Prop :=
  a + b = 2 * (x + y) ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9

theorem weight_shifted_count : 
  ∃ count : ℕ, count = 225 ∧ 
  (∀ (a b x y : ℕ), is_weight_shifted a b x y → count = 225) := 
sorry

end weight_shifted_count_l51_51406


namespace max_value_g_f_less_than_e_x_div_x_sq_l51_51573

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem max_value_g : ∃ x, x = 3 ∧ g x = 2 * Real.log 2 - 7 / 4 := by
  sorry

theorem f_less_than_e_x_div_x_sq (x : ℝ) (hx : x > 0) : f x < (Real.exp x - 1) / x ^ 2 := by
  sorry

end max_value_g_f_less_than_e_x_div_x_sq_l51_51573


namespace dot_product_EC_ED_l51_51768

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l51_51768


namespace red_pairs_count_l51_51320

theorem red_pairs_count (blue_shirts red_shirts total_pairs blue_blue_pairs : ℕ)
  (h1 : blue_shirts = 63) 
  (h2 : red_shirts = 81) 
  (h3 : total_pairs = 72) 
  (h4 : blue_blue_pairs = 21)
  : (red_shirts - (blue_shirts - blue_blue_pairs * 2)) / 2 = 30 :=
by
  sorry

end red_pairs_count_l51_51320


namespace smallest_n_for_isosceles_trapezoid_coloring_l51_51887

def isIsoscelesTrapezoid (a b c d : ℕ) : Prop :=
  -- conditions to check if vertices a, b, c, d form an isosceles trapezoid in a regular n-gon
  sorry  -- definition of an isosceles trapezoid

def vertexColors (n : ℕ) : Fin n → Fin 3 :=
  sorry  -- vertex coloring function

theorem smallest_n_for_isosceles_trapezoid_coloring :
  ∃ n : ℕ, (∀ (vertices : Fin n → Fin 3), ∃ (a b c d : Fin n),
    vertexColors n a = vertexColors n b ∧
    vertexColors n b = vertexColors n c ∧
    vertexColors n c = vertexColors n d ∧
    isIsoscelesTrapezoid a b c d) ∧ n = 17 :=
by
  sorry

end smallest_n_for_isosceles_trapezoid_coloring_l51_51887


namespace evaluate_ceiling_expression_l51_51282

theorem evaluate_ceiling_expression:
  (Int.ceil ((23 : ℚ) / 9 - Int.ceil ((35 : ℚ) / 23)))
  / (Int.ceil ((35 : ℚ) / 9 + Int.ceil ((9 * 23 : ℚ) / 35))) = 1 / 12 := by
  sorry

end evaluate_ceiling_expression_l51_51282


namespace tonya_hamburgers_to_beat_winner_l51_51375

-- Given conditions
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Calculate the number of hamburgers eaten last year
def hamburgers_eaten_last_year : ℕ := ounces_eaten_last_year / ounces_per_hamburger

-- Prove the number of hamburgers Tonya needs to eat to beat last year's winner
theorem tonya_hamburgers_to_beat_winner : 
  hamburgers_eaten_last_year + 1 = 22 :=
by
  -- It remains to be proven
  sorry

end tonya_hamburgers_to_beat_winner_l51_51375


namespace roof_area_l51_51239

theorem roof_area (w l : ℕ) (h1 : l = 4 * w) (h2 : l - w = 42) : l * w = 784 :=
by
  sorry

end roof_area_l51_51239


namespace sine_of_angle_from_point_l51_51303

theorem sine_of_angle_from_point (x y : ℤ) (r : ℝ) (h : r = Real.sqrt ((x : ℝ)^2 + (y : ℝ)^2)) (hx : x = -12) (hy : y = 5) :
  Real.sin (Real.arctan (y / x)) = y / r := 
by
  sorry

end sine_of_angle_from_point_l51_51303


namespace stream_speed_l51_51846

theorem stream_speed (v : ℝ) : (24 + v) = 168 / 6 → v = 4 :=
by
  intro h
  sorry

end stream_speed_l51_51846


namespace count_three_digit_numbers_divisible_by_13_l51_51149

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l51_51149


namespace count_3_digit_numbers_divisible_by_13_l51_51061

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l51_51061


namespace range_of_2x_plus_y_l51_51051

-- Given that positive numbers x and y satisfy this equation:
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x + y + 4 * x * y = 15 / 2

-- Define the range for 2x + y
def range_2x_plus_y (x y : ℝ) : ℝ :=
  2 * x + y

-- State the theorem.
theorem range_of_2x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : satisfies_equation x y) :
  3 ≤ range_2x_plus_y x y :=
by
  sorry

end range_of_2x_plus_y_l51_51051


namespace largest_prime_factor_of_1729_is_19_l51_51999

theorem largest_prime_factor_of_1729_is_19 :
  ∃ p : ℕ, prime p ∧ p ∣ 1729 ∧ (∀ q : ℕ, prime q ∧ q ∣ 1729 → q ≤ p) :=
sorry

end largest_prime_factor_of_1729_is_19_l51_51999


namespace find_m_l51_51908

theorem find_m (m x_1 x_2 : ℝ) 
  (h1 : x_1^2 + m * x_1 - 3 = 0) 
  (h2 : x_2^2 + m * x_2 - 3 = 0) 
  (h3 : x_1 + x_2 - x_1 * x_2 = 5) : 
  m = -2 :=
sorry

end find_m_l51_51908


namespace correct_LCM_of_fractions_l51_51669

noncomputable def lcm_of_fractions : ℚ :=
  let denominators := [10, 9, 8, 12] in
  let numerators := [7, 8, 3, 5] in
  let lcm_denominators := denominators.foldl lcm 1 in
  let gcd_numerators := numerators.foldl gcd 0 in
  (lcm_denominators : ℚ) / gcd_numerators

theorem correct_LCM_of_fractions :
  lcm_of_fractions = 360 := by sorry

end correct_LCM_of_fractions_l51_51669


namespace complex_fraction_eval_l51_51615

theorem complex_fraction_eval (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + a * b + b^2 = 0) :
  (a^15 + b^15) / (a + b)^15 = -2 := by
sorry

end complex_fraction_eval_l51_51615


namespace markese_earned_16_l51_51793

def evan_earnings (E : ℕ) : Prop :=
  (E : ℕ)

def markese_earnings (M : ℕ) (E : ℕ) : Prop :=
  (M : ℕ) = E - 5

def total_earnings (E M : ℕ) : Prop :=
  E + M = 37

theorem markese_earned_16 (E : ℕ) (M : ℕ) 
  (h1 : markese_earnings M E) 
  (h2 : total_earnings E M) : M = 16 :=
sorry

end markese_earned_16_l51_51793


namespace k_9_pow_4_eq_81_l51_51499

theorem k_9_pow_4_eq_81 
  (h k : ℝ → ℝ) 
  (hk1 : ∀ (x : ℝ), x ≥ 1 → h (k x) = x^3) 
  (hk2 : ∀ (x : ℝ), x ≥ 1 → k (h x) = x^4) 
  (k81_eq_9 : k 81 = 9) :
  (k 9)^4 = 81 :=
by
  sorry

end k_9_pow_4_eq_81_l51_51499


namespace num_three_digit_div_by_13_l51_51163

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l51_51163


namespace max_d_value_l51_51285

theorem max_d_value : ∀ (d e : ℕ), (d < 10) → (e < 10) → (5 * 10^5 + d * 10^4 + 5 * 10^3 + 2 * 10^2 + 2 * 10 + e ≡ 0 [MOD 22]) → (e % 2 = 0) → (d + e = 10) → d ≤ 8 :=
by
  intros d e h1 h2 h3 h4 h5
  sorry

end max_d_value_l51_51285


namespace arithmetic_seq_max_n_l51_51179

def arithmetic_seq_max_sum (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 > 0) ∧ (3 * (a 1 + 4 * d) = 5 * (a 1 + 7 * d)) ∧
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) ∧
  (S 12 = -72 * d)

theorem arithmetic_seq_max_n
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : 
  arithmetic_seq_max_sum a d S → n = 12 :=
by
  sorry

end arithmetic_seq_max_n_l51_51179


namespace can_adjust_to_357_l51_51973

structure Ratio (L O V : ℕ) :=
(lemon : ℕ)
(oil : ℕ)
(vinegar : ℕ)

def MixA : Ratio 1 2 3 := ⟨1, 2, 3⟩
def MixB : Ratio 3 4 5 := ⟨3, 4, 5⟩
def TargetC : Ratio 3 5 7 := ⟨3, 5, 7⟩

theorem can_adjust_to_357 (x y : ℕ) (hA : x * MixA.lemon + y * MixB.lemon = 3 * (x + y))
    (hO : x * MixA.oil + y * MixB.oil = 5 * (x + y))
    (hV : x * MixA.vinegar + y * MixB.vinegar = 7 * (x + y)) :
    (∃ a b : ℕ, x = 3 * a ∧ y = 2 * b) :=
sorry

end can_adjust_to_357_l51_51973


namespace find_integer_x_l51_51265

theorem find_integer_x (x : ℤ) :
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  0 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 
  → x = 3 :=
by
  intros h
  sorry

end find_integer_x_l51_51265


namespace willie_currency_exchange_l51_51390

theorem willie_currency_exchange :
  let euro_amount := 70
  let pound_amount := 50
  let franc_amount := 30

  let euro_to_dollar := 1.2
  let pound_to_dollar := 1.5
  let franc_to_dollar := 1.1

  let airport_euro_rate := 5 / 7
  let airport_pound_rate := 3 / 4
  let airport_franc_rate := 9 / 10

  let flat_fee := 5

  let official_euro_dollars := euro_amount * euro_to_dollar
  let official_pound_dollars := pound_amount * pound_to_dollar
  let official_franc_dollars := franc_amount * franc_to_dollar

  let airport_euro_dollars := official_euro_dollars * airport_euro_rate
  let airport_pound_dollars := official_pound_dollars * airport_pound_rate
  let airport_franc_dollars := official_franc_dollars * airport_franc_rate

  let final_euro_dollars := airport_euro_dollars - flat_fee
  let final_pound_dollars := airport_pound_dollars - flat_fee
  let final_franc_dollars := airport_franc_dollars - flat_fee

  let total_dollars := final_euro_dollars + final_pound_dollars + final_franc_dollars

  total_dollars = 130.95 :=
by
  sorry

end willie_currency_exchange_l51_51390


namespace trig_identity_l51_51458

theorem trig_identity
  (x : ℝ)
  (h : Real.tan (π / 4 + x) = 2014) :
  1 / Real.cos (2 * x) + Real.tan (2 * x) = 2014 :=
by
  sorry

end trig_identity_l51_51458


namespace number_of_cities_l51_51371

theorem number_of_cities (n : ℕ) (h : n * (n - 1) / 2 = 15) : n = 6 :=
sorry

end number_of_cities_l51_51371


namespace remainder_divisor_l51_51747

theorem remainder_divisor (d r : ℤ) (h1 : d > 1) 
  (h2 : 2024 % d = r) (h3 : 3250 % d = r) (h4 : 4330 % d = r) : d - r = 2 := 
by
  sorry

end remainder_divisor_l51_51747


namespace probability_of_singing_on_Saturday_l51_51462

variable (P : Prop → ℝ)

-- Conditions
axiom given_data :
  P (¬(sings_on_saturday)) → sings_on_sunday = 0.7 ∧
  P sings_on_sunday = 0.5 ∧
  P (sings_on_saturday → ¬sings_on_sunday) = 1

-- Theorem: Find the probability that Alex sings on Saturday
theorem probability_of_singing_on_Saturday
  (h1 : P (¬(sings_on_saturday)) → sings_on_sunday = 0.7)
  (h2 : P sings_on_sunday = 0.5)
  (h3 : P (¬sings_on_saturday ∧ sings_on_sunday)) :
  P (sings_on_saturday) = 2/7 := 
sorry

end probability_of_singing_on_Saturday_l51_51462


namespace min_distance_PM_l51_51463

-- Define the line l1
def l1 (P : Point) : Prop := P.x + P.y + 3 = 0

-- Define the curve C (circle)
def C (M : Point) : Prop := (M.x - 5)^2 + M.y^2 = 16

-- Given conditions
theorem min_distance_PM :
  ∃ P M : Point, l1 P ∧ C M ∧ (∃ l2 : Line, line_passing_through l2 P ∧ tangent_to l2 C M) →
  min_value (dist P M) = 4 := by
  sorry

end min_distance_PM_l51_51463


namespace inequality_solution_set_l51_51431

noncomputable def solution_set := { x : ℝ | 0 < x ∧ x < 2 }

theorem inequality_solution_set : 
  { x : ℝ | (4 / x > |x|) } = solution_set :=
by sorry

end inequality_solution_set_l51_51431


namespace count_3_digit_multiples_of_13_l51_51094

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l51_51094


namespace min_positive_integer_expression_l51_51714

theorem min_positive_integer_expression : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (m: ℝ) / 3 + 27 / (m: ℝ) ≥ (n: ℝ) / 3 + 27 / (n: ℝ)) ∧ (n / 3 + 27 / n = 6) :=
sorry

end min_positive_integer_expression_l51_51714


namespace find_b_l51_51671

noncomputable def f (x : ℝ) : ℝ := -1 / x

theorem find_b (a b : ℝ) (h1 : f a = -1 / 3) (h2 : f (a * b) = 1 / 6) : b = -2 := 
by
  sorry

end find_b_l51_51671


namespace smallest_angle_pentagon_l51_51235

theorem smallest_angle_pentagon (x : ℝ) (h : 16 * x = 540) : 2 * x = 67.5 := 
by 
  sorry

end smallest_angle_pentagon_l51_51235


namespace max_power_speed_l51_51816

def aerodynamic_force (C S ρ v₀ v : ℝ) : ℝ :=
  (C * S * ρ * (v₀ - v)^2) / 2

def power (C S ρ v₀ v : ℝ) : ℝ :=
  aerodynamic_force C S ρ v₀ v * v

theorem max_power_speed (C S ρ v₀ v : ℝ) (h₁ : v = v₀ / 3) :
  ∃ v, power C S ρ v₀ v = (C * S * ρ * v₀^3) / 54 :=
begin
  use v₀ / 3,
  sorry
end

end max_power_speed_l51_51816


namespace triangle_third_side_l51_51347

theorem triangle_third_side (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = 3 * C) (h2 : b = 6) (h3 : c = 18) 
  (law_of_cosines : cos C = (a^2 + c^2 - b^2) / (2 * a * c))
  (law_of_sines : sin C = (3 * sin C - 4 * (sin C)^3)) :
  a = 72 := 
sorry

end triangle_third_side_l51_51347


namespace find_first_number_l51_51521

variable (x y : ℕ)

theorem find_first_number (h1 : y = 11) (h2 : x + (y + 3) = 19) : x = 5 :=
by
  sorry

end find_first_number_l51_51521


namespace number_of_three_digit_numbers_divisible_by_13_l51_51119

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l51_51119


namespace mary_fruits_left_l51_51342

-- Conditions as definitions:
def mary_bought_apples : ℕ := 14
def mary_bought_oranges : ℕ := 9
def mary_bought_blueberries : ℕ := 6

def mary_ate_apples : ℕ := 1
def mary_ate_oranges : ℕ := 1
def mary_ate_blueberries : ℕ := 1

-- The problem statement:
theorem mary_fruits_left : 
  (mary_bought_apples - mary_ate_apples) + 
  (mary_bought_oranges - mary_ate_oranges) + 
  (mary_bought_blueberries - mary_ate_blueberries) = 26 := by
  sorry

end mary_fruits_left_l51_51342


namespace speed_of_man_l51_51400

theorem speed_of_man 
  (L : ℝ) 
  (V_t : ℝ) 
  (T : ℝ) 
  (conversion_factor : ℝ) 
  (kmph_to_mps : ℝ → ℝ)
  (final_conversion : ℝ → ℝ) 
  (relative_speed : ℝ) 
  (Vm : ℝ) : Prop := 
L = 220 ∧ V_t = 59 ∧ T = 12 ∧ 
conversion_factor = 1000 / 3600 ∧ 
kmph_to_mps V_t = V_t * conversion_factor ∧ 
relative_speed = L / T ∧ 
Vm = relative_speed - (kmph_to_mps V_t) ∧ 
final_conversion Vm = Vm * 3.6 ∧ 
final_conversion Vm = 6.984

end speed_of_man_l51_51400


namespace number_of_three_digit_numbers_divisible_by_13_l51_51121

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l51_51121


namespace not_divisible_by_2003_l51_51647

def seq_a : ℕ → ℕ 
| 0       := 1
| (n + 1) := (seq_a n) ^ 2001 + (seq_b n)
and seq_b : ℕ → ℕ
| 0       := 4
| (n + 1) := (seq_b n) ^ 2001 + (seq_a n)

theorem not_divisible_by_2003 (n : ℕ) : ¬(2003 ∣ seq_a n) ∧ ¬(2003 ∣ seq_b n) :=
sorry

end not_divisible_by_2003_l51_51647


namespace count_3_digit_numbers_divisible_by_13_l51_51067

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51067


namespace three_digit_numbers_divisible_by_13_count_l51_51130

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l51_51130


namespace certain_number_l51_51674

theorem certain_number (x : ℝ) (h : (2.28 * x) / 6 = 480.7) : x = 1265.0 := 
by 
  sorry

end certain_number_l51_51674


namespace johns_final_push_time_l51_51188

-- Definitions and assumptions
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.8
def initial_gap : ℝ := 15
def final_gap : ℝ := 2

theorem johns_final_push_time :
  ∃ t : ℝ, john_speed * t = steve_speed * t + initial_gap + final_gap ∧ t = 42.5 :=
by
  sorry

end johns_final_push_time_l51_51188


namespace solution_set_of_inequality_l51_51503

theorem solution_set_of_inequality (x : ℝ) : |5 * x - x^2| < 6 ↔ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) :=
sorry

end solution_set_of_inequality_l51_51503


namespace pool_water_after_eight_hours_l51_51422

-- Define the conditions
def hour1_fill_rate := 8
def hour2_and_hour3_fill_rate := 10
def hour4_and_hour5_fill_rate := 14
def hour6_fill_rate := 12
def hour7_fill_rate := 12
def hour8_fill_rate := 12
def hour7_leak := -8
def hour8_leak := -5

-- Calculate the water added in each time period
def water_added := hour1_fill_rate +
                   (hour2_and_hour3_fill_rate * 2) +
                   (hour4_and_hour5_fill_rate * 2) +
                   (hour6_fill_rate + hour7_fill_rate + hour8_fill_rate)

-- Calculate the water lost due to leaks
def water_lost := hour7_leak + hour8_leak  -- Note: Leaks are already negative

-- The final calculation: total water added minus total water lost
def final_water := water_added + water_lost

theorem pool_water_after_eight_hours : final_water = 79 :=
by {
  -- proof steps to check equality are omitted here
  sorry
}

end pool_water_after_eight_hours_l51_51422


namespace system_of_inequalities_solution_l51_51036

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end system_of_inequalities_solution_l51_51036


namespace max_value_of_expression_l51_51356

noncomputable def maximum_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (h : x^2 - x*y + 2*y^2 = 8) : ℝ :=
  x^2 + x*y + 2*y^2

theorem max_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x^2 - x*y + 2*y^2 = 8) : maximum_value hx hy h = (72 + 32 * Real.sqrt 2) / 7 :=
by
  sorry

end max_value_of_expression_l51_51356


namespace passing_probability_l51_51322

def probability_of_passing (p : ℝ) : ℝ :=
  p^3 + p^2 * (1 - p) + (1 - p) * p^2

theorem passing_probability :
  probability_of_passing 0.6 = 0.504 :=
by {
  sorry
}

end passing_probability_l51_51322


namespace find_distance_l51_51311

theorem find_distance (T D : ℝ) 
  (h1 : D = 5 * (T + 0.2)) 
  (h2 : D = 6 * (T - 0.25)) : 
  D = 13.5 :=
by
  sorry

end find_distance_l51_51311


namespace find_ratio_l51_51468

noncomputable def decagon_area : ℝ := 12
noncomputable def area_below_PQ : ℝ := 6
noncomputable def unit_square_area : ℝ := 1
noncomputable def triangle_base : ℝ := 6
noncomputable def area_above_PQ : ℝ := 6
noncomputable def XQ : ℝ := 4
noncomputable def QY : ℝ := 2

theorem find_ratio {XQ QY : ℝ} (h1 : decagon_area = 12) (h2 : area_below_PQ = 6)
                   (h3 : unit_square_area = 1) (h4 : triangle_base = 6)
                   (h5 : area_above_PQ = 6) (h6 : XQ + QY = 6) :
  XQ / QY = 2 := by { sorry }

end find_ratio_l51_51468


namespace verify_mass_percentage_l51_51882

-- Define the elements in HBrO3
def hydrogen : String := "H"
def bromine : String := "Br"
def oxygen : String := "O"

-- Define the given molar masses
def molar_masses (e : String) : Float :=
  if e = hydrogen then 1.01
  else if e = bromine then 79.90
  else if e = oxygen then 16.00
  else 0.0

-- Define the molar mass of HBrO3
def molar_mass_HBrO3 : Float := 128.91

-- Function to calculate mass percentage of a given element in HBrO3
def mass_percentage (e : String) : Float :=
  if e = bromine then 79.90 / molar_mass_HBrO3 * 100
  else if e = hydrogen then 1.01 / molar_mass_HBrO3 * 100
  else if e = oxygen then 48.00 / molar_mass_HBrO3 * 100
  else 0.0

-- The proof problem statement
theorem verify_mass_percentage (e : String) (h : e ∈ [hydrogen, bromine, oxygen]) : mass_percentage e = 0.78 :=
sorry

end verify_mass_percentage_l51_51882


namespace count100DigitEvenNumbers_is_correct_l51_51918

noncomputable def count100DigitEvenNumbers : ℕ :=
  let validDigits : Finset ℕ := {0, 1, 3}
  let firstDigitChoices : ℕ := 2  -- Only 1 or 3
  let middleDigitsChoices : ℕ := 3 ^ 98  -- 3 choices for each of the 98 middle positions
  let lastDigitChoices : ℕ := 1  -- Only 0 (even number requirement)
  firstDigitChoices * middleDigitsChoices * lastDigitChoices

theorem count100DigitEvenNumbers_is_correct :
  count100DigitEvenNumbers = 2 * 3 ^ 98 := by
  sorry

end count100DigitEvenNumbers_is_correct_l51_51918


namespace problem_statement_l51_51902

variable (x : ℝ)
def A := ({-3, x^2, x + 1} : Set ℝ)
def B := ({x - 3, 2 * x - 1, x^2 + 1} : Set ℝ)

theorem problem_statement (hx : A x ∩ B x = {-3}) : 
  x = -1 ∧ A x ∪ B x = ({-4, -3, 0, 1, 2} : Set ℝ) :=
by
  sorry

end problem_statement_l51_51902


namespace base_8_not_divisible_by_five_l51_51892

def base_b_subtraction_not_divisible_by_five (b : ℕ) : Prop :=
  let num1 := 3 * b^3 + 1 * b^2 + 0 * b + 2
  let num2 := 3 * b^2 + 0 * b + 2
  let diff := num1 - num2
  ¬ (diff % 5 = 0)

theorem base_8_not_divisible_by_five : base_b_subtraction_not_divisible_by_five 8 := 
by
  sorry

end base_8_not_divisible_by_five_l51_51892


namespace roots_of_polynomial_l51_51286

def poly (x : ℝ) : ℝ := x^3 - 3 * x^2 - 4 * x + 12

theorem roots_of_polynomial : 
  (poly 2 = 0) ∧ (poly (-2) = 0) ∧ (poly 3 = 0) ∧ 
  (∀ x, poly x = 0 → x = 2 ∨ x = -2 ∨ x = 3) :=
by
  sorry

end roots_of_polynomial_l51_51286


namespace Amy_total_crumbs_eq_3z_l51_51244

variable (T C z : ℕ)

-- Given conditions
def total_crumbs_Arthur := T * C = z
def trips_Amy := 2 * T
def crumbs_per_trip_Amy := 3 * C / 2

-- Problem statement
theorem Amy_total_crumbs_eq_3z (h : total_crumbs_Arthur T C z) :
  (trips_Amy T) * (crumbs_per_trip_Amy C) = 3 * z :=
sorry

end Amy_total_crumbs_eq_3z_l51_51244


namespace simplify_expression_l51_51542

variable {p q r : ℚ}

theorem simplify_expression (hp : p ≠ 2) (hq : q ≠ 5) (hr : r ≠ 7) :
  ( (p - 2) / (7 - r) * (q - 5) / (2 - p) * (r - 7) / (5 - q) ) = -1 := by
  sorry

end simplify_expression_l51_51542


namespace exist_equilateral_triangle_on_parallel_lines_l51_51743

-- Define the concept of lines and points in a relation to them
def Line := ℝ → ℝ -- For simplicity, let's assume lines are functions

-- Define the points A1, A2, A3
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the concept of parallel lines
def parallel (D1 D2 : Line) : Prop :=
  ∀ x y, D1 x - D2 x = D1 y - D2 y

axiom D1 : Line
axiom D2 : Line
axiom D3 : Line

-- Ensure the lines are parallel
axiom parallel_D1_D2 : parallel D1 D2
axiom parallel_D2_D3 : parallel D2 D3

-- Main statement to prove
theorem exist_equilateral_triangle_on_parallel_lines :
  ∃ (A1 A2 A3 : Point), 
    (A1.y = D1 A1.x) ∧ 
    (A2.y = D2 A2.x) ∧ 
    (A3.y = D3 A3.x) ∧ 
    ((A1.x - A2.x)^2 + (A1.y - A2.y)^2 = (A2.x - A3.x)^2 + (A2.y - A3.y)^2) ∧ 
    ((A2.x - A3.x)^2 + (A2.y - A3.y)^2 = (A3.x - A1.x)^2 + (A3.y - A1.y)^2) := sorry

end exist_equilateral_triangle_on_parallel_lines_l51_51743


namespace unique_triangled_pair_l51_51872

theorem unique_triangled_pair (a b x y : ℝ) (h : ∀ a b : ℝ, (a, b) = (a * x + b * y, a * y + b * x)) : (x, y) = (1, 0) :=
by sorry

end unique_triangled_pair_l51_51872


namespace find_n_l51_51584

theorem find_n (a b : ℤ) (h₁ : a ≡ 25 [ZMOD 42]) (h₂ : b ≡ 63 [ZMOD 42]) :
  ∃ n, 200 ≤ n ∧ n ≤ 241 ∧ (a - b ≡ n [ZMOD 42]) ∧ n = 214 :=
by
  sorry

end find_n_l51_51584


namespace volume_ratio_l51_51306

def volume_of_cube (side_length : ℕ) : ℕ :=
  side_length * side_length * side_length

theorem volume_ratio 
  (hyungjin_side_length_cm : ℕ)
  (kyujun_side_length_m : ℕ)
  (h1 : hyungjin_side_length_cm = 100)
  (h2 : kyujun_side_length_m = 2) :
  volume_of_cube (kyujun_side_length_m * 100) = 8 * volume_of_cube hyungjin_side_length_cm :=
by
  sorry

end volume_ratio_l51_51306


namespace ways_to_place_letters_l51_51507

-- defining the conditions of the problem
def num_letters : Nat := 4
def num_mailboxes : Nat := 3

-- the theorem we need to prove
theorem ways_to_place_letters : 
  (num_mailboxes ^ num_letters) = 81 := 
by 
  sorry

end ways_to_place_letters_l51_51507


namespace remainder_when_divided_by_product_l51_51966

noncomputable def Q : Polynomial ℝ := sorry

theorem remainder_when_divided_by_product (Q : Polynomial ℝ)
    (h1 : Q.eval 20 = 100)
    (h2 : Q.eval 100 = 20) :
    ∃ R : Polynomial ℝ, ∃ a b : ℝ, Q = (Polynomial.X - 20) * (Polynomial.X - 100) * R + Polynomial.C a * Polynomial.X + Polynomial.C b ∧
    a = -1 ∧ b = 120 :=
by
  sorry

end remainder_when_divided_by_product_l51_51966


namespace number_of_3_digit_divisible_by_13_l51_51074

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l51_51074


namespace length_of_bridge_is_230_l51_51525

noncomputable def train_length : ℚ := 145
noncomputable def train_speed_kmh : ℚ := 45
noncomputable def time_to_cross_bridge : ℚ := 30
noncomputable def train_speed_ms : ℚ := (train_speed_kmh * 1000) / 3600
noncomputable def bridge_length : ℚ := (train_speed_ms * time_to_cross_bridge) - train_length

theorem length_of_bridge_is_230 :
  bridge_length = 230 :=
sorry

end length_of_bridge_is_230_l51_51525


namespace evaluate_series_l51_51554

noncomputable def infinite_series :=
  ∑' n, (n^3 + 2*n^2 - 3) / (n+3).factorial

theorem evaluate_series : infinite_series = 1 / 4 :=
by
  sorry

end evaluate_series_l51_51554


namespace part1_part2_l51_51912

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 3 + x ^ 2

noncomputable def g (f : ℝ -> ℝ) (x : ℝ) : ℝ := f x * Real.exp x

theorem part1 (a : ℝ) (h : deriv (f a) (-4 / 3) = 0) : a = 1 / 2 :=
by
  sorry

theorem part2 : 
  ∃ (I_decreasing I_increasing : Set ℝ), 
    I_decreasing = {x | x < -4} ∪ {x | -1 < x ∧ x ≤ 0} ∧
    I_increasing = {x | -4 < x ∧ x ≤ -1} ∪ {x | 0 < x} ∧
    ∀ x, (deriv (g (f (1 / 2)) x) < 0 ↔ x ∈ I_decreasing) ∧ 
         (deriv (g (f (1 / 2)) x) > 0 ↔ x ∈ I_increasing) :=
by
  sorry

end part1_part2_l51_51912


namespace find_x_l51_51889

theorem find_x : 
  (∃ x : ℝ, 
    2.5 * ((3.6 * 0.48 * 2.5) / (0.12 * x * 0.5)) = 2000.0000000000002) → 
  x = 0.225 :=
by
  sorry

end find_x_l51_51889


namespace find_speeds_l51_51720

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end find_speeds_l51_51720


namespace yoongi_stacked_higher_by_one_cm_l51_51509

def height_box_A : ℝ := 3
def height_box_B : ℝ := 3.5
def boxes_stacked_by_Taehyung : ℕ := 16
def boxes_stacked_by_Yoongi : ℕ := 14
def height_Taehyung_stack : ℝ := height_box_A * boxes_stacked_by_Taehyung
def height_Yoongi_stack : ℝ := height_box_B * boxes_stacked_by_Yoongi

theorem yoongi_stacked_higher_by_one_cm :
  height_Yoongi_stack = height_Taehyung_stack + 1 :=
by
  sorry

end yoongi_stacked_higher_by_one_cm_l51_51509


namespace sets_given_to_friend_l51_51612

theorem sets_given_to_friend (total_cards : ℕ) (total_given_away : ℕ) (sets_brother : ℕ) 
  (sets_sister : ℕ) (cards_per_set : ℕ) (sets_friend : ℕ) 
  (h1 : total_cards = 365) 
  (h2 : total_given_away = 195) 
  (h3 : sets_brother = 8) 
  (h4 : sets_sister = 5) 
  (h5 : cards_per_set = 13) 
  (h6 : total_given_away = (sets_brother + sets_sister + sets_friend) * cards_per_set) : 
  sets_friend = 2 :=
by
  sorry

end sets_given_to_friend_l51_51612


namespace count_3_digit_numbers_divisible_by_13_l51_51137

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51137


namespace sum_of_diagonals_l51_51959

def FG : ℝ := 4
def HI : ℝ := 4
def GH : ℝ := 11
def IJ : ℝ := 11
def FJ : ℝ := 15

theorem sum_of_diagonals (x y z : ℝ) (h1 : z^2 = 4 * x + 121) (h2 : z^2 = 11 * y + 16)
  (h3 : x * y = 44 + 15 * z) (h4 : x * z = 4 * z + 225) (h5 : y * z = 11 * z + 60) :
  3 * z + x + y = 90 :=
sorry

end sum_of_diagonals_l51_51959


namespace max_length_AB_l51_51370

theorem max_length_AB : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 3 → ∃ M, M = 81 / 8 ∧ ∀ t, -2 * (t - 3/4)^2 + 81 / 8 = M :=
by sorry

end max_length_AB_l51_51370


namespace percentage_problem_l51_51312

theorem percentage_problem (P : ℕ) : (P / 100 * 400 = 20 / 100 * 700) → P = 35 :=
by
  intro h
  sorry

end percentage_problem_l51_51312


namespace max_possible_ratio_squared_l51_51622

noncomputable def maxRatioSquared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) (h4 : ∃ x y, (0 ≤ x) ∧ (x < a) ∧ (0 ≤ y) ∧ (y < b) ∧ (a^2 + y^2 = b^2 + x^2) ∧ (b^2 + x^2 = (a - x)^2 + (b + y)^2)) : ℝ :=
  2

theorem max_possible_ratio_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) (h4 : ∃ x y, (0 ≤ x) ∧ (x < a) ∧ (0 ≤ y) ∧ (y < b) ∧ (a^2 + y^2 = b^2 + x^2) ∧ (b^2 + x^2 = (a - x)^2 + (b + y)^2)) : maxRatioSquared a b h1 h2 h3 h4 = 2 :=
sorry

end max_possible_ratio_squared_l51_51622


namespace range_of_m_l51_51465

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), x^2 - 4 * x - 2 * m + 1 ≤ 0) ↔ m ∈ Set.Ici (3 : ℝ) := 
sorry

end range_of_m_l51_51465


namespace cube_root_simplification_l51_51583

theorem cube_root_simplification (N : ℝ) (h : N > 1) : (N^3)^(1/3) * ((N^5)^(1/3) * ((N^3)^(1/3)))^(1/3) = N^(5/3) :=
by sorry

end cube_root_simplification_l51_51583


namespace period_of_f_g_is_2_sin_x_g_is_odd_l51_51738

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 3)

-- Theorem 1: Prove that f has period 2π.
theorem period_of_f : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

-- Define g and prove the related properties.
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 3)

-- Theorem 2: Prove that g(x) = 2 * sin x.
theorem g_is_2_sin_x : ∀ x : ℝ, g x = 2 * Real.sin x := by
  sorry

-- Theorem 3: Prove that g is an odd function.
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end period_of_f_g_is_2_sin_x_g_is_odd_l51_51738


namespace find_rstu_l51_51985

theorem find_rstu (a x y c : ℝ) (r s t u : ℤ) (hc : a^10 * x * y - a^8 * y - a^7 * x = a^6 * (c^3 - 1)) :
  (a^r * x - a^s) * (a^t * y - a^u) = a^6 * c^3 ∧ r * s * t * u = 0 :=
by
  sorry

end find_rstu_l51_51985


namespace greatest_multiple_of_30_less_than_800_l51_51514

theorem greatest_multiple_of_30_less_than_800 : 
    ∃ n : ℤ, (n % 30 = 0) ∧ (n < 800) ∧ (∀ m : ℤ, (m % 30 = 0) ∧ (m < 800) → m ≤ n) ∧ n = 780 :=
by
  sorry

end greatest_multiple_of_30_less_than_800_l51_51514


namespace equal_circumradii_l51_51349

-- Define the points and triangles involved
variable (A B C M : Type*) 

-- The circumcircle radius of a triangle is at least R
variable (R R1 R2 R3 : ℝ)

-- Hypotheses: the given conditions
variable (hR1 : R1 ≥ R)
variable (hR2 : R2 ≥ R)
variable (hR3 : R3 ≥ R)

-- The goal: to show that all four radii are equal
theorem equal_circumradii {A B C M : Type*} (R R1 R2 R3 : ℝ) 
    (hR1 : R1 ≥ R) 
    (hR2 : R2 ≥ R) 
    (hR3 : R3 ≥ R): 
    R1 = R ∧ R2 = R ∧ R3 = R := 
by 
  sorry

end equal_circumradii_l51_51349


namespace exists_positive_real_u_l51_51978

theorem exists_positive_real_u (n : ℕ) (h_pos : n > 0) : 
  ∃ u : ℝ, u > 0 ∧ ∀ n : ℕ, n > 0 → (⌊u^n⌋ - n) % 2 = 0 :=
sorry

end exists_positive_real_u_l51_51978


namespace total_students_l51_51810

theorem total_students (N : ℕ)
  (h1 : (84 + 128 + 13 = 15 * N))
  : N = 15 :=
sorry

end total_students_l51_51810


namespace only_three_A_l51_51558

def student := Type
variable (Alan Beth Carlos Diana Eliza : student)

variable (gets_A : student → Prop)

variable (H1 : gets_A Alan → gets_A Beth)
variable (H2 : gets_A Beth → gets_A Carlos)
variable (H3 : gets_A Carlos → gets_A Diana)
variable (H4 : gets_A Diana → gets_A Eliza)
variable (H5 : gets_A Eliza → gets_A Alan)
variable (H6 : ∃ a b c : student, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ gets_A a ∧ gets_A b ∧ gets_A c ∧ ∀ d : student, gets_A d → d = a ∨ d = b ∨ d = c)

theorem only_three_A : gets_A Carlos ∧ gets_A Diana ∧ gets_A Eliza :=
by
  sorry

end only_three_A_l51_51558


namespace A_and_D_mut_exclusive_not_complementary_l51_51658

-- Define the events based on the conditions
inductive Die
| one | two | three | four | five | six

def is_odd (d : Die) : Prop :=
  d = Die.one ∨ d = Die.three ∨ d = Die.five

def is_even (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four ∨ d = Die.six

def is_multiple_of_2 (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four ∨ d = Die.six

def is_two_or_four (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four

-- Define the predicate for mutually exclusive but not complementary
def mutually_exclusive_but_not_complementary (P Q : Die → Prop) : Prop :=
  (∀ d, ¬ (P d ∧ Q d)) ∧ ¬ (∀ d, P d ∨ Q d)

-- Verify that "A and D" are mutually exclusive but not complementary
theorem A_and_D_mut_exclusive_not_complementary :
  mutually_exclusive_but_not_complementary is_odd is_two_or_four :=
  by
    sorry

end A_and_D_mut_exclusive_not_complementary_l51_51658


namespace edward_remaining_money_l51_51711

def initial_amount : ℕ := 19
def spent_amount : ℕ := 13
def remaining_amount : ℕ := initial_amount - spent_amount

theorem edward_remaining_money : remaining_amount = 6 := by
  sorry

end edward_remaining_money_l51_51711


namespace koschei_coin_count_l51_51332

theorem koschei_coin_count (a : ℕ) :
  (a % 10 = 7) ∧
  (a % 12 = 9) ∧
  (300 ≤ a ∧ a ≤ 400) →
  a = 357 :=
sorry

end koschei_coin_count_l51_51332


namespace melanie_plums_l51_51800

variable (initialPlums : ℕ) (givenPlums : ℕ)

theorem melanie_plums :
  initialPlums = 7 → givenPlums = 3 → initialPlums - givenPlums = 4 :=
by
  intro h1 h2
  -- proof omitted
  exact sorry

end melanie_plums_l51_51800


namespace number_of_factors_180_l51_51923

def number_of_factors (n : ℕ) : ℕ :=
  (n.factors.eraseDuplicates.map (λ p, n.factors.count p + 1)).foldr (· * ·) 1

theorem number_of_factors_180 : number_of_factors 180 = 18 := by
  sorry

end number_of_factors_180_l51_51923


namespace sym_coords_origin_l51_51587

theorem sym_coords_origin (a b : ℝ) (h : |a - 3| + (b + 4)^2 = 0) :
  (-a, -b) = (-3, 4) :=
sorry

end sym_coords_origin_l51_51587


namespace count_3_digit_numbers_divisible_by_13_l51_51060

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l51_51060


namespace valid_integers_count_l51_51919

def count_valid_integers : ℕ :=
  let digits : List ℕ := [0, 1, 2, 3, 4, 6, 7, 8, 9]
  let first_digit_count := 7  -- from 2 to 9 excluding 5
  let second_digit_count := 8
  let third_digit_count := 7
  let fourth_digit_count := 6
  first_digit_count * second_digit_count * third_digit_count * fourth_digit_count

theorem valid_integers_count : count_valid_integers = 2352 := by
  -- intermediate step might include nice counting macros
  sorry

end valid_integers_count_l51_51919


namespace largest_value_of_n_l51_51013

noncomputable def largest_n_under_200000 : ℕ :=
  if h : 199999 < 200000 ∧ (8 * (199999 - 3)^5 - 2 * 199999^2 + 18 * 199999 - 36) % 7 = 0 then 199999 else 0

theorem largest_value_of_n (n : ℕ) :
  n < 200000 → (8 * (n - 3)^5 - 2 * n^2 + 18 * n - 36) % 7 = 0 → n = 199999 :=
by sorry

end largest_value_of_n_l51_51013


namespace compute_expression_l51_51871

theorem compute_expression : 2 * (Real.sqrt 144)^2 = 288 := by
  sorry

end compute_expression_l51_51871


namespace solve_inequality_l51_51808

theorem solve_inequality (x : ℝ) : 
  (3 * x^2 - 5 * x + 2 > 0) ↔ (x < 2 / 3 ∨ x > 1) := 
by
  sorry

end solve_inequality_l51_51808


namespace expression_evaluation_l51_51018

theorem expression_evaluation : 
  ( ((2 + 2)^2 / 2^2) * ((3 + 3 + 3 + 3)^3 / (3 + 3 + 3)^3) * ((6 + 6 + 6 + 6 + 6 + 6)^6 / (6 + 6 + 6 + 6)^6) = 108 ) := 
by 
  sorry

end expression_evaluation_l51_51018


namespace rectangular_board_area_l51_51693

variable (length width : ℕ)

theorem rectangular_board_area
  (h1 : length = 2 * width)
  (h2 : 2 * length + 2 * width = 84) :
  length * width = 392 := 
by
  sorry

end rectangular_board_area_l51_51693


namespace profit_from_ad_l51_51325

def advertising_cost : ℝ := 1000
def customers : ℕ := 100
def purchase_rate : ℝ := 0.8
def purchase_price : ℝ := 25

theorem profit_from_ad (advertising_cost customers purchase_rate purchase_price : ℝ) : 
  (customers * purchase_rate * purchase_price - advertising_cost) = 1000 :=
by
  -- assumptions as conditions
  let bought_customers := (customers : ℝ) * purchase_rate
  let revenue := bought_customers * purchase_price
  let profit := revenue - advertising_cost
  -- state the proof goal
  have goal : profit = 1000 :=
    sorry
  exact goal

end profit_from_ad_l51_51325


namespace median_hypotenuse_right_triangle_l51_51469

/-- Prove that in a right triangle with legs of lengths 5 and 12,
  the median on the hypotenuse can be either 6 or 6.5. -/
theorem median_hypotenuse_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) :
  ∃ c : ℝ, (c = 6 ∨ c = 6.5) :=
sorry

end median_hypotenuse_right_triangle_l51_51469


namespace maximum_area_of_rectangular_farm_l51_51011

theorem maximum_area_of_rectangular_farm :
  ∃ l w : ℕ, 2 * (l + w) = 160 ∧ l * w = 1600 :=
by
  sorry

end maximum_area_of_rectangular_farm_l51_51011


namespace average_daily_net_income_correct_l51_51847

-- Define the income, tips, and expenses for each day.
def day1_income := 300
def day1_tips := 50
def day1_expenses := 80

def day2_income := 150
def day2_tips := 20
def day2_expenses := 40

def day3_income := 750
def day3_tips := 100
def day3_expenses := 150

def day4_income := 200
def day4_tips := 30
def day4_expenses := 50

def day5_income := 600
def day5_tips := 70
def day5_expenses := 120

-- Define the net income for each day as income + tips - expenses.
def day1_net_income := day1_income + day1_tips - day1_expenses
def day2_net_income := day2_income + day2_tips - day2_expenses
def day3_net_income := day3_income + day3_tips - day3_expenses
def day4_net_income := day4_income + day4_tips - day4_expenses
def day5_net_income := day5_income + day5_tips - day5_expenses

-- Calculate the total net income over the 5 days.
def total_net_income := 
  day1_net_income + day2_net_income + day3_net_income + day4_net_income + day5_net_income

-- Define the number of days.
def number_of_days := 5

-- Calculate the average daily net income.
def average_daily_net_income := total_net_income / number_of_days

-- Statement to prove that the average daily net income is $366.
theorem average_daily_net_income_correct :
  average_daily_net_income = 366 := by
  sorry

end average_daily_net_income_correct_l51_51847


namespace run_time_is_48_minutes_l51_51837

noncomputable def cycling_speed : ℚ := 5 / 2
noncomputable def running_speed : ℚ := cycling_speed * 0.5
noncomputable def walking_speed : ℚ := running_speed * 0.5

theorem run_time_is_48_minutes (d : ℚ) (h : (d / cycling_speed) + (d / walking_speed) = 2) : 
  (60 * d / running_speed) = 48 :=
by
  sorry

end run_time_is_48_minutes_l51_51837


namespace brenda_age_l51_51414

theorem brenda_age (A B J : ℝ)
  (h1 : A = 4 * B)
  (h2 : J = B + 8)
  (h3 : A = J + 2) :
  B = 10 / 3 :=
by
  sorry

end brenda_age_l51_51414


namespace prob_students_on_both_days_l51_51294
noncomputable def probability_event_on_both_days: ℚ := by
  let total_days := 2
  let total_students := 4
  let prob_single_day := (1 / total_days : ℚ) ^ total_students
  let prob_all_same_day := 2 * prob_single_day
  let prob_both_days := 1 - prob_all_same_day
  exact prob_both_days

theorem prob_students_on_both_days : probability_event_on_both_days = 7 / 8 :=
by
  exact sorry

end prob_students_on_both_days_l51_51294


namespace count_three_digit_numbers_divisible_by_13_l51_51151

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l51_51151


namespace card_M_l51_51820

open Set

noncomputable def M : Set ℤ := {x | (x - 5) * (x - 1) ≤ 0 ∧ x ≠ 1}

theorem card_M : (card M) = 4 :=
by
  sorry

end card_M_l51_51820


namespace three_digit_numbers_divisible_by_13_count_l51_51132

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l51_51132


namespace number_of_cats_l51_51600

variable (C D : ℕ)

-- Conditions
def condition1 : Prop := C = 15 * D / 7
def condition2 : Prop := C = 15 * (D + 12) / 11

-- Proof problem
theorem number_of_cats (h1 : condition1 C D) (h2 : condition2 C D) : C = 45 := sorry

end number_of_cats_l51_51600


namespace value_of_a4_l51_51183

variables {a : ℕ → ℝ} -- Define the sequence as a function from natural numbers to real numbers.

-- Conditions: The sequence is geometric, positive and satisfies the given product condition.
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k, a (n + k) = (a n) * (a k)

-- Condition: All terms are positive.
def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

-- Given product condition:
axiom a1_a7_product : a 1 * a 7 = 36

-- The theorem to prove:
theorem value_of_a4 (h_geo : is_geometric_sequence a) (h_pos : all_terms_positive a) : 
  a 4 = 6 :=
sorry

end value_of_a4_l51_51183


namespace original_number_of_people_l51_51802

theorem original_number_of_people (x : ℕ) (h1 : 3 ∣ x) (h2 : 6 ∣ x) (h3 : (x / 2) = 18) : x = 36 :=
by
  sorry

end original_number_of_people_l51_51802


namespace malcolm_initial_white_lights_l51_51230

theorem malcolm_initial_white_lights :
  ∀ (red blue green remaining total_initial : ℕ),
    red = 12 →
    blue = 3 * red →
    green = 6 →
    remaining = 5 →
    total_initial = red + blue + green + remaining →
    total_initial = 59 :=
by
  intros red blue green remaining total_initial h1 h2 h3 h4 h5
  -- Add details if necessary for illustration
  -- sorry typically as per instructions
  sorry

end malcolm_initial_white_lights_l51_51230


namespace oranges_weight_is_10_l51_51855

def applesWeight (A : ℕ) : ℕ := A
def orangesWeight (A : ℕ) : ℕ := 5 * A
def totalWeight (A : ℕ) (O : ℕ) : ℕ := A + O
def totalCost (A : ℕ) (x : ℕ) (O : ℕ) (y : ℕ) : ℕ := A * x + O * y

theorem oranges_weight_is_10 (A O : ℕ) (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 := by
  sorry

end oranges_weight_is_10_l51_51855


namespace electricity_usage_A_B_l51_51494

def electricity_cost (x : ℕ) : ℝ :=
  if h₁ : 0 ≤ x ∧ x ≤ 24 then 4.2 * x
  else if h₂ : 24 < x ∧ x ≤ 60 then 5.2 * x - 24
  else if h₃ : 60 < x ∧ x ≤ 100 then 6.6 * x - 108
  else if h₄ : 100 < x ∧ x ≤ 150 then 7.6 * x - 208
  else if h₅ : 150 < x ∧ x ≤ 250 then 8 * x - 268
  else 8.4 * x - 368

theorem electricity_usage_A_B (x : ℕ) (h : electricity_cost x = 486) :
  60 < x ∧ x ≤ 100 ∧ 5 * x = 450 ∧ 2 * x = 180 :=
by
  sorry

end electricity_usage_A_B_l51_51494


namespace count_three_digit_numbers_divisible_by_13_l51_51150

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l51_51150


namespace divide_cakes_l51_51877

/-- Statement: Eleven cakes can be divided equally among six girls without cutting any cake into 
exactly six equal parts such that each girl receives 1 + 1/2 + 1/4 + 1/12 cakes -/
theorem divide_cakes (cakes girls : ℕ) (h_cakes : cakes = 11) (h_girls : girls = 6) :
  ∃ (parts : ℕ → ℝ), (∀ i, parts i = 1 + 1 / 2 + 1 / 4 + 1 / 12) ∧ (cakes = girls * (1 + 1 / 2 + 1 / 4 + 1 / 12)) :=
by
  sorry

end divide_cakes_l51_51877


namespace Kyler_wins_l51_51977

variable (K : ℕ) -- Kyler's wins

/- Constants based on the problem statement -/
def Peter_wins := 5
def Peter_losses := 3
def Emma_wins := 2
def Emma_losses := 4
def Total_games := 15
def Kyler_losses := 4

/- Definition that calculates total games played -/
def total_games_played := 2 * Total_games

/- Game equation based on the total count of played games -/
def game_equation := Peter_wins + Peter_losses + Emma_wins + Emma_losses + K + Kyler_losses = total_games_played

/- Question: Calculate Kyler's wins assuming the given conditions -/
theorem Kyler_wins : K = 1 :=
by
  sorry

end Kyler_wins_l51_51977


namespace no_polyhedron_with_surface_2015_l51_51186

/--
It is impossible to glue together 1 × 1 × 1 cubes to form a polyhedron whose surface area is 2015.
-/
theorem no_polyhedron_with_surface_2015 (n k : ℕ) : 6 * n - 2 * k ≠ 2015 :=
by
  sorry

end no_polyhedron_with_surface_2015_l51_51186


namespace difference_between_numbers_l51_51226

theorem difference_between_numbers :
  ∃ S : ℝ, L = 1650 ∧ L = 6 * S + 15 ∧ L - S = 1377.5 :=
sorry

end difference_between_numbers_l51_51226


namespace find_x_six_l51_51262

noncomputable def positive_real : Type := { x : ℝ // 0 < x }

theorem find_x_six (x : positive_real)
  (h : (1 - x.val ^ 3) ^ (1/3) + (1 + x.val ^ 3) ^ (1/3) = 1) :
  x.val ^ 6 = 28 / 27 := 
sorry

end find_x_six_l51_51262


namespace smallest_m_for_integral_solutions_l51_51014

theorem smallest_m_for_integral_solutions :
  ∃ (m : ℕ), (∀ (x : ℤ), (12 * x^2 - m * x + 504 = 0 → ∃ (p q : ℤ), p + q = m / 12 ∧ p * q = 42)) ∧
  m = 156 := by
sorry

end smallest_m_for_integral_solutions_l51_51014


namespace probability_of_sum_12_with_at_least_one_4_or_more_l51_51247

-- Definitions for the problem conditions
def outcomes := {x : ℕ × ℕ × ℕ | x.1 + x.2.1 + x.2.2 = 12 ∧ 
  (x.1 ≥ 4 ∨ x.2.1 ≥ 4 ∨ x.2.2 ≥ 4)}

def total_possibilities := 6 * 6 * 6

noncomputable def count_outcomes : ℕ :=
  Set.card outcomes

-- The final probability we need to prove matches the expected result
theorem probability_of_sum_12_with_at_least_one_4_or_more :
  (count_outcomes : ℚ) / total_possibilities = 4 / 54 :=
  sorry

end probability_of_sum_12_with_at_least_one_4_or_more_l51_51247


namespace jennifer_spent_124_dollars_l51_51957

theorem jennifer_spent_124_dollars 
  (initial_cans : ℕ := 40)
  (cans_per_set : ℕ := 5)
  (additional_cans_per_set : ℕ := 6)
  (total_cans_mark : ℕ := 30)
  (price_per_can_whole : ℕ := 2)
  (discount_threshold_whole : ℕ := 10)
  (discount_amount_whole : ℕ := 4) : 
  (initial_cans + additional_cans_per_set * (total_cans_mark / cans_per_set)) * price_per_can_whole - 
  (discount_amount_whole * ((initial_cans + additional_cans_per_set * (total_cans_mark / cans_per_set)) / discount_threshold_whole)) = 124 := by
  sorry

end jennifer_spent_124_dollars_l51_51957


namespace fifteenth_digit_sum_l51_51512

/-- The 15th digit after the decimal point of the sum of decimal equivalents of 1/9 and 1/11 is 1. -/
theorem fifteenth_digit_sum (d1 d2 : Nat) (h1 : (1/9 : Rat) = 0.1111111 -- overline 1 represents repeating 1
                    h2 : (1/11 : Rat) = 0.090909) -- overline 090909 represents repeating 090909
                   (repeating_block : String := "10")
                    : repeating_block[15 % 2] = '1' := -- finding the 15th digit
by
  sorry

end fifteenth_digit_sum_l51_51512


namespace y_intercept_tangent_line_l51_51256

noncomputable def tangent_line_y_intercept (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (htangent: Prop) : ℝ :=
  if r1 = 3 ∧ r2 = 2 ∧ c1 = (3, 0) ∧ c2 = (8, 0) ∧ htangent = true then 6 * Real.sqrt 6 else 0

theorem y_intercept_tangent_line (h : tangent_line_y_intercept 3 2 (3, 0) (8, 0) true = 6 * Real.sqrt 6) :
  tangent_line_y_intercept 3 2 (3, 0) (8, 0) true = 6 * Real.sqrt 6 :=
by
  exact h

end y_intercept_tangent_line_l51_51256


namespace part_a_exists_part_b_not_exists_l51_51709

theorem part_a_exists :
  ∃ (a b : ℤ), (∀ x : ℝ, x^2 + a*x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + a*x + b = 0) :=
sorry

theorem part_b_not_exists :
  ¬ ∃ (a b : ℤ), (∀ x : ℝ, x^2 + 2*a*x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + 2*a*x + b = 0) :=
sorry

end part_a_exists_part_b_not_exists_l51_51709


namespace ab_plus_a_plus_b_l51_51964

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 6 * x^2 - x + 2
-- Define the conditions on a and b
def is_root (x : ℝ) : Prop := poly x = 0

-- State the theorem
theorem ab_plus_a_plus_b (a b : ℝ) (ha : is_root a) (hb : is_root b) : a * b + a + b = 1 :=
sorry

end ab_plus_a_plus_b_l51_51964


namespace dad_strawberries_weight_proof_l51_51792

/-
Conditions:
1. total_weight (the combined weight of Marco's and his dad's strawberries) is 23 pounds.
2. marco_weight (the weight of Marco's strawberries) is 14 pounds.
We need to prove that dad_weight (the weight of dad's strawberries) is 9 pounds.
-/

def total_weight : ℕ := 23
def marco_weight : ℕ := 14

def dad_weight : ℕ := total_weight - marco_weight

theorem dad_strawberries_weight_proof : dad_weight = 9 := by
  sorry

end dad_strawberries_weight_proof_l51_51792


namespace problem_statement_l51_51196

noncomputable def p := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def q := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def r := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7
noncomputable def s := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7

theorem problem_statement :
  (1 / p + 1 / q + 1 / r + 1 / s)^2 = 112 / 3481 :=
sorry

end problem_statement_l51_51196


namespace count_three_digit_numbers_divisible_by_13_l51_51087

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l51_51087


namespace inequality_holds_l51_51873

variable (x a : ℝ)

def tensor (x y : ℝ) : ℝ :=
  (1 - x) * (1 + y)

theorem inequality_holds (h : ∀ x : ℝ, tensor (x - a) (x + a) < 1) : -2 < a ∧ a < 0 := by
  sorry

end inequality_holds_l51_51873


namespace find_largest_number_l51_51650

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
sorry

end find_largest_number_l51_51650


namespace max_squares_covered_l51_51851

theorem max_squares_covered 
    (board_square_side : ℝ) 
    (card_side : ℝ) 
    (n : ℕ) 
    (h1 : board_square_side = 1) 
    (h2 : card_side = 2) 
    (h3 : ∀ x y : ℝ, (x*x + y*y ≤ card_side*card_side) → card_side*card_side ≤ 4) :
    n ≤ 9 := sorry

end max_squares_covered_l51_51851


namespace max_value_log_div_x_l51_51234

noncomputable def func (x : ℝ) := (Real.log x) / x

theorem max_value_log_div_x : ∃ x > 0, func x = 1 / Real.exp 1 ∧ 
(∀ t > 0, t ≠ x → func t ≤ func x) :=
sorry

end max_value_log_div_x_l51_51234


namespace number_of_paths_l51_51640

theorem number_of_paths (n : ℕ) (h1 : n > 3) : 
  (2 * (8 * n^3 - 48 * n^2 + 88 * n - 48) + (4 * n^2 - 12 * n + 8) + (2 * n - 2)) = 16 * n^3 - 92 * n^2 + 166 * n - 90 :=
by
  sorry

end number_of_paths_l51_51640


namespace books_from_second_shop_l51_51495

-- Define the conditions
def num_books_first_shop : ℕ := 65
def cost_first_shop : ℕ := 1280
def cost_second_shop : ℕ := 880
def total_cost : ℤ := cost_first_shop + cost_second_shop
def average_price_per_book : ℤ := 18

-- Define the statement to be proved
theorem books_from_second_shop (x : ℕ) :
  (num_books_first_shop + x) * average_price_per_book = total_cost →
  x = 55 :=
by
  sorry

end books_from_second_shop_l51_51495


namespace points_meet_every_720_seconds_l51_51659

theorem points_meet_every_720_seconds
    (v1 v2 : ℝ) 
    (h1 : v1 - v2 = 1/720) 
    (h2 : (1/v2) - (1/v1) = 10) :
    v1 = 1/80 ∧ v2 = 1/90 :=
by
  sorry

end points_meet_every_720_seconds_l51_51659


namespace fraction_equivalence_l51_51666

theorem fraction_equivalence : 
    (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 := by sorry

end fraction_equivalence_l51_51666


namespace peak_infection_day_l51_51601

-- Given conditions
def initial_cases : Nat := 20
def increase_rate : Nat := 50
def decrease_rate : Nat := 30
def total_infections : Nat := 8670
def total_days : Nat := 30

-- Peak Day and infections on that day
def peak_day : Nat := 12

-- Theorem stating what we want to prove
theorem peak_infection_day :
  ∃ n : Nat, n = initial_cases + increase_rate * (peak_day - 1) - decrease_rate * (30 - peak_day) :=
sorry

end peak_infection_day_l51_51601


namespace average_minutes_run_per_day_l51_51698

theorem average_minutes_run_per_day (f : ℕ) :
  let third_grade_minutes := 12
  let fourth_grade_minutes := 15
  let fifth_grade_minutes := 10
  let third_graders := 4 * f
  let fourth_graders := 2 * f
  let fifth_graders := f
  let total_minutes := third_graders * third_grade_minutes + fourth_graders * fourth_grade_minutes + fifth_graders * fifth_grade_minutes
  let total_students := third_graders + fourth_graders + fifth_graders
  total_minutes / total_students = 88 / 7 :=
by
  sorry

end average_minutes_run_per_day_l51_51698


namespace usual_time_72_l51_51397

namespace TypicalTimeProof

variables (S T : ℝ) 

theorem usual_time_72 (h : T ≠ 0) (h2 : 0.75 * S ≠ 0) (h3 : 4 * T = 3 * (T + 24)) : T = 72 := by
  sorry

end TypicalTimeProof

end usual_time_72_l51_51397


namespace four_consecutive_integers_product_2520_l51_51044

theorem four_consecutive_integers_product_2520 {a b c d : ℕ}
  (h1 : a + 1 = b) 
  (h2 : b + 1 = c) 
  (h3 : c + 1 = d) 
  (h4 : a * b * c * d = 2520) : 
  a = 6 := 
sorry

end four_consecutive_integers_product_2520_l51_51044


namespace more_red_flowers_than_white_l51_51539

-- Definitions based on given conditions
def yellow_and_white := 13
def red_and_yellow := 17
def red_and_white := 14
def blue_and_yellow := 16

-- Definitions based on the requirements of the problem
def red_flowers := red_and_yellow + red_and_white
def white_flowers := yellow_and_white + red_and_white

-- Theorem to prove the number of more flowers containing red than white
theorem more_red_flowers_than_white : red_flowers - white_flowers = 4 := by
  sorry

end more_red_flowers_than_white_l51_51539


namespace total_number_of_coins_l51_51007

-- Define conditions
def pennies : Nat := 38
def nickels : Nat := 27
def dimes : Nat := 19
def quarters : Nat := 24
def half_dollars : Nat := 13
def one_dollar_coins : Nat := 17
def two_dollar_coins : Nat := 5
def australian_fifty_cent_coins : Nat := 4
def mexican_one_peso_coins : Nat := 12

-- Define the problem as a theorem
theorem total_number_of_coins : 
  pennies + nickels + dimes + quarters + half_dollars + one_dollar_coins + two_dollar_coins + australian_fifty_cent_coins + mexican_one_peso_coins = 159 := by
  sorry

end total_number_of_coins_l51_51007


namespace num_100_digit_even_numbers_l51_51915

theorem num_100_digit_even_numbers : 
  let digit_set := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let valid_number (digits : list ℕ) := 
    digits.length = 100 ∧ digits.head ∈ {1, 3} ∧ 
    digits.last ∈ {0} ∧ 
    ∀ d ∈ digits.tail.init, d ∈ digit_set
  (∃ (m : ℕ), valid_number (m.digits 10)) = 2 * 3^98 := 
sorry

end num_100_digit_even_numbers_l51_51915


namespace min_value_quadratic_l51_51388

theorem min_value_quadratic (x : ℝ) : ∃ x, x = -7 ∧ (x^2 + 14 * x + 24 = -25) := sorry

end min_value_quadratic_l51_51388


namespace count_three_digit_div_by_13_l51_51104

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l51_51104


namespace members_count_l51_51394

theorem members_count (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end members_count_l51_51394


namespace cubic_roots_identity_l51_51627

theorem cubic_roots_identity 
  (x1 x2 x3 p q : ℝ) 
  (hq : ∀ x, x^3 + p * x + q = (x - x1) * (x - x2) * (x - x3))
  (h_sum : x1 + x2 + x3 = 0)
  (h_prod : x1 * x2 + x2 * x3 + x3 * x1 = p):
  x2^2 + x2 * x3 + x3^2 = -p ∧ x1^2 + x1 * x3 + x3^2 = -p ∧ x1^2 + x1 * x2 + x2^2 = -p :=
sorry

end cubic_roots_identity_l51_51627


namespace minimum_value_of_f_l51_51883

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x + 1/x)

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ 5/2) ∧ (f 1 = 5/2) := by
  sorry

end minimum_value_of_f_l51_51883


namespace table_to_chair_ratio_l51_51613

noncomputable def price_chair : ℤ := 20
noncomputable def price_table : ℤ := 60
noncomputable def price_couch : ℤ := 300

theorem table_to_chair_ratio 
  (h1 : price_couch = 300)
  (h2 : price_couch = 5 * price_table)
  (h3 : price_chair + price_table + price_couch = 380)
  : price_table / price_chair = 3 := 
by 
  sorry

end table_to_chair_ratio_l51_51613


namespace symmetric_axis_of_parabola_l51_51988

theorem symmetric_axis_of_parabola :
  (∃ x : ℝ, x = 6 ∧ (∀ y : ℝ, y = 1/2 * x^2 - 6 * x + 21)) :=
sorry

end symmetric_axis_of_parabola_l51_51988


namespace product_largest_smallest_using_digits_l51_51560

theorem product_largest_smallest_using_digits (a b : ℕ) (h1 : 100 * 6 + 10 * 2 + 0 = a) (h2 : 100 * 2 + 10 * 0 + 6 = b) : a * b = 127720 := by
  -- The proof will go here
  sorry

end product_largest_smallest_using_digits_l51_51560


namespace min_value_of_function_l51_51563

theorem min_value_of_function (x : ℝ) (h : x > -1) : 
  (∀ x₀ : ℝ, x₀ > -1 → (x₀ + 1 + 1 / (x₀ + 1) - 1) ≥ 1) ∧ (x = 0) :=
sorry

end min_value_of_function_l51_51563


namespace triangle_shading_probability_l51_51751

theorem triangle_shading_probability (n_triangles: ℕ) (n_shaded: ℕ) (h1: n_triangles > 4) (h2: n_shaded = 4) (h3: n_triangles = 10) :
  (n_shaded / n_triangles) = 2 / 5 := 
by
  sorry

end triangle_shading_probability_l51_51751


namespace monthly_income_l51_51220

def average_expenditure_6_months (expenditure_6_months : ℕ) (average : ℕ) : Prop :=
  average = expenditure_6_months / 6

def expenditure_next_4_months (expenditure_4_months : ℕ) (monthly_expense : ℕ) : Prop :=
  expenditure_4_months = 4 * monthly_expense

def cleared_debt_and_saved (income_4_months : ℕ) (debt : ℕ) (savings : ℕ)  (condition : ℕ) : Prop :=
  income_4_months = debt + savings + condition

theorem monthly_income 
(income : ℕ) 
(avg_6m_exp : ℕ) 
(exp_4m : ℕ) 
(debt: ℕ) 
(savings: ℕ )
(condition: ℕ) 
    (h1 : average_expenditure_6_months avg_6m_exp 85) 
    (h2 : expenditure_next_4_months exp_4m 60) 
    (h3 : cleared_debt_and_saved (income * 4) debt savings 30) 
    (h4 : income * 6 < 6 * avg_6m_exp) 
    : income = 78 :=
sorry

end monthly_income_l51_51220


namespace sheet_length_l51_51865

theorem sheet_length (L : ℝ) : 
  (20 * L > 0) → 
  ((16 * (L - 6)) / (20 * L) = 0.64) → 
  L = 30 :=
by
  intro h1 h2
  sorry

end sheet_length_l51_51865


namespace symmetry_center_of_g_l51_51377

noncomputable def g (x : ℝ) : ℝ := 2 * real.cos (2 * x - 2 * real.pi / 3) - 1

theorem symmetry_center_of_g : ∃ k : ℤ, g (k * real.pi / 2 + real.pi / 12) = -1 :=
by
  sorry

end symmetry_center_of_g_l51_51377


namespace correct_time_fraction_l51_51534

theorem correct_time_fraction : 
  let incorrect_hours := {1, 10, 11, 12}
  let total_hours := 12
  let correct_hours := total_hours - incorrect_hours.size
  let total_minutes := 60
  let incorrect_minutes := 10 + 5 -- minutes with '1' in tens or ones place
  let correct_minutes := total_minutes - incorrect_minutes
  correct_hours * correct_minutes / (total_hours * total_minutes) = 1 / 2 :=
by
  let incorrect_hours := {1, 10, 11, 12}
  let total_hours := 12
  let correct_hours := total_hours - incorrect_hours.size
  let total_minutes := 60
  let incorrect_minutes := 10 + 5
  let correct_minutes := total_minutes - incorrect_minutes
  have hours_fraction := correct_hours / total_hours
  have minutes_fraction := correct_minutes / total_minutes
  have day_fraction := hours_fraction * minutes_fraction
  sorry

end correct_time_fraction_l51_51534


namespace find_speeds_l51_51718

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end find_speeds_l51_51718


namespace smallest_value_of_n_l51_51423

theorem smallest_value_of_n :
  ∃ o y m n : ℕ, 10 * o = 16 * y ∧ 16 * y = 18 * m ∧ 18 * m = 18 * n ∧ n = 40 := 
sorry

end smallest_value_of_n_l51_51423


namespace solve_for_x_l51_51046

-- Lean 4 statement for the problem
theorem solve_for_x (x : ℝ) (h : (x + 1)^3 = -27) : x = -4 := by
  sorry

end solve_for_x_l51_51046


namespace shooting_accuracy_l51_51483

theorem shooting_accuracy (S : ℕ → ℕ) (H1 : ∀ n, S n < 10 * n / 9) (H2 : ∀ n, S n > 10 * n / 9) :
  ∃ n, 10 * (S n) = 9 * n :=
by
  sorry

end shooting_accuracy_l51_51483


namespace shorten_ellipse_parametric_form_l51_51500

theorem shorten_ellipse_parametric_form :
  ∀ (θ : ℝ), 
  ∃ (x' y' : ℝ),
    x' = 4 * Real.cos θ ∧ y' = 2 * Real.sin θ ∧
    (∃ (x y : ℝ),
      x' = 2 * x ∧ y' = y ∧
      x = 2 * Real.cos θ ∧ y = 2 * Real.sin θ) :=
by
  sorry

end shorten_ellipse_parametric_form_l51_51500


namespace theater_earnings_l51_51691

theorem theater_earnings :
  let matinee_price := 5
  let evening_price := 7
  let opening_night_price := 10
  let popcorn_price := 10
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let half_of_customers_that_bought_popcorn := 
    (matinee_customers + evening_customers + opening_night_customers) / 2
  let total_earnings := 
    (matinee_price * matinee_customers) + 
    (evening_price * evening_customers) + 
    (opening_night_price * opening_night_customers) + 
    (popcorn_price * half_of_customers_that_bought_popcorn)
  total_earnings = 1670 :=
by
  sorry

end theater_earnings_l51_51691


namespace least_number_of_plates_needed_l51_51404

theorem least_number_of_plates_needed
  (cubes : ℕ)
  (cube_dim : ℕ)
  (temp_limit : ℕ)
  (plates_exist : ∀ (n : ℕ), n > temp_limit → ∃ (p : ℕ), p = 21) :
  cubes = 512 ∧ cube_dim = 8 → temp_limit > 0 → 21 = 7 + 7 + 7 :=
by {
  sorry
}

end least_number_of_plates_needed_l51_51404


namespace discount_price_l51_51536

theorem discount_price (original_price : ℝ) (discount_rate : ℝ) (current_price : ℝ) 
  (h1 : original_price = 120) 
  (h2 : discount_rate = 0.8) 
  (h3 : current_price = original_price * discount_rate) : 
  current_price = 96 := 
by
  sorry

end discount_price_l51_51536


namespace solve_abc_l51_51337

def f (x a b c : ℤ) : ℤ := x^3 + a*x^2 + b*x + c

theorem solve_abc (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_fa : f a a b c = a^3) (h_fb : f b b a c = b^3) : 
  a = -2 ∧ b = 4 ∧ c = 16 := 
sorry

end solve_abc_l51_51337


namespace matrices_commute_l51_51193

variable {n : Nat}
variable (A B X : Matrix (Fin n) (Fin n) ℝ)

theorem matrices_commute (h : A * X * B + A + B = 0) : A * X * B = B * X * A :=
by
  sorry

end matrices_commute_l51_51193


namespace three_digit_numbers_div_by_13_l51_51110

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l51_51110


namespace clock_correct_fraction_l51_51535

/--
A 12-hour digital clock displays the hour and minute of a day. 
Whenever it is supposed to display a 1, it mistakenly displays a 9. 
Prove that the fraction of the day the clock shows the correct time is 1/2.
-/
def correct_fraction_hours : ℚ := 2 / 3

def correct_fraction_minutes : ℚ := 3 / 4

theorem clock_correct_fraction : correct_fraction_hours * correct_fraction_minutes = 1 / 2 :=
by
  have hours_correct := correct_fraction_hours
  have minutes_correct := correct_fraction_minutes
  calc
    (correct_fraction_hours * correct_fraction_minutes) = (2 / 3 * 3 / 4) : by sorry
    ... = 1 / 2 : by sorry

end clock_correct_fraction_l51_51535


namespace range_of_a_l51_51569

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x^2 + 2 * a * x + 1) → -1 ≤ a ∧ a ≤ 1 :=
by
  intro h
  sorry

end range_of_a_l51_51569


namespace quadrant_of_tan_and_cos_l51_51745

theorem quadrant_of_tan_and_cos (α : ℝ) (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  ∃ Q, (Q = 2) :=
by
  sorry


end quadrant_of_tan_and_cos_l51_51745


namespace decreased_and_divided_l51_51313

theorem decreased_and_divided (x : ℝ) (h : (x - 5) / 7 = 7) : (x - 14) / 10 = 4 := by
  sorry

end decreased_and_divided_l51_51313


namespace combined_height_of_rockets_l51_51189

noncomputable def height_of_rocket (a t : ℝ) : ℝ := (1/2) * a * t^2

theorem combined_height_of_rockets
  (h_A_ft : ℝ)
  (fuel_type_B_coeff : ℝ)
  (g : ℝ)
  (ft_to_m : ℝ)
  (h_combined : ℝ) :
  h_A_ft = 850 →
  fuel_type_B_coeff = 1.7 →
  g = 9.81 →
  ft_to_m = 0.3048 →
  h_combined = 348.96 :=
by sorry

end combined_height_of_rockets_l51_51189


namespace problem_inequality_l51_51787

theorem problem_inequality (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  -- proof here
  sorry

end problem_inequality_l51_51787


namespace value_of_f_g_3_l51_51746

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x^2 - 2*x + 1

theorem value_of_f_g_3 : f (g 3) = 2134 :=
by 
  sorry

end value_of_f_g_3_l51_51746


namespace complement_of_60_is_30_l51_51939

noncomputable def complement (angle : ℝ) : ℝ := 90 - angle

theorem complement_of_60_is_30 : complement 60 = 30 :=
by 
  sorry

end complement_of_60_is_30_l51_51939


namespace symmetric_function_cannot_be_even_l51_51448

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_function_cannot_be_even :
  (∀ x, f (f x) = x^2) ∧ (∀ x ≥ 0, f (x^2) = x) → ¬ (∀ x, f x = f (-x)) :=
by 
  intros
  sorry -- Proof is not required

end symmetric_function_cannot_be_even_l51_51448


namespace inradius_plus_circumradius_le_height_l51_51788

theorem inradius_plus_circumradius_le_height {α β γ : ℝ} 
    (h : ℝ) (r R : ℝ)
    (h_triangle : α ≥ β ∧ β ≥ γ ∧ γ ≥ 0 ∧ α + β + γ = π )
    (h_non_obtuse : π / 2 ≥ α ∧ π / 2 ≥ β ∧ π / 2 ≥ γ)
    (h_greatest_height : true) -- Assuming this condition holds as given
    :
    r + R ≤ h :=
sorry

end inradius_plus_circumradius_le_height_l51_51788


namespace evaluate_expression_l51_51546

-- Defining the conditions for the cosine and sine values
def cos_0 : Real := 1
def sin_3pi_2 : Real := -1

-- Proving the given expression equals -1
theorem evaluate_expression : 3 * cos_0 + 4 * sin_3pi_2 = -1 :=
by 
  -- Given the definitions, this will simplify as expected.
  sorry

end evaluate_expression_l51_51546


namespace pentagon_angle_E_l51_51324

theorem pentagon_angle_E 
    (A B C D E : Type)
    [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
    (AB BC CD DE : ℝ)
    (angle_B angle_C angle_D : ℝ)
    (h1 : AB = BC)
    (h2 : BC = CD)
    (h3 : CD = DE)
    (h4 : angle_B = 96)
    (h5 : angle_C = 108)
    (h6 : angle_D = 108) :
    ∃ angle_E : ℝ, angle_E = 102 := 
by
  sorry

end pentagon_angle_E_l51_51324


namespace num_pos_3_digit_div_by_13_l51_51169

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l51_51169


namespace football_basketball_problem_l51_51260

theorem football_basketball_problem :
  ∃ (football_cost basketball_cost : ℕ),
    (3 * football_cost + basketball_cost = 230) ∧
    (2 * football_cost + 3 * basketball_cost = 340) ∧
    football_cost = 50 ∧
    basketball_cost = 80 ∧
    ∃ (basketballs footballs : ℕ),
      (basketballs + footballs = 20) ∧
      (footballs < basketballs) ∧
      (80 * basketballs + 50 * footballs ≤ 1400) ∧
      ((basketballs = 11 ∧ footballs = 9) ∨
       (basketballs = 12 ∧ footballs = 8) ∨
       (basketballs = 13 ∧ footballs = 7)) :=
by
  sorry

end football_basketball_problem_l51_51260


namespace inequality_of_positive_numbers_l51_51446

theorem inequality_of_positive_numbers (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c :=
sorry

end inequality_of_positive_numbers_l51_51446


namespace minimum_guests_l51_51348

theorem minimum_guests (x : ℕ) : (120 + 18 * x > 250 + 15 * x) → (x ≥ 44) := by
  intro h
  sorry

end minimum_guests_l51_51348


namespace two_leq_one_add_one_div_n_pow_n_lt_three_l51_51488

theorem two_leq_one_add_one_div_n_pow_n_lt_three :
  ∀ (n : ℕ), 2 ≤ (1 + (1 : ℝ) / n) ^ n ∧ (1 + (1 : ℝ) / n) ^ n < 3 := 
by 
  sorry

end two_leq_one_add_one_div_n_pow_n_lt_three_l51_51488


namespace speeds_correct_l51_51725

-- Definitions for conditions
def distance (A B : Type) := 40 -- given distance between A and B is 40 km
def start_time_pedestrian : Real := 4 -- pedestrian starts at 4:00 AM
def start_time_cyclist : Real := 7 + (20 / 60) -- cyclist starts at 7:20 AM
def midpoint_distance : Real := 20 -- the midpoint distance where cyclist catches up with pedestrian is 20 km

noncomputable def speeds (x y : Real) : Prop :=
  let t_catch_up := (20 - (10 / 3) * x) / (y - x) in -- time taken by the cyclist to catch up
  let t_total := (10 / 3) + t_catch_up + 1 in -- total time for pedestrian until meeting second cyclist
  4.5 = t_total ∧ -- total time in hours from 4:00 AM to 8:30 AM
  10 * x * (y - x) + 60 * x - 10 * x^2 = 60 * y - 60 * x ∧ -- initial condition simplification step
  y = 6 * x -- relationship between speeds based on derived equations

-- The proposition to prove
theorem speeds_correct : ∃ x y : Real, speeds x y ∧ x = 5 ∧ y = 30 :=
by
  sorry

end speeds_correct_l51_51725


namespace count_3digit_numbers_div_by_13_l51_51115

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l51_51115


namespace find_speeds_l51_51728

noncomputable def speed_pedestrian := 5
noncomputable def speed_cyclist := 30

def distance_AB := 40
def starting_time_pedestrian := 4 -- In hours (24-hour format)
def starting_time_cyclist_1 := 7 + 20 / 60 -- 7:20 AM in hours
def halfway_distance := distance_AB / 2
def midpoint_meeting_time := 1 -- Time (in hours) after the first meeting
def starting_time_cyclist_2 := 8 + 30 / 60 -- 8:30 AM in hours

theorem find_speeds (x y : ℝ) (hx : x = speed_pedestrian) (hy : y = speed_cyclist) :
  let time_to_halfway := halfway_distance / x in
  let cyclist_time := (midpoint_meeting_time + time_to_halfway) in
  distance_AB = 
    cyclist_time * y + 
    time_to_halfway * x + 
    (midpoint_meeting_time - 1) * x :=
    x = speed_pedestrian ∧ y = speed_cyclist :=
begin
  sorry
end

end find_speeds_l51_51728


namespace rebecca_haircut_charge_l51_51980

-- Define the conditions
variable (H : ℕ) -- Charge for a haircut
def perm_charge : ℕ := 40
def dye_charge : ℕ := 60
def dye_cost : ℕ := 10
def haircuts_today : ℕ := 4
def perms_today : ℕ := 1
def dye_jobs_today : ℕ := 2
def tips_today : ℕ := 50
def total_amount_end_day : ℕ := 310

-- State the proof problem
theorem rebecca_haircut_charge :
  4 * H + perms_today * perm_charge + dye_jobs_today * dye_charge + tips_today - dye_jobs_today * dye_cost = total_amount_end_day →
  H = 30 :=
by
  sorry

end rebecca_haircut_charge_l51_51980


namespace dot_product_square_ABCD_l51_51762

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l51_51762


namespace anne_cleaning_time_l51_51839

theorem anne_cleaning_time :
  ∃ (A B : ℝ), (4 * (B + A) = 1) ∧ (3 * (B + 2 * A) = 1) ∧ (1 / A = 12) :=
by
  sorry

end anne_cleaning_time_l51_51839


namespace percentage_of_whole_l51_51845

theorem percentage_of_whole (part whole percent : ℕ) (h1 : part = 120) (h2 : whole = 80) (h3 : percent = 150) : 
  part = (percent / 100) * whole :=
by
  sorry

end percentage_of_whole_l51_51845


namespace count_three_digit_numbers_divisible_by_13_l51_51124

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l51_51124


namespace area_of_trapezoid_EFGH_l51_51248

-- Define the vertices of the trapezoid
structure Point where
  x : ℤ
  y : ℤ

def E : Point := ⟨-2, -3⟩
def F : Point := ⟨-2, 2⟩
def G : Point := ⟨4, 5⟩
def H : Point := ⟨4, 0⟩

-- Define the formula for the area of a trapezoid
def trapezoid_area (b1 b2 height : ℤ) : ℤ :=
  (b1 + b2) * height / 2

-- The proof statement
theorem area_of_trapezoid_EFGH : trapezoid_area (F.y - E.y) (G.y - H.y) (G.x - E.x) = 30 := by
  sorry -- proof not required

end area_of_trapezoid_EFGH_l51_51248


namespace smallest_k_l51_51464

theorem smallest_k (m n k : ℤ) (h : 221 * m + 247 * n + 323 * k = 2001) (hk : k > 100) : 
∃ k', k' = 111 ∧ k' > 100 :=
by
  sorry

end smallest_k_l51_51464


namespace average_minutes_correct_l51_51700

noncomputable def average_minutes_run_per_day : ℚ :=
  let f (fifth_graders : ℕ) : ℚ := (48 * (4 * fifth_graders) + 30 * (2 * fifth_graders) + 10 * fifth_graders) / (4 * fifth_graders + 2 * fifth_graders + fifth_graders)
  f 1

theorem average_minutes_correct :
  average_minutes_run_per_day = 88 / 7 :=
by
  sorry

end average_minutes_correct_l51_51700


namespace part1_part2_l51_51730

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 1 / (x + a)

theorem part1 (a x : ℝ):
  a ≥ 1 → x > 0 → f x a ≥ 0 := 
sorry

theorem part2 (a : ℝ):
  0 < a ∧ a ≤ 2 / 3 → ∃! x, x > -a ∧ f x a = 0 :=
sorry

end part1_part2_l51_51730


namespace monthly_income_l51_51219

def average_expenditure_6_months (expenditure_6_months : ℕ) (average : ℕ) : Prop :=
  average = expenditure_6_months / 6

def expenditure_next_4_months (expenditure_4_months : ℕ) (monthly_expense : ℕ) : Prop :=
  expenditure_4_months = 4 * monthly_expense

def cleared_debt_and_saved (income_4_months : ℕ) (debt : ℕ) (savings : ℕ)  (condition : ℕ) : Prop :=
  income_4_months = debt + savings + condition

theorem monthly_income 
(income : ℕ) 
(avg_6m_exp : ℕ) 
(exp_4m : ℕ) 
(debt: ℕ) 
(savings: ℕ )
(condition: ℕ) 
    (h1 : average_expenditure_6_months avg_6m_exp 85) 
    (h2 : expenditure_next_4_months exp_4m 60) 
    (h3 : cleared_debt_and_saved (income * 4) debt savings 30) 
    (h4 : income * 6 < 6 * avg_6m_exp) 
    : income = 78 :=
sorry

end monthly_income_l51_51219


namespace type_of_graph_displays_trend_l51_51275

theorem type_of_graph_displays_trend :
  (∃ graph_type : Type, graph_type = "line graph") :=
sorry

end type_of_graph_displays_trend_l51_51275


namespace max_plus_ten_min_eq_zero_l51_51492

theorem max_plus_ten_min_eq_zero (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  let M := max (x * y + x * z + y * z)
  let m := min (x * y + x * z + y * z)
  M + 10 * m = 0 :=
by
  sorry

end max_plus_ten_min_eq_zero_l51_51492


namespace sum_abcd_l51_51838

variables (a b c d : ℚ)

theorem sum_abcd :
  3 * a + 4 * b + 6 * c + 8 * d = 48 →
  4 * (d + c) = b →
  4 * b + 2 * c = a →
  c + 1 = d →
  a + b + c + d = 513 / 37 :=
by
sorry

end sum_abcd_l51_51838


namespace geometric_sequence_a4_l51_51950

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ {m n p q}, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_a4 (a : ℕ → ℝ) (h : geometric_sequence a) (h2 : a 2 = 4) (h6 : a 6 = 16) :
  a 4 = 8 :=
by {
  -- Here you can provide the proof steps if needed
  sorry
}

end geometric_sequence_a4_l51_51950


namespace crucian_carps_heavier_l51_51528

-- Variables representing the weights
variables (K O L : ℝ)

-- Given conditions
axiom weight_6K_lt_5O : 6 * K < 5 * O
axiom weight_6K_gt_10L : 6 * K > 10 * L

-- The proof statement
theorem crucian_carps_heavier : 2 * K > 3 * L :=
by
  -- Proof would go here
  sorry

end crucian_carps_heavier_l51_51528


namespace theater_earnings_l51_51690

theorem theater_earnings :
  let matinee_price := 5
  let evening_price := 7
  let opening_night_price := 10
  let popcorn_price := 10
  let matinee_customers := 32
  let evening_customers := 40
  let opening_night_customers := 58
  let half_of_customers_that_bought_popcorn := 
    (matinee_customers + evening_customers + opening_night_customers) / 2
  let total_earnings := 
    (matinee_price * matinee_customers) + 
    (evening_price * evening_customers) + 
    (opening_night_price * opening_night_customers) + 
    (popcorn_price * half_of_customers_that_bought_popcorn)
  total_earnings = 1670 :=
by
  sorry

end theater_earnings_l51_51690


namespace four_digit_numbers_divisible_by_5_l51_51456

theorem four_digit_numbers_divisible_by_5 : 
  let smallest_4_digit := 1000
  let largest_4_digit := 9999
  let divisible_by_5 (n : ℕ) := ∃ k : ℕ, n = 5 * k
  ∃ n : ℕ, ( ∀ x : ℕ, smallest_4_digit ≤ x ∧ x ≤ largest_4_digit ∧ divisible_by_5 x ↔ (smallest_4_digit + (n-1) * 5 = x) ) ∧ n = 1800 :=
by
  sorry

end four_digit_numbers_divisible_by_5_l51_51456


namespace total_drink_volume_l51_51257

variable (T : ℝ)

theorem total_drink_volume :
  (0.15 * T + 0.60 * T + 0.25 * T = 35) → T = 140 :=
by
  intros h
  have h1 : (0.25 * T) = 35 := by sorry
  have h2 : T = 140 := by sorry
  exact h2

end total_drink_volume_l51_51257


namespace sodium_chloride_solution_l51_51954

theorem sodium_chloride_solution (n y : ℝ) (h1 : n > 30) 
  (h2 : 0.01 * n * n = 0.01 * (n - 8) * (n + y)) : 
  y = 8 * n / (n + 8) :=
sorry

end sodium_chloride_solution_l51_51954


namespace number_of_valid_house_numbers_l51_51016

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

def digit_sum_odd (n : ℕ) : Prop :=
  (n / 10 + n % 10) % 2 = 1

def valid_house_number (W X Y Z : ℕ) : Prop :=
  W ≠ 0 ∧ X ≠ 0 ∧ Y ≠ 0 ∧ Z ≠ 0 ∧
  is_two_digit_prime (10 * W + X) ∧ is_two_digit_prime (10 * Y + Z) ∧
  10 * W + X ≠ 10 * Y + Z ∧
  10 * W + X < 60 ∧ 10 * Y + Z < 60 ∧
  digit_sum_odd (10 * W + X)

theorem number_of_valid_house_numbers : ∃ n, n = 108 ∧
  (∀ W X Y Z, valid_house_number W X Y Z → valid_house_number_count = 108) :=
sorry

end number_of_valid_house_numbers_l51_51016


namespace malcolm_initial_white_lights_l51_51232

-- Definitions based on the conditions
def red_lights : Nat := 12
def blue_lights : Nat := 3 * red_lights
def green_lights : Nat := 6
def total_colored_lights := red_lights + blue_lights + green_lights
def lights_left_to_buy : Nat := 5
def initially_white_lights := total_colored_lights + lights_left_to_buy

-- Proof statement
theorem malcolm_initial_white_lights : initially_white_lights = 59 := by
  sorry

end malcolm_initial_white_lights_l51_51232


namespace markese_earnings_16_l51_51798

theorem markese_earnings_16 (E M : ℕ) (h1 : M = E - 5) (h2 : E + M = 37) : M = 16 :=
by
  sorry

end markese_earnings_16_l51_51798


namespace probability_team_A_champions_l51_51604

theorem probability_team_A_champions : 
  let p : ℚ := 1 / 2 
  let prob_team_A_win_next := p
  let prob_team_B_win_next_A_win_after := p * p
  prob_team_A_win_next + prob_team_B_win_next_A_win_after = 3 / 4 :=
by
  sorry

end probability_team_A_champions_l51_51604


namespace ratio_a_c_l51_51364

variable (a b c d : ℕ)

/-- The given conditions -/
axiom ratio_a_b : a / b = 5 / 2
axiom ratio_c_d : c / d = 4 / 1
axiom ratio_d_b : d / b = 1 / 3

/-- The proof problem -/
theorem ratio_a_c : a / c = 15 / 8 := by
  sorry

end ratio_a_c_l51_51364


namespace range_of_m_l51_51901

theorem range_of_m (f g : ℝ → ℝ) (h1 : ∃ m : ℝ, ∀ x : ℝ, f x = m * (x - m) * (x + m + 3))
  (h2 : ∀ x : ℝ, g x = 2 ^ x - 4)
  (h3 : ∀ x : ℝ, f x < 0 ∨ g x < 0) :
  ∃ m : ℝ, -5 < m ∧ m < 0 :=
sorry

end range_of_m_l51_51901


namespace proposition_not_true_at_9_l51_51862

variable {P : ℕ → Prop}

theorem proposition_not_true_at_9 (h : ∀ k : ℕ, k ≥ 1 → P k → P (k + 1)) (h10 : ¬P 10) : ¬P 9 :=
by
  sorry

end proposition_not_true_at_9_l51_51862


namespace exclude_chairs_l51_51675

-- Definitions
def total_chairs : ℕ := 10000
def perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Statement
theorem exclude_chairs (n : ℕ) (h₁ : n = total_chairs) :
  perfect_square n → (n - total_chairs) = 0 := 
sorry

end exclude_chairs_l51_51675


namespace num_three_digit_div_by_13_l51_51164

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l51_51164


namespace xy_yz_zx_equal_zero_l51_51628

noncomputable def side1 (x y z : ℝ) : ℝ := 1 / abs (x^2 + 2 * y * z)
noncomputable def side2 (x y z : ℝ) : ℝ := 1 / abs (y^2 + 2 * z * x)
noncomputable def side3 (x y z : ℝ) : ℝ := 1 / abs (z^2 + 2 * x * y)

def non_degenerate_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem xy_yz_zx_equal_zero
  (x y z : ℝ)
  (h1 : non_degenerate_triangle (side1 x y z) (side2 x y z) (side3 x y z)) :
  xy + yz + zx = 0 := sorry

end xy_yz_zx_equal_zero_l51_51628


namespace total_employees_l51_51533

-- Defining the number of part-time and full-time employees
def p : ℕ := 2041
def f : ℕ := 63093

-- Statement that the total number of employees is the sum of part-time and full-time employees
theorem total_employees : p + f = 65134 :=
by
  -- Use Lean's built-in arithmetic to calculate the sum
  rfl

end total_employees_l51_51533


namespace Parabola_vertex_form_l51_51182

theorem Parabola_vertex_form (x : ℝ) (y : ℝ) : 
  (∃ h k : ℝ, (h = -2) ∧ (k = 1) ∧ (y = (x + h)^2 + k) ) ↔ (y = (x + 2)^2 + 1) :=
by
  sorry

end Parabola_vertex_form_l51_51182


namespace greatest_multiple_of_30_less_than_800_l51_51515

theorem greatest_multiple_of_30_less_than_800 : 
    ∃ n : ℤ, (n % 30 = 0) ∧ (n < 800) ∧ (∀ m : ℤ, (m % 30 = 0) ∧ (m < 800) → m ≤ n) ∧ n = 780 :=
by
  sorry

end greatest_multiple_of_30_less_than_800_l51_51515


namespace solve_system_of_inequalities_l51_51028

theorem solve_system_of_inequalities (p : ℚ) : (19 * p < 10) ∧ (1 / 2 < p) → (1 / 2 < p ∧ p < 10 / 19) :=
by
  sorry

end solve_system_of_inequalities_l51_51028


namespace modulus_sum_l51_51553

def z1 : ℂ := 3 - 5 * Complex.I
def z2 : ℂ := 3 + 5 * Complex.I

theorem modulus_sum : Complex.abs z1 + Complex.abs z2 = 2 * Real.sqrt 34 := 
by 
  sorry

end modulus_sum_l51_51553


namespace ways_to_go_home_via_library_l51_51236

def ways_from_school_to_library := 2
def ways_from_library_to_home := 3

theorem ways_to_go_home_via_library : 
  ways_from_school_to_library * ways_from_library_to_home = 6 :=
by 
  sorry

end ways_to_go_home_via_library_l51_51236


namespace nth_derivative_ln_correct_l51_51941

noncomputable def nth_derivative_ln (n : ℕ) : ℝ → ℝ
| x => (-1)^(n-1) * (Nat.factorial (n-1)) / (1 + x) ^ n

theorem nth_derivative_ln_correct (n : ℕ) (x : ℝ) :
  deriv^[n] (λ x => Real.log (1 + x)) x = nth_derivative_ln n x := 
by
  sorry

end nth_derivative_ln_correct_l51_51941


namespace ladder_distance_from_wall_l51_51253

theorem ladder_distance_from_wall (h a b : ℕ) (h_hyp : h = 13) (h_wall : a = 12) :
  a^2 + b^2 = h^2 → b = 5 :=
by
  intros h_eq
  sorry

end ladder_distance_from_wall_l51_51253


namespace sum_first_four_terms_geometric_sequence_l51_51951

theorem sum_first_four_terms_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (a₁ : ℝ)
    (h₁ : a 2 = 9)
    (h₂ : a 5 = 243)
    (h₃ : ∀ n, a (n + 1) = a n * r) :
    a₁ + a₁ * r + a₁ * r^2 + a₁ * r^3 = 120 := 
by 
  sorry

end sum_first_four_terms_geometric_sequence_l51_51951


namespace positive_3_digit_numbers_divisible_by_13_count_l51_51078

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l51_51078


namespace area_of_rectangle_inscribed_in_triangle_l51_51493

theorem area_of_rectangle_inscribed_in_triangle :
  ∀ (E F G A B C D : ℝ) (EG altitude_ABCD : ℝ),
    E < F ∧ F < G ∧ A < B ∧ B < C ∧ C < D ∧ A < D ∧ D < G ∧ A < G ∧
    EG = 10 ∧ 
    altitude_ABCD = 7 ∧ 
    B = C ∧ 
    A + D = EG ∧ 
    A + 2 * B = EG →
    ((A * B) = (1225 / 72)) :=
by
  intros E F G A B C D EG altitude_ABCD
  intro h
  sorry

end area_of_rectangle_inscribed_in_triangle_l51_51493


namespace inequality_solution_l51_51557

theorem inequality_solution (x : ℝ) : 3 * x^2 + 7 * x < 6 ↔ -3 < x ∧ x < 2 / 3 := 
sorry

end inequality_solution_l51_51557


namespace count_three_digit_numbers_divisible_by_13_l51_51152

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l51_51152


namespace triangle_sides_and_angles_l51_51241

theorem triangle_sides_and_angles (a : Real) (α β : Real) :
  (a ≥ 0) →
  let sides := [a, a + 1, a + 2]
  let angles := [α, β, 2 * α]
  (∀ s, s ∈ sides) → (∀ θ, θ ∈ angles) →
  a = 4 ∧ a + 1 = 5 ∧ a + 2 = 6 := 
by {
  sorry
}

end triangle_sides_and_angles_l51_51241


namespace count_three_digit_numbers_divisible_by_13_l51_51126

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l51_51126


namespace prop_p_iff_prop_q_iff_not_or_p_q_l51_51338

theorem prop_p_iff (m : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + (2 + m) = 0) ↔ (m ≤ -1 ∨ m ≥ 2) :=
sorry

theorem prop_q_iff (m : ℝ) :
  (∃ x y : ℝ, (x^2)/(1 - 2*m) + (y^2)/(m + 2) = 1) ↔ (m < -2 ∨ m > 1/2) :=
sorry

theorem not_or_p_q (m : ℝ) :
  ¬(∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + (2 + m) = 0) ∧
  ¬(∃ x y : ℝ, (x^2)/(1 - 2*m) + (y^2)/(m + 2) = 1) ↔
  (-1 < m ∧ m ≤ 1/2) :=
sorry

end prop_p_iff_prop_q_iff_not_or_p_q_l51_51338


namespace min_value_inequality_l51_51336

theorem min_value_inequality (θ φ : ℝ) : 
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 549 - 140 * Real.sqrt 5 := 
by
  sorry

end min_value_inequality_l51_51336


namespace triangle_DFG_area_l51_51801

theorem triangle_DFG_area (a b x y : ℝ) (h_ab : a * b = 20) (h_xy : x * y = 8) : 
  (a * b - x * y) / 2 = 6 := 
by
  sorry

end triangle_DFG_area_l51_51801


namespace num_3_digit_div_by_13_l51_51158

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l51_51158


namespace inequality_one_solution_inequality_two_solution_cases_l51_51634

-- Setting up the problem for the first inequality
theorem inequality_one_solution :
  {x : ℝ | -1 ≤ x ∧ x ≤ 4} = {x : ℝ |  -x ^ 2 + 3 * x + 4 ≥ 0} :=
sorry

-- Setting up the problem for the second inequality with different cases of 'a'
theorem inequality_two_solution_cases (a : ℝ) :
  (a = 0 ∧ {x : ℝ | true} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0})
  ∧ (a > 0 ∧ {x : ℝ | x ≥ a - 1 ∨ x ≤ -a - 1} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0})
  ∧ (a < 0 ∧ {x : ℝ | x ≥ -a - 1 ∨ x ≤ a - 1} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0}) :=
sorry

end inequality_one_solution_inequality_two_solution_cases_l51_51634


namespace probability_A_selected_is_three_fourths_l51_51045

-- Definition and the theorem based on the given conditions and question

noncomputable def total_events : ℕ := (nat.choose 4 3)
noncomputable def favorable_events : ℕ := (nat.choose 1 1) * (nat.choose 3 2)
noncomputable def probability_A_selected : ℚ := favorable_events / total_events

theorem probability_A_selected_is_three_fourths : probability_A_selected = 3 / 4 := 
by
  sorry

end probability_A_selected_is_three_fourths_l51_51045


namespace largest_prime_factor_1729_l51_51995

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l51_51995


namespace calculate_amount_l51_51870

theorem calculate_amount (p1 p2 p3: ℝ) : 
  p1 = 0.15 * 4000 ∧ 
  p2 = p1 - 0.25 * p1 ∧ 
  p3 = 0.07 * p2 -> 
  (p3 + 0.10 * p3) = 34.65 := 
by 
  sorry

end calculate_amount_l51_51870


namespace find_k_l51_51185

def triangle_sides (a b c : ℕ) : Prop :=
a < b + c ∧ b < a + c ∧ c < a + b

def is_right_triangle (a b c : ℕ) : Prop :=
a * a + b * b = c * c

def angle_bisector_length (a b c l : ℕ) : Prop :=
∃ k : ℚ, l = k * Real.sqrt 2 ∧ k = 5 / 2

theorem find_k :
  ∀ (AB BC AC BD : ℕ),
  triangle_sides AB BC AC ∧ is_right_triangle AB BC AC ∧
  AB = 5 ∧ BC = 12 ∧ AC = 13 ∧ angle_bisector_length 5 12 13 BD →
  ∃ k : ℚ, BD = k * Real.sqrt 2 ∧ k = 5 / 2 := by
  sorry

end find_k_l51_51185


namespace luxury_class_adults_l51_51351

def total_passengers : ℕ := 300
def adult_percentage : ℝ := 0.70
def luxury_percentage : ℝ := 0.15

def total_adults (p : ℕ) : ℕ := (p * 70) / 100
def adults_in_luxury (a : ℕ) : ℕ := (a * 15) / 100

theorem luxury_class_adults :
  adults_in_luxury (total_adults total_passengers) = 31 :=
by
  sorry

end luxury_class_adults_l51_51351


namespace perimeter_triangle_PQR_is_24_l51_51948

noncomputable def perimeter_triangle_PQR (QR PR : ℝ) : ℝ :=
  let PQ := Real.sqrt (QR^2 + PR^2)
  PQ + QR + PR

theorem perimeter_triangle_PQR_is_24 :
  perimeter_triangle_PQR 8 6 = 24 := by
  sorry

end perimeter_triangle_PQR_is_24_l51_51948


namespace triangle_shape_l51_51598

-- Let there be a triangle ABC with sides opposite to angles A, B, and C being a, b, and c respectively
variables (A B C : ℝ) (a b c : ℝ) (b_ne_1 : b ≠ 1)
          (h1 : (log (b) (C / A)) = (log (sqrt (b)) (2)))
          (h2 : (log (b) (sin B / sin A)) = (log (sqrt (b)) (2)))

-- Define the theorem that states the shape of the triangle
theorem triangle_shape : A = π / 6 ∧ B = π / 2 ∧ C = π / 3 ∧ (A + B + C = π) :=
by
  -- Proof is provided in the solution, skipping proof here
  sorry

end triangle_shape_l51_51598


namespace number_of_3_digit_divisible_by_13_l51_51075

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l51_51075


namespace students_who_chose_water_l51_51017

-- Defining the conditions
def percent_juice : ℚ := 75 / 100
def percent_water : ℚ := 25 / 100
def students_who_chose_juice : ℚ := 90
def ratio_water_to_juice : ℚ := percent_water / percent_juice  -- This should equal 1/3

-- The theorem we need to prove
theorem students_who_chose_water : students_who_chose_juice * ratio_water_to_juice = 30 := 
by
  sorry

end students_who_chose_water_l51_51017


namespace count_three_digit_numbers_divisible_by_13_l51_51129

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l51_51129


namespace speed_of_sound_l51_51531

theorem speed_of_sound (time_blasts : ℝ) (distance_traveled : ℝ) (time_heard : ℝ) (speed : ℝ) 
  (h_blasts : time_blasts = 30 * 60) -- time between the two blasts in seconds 
  (h_distance : distance_traveled = 8250) -- distance in meters
  (h_heard : time_heard = 30 * 60 + 25) -- time when man heard the second blast
  (h_relationship : speed = distance_traveled / (time_heard - time_blasts)) : 
  speed = 330 :=
sorry

end speed_of_sound_l51_51531


namespace count_3_digit_numbers_divisible_by_13_l51_51139

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51139


namespace ny_sales_tax_l51_51389

theorem ny_sales_tax {x : ℝ} 
  (h1 : 100 + x * 1 + 6/100 * (100 + x * 1) = 110) : 
  x = 3.77 :=
by
  sorry

end ny_sales_tax_l51_51389


namespace total_earnings_per_week_correct_l51_51779

noncomputable def weekday_fee_kid : ℝ := 3
noncomputable def weekday_fee_adult : ℝ := 6
noncomputable def weekend_surcharge_ratio : ℝ := 0.5

noncomputable def num_kids_weekday : ℕ := 8
noncomputable def num_adults_weekday : ℕ := 10

noncomputable def num_kids_weekend : ℕ := 12
noncomputable def num_adults_weekend : ℕ := 15

noncomputable def weekday_earnings_kids : ℝ := (num_kids_weekday : ℝ) * weekday_fee_kid
noncomputable def weekday_earnings_adults : ℝ := (num_adults_weekday : ℝ) * weekday_fee_adult

noncomputable def weekday_earnings_total : ℝ := weekday_earnings_kids + weekday_earnings_adults

noncomputable def weekday_earning_per_week : ℝ := weekday_earnings_total * 5

noncomputable def weekend_fee_kid : ℝ := weekday_fee_kid * (1 + weekend_surcharge_ratio)
noncomputable def weekend_fee_adult : ℝ := weekday_fee_adult * (1 + weekend_surcharge_ratio)

noncomputable def weekend_earnings_kids : ℝ := (num_kids_weekend : ℝ) * weekend_fee_kid
noncomputable def weekend_earnings_adults : ℝ := (num_adults_weekend : ℝ) * weekend_fee_adult

noncomputable def weekend_earnings_total : ℝ := weekend_earnings_kids + weekend_earnings_adults

noncomputable def weekend_earning_per_week : ℝ := weekend_earnings_total * 2

noncomputable def total_weekly_earnings : ℝ := weekday_earning_per_week + weekend_earning_per_week

theorem total_earnings_per_week_correct : total_weekly_earnings = 798 := by
  sorry

end total_earnings_per_week_correct_l51_51779


namespace no_adjacent_birch_trees_probability_l51_51407

-- Define the number of trees in each category
def pines : ℕ := 2
def oaks : ℕ := 4
def birches : ℕ := 6

-- Total number of trees
def total_trees : ℕ := pines + oaks + birches

-- Total number of non-birch trees
def non_birch_trees : ℕ := pines + oaks

-- Number of slots created by non-birch trees
def slots : ℕ := non_birch_trees + 1

-- Combinatorial functions to calculate the number of ways
def choose (n k : ℕ) : ℕ := Nat.choose n k

def ways_to_place_birches := choose slots birches
def total_arrangements := choose total_trees birches

-- The probability that no two birch trees are adjacent
def probability_no_adjacent_birch : ℚ :=
  (ways_to_place_birches : ℚ) / total_arrangements

-- The simplified answer should be 1/132
theorem no_adjacent_birch_trees_probability :
  probability_no_adjacent_birch = 1 / 132 :=
begin
  sorry
end

end no_adjacent_birch_trees_probability_l51_51407


namespace find_speeds_l51_51721

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end find_speeds_l51_51721


namespace students_doing_hula_hoops_l51_51496

def number_of_students_jumping_rope : ℕ := 7
def number_of_students_doing_hula_hoops : ℕ := 5 * number_of_students_jumping_rope

theorem students_doing_hula_hoops : number_of_students_doing_hula_hoops = 35 :=
by
  sorry

end students_doing_hula_hoops_l51_51496


namespace tetrahedron_sum_eq_14_l51_51614

theorem tetrahedron_sum_eq_14 :
  let edges := 6
  let corners := 4
  let faces := 4
  edges + corners + faces = 14 :=
by
  let edges := 6
  let corners := 4
  let faces := 4
  show edges + corners + faces = 14
  sorry

end tetrahedron_sum_eq_14_l51_51614


namespace finite_decimals_are_rational_l51_51696

-- Conditions as definitions
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_infinite_decimal (x : ℝ) : Prop := ¬∃ (n : ℤ), x = ↑n
def is_finite_decimal (x : ℝ) : Prop := ∃ (a b : ℕ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- Equivalence to statement C: Finite decimals are rational numbers
theorem finite_decimals_are_rational : ∀ (x : ℝ), is_finite_decimal x → is_rational x := by
  sorry

end finite_decimals_are_rational_l51_51696


namespace Toms_swimming_speed_is_2_l51_51992

theorem Toms_swimming_speed_is_2
  (S : ℝ)
  (h1 : 2 * S + 4 * S = 12) :
  S = 2 :=
by
  sorry

end Toms_swimming_speed_is_2_l51_51992


namespace minimum_value_occurs_at_4_l51_51548

noncomputable def minimum_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, f x ≤ f y

def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 15

theorem minimum_value_occurs_at_4 :
  minimum_value_at quadratic_expression 4 :=
sorry

end minimum_value_occurs_at_4_l51_51548


namespace ten_unique_positive_odd_integers_equality_l51_51367

theorem ten_unique_positive_odd_integers_equality {x : ℕ} (h1: x = 3):
  ∃ S : Finset ℕ, S.card = 10 ∧ 
    (∀ n ∈ S, n < 100 ∧ n % 2 = 1 ∧ 
      ∃ k : ℕ, k % 2 = 1 ∧ n = k * x) :=
by
  sorry

end ten_unique_positive_odd_integers_equality_l51_51367


namespace connor_cats_l51_51201

theorem connor_cats (j : ℕ) (a : ℕ) (m : ℕ) (c : ℕ) (co : ℕ) (x : ℕ) 
  (h1 : a = j / 3)
  (h2 : m = 2 * a)
  (h3 : c = a / 2)
  (h4 : c = co + 5)
  (h5 : j = 90)
  (h6 : x = j + a + m + c + co) : 
  co = 10 := 
by
  sorry

end connor_cats_l51_51201


namespace part_a_l51_51526

theorem part_a (x : ℝ) (hx : x ≥ 1) : x^3 - 5 * x^2 + 8 * x - 4 ≥ 0 := 
  sorry

end part_a_l51_51526


namespace total_items_proof_l51_51317

noncomputable def totalItemsBought (budget : ℕ) (sandwichCost : ℕ) 
  (pastryCost : ℕ) (maxSandwiches : ℕ) : ℕ :=
  let s := min (budget / sandwichCost) maxSandwiches
  let remainingMoney := budget - s * sandwichCost
  let p := remainingMoney / pastryCost
  s + p

theorem total_items_proof : totalItemsBought 50 6 2 7 = 11 := by
  sorry

end total_items_proof_l51_51317


namespace count_3_digit_numbers_divisible_by_13_l51_51092

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51092


namespace friends_professions_l51_51895

theorem friends_professions
  (architect barista veterinarian guitarist : ℕ)
  (P : ℕ → Sort)
  (Andrey Boris Vyacheslav Gennady : P)
  (seats : P → ℕ)
  (h1 : ∀ {x y}, seats x = veterinarian → seats y = architect → (seats x = seats y + 1 ∨ seats x = seats y - 1))
  (h2 : ∀ {x y}, seats x = barista → seats y = Boris → seats y = seats x + 1)
  (h3 : seats Vyacheslav > seats Andrey ∧ seats Vyacheslav > seats Boris)
  (h4 : ∀ {x}, x = Andrey → ∃ y z, (seats y = seats x + 1 ∨ seats y = seats x - 1) ∧ (seats z = seats x + 1 ∨ seats z = seats x - 1))
  (h5 : ∀ {x y}, seats x = guitarist → seats y = barista → seats x ≠ seats y + 1 ∧ seats x ≠ seats y - 1)
  : (∀ x, (x = Gennady → seats x = barista)
    ∧ (x = Boris → seats x = architect)
    ∧ (x = Andrey → seats x = veterinarian)
    ∧ (x = Vyacheslav → seats x = guitarist)) :=
by
  sorry

end friends_professions_l51_51895


namespace exists_good_10_element_subset_l51_51786

theorem exists_good_10_element_subset (f : Finset ℕ → ℕ)
  (h₁ : ∀ S, S.card = 9 → ∃ n, n ∈ S ∧ f S = n) :
  ∃ T : Finset ℕ, T.card = 10 ∧ ∀ k ∈ T, f (T.erase k) ≠ k :=
by
  let M := (Finset.range 21).erase 0
  have hM : M.card = 20 := by simp [Finset.card_erase_of_mem, Finset.mem_range]
  obtain ⟨T, hT₁, hT₂⟩ := exists_good_set M f h₁ hM
  exact ⟨T, hT₁, hT₂⟩


end exists_good_10_element_subset_l51_51786


namespace largest_number_value_l51_51651

theorem largest_number_value 
  (a b c : ℚ)
  (h_sum : a + b + c = 100)
  (h_diff1 : c - b = 10)
  (h_diff2 : b - a = 5) : 
  c = 125 / 3 := 
sorry

end largest_number_value_l51_51651


namespace find_x_l51_51309

theorem find_x 
  (x : ℕ)
  (h : 3^x = 3^(20) * 3^(20) * 3^(18) + 3^(19) * 3^(20) * 3^(19) + 3^(18) * 3^(21) * 3^(19)) :
  x = 59 :=
sorry

end find_x_l51_51309


namespace quadratic_solution_l51_51505

theorem quadratic_solution (x : ℝ) (h : 2 * x ^ 2 - 2 = 0) : x = 1 ∨ x = -1 :=
sorry

end quadratic_solution_l51_51505


namespace count_3_digit_numbers_divisible_by_13_l51_51089

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51089


namespace find_speeds_l51_51719

noncomputable def speed_proof_problem (x y: ℝ) : Prop :=
  let distance_AB := 40
  let time_cyclist_start := 7 + 20 / 60
  let time_pedestrian_start := 4
  let time_cyclist_to_catch_up := (distance_AB / 2 - 10 / 3 * x) / (y - x)
  let time_pedestrian_meet := 10 / 3 + time_cyclist_to_catch_up + 1
  let time_second_cyclist_start := 8.5
  let dist_cyclist := y * (time_second_cyclist_start - time_pedestrian_start)
  let dist_pedestrian := x * time_pedestrian_meet 
  (x = 5 ∧ y = 30) ∧
  (time_cyclist_start - time_pedestrian_start = 10 / 3) ∧
  (dist_pedestrian + time_cyclist_to_catch_up * x = distance_AB / 2) ∧
  (dist_pedestrian + y * 1 = 40)

theorem find_speeds (x y: ℝ) :
  speed_proof_problem x y :=
sorry

end find_speeds_l51_51719


namespace exp_inequality_l51_51490

theorem exp_inequality (n : ℕ) (h : 0 < n) : 2 ≤ (1 + 1 / (n : ℝ)) ^ n ∧ (1 + 1 / (n : ℝ)) ^ n < 3 :=
sorry

end exp_inequality_l51_51490


namespace value_increase_factor_l51_51372

theorem value_increase_factor (P S : ℝ) (frac F : ℝ) (hP : P = 200) (hS : S = 240) (hfrac : frac = 0.40) :
  frac * (P * F) = S -> F = 3 := by
  sorry

end value_increase_factor_l51_51372


namespace train_time_l51_51001

theorem train_time (T : ℕ) (D : ℝ) (h1 : D = 48 * (T / 60)) (h2 : D = 60 * (40 / 60)) : T = 50 :=
by
  sorry

end train_time_l51_51001


namespace proof_problem_theorem_l51_51757

noncomputable def proof_problem : Prop :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 2)
  let D : ℝ × ℝ := (0, 2)
  let E : ℝ × ℝ := (1, 0)
  let vector := (p1 p2 : ℝ × ℝ) → (p2.1 - p1.1, p2.2 - p1.2)
  let dot_product := (u v : ℝ × ℝ) → u.1 * v.1 + u.2 * v.2
  let EC := vector E C
  let ED := vector E D
  EC ∘ ED = 3

theorem proof_problem_theorem : proof_problem := 
by 
  sorry

end proof_problem_theorem_l51_51757


namespace animath_workshop_lists_l51_51217

/-- The 79 trainees of the Animath workshop each choose an activity for the free afternoon 
among 5 offered activities. It is known that:
- The swimming pool was at least as popular as soccer.
- The students went shopping in groups of 5.
- No more than 4 students played cards.
- At most one student stayed in their room.
We write down the number of students who participated in each activity.
How many different lists could we have written? --/
theorem animath_workshop_lists :
  ∃ (l : ℕ), l = Nat.choose 81 2 := 
sorry

end animath_workshop_lists_l51_51217


namespace three_digit_numbers_div_by_13_l51_51109

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l51_51109


namespace clock_correct_time_fraction_l51_51529

/-- 
  A 24-hour digital clock displays the hour and minute of a day, 
  counting from 00:00 to 23:59. However, due to a glitch, whenever 
  the clock is supposed to display a '2', it mistakenly displays a '5'.

  Prove that the fraction of a day during which the clock shows the correct 
  time is 23/40.
-/
theorem clock_correct_time_fraction :
  let total_hours := 24
  let affected_hours := 6
  let correct_hours := total_hours - affected_hours
  let total_minutes := 60
  let affected_minutes := 14
  let correct_minutes := total_minutes - affected_minutes
  (correct_hours / total_hours) * (correct_minutes / total_minutes) = 23 / 40 :=
by
  let total_hours := 24
  let affected_hours := 6
  let correct_hours := total_hours - affected_hours
  let total_minutes := 60
  let affected_minutes := 14
  let correct_minutes := total_minutes - affected_minutes
  have h1 : correct_hours = 18 := rfl
  have h2 : correct_minutes = 46 := rfl
  have h3 : 18 / 24 = 3 / 4 := by norm_num
  have h4 : 46 / 60 = 23 / 30 := by norm_num
  have h5 : (3 / 4) * (23 / 30) = 23 / 40 := by norm_num
  exact h5

end clock_correct_time_fraction_l51_51529


namespace solve_system_of_inequalities_l51_51027

theorem solve_system_of_inequalities (p : ℚ) : (19 * p < 10) ∧ (1 / 2 < p) → (1 / 2 < p ∧ p < 10 / 19) :=
by
  sorry

end solve_system_of_inequalities_l51_51027


namespace contractor_fine_per_absent_day_l51_51680

noncomputable def fine_per_absent_day (total_days : ℕ) (pay_per_day : ℝ) (total_amount_received : ℝ) (days_absent : ℕ) : ℝ :=
  let days_worked := total_days - days_absent
  let earned := days_worked * pay_per_day
  let fine := (earned - total_amount_received) / days_absent
  fine

theorem contractor_fine_per_absent_day :
  fine_per_absent_day 30 25 425 10 = 7.5 := by
  sorry

end contractor_fine_per_absent_day_l51_51680


namespace valentines_count_l51_51486

theorem valentines_count (x y : ℕ) (h1 : (x = 2 ∧ y = 48) ∨ (x = 48 ∧ y = 2)) : 
  x * y - (x + y) = 46 := by
  sorry

end valentines_count_l51_51486


namespace max_min_sum_difference_l51_51200

-- The statement that we need to prove
theorem max_min_sum_difference : 
  ∃ (max_sum min_sum: ℕ), (∀ (RST UVW XYZ : ℕ),
   -- Constraints for Max's and Minnie's sums respectively
   (RST = 100 * 9 + 10 * 6 + 3 ∧ UVW = 100 * 8 + 10 * 5 + 2 ∧ XYZ = 100 * 7 + 10 * 4 + 1 → max_sum = 2556) ∧ 
   (RST = 100 * 1 + 10 * 0 + 6 ∧ UVW = 100 * 2 + 10 * 4 + 7 ∧ XYZ = 100 * 3 + 10 * 5 + 8 → min_sum = 711)) → 
    max_sum - min_sum = 1845 :=
by
  sorry

end max_min_sum_difference_l51_51200


namespace solve_system_of_inequalities_l51_51026

theorem solve_system_of_inequalities (p : ℚ) : (19 * p < 10) ∧ (1 / 2 < p) → (1 / 2 < p ∧ p < 10 / 19) :=
by
  sorry

end solve_system_of_inequalities_l51_51026


namespace min_y_value_l51_51478

noncomputable def min_value_y : ℝ :=
  18 - 2 * Real.sqrt 106

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20 * x + 36 * y) : 
  y >= 18 - 2 * Real.sqrt 106 :=
sorry

end min_y_value_l51_51478


namespace minimize_water_tank_construction_cost_l51_51684

theorem minimize_water_tank_construction_cost 
  (volume : ℝ := 4800)
  (depth : ℝ := 3)
  (cost_bottom_per_m2 : ℝ := 150)
  (cost_walls_per_m2 : ℝ := 120)
  (x : ℝ) :
  (volume = x * x * depth) →
  (∀ y, y = cost_bottom_per_m2 * x * x + cost_walls_per_m2 * 4 * x * depth) →
  (x = 40) ∧ (y = 297600) :=
by
  sorry

end minimize_water_tank_construction_cost_l51_51684


namespace additional_houses_built_by_october_l51_51005

def total_houses : ℕ := 2000
def fraction_built_first_half : ℚ := 3 / 5
def houses_needed_by_october : ℕ := 500

def houses_built_first_half : ℚ := fraction_built_first_half * total_houses
def houses_built_by_october : ℕ := total_houses - houses_needed_by_october

theorem additional_houses_built_by_october :
  (houses_built_by_october - houses_built_first_half) = 300 := by
  sorry

end additional_houses_built_by_october_l51_51005


namespace geometric_sequence_common_ratio_l51_51605

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = -1) 
  (h2 : a 2 + a 3 = -2) 
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  q = -2 ∨ q = 1 := 
by sorry

end geometric_sequence_common_ratio_l51_51605


namespace find_speeds_l51_51722

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end find_speeds_l51_51722


namespace three_digit_numbers_divisible_by_13_count_l51_51135

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l51_51135


namespace small_bottles_initial_l51_51000

theorem small_bottles_initial
  (S : ℤ)
  (big_bottles_initial : ℤ := 15000)
  (sold_small_bottles_percentage : ℚ := 0.11)
  (sold_big_bottles_percentage : ℚ := 0.12)
  (remaining_bottles_in_storage : ℤ := 18540)
  (remaining_small_bottles : ℚ := 0.89 * S)
  (remaining_big_bottles : ℚ := 0.88 * big_bottles_initial)
  (h : remaining_small_bottles + remaining_big_bottles = remaining_bottles_in_storage)
  : S = 6000 :=
by
  sorry

end small_bottles_initial_l51_51000


namespace unique_integer_sequence_l51_51630

theorem unique_integer_sequence (a : ℕ → ℤ) :
  a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, a (n + 1)^3 + 1 = a n * a (n + 2)) →
  ∃! (a : ℕ → ℤ), a 1 = 1 ∧ a 2 > 1 ∧ (∀ n ≥ 1, a (n + 1)^3 + 1 = a n * a (n + 2)) :=
sorry

end unique_integer_sequence_l51_51630


namespace jordan_rect_width_is_10_l51_51670

def carol_rect_length : ℕ := 5
def carol_rect_width : ℕ := 24
def jordan_rect_length : ℕ := 12

def carol_rect_area : ℕ := carol_rect_length * carol_rect_width
def jordan_rect_width := carol_rect_area / jordan_rect_length

theorem jordan_rect_width_is_10 : jordan_rect_width = 10 :=
by
  sorry

end jordan_rect_width_is_10_l51_51670


namespace num_pos_3_digit_div_by_13_l51_51171

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l51_51171


namespace average_minutes_run_per_day_l51_51699

theorem average_minutes_run_per_day (f : ℕ) :
  let third_grade_minutes := 12
  let fourth_grade_minutes := 15
  let fifth_grade_minutes := 10
  let third_graders := 4 * f
  let fourth_graders := 2 * f
  let fifth_graders := f
  let total_minutes := third_graders * third_grade_minutes + fourth_graders * fourth_grade_minutes + fifth_graders * fifth_grade_minutes
  let total_students := third_graders + fourth_graders + fifth_graders
  total_minutes / total_students = 88 / 7 :=
by
  sorry

end average_minutes_run_per_day_l51_51699


namespace merchants_tea_cups_l51_51510

theorem merchants_tea_cups (a b c : ℕ) 
  (h1 : a + b = 11)
  (h2 : b + c = 15)
  (h3 : a + c = 14) : 
  a + b + c = 20 :=
by
  sorry

end merchants_tea_cups_l51_51510


namespace ratio_copper_to_zinc_l51_51867

theorem ratio_copper_to_zinc (copper zinc : ℝ) (hc : copper = 24) (hz : zinc = 10.67) : (copper / zinc) = 2.25 :=
by
  rw [hc, hz]
  -- Add the arithmetic operation
  sorry

end ratio_copper_to_zinc_l51_51867


namespace count_three_digit_numbers_divisible_by_13_l51_51125

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l51_51125


namespace directrix_of_parabola_l51_51712

-- Define the given parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

-- Define the directrix equation for the parabola
def directrix_eq : ℝ := -13 / 12

-- Problem statement: Proof that the equation of the directrix of the parabola is given by directrix_eq.
theorem directrix_of_parabola : ∀ x : ℝ, (∃ (a b c : ℝ), parabola x = a*x^2 + b*x + c) → (∃ y : ℝ, y = directrix_eq) :=
by
  intros x H
  use directrix_eq
  sorry

end directrix_of_parabola_l51_51712


namespace find_speeds_l51_51727

noncomputable def speed_pedestrian := 5
noncomputable def speed_cyclist := 30

def distance_AB := 40
def starting_time_pedestrian := 4 -- In hours (24-hour format)
def starting_time_cyclist_1 := 7 + 20 / 60 -- 7:20 AM in hours
def halfway_distance := distance_AB / 2
def midpoint_meeting_time := 1 -- Time (in hours) after the first meeting
def starting_time_cyclist_2 := 8 + 30 / 60 -- 8:30 AM in hours

theorem find_speeds (x y : ℝ) (hx : x = speed_pedestrian) (hy : y = speed_cyclist) :
  let time_to_halfway := halfway_distance / x in
  let cyclist_time := (midpoint_meeting_time + time_to_halfway) in
  distance_AB = 
    cyclist_time * y + 
    time_to_halfway * x + 
    (midpoint_meeting_time - 1) * x :=
    x = speed_pedestrian ∧ y = speed_cyclist :=
begin
  sorry
end

end find_speeds_l51_51727


namespace profit_calculation_l51_51350

def totalProfit (totalMoney part1 interest1 interest2 time : ℕ) : ℕ :=
  let part2 := totalMoney - part1
  let interestFromPart1 := part1 * interest1 / 100 * time
  let interestFromPart2 := part2 * interest2 / 100 * time
  interestFromPart1 + interestFromPart2

theorem profit_calculation : 
  totalProfit 80000 70000 10 20 1 = 9000 :=
  by 
    -- Rather than providing a full proof, we insert 'sorry' as per the instruction.
    sorry

end profit_calculation_l51_51350


namespace no_three_segments_form_triangle_l51_51474

theorem no_three_segments_form_triangle :
  ∃ (a : Fin 10 → ℕ), ∀ {i j k : Fin 10}, i < j → j < k → a i + a j ≤ a k :=
by
  sorry

end no_three_segments_form_triangle_l51_51474


namespace system_of_equations_solution_l51_51436

theorem system_of_equations_solution :
  ∃ x y : ℚ, x = 2 * y ∧ 2 * x - y = 5 ∧ x = 10 / 3 ∧ y = 5 / 3 :=
by
  sorry

end system_of_equations_solution_l51_51436


namespace sum_of_diagonals_l51_51958

def FG : ℝ := 4
def HI : ℝ := 4
def GH : ℝ := 11
def IJ : ℝ := 11
def FJ : ℝ := 15

theorem sum_of_diagonals (x y z : ℝ) (h1 : z^2 = 4 * x + 121) (h2 : z^2 = 11 * y + 16)
  (h3 : x * y = 44 + 15 * z) (h4 : x * z = 4 * z + 225) (h5 : y * z = 11 * z + 60) :
  3 * z + x + y = 90 :=
sorry

end sum_of_diagonals_l51_51958


namespace find_P_Q_R_l51_51287

theorem find_P_Q_R :
  ∃ P Q R : ℝ, (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 → 
    (5 * x / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2)) 
    ∧ P = 5 ∧ Q = -5 ∧ R = -5 :=
by
  sorry

end find_P_Q_R_l51_51287


namespace choose_3_from_12_l51_51752

theorem choose_3_from_12 : (Nat.choose 12 3) = 220 := by
  sorry

end choose_3_from_12_l51_51752


namespace maximize_wind_power_l51_51815

variable {C S ρ v_0 : ℝ}

theorem maximize_wind_power : 
  ∃ v : ℝ, (∀ (v' : ℝ),
           let F := (C * S * ρ * (v_0 - v)^2) / 2;
           let N := F * v;
           let N' := (C * S * ρ / 2) * (v_0^2 - 4 * v_0 * v + 3 * v^2);
           N' = 0
         → N ≤ (C * S * ρ / 2) * (v_0^2 * (v_0/3) - 2 * v_0 * (v_0/3)^2 + (v_0/3)^3)) ∧ v = v_0 / 3 :=
by sorry

end maximize_wind_power_l51_51815


namespace larry_channel_reduction_l51_51192

theorem larry_channel_reduction
  (initial_channels new_channels final_channels sports_package supreme_sports_package channels_at_end : ℕ)
  (h_initial : initial_channels = 150)
  (h_adjustment : new_channels = initial_channels - 20 + 12)
  (h_sports : sports_package = 8)
  (h_supreme_sports : supreme_sports_package = 7)
  (h_channels_at_end : channels_at_end = 147)
  (h_final : final_channels = channels_at_end - sports_package - supreme_sports_package) :
  initial_channels - 20 + 12 - final_channels = 10 := 
sorry

end larry_channel_reduction_l51_51192


namespace evaluate_expression_l51_51280

def ceil (x : ℚ) : ℤ := sorry -- Implement the ceiling function for rational numbers as needed

theorem evaluate_expression :
  (ceil ((23 : ℚ) / 9 - ceil ((35 : ℚ) / 23))) 
  / (ceil ((35 : ℚ) / 9 + ceil ((9 * 23 : ℚ) / 35))) = (1 / 10 : ℚ) :=
by
  intros
  -- Proof goes here
  sorry

end evaluate_expression_l51_51280


namespace maximize_power_speed_l51_51814

variable (C S ρ v₀ : ℝ)

-- Given the formula for force F
def force (v : ℝ) : ℝ := (C * S * ρ * (v₀ - v)^2) / 2

-- Given the formula for power N
def power (v : ℝ) : ℝ := force C S ρ v₀ v * v

theorem maximize_power_speed : ∀ C S ρ v₀ : ℝ, ∃ v : ℝ, v = v₀ / 3 ∧ (∀ v' : ℝ, power C S ρ v₀ v ≤ power C S ρ v₀ v') :=
by
  sorry

end maximize_power_speed_l51_51814


namespace solution_for_system_of_inequalities_l51_51037

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end solution_for_system_of_inequalities_l51_51037


namespace regular_square_pyramid_side_edge_length_l51_51736

theorem regular_square_pyramid_side_edge_length 
  (base_edge_length : ℝ)
  (volume : ℝ)
  (h_base_edge_length : base_edge_length = 4 * Real.sqrt 2)
  (h_volume : volume = 32) :
  ∃ side_edge_length : ℝ, side_edge_length = 5 :=
by sorry

end regular_square_pyramid_side_edge_length_l51_51736


namespace inverse_of_square_l51_51173

theorem inverse_of_square (A : Matrix (Fin 2) (Fin 2) ℝ) (hA_inv : A⁻¹ = ![![3, 4], ![-2, -2]]) :
  (A^2)⁻¹ = ![![1, 4], ![-2, -4]] :=
by
  sorry

end inverse_of_square_l51_51173


namespace trig_identity_simplify_l51_51982

-- Define the problem in Lean 4
theorem trig_identity_simplify (α : Real) : (Real.sin (α - Real.pi / 2) * Real.tan (Real.pi - α)) = Real.sin α :=
by
  sorry

end trig_identity_simplify_l51_51982


namespace distance_between_points_l51_51663

theorem distance_between_points : 
  let p1 := (3, -2) 
  let p2 := (-7, 4) 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 136 :=
by
  sorry

end distance_between_points_l51_51663


namespace find_k_l51_51987

theorem find_k (σ μ : ℝ) (hσ : σ = 2) (hμ : μ = 55) :
  ∃ k : ℝ, μ - k * σ > 48 ∧ k = 3 :=
by
  sorry

end find_k_l51_51987


namespace greatest_multiple_of_5_and_6_less_than_800_l51_51517

theorem greatest_multiple_of_5_and_6_less_than_800 : 
  ∃ n : ℕ, n < 800 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 800 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
sorry

end greatest_multiple_of_5_and_6_less_than_800_l51_51517


namespace solve_abs_eq_l51_51875

theorem solve_abs_eq (x : ℝ) : 
    (3 * x + 9 = abs (-20 + 4 * x)) ↔ 
    (x = 29) ∨ (x = 11 / 7) := 
by sorry

end solve_abs_eq_l51_51875


namespace full_price_tickets_revenue_l51_51261

-- Define the conditions and then prove the statement
theorem full_price_tickets_revenue (f d p : ℕ) (h1 : f + d = 200) (h2 : f * p + d * (p / 3) = 3000) : f * p = 1500 := by
  sorry

end full_price_tickets_revenue_l51_51261


namespace quadratic_roots_squared_sum_l51_51298

theorem quadratic_roots_squared_sum (m n : ℝ) (h1 : m^2 - 2 * m - 1 = 0) (h2 : n^2 - 2 * n - 1 = 0) : m^2 + n^2 = 6 :=
sorry

end quadratic_roots_squared_sum_l51_51298


namespace find_hourly_rate_l51_51540

theorem find_hourly_rate (x : ℝ) (h1 : 40 * x + 10.75 * 16 = 622) : x = 11.25 :=
sorry

end find_hourly_rate_l51_51540


namespace preimage_of_point_l51_51905

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

-- Define the statement of the problem
theorem preimage_of_point {x y : ℝ} (h1 : f x y = (3, 1)) : (x = 2 ∧ y = 1) :=
by
  sorry

end preimage_of_point_l51_51905


namespace count_100_digit_even_numbers_l51_51916

theorem count_100_digit_even_numbers : 
  let valid_digits := {0, 1, 3}
  let num_digits := 100
  let num_even_digits := 2 * 3^98
  ∀ n : ℕ, n = num_digits → (∃ (digits : Fin n → ℕ), 
    (∀ i, digits i ∈ valid_digits) ∧ 
    digits 0 ≠ 0 ∧ 
    digits (n-1) = 0) → 
    (num_even_digits = 2 * 3^98) :=
by
  sorry

end count_100_digit_even_numbers_l51_51916


namespace pow_mod_1000_of_6_eq_296_l51_51660

theorem pow_mod_1000_of_6_eq_296 : (6 ^ 1993) % 1000 = 296 := by
  sorry

end pow_mod_1000_of_6_eq_296_l51_51660


namespace sum_first_twelve_multiples_17_sum_squares_first_twelve_multiples_17_l51_51832

-- Definitions based on conditions
def sum_arithmetic (n : ℕ) : ℕ := n * (n + 1) / 2
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Theorem statements based on the correct answers
theorem sum_first_twelve_multiples_17 : 
  17 * sum_arithmetic 12 = 1326 := 
by
  sorry

theorem sum_squares_first_twelve_multiples_17 : 
  17^2 * sum_squares 12 = 187850 :=
by
  sorry

end sum_first_twelve_multiples_17_sum_squares_first_twelve_multiples_17_l51_51832


namespace fraction_value_unchanged_l51_51949

theorem fraction_value_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / (x + y) = (2 * x) / (2 * (x + y))) :=
by sorry

end fraction_value_unchanged_l51_51949


namespace minimize_quadratic_expression_l51_51550

theorem minimize_quadratic_expression:
  ∀ x : ℝ, (∃ a b c : ℝ, a = 1 ∧ b = -8 ∧ c = 15 ∧ x^2 + b * x + c ≥ (4 - 4)^2 - 1) :=
by
  sorry

end minimize_quadratic_expression_l51_51550


namespace return_speed_is_48_l51_51849

variable (d r : ℕ)
variable (t_1 t_2 : ℚ)

-- Given conditions
def distance_each_way : Prop := d = 120
def time_to_travel_A_to_B : Prop := t_1 = d / 80
def time_to_travel_B_to_A : Prop := t_2 = d / r
def average_speed_round_trip : Prop := 60 * (t_1 + t_2) = 2 * d

-- Statement to prove
theorem return_speed_is_48 :
  distance_each_way d ∧
  time_to_travel_A_to_B d t_1 ∧
  time_to_travel_B_to_A d r t_2 ∧
  average_speed_round_trip d t_1 t_2 →
  r = 48 :=
by
  intros
  sorry

end return_speed_is_48_l51_51849


namespace ellipse_distance_range_l51_51603

theorem ellipse_distance_range {d : ℝ} :
  let f1 := (-1 : ℝ, 0 : ℝ)
  let f2 := (1 : ℝ, 0 : ℝ)
  let ellipse := { p : ℝ × ℝ | p.1^2 + 2 * p.2^2 = 2 }
  (∃ (k m : ℝ), m ≠ -1 ∧
    let l := { p : ℝ × ℝ | p.1 = k * p.2 + m }
    (∃ (A B : ℝ × ℝ), A ≠ B ∧ A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ l ∧ B ∈ l ∧
     ∃ (a1 a2 : ℝ), 
      a1 = (A.2 - 0) / (A.1 + 1) ∧
      a2 = (B.2 - 0) / (B.1 + 1) ∧
      let mid_slope := (a1 + a2) / 2
      (a1, mid_slope, a2) being an arithmetic_seq ∧
    (distance f2 l) = d)) ∧
  (d > sqrt 3 ∧ d < 2) :=
begin
  sorry
end

end ellipse_distance_range_l51_51603


namespace compare_doubling_l51_51174

theorem compare_doubling (a b : ℝ) (h : a > b) : 2 * a > 2 * b :=
  sorry

end compare_doubling_l51_51174


namespace volume_of_prism_l51_51520

noncomputable def prismVolume {x y z : ℝ} 
  (h1 : x * y = 20) 
  (h2 : y * z = 12) 
  (h3 : x * z = 8) : ℝ :=
  x * y * z

theorem volume_of_prism (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 12)
  (h3 : x * z = 8) : prismVolume h1 h2 h3 = 8 * Real.sqrt 15 :=
by
  sorry

end volume_of_prism_l51_51520


namespace soccer_games_per_month_l51_51656

theorem soccer_games_per_month (total_games : ℕ) (months : ℕ) (h1 : total_games = 27) (h2 : months = 3) : total_games / months = 9 :=
by 
  sorry

end soccer_games_per_month_l51_51656


namespace find_x_l51_51556

noncomputable def x : ℝ := 10.3

theorem find_x (h1 : x + (⌈x⌉ : ℝ) = 21.3) (h2 : x > 0) : x = 10.3 :=
sorry

end find_x_l51_51556


namespace solve_for_x_l51_51807

theorem solve_for_x :
  ∀ x : ℝ, 4 * x + 9 * x = 360 - 9 * (x - 4) → x = 18 :=
by
  intros x h
  sorry

end solve_for_x_l51_51807


namespace tom_cheaper_than_jane_l51_51829

-- Define constants for Store A
def store_a_full_price : ℝ := 125
def store_a_discount_one : ℝ := 0.08
def store_a_discount_two : ℝ := 0.12
def store_a_tax : ℝ := 0.07

-- Define constants for Store B
def store_b_full_price : ℝ := 130
def store_b_discount_one : ℝ := 0.10
def store_b_discount_three : ℝ := 0.15
def store_b_tax : ℝ := 0.05

-- Define the number of smartphones bought by Tom and Jane
def tom_quantity : ℕ := 2
def jane_quantity : ℕ := 3

-- Define the final amount Tom pays
def final_amount_tom : ℝ :=
  let full_price := tom_quantity * store_a_full_price
  let discount := store_a_discount_two * full_price
  let discounted_price := full_price - discount
  let tax := store_a_tax * discounted_price
  discounted_price + tax

-- Define the final amount Jane pays
def final_amount_jane : ℝ :=
  let full_price := jane_quantity * store_b_full_price
  let discount := store_b_discount_three * full_price
  let discounted_price := full_price - discount
  let tax := store_b_tax * discounted_price
  discounted_price + tax

-- Prove that Tom's total cost is $112.68 cheaper than Jane's total cost
theorem tom_cheaper_than_jane : final_amount_jane - final_amount_tom = 112.68 :=
by
  have tom := final_amount_tom
  have jane := final_amount_jane
  sorry

end tom_cheaper_than_jane_l51_51829


namespace max_g_f_inequality_l51_51576

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := f x - x / 4 - 1

theorem max_g : ∃ x : ℝ, g x = 2 * Real.log 2 - 7 / 4 :=
sorry

theorem f_inequality (x : ℝ) (hx : 0 < x) : f x < (Real.exp x - 1) / x^2 :=
sorry

end max_g_f_inequality_l51_51576


namespace count_3digit_numbers_div_by_13_l51_51116

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l51_51116


namespace count_3_digit_numbers_divisible_by_13_l51_51066

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51066


namespace q_sufficient_not_necessary_for_p_l51_51562

def p (x : ℝ) : Prop := abs x < 2
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

theorem q_sufficient_not_necessary_for_p (x : ℝ) : (q x → p x) ∧ ¬(p x → q x) := 
by
  sorry

end q_sufficient_not_necessary_for_p_l51_51562


namespace dance_problem_l51_51365

theorem dance_problem :
  ∃ (G : ℝ) (B T : ℝ),
    B / G = 3 / 4 ∧
    T = 0.20 * B ∧
    B + G + T = 114 ∧
    G = 60 :=
by
  sorry

end dance_problem_l51_51365


namespace distance_between_A_and_B_l51_51245

variable (d : ℝ) -- Total distance between A and B

def car_speeds (vA vB t : ℝ) : Prop :=
vA = 80 ∧ vB = 100 ∧ t = 2

def total_covered_distance (vA vB t : ℝ) : ℝ :=
(vA + vB) * t

def percentage_distance (total_distance covered_distance : ℝ) : Prop :=
0.6 * total_distance = covered_distance

theorem distance_between_A_and_B (vA vB t : ℝ) (H1 : car_speeds vA vB t) 
  (H2 : percentage_distance d (total_covered_distance vA vB t)) : d = 600 := by
  sorry

end distance_between_A_and_B_l51_51245


namespace answer_l51_51759

-- Definitions of geometric entities in terms of vectors
structure Square :=
  (A B C D E : ℝ × ℝ)
  (side_length : ℝ)
  (hAB_eq : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = side_length ^ 2)
  (hBC_eq : (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = side_length ^ 2)
  (hCD_eq : (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = side_length ^ 2)
  (hDA_eq : (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2 = side_length ^ 2)
  (hE_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def EC_ED_dot_product (s : Square) : ℝ :=
  let EC := (s.C.1 - s.E.1, s.C.2 - s.E.2)
  let ED := (s.D.1 - s.E.1, s.D.2 - s.E.2)
  dot_product EC ED

theorem answer (s : Square) (h_side_length : s.side_length = 2) :
  EC_ED_dot_product s = 3 :=
sorry

end answer_l51_51759


namespace number_of_cats_adopted_l51_51955

theorem number_of_cats_adopted (c : ℕ) 
  (h1 : 50 * c + 3 * 100 + 2 * 150 = 700) :
  c = 2 :=
by
  sorry

end number_of_cats_adopted_l51_51955


namespace sin_eq_solutions_l51_51930

theorem sin_eq_solutions :
  (∃ count : ℕ, 
    count = 4007 ∧ 
    (∀ (x : ℝ), 
      0 ≤ x ∧ x ≤ 2 * Real.pi → 
      (∃ (k1 k2 : ℤ), 
        x = -2 * k1 * Real.pi ∨ 
        x = 2 * Real.pi ∨ 
        x = (2 * k2 + 1) * Real.pi / 4005)
    )) :=
sorry

end sin_eq_solutions_l51_51930


namespace factors_of_180_l51_51926

theorem factors_of_180 : ∃ n : ℕ, n = 18 ∧ ∀ p q r : ℕ, 180 = p^2 * q^2 * r^1 → 
  n = (p + 1) * (q + 1) * (r) :=
by
  sorry

end factors_of_180_l51_51926


namespace three_digit_numbers_divisible_by_13_l51_51145

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l51_51145


namespace cafeteria_problem_l51_51599

theorem cafeteria_problem (C : ℕ) 
    (h1 : ∃ h : ℕ, h = 4 * C)
    (h2 : 5 = 5)
    (h3 : C + 4 * C + 5 = 40) : 
    C = 7 := sorry

end cafeteria_problem_l51_51599


namespace find_non_integer_solution_l51_51476

noncomputable def q (x y : ℝ) (b : Fin 10 → ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 +
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3

theorem find_non_integer_solution (b : Fin 10 → ℝ)
  (h0 : q 0 0 b = 0)
  (h1 : q 1 0 b = 0)
  (h2 : q (-1) 0 b = 0)
  (h3 : q 0 1 b = 0)
  (h4 : q 0 (-1) b = 0)
  (h5 : q 1 1 b = 0)
  (h6 : q 1 (-1) b = 0)
  (h7 : q (-1) 1 b = 0)
  (h8 : q (-1) (-1) b = 0) :
  ∃ r s : ℝ, q r s b = 0 ∧ ¬ (∃ n : ℤ, r = n) ∧ ¬ (∃ n : ℤ, s = n) :=
sorry

end find_non_integer_solution_l51_51476


namespace koschei_coins_l51_51329

theorem koschei_coins :
  ∃ a : ℕ, (a % 10 = 7) ∧ (a % 12 = 9) ∧ (300 ≤ a) ∧ (a ≤ 400) ∧ a = 357 :=
by 
  sorry

end koschei_coins_l51_51329


namespace monthly_income_of_labourer_l51_51222

variable (I : ℕ) -- Monthly income

-- Conditions: 
def condition1 := (85 * 6) - (6 * I) -- A boolean expression depicting the labourer fell into debt
def condition2 := (60 * 4) + (85 * 6 - 6 * I) + 30 -- Total income covers debt and saving 30

-- Statement to be proven
theorem monthly_income_of_labourer : 
  ∃ I : ℕ, condition1 I = 0 ∧ condition2 I = 4 * I → I = 78 :=
by
  sorry

end monthly_income_of_labourer_l51_51222


namespace candy_remainder_l51_51461

theorem candy_remainder :
  38759863 % 6 = 1 :=
by
  sorry

end candy_remainder_l51_51461


namespace num_3_digit_div_by_13_l51_51157

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l51_51157


namespace count_three_digit_numbers_divisible_by_13_l51_51128

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l51_51128


namespace inequality_positives_l51_51979

theorem inequality_positives (x1 x2 x3 x4 x5 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) (hx5 : 0 < x5) : 
  (x1 + x2 + x3 + x4 + x5)^2 ≥ 4 * (x1 * x2 + x3 * x4 + x5 * x1 + x2 * x3 + x4 * x5) :=
sorry

end inequality_positives_l51_51979


namespace base_of_exponent_l51_51310

theorem base_of_exponent (b x y : ℕ) (h1 : x - y = 12) (h2 : x = 12) (h3 : b^x * 4^y = 531441) : b = 3 :=
by
  sorry

end base_of_exponent_l51_51310


namespace num_3_digit_div_by_13_l51_51159

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l51_51159


namespace proof_problem_theorem_l51_51755

noncomputable def proof_problem : Prop :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 2)
  let D : ℝ × ℝ := (0, 2)
  let E : ℝ × ℝ := (1, 0)
  let vector := (p1 p2 : ℝ × ℝ) → (p2.1 - p1.1, p2.2 - p1.2)
  let dot_product := (u v : ℝ × ℝ) → u.1 * v.1 + u.2 * v.2
  let EC := vector E C
  let ED := vector E D
  EC ∘ ED = 3

theorem proof_problem_theorem : proof_problem := 
by 
  sorry

end proof_problem_theorem_l51_51755


namespace total_cost_of_shirt_and_sweater_l51_51989

-- Define the given conditions
def price_of_shirt := 36.46
def diff_price_shirt_sweater := 7.43
def price_of_sweater := price_of_shirt + diff_price_shirt_sweater

-- Statement to prove
theorem total_cost_of_shirt_and_sweater :
  price_of_shirt + price_of_sweater = 80.35 :=
by
  -- Proof goes here
  sorry

end total_cost_of_shirt_and_sweater_l51_51989


namespace minimum_surface_area_of_circumscribed_sphere_of_prism_l51_51952

theorem minimum_surface_area_of_circumscribed_sphere_of_prism :
  ∃ S : ℝ, 
    (∀ h r, r^2 * h = 4 → r^2 + (h^2 / 4) = R → 4 * π * R^2 = S) ∧ 
    (∀ S', S' ≤ S) ∧ 
    S = 12 * π :=
sorry

end minimum_surface_area_of_circumscribed_sphere_of_prism_l51_51952


namespace sequence_sum_l51_51664

def alternating_sum : List ℤ := [2, -7, 10, -15, 18, -23, 26, -31, 34, -39, 40, -45, 48]

theorem sequence_sum : alternating_sum.sum = 13 := by
  sorry

end sequence_sum_l51_51664


namespace misha_darts_score_l51_51972

theorem misha_darts_score (x : ℕ) 
  (h1 : x >= 24)
  (h2 : x * 3 <= 72) : 
  2 * x = 48 :=
by
  sorry

end misha_darts_score_l51_51972


namespace painted_cells_l51_51781

theorem painted_cells (k l : ℕ) (h : k * l = 74) :
    (2 * k + 1) * (2 * l + 1) - k * l = 301 ∨ 
    (2 * k + 1) * (2 * l + 1) - k * l = 373 :=
sorry

end painted_cells_l51_51781


namespace inequality_solution_set_l51_51504

theorem inequality_solution_set (x : ℝ) : (2 * x + 1 ≥ 3) ∧ (4 * x - 1 < 7) ↔ (1 ≤ x ∧ x < 2) :=
by
  sorry

end inequality_solution_set_l51_51504


namespace count_3_digit_numbers_divisible_by_13_l51_51091

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51091


namespace jigsaw_puzzle_completion_l51_51971

theorem jigsaw_puzzle_completion (p : ℝ) :
  let total_pieces := 1000
  let pieces_first_day := total_pieces * 0.10
  let remaining_after_first_day := total_pieces - pieces_first_day

  let pieces_second_day := remaining_after_first_day * (p / 100)
  let remaining_after_second_day := remaining_after_first_day - pieces_second_day

  let pieces_third_day := remaining_after_second_day * 0.30
  let remaining_after_third_day := remaining_after_second_day - pieces_third_day

  remaining_after_third_day = 504 ↔ p = 20 := 
by {
    sorry
}

end jigsaw_puzzle_completion_l51_51971


namespace remainder_gx12_div_gx_l51_51965

-- Definition of the polynomial g(x)
def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Theorem stating the problem
theorem remainder_gx12_div_gx : ∀ x : ℂ, (g (x^12)) % (g x) = 6 := by
  sorry

end remainder_gx12_div_gx_l51_51965


namespace distance_from_P_to_other_focus_on_ellipse_l51_51048

noncomputable def distance_to_other_focus (P : ℝ × ℝ) : ℝ :=
  let a := 5 in   -- derived from a^2 = 25
  2 * a - 4

theorem distance_from_P_to_other_focus_on_ellipse {x y : ℝ}
  (h_ellipse : x^2 / 25 + y^2 / 16 = 1)
  (h_distance : ∃ P1, (x = 5 * real.cos P1) ∧ (y = 4 * real.sin P1) ∧ dist (x, y) (5, 0) = 4) :
  distance_to_other_focus (x, y) = 6 :=
sorry

end distance_from_P_to_other_focus_on_ellipse_l51_51048


namespace remaining_bread_after_three_days_l51_51791

namespace BreadProblem

def InitialBreadCount : ℕ := 200

def FirstDayConsumption (bread : ℕ) : ℕ := bread / 4
def SecondDayConsumption (remainingBreadAfterFirstDay : ℕ) : ℕ := 2 * remainingBreadAfterFirstDay / 5
def ThirdDayConsumption (remainingBreadAfterSecondDay : ℕ) : ℕ := remainingBreadAfterSecondDay / 2

theorem remaining_bread_after_three_days : 
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  breadAfterThirdDay = 45 := 
by
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  have : breadAfterThirdDay = 45 := sorry
  exact this

end BreadProblem

end remaining_bread_after_three_days_l51_51791


namespace part1_part2_case1_part2_case2_part2_case3_1_part2_case3_2_part2_case3_3_l51_51479

def f (a x : ℝ) : ℝ := a * x ^ 2 + (1 - a) * x + a - 2

theorem part1 (a : ℝ) : (∀ x : ℝ, f a x ≥ -2) ↔ a ≥ 1/3 :=
sorry

theorem part2_case1 (a : ℝ) (ha : a = 0) : ∀ x : ℝ, f a x < a - 1 ↔ x < 1 :=
sorry

theorem part2_case2 (a : ℝ) (ha : a > 0) : ∀ x : ℝ, (f a x < a - 1) ↔ (-1 / a < x ∧ x < 1) :=
sorry

theorem part2_case3_1 (a : ℝ) (ha : a = -1) : ∀ x : ℝ, (f a x < a - 1) ↔ x ≠ 1 :=
sorry

theorem part2_case3_2 (a : ℝ) (ha : -1 < a ∧ a < 0) : ∀ x : ℝ, (f a x < a - 1) ↔ (x > -1 / a ∨ x < 1) :=
sorry

theorem part2_case3_3 (a : ℝ) (ha : a < -1) : ∀ x : ℝ, (f a x < a - 1) ↔ (x > 1 ∨ x < -1 / a) :=
sorry

end part1_part2_case1_part2_case2_part2_case3_1_part2_case3_2_part2_case3_3_l51_51479


namespace count_three_digit_div_by_13_l51_51101

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l51_51101


namespace problem_statement_l51_51909

def f : ℝ → ℝ :=
  sorry

lemma even_function (x : ℝ) : f (-x) = f x :=
  sorry

lemma periodicity (x : ℝ) (hx : 0 ≤ x) : f (x + 2) = -f x :=
  sorry

lemma value_in_interval (x : ℝ) (hx : 0 ≤ x ∧ x < 2) : f x = Real.log (x + 1) :=
  sorry

theorem problem_statement : f (-2001) + f 2012 = 1 :=
  sorry

end problem_statement_l51_51909


namespace smallest_positive_integer_congruence_l51_51381

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 5 * x ≡ 18 [MOD 31] ∧ 0 < x ∧ x < 31 ∧ x = 16 := 
by sorry

end smallest_positive_integer_congruence_l51_51381


namespace isoland_license_plates_proof_l51_51813

def isoland_license_plates : ℕ :=
  let letters := ['A', 'B', 'D', 'E', 'I', 'L', 'N', 'O', 'R', 'U']
  let valid_letters := letters.erase 'B'
  let first_letter_choices := ['A', 'I']
  let last_letter := 'R'
  let remaining_letters:= valid_letters.erase last_letter
  (first_letter_choices.length * (remaining_letters.length - first_letter_choices.length) * (remaining_letters.length - first_letter_choices.length - 1) * (remaining_letters.length - first_letter_choices.length - 2))

theorem isoland_license_plates_proof :
  isoland_license_plates = 420 := by
  sorry

end isoland_license_plates_proof_l51_51813


namespace part1_part2_l51_51561

theorem part1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1) (h4 : a^2 + 4 * b^2 + c^2 - 2 * c = 2) : 
  a + 2 * b + c ≤ 4 :=
sorry

theorem part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1) (h4 : a^2 + 4 * b^2 + c^2 - 2 * c = 2) (h5 : a = 2 * b) : 
  1 / b + 1 / (c - 1) ≥ 3 :=
sorry

end part1_part2_l51_51561


namespace approx_d_l51_51427

noncomputable def close_approx_d : ℝ :=
  let d := (69.28 * (0.004)^3 - Real.log 27) / (0.03 * Real.cos (55 * Real.pi / 180))
  d

theorem approx_d : |close_approx_d + 191.297| < 0.001 :=
  by
    -- Proof goes here.
    sorry

end approx_d_l51_51427


namespace count_3_digit_numbers_divisible_by_13_l51_51069

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51069


namespace part_one_part_two_l51_51559

theorem part_one (a b : ℝ) (h : a ≠ 0) : |a + b| + |a - b| ≥ 2 * |a| :=
by sorry

theorem part_two (x : ℝ) : |x - 1| + |x - 2| ≤ 2 ↔ (1 / 2 : ℝ) ≤ x ∧ x ≤ (5 / 2 : ℝ) :=
by sorry

end part_one_part_two_l51_51559


namespace anns_age_l51_51417

theorem anns_age (a b : ℕ) (h1 : a + b = 54) 
(h2 : b = a - (a - b) + (a - b)): a = 29 :=
sorry

end anns_age_l51_51417


namespace calculate_expression_l51_51420

theorem calculate_expression (a b : ℝ) : (a - b) * (a + b) * (a^2 - b^2) = a^4 - 2 * a^2 * b^2 + b^4 := 
by
  sorry

end calculate_expression_l51_51420


namespace combined_weight_of_boxes_l51_51611

-- Defining the weights of each box as constants
def weight1 : ℝ := 2.5
def weight2 : ℝ := 11.3
def weight3 : ℝ := 5.75
def weight4 : ℝ := 7.2
def weight5 : ℝ := 3.25

-- The main theorem statement
theorem combined_weight_of_boxes : weight1 + weight2 + weight3 + weight4 + weight5 = 30 := by
  sorry

end combined_weight_of_boxes_l51_51611


namespace find_number_l51_51292

theorem find_number (x : ℝ) (h : x = 12) : ( ( 17.28 / x ) / ( 3.6 * 0.2 ) ) = 2 := 
by
  -- Proof will be here
  sorry

end find_number_l51_51292


namespace total_tiles_in_square_hall_l51_51687

theorem total_tiles_in_square_hall
  (s : ℕ) -- integer side length of the square hall
  (black_tiles : ℕ)
  (total_tiles : ℕ)
  (all_tiles_white_or_black : ∀ (x : ℕ), x ≤ total_tiles → x = black_tiles ∨ x = total_tiles - black_tiles)
  (black_tiles_count : black_tiles = 153 + 3) : total_tiles = 6084 :=
by
  sorry

end total_tiles_in_square_hall_l51_51687


namespace log_inequality_l51_51932

theorem log_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : 
  Real.log b / Real.log a + Real.log a / Real.log b ≤ -2 := sorry

end log_inequality_l51_51932


namespace greatest_possible_value_of_a_l51_51639

theorem greatest_possible_value_of_a :
  ∃ a : ℕ, (∀ x : ℤ, x * (x + a) = -12) → a = 13 := by
  sorry

end greatest_possible_value_of_a_l51_51639


namespace product_is_48_l51_51637

-- Define the conditions and the target product
def problem (x y : ℝ) := 
  x ≠ y ∧ (x + y) / (x - y) = 7 ∧ (x * y) / (x - y) = 24

-- Prove that the product is 48 given the conditions
theorem product_is_48 (x y : ℝ) (h : problem x y) : x * y = 48 :=
sorry

end product_is_48_l51_51637


namespace solution_difference_l51_51480

theorem solution_difference (m n : ℝ) (h_eq : ∀ x : ℝ, (x - 4) * (x + 4) = 24 * x - 96 ↔ x = m ∨ x = n) (h_distinct : m ≠ n) (h_order : m > n) : m - n = 16 :=
sorry

end solution_difference_l51_51480


namespace number_of_3_digit_divisible_by_13_l51_51071

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l51_51071


namespace range_of_a_l51_51361

open Real

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem range_of_a (a : ℝ) (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) :
  (f (a * sin θ) + f (1 - a) > 0) → a ≤ 1 :=
sorry

end range_of_a_l51_51361


namespace system_of_inequalities_solution_l51_51033

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end system_of_inequalities_solution_l51_51033


namespace relationship_p_q_l51_51482

noncomputable def p (α β : ℝ) : ℝ := Real.cos α * Real.cos β
noncomputable def q (α β : ℝ) : ℝ := Real.cos ((α + β) / 2) ^ 2

theorem relationship_p_q (α β : ℝ) : p α β ≤ q α β :=
by
  sorry

end relationship_p_q_l51_51482


namespace convert_10203_base4_to_base10_l51_51705

def base4_to_base10 (n : ℕ) (d₀ d₁ d₂ d₃ d₄ : ℕ) : ℕ :=
  d₄ * 4^4 + d₃ * 4^3 + d₂ * 4^2 + d₁ * 4^1 + d₀ * 4^0

theorem convert_10203_base4_to_base10 :
  base4_to_base10 10203 3 0 2 0 1 = 291 :=
by
  -- proof goes here
  sorry

end convert_10203_base4_to_base10_l51_51705


namespace count_3_digit_numbers_divisible_by_13_l51_51058

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l51_51058


namespace proof_problem_theorem_l51_51754

noncomputable def proof_problem : Prop :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 2)
  let D : ℝ × ℝ := (0, 2)
  let E : ℝ × ℝ := (1, 0)
  let vector := (p1 p2 : ℝ × ℝ) → (p2.1 - p1.1, p2.2 - p1.2)
  let dot_product := (u v : ℝ × ℝ) → u.1 * v.1 + u.2 * v.2
  let EC := vector E C
  let ED := vector E D
  EC ∘ ED = 3

theorem proof_problem_theorem : proof_problem := 
by 
  sorry

end proof_problem_theorem_l51_51754


namespace max_value_of_quadratic_l51_51518

theorem max_value_of_quadratic :
  ∃ (x : ℝ), ∀ (y : ℝ), -3 * y^2 + 18 * y - 5 ≤ -3 * x^2 + 18 * x - 5 ∧ -3 * x^2 + 18 * x - 5 = 22 :=
sorry

end max_value_of_quadratic_l51_51518


namespace correct_M_min_t_for_inequality_l51_51302

-- Define the set M
def M : Set ℝ := {a | 0 ≤ a ∧ a < 4}

-- Prove that M is correct given ax^2 + ax + 2 > 0 for all x ∈ ℝ implies 0 ≤ a < 4
theorem correct_M (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + 2 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

-- Prove the minimum value of t given t > 0 and the inequality holds for all a ∈ M
theorem min_t_for_inequality (t : ℝ) (h : 0 < t) : 
  (∀ a ∈ M, (a^2 - 2 * a) * t ≤ t^2 + 3 * t - 46) ↔ 46 ≤ t :=
sorry

end correct_M_min_t_for_inequality_l51_51302


namespace find_angle_A_l51_51597

theorem find_angle_A (a b : ℝ) (B A : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 2) (h3 : B = Real.pi / 4) : A = Real.pi / 6 :=
by
  sorry

end find_angle_A_l51_51597


namespace initial_garrison_men_l51_51853

theorem initial_garrison_men (M : ℕ) (h1 : 62 * M = 62 * M) 
  (h2 : M * 47 = (M + 2700) * 20) : M = 2000 := by
  sorry

end initial_garrison_men_l51_51853


namespace caterpillar_to_scorpion_ratio_l51_51010

theorem caterpillar_to_scorpion_ratio 
  (roach_count : ℕ) (scorpion_count : ℕ) (total_insects : ℕ) 
  (h_roach : roach_count = 12) 
  (h_scorpion : scorpion_count = 3) 
  (h_cricket : cricket_count = roach_count / 2) 
  (h_total : total_insects = 27) 
  (h_non_cricket_count : non_cricket_count = roach_count + scorpion_count + cricket_count) 
  (h_caterpillar_count : caterpillar_count = total_insects - non_cricket_count) : 
  (caterpillar_count / scorpion_count) = 2 := 
by 
  sorry

end caterpillar_to_scorpion_ratio_l51_51010


namespace pencils_loss_equates_20_l51_51206

/--
Patrick purchased 70 pencils and sold them at a loss equal to the selling price of some pencils. The cost of 70 pencils is 1.2857142857142856 times the selling price of 70 pencils. Prove that the loss equates to the selling price of 20 pencils.
-/
theorem pencils_loss_equates_20 
  (C S : ℝ) 
  (h1 : C = 1.2857142857142856 * S) :
  (70 * C - 70 * S) = 20 * S :=
by
  sorry

end pencils_loss_equates_20_l51_51206


namespace original_cost_of_tomatoes_correct_l51_51931

noncomputable def original_cost_of_tomatoes := 
  let original_order := 25
  let new_tomatoes := 2.20
  let new_lettuce := 1.75
  let old_lettuce := 1.00
  let new_celery := 2.00
  let old_celery := 1.96
  let delivery_tip := 8
  let new_total_bill := 35
  let new_groceries := new_total_bill - delivery_tip
  let increase_in_cost := (new_lettuce - old_lettuce) + (new_celery - old_celery)
  let difference_due_to_substitutions := new_groceries - original_order
  let x := new_tomatoes + (difference_due_to_substitutions - increase_in_cost)
  x

theorem original_cost_of_tomatoes_correct :
  original_cost_of_tomatoes = 3.41 := by
  sorry

end original_cost_of_tomatoes_correct_l51_51931


namespace count_three_digit_div_by_13_l51_51100

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l51_51100


namespace total_cost_of_shirt_and_sweater_l51_51990

theorem total_cost_of_shirt_and_sweater (S : ℝ) : 
  (S - 7.43 = 36.46) → (36.46 + S = 80.35) :=
by
  assume h1 : S - 7.43 = 36.46
  sorry

end total_cost_of_shirt_and_sweater_l51_51990


namespace divisible_by_6_l51_51881

theorem divisible_by_6 (n : ℤ) : 6 ∣ (n^3 - n + 6) :=
by
  sorry

end divisible_by_6_l51_51881


namespace count_three_digit_div_by_13_l51_51103

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l51_51103


namespace variance_of_data_l51_51825

def data : List ℝ := [0.7, 1, 0.8, 0.9, 1.1]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.foldr (λ x acc => x + acc) 0) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.foldr (λ x acc => (x - m) ^ 2 + acc) 0) / l.length

theorem variance_of_data :
  variance data = 0.02 :=
by
  sorry

end variance_of_data_l51_51825


namespace increasing_log_condition_range_of_a_l51_51052

noncomputable def t (x a : ℝ) := x^2 - a*x + 3*a

theorem increasing_log_condition :
  (∀ x ≥ 2, 2 * x - a ≥ 0) ∧ a > -4 ∧ a ≤ 4 →
  ∀ x ≥ 2, x^2 - a*x + 3*a > 0 :=
by
  sorry

theorem range_of_a
  (h1 : ∀ x ≥ 2, 2 * x - a ≥ 0)
  (h2 : 4 - 2 * a + 3 * a > 0)
  (h3 : ∀ x ≥ 2, t x a > 0)
  : a > -4 ∧ a ≤ 4 :=
by
  sorry

end increasing_log_condition_range_of_a_l51_51052


namespace roots_quadratic_solution_l51_51567

theorem roots_quadratic_solution (α β : ℝ) (hα : α^2 - 3*α - 2 = 0) (hβ : β^2 - 3*β - 2 = 0) :
  3*α^3 + 8*β^4 = 1229 := by
  sorry

end roots_quadratic_solution_l51_51567


namespace term_with_largest_binomial_coeffs_and_largest_coefficient_l51_51737

theorem term_with_largest_binomial_coeffs_and_largest_coefficient :
  ∀ x : ℝ,
    (∀ k : ℕ, k = 2 → (Nat.choose 5 k) * (x ^ (2 / 3)) ^ (5 - k) * (3 * x ^ 2) ^ k = 90 * x ^ 6) ∧
    (∀ k : ℕ, k = 3 → (Nat.choose 5 k) * (x ^ (2 / 3)) ^ (5 - k) * (3 * x ^ 2) ^ k = 270 * x ^ (22 / 3)) ∧
    (∀ r : ℕ, r = 4 → (Nat.choose 5 4) * (x ^ (2 / 3)) ^ (5 - 4) * (3 * x ^ 2) ^ 4 = 405 * x ^ (26 / 3)) :=
by sorry

end term_with_largest_binomial_coeffs_and_largest_coefficient_l51_51737


namespace count_3_digit_multiples_of_13_l51_51095

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l51_51095


namespace only_D_is_quadratic_l51_51835

-- Conditions
def eq_A (x : ℝ) : Prop := x^2 + 1/x - 1 = 0
def eq_B (x : ℝ) : Prop := (2*x + 1) + x = 0
def eq_C (m x : ℝ) : Prop := 2*m^2 + x = 3
def eq_D (x : ℝ) : Prop := x^2 - x = 0

-- Proof statement
theorem only_D_is_quadratic :
  ∃ (x : ℝ), eq_D x ∧ 
  (¬(∃ x : ℝ, eq_A x) ∧ ¬(∃ x : ℝ, eq_B x) ∧ ¬(∃ (m x : ℝ), eq_C m x)) :=
by
  sorry

end only_D_is_quadratic_l51_51835


namespace hendecagon_diagonals_l51_51879

-- Define the number of sides n of the hendecagon
def n : ℕ := 11

-- Define the formula for calculating the number of diagonals in an n-sided polygon
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem that there are 44 diagonals in a hendecagon
theorem hendecagon_diagonals : diagonals n = 44 :=
by
  -- Proof is skipped using sorry
  sorry

end hendecagon_diagonals_l51_51879


namespace positive_3_digit_numbers_divisible_by_13_count_l51_51081

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l51_51081


namespace sum_of_variables_l51_51731

theorem sum_of_variables (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0) : 
  x + y + z = 2 :=
sorry

end sum_of_variables_l51_51731


namespace three_digit_numbers_divisible_by_13_l51_51146

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l51_51146


namespace solution_for_system_of_inequalities_l51_51039

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end solution_for_system_of_inequalities_l51_51039


namespace tetrahedron_area_theorem_l51_51357

noncomputable def tetrahedron_faces_areas_and_angles
  (a b c d : ℝ) (α β γ : ℝ) : Prop :=
  d^2 = a^2 + b^2 + c^2 - 2 * a * b * Real.cos γ - 2 * b * c * Real.cos α - 2 * c * a * Real.cos β

theorem tetrahedron_area_theorem
  (a b c d : ℝ) (α β γ : ℝ) :
  tetrahedron_faces_areas_and_angles a b c d α β γ :=
sorry

end tetrahedron_area_theorem_l51_51357


namespace dot_product_EC_ED_l51_51767

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l51_51767


namespace coterminal_angle_l51_51184

theorem coterminal_angle (theta : ℝ) (lower : ℝ) (upper : ℝ) (k : ℤ) : 
  -950 = k * 360 + theta ∧ (lower ≤ theta ∧ theta ≤ upper) → theta = 130 :=
by
  -- Given conditions
  sorry

end coterminal_angle_l51_51184


namespace mapping_has_output_l51_51565

variable (M N : Type) (f : M → N)

theorem mapping_has_output (x : M) : ∃ y : N, f x = y :=
by
  sorry

end mapping_has_output_l51_51565


namespace num_three_digit_div_by_13_l51_51162

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l51_51162


namespace count_6_digit_palindromes_with_even_middle_digits_l51_51305

theorem count_6_digit_palindromes_with_even_middle_digits :
  let a_values := 9
  let b_even_values := 5
  let c_values := 10
  a_values * b_even_values * c_values = 450 :=
by {
  sorry
}

end count_6_digit_palindromes_with_even_middle_digits_l51_51305


namespace sum_of_perpendiculars_eq_altitude_l51_51426

variables {A B C P A' B' C' : Type*}
variables (AB AC BC PA' PB' PC' h : ℝ)

-- Conditions
def is_isosceles_triangle (AB AC BC : ℝ) : Prop :=
  AB = AC

def point_inside_triangle (P A B C : Type*) : Prop :=
  true -- Assume point P is inside the triangle

def is_perpendiculars_dropped (PA' PB' PC' : ℝ) : Prop :=
  true -- Assume PA', PB', PC' are the lengths of the perpendiculars from P to the sides BC, CA, AB

def base_of_triangle (BC : ℝ) : Prop :=
  true -- Assume BC is the base of triangle

-- Theorem statement
theorem sum_of_perpendiculars_eq_altitude
  (h : ℝ) (AB AC BC PA' PB' PC' : ℝ)
  (isosceles : is_isosceles_triangle AB AC BC)
  (point_inside_triangle' : point_inside_triangle P A B C)
  (perpendiculars_dropped : is_perpendiculars_dropped PA' PB' PC')
  (base_of_triangle' : base_of_triangle BC) : 
  PA' + PB' + PC' = h := 
sorry

end sum_of_perpendiculars_eq_altitude_l51_51426


namespace positive_3_digit_numbers_divisible_by_13_count_l51_51076

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l51_51076


namespace verify_calculations_l51_51834

theorem verify_calculations (m n x y a b : ℝ) :
  (2 * m - 3 * n) ^ 2 = 4 * m ^ 2 - 12 * m * n + 9 * n ^ 2 ∧
  (-x + y) ^ 2 = x ^ 2 - 2 * x * y + y ^ 2 ∧
  (a + 2 * b) * (a - 2 * b) = a ^ 2 - 4 * b ^ 2 ∧
  (-2 * x ^ 2 * y ^ 2) ^ 3 / (- x * y) ^ 3 ≠ -2 * x ^ 3 * y ^ 3 :=
by
  sorry

end verify_calculations_l51_51834


namespace solve_for_s_l51_51353

theorem solve_for_s (s : ℝ) (t : ℝ) (h1 : t = 8 * s^2) (h2 : t = 4.8) : s = Real.sqrt 0.6 ∨ s = -Real.sqrt 0.6 := by
  sorry

end solve_for_s_l51_51353


namespace part_a_part_b_part_c_l51_51654

-- Definitions for the convex polyhedron, volume, and surface area
structure ConvexPolyhedron :=
  (volume : ℝ)
  (surface_area : ℝ)

variable {P : ConvexPolyhedron}

-- Statement for Part (a)
theorem part_a (r : ℝ) (h_r : r ≤ P.surface_area) :
  P.volume / P.surface_area ≥ r / 3 := sorry

-- Statement for Part (b)
theorem part_b :
  Exists (fun r : ℝ => r = P.volume / P.surface_area) := sorry

-- Definitions and conditions for the outer and inner polyhedron
structure ConvexPolyhedronPair :=
  (outer_polyhedron : ConvexPolyhedron)
  (inner_polyhedron : ConvexPolyhedron)

variable {CP : ConvexPolyhedronPair}

-- Statement for Part (c)
theorem part_c :
  3 * CP.outer_polyhedron.volume / CP.outer_polyhedron.surface_area ≥
  CP.inner_polyhedron.volume / CP.inner_polyhedron.surface_area := sorry

end part_a_part_b_part_c_l51_51654


namespace number_of_3_digit_divisible_by_13_l51_51072

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l51_51072


namespace masha_lives_on_seventh_floor_l51_51799

/-- Masha lives in apartment No. 290, which is in the 4th entrance of a 17-story building.
The number of apartments is the same in all entrances of the building on all 17 floors; apartment numbers start from 1.
We need to prove that Masha lives on the 7th floor. -/
theorem masha_lives_on_seventh_floor 
  (n_apartments_per_floor : ℕ) 
  (total_floors : ℕ := 17) 
  (entrances : ℕ := 4) 
  (masha_apartment : ℕ := 290) 
  (start_apartment : ℕ := 1) 
  (h1 : (masha_apartment - start_apartment + 1) > 0) 
  (h2 : masha_apartment ≤ entrances * total_floors * n_apartments_per_floor)
  (h4 : masha_apartment > (entrances - 1) * total_floors * n_apartments_per_floor)  
   : ((masha_apartment - ((entrances - 1) * total_floors * n_apartments_per_floor) - 1) / n_apartments_per_floor) + 1 = 7 := 
by
  sorry

end masha_lives_on_seventh_floor_l51_51799


namespace seating_arrangement_l51_51898

-- Definitions of friends and their professions
inductive Friend
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq

-- Seating condition definitions
def seated_right (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = a ∧ seating (i + 1) = b)

def seated_between (a b c : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, seating i = b ∧ seating (i - 1) = a ∧ seating (i + 1) = c)

def neighbors (a b : Friend) (seating : ℕ → Friend) : Prop :=
  (∃ i, (seating i = a ∧ seating (i - 1) = b) ∨ (seating i = a ∧ seating (i + 1) = b))

-- The proposition to be proved
theorem seating_arrangement
  (seating : ℕ → Friend)
  (profession : Friend → Profession)
  (h1 : ∃ i, seating_between Profession.Veterinarian Profession.Architect Profession.Guitarist seating)
  (h2 : seated_right Profession.Barista Profession.Boris seating)
  (h3 : ∃ i, seating i = Friend.Vyacheslav ∧ (∃ j, seating j = Friend.Andrey ∧ j < i) ∧ (∃ k, seating k = Friend.Boris ∧ k < i))
  (h4 : ∀ a, seating a = Friend.Andrey → neighbors Friend.Andrey seating)
  (h5 : ∀ a, seating a = Profession.Guitarist → ∀ b, seating (b - 1) = Profession.Barista ∨ seating (b + 1) = Profession.Barista → false) :
  (profession Friend.Gennady = Profession.Barista ∧
   profession Friend.Boris = Profession.Architect ∧
   profession Friend.Andrey = Profession.Veterinarian ∧
   profession Friend.Vyacheslav = Profession.Guitarist) :=
begin
  sorry
end

end seating_arrangement_l51_51898


namespace count_3_digit_numbers_divisible_by_13_l51_51088

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51088


namespace dot_product_EC_ED_l51_51771

-- Define the context of the square and the points E, C, D
def midpoint (A B: ℝ × ℝ): ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem dot_product_EC_ED :
  ∀ (A B D C E: ℝ × ℝ),
    ABCD_is_square A B C D →
    side_length (A B C D) = 2 →
    E = midpoint A B →
    vector_dot_product (vector_range E C) (vector_range E D) = 3 :=
by
  sorry

end dot_product_EC_ED_l51_51771


namespace problem_solution_l51_51345

theorem problem_solution (a b : ℝ) (ha : |a| = 5) (hb : b = -3) :
  a + b = 2 ∨ a + b = -8 :=
by sorry

end problem_solution_l51_51345


namespace simplify_and_evaluate_l51_51212

theorem simplify_and_evaluate (x : ℝ) (h : x^2 - 3*x - 2 = 0) :
  (x + 1) * (x - 1) - (x + 3)^2 + 2 * x^2 = -6 := 
by {
  sorry
}

end simplify_and_evaluate_l51_51212


namespace delivery_in_april_l51_51506

theorem delivery_in_april (n_jan n_mar : ℕ) (growth_rate : ℝ) :
  n_jan = 100000 → n_mar = 121000 → (1 + growth_rate) ^ 2 = n_mar / n_jan →
  (n_mar * (1 + growth_rate) = 133100) :=
by
  intros n_jan_eq n_mar_eq growth_eq
  sorry

end delivery_in_april_l51_51506


namespace given_expression_simplifies_to_l51_51393

-- Given conditions: a ≠ ±1, a ≠ 0, b ≠ -1, b ≠ 0
variable (a b : ℝ)
variable (ha1 : a ≠ 1)
variable (ha2 : a ≠ -1)
variable (ha3 : a ≠ 0)
variable (hb1 : b ≠ 0)
variable (hb2 : b ≠ -1)

theorem given_expression_simplifies_to (h1 : a ≠ 1) (h2 : a ≠ -1) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : b ≠ -1) :
    (a * b^(2/3) - b^(2/3) - a + 1) / ((1 - a^(1/3)) * ((a^(1/3) + 1)^2 - a^(1/3)) * (b^(1/3) + 1))
  + (a * b)^(1/3) * (1/a^(1/3) + 1/b^(1/3)) = 1 + a^(1/3) := by
  sorry

end given_expression_simplifies_to_l51_51393


namespace verify_statements_l51_51907

theorem verify_statements (a b : ℝ) :
  ( (ab < 0 ∧ (a < 0 ∧ b > 0 ∨ a > 0 ∧ b < 0)) → (a / b = -1)) ∧
  ( (a + b < 0 ∧ ab > 0) → (|2 * a + 3 * b| = -(2 * a + 3 * b)) ) ∧
  ( (|a - b| + a - b = 0) → (b > a) = False ) ∧
  ( (|a| > |b|) → ((a + b) * (a - b) < 0) = False ) :=
by
  sorry

end verify_statements_l51_51907


namespace negation_of_forall_geq_l51_51819

theorem negation_of_forall_geq {x : ℝ} : ¬ (∀ x : ℝ, x^2 - x ≥ 0) ↔ ∃ x : ℝ, x^2 - x < 0 :=
by
  sorry

end negation_of_forall_geq_l51_51819


namespace exists_real_root_in_interval_l51_51362

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 3

theorem exists_real_root_in_interval (f : ℝ → ℝ)
  (h_mono : ∀ x y, x < y → f x < f y)
  (h1 : f 1 < 0)
  (h2 : f 2 > 0) : 
  ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 := 
sorry

end exists_real_root_in_interval_l51_51362


namespace area_of_octagon_in_square_l51_51440

theorem area_of_octagon_in_square : 
  let A := (0, 0)
  let B := (6, 0)
  let C := (6, 6)
  let D := (0, 6)
  let E := (3, 0)
  let F := (6, 3)
  let G := (3, 6)
  let H := (0, 3)
  ∃ (octagon_area : ℚ),
    octagon_area = 6 :=
by
  sorry

end area_of_octagon_in_square_l51_51440


namespace solution_of_system_of_inequalities_l51_51029

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end solution_of_system_of_inequalities_l51_51029


namespace fixed_point_graph_l51_51708

theorem fixed_point_graph (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) : ∃ x y : ℝ, (x = 2 ∧ y = 2 ∧ y = a^(x-2) + 1) :=
by
  use 2
  use 2
  sorry

end fixed_point_graph_l51_51708


namespace sum_of_angles_l51_51223

theorem sum_of_angles (x y : ℝ) (n : ℕ) :
  n = 16 →
  (∃ k l : ℕ, k = 3 ∧ l = 5 ∧ 
  x = (k * (360 / n)) / 2 ∧ y = (l * (360 / n)) / 2) →
  x + y = 90 :=
by
  intros
  sorry

end sum_of_angles_l51_51223


namespace total_students_in_college_l51_51396

theorem total_students_in_college (B G : ℕ) (h_ratio: 8 * G = 5 * B) (h_girls: G = 175) :
  B + G = 455 := 
  sorry

end total_students_in_college_l51_51396


namespace final_professions_correct_l51_51893

-- Define the people and their positions
inductive Person
| Andrey | Boris | Vyacheslav | Gennady
deriving DecidableEq, Inhabited

inductive Profession
| Architect | Barista | Veterinarian | Guitarist
deriving DecidableEq, Inhabited

open Person Profession

-- Given positions in seats
def seatingArrangement : Person → Nat := 
λ p, 
  match p with
  | Andrey    => 3
  | Boris     => 2
  | Vyacheslav => 4
  | Gennady   => 1

-- Given professions
def profession : Person → Profession := 
λ p, 
  match p with
  | Andrey    => Veterinarian
  | Boris     => Architect
  | Vyacheslav => Guitarist
  | Gennady   => Barista

-- Define the conditions
def condition1 : Prop :=
  (seatingArrangement Andrey = 3) ∧ 
  (seatingArrangement Boris = 2) ∧
  (seatingArrangement Vyacheslav = 4) ∧ 
  (seatingArrangement Gennady = 1)

def condition2 : Prop := 
  seatingArrangement (Person.Barista) + 1 = seatingArrangement Boris

def condition3 : Prop := 
  seatingArrangement Vyacheslav > seatingArrangement Andrey ∧ 
  seatingArrangement Vyacheslav > seatingArrangement Boris

def condition4 : Prop := 
  seatingArrangement (Person.Veterinarian) ≠ 1 ∧ seatingArrangement (Person.Veterinarian) ≠ 4

def condition5 : Prop := 
  (seatingArrangement (Person.Barista) ≠ seatingArrangement (Person.Guitarist) + 1 ∧
  seatingArrangement (Person.Guitarist) ≠ seatingArrangement (Person.Barista) + 1)

-- Prove the final professions
theorem final_professions_correct :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 →
  profession Gennady = Barista ∧
  profession Boris = Architect ∧
  profession Andrey = Veterinarian ∧
  profession Vyacheslav = Guitarist :=
by sorry

end final_professions_correct_l51_51893


namespace range_f_sum_l51_51432

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x ^ 2)

theorem range_f_sum {a b : ℝ} (h₁ : Set.Ioo a b = (Set.Ioo (0 : ℝ) (3 : ℝ))) :
  a + b = 3 :=
sorry

end range_f_sum_l51_51432


namespace count_100_digit_even_numbers_l51_51917

theorem count_100_digit_even_numbers : 
  let digits := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let count_valid_numbers : ℕ := 
    let last_digit := 0
    let first_digit_choices := 2
    let middle_digits_choices := 3 ^ 98
    first_digit_choices * middle_digits_choices * 1
  in
  count_valid_numbers = 2 * 3 ^ 98 :=
by 
  let digits := {0, 1, 3}
  let is_even (n : ℕ) := n % 2 = 0
  let count_valid_numbers : ℕ := 
    let last_digit := 0
    let first_digit_choices := 2
    let middle_digits_choices := 3 ^ 98
    first_digit_choices * middle_digits_choices * 1
  have : count_valid_numbers = 2 * 3 ^ 98 := by sorry
  exact this

end count_100_digit_even_numbers_l51_51917


namespace digit_15_of_sum_reciprocals_l51_51513

/-- 
What is the 15th digit after the decimal point of the sum of the decimal equivalents 
for the fractions 1/9 and 1/11?
-/
theorem digit_15_of_sum_reciprocals :
  let r := (1/9 + 1/11) in
  let d15 := Real.frac (10^(15:ℕ) * r) in
  Int.floor (10 * d15) = 1 :=
by
  let r := (1/9 + 1/11)
  let d15 := Real.frac (10^(15:ℕ) * r)
  have h : Real.toRat d15 = 1 / 10 + d15 - Int.floor d15
  have : Real.frac (10^(15:ℕ) * r) = r
  sorry

end digit_15_of_sum_reciprocals_l51_51513


namespace sum_of_squares_l51_51648

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 23) (h2 : a * b + b * c + a * c = 131) :
  a^2 + b^2 + c^2 = 267 :=
by
  sorry

end sum_of_squares_l51_51648


namespace sin_product_identity_l51_51888

noncomputable def sin_15_deg := Real.sin (15 * Real.pi / 180)
noncomputable def sin_30_deg := Real.sin (30 * Real.pi / 180)
noncomputable def sin_75_deg := Real.sin (75 * Real.pi / 180)

theorem sin_product_identity :
  sin_15_deg * sin_30_deg * sin_75_deg = 1 / 8 :=
by
  sorry

end sin_product_identity_l51_51888


namespace evaluate_product_eq_l51_51434

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product_eq : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 885735 := 
sorry

end evaluate_product_eq_l51_51434


namespace function_range_ge_4_l51_51238

variable {x : ℝ}

theorem function_range_ge_4 (h : x > 0) : 2 * x + 2 * x⁻¹ ≥ 4 :=
sorry

end function_range_ge_4_l51_51238


namespace initial_plants_count_l51_51369

theorem initial_plants_count (p : ℕ) 
    (h1 : p - 20 > 0)
    (h2 : (p - 20) / 2 > 0)
    (h3 : ((p - 20) / 2) - 1 > 0)
    (h4 : ((p - 20) / 2) - 1 = 4) : 
    p = 30 :=
by
  sorry

end initial_plants_count_l51_51369


namespace smallest_x_y_sum_l51_51734

theorem smallest_x_y_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hne : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 24) :
  x + y = 100 :=
sorry

end smallest_x_y_sum_l51_51734


namespace price_per_pound_salt_is_50_l51_51259

-- Given conditions
def totalWeight : ℕ := 60
def weightSalt1 : ℕ := 20
def priceSalt2 : ℕ := 35
def weightSalt2 : ℕ := 40
def sellingPricePerPound : ℕ := 48
def desiredProfitRate : ℚ := 0.20

-- Mathematical definitions derived from conditions
def costSalt1 (priceSalt1 : ℕ) : ℕ := weightSalt1 * priceSalt1
def costSalt2 : ℕ := weightSalt2 * priceSalt2
def totalCost (priceSalt1 : ℕ) : ℕ := costSalt1 priceSalt1 + costSalt2
def totalRevenue : ℕ := totalWeight * sellingPricePerPound
def profit (priceSalt1 : ℕ) : ℚ := desiredProfitRate * totalCost priceSalt1
def totalProfit (priceSalt1 : ℕ) : ℚ := totalCost priceSalt1 + profit priceSalt1

-- Proof statement
theorem price_per_pound_salt_is_50 : ∃ (priceSalt1 : ℕ), totalRevenue = totalProfit priceSalt1 ∧ priceSalt1 = 50 := by
  -- We provide the prove structure, exact proof steps are skipped with sorry
  sorry

end price_per_pound_salt_is_50_l51_51259


namespace train_crosses_post_in_approximately_18_seconds_l51_51677

noncomputable def train_length : ℕ := 300
noncomputable def platform_length : ℕ := 350
noncomputable def crossing_time_platform : ℕ := 39

noncomputable def combined_length : ℕ := train_length + platform_length
noncomputable def speed_train : ℝ := combined_length / crossing_time_platform

noncomputable def crossing_time_post : ℝ := train_length / speed_train

theorem train_crosses_post_in_approximately_18_seconds :
  abs (crossing_time_post - 18) < 1 :=
by
  admit

end train_crosses_post_in_approximately_18_seconds_l51_51677


namespace tank_capacity_l51_51685

theorem tank_capacity (x : ℝ) (h₁ : (3/4) * x = (1/3) * x + 18) : x = 43.2 := sorry

end tank_capacity_l51_51685


namespace find_px_value_l51_51481

noncomputable def p (a b c x : ℤ) := a * x^2 + b * x + c

theorem find_px_value {a b c : ℤ} 
  (h1 : p a b c 2 = 2) 
  (h2 : p a b c (-2) = -2) 
  (h3 : p a b c 9 = 3) 
  (h : a = -2 / 11) 
  (h4 : b = 1)
  (h5 : c = 8 / 11) :
  p a b c 14 = -230 / 11 :=
by
  sorry

end find_px_value_l51_51481


namespace speedster_convertibles_count_l51_51402

theorem speedster_convertibles_count (T : ℕ)
  (h1 : 3 * T / 4 isNat)
  (h2 : (T / 4 : ℕ) = 30)
  (h3 : ∃ (convertibles : ℕ), convertibles = 3 * (3 * T / 4) / 5) :
  ∃ (convertibles : ℕ), convertibles = 54 :=
by
  cases h3 with convertibles h3
  use convertibles
  rw [Nat.mul_div_cancel_left, Nat.mul_div_cancel_left] at h3
  sorry

end speedster_convertibles_count_l51_51402


namespace roots_of_f_l51_51484

noncomputable def f (a x : ℝ) : ℝ := x - Real.log (a * x)

theorem roots_of_f (a : ℝ) :
  (a < 0 → ¬∃ x : ℝ, f a x = 0) ∧
  (0 < a ∧ a < Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a = Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a > Real.exp 1 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :=
sorry

end roots_of_f_l51_51484


namespace hyperbola_asymptote_slope_l51_51227

theorem hyperbola_asymptote_slope :
  (∀ x y : ℝ, (x^2 / 100 - y^2 / 64 = 1) → y = (4/5) * x ∨ y = -(4/5) * x) :=
by
  sorry

end hyperbola_asymptote_slope_l51_51227


namespace count_3_digit_multiples_of_13_l51_51099

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l51_51099


namespace trajectory_of_M_l51_51441

theorem trajectory_of_M (x y t : ℝ) (M P F : ℝ × ℝ)
    (hF : F = (1, 0))
    (hP : P = (1/4 * t^2, t))
    (hFP : (P.1 - F.1, P.2 - F.2) = (1/4 * t^2 - 1, t))
    (hFM : (M.1 - F.1, M.2 - F.2) = (x - 1, y))
    (hFP_FM : (P.1 - F.1, P.2 - F.2) = (2 * (M.1 - F.1), 2 * (M.2 - F.2))) :
  y^2 = 2 * x - 1 :=
by
  sorry

end trajectory_of_M_l51_51441


namespace additional_tickets_won_l51_51419

-- Definitions from the problem
def initial_tickets : ℕ := 49
def spent_tickets : ℕ := 25
def final_tickets : ℕ := 30

-- The main statement we need to prove
theorem additional_tickets_won (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : 
  final_tickets - (initial_tickets - spent_tickets) = 6 :=
by
  sorry

end additional_tickets_won_l51_51419


namespace malcolm_initial_white_lights_l51_51229

theorem malcolm_initial_white_lights :
  ∀ (red blue green remaining total_initial : ℕ),
    red = 12 →
    blue = 3 * red →
    green = 6 →
    remaining = 5 →
    total_initial = red + blue + green + remaining →
    total_initial = 59 :=
by
  intros red blue green remaining total_initial h1 h2 h3 h4 h5
  -- Add details if necessary for illustration
  -- sorry typically as per instructions
  sorry

end malcolm_initial_white_lights_l51_51229


namespace find_k_in_geometric_sequence_l51_51473

theorem find_k_in_geometric_sequence (c k : ℝ) (h1_nonzero : c ≠ 0)
  (S : ℕ → ℝ) (a : ℕ → ℝ) (h2 : ∀ n, a (n + 1) = c * a n)
  (h3 : ∀ n, S n = 3^n + k)
  (h4 : a 1 = 3 + k)
  (h5 : a 2 = S 2 - S 1)
  (h6 : a 3 = S 3 - S 2) : k = -1 := by
  sorry

end find_k_in_geometric_sequence_l51_51473


namespace find_number_l51_51946

theorem find_number (x : ℝ) (h : (5 / 6) * x = (5 / 16) * x + 300) : x = 576 :=
sorry

end find_number_l51_51946


namespace large_beds_l51_51914

theorem large_beds {L : ℕ} {M : ℕ} 
    (h1 : M = 2) 
    (h2 : ∀ (x : ℕ), 100 <= x → L = (320 - 60 * M) / 100) : 
  L = 2 :=
by
  sorry

end large_beds_l51_51914


namespace problem_statement_l51_51621

theorem problem_statement (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := 
sorry

end problem_statement_l51_51621


namespace three_digit_numbers_div_by_13_l51_51106

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l51_51106


namespace simplify_expr_l51_51631

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) : x ^ (-2) + 2 * x ^ (-1) - 3 = (1 + 2 * x - 3 * x ^ 2) / x ^ 2 :=
by sorry

end simplify_expr_l51_51631


namespace specified_time_eq_l51_51410

noncomputable def slow_horse_days (x : ℝ) := x + 1
noncomputable def fast_horse_days (x : ℝ) := x - 3

theorem specified_time_eq (x : ℝ) (h1 : slow_horse_days x > 0) (h2 : fast_horse_days x > 0) :
  (900 / slow_horse_days x) * 2 = 900 / fast_horse_days x :=
by
  sorry

end specified_time_eq_l51_51410


namespace number_of_days_A_left_l51_51522

noncomputable def work_problem (W : ℝ) : Prop :=
  let A_rate := W / 45
  let B_rate := W / 40
  let days_B_alone := 23
  ∃ x : ℝ, x * (A_rate + B_rate) + days_B_alone * B_rate = W ∧ x = 9

theorem number_of_days_A_left (W : ℝ) : work_problem W :=
  sorry

end number_of_days_A_left_l51_51522


namespace upgraded_video_card_multiple_l51_51191

noncomputable def multiple_of_video_card_cost (computer_cost monitor_cost_peripheral_cost base_video_card_cost total_spent upgraded_video_card_cost : ℝ) : ℝ :=
  upgraded_video_card_cost / base_video_card_cost

theorem upgraded_video_card_multiple
  (computer_cost : ℝ)
  (monitor_cost_ratio : ℝ)
  (base_video_card_cost : ℝ)
  (total_spent : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : monitor_cost_ratio = 1/5)
  (h3 : base_video_card_cost = 300)
  (h4 : total_spent = 2100) :
  multiple_of_video_card_cost computer_cost (computer_cost * monitor_cost_ratio) base_video_card_cost total_spent (total_spent - (computer_cost + computer_cost * monitor_cost_ratio)) = 1 :=
by
  sorry

end upgraded_video_card_multiple_l51_51191


namespace simplify_expression_l51_51214

theorem simplify_expression (x : ℝ) : 2 * (x - 3) - (-x + 4) = 3 * x - 10 :=
by
  -- The proof is omitted, so use sorry to skip it
  sorry

end simplify_expression_l51_51214


namespace frank_hawaiian_slices_l51_51706

theorem frank_hawaiian_slices:
  ∀ (total_slices dean_slices sammy_slices leftover_slices frank_slices : ℕ),
  total_slices = 24 →
  dean_slices = 6 →
  sammy_slices = 4 →
  leftover_slices = 11 →
  (total_slices - leftover_slices) = (dean_slices + sammy_slices + frank_slices) →
  frank_slices = 3 :=
by
  intros total_slices dean_slices sammy_slices leftover_slices frank_slices
  intros h_total h_dean h_sammy h_leftovers h_total_eaten
  sorry

end frank_hawaiian_slices_l51_51706


namespace meals_given_away_l51_51545

def initial_meals_colt_and_curt : ℕ := 113
def additional_meals_sole_mart : ℕ := 50
def remaining_meals : ℕ := 78
def total_initial_meals : ℕ := initial_meals_colt_and_curt + additional_meals_sole_mart
def given_away_meals (total : ℕ) (remaining : ℕ) : ℕ := total - remaining

theorem meals_given_away : given_away_meals total_initial_meals remaining_meals = 85 :=
by
  sorry

end meals_given_away_l51_51545


namespace number_of_three_digit_numbers_divisible_by_13_l51_51122

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l51_51122


namespace triangle_area_ab_l51_51308

theorem triangle_area_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0.5 * (12 / a) * (12 / b) = 12) : a * b = 6 :=
by
  sorry

end triangle_area_ab_l51_51308


namespace hard_candy_food_coloring_l51_51848

theorem hard_candy_food_coloring
  (lollipop_coloring : ℕ) (hard_candy_coloring : ℕ)
  (num_lollipops : ℕ) (num_hardcandies : ℕ)
  (total_coloring : ℕ)
  (H1 : lollipop_coloring = 8)
  (H2 : num_lollipops = 150)
  (H3 : num_hardcandies = 20)
  (H4 : total_coloring = 1800) :
  (20 * hard_candy_coloring + 150 * lollipop_coloring = total_coloring) → 
  hard_candy_coloring = 30 :=
by
  sorry

end hard_candy_food_coloring_l51_51848


namespace ratio_of_p_to_q_l51_51384

theorem ratio_of_p_to_q (p q : ℝ) (h₁ : (p + q) / (p - q) = 4 / 3) (h₂ : p / q = r) : r = 7 :=
sorry

end ratio_of_p_to_q_l51_51384


namespace parabola_directrix_eq_l51_51713

theorem parabola_directrix_eq (x : ℝ) : 
  (∀ y : ℝ, y = 3 * x^2 - 6 * x + 2 → True) →
  y = -13/12 := 
  sorry

end parabola_directrix_eq_l51_51713


namespace three_digit_numbers_divisible_by_13_l51_51142

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l51_51142


namespace pow_99_square_pow_neg8_mult_l51_51421

theorem pow_99_square :
  99^2 = 9801 := 
by
  -- Proof omitted
  sorry

theorem pow_neg8_mult :
  (-8) ^ 2009 * (-1/8) ^ 2008 = -8 :=
by
  -- Proof omitted
  sorry

end pow_99_square_pow_neg8_mult_l51_51421


namespace polynomial_sum_l51_51616

def f (x : ℝ) : ℝ := -4 * x^3 - 3 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 + x - 7
def h (x : ℝ) : ℝ := 3 * x^3 + 6 * x^2 + 3 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = x^3 - 2 * x^2 + 6 * x - 10 := by
  sorry

end polynomial_sum_l51_51616


namespace system_of_equations_solution_l51_51477

theorem system_of_equations_solution :
  ∀ (a b : ℝ),
  (-2 * a + b^2 = Real.cos (π * a + b^2) - 1 ∧ b^2 = Real.cos (2 * π * a + b^2) - 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 0) :=
by
  intro a b
  sorry

end system_of_equations_solution_l51_51477


namespace prime_list_count_l51_51880

theorem prime_list_count {L : ℕ → ℕ} 
  (hL₀ : L 0 = 29)
  (hL : ∀ (n : ℕ), L (n + 1) = L n * 101 + L 0) :
  (∃! n, n = 0 ∧ Prime (L n)) ∧ ∀ m > 0, ¬ Prime (L m) := 
by
  sorry

end prime_list_count_l51_51880


namespace fewest_tiles_needed_to_cover_rectangle_l51_51864

noncomputable def height_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (Real.sqrt 3 / 2) * side_length

noncomputable def area_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (1 / 2) * side_length * height_of_equilateral_triangle side_length

noncomputable def area_of_floor_in_square_inches (length_in_feet : ℝ) (width_in_feet : ℝ) : ℝ :=
  length_in_feet * width_in_feet * (12 * 12)

noncomputable def number_of_tiles_required (floor_area : ℝ) (tile_area : ℝ) : ℝ :=
  floor_area / tile_area

theorem fewest_tiles_needed_to_cover_rectangle :
  number_of_tiles_required (area_of_floor_in_square_inches 3 4) (area_of_equilateral_triangle 2) = 997 := 
by
  sorry

end fewest_tiles_needed_to_cover_rectangle_l51_51864


namespace number_of_workers_l51_51012

theorem number_of_workers (N C : ℕ) 
  (h1 : N * C = 300000) 
  (h2 : N * (C + 50) = 325000) : 
  N = 500 :=
sorry

end number_of_workers_l51_51012


namespace dot_product_EC_ED_l51_51773

-- Define the context of the square and the points E, C, D
def midpoint (A B: ℝ × ℝ): ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem dot_product_EC_ED :
  ∀ (A B D C E: ℝ × ℝ),
    ABCD_is_square A B C D →
    side_length (A B C D) = 2 →
    E = midpoint A B →
    vector_dot_product (vector_range E C) (vector_range E D) = 3 :=
by
  sorry

end dot_product_EC_ED_l51_51773


namespace count_three_digit_numbers_divisible_by_13_l51_51085

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l51_51085


namespace sin_minus_cos_eq_l51_51906

variable {α : ℝ} (h₁ : 0 < α ∧ α < π) (h₂ : Real.sin α + Real.cos α = 1/3)

theorem sin_minus_cos_eq : Real.sin α - Real.cos α = Real.sqrt 17 / 3 :=
by 
  -- Proof goes here
  sorry

end sin_minus_cos_eq_l51_51906


namespace polynomial_has_no_real_roots_l51_51429

theorem polynomial_has_no_real_roots :
  ∀ x : ℝ, x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + 5/2 ≠ 0 :=
by
  sorry

end polynomial_has_no_real_roots_l51_51429


namespace dot_product_square_ABCD_l51_51764

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l51_51764


namespace Parabola_vertex_form_l51_51181

theorem Parabola_vertex_form (x : ℝ) (y : ℝ) : 
  (∃ h k : ℝ, (h = -2) ∧ (k = 1) ∧ (y = (x + h)^2 + k) ) ↔ (y = (x + 2)^2 + 1) :=
by
  sorry

end Parabola_vertex_form_l51_51181


namespace ab_range_l51_51296

variable (a b : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b)
variable (h_eq : a * b = a + b + 8)

theorem ab_range (h : a * b = a + b + 8) : 16 ≤ a * b :=
by sorry

end ab_range_l51_51296


namespace common_ratio_half_l51_51450

-- Definitions based on conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n+1) = a n * q
def arith_seq (x y z : ℝ) := x + z = 2 * y

-- Theorem statement
theorem common_ratio_half (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q)
  (h_arith : arith_seq (a 5) (a 6 + a 8) (a 7)) : q = 1 / 2 := 
sorry

end common_ratio_half_l51_51450


namespace number_of_paths_to_spell_BINGO_l51_51467

theorem number_of_paths_to_spell_BINGO : 
  ∃ (paths : ℕ), paths = 36 :=
by
  sorry

end number_of_paths_to_spell_BINGO_l51_51467


namespace find_k_for_parallel_lines_l51_51449

theorem find_k_for_parallel_lines (k : ℝ) :
  (∀ x y : ℝ, (k - 2) * x + (4 - k) * y + 1 = 0) →
  (∀ x y : ℝ, 2 * (k - 2) * x - 2 * y + 3 = 0) →
  (k = 2 ∨ k = 5) :=
sorry

end find_k_for_parallel_lines_l51_51449


namespace number_of_three_digit_numbers_divisible_by_13_l51_51118

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l51_51118


namespace cost_per_gallon_l51_51974

theorem cost_per_gallon (weekly_spend : ℝ) (two_week_usage : ℝ) (weekly_spend_eq : weekly_spend = 36) (two_week_usage_eq : two_week_usage = 24) : 
  (2 * weekly_spend / two_week_usage) = 3 :=
by sorry

end cost_per_gallon_l51_51974


namespace system_of_inequalities_solution_l51_51034

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end system_of_inequalities_solution_l51_51034


namespace count_3_digit_numbers_divisible_by_13_l51_51090

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51090


namespace range_of_a_l51_51335

def A := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2*(a+1)*x + (a^2 -1) = 0}

theorem range_of_a (a : ℝ) :
  (A ∩ B a = B a) → (a = 1 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l51_51335


namespace local_value_of_4_in_564823_l51_51607

def face_value (d : ℕ) : ℕ := d
def place_value_of_thousands : ℕ := 1000
def local_value (d : ℕ) (p : ℕ) : ℕ := d * p

theorem local_value_of_4_in_564823 :
  local_value (face_value 4) place_value_of_thousands = 4000 :=
by 
  sorry

end local_value_of_4_in_564823_l51_51607


namespace ratio_proof_l51_51702

noncomputable def total_capacity : ℝ := 10 -- million gallons
noncomputable def amount_end_month : ℝ := 6 -- million gallons
noncomputable def normal_level : ℝ := total_capacity - 5 -- million gallons

theorem ratio_proof (h1 : amount_end_month = 0.6 * total_capacity)
                    (h2 : normal_level = total_capacity - 5) :
  (amount_end_month / normal_level) = 1.2 :=
by sorry

end ratio_proof_l51_51702


namespace final_result_is_110_l51_51863

theorem final_result_is_110 (x : ℕ) (h1 : x = 155) : (x * 2 - 200) = 110 :=
by
  -- placeholder for the solution proof
  sorry

end final_result_is_110_l51_51863


namespace algebra_expression_correct_l51_51911

theorem algebra_expression_correct {x y : ℤ} (h : x + 2 * y + 1 = 3) : 2 * x + 4 * y + 1 = 5 :=
  sorry

end algebra_expression_correct_l51_51911


namespace negation_proof_l51_51502

theorem negation_proof :
  ¬ (∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 :=
by
  -- proof goes here
  sorry

end negation_proof_l51_51502


namespace smallest_positive_period_pi_increasing_intervals_in_0_to_pi_range_of_m_l51_51197

noncomputable def f (x m : ℝ) := 2 * (Real.cos x) ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + m

theorem smallest_positive_period_pi (m : ℝ) :
  ∀ x : ℝ, f (x + π) m = f x m := sorry

theorem increasing_intervals_in_0_to_pi (m : ℝ) :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 6) ∨ (2 * π / 3 ≤ x ∧ x ≤ π) →
  ∀ y : ℝ, ((0 ≤ y ∧ y ≤ π / 6 ∨ (2 * π / 3 ≤ y ∧ y ≤ π)) ∧ x < y) → f x m < f y m := sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 6) → -4 < f x m ∧ f x m < 4) ↔ (-6 < m ∧ m < 1) := sorry

end smallest_positive_period_pi_increasing_intervals_in_0_to_pi_range_of_m_l51_51197


namespace koschei_coin_count_l51_51333

theorem koschei_coin_count (a : ℕ) :
  (a % 10 = 7) ∧
  (a % 12 = 9) ∧
  (300 ≤ a ∧ a ≤ 400) →
  a = 357 :=
sorry

end koschei_coin_count_l51_51333


namespace inequality_least_one_l51_51784

theorem inequality_least_one {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4) :=
by
  sorry

end inequality_least_one_l51_51784


namespace mixed_number_arithmetic_l51_51269

theorem mixed_number_arithmetic :
  26 * (2 + 4 / 7 - (3 + 1 / 3)) + (3 + 1 / 5 + (2 + 3 / 7)) = -14 - 223 / 735 :=
by
  sorry

end mixed_number_arithmetic_l51_51269


namespace monotonic_decreasing_interval_l51_51644

open Real

noncomputable def decreasing_interval (k: ℤ): Set ℝ :=
  {x | k * π - π / 3 < x ∧ x < k * π + π / 6 }

theorem monotonic_decreasing_interval (k : ℤ) :
  ∀ x, x ∈ decreasing_interval k ↔ (k * π - π / 3 < x ∧ x < k * π + π / 6) :=
by 
  intros x
  sorry

end monotonic_decreasing_interval_l51_51644


namespace inverse_function_of_1_div_2_pow_eq_log_base_1_div_2_l51_51943

noncomputable def inverse_of_half_pow (x : ℝ) : ℝ := Real.log x / Real.log (1 / 2)

theorem inverse_function_of_1_div_2_pow_eq_log_base_1_div_2 (x : ℝ) (hx : 0 < x) :
  inverse_of_half_pow x = Real.log x / Real.log (1 / 2) :=
by
  sorry

end inverse_function_of_1_div_2_pow_eq_log_base_1_div_2_l51_51943


namespace inequality_holds_for_interval_l51_51642

theorem inequality_holds_for_interval (a : ℝ) : 
  (∀ x, 1 < x ∧ x < 5 → x^2 - 2 * (a - 2) * x + a < 0) → a ≥ 5 :=
by
  intros h
  sorry

end inequality_holds_for_interval_l51_51642


namespace solution_of_system_of_inequalities_l51_51031

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end solution_of_system_of_inequalities_l51_51031


namespace boxes_containing_neither_l51_51694

theorem boxes_containing_neither (total_boxes markers erasers both : ℕ) 
  (h_total : total_boxes = 15) (h_markers : markers = 8) (h_erasers : erasers = 5) (h_both : both = 4) :
  total_boxes - (markers + erasers - both) = 6 :=
by
  sorry

end boxes_containing_neither_l51_51694


namespace water_consumption_correct_l51_51023

theorem water_consumption_correct (w n r : ℝ) 
  (hw : w = 21428) 
  (hn : n = 26848.55) 
  (hr : r = 302790.13) :
  w = 21428 ∧ n = 26848.55 ∧ r = 302790.13 :=
by 
  sorry

end water_consumption_correct_l51_51023


namespace problem_relationship_l51_51437

theorem problem_relationship (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a :=
by {
  sorry
}

end problem_relationship_l51_51437


namespace Adam_bought_9_cat_food_packages_l51_51002

def num_cat_food_packages (c : ℕ) : Prop :=
  let cat_cans := 10 * c
  let dog_cans := 7 * 5
  cat_cans = dog_cans + 55

theorem Adam_bought_9_cat_food_packages : num_cat_food_packages 9 :=
by
  unfold num_cat_food_packages
  sorry

end Adam_bought_9_cat_food_packages_l51_51002


namespace sue_cost_l51_51355

def cost_of_car : ℝ := 2100
def total_days_in_week : ℝ := 7
def sue_days : ℝ := 3

theorem sue_cost : (cost_of_car * (sue_days / total_days_in_week)) = 899.99 :=
by
  sorry

end sue_cost_l51_51355


namespace largest_prime_factor_1729_l51_51994

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l51_51994


namespace percentage_of_ll_watchers_l51_51204

theorem percentage_of_ll_watchers 
  (T : ℕ) 
  (IS : ℕ) 
  (ME : ℕ) 
  (E2 : ℕ) 
  (A3 : ℕ) 
  (total_residents : T = 600)
  (is_watchers : IS = 210)
  (me_watchers : ME = 300)
  (e2_watchers : E2 = 108)
  (a3_watchers : A3 = 21)
  (at_least_one_show : IS + (by sorry) + ME - E2 + A3 = T) :
  ∃ x : ℕ, (x * 100 / T) = 115 :=
by sorry

end percentage_of_ll_watchers_l51_51204


namespace speed_of_current_eq_l51_51688

theorem speed_of_current_eq :
  ∃ (m c : ℝ), (m + c = 15) ∧ (m - c = 8.6) ∧ (c = 3.2) :=
by
  sorry

end speed_of_current_eq_l51_51688


namespace rugby_tournament_n_count_l51_51264

noncomputable def valid_n_count : ℕ :=
  (10 to 2017).count (λ n,
    (n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)) % (2^5 * 3^2 * 5 * 7) = 0
  )

theorem rugby_tournament_n_count : valid_n_count = 562 :=
by {
  sorry
}

end rugby_tournament_n_count_l51_51264


namespace simplify_fraction_l51_51519

theorem simplify_fraction (a b : ℕ) (h : a = 2020) (h2 : b = 2018) :
  (2 ^ a - 2 ^ b) / (2 ^ a + 2 ^ b) = 3 / 5 := by
  sorry

end simplify_fraction_l51_51519


namespace three_digit_numbers_div_by_13_l51_51107

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l51_51107


namespace count_three_digit_numbers_divisible_by_13_l51_51083

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l51_51083


namespace alpha_beta_sum_l51_51501

theorem alpha_beta_sum (α β : ℝ) (h1 : α^3 - 3 * α^2 + 5 * α - 17 = 0) (h2 : β^3 - 3 * β^2 + 5 * β + 11 = 0) : α + β = 2 := 
by
  sorry

end alpha_beta_sum_l51_51501


namespace min_questions_to_find_phone_number_min_questions_to_find_phone_number_is_17_l51_51392

theorem min_questions_to_find_phone_number : 
  ∃ n : ℕ, ∀ (N : ℕ), (N = 100000 → 2 ^ n ≥ N) ∧ (2 ^ (n - 1) < N) := sorry

-- In simpler form, since log_2(100000) ≈ 16.60965, we have:
theorem min_questions_to_find_phone_number_is_17 : 
  ∀ (N : ℕ), (N = 100000 → 17 = Nat.ceil (Real.logb 2 100000)) := sorry

end min_questions_to_find_phone_number_min_questions_to_find_phone_number_is_17_l51_51392


namespace count_three_digit_div_by_13_l51_51102

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l51_51102


namespace find_max_m_l51_51194

-- We define real numbers a, b, c that satisfy the given conditions
variable (a b c m : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum : a + b + c = 12)
variable (h_prod_sum : a * b + b * c + c * a = 30)
variable (m_def : m = min (a * b) (min (b * c) (c * a)))

-- We state the main theorem to be proved
theorem find_max_m : m ≤ 2 :=
by
  sorry

end find_max_m_l51_51194


namespace range_of_a_sq_l51_51891

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ m n : ℕ, a (m + n) = a m + a n

theorem range_of_a_sq {n : ℕ}
  (h_arith : arithmetic_sequence a)
  (h_cond : a 1 ^ 2 + a (2 * n + 1) ^ 2 = 1) :
  ∃ (L R : ℝ), (L = 2) ∧ (∀ k : ℕ, a (n+1) ^ 2 + a (3*n+1) ^ 2 ≥ L) := sorry

end range_of_a_sq_l51_51891


namespace total_participants_l51_51207

theorem total_participants (Petya Vasya total : ℕ) 
  (h1 : Petya = Vasya + 1) 
  (h2 : Petya = 10)
  (h3 : Vasya + 15 = total + 1) : 
  total = 23 :=
by
  sorry

end total_participants_l51_51207


namespace proof_problem_theorem_l51_51756

noncomputable def proof_problem : Prop :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (2, 2)
  let D : ℝ × ℝ := (0, 2)
  let E : ℝ × ℝ := (1, 0)
  let vector := (p1 p2 : ℝ × ℝ) → (p2.1 - p1.1, p2.2 - p1.2)
  let dot_product := (u v : ℝ × ℝ) → u.1 * v.1 + u.2 * v.2
  let EC := vector E C
  let ED := vector E D
  EC ∘ ED = 3

theorem proof_problem_theorem : proof_problem := 
by 
  sorry

end proof_problem_theorem_l51_51756


namespace solve_system_of_inequalities_l51_51025

theorem solve_system_of_inequalities (p : ℚ) : (19 * p < 10) ∧ (1 / 2 < p) → (1 / 2 < p ∧ p < 10 / 19) :=
by
  sorry

end solve_system_of_inequalities_l51_51025


namespace minimum_benches_for_equal_occupancy_l51_51498

theorem minimum_benches_for_equal_occupancy (M : ℕ) :
  (∃ x y, x = y ∧ 8 * M = x ∧ 12 * M = y) ↔ M = 3 := by
  sorry

end minimum_benches_for_equal_occupancy_l51_51498


namespace largest_prime_factor_of_1729_l51_51997

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l51_51997


namespace arithmetic_seq_a7_geometric_seq_b6_geometric_common_ratio_l51_51445

noncomputable def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def geometric_seq (b₁ q : ℕ) (n : ℕ) : ℕ :=
  b₁ * q^(n - 1)

theorem arithmetic_seq_a7 
  (a₁ d : ℕ)
  (h : 2 * arithmetic_seq a₁ d 5 - arithmetic_seq a₁ d 3 = 3) :
  arithmetic_seq a₁ d 7 = 3 :=
sorry

theorem geometric_seq_b6 
  (b₁ q : ℕ)
  (h1 : geometric_seq b₁ q 2 = 1)
  (h2 : geometric_seq b₁ q 4 = 4) :
  geometric_seq b₁ q 6 = 16 :=
sorry

theorem geometric_common_ratio 
  (b₁ q : ℕ)
  (h1 : geometric_seq b₁ q 2 = 1)
  (h2 : geometric_seq b₁ q 4 = 4) :
  q = 2 ∨ q = -2 :=
sorry

end arithmetic_seq_a7_geometric_seq_b6_geometric_common_ratio_l51_51445


namespace count_3digit_numbers_div_by_13_l51_51113

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l51_51113


namespace sum_of_sequences_l51_51425

-- Define the sequences and their type
def seq1 : List ℕ := [2, 12, 22, 32, 42]
def seq2 : List ℕ := [10, 20, 30, 40, 50]

-- The property we wish to prove
theorem sum_of_sequences : seq1.sum + seq2.sum = 260 :=
by
  sorry

end sum_of_sequences_l51_51425


namespace sequence_periodic_l51_51042

def last_digit (n : ℕ) : ℕ := n % 10

noncomputable def a_n (n : ℕ) : ℕ := last_digit (n^(n^n))

theorem sequence_periodic :
  ∃ period : ℕ, period = 20 ∧ ∀ n m : ℕ, n ≡ m [MOD period] → a_n n = a_n m :=
sorry

end sequence_periodic_l51_51042


namespace positive_3_digit_numbers_divisible_by_13_count_l51_51077

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l51_51077


namespace infinite_series_eq_15_l51_51291

theorem infinite_series_eq_15 (x : ℝ) :
  (∑' (n : ℕ), (5 + n * x) / 3^n) = 15 ↔ x = 10 :=
by
  sorry

end infinite_series_eq_15_l51_51291


namespace melinda_doughnuts_picked_l51_51975

theorem melinda_doughnuts_picked :
  (∀ d h_coffee m_coffee : ℕ, d = 3 → h_coffee = 4 → m_coffee = 6 →
    ∀ cost_d cost_h cost_m : ℝ, cost_d = 0.45 → 
    cost_h = 4.91 → cost_m = 7.59 → 
    ∃ m_doughnuts : ℕ, cost_m - m_coffee * ((cost_h - d * cost_d) / h_coffee) = m_doughnuts * cost_d) → 
  ∃ n : ℕ, n = 5 := 
by sorry

end melinda_doughnuts_picked_l51_51975


namespace current_prices_l51_51678

theorem current_prices (initial_ram_price initial_ssd_price : ℝ) 
  (ram_increase_1 ram_decrease_1 ram_decrease_2 : ℝ) 
  (ssd_increase_1 ssd_decrease_1 ssd_decrease_2 : ℝ) 
  (initial_ram : initial_ram_price = 50) 
  (initial_ssd : initial_ssd_price = 100) 
  (ram_increase_factor : ram_increase_1 = 0.30 * initial_ram_price) 
  (ram_decrease_factor_1 : ram_decrease_1 = 0.15 * (initial_ram_price + ram_increase_1)) 
  (ram_decrease_factor_2 : ram_decrease_2 = 0.20 * ((initial_ram_price + ram_increase_1) - ram_decrease_1)) 
  (ssd_increase_factor : ssd_increase_1 = 0.10 * initial_ssd_price) 
  (ssd_decrease_factor_1 : ssd_decrease_1 = 0.05 * (initial_ssd_price + ssd_increase_1)) 
  (ssd_decrease_factor_2 : ssd_decrease_2 = 0.12 * ((initial_ssd_price + ssd_increase_1) - ssd_decrease_1)) 
  : 
  ((initial_ram_price + ram_increase_1 - ram_decrease_1 - ram_decrease_2) = 44.20) ∧ 
  ((initial_ssd_price + ssd_increase_1 - ssd_decrease_1 - ssd_decrease_2) = 91.96) := 
by
  sorry

end current_prices_l51_51678


namespace toys_in_stock_l51_51541

theorem toys_in_stock (sold_first_week sold_second_week toys_left toys_initial: ℕ) :
  sold_first_week = 38 → 
  sold_second_week = 26 → 
  toys_left = 19 → 
  toys_initial = sold_first_week + sold_second_week + toys_left → 
  toys_initial = 83 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end toys_in_stock_l51_51541


namespace fraction_of_money_left_l51_51544

theorem fraction_of_money_left (m : ℝ) (b : ℝ) (h1 : (1 / 4) * m = (1 / 2) * b) :
  m - b - 50 = m / 2 - 50 → (m - b - 50) / m = 1 / 2 - 50 / m :=
by sorry

end fraction_of_money_left_l51_51544


namespace find_C_l51_51050

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def A : ℕ := sum_of_digits (4568 ^ 7777)
noncomputable def B : ℕ := sum_of_digits A
noncomputable def C : ℕ := sum_of_digits B

theorem find_C : C = 5 :=
by
  sorry

end find_C_l51_51050


namespace interval_representation_l51_51811

def S : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem interval_representation : S = Set.Ioc (-1) 3 :=
sorry

end interval_representation_l51_51811


namespace number_of_factors_180_l51_51922

def number_of_factors (n : ℕ) : ℕ :=
  (n.factors.eraseDuplicates.map (λ p, n.factors.count p + 1)).foldr (· * ·) 1

theorem number_of_factors_180 : number_of_factors 180 = 18 := by
  sorry

end number_of_factors_180_l51_51922


namespace part1_part2_l51_51842

-- Part (1) prove maximum value of 4 - 2x - 1/x when x > 0 is 0.
theorem part1 (x : ℝ) (h : 0 < x) : 
  4 - 2 * x - (2 / x) ≤ 0 :=
sorry

-- Part (2) prove minimum value of 1/a + 1/b when a + 2b = 1 and a > 0, b > 0 is 3 + 2 * sqrt 2.
theorem part2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 2 * b = 1) :
  (1 / a) + (1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end part1_part2_l51_51842


namespace solution_for_system_of_inequalities_l51_51038

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end solution_for_system_of_inequalities_l51_51038


namespace rename_not_always_possible_l51_51841

variable (G : SimpleGraph ℕ)
variable (A B : ℕ)
variable (adj : G.Adj)

theorem rename_not_always_possible : 
  ¬(∀ (A B : ℕ) (G' : SimpleGraph ℕ), (∀ W, (G.Adj W A ↔ G'.Adj W B) ∧ (G.Adj W B ↔ G'.Adj W A)) → (G = G')) :=
sorry

end rename_not_always_possible_l51_51841


namespace factors_of_180_l51_51927

theorem factors_of_180 : ∃ n : ℕ, n = 18 ∧ ∀ p q r : ℕ, 180 = p^2 * q^2 * r^1 → 
  n = (p + 1) * (q + 1) * (r) :=
by
  sorry

end factors_of_180_l51_51927


namespace num_pos_3_digit_div_by_13_l51_51170

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l51_51170


namespace part1_part2_part3_l51_51295

section CircleLine

-- Given: Circle C with equation x^2 + y^2 - 2x - 2y + 1 = 0
-- Tangent to line l intersecting the x-axis at A and the y-axis at B
variable (a b : ℝ) (ha : a > 2) (hb : b > 2)

-- Ⅰ. Prove that (a - 2)(b - 2) = 2
theorem part1 : (a - 2) * (b - 2) = 2 :=
sorry

-- Ⅱ. Find the equation of the trajectory of the midpoint of segment AB
theorem part2 (x y : ℝ) (hx : x > 1) (hy : y > 1) : (x - 1) * (y - 1) = 1 :=
sorry

-- Ⅲ. Find the minimum value of the area of triangle AOB
theorem part3 : ∃ (area : ℝ), area = 6 :=
sorry

end CircleLine

end part1_part2_part3_l51_51295


namespace range_of_first_term_l51_51301

-- Define the arithmetic sequence and its common difference.
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Define the sum of the first n terms of the sequence.
def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Prove the range of the first term a1 given the conditions.
theorem range_of_first_term (a d : ℤ) (S : ℕ → ℤ) (h1 : d = -2)
  (h2 : ∀ n, S n = sum_of_first_n_terms a d n)
  (h3 : S 7 = S 7)
  (h4 : ∀ n, n ≠ 7 → S n < S 7) :
  12 < a ∧ a < 14 :=
by
  sorry

end range_of_first_term_l51_51301


namespace quadratic_has_two_distinct_real_roots_l51_51054

theorem quadratic_has_two_distinct_real_roots (k : ℝ) : 
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 + 2 * k * r1 + (k - 1) = 0 ∧ r2^2 + 2 * k * r2 + (k - 1) = 0 := 
by 
  sorry

end quadratic_has_two_distinct_real_roots_l51_51054


namespace length_of_side_b_max_area_of_triangle_l51_51609

variable {A B C a b c : ℝ}
variable {triangle_ABC : a + c = 6}
variable {eq1 : (3 - Real.cos A) * Real.sin B = Real.sin A * (1 + Real.cos B)}

-- Theorem for part (1) length of side b
theorem length_of_side_b :
  b = 2 :=
sorry

-- Theorem for part (2) maximum area of the triangle
theorem max_area_of_triangle :
  ∃ (S : ℝ), S = 2 * Real.sqrt 2 :=
sorry

end length_of_side_b_max_area_of_triangle_l51_51609


namespace solution_of_system_of_inequalities_l51_51032

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end solution_of_system_of_inequalities_l51_51032


namespace minimum_value_of_product_l51_51968

theorem minimum_value_of_product (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 9) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 30 := 
sorry

end minimum_value_of_product_l51_51968


namespace area_of_rectangle_stage_8_l51_51933

theorem area_of_rectangle_stage_8 : 
  (∀ n, 4 * 4 = 16) →
  (∀ k, k ≤ 8 → k = k) →
  (8 * 16 = 128) :=
by
  intros h_sq_area h_sequence
  sorry

end area_of_rectangle_stage_8_l51_51933


namespace dot_product_EC_ED_l51_51769

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l51_51769


namespace num_three_digit_div_by_13_l51_51165

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l51_51165


namespace solution_of_system_of_inequalities_l51_51030

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end solution_of_system_of_inequalities_l51_51030


namespace find_speeds_l51_51723

/--
From point A to point B, which are 40 km apart, a pedestrian set out at 4:00 AM,
and a cyclist set out at 7:20 AM. The cyclist caught up with the pedestrian exactly
halfway between A and B, after which both continued their journey. A second cyclist
with the same speed as the first cyclist set out from B to A at 8:30 AM and met the
pedestrian one hour after the pedestrian's meeting with the first cyclist. Prove that
the speed of the pedestrian is 5 km/h and the speed of the cyclists is 30 km/h.
-/
theorem find_speeds (x y : ℝ) : 
  (∀ t : ℝ, (0 <= t ∧ t < (7 + (1/3)) ∨ (7 + (1/3)) <= t ∧ t <= 20) -> (x * t + 20 = y * ((7 + (1/3)) - t))) ∧ -- Midpoint and catch-up condition
  (∀ t, (8 + (1/2) <= t) -> (40 - (x * (8 + (1/2))) = y * (t - (8 + (1/2))))) -> -- Second meeting condition
  x = 5 ∧ y = 30 := 
sorry

end find_speeds_l51_51723


namespace gcf_of_24_and_16_l51_51378

theorem gcf_of_24_and_16 :
  let n := 24
  let lcm := 48
  gcd n 16 = 8 :=
by
  sorry

end gcf_of_24_and_16_l51_51378


namespace max_lg_value_l51_51903

noncomputable def max_lg_product (x y : ℝ) (hx: x > 1) (hy: y > 1) (hxy: Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) : ℝ :=
  4

theorem max_lg_value (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  max_lg_product x y hx hy hxy = 4 := 
by
  unfold max_lg_product
  sorry

end max_lg_value_l51_51903


namespace three_digit_numbers_div_by_13_l51_51108

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l51_51108


namespace kids_french_fries_cost_l51_51008

noncomputable def cost_burger : ℝ := 5
noncomputable def cost_fries : ℝ := 3
noncomputable def cost_soft_drink : ℝ := 3
noncomputable def cost_special_burger_meal : ℝ := 9.50
noncomputable def cost_kids_burger : ℝ := 3
noncomputable def cost_kids_juice_box : ℝ := 2
noncomputable def cost_kids_meal : ℝ := 5
noncomputable def savings : ℝ := 10

noncomputable def total_adult_meal_individual : ℝ := 2 * cost_burger + 2 * cost_fries + 2 * cost_soft_drink
noncomputable def total_adult_meal_deal : ℝ := 2 * cost_special_burger_meal

noncomputable def total_kids_meal_individual (F : ℝ) : ℝ := 2 * cost_kids_burger + 2 * F + 2 * cost_kids_juice_box
noncomputable def total_kids_meal_deal : ℝ := 2 * cost_kids_meal

noncomputable def total_cost_individual (F : ℝ) : ℝ := total_adult_meal_individual + total_kids_meal_individual F
noncomputable def total_cost_deal : ℝ := total_adult_meal_deal + total_kids_meal_deal

theorem kids_french_fries_cost : ∃ F : ℝ, total_cost_individual F - total_cost_deal = savings ∧ F = 3.50 := 
by
  use 3.50
  sorry

end kids_french_fries_cost_l51_51008


namespace find_a_l51_51741

theorem find_a {a : ℝ} :
  (∀ x : ℝ, (ax - 1) / (x + 1) < 0 → (x < -1 ∨ x > -1 / 2)) → a = -2 :=
by 
  intros h
  sorry

end find_a_l51_51741


namespace expression_is_five_l51_51716

-- Define the expression
def given_expression : ℤ := abs (abs (-abs (-2 + 1) - 2) + 2)

-- Prove that the expression equals 5
theorem expression_is_five : given_expression = 5 :=
by
  -- We skip the proof for now
  sorry

end expression_is_five_l51_51716


namespace dot_product_EC_ED_l51_51772

-- Define the context of the square and the points E, C, D
def midpoint (A B: ℝ × ℝ): ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem dot_product_EC_ED :
  ∀ (A B D C E: ℝ × ℝ),
    ABCD_is_square A B C D →
    side_length (A B C D) = 2 →
    E = midpoint A B →
    vector_dot_product (vector_range E C) (vector_range E D) = 3 :=
by
  sorry

end dot_product_EC_ED_l51_51772


namespace power_of_two_expression_l51_51172

theorem power_of_two_expression :
  2^2010 - 2^2009 - 2^2008 + 2^2007 - 2^2006 = 5 * 2^2006 :=
by
  sorry

end power_of_two_expression_l51_51172


namespace dot_product_square_ABCD_l51_51763

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l51_51763


namespace solveProfessions_l51_51896

-- Define our types for Friends and Professions
inductive Friend | Andrey | Boris | Vyacheslav | Gennady
inductive Profession | Architect | Barista | Veterinarian | Guitarist

-- Define the seating arrangement and conditions
def seatingArrangement : Friend → Position := sorry
def profession : Friend → Profession := sorry

-- Conditions translated to Lean
axiom condition1 : ∀ (f : Friend), 
  profession f = Profession.Veterinarian → 
  ∃ (neighbor1 neighbor2 : Friend), 
    seatingArrangement neighbor1 = seatingArrangement f + 1 ∨ seatingArrangement neighbor1 = seatingArrangement f - 1 ∧ 
    seatingArrangement neighbor2 = seatingArrangement f + 1 ∨ seatingArrangement neighbor2 = seatingArrangement f - 1 ∧
    (profession neighbor1 = Profession.Architect ∧ profession neighbor2 = Profession.Guitarist) 

axiom condition2 : ∀ (f : Friend), 
  profession f = Profession.Barista → 
  ∃ (rightNeighbor : Friend), 
    seatingArrangement rightNeighbor = seatingArrangement f + 1 ∧ 
    rightNeighbor = Friend.Boris

axiom condition3 : seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Andrey ∧ seatingArrangement Friend.Vyacheslav > seatingArrangement Friend.Boris

axiom condition4 : ∀ (f : Friend), 
  f = Friend.Andrey → 
  ∃ (leftNeighbor rightNeighbor : Friend), 
    seatingArrangement leftNeighbor = seatingArrangement f - 1 ∧ 
    seatingArrangement rightNeighbor = seatingArrangement f + 1

axiom condition5 : ∀ (f1 f2 : Friend), 
  (profession f1 = Profession.Guitarist ∧ profession f2 = Profession.Barista) → 
  seatingArrangement f1 ≠ seatingArrangement f2 + 1 ∧ seatingArrangement f1 ≠ seatingArrangement f2 - 1

-- Define the positions
def Position : Type := Fin 4

-- The main theorem to prove the assignments
theorem solveProfessions :
  profession Friend.Gennady = Profession.Barista ∧ 
  profession Friend.Boris = Profession.Architect ∧ 
  profession Friend.Andrey = Profession.Veterinarian ∧ 
  profession Friend.Vyacheslav = Profession.Guitarist :=
sorry

end solveProfessions_l51_51896


namespace midpoint_translation_l51_51806

open Real

theorem midpoint_translation :
  let p1 := (3 : ℝ, -2 : ℝ)
      p2 := (-7 : ℝ, 6 : ℝ)
      mp_s3 := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
      translation := (3, -4)
      mp_s4 := (mp_s3.1 + translation.1, mp_s3.2 + translation.2)
  in mp_s4 = (1, -2) :=
by
  let p1 := (3 : ℝ, -2 : ℝ)
  let p2 := (-7 : ℝ, 6 : ℝ)
  let mp_s3 := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let translation := (3, -4)
  let mp_s4 := (mp_s3.1 + translation.1, mp_s3.2 + translation.2)
  show mp_s4 = (1, -2)
  sorry

end midpoint_translation_l51_51806


namespace digit_A_unique_solution_l51_51833

theorem digit_A_unique_solution :
  ∃ (A : ℕ), 0 ≤ A ∧ A < 10 ∧ (100 * A + 72 - 23 = 549) ∧ A = 5 :=
by
  sorry

end digit_A_unique_solution_l51_51833


namespace part1_part2_l51_51572

noncomputable def f (a x : ℝ) : ℝ := (a / x) - Real.log x

theorem part1 (a : ℝ) (x1 x2 : ℝ) (hx1pos : 0 < x1) (hx2pos : 0 < x2) (hxdist : x1 ≠ x2) 
(hf : f a x1 = -3) (hf2 : f a x2 = -3) : a ∈ Set.Ioo (-Real.exp 2) 0 :=
sorry

theorem part2 (x1 x2 : ℝ) (hx1pos : 0 < x1) (hx2pos : 0 < x2) (hxdist : x1 ≠ x2)
(hfa : f (-2) x1 = -3) (hfb : f (-2) x2 = -3) : x1 + x2 > 4 :=
sorry

end part1_part2_l51_51572


namespace smallest_nat_mul_47_last_four_digits_l51_51290

theorem smallest_nat_mul_47_last_four_digits (N : ℕ) :
  (47 * N) % 10000 = 1969 ↔ N = 8127 :=
sorry

end smallest_nat_mul_47_last_four_digits_l51_51290


namespace problem_statement_l51_51619

theorem problem_statement (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
sorry

end problem_statement_l51_51619


namespace common_tangent_intersects_x_axis_at_point_A_l51_51304

-- Define the ellipses using their equations
def ellipse_C1 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def ellipse_C2 (x y : ℝ) : Prop := (x - 2)^2 + 4 * y^2 = 1

-- The theorem stating the coordinates of the point where the common tangent intersects the x-axis
theorem common_tangent_intersects_x_axis_at_point_A :
  (∃ x : ℝ, (ellipse_C1 x 0 ∧ ellipse_C2 x 0) ↔ x = 4) :=
sorry

end common_tangent_intersects_x_axis_at_point_A_l51_51304


namespace number_of_acceptable_outfits_l51_51391

-- Definitions based on conditions
def total_shirts := 5
def total_pants := 4
def restricted_shirts := 2
def restricted_pants := 1

-- Defining the problem statement
theorem number_of_acceptable_outfits : 
  (total_shirts * total_pants - restricted_shirts * restricted_pants + restricted_shirts * (total_pants - restricted_pants)) = 18 :=
by sorry

end number_of_acceptable_outfits_l51_51391


namespace maintenance_check_days_l51_51697

theorem maintenance_check_days (x : ℝ) (hx : x + 0.20 * x = 60) : x = 50 :=
by
  -- this is where the proof would go
  sorry

end maintenance_check_days_l51_51697


namespace koschei_coins_l51_51326

theorem koschei_coins :
  ∃ a : ℕ, a % 10 = 7 ∧ a % 12 = 9 ∧ 300 ≤ a ∧ a ≤ 400 ∧ a = 357 :=
by
  sorry

end koschei_coins_l51_51326


namespace Megatech_budget_allocation_l51_51679

theorem Megatech_budget_allocation :
  let total_degrees := 360
  let degrees_astrophysics := 90
  let home_electronics := 19
  let food_additives := 10
  let genetically_modified_microorganisms := 24
  let industrial_lubricants := 8

  let percentage_astrophysics := (degrees_astrophysics / total_degrees) * 100
  let known_percentages_sum := home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants + percentage_astrophysics
  let percentage_microphotonics := 100 - known_percentages_sum

  percentage_microphotonics = 14 :=
by
  sorry

end Megatech_budget_allocation_l51_51679


namespace find_integers_satisfying_condition_l51_51428

-- Define the inequality condition
def condition (x : ℤ) : Prop := x * x < 3 * x

-- Prove that the set of integers satisfying the condition is {1, 2}
theorem find_integers_satisfying_condition :
  { x : ℤ | condition x } = {1, 2} := 
by {
  sorry
}

end find_integers_satisfying_condition_l51_51428


namespace scientific_notation_of_600000_l51_51283

theorem scientific_notation_of_600000 :
  600000 = 6 * 10^5 :=
sorry

end scientific_notation_of_600000_l51_51283


namespace john_thrice_tom_years_ago_l51_51190

-- Define the ages of Tom and John
def T : ℕ := 16
def J : ℕ := 36

-- Condition that John will be 2 times Tom's age in 4 years
def john_twice_tom_in_4_years (J T : ℕ) : Prop := J + 4 = 2 * (T + 4)

-- The number of years ago John was thrice as old as Tom
def years_ago (J T x : ℕ) : Prop := J - x = 3 * (T - x)

-- Prove that the number of years ago John was thrice as old as Tom is 6
theorem john_thrice_tom_years_ago (h1 : john_twice_tom_in_4_years 36 16) : years_ago 36 16 6 :=
by
  -- Import initial values into the context
  unfold john_twice_tom_in_4_years at h1
  unfold years_ago
  -- Solve the steps, more details in the actual solution
  sorry

end john_thrice_tom_years_ago_l51_51190


namespace sum_of_parts_l51_51252

variable (x y : ℤ)
variable (h1 : x + y = 60)
variable (h2 : y = 45)

theorem sum_of_parts : 10 * x + 22 * y = 1140 :=
by
  sorry

end sum_of_parts_l51_51252


namespace amount_saved_l51_51277

theorem amount_saved (list_price : ℝ) (tech_deals_discount : ℝ) (electro_bargains_discount : ℝ)
    (tech_deals_price : ℝ) (electro_bargains_price : ℝ) (amount_saved : ℝ) :
  tech_deals_discount = 0.15 →
  list_price = 120 →
  tech_deals_price = list_price * (1 - tech_deals_discount) →
  electro_bargains_discount = 20 →
  electro_bargains_price = list_price - electro_bargains_discount →
  amount_saved = tech_deals_price - electro_bargains_price →
  amount_saved = 2 :=
by
  -- proof steps would go here
  sorry

end amount_saved_l51_51277


namespace range_of_y_l51_51585

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 120) : y ∈ Set.Ioo (-11 : ℝ) (-10 : ℝ) :=
sorry

end range_of_y_l51_51585


namespace number_of_three_digit_numbers_divisible_by_13_l51_51123

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l51_51123


namespace remainder_when_divided_by_39_l51_51403

theorem remainder_when_divided_by_39 (N : ℤ) (h1 : ∃ k : ℤ, N = 13 * k + 3) : N % 39 = 3 :=
sorry

end remainder_when_divided_by_39_l51_51403


namespace sum_of_arithmetic_sequence_l51_51470

theorem sum_of_arithmetic_sequence (a d1 d2 : ℕ) 
  (h1 : d1 = d2 + 2) 
  (h2 : d1 + d2 = 24) 
  (a_pos : 0 < a) : 
  (a + (a + d1) + (a + d1) + (a + d1 + d2) = 54) := 
by 
  sorry

end sum_of_arithmetic_sequence_l51_51470


namespace hair_cut_amount_l51_51004

theorem hair_cut_amount (initial_length final_length cut_length : ℕ) (h1 : initial_length = 11) (h2 : final_length = 7) : cut_length = 4 :=
by 
  sorry

end hair_cut_amount_l51_51004


namespace max_intersections_l51_51246

/-- Given two different circles and three different straight lines, the maximum number of
points of intersection on a plane is 17. -/
theorem max_intersections (c1 c2 : Circle) (l1 l2 l3 : Line) (h_distinct_cir : c1 ≠ c2) (h_distinct_lines : ∀ (l1 l2 : Line), l1 ≠ l2) :
  ∃ (n : ℕ), n = 17 :=
by
  sorry

end max_intersections_l51_51246


namespace upper_limit_for_y_l51_51588

theorem upper_limit_for_y (x y : ℝ) (hx : 5 < x) (hx' : x < 8) (hy : 8 < y) (h_diff : y - x = 7) : y ≤ 14 :=
by
  sorry

end upper_limit_for_y_l51_51588


namespace max_value_6a_3b_10c_l51_51789

theorem max_value_6a_3b_10c (a b c : ℝ) (h : 9 * a ^ 2 + 4 * b ^ 2 + 25 * c ^ 2 = 1) : 
  6 * a + 3 * b + 10 * c ≤ (Real.sqrt 41) / 2 :=
sorry

end max_value_6a_3b_10c_l51_51789


namespace rectangle_with_perpendicular_diagonals_is_square_l51_51249

-- Define rectangle and its properties
structure Rectangle where
  length : ℝ
  width : ℝ
  opposite_sides_equal : length = width

-- Define the condition that the diagonals of the rectangle are perpendicular
axiom perpendicular_diagonals {r : Rectangle} : r.length = r.width → True

-- Define the square property that a rectangle with all sides equal is a square
structure Square extends Rectangle where
  all_sides_equal : length = width

-- The main theorem to be proven
theorem rectangle_with_perpendicular_diagonals_is_square (r : Rectangle) (h : r.length = r.width) : Square := by
  sorry

end rectangle_with_perpendicular_diagonals_is_square_l51_51249


namespace domain_of_f_3x_minus_1_domain_of_f_l51_51251

-- Problem (1): Domain of f(3x - 1)
theorem domain_of_f_3x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -2 ≤ f x ∧ f x ≤ 1) →
  (∀ x, -1 / 3 ≤ x ∧ x ≤ 2 / 3) :=
by
  intro h
  sorry

-- Problem (2): Domain of f(x)
theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, -1 ≤ 2*x + 5 ∧ 2*x + 5 ≤ 4) →
  (∀ y, 3 ≤ y ∧ y ≤ 13) :=
by
  intro h
  sorry

end domain_of_f_3x_minus_1_domain_of_f_l51_51251


namespace quadratic_expression_representation_quadratic_expression_integer_iff_l51_51673

theorem quadratic_expression_representation (A B C : ℤ) :
  ∃ (k l m : ℤ), 
    (k = 2 * A) ∧ 
    (l = A + B) ∧ 
    (m = C) ∧ 
    (∀ x : ℤ, A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m) := 
sorry

theorem quadratic_expression_integer_iff (A B C : ℤ) :
  (∀ x : ℤ, ∃ k l m : ℤ, (k = 2 * A) ∧ (l = A + B) ∧ (m = C) ∧ (A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m)) ↔ 
  (A % 1 = 0 ∧ B % 1 = 0 ∧ C % 1 = 0) := 
sorry

end quadratic_expression_representation_quadratic_expression_integer_iff_l51_51673


namespace stone_length_is_correct_l51_51686

variable (length_m width_m : ℕ)
variable (num_stones : ℕ)
variable (width_stone dm : ℕ)

def length_of_each_stone (length_m : ℕ) (width_m : ℕ) (num_stones : ℕ) (width_stone : ℕ) : ℕ :=
  let length_dm := length_m * 10
  let width_dm := width_m * 10
  let area_hall := length_dm * width_dm
  let area_stone := width_stone * 5
  (area_hall / num_stones) / width_stone

theorem stone_length_is_correct :
  length_of_each_stone 36 15 5400 5 = 2 := by
  sorry

end stone_length_is_correct_l51_51686


namespace count_three_digit_numbers_divisible_by_13_l51_51084

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l51_51084


namespace rectangle_area_l51_51775

theorem rectangle_area
  (x : ℝ)
  (perimeter_eq_160 : 10 * x = 160) :
  4 * (4 * x * x) = 1024 :=
by
  -- We would solve the problem and show the steps here
  sorry

end rectangle_area_l51_51775


namespace possible_values_f_l51_51785

noncomputable def f (x y z : ℝ) : ℝ := (y / (y + x)) + (z / (z + y)) + (x / (x + z))

theorem possible_values_f (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) (h7 : x^2 + y^3 = z^4) : 
  1 < f x y z ∧ f x y z < 2 :=
sorry

end possible_values_f_l51_51785


namespace three_digit_numbers_div_by_13_l51_51111

theorem three_digit_numbers_div_by_13 : 
  card {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ 13 ∣ n} = 69 :=
by
  sorry

end three_digit_numbers_div_by_13_l51_51111


namespace repeating_sequence_length_1_over_221_l51_51986

theorem repeating_sequence_length_1_over_221 : ∃ n : ℕ, (10 ^ n ≡ 1 [MOD 221]) ∧ (∀ m : ℕ, (10 ^ m ≡ 1 [MOD 221]) → (n ≤ m)) ∧ n = 48 :=
by
  sorry

end repeating_sequence_length_1_over_221_l51_51986


namespace range_of_quadratic_function_is_geq_11_over_4_l51_51240

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - x + 3

-- Define the range of the quadratic function
def range_of_quadratic_function := {y : ℝ | ∃ x : ℝ, quadratic_function x = y}

-- Prove the statement
theorem range_of_quadratic_function_is_geq_11_over_4 : range_of_quadratic_function = {y : ℝ | y ≥ 11 / 4} :=
by
  sorry

end range_of_quadratic_function_is_geq_11_over_4_l51_51240


namespace answer_l51_51760

-- Definitions of geometric entities in terms of vectors
structure Square :=
  (A B C D E : ℝ × ℝ)
  (side_length : ℝ)
  (hAB_eq : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = side_length ^ 2)
  (hBC_eq : (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = side_length ^ 2)
  (hCD_eq : (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = side_length ^ 2)
  (hDA_eq : (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2 = side_length ^ 2)
  (hE_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def EC_ED_dot_product (s : Square) : ℝ :=
  let EC := (s.C.1 - s.E.1, s.C.2 - s.E.2)
  let ED := (s.D.1 - s.E.1, s.D.2 - s.E.2)
  dot_product EC ED

theorem answer (s : Square) (h_side_length : s.side_length = 2) :
  EC_ED_dot_product s = 3 :=
sorry

end answer_l51_51760


namespace range_of_a_zero_value_of_a_minimum_l51_51739

noncomputable def f (x a : ℝ) : ℝ := Real.log x + (7 * a) / x

-- Problem 1: Range of a where f(x) has exactly one zero in its domain
theorem range_of_a_zero (a : ℝ) : 
  (∃! x : ℝ, (0 < x) ∧ f x a = 0) ↔ (a ∈ Set.Iic 0 ∪ {1 / (7 * Real.exp 1)}) := sorry

-- Problem 2: Value of a such that the minimum value of f(x) on [e, e^2] is 3
theorem value_of_a_minimum (a : ℝ) : 
  (∃ x : ℝ, (Real.exp 1 ≤ x ∧ x ≤ Real.exp 2) ∧ f x a = 3) ↔ (a = (Real.exp 2)^2 / 7) := sorry

end range_of_a_zero_value_of_a_minimum_l51_51739


namespace count_3_digit_numbers_divisible_by_13_l51_51141

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51141


namespace divide_5440_K_l51_51876

theorem divide_5440_K (a b c d : ℕ) 
  (h1 : 5440 = a + b + c + d)
  (h2 : 2 * b = 3 * a)
  (h3 : 3 * c = 5 * b)
  (h4 : 5 * d = 6 * c) : 
  a = 680 ∧ b = 1020 ∧ c = 1700 ∧ d = 2040 :=
by 
  sorry

end divide_5440_K_l51_51876


namespace solve_for_p_l51_51352

theorem solve_for_p (q p : ℝ) (h : p^2 * q = p * q + p^2) : 
  p = 0 ∨ p = q / (q - 1) :=
by
  sorry

end solve_for_p_l51_51352


namespace total_money_l51_51401

theorem total_money (A B C : ℝ) (h1 : A = 1 / 2 * (B + C))
  (h2 : B = 2 / 3 * (A + C)) (h3 : A = 122) :
  A + B + C = 366 := by
  sorry

end total_money_l51_51401


namespace newLampTaller_l51_51777

-- Define the heights of the old and new lamps
def oldLampHeight : ℝ := 1
def newLampHeight : ℝ := 2.33

-- Define the proof statement
theorem newLampTaller : newLampHeight - oldLampHeight = 1.33 :=
by
  sorry

end newLampTaller_l51_51777


namespace num_3_digit_div_by_13_l51_51156

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l51_51156


namespace divisors_squared_prime_l51_51860

theorem divisors_squared_prime (p : ℕ) [hp : Fact (Nat.Prime p)] (m : ℕ) (h : m = p^3) (hm_div : Nat.divisors m = 4) :
  Nat.divisors (m^2) = 7 :=
sorry

end divisors_squared_prime_l51_51860


namespace expression_value_l51_51271

theorem expression_value :
  (1 / (3 - (1 / (3 + (1 / (3 - (1 / 3))))))) = (27 / 73) :=
by 
  sorry

end expression_value_l51_51271


namespace base_log_eq_l51_51823

theorem base_log_eq (x : ℝ) : (5 : ℝ)^(x + 7) = (6 : ℝ)^x → x = Real.logb (6 / 5 : ℝ) (5^7 : ℝ) := by
  sorry

end base_log_eq_l51_51823


namespace find_other_number_l51_51228

theorem find_other_number (x y : ℕ) (h_gcd : Nat.gcd x y = 22) (h_lcm : Nat.lcm x y = 5940) (h_x : x = 220) :
  y = 594 :=
sorry

end find_other_number_l51_51228


namespace tangent_line_ellipse_l51_51047

variable (a b x0 y0 : ℝ)
variable (x y : ℝ)

def ellipse (x y a b : ℝ) := (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

theorem tangent_line_ellipse :
  ellipse x y a b ∧ a > b ∧ (x0 ≠ 0 ∨ y0 ≠ 0) ∧ (x0 ^ 2) / (a ^ 2) + (y0 ^ 2) / (b ^ 2) > 1 →
  (x0 * x) / (a ^ 2) + (y0 * y) / (b ^ 2) = 1 :=
  sorry

end tangent_line_ellipse_l51_51047


namespace necessary_condition_for_inequality_l51_51947

-- Definitions based on the conditions in a)
variables (A B C D : ℝ)

-- Main statement translating c) into Lean
theorem necessary_condition_for_inequality (h : C < D) : A > B :=
by sorry

end necessary_condition_for_inequality_l51_51947


namespace normal_distribution_symmetry_proof_l51_51570

noncomputable theory

open Real

variables (σ : ℝ)
def X : Type := ℝ

def P (X : ℝ) := sorry -- Placeholder for probability function on X

theorem normal_distribution_symmetry_proof
  (h1 : ∀ X, X ~ Normal(2, σ^2))
  (h2 : P(X < 4) = 0.84) :
  P(X ≤ 0) = 0.16 :=
sorry

end normal_distribution_symmetry_proof_l51_51570


namespace count_3_digit_numbers_divisible_by_13_l51_51140

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51140


namespace abc_order_l51_51866

noncomputable def a : ℝ := Real.log (3 / 2) - 3 / 2
noncomputable def b : ℝ := Real.log Real.pi - Real.pi
noncomputable def c : ℝ := Real.log 3 - 3

theorem abc_order : a > c ∧ c > b := by
  have h₁: a = Real.log (3 / 2) - 3 / 2 := rfl
  have h₂: b = Real.log Real.pi - Real.pi := rfl
  have h₃: c = Real.log 3 - 3 := rfl
  sorry

end abc_order_l51_51866


namespace num_three_digit_div_by_13_l51_51161

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l51_51161


namespace two_leq_one_add_one_div_n_pow_n_lt_three_l51_51489

theorem two_leq_one_add_one_div_n_pow_n_lt_three :
  ∀ (n : ℕ), 2 ≤ (1 + (1 : ℝ) / n) ^ n ∧ (1 + (1 : ℝ) / n) ^ n < 3 := 
by 
  sorry

end two_leq_one_add_one_div_n_pow_n_lt_three_l51_51489


namespace domain_of_composite_function_l51_51638

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x → x ≤ 2 → f x = f x) →
  (∀ (x : ℝ), -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2 → f (x^2) = f (x^2)) :=
by
  sorry

end domain_of_composite_function_l51_51638


namespace answer_l51_51761

-- Definitions of geometric entities in terms of vectors
structure Square :=
  (A B C D E : ℝ × ℝ)
  (side_length : ℝ)
  (hAB_eq : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = side_length ^ 2)
  (hBC_eq : (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = side_length ^ 2)
  (hCD_eq : (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = side_length ^ 2)
  (hDA_eq : (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2 = side_length ^ 2)
  (hE_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def EC_ED_dot_product (s : Square) : ℝ :=
  let EC := (s.C.1 - s.E.1, s.C.2 - s.E.2)
  let ED := (s.D.1 - s.E.1, s.D.2 - s.E.2)
  dot_product EC ED

theorem answer (s : Square) (h_side_length : s.side_length = 2) :
  EC_ED_dot_product s = 3 :=
sorry

end answer_l51_51761


namespace boxes_sold_l51_51187

theorem boxes_sold (start_boxes sold_boxes left_boxes : ℕ) (h1 : start_boxes = 10) (h2 : left_boxes = 5) (h3 : start_boxes - sold_boxes = left_boxes) : sold_boxes = 5 :=
by
  sorry

end boxes_sold_l51_51187


namespace sampling_method_is_systematic_l51_51318

def conveyor_belt_sampling (interval: ℕ) (product_picking: ℕ → ℕ) : Prop :=
  ∀ (n: ℕ), product_picking n = n * interval

theorem sampling_method_is_systematic
  (interval: ℕ)
  (product_picking: ℕ → ℕ)
  (h: conveyor_belt_sampling interval product_picking) :
  interval = 30 → product_picking = systematic_sampling := 
sorry

end sampling_method_is_systematic_l51_51318


namespace prob_both_white_is_two_fifth_prob_one_white_one_black_is_eight_fifteenth_l51_51242

-- Conditions
def total_balls : ℕ := 6
def white_balls : ℕ := 4
def black_balls : ℕ := 2

-- Events
def total_outcomes : ℕ := (total_balls * (total_balls - 1)) / 2
def white_white_outcomes : ℕ := (white_balls * (white_balls - 1)) / 2
def white_black_outcomes : ℕ := white_balls * black_balls

-- Probabilities
def prob_both_white : ℚ := (white_white_outcomes : ℚ) / (total_outcomes : ℚ)
def prob_one_white_one_black : ℚ := (white_black_outcomes : ℚ) / (total_outcomes : ℚ)

theorem prob_both_white_is_two_fifth 
  (h1 : total_balls = 6)
  (h2 : white_balls = 4)
  (h3 : black_balls = 2)
  (h4 : white_white_outcomes = 6)
  (h5 : total_outcomes = 15) :
  prob_both_white = 2 / 5 := by
  sorry

theorem prob_one_white_one_black_is_eight_fifteenth
  (h1 : total_balls = 6)
  (h2 : white_balls = 4)
  (h3 : black_balls = 2)
  (h4 : white_black_outcomes = 8)
  (h5 : total_outcomes = 15) :
  prob_one_white_one_black = 8 / 15 := by
  sorry

end prob_both_white_is_two_fifth_prob_one_white_one_black_is_eight_fifteenth_l51_51242


namespace number_of_three_digit_numbers_divisible_by_13_l51_51120

-- Definitions related to conditions:
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n ≤ 999)
def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

-- Main theorem statement: proof that there are exactly 69 three-digit numbers divisible by 13
theorem number_of_three_digit_numbers_divisible_by_13 :
  {n : ℕ | is_three_digit n ∧ is_divisible_by_13 n}.card = 69 := 
begin
  sorry -- proof is omitted
end

end number_of_three_digit_numbers_divisible_by_13_l51_51120


namespace boxcar_capacity_ratio_l51_51629

-- The known conditions translated into Lean definitions
def red_boxcar_capacity (B : ℕ) : ℕ := 3 * B
def blue_boxcar_count : ℕ := 4
def red_boxcar_count : ℕ := 3
def black_boxcar_count : ℕ := 7
def black_boxcar_capacity : ℕ := 4000
def total_capacity : ℕ := 132000

-- The mathematical condition as a Lean theorem statement.
theorem boxcar_capacity_ratio 
  (B : ℕ)
  (h_condition : (red_boxcar_count * red_boxcar_capacity B + 
                  blue_boxcar_count * B + 
                  black_boxcar_count * black_boxcar_capacity = 
                  total_capacity)) : 
  black_boxcar_capacity / B = 1 / 2 := 
sorry

end boxcar_capacity_ratio_l51_51629


namespace problem_solution_l51_51383

theorem problem_solution :
  (3012 - 2933)^2 / 196 = 32 := sorry

end problem_solution_l51_51383


namespace bowling_ball_weight_l51_51890

theorem bowling_ball_weight (b c : ℕ) 
  (h1 : 5 * b = 3 * c) 
  (h2 : 3 * c = 105) : 
  b = 21 := 
  sorry

end bowling_ball_weight_l51_51890


namespace num_pos_3_digit_div_by_13_l51_51168

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l51_51168


namespace odd_function_increasing_function_l51_51053

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem odd_function (x : ℝ) : 
  (f (1 / 2) (-x)) = -(f (1 / 2) x) := 
by
  sorry

theorem increasing_function : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f (1 / 2) x₁ < f (1 / 2) x₂ := 
by
  sorry

end odd_function_increasing_function_l51_51053


namespace kathleen_allowance_l51_51475

theorem kathleen_allowance (x : ℝ) :
  let middle_school_allowance := 10
  let senior_year_allowance := 10 * x + 5
  let percentage_increase := ((senior_year_allowance - middle_school_allowance) / middle_school_allowance) * 100
  percentage_increase = 150 → x = 2 :=
by
  -- Definitions and conditions setup
  let middle_school_allowance := 10
  let senior_year_allowance := 10 * x + 5
  let percentage_increase := ((senior_year_allowance - middle_school_allowance) / middle_school_allowance) * 100
  intros h
  -- Skipping the proof
  sorry

end kathleen_allowance_l51_51475


namespace speeds_correct_l51_51724

-- Definitions for conditions
def distance (A B : Type) := 40 -- given distance between A and B is 40 km
def start_time_pedestrian : Real := 4 -- pedestrian starts at 4:00 AM
def start_time_cyclist : Real := 7 + (20 / 60) -- cyclist starts at 7:20 AM
def midpoint_distance : Real := 20 -- the midpoint distance where cyclist catches up with pedestrian is 20 km

noncomputable def speeds (x y : Real) : Prop :=
  let t_catch_up := (20 - (10 / 3) * x) / (y - x) in -- time taken by the cyclist to catch up
  let t_total := (10 / 3) + t_catch_up + 1 in -- total time for pedestrian until meeting second cyclist
  4.5 = t_total ∧ -- total time in hours from 4:00 AM to 8:30 AM
  10 * x * (y - x) + 60 * x - 10 * x^2 = 60 * y - 60 * x ∧ -- initial condition simplification step
  y = 6 * x -- relationship between speeds based on derived equations

-- The proposition to prove
theorem speeds_correct : ∃ x y : Real, speeds x y ∧ x = 5 ∧ y = 30 :=
by
  sorry

end speeds_correct_l51_51724


namespace distance_sum_conditions_l51_51790

theorem distance_sum_conditions (a : ℚ) (k : ℚ) :
  abs (20 * a - 20 * k - 190) = 4460 ∧ abs (20 * a^2 - 20 * k - 190) = 2755 →
  a = -37 / 2 ∨ a = 39 / 2 :=
sorry

end distance_sum_conditions_l51_51790


namespace answer_l51_51758

-- Definitions of geometric entities in terms of vectors
structure Square :=
  (A B C D E : ℝ × ℝ)
  (side_length : ℝ)
  (hAB_eq : (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = side_length ^ 2)
  (hBC_eq : (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = side_length ^ 2)
  (hCD_eq : (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = side_length ^ 2)
  (hDA_eq : (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2 = side_length ^ 2)
  (hE_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def EC_ED_dot_product (s : Square) : ℝ :=
  let EC := (s.C.1 - s.E.1, s.C.2 - s.E.2)
  let ED := (s.D.1 - s.E.1, s.D.2 - s.E.2)
  dot_product EC ED

theorem answer (s : Square) (h_side_length : s.side_length = 2) :
  EC_ED_dot_product s = 3 :=
sorry

end answer_l51_51758


namespace count_three_digit_numbers_divisible_by_13_l51_51148

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l51_51148


namespace problem_statement_l51_51620

theorem problem_statement (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := 
sorry

end problem_statement_l51_51620


namespace derivative_f_at_zero_l51_51843

noncomputable def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then 4 * x * (1 - |x|) else 0

theorem derivative_f_at_zero : HasDerivAt f 4 0 :=
by
  -- Proof omitted
  sorry

end derivative_f_at_zero_l51_51843


namespace assign_teachers_to_classes_l51_51844

-- Define the given conditions as variables and constants
theorem assign_teachers_to_classes :
  (∃ ways : ℕ, ways = 36) :=
by
  sorry

end assign_teachers_to_classes_l51_51844


namespace line_equation_l51_51408

theorem line_equation {L : ℝ → ℝ → Prop} (h1 : L (-3) (-2)) 
  (h2 : ∃ a : ℝ, a ≠ 0 ∧ (L a 0 ∧ L 0 a)) :
  (∀ x y, L x y ↔ 2 * x - 3 * y = 0) ∨ (∀ x y, L x y ↔ x + y + 5 = 0) :=
by 
  sorry

end line_equation_l51_51408


namespace number_of_sister_pairs_is_two_l51_51940

noncomputable def f : ℝ → ℝ :=
λ x, if x < 0 then x^2 + 2 * x else 2 / (Real.exp x)

def symmetric_about_origin (A B : ℝ × ℝ) : Prop :=
B.1 = -A.1 ∧ B.2 = -A.2

def sister_pairs (f : ℝ → ℝ) : ℕ :=
Set.card {pair : ℝ × ℝ | (symmetric_about_origin pair.fst pair.snd) ∧ pair.fst.2 = f pair.fst.1 ∧ pair.snd.2 = f pair.snd.1}

theorem number_of_sister_pairs_is_two : sister_pairs f = 2 := by
  sorry

end number_of_sister_pairs_is_two_l51_51940


namespace select_representatives_l51_51368

theorem select_representatives
  (female_count : ℕ) (male_count : ℕ)
  (female_count_eq : female_count = 4)
  (male_count_eq : male_count = 6) :
  female_count * male_count = 24 := by
  sorry

end select_representatives_l51_51368


namespace problem_statement_l51_51900

noncomputable def a : ℝ := Real.tan (1 / 2)
noncomputable def b : ℝ := Real.tan (2 / Real.pi)
noncomputable def c : ℝ := Real.sqrt 3 / Real.pi

theorem problem_statement : a < c ∧ c < b := by
  sorry

end problem_statement_l51_51900


namespace count_3_digit_numbers_divisible_by_13_l51_51093

noncomputable def num_3_digit_div_by_13 : ℕ :=
  let a := 104
  let d := 13
  let l := 988
  in ((l - a) / d) + 1

theorem count_3_digit_numbers_divisible_by_13 : num_3_digit_div_by_13 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51093


namespace angle_equality_l51_51782

open Real

structure Point2D (α : Type) :=
  (x : α)
  (y : α)

noncomputable def semicircle (A B : Point2D ℝ) : set (Point2D ℝ) :=
  {C | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧
       C.x = (A.x + B.x) / 2 + ((A.x - B.x) / 2) * cos θ ∧
       C.y = ((A.x - B.x) / 2) * sin θ}

noncomputable structure Triangle (α : Type) :=
  (A B P : Point2D α)

noncomputable def incircle (T : Triangle ℝ) : set (Point2D ℝ) :=
  {C | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * π ∧
       let R := (dist T.A T.B + dist T.B T.P + dist T.P T.A) / 2 in
       dist C T.A = dist C T.B ∧
       dist C T.B = dist C T.P ∧
       dist C T.P = R}

axiom angle_obtuse (T : Triangle ℝ) : Prop

noncomputable def tangent_points (T : Triangle ℝ) (IC : set (Point2D ℝ)) :
  Point2D ℝ × Point2D ℝ :=
  ({ x := 0, y := 0 }, { x := 0, y := 0 }) -- Placeholder values

noncomputable def line_intersect_semicircle
  (M N : Point2D ℝ) (semicircle : set (Point2D ℝ)) :
  Point2D ℝ × Point2D ℝ :=
  ({ x := 0, y := 0 }, { x := 0, y := 0 }) -- Placeholder values

theorem angle_equality
  (A B P : Point2D ℝ)
  (hP_in_semicircle : P ∈ semicircle A B)
  (hAPB_obtuse : angle_obtuse { A := A, B := B, P := P } )
  (incircle_of_triangle : set (Point2D ℝ) := incircle { A := A, B := B, P := P })
  (M N : Point2D ℝ := (tangent_points { A := A, B := B, P := P } incircle_of_triangle).fst,
   (tangent_points { A := A, B := B, P := P } incircle_of_triangle).snd)
  (X Y : Point2D ℝ := (line_intersect_semicircle M N (semicircle A B)).fst,
    (line_intersect_semicircle M N (semicircle A B)).snd) :
  -- Placeholder proof outline
  sorry

end angle_equality_l51_51782


namespace range_of_m_l51_51485

theorem range_of_m (m : ℝ) :
  (∀ x, |x^2 - 4 * x + m| ≤ x + 4 ↔ (-4 ≤ m ∧ m ≤ 4)) ∧
  (∀ x, (x = 0 → |0^2 - 4 * 0 + m| ≤ 0 + 4) ∧ (x = 2 → ¬(|2^2 - 4 * 2 + m| ≤ 2 + 4))) →
  (-4 ≤ m ∧ m < -2) :=
by
  sorry

end range_of_m_l51_51485


namespace luca_lost_more_weight_l51_51268

theorem luca_lost_more_weight (barbi_kg_month : ℝ) (luca_kg_year : ℝ) (months_in_year : ℕ) (years : ℕ) 
(h_barbi : barbi_kg_month = 1.5) (h_luca : luca_kg_year = 9) (h_months_in_year : months_in_year = 12) (h_years : years = 11) : 
  (luca_kg_year * years) - (barbi_kg_month * months_in_year * (years / 11)) = 81 := 
by 
  sorry

end luca_lost_more_weight_l51_51268


namespace common_property_of_rectangles_rhombuses_and_squares_l51_51646

-- Definitions of shapes and properties

-- Assume properties P1 = "Diagonals are equal", P2 = "Diagonals bisect each other", 
-- P3 = "Diagonals are perpendicular to each other", and P4 = "Diagonals bisect each other and are equal"

def is_rectangle (R : Type) : Prop := sorry
def is_rhombus (R : Type) : Prop := sorry
def is_square (R : Type) : Prop := sorry

def diagonals_bisect_each_other (R : Type) : Prop := sorry

-- Theorem stating the common property
theorem common_property_of_rectangles_rhombuses_and_squares 
  (R : Type)
  (H_rect : is_rectangle R)
  (H_rhomb : is_rhombus R)
  (H_square : is_square R) :
  diagonals_bisect_each_other R := 
  sorry

end common_property_of_rectangles_rhombuses_and_squares_l51_51646


namespace full_price_ticket_revenue_l51_51537

theorem full_price_ticket_revenue 
  (f h p : ℕ)
  (h1 : f + h = 160)
  (h2 : f * p + h * (p / 3) = 2400) :
  f * p = 400 := 
sorry

end full_price_ticket_revenue_l51_51537


namespace kelly_spends_correct_amount_l51_51418

noncomputable def total_cost_with_discount : ℝ :=
  let mango_cost_per_pound := (0.60 : ℝ) * 2
  let orange_cost_per_pound := (0.40 : ℝ) * 4
  let mango_total_cost := 5 * mango_cost_per_pound
  let orange_total_cost := 5 * orange_cost_per_pound
  let total_cost_without_discount := mango_total_cost + orange_total_cost
  let discount := 0.10 * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount
  total_cost_with_discount

theorem kelly_spends_correct_amount :
  total_cost_with_discount = 12.60 := by
  sorry

end kelly_spends_correct_amount_l51_51418


namespace proof_of_p_and_not_q_l51_51578

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x
def q : Prop := ∀ x : ℝ, exp x > 1

theorem proof_of_p_and_not_q : p ∧ ¬q :=
by {
  sorry
}

end proof_of_p_and_not_q_l51_51578


namespace min_value_at_neg7_l51_51386

noncomputable def f (x : ℝ) : ℝ := x^2 + 14 * x + 24

theorem min_value_at_neg7 : ∀ x : ℝ, f (-7) ≤ f x :=
by
  sorry

end min_value_at_neg7_l51_51386


namespace intersection_M_N_l51_51199

open Set

def M : Set ℝ := {x | x ≥ 0}
def N : Set ℝ := {x | x^2 < 1}

theorem intersection_M_N : M ∩ N = Ico 0 1 := 
sorry

end intersection_M_N_l51_51199


namespace tan_a1_a13_eq_sqrt3_l51_51910

-- Definition of required constants and properties of the geometric sequence
noncomputable def a (n : Nat) : ℝ := sorry -- Geometric sequence definition (abstract)

-- Given condition: a_3 * a_11 + 2 * a_7^2 = 4π
axiom geom_seq_cond : a 3 * a 11 + 2 * (a 7)^2 = 4 * Real.pi

-- Property of geometric sequence: a_3 * a_11 = a_7^2
axiom geom_seq_property : a 3 * a 11 = (a 7)^2

-- To prove: tan(a_1 * a_13) = √3
theorem tan_a1_a13_eq_sqrt3 : Real.tan (a 1 * a 13) = Real.sqrt 3 := by
  sorry

end tan_a1_a13_eq_sqrt3_l51_51910


namespace cube_dimension_ratio_l51_51682

theorem cube_dimension_ratio (V1 V2 : ℕ) (h1 : V1 = 27) (h2 : V2 = 216) :
  ∃ r : ℕ, r = 2 ∧ (∃ l1 l2 : ℕ, l1 * l1 * l1 = V1 ∧ l2 * l2 * l2 = V2 ∧ l2 = r * l1) :=
by
  sorry

end cube_dimension_ratio_l51_51682


namespace algebraic_expression_value_l51_51568

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^2 + a + 1 = 2 :=
sorry

end algebraic_expression_value_l51_51568


namespace geometric_sequence_general_formula_no_arithmetic_sequence_l51_51366

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Condition: Sum of the first n terms of the sequence {a_n} is S_n
-- and S_n = 2a_n - n for n \in \mathbb{N}^*.
axiom sum_condition (n : ℕ) (h : n > 0) : S n = 2 * a n - n

-- Question 1: Prove that the sequence {a_n + 1} forms a geometric sequence.
theorem geometric_sequence (n : ℕ) (h : n > 0) : ∃ r, r ≠ 0 ∧ ∀ m, m > 0 → a (m + 1) + 1 = r * (a m + 1) := 
sorry

-- Question 2: Find the general formula for the sequence {a_n}.
theorem general_formula (n : ℕ) (h : n > 0) : a n = 2 ^ n - 1 := 
sorry

-- Question 3: Prove that there do not exist three consecutive terms in the sequence {a_n} that can form an arithmetic sequence.
theorem no_arithmetic_sequence (k : ℕ) (h : k > 0) : ¬ ∃ k, k > 0 ∧ a k = (a (k + 1) + a (k + 2)) / 2 := 
sorry

end geometric_sequence_general_formula_no_arithmetic_sequence_l51_51366


namespace solution_for_system_of_inequalities_l51_51040

theorem solution_for_system_of_inequalities (p : ℝ) : 
  (19 * p < 10) ∧ (p > 0.5) ↔ (0.5 < p ∧ p < 10/19) := 
by {
  sorry
}

end solution_for_system_of_inequalities_l51_51040


namespace eight_odot_six_eq_ten_l51_51363

-- Define the operation ⊙ as given in the problem statement
def operation (a b : ℕ) : ℕ := a + (3 * a) / (2 * b)

-- State the theorem to prove
theorem eight_odot_six_eq_ten : operation 8 6 = 10 :=
by
  -- Here you will provide the proof, but we skip it with sorry
  sorry

end eight_odot_six_eq_ten_l51_51363


namespace number_of_children_bikes_l51_51869

theorem number_of_children_bikes (c : ℕ) 
  (regular_bikes : ℕ) (wheels_per_regular_bike : ℕ) 
  (wheels_per_children_bike : ℕ) (total_wheels : ℕ)
  (h1 : regular_bikes = 7) 
  (h2 : wheels_per_regular_bike = 2) 
  (h3 : wheels_per_children_bike = 4) 
  (h4 : total_wheels = 58) 
  (h5 : total_wheels = (regular_bikes * wheels_per_regular_bike) + (c * wheels_per_children_bike)) 
  : c = 11 :=
by
  sorry

end number_of_children_bikes_l51_51869


namespace largest_prime_factor_1729_l51_51996

theorem largest_prime_factor_1729 : ∃ p, nat.prime p ∧ p ∣ 1729 ∧ ∀ q, nat.prime q ∧ q ∣ 1729 → q ≤ p :=
by {
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { norm_num },
  { intros q hq hq_div,
    have h := nat.dvd_prime_mul fs nat.prime_7 nat.prime_13 nat.prime_19 1729,
    specialize h q hq_div,
    cases h,
    { exact nat.le_of_dvd hq_div (by norm_num) },
    cases h,
    { norm_num at h,
      subst h },
    { exfalso,
      exact nat.prime.ne_zero hq (by norm_num) }
  }
}

end largest_prime_factor_1729_l51_51996


namespace exists_term_not_of_form_l51_51803

theorem exists_term_not_of_form (a d : ℕ) (h_seq : ∀ i j : ℕ, (i < 40 ∧ j < 40 ∧ i ≠ j) → a + i * d ≠ a + j * d)
  (pos_a : a > 0) (pos_d : d > 0) 
  : ∃ h : ℕ, h < 40 ∧ ¬ ∃ k l : ℕ, a + h * d = 2^k + 3^l :=
by {
  sorry
}

end exists_term_not_of_form_l51_51803


namespace max_distance_circle_ellipse_l51_51969

theorem max_distance_circle_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 2}
  let ellipse := {p : ℝ × ℝ | p.1^2 / 10 + p.2^2 = 1}
  ∀ (P Q : ℝ × ℝ), P ∈ circle → Q ∈ ellipse → 
  dist P Q ≤ 6 * Real.sqrt 2 :=
by
  intro circle ellipse P Q hP hQ
  sorry

end max_distance_circle_ellipse_l51_51969


namespace unit_prices_minimum_B_seedlings_l51_51878

-- Definition of the problem conditions and the results of Part 1
theorem unit_prices (x : ℝ) : 
  (1200 / (1.5 * x) + 10 = 900 / x) ↔ x = 10 :=
by
  sorry

-- Definition of the problem conditions and the result of Part 2
theorem minimum_B_seedlings (m : ℕ) : 
  (10 * m + 15 * (100 - m) ≤ 1314) ↔ m ≥ 38 :=
by
  sorry

end unit_prices_minimum_B_seedlings_l51_51878


namespace largest_number_value_l51_51652

theorem largest_number_value 
  (a b c : ℚ)
  (h_sum : a + b + c = 100)
  (h_diff1 : c - b = 10)
  (h_diff2 : b - a = 5) : 
  c = 125 / 3 := 
sorry

end largest_number_value_l51_51652


namespace range_of_x_plus_2y_minus_2z_l51_51591

theorem range_of_x_plus_2y_minus_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) : -6 ≤ x + 2 * y - 2 * z ∧ x + 2 * y - 2 * z ≤ 6 :=
sorry

end range_of_x_plus_2y_minus_2z_l51_51591


namespace distance_center_to_plane_of_trapezoid_l51_51625

variable (R α : ℝ)
variable (h1 : 0 < α) (h2 : α < real.pi / 2)

theorem distance_center_to_plane_of_trapezoid {R α : ℝ} (h1 : 0 < α) (h2 : α < real.pi / 2) : 
  let sin_part1 := real.sin (3 * α / 2) in
  let sqrt_part := real.sqrt (real.sin ((3 * α / 2) + (real.pi / 6)) * real.sin ((3 * α / 2) - (real.pi / 6))) in
  O1O2 = (R / sin_part1) * sqrt_part :=
sorry

end distance_center_to_plane_of_trapezoid_l51_51625


namespace large_planter_holds_seeds_l51_51205

theorem large_planter_holds_seeds (total_seeds : ℕ) (small_planter_capacity : ℕ) (num_small_planters : ℕ) (num_large_planters : ℕ) 
  (h1 : total_seeds = 200)
  (h2 : small_planter_capacity = 4)
  (h3 : num_small_planters = 30)
  (h4 : num_large_planters = 4) : 
  (total_seeds - num_small_planters * small_planter_capacity) / num_large_planters = 20 := by
  sorry

end large_planter_holds_seeds_l51_51205


namespace hexagon_monochromatic_triangle_probability_l51_51552

open Classical

-- Define the total number of edges in the hexagon
def total_edges : ℕ := 15

-- Define the number of triangles from 6 vertices
def total_triangles : ℕ := Nat.choose 6 3

-- Define the probability that a given triangle is not monochromatic
def prob_not_monochromatic_triangle : ℚ := 3 / 4

-- Calculate the probability of having at least one monochromatic triangle
def prob_at_least_one_monochromatic_triangle : ℚ := 
  1 - (prob_not_monochromatic_triangle ^ total_triangles)

theorem hexagon_monochromatic_triangle_probability :
  abs ((prob_at_least_one_monochromatic_triangle : ℝ) - 0.9968) < 0.0001 :=
by
  sorry

end hexagon_monochromatic_triangle_probability_l51_51552


namespace three_digit_numbers_divisible_by_13_l51_51143

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l51_51143


namespace find_third_integer_l51_51398

noncomputable def third_odd_integer (x : ℤ) :=
  x + 4

theorem find_third_integer (x : ℤ) (h : 3 * x = 2 * (x + 4) + 3) : third_odd_integer x = 15 :=
by
  sorry

end find_third_integer_l51_51398


namespace necessary_but_not_sufficient_l51_51340

-- Define sets M and N
def M (x : ℝ) : Prop := x < 5
def N (x : ℝ) : Prop := x > 3

-- Define the union and intersection of M and N
def M_union_N (x : ℝ) : Prop := M x ∨ N x
def M_inter_N (x : ℝ) : Prop := M x ∧ N x

-- Theorem statement: Prove the necessity but not sufficiency
theorem necessary_but_not_sufficient (x : ℝ) :
  M_inter_N x → M_union_N x ∧ ¬(M_union_N x → M_inter_N x) := 
sorry

end necessary_but_not_sufficient_l51_51340


namespace eccentricity_of_hyperbola_l51_51812

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  let c := 2 * b
  let e := c / a
  e

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_cond : hyperbola_eccentricity a b h_a h_b = 2 * (b / a)) :
  hyperbola_eccentricity a b h_a h_b = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end eccentricity_of_hyperbola_l51_51812


namespace circle_equation_a_value_l51_51942

theorem circle_equation_a_value (a : ℝ) : (∀ x y : ℝ, x^2 + (a + 2) * y^2 + 2 * a * x + a = 0) → (a = -1) :=
sorry

end circle_equation_a_value_l51_51942


namespace least_froods_l51_51323

theorem least_froods (n : ℕ) :
  (∃ n, n ≥ 1 ∧ (n * (n + 1)) / 2 > 20 * n) → (∃ n, n = 40) :=
by {
  sorry
}

end least_froods_l51_51323


namespace mod_inverse_35_36_l51_51884

theorem mod_inverse_35_36 : ∃ a : ℤ, 0 ≤ a ∧ a < 36 ∧ (35 * a) % 36 = 1 :=
  ⟨35, by sorry⟩

end mod_inverse_35_36_l51_51884


namespace minimize_quadratic_expression_l51_51551

theorem minimize_quadratic_expression:
  ∀ x : ℝ, (∃ a b c : ℝ, a = 1 ∧ b = -8 ∧ c = 15 ∧ x^2 + b * x + c ≥ (4 - 4)^2 - 1) :=
by
  sorry

end minimize_quadratic_expression_l51_51551


namespace line_graph_displays_trend_l51_51274

-- Define the types of statistical graphs
inductive StatisticalGraph : Type
| barGraph : StatisticalGraph
| lineGraph : StatisticalGraph
| pieChart : StatisticalGraph
| histogram : StatisticalGraph

-- Define the property of displaying trends over time
def displaysTrend (g : StatisticalGraph) : Prop := 
  g = StatisticalGraph.lineGraph

-- Theorem to prove that the type of statistical graph that displays the trend of data is the line graph
theorem line_graph_displays_trend : displaysTrend StatisticalGraph.lineGraph :=
sorry

end line_graph_displays_trend_l51_51274


namespace crow_distance_l51_51681

theorem crow_distance (trips: ℕ) (hours: ℝ) (speed: ℝ) (distance: ℝ) :
  trips = 15 → hours = 1.5 → speed = 4 → (trips * 2 * distance) = (speed * hours) → distance = 200 / 1000 :=
by
  intros h_trips h_hours h_speed h_eq
  sorry

end crow_distance_l51_51681


namespace rectangle_area_stage_8_l51_51938

def square_side_length : ℕ := 4
def stage_count : ℕ := 8

-- The function to compute the area of one square
def square_area (side_length: ℕ) : ℕ :=
  side_length * side_length

-- The function to compute the total area at a given stage
def total_area_at_stage (side_length: ℕ) (stages: ℕ) : ℕ :=
  stages * (square_area side_length)

theorem rectangle_area_stage_8 :
  total_area_at_stage square_side_length stage_count = 128 :=
  by
    sorry

end rectangle_area_stage_8_l51_51938


namespace cube_division_l51_51852

theorem cube_division :
  ∃ N, 
    let edge_large := 6 
    let volume_large := edge_large^3;
    volumes = [1, 2, 3, 6]
    N = 164 ∧ 
    volume_large = 2 * 3^3 + 162 * 1^3 :=
begin
  sorry
end

end cube_division_l51_51852


namespace find_first_term_l51_51435

noncomputable def firstTermOfGeometricSeries (S : ℝ) (r : ℝ) : ℝ :=
  S * (1 - r) / (1 - r)

theorem find_first_term
  (S : ℝ)
  (r : ℝ)
  (hS : S = 20)
  (hr : r = -3/7) :
  firstTermOfGeometricSeries S r = 200 / 7 :=
  by
    rw [hS, hr]
    sorry

end find_first_term_l51_51435


namespace emails_left_in_inbox_l51_51809

-- Define the initial conditions and operations
def initial_emails : ℕ := 600

def move_half_to_trash (emails : ℕ) : ℕ := emails / 2
def move_40_percent_to_work (emails : ℕ) : ℕ := emails - (emails * 40 / 100)
def move_25_percent_to_personal (emails : ℕ) : ℕ := emails - (emails * 25 / 100)
def move_10_percent_to_miscellaneous (emails : ℕ) : ℕ := emails - (emails * 10 / 100)
def filter_30_percent_to_subfolders (emails : ℕ) : ℕ := emails - (emails * 30 / 100)
def archive_20_percent (emails : ℕ) : ℕ := emails - (emails * 20 / 100)

-- Statement we need to prove
theorem emails_left_in_inbox : 
  archive_20_percent
    (filter_30_percent_to_subfolders
      (move_10_percent_to_miscellaneous
        (move_25_percent_to_personal
          (move_40_percent_to_work
            (move_half_to_trash initial_emails))))) = 69 := 
by sorry

end emails_left_in_inbox_l51_51809


namespace rows_needed_correct_l51_51555

variable (pencils rows_needed : Nat)

def total_pencils : Nat := 35
def pencils_per_row : Nat := 5
def rows_expected : Nat := 7

theorem rows_needed_correct : rows_needed = total_pencils / pencils_per_row →
  rows_needed = rows_expected := by
  sorry

end rows_needed_correct_l51_51555


namespace triangle_largest_angle_l51_51818

theorem triangle_largest_angle (k : ℕ) 
  (h1 : 3 * k + 4 * k + 5 * k = 180)
  (h2 : ∃ k, 3 * k + 4 * k + 5 * k = 180) :
  5 * k = 75 :=
sorry

end triangle_largest_angle_l51_51818


namespace customOp_eval_l51_51176

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- State the theorem we need to prove
theorem customOp_eval : customOp 4 (-1) = -4 :=
  by
    sorry

end customOp_eval_l51_51176


namespace man_son_work_together_l51_51857

theorem man_son_work_together (man_days : ℝ) (son_days : ℝ) (combined_days : ℝ) :
  man_days = 4 → son_days = 12 → (1 / man_days + 1 / son_days) = 1 / combined_days → combined_days = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end man_son_work_together_l51_51857


namespace count_3digit_numbers_div_by_13_l51_51112

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l51_51112


namespace perimeter_C_l51_51284

def is_square (n : ℕ) : Prop := n > 0 ∧ ∃ s : ℕ, s * s = n

variable (A B C : ℕ) -- Defining the squares
variable (sA sB sC : ℕ) -- Defining the side lengths

-- Conditions as definitions
axiom square_figures : is_square A ∧ is_square B ∧ is_square C 
axiom perimeter_A : 4 * sA = 20
axiom perimeter_B : 4 * sB = 40
axiom side_length_C : sC = 2 * (sA + sB)

-- The equivalent proof problem statement
theorem perimeter_C : 4 * sC = 120 :=
by
  -- Proof will go here
  sorry

end perimeter_C_l51_51284


namespace max_g_f_inequality_l51_51575

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := f x - x / 4 - 1

theorem max_g : ∃ x : ℝ, g x = 2 * Real.log 2 - 7 / 4 :=
sorry

theorem f_inequality (x : ℝ) (hx : 0 < x) : f x < (Real.exp x - 1) / x^2 :=
sorry

end max_g_f_inequality_l51_51575


namespace distance_between_foci_l51_51288

-- Define the ellipse
def ellipse_eq (x y : ℝ) := 9 * x^2 + 36 * y^2 = 1296

-- Define the semi-major and semi-minor axes
def semi_major_axis := 12
def semi_minor_axis := 6

-- Distance between the foci of the ellipse
theorem distance_between_foci : 
  (∃ x y : ℝ, ellipse_eq x y) → 2 * Real.sqrt (semi_major_axis^2 - semi_minor_axis^2) = 12 * Real.sqrt 3 :=
by
  sorry

end distance_between_foci_l51_51288


namespace train_length_l51_51695

theorem train_length (speed_kph : ℕ) (tunnel_length_m : ℕ) (time_s : ℕ) : 
  speed_kph = 54 → 
  tunnel_length_m = 1200 → 
  time_s = 100 → 
  ∃ train_length_m : ℕ, train_length_m = 300 := 
by
  intros h1 h2 h3
  have speed_mps : ℕ := (speed_kph * 1000) / 3600 
  have total_distance_m : ℕ := speed_mps * time_s
  have train_length_m : ℕ := total_distance_m - tunnel_length_m
  use train_length_m
  sorry

end train_length_l51_51695


namespace total_bales_in_barn_l51_51827

-- Definitions based on the conditions 
def initial_bales : ℕ := 47
def added_bales : ℕ := 35

-- Statement to prove the final number of bales in the barn
theorem total_bales_in_barn : initial_bales + added_bales = 82 :=
by
  sorry

end total_bales_in_barn_l51_51827


namespace markese_earned_16_l51_51794

def evan_earnings (E : ℕ) : Prop :=
  (E : ℕ)

def markese_earnings (M : ℕ) (E : ℕ) : Prop :=
  (M : ℕ) = E - 5

def total_earnings (E M : ℕ) : Prop :=
  E + M = 37

theorem markese_earned_16 (E : ℕ) (M : ℕ) 
  (h1 : markese_earnings M E) 
  (h2 : total_earnings E M) : M = 16 :=
sorry

end markese_earned_16_l51_51794


namespace minnie_takes_more_time_l51_51202

def minnie_speed_flat : ℝ := 25
def minnie_speed_downhill : ℝ := 35
def minnie_speed_uphill : ℝ := 10
def penny_speed_flat : ℝ := 35
def penny_speed_downhill : ℝ := 45
def penny_speed_uphill : ℝ := 15

def distance_A_to_B : ℝ := 15
def distance_B_to_D : ℝ := 20
def distance_D_to_C : ℝ := 25

def distance_C_to_B : ℝ := 20
def distance_D_to_A : ℝ := 25

noncomputable def time_minnie : ℝ :=
(distance_A_to_B / minnie_speed_uphill) + 
(distance_B_to_D / minnie_speed_downhill) + 
(distance_D_to_C / minnie_speed_flat)

noncomputable def time_penny : ℝ :=
(distance_C_to_B / penny_speed_uphill) + 
(distance_B_to_D / penny_speed_downhill) + 
(distance_D_to_A / penny_speed_flat)

noncomputable def time_diff : ℝ := (time_minnie - time_penny) * 60

theorem minnie_takes_more_time : time_diff = 10 := by
  sorry

end minnie_takes_more_time_l51_51202


namespace rectangle_area_l51_51024

theorem rectangle_area (P : ℝ) (twice : ℝ → ℝ) (L W A : ℝ) 
  (h1 : P = 40) 
  (h2 : ∀ W, L = twice W) 
  (h3 : ∀ L W, P = 2 * L + 2 * W) 
  (h4 : ∀ L W, A = L * W) 
  (h5 : twice = (λ W, 2 * W)) :
  A = 800 / 9 := 
sorry

end rectangle_area_l51_51024


namespace general_formula_for_a_n_l51_51963

-- Given conditions
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
variable (h1 : ∀ n : ℕ, a n > 0)
variable (h2 : ∀ n : ℕ, 4 * S n = (a n - 1) * (a n + 3))

theorem general_formula_for_a_n :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end general_formula_for_a_n_l51_51963


namespace odd_square_minus_one_div_by_eight_l51_51316

theorem odd_square_minus_one_div_by_eight (n : ℤ) : ∃ k : ℤ, (2 * n + 1) ^ 2 - 1 = 8 * k :=
by
  sorry

end odd_square_minus_one_div_by_eight_l51_51316


namespace river_depth_mid_July_l51_51180

theorem river_depth_mid_July :
  let d_May := 5
  let d_June := d_May + 10
  let d_July := 3 * d_June
  d_July = 45 :=
by
  sorry

end river_depth_mid_July_l51_51180


namespace num_pos_3_digit_div_by_13_l51_51166

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l51_51166


namespace area_of_rectangle_stage_8_l51_51934

theorem area_of_rectangle_stage_8 : 
  (∀ n, 4 * 4 = 16) →
  (∀ k, k ≤ 8 → k = k) →
  (8 * 16 = 128) :=
by
  intros h_sq_area h_sequence
  sorry

end area_of_rectangle_stage_8_l51_51934


namespace range_of_m_for_inequality_l51_51676

theorem range_of_m_for_inequality (x y m : ℝ) :
  (∀ x y : ℝ, 3*x^2 + y^2 ≥ m * x * (x + y)) ↔ (-6 ≤ m ∧ m ≤ 2) := sorry

end range_of_m_for_inequality_l51_51676


namespace slow_population_growth_before_ir_l51_51009

-- Define the conditions
def low_level_social_productivity_before_ir : Prop := sorry
def high_birth_rate_before_ir : Prop := sorry
def high_mortality_rate_before_ir : Prop := sorry

-- The correct answer
def low_natural_population_growth_rate_before_ir : Prop := sorry

-- The theorem to prove
theorem slow_population_growth_before_ir 
  (h1 : low_level_social_productivity_before_ir) 
  (h2 : high_birth_rate_before_ir) 
  (h3 : high_mortality_rate_before_ir) : low_natural_population_growth_rate_before_ir := 
sorry

end slow_population_growth_before_ir_l51_51009


namespace find_ab_l51_51641

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 + b

theorem find_ab (a b : ℝ) (h₀ : a ≠ 0) (h₁ : f a b 2 = 2) (h₂ : f a b 3 = 5) :
    (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 3) :=
by 
  sorry

end find_ab_l51_51641


namespace sum_of_coefficients_l51_51824

theorem sum_of_coefficients (a : ℤ) (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (a + x) * (1 + x) ^ 4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + a_3 + a_5 = 32 →
  a = 3 :=
by sorry

end sum_of_coefficients_l51_51824


namespace digit_for_divisibility_by_45_l51_51874

theorem digit_for_divisibility_by_45 (n : ℕ) (h₀ : n < 10)
  (h₁ : 5 ∣ (5 + 10 * (7 + 4 * (1 + 5 * (8 + n))))) 
  (h₂ : 9 ∣ (5 + 7 + 4 + n + 5 + 8)) : 
  n = 7 :=
by { sorry }

end digit_for_divisibility_by_45_l51_51874


namespace circle_reflection_l51_51358

-- Definitions provided in conditions
def initial_center : ℝ × ℝ := (6, -5)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.snd, p.fst)
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.fst, p.snd)

-- The final statement we need to prove
theorem circle_reflection :
  reflect_y_axis (reflect_y_eq_x initial_center) = (5, 6) :=
by
  -- By reflecting the point (6, -5) over y = x and then over the y-axis, we should get (5, 6)
  sorry

end circle_reflection_l51_51358


namespace older_brother_catches_younger_brother_l51_51826

theorem older_brother_catches_younger_brother
  (y_time_reach_school o_time_reach_school : ℕ) 
  (delay : ℕ) 
  (catchup_time : ℕ) 
  (h1 : y_time_reach_school = 25) 
  (h2 : o_time_reach_school = 15) 
  (h3 : delay = 8) 
  (h4 : catchup_time = 17):
  catchup_time = delay + ((8 * y_time_reach_school) / (o_time_reach_school - y_time_reach_school) * (y_time_reach_school / 25)) :=
by
  sorry

end older_brother_catches_younger_brother_l51_51826


namespace find_m_l51_51594

theorem find_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + x - m > 0 ↔ x < -3 ∨ x > 2) → m = 6 :=
by
  intros h
  sorry

end find_m_l51_51594


namespace unique_involution_l51_51019

noncomputable def f (x : ℤ) : ℤ := sorry

theorem unique_involution (f : ℤ → ℤ) :
  (∀ x : ℤ, f (f x) = x) →
  (∀ x y : ℤ, (x + y) % 2 = 1 → f x + f y ≥ x + y) →
  (∀ x : ℤ, f x = x) :=
sorry

end unique_involution_l51_51019


namespace value_of_fraction_l51_51635

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
variable (h : (4 * x + y) / (x - 4 * y) = -3)

theorem value_of_fraction : (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end value_of_fraction_l51_51635


namespace divisibility_by_30_l51_51460

theorem divisibility_by_30 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_3 : p ≥ 3) : 30 ∣ (p^3 - 1) ↔ p % 15 = 1 := 
  sorry

end divisibility_by_30_l51_51460


namespace find_min_max_value_l51_51289

open Real

theorem find_min_max_value (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) (h_det : b^2 - 4 * a * c < 0) :
  ∃ (min_val max_val: ℝ),
    min_val = (2 * d * sqrt (a * c)) / (b + 2 * sqrt (a * c)) ∧ 
    max_val = (2 * d * sqrt (a * c)) / (b - 2 * sqrt (a * c)) ∧
    (∀ x y : ℝ, a * x^2 + c * y^2 ≥ min_val ∧ a * x^2 + c * y^2 ≤ max_val) :=
by
  -- Proof goes here
  sorry

end find_min_max_value_l51_51289


namespace problem_intersection_l51_51300

theorem problem_intersection (a b : ℝ) 
    (h1 : b = - 2 / a) 
    (h2 : b = a + 3) 
    : 1 / a - 1 / b = -3 / 2 :=
by
  sorry

end problem_intersection_l51_51300


namespace diagonals_in_octadecagon_l51_51243

def num_sides : ℕ := 18

def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_octadecagon : num_diagonals num_sides = 135 := by 
  sorry

end diagonals_in_octadecagon_l51_51243


namespace max_geometric_progression_terms_l51_51661

theorem max_geometric_progression_terms :
  ∀ a0 q : ℕ, (∀ k, a0 * q^k ≥ 100 ∧ a0 * q^k < 1000) →
  (∃ r s : ℕ, r > s ∧ q = r / s) →
  (∀ n, ∃ r s : ℕ, (r^n < 1000) ∧ ((r / s)^n < 10)) →
  n ≤ 5 :=
sorry

end max_geometric_progression_terms_l51_51661


namespace length_PC_in_rectangle_l51_51962

theorem length_PC_in_rectangle (PA PB PD: ℝ) (P_inside: True) 
(h1: PA = 5) (h2: PB = 7) (h3: PD = 3) : PC = Real.sqrt 65 := 
sorry

end length_PC_in_rectangle_l51_51962


namespace malcolm_initial_white_lights_l51_51231

-- Definitions based on the conditions
def red_lights : Nat := 12
def blue_lights : Nat := 3 * red_lights
def green_lights : Nat := 6
def total_colored_lights := red_lights + blue_lights + green_lights
def lights_left_to_buy : Nat := 5
def initially_white_lights := total_colored_lights + lights_left_to_buy

-- Proof statement
theorem malcolm_initial_white_lights : initially_white_lights = 59 := by
  sorry

end malcolm_initial_white_lights_l51_51231


namespace max_pairwise_disjoint_subsets_l51_51783

noncomputable def maxSubsets (n : ℕ) : ℕ :=
  Nat.choose n (n / 2)

theorem max_pairwise_disjoint_subsets (n : ℕ):
  ∀ (A : Finset (Fin n)) (A_subsets : Finset (Finset (Fin n))),
    (∀ X Y ∈ A_subsets, X ≠ Y → X ∩ Y = ∅) →
    A_subset.card = n →
    A_subsets.card ≤ maxSubsets n := sorry

end max_pairwise_disjoint_subsets_l51_51783


namespace count_3_digit_numbers_divisible_by_13_l51_51063

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l51_51063


namespace p_q_r_cubic_sum_l51_51704

theorem p_q_r_cubic_sum (p q r : ℚ) (h1 : p + q + r = 4) (h2 : p * q + p * r + q * r = 6) (h3 : p * q * r = -8) : 
  p^3 + q^3 + r^3 = 8 := by
  sorry

end p_q_r_cubic_sum_l51_51704


namespace evaluate_expression_l51_51279

def ceil (x : ℚ) : ℤ := sorry -- Implement the ceiling function for rational numbers as needed

theorem evaluate_expression :
  (ceil ((23 : ℚ) / 9 - ceil ((35 : ℚ) / 23))) 
  / (ceil ((35 : ℚ) / 9 + ceil ((9 * 23 : ℚ) / 35))) = (1 / 10 : ℚ) :=
by
  intros
  -- Proof goes here
  sorry

end evaluate_expression_l51_51279


namespace animal_arrangement_l51_51218

theorem animal_arrangement :
  let chickens := 3
  let dogs := 3
  let cats := 4 
  let rabbits := 2
  let total_animals := chickens + dogs + cats + rabbits
  let factorial := Nat.factorial
  total_animals = 12 ∧
  (factorial 4 * factorial chickens * factorial dogs * factorial cats * factorial rabbits = 41472) :=
by
  { sorry }

end animal_arrangement_l51_51218


namespace annual_increase_of_chickens_l51_51623

theorem annual_increase_of_chickens 
  (chickens_now : ℕ)
  (chickens_after_9_years : ℕ)
  (years : ℕ)
  (chickens_now_eq : chickens_now = 550)
  (chickens_after_9_years_eq : chickens_after_9_years = 1900)
  (years_eq : years = 9)
  : ((chickens_after_9_years - chickens_now) / years) = 150 :=
by
  sorry

end annual_increase_of_chickens_l51_51623


namespace total_eggs_l51_51455

-- Define the number of eggs eaten in each meal
def breakfast_eggs : ℕ := 2
def lunch_eggs : ℕ := 3
def dinner_eggs : ℕ := 1

-- Prove the total number of eggs eaten is 6
theorem total_eggs : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  sorry

end total_eggs_l51_51455


namespace three_digit_numbers_divisible_by_13_count_l51_51134

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l51_51134


namespace dot_product_EC_ED_l51_51770

-- Define the context of the square and the points E, C, D
def midpoint (A B: ℝ × ℝ): ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem dot_product_EC_ED :
  ∀ (A B D C E: ℝ × ℝ),
    ABCD_is_square A B C D →
    side_length (A B C D) = 2 →
    E = midpoint A B →
    vector_dot_product (vector_range E C) (vector_range E D) = 3 :=
by
  sorry

end dot_product_EC_ED_l51_51770


namespace problem_statement_l51_51618

theorem problem_statement (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
sorry

end problem_statement_l51_51618


namespace prime_numbers_in_list_l51_51057

noncomputable def list_numbers : ℕ → ℕ
| 0       => 43
| (n + 1) => 43 * ((10 ^ (2 * n + 2) - 1) / 99) 

theorem prime_numbers_in_list : ∃ n:ℕ, (∀ m, (m > n) → ¬ Prime (list_numbers m)) ∧ Prime (list_numbers 0) := 
by
  sorry

end prime_numbers_in_list_l51_51057


namespace cost_of_dozen_pens_l51_51524

variable (x : ℝ) (pen_cost pencil_cost : ℝ)
variable (h1 : 3 * pen_cost + 5 * pencil_cost = 260)
variable (h2 : pen_cost / pencil_cost = 5)

theorem cost_of_dozen_pens (x_pos : 0 < x) 
    (pen_cost_def : pen_cost = 5 * x) 
    (pencil_cost_def : pencil_cost = x) :
    12 * pen_cost = 780 := by
  sorry

end cost_of_dozen_pens_l51_51524


namespace red_cards_pick_ordered_count_l51_51683

theorem red_cards_pick_ordered_count :
  let deck_size := 36
  let suits := 3
  let suit_size := 12
  let red_suits := 2
  let red_cards := red_suits * suit_size
  (red_cards * (red_cards - 1) = 552) :=
by
  let deck_size := 36
  let suits := 3
  let suit_size := 12
  let red_suits := 2
  let red_cards := red_suits * suit_size
  show (red_cards * (red_cards - 1) = 552)
  sorry

end red_cards_pick_ordered_count_l51_51683


namespace simplify_expression_l51_51213

theorem simplify_expression (x y : ℝ) : 
  8 * x + 3 * y - 2 * x + y + 20 + 15 = 6 * x + 4 * y + 35 :=
by
  sorry

end simplify_expression_l51_51213


namespace tyrone_gave_marbles_to_eric_l51_51610

theorem tyrone_gave_marbles_to_eric (initial_tyrone_marbles : ℕ) (initial_eric_marbles : ℕ) (marbles_given : ℕ) :
  initial_tyrone_marbles = 150 ∧ initial_eric_marbles = 30 ∧ (initial_tyrone_marbles - marbles_given = 3 * initial_eric_marbles) → marbles_given = 60 :=
by
  sorry

end tyrone_gave_marbles_to_eric_l51_51610


namespace necessary_but_not_sufficient_l51_51224

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 + 2 * x - 8 > 0) ↔ (x > 2) ∨ (x < -4) := by
sorry

end necessary_but_not_sufficient_l51_51224


namespace largest_prime_factor_1729_l51_51993

theorem largest_prime_factor_1729 : ∃ p, prime p ∧ p ∣ 1729 ∧ ∀ q, prime q ∧ q ∣ 1729 → q ≤ p := by
  let n := 1729
  have h1 : prime 7 := by sorry
  have h2 : n = 7 * 247 := by sorry
  have h3 : prime 13 := by sorry
  have h4 : 247 = 13 * 19 := by sorry
  have h5 : prime 19 := by sorry
  have h6 : n = 7 * (13 * 19) := by
    rw [h2, h4]
    ring
  use 19
  split
  exact h5
  split
  rw [h6]
  exact dvd_mul_of_dvd_right (dvd_mul_right 13 19) 7
  intros q pk q_div
  by_contra
  rw [h6, prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h1 prime.not_associated_of_lt pk (dvd.trans q_div h2) (dvd_trans q_div (dvd_mul_right (13 * 19) 7)) h_1
  rw[prime.dvd_mul pk] at q_div
  cases q_div
  exact not_prime_of_dvd_prime h3 (prime.not_associated_of_lt pk) (dvd.trans q_div h4) (dvd.trans q_div (dvd_mul_right 19 13)) h_1
  rw [←dvd_prime_iff_dvd_of_prime h5 pk] at q_div
  exact q_div

end largest_prime_factor_1729_l51_51993


namespace num_three_digit_div_by_13_l51_51160

theorem num_three_digit_div_by_13 : 
  ∃ n : ℕ, n = 68 ∧ 
  let a := 117 in 
  let d := 13 in 
  let l := 988 in 
  (∀ k : ℕ, (k ≥ 1 ∧ k ≤ n) → (k = n → (a + (n-1) * d = l))):
  ∃ n : ℕ, 
  (117 + (n-1) * 13 = 988) ∧ n = 68 :=
sorry

end num_three_digit_div_by_13_l51_51160


namespace count_3_digit_numbers_divisible_by_13_l51_51064

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51064


namespace divisors_of_m_squared_l51_51859

theorem divisors_of_m_squared {m : ℕ} (h₁ : ∀ d, d ∣ m → d = 1 ∨ d = m ∨ prime d) (h₂ : nat.divisors m = 4) :
  (nat.divisors (m ^ 2) = 7 ∨ nat.divisors (m ^ 2) = 9) :=
sorry

end divisors_of_m_squared_l51_51859


namespace total_cost_l51_51359

def cost(M R F : ℝ) := 10 * M = 24 * R ∧ 6 * F = 2 * R ∧ F = 23

theorem total_cost (M R F : ℝ) (h : cost M R F) : 
  4 * M + 3 * R + 5 * F = 984.40 :=
by
  sorry

end total_cost_l51_51359


namespace count_3digit_numbers_div_by_13_l51_51117

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l51_51117


namespace least_four_digit_9_heavy_l51_51413

def is_9_heavy (n : ℕ) : Prop := n % 9 > 5

def four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

theorem least_four_digit_9_heavy : ∃ n, four_digit n ∧ is_9_heavy n ∧ ∀ m, (four_digit m ∧ is_9_heavy m) → n ≤ m :=
by
  exists 1005
  sorry

end least_four_digit_9_heavy_l51_51413


namespace proof_problem_l51_51735

noncomputable def A := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
noncomputable def B := {(x, y) : ℝ × ℝ | y = x^2 + 1}

theorem proof_problem :
  ((1, 2) ∈ B) ∧
  (0 ∉ A) ∧
  ((0, 0) ∉ B) :=
by
  sorry

end proof_problem_l51_51735


namespace days_before_reinforcement_l51_51854

theorem days_before_reinforcement
    (garrison_1 : ℕ)
    (initial_days : ℕ)
    (reinforcement : ℕ)
    (additional_days : ℕ)
    (total_men_after_reinforcement : ℕ)
    (man_days_initial : ℕ)
    (man_days_after : ℕ)
    (x : ℕ) :
    garrison_1 * (initial_days - x) = total_men_after_reinforcement * additional_days →
    garrison_1 = 2000 →
    initial_days = 54 →
    reinforcement = 1600 →
    additional_days = 20 →
    total_men_after_reinforcement = garrison_1 + reinforcement →
    man_days_initial = garrison_1 * initial_days →
    man_days_after = total_men_after_reinforcement * additional_days →
    x = 18 :=
by
  intros h_eq g_1 i_days r_f a_days total_men m_days_i m_days_a
  sorry

end days_before_reinforcement_l51_51854


namespace count_3_digit_numbers_divisible_by_13_l51_51059

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l51_51059


namespace power_problem_l51_51748

theorem power_problem (k : ℕ) (h : 6 ^ k = 4) : 6 ^ (2 * k + 3) = 3456 := 
by 
  sorry

end power_problem_l51_51748


namespace min_value_quadratic_l51_51387

theorem min_value_quadratic (x : ℝ) : ∃ x, x = -7 ∧ (x^2 + 14 * x + 24 = -25) := sorry

end min_value_quadratic_l51_51387


namespace number_of_3_digit_divisible_by_13_l51_51073

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l51_51073


namespace sum_of_two_numbers_l51_51237

theorem sum_of_two_numbers (a b : ℝ) (h1 : a * b = 16) (h2 : (1 / a) = 3 * (1 / b)) (ha : 0 < a) (hb : 0 < b) :
  a + b = 16 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end sum_of_two_numbers_l51_51237


namespace koschei_coins_l51_51327

theorem koschei_coins :
  ∃ a : ℕ, a % 10 = 7 ∧ a % 12 = 9 ∧ 300 ≤ a ∧ a ≤ 400 ∧ a = 357 :=
by
  sorry

end koschei_coins_l51_51327


namespace area_at_stage_8_l51_51936

theorem area_at_stage_8 
  (side_length : ℕ)
  (stage : ℕ)
  (num_squares : ℕ)
  (square_area : ℕ) 
  (total_area : ℕ) 
  (h1 : side_length = 4) 
  (h2 : stage = 8) 
  (h3 : num_squares = stage) 
  (h4 : square_area = side_length * side_length) 
  (h5 : total_area = num_squares * square_area) :
  total_area = 128 :=
sorry

end area_at_stage_8_l51_51936


namespace find_some_number_l51_51466

def some_number (x : Int) (some_num : Int) : Prop :=
  (3 < x ∧ x < 10) ∧
  (5 < x ∧ x < 18) ∧
  (9 > x ∧ x > -2) ∧
  (8 > x ∧ x > 0) ∧
  (x + some_num < 9)

theorem find_some_number :
  ∀ (some_num : Int), some_number 7 some_num → some_num < 2 :=
by
  intros some_num H
  sorry

end find_some_number_l51_51466


namespace smallest_x_for_max_f_l51_51276

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 12)

theorem smallest_x_for_max_f : ∃ x > 0, f x = 2 ∧ ∀ y > 0, (f y = 2 → y ≥ x) :=
sorry

end smallest_x_for_max_f_l51_51276


namespace count_3_digit_numbers_divisible_by_13_l51_51062

theorem count_3_digit_numbers_divisible_by_13 : 
  (∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) → (13 * 8 ≤ k ∧ k ≤ 13 * 76))) :=
begin
  use 69,
  split,
  { refl, },
  { 
    intros k hk,
    have h1 : 13 * 8 = 104, by norm_num,
    have h2 : 13 * 76 = 988, by norm_num,
    rw [h1, h2],
    exact ⟨hk.1.trans (le_refl 104), hk.2.right.trans (le_refl 988)⟩,
  }
end

end count_3_digit_numbers_divisible_by_13_l51_51062


namespace total_score_is_938_l51_51178

-- Define the average score condition
def average_score (S : ℤ) : Prop := 85.25 ≤ (S : ℚ) / 11 ∧ (S : ℚ) / 11 < 85.35

-- Define the condition that each student's score is an integer
def total_score (S : ℤ) : Prop := average_score S ∧ ∃ n : ℕ, S = n

-- Lean 4 statement for the proof problem
theorem total_score_is_938 : ∃ S : ℤ, total_score S ∧ S = 938 :=
by
  sorry

end total_score_is_938_l51_51178


namespace lottery_probability_correct_l51_51643

noncomputable def probability_winning_lottery : ℚ :=
  let starBall_probability := 1 / 30
  let combinations (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  let magicBalls_probability := 1 / (combinations 49 6)
  starBall_probability * magicBalls_probability

theorem lottery_probability_correct :
  probability_winning_lottery = 1 / 419514480 := by
  sorry

end lottery_probability_correct_l51_51643


namespace count_3digit_numbers_div_by_13_l51_51114

-- Definition of the smallest 3-digit number divisible by 13
def smallest_div_by_13 : ℕ := 104

-- Definition of the largest 3-digit number divisible by 13
def largest_div_by_13 : ℕ := 988

-- Definition of the arithmetic sequence
def arithmetic_seq_count (a l d : ℕ) : ℕ := (l - a + d) / d

-- Theorem statement
theorem count_3digit_numbers_div_by_13 : arithmetic_seq_count 8 76 1 = 69 :=
by
  sorry

end count_3digit_numbers_div_by_13_l51_51114


namespace arithmetic_sequence_sum_first_nine_terms_l51_51315

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (d : ℤ)

-- The sequence {a_n} is an arithmetic sequence.
def arithmetic_sequence := ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- The sum of the first n terms of the sequence.
def sum_first_n_terms := ∀ n : ℕ, S_n n = (n * (a_n 1 + a_n n)) / 2

-- Given condition: a_2 = 3 * a_4 - 6
def given_condition := a_n 2 = 3 * a_n 4 - 6

-- The main theorem to prove S_9 = 27
theorem arithmetic_sequence_sum_first_nine_terms (h_arith : arithmetic_sequence a_n d) (h_sum : sum_first_n_terms a_n S_n) (h_condition : given_condition a_n) : 
  S_n 9 = 27 := 
by
  sorry

end arithmetic_sequence_sum_first_nine_terms_l51_51315


namespace correct_grammatical_phrase_l51_51003

-- Define the conditions as lean definitions 
def number_of_cars_produced_previous_year : ℕ := sorry  -- number of cars produced in previous year
def number_of_cars_produced_2004 : ℕ := 3 * number_of_cars_produced_previous_year  -- number of cars produced in 2004

-- Define the theorem stating the correct phrase to describe the production numbers
theorem correct_grammatical_phrase : 
  (3 * number_of_cars_produced_previous_year = number_of_cars_produced_2004) → 
  ("three times as many cars" = "three times as many cars") := 
by
  sorry

end correct_grammatical_phrase_l51_51003


namespace positive_3_digit_numbers_divisible_by_13_count_l51_51079

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l51_51079


namespace jamie_nickels_l51_51956

theorem jamie_nickels (x : ℕ) (hx : 5 * x + 10 * x + 25 * x = 1320) : x = 33 :=
sorry

end jamie_nickels_l51_51956


namespace mod_inverse_17_1200_l51_51662

theorem mod_inverse_17_1200 : ∃ x : ℕ, x < 1200 ∧ 17 * x % 1200 = 1 := 
by
  use 353
  sorry

end mod_inverse_17_1200_l51_51662


namespace polynomial_perfect_square_l51_51944

theorem polynomial_perfect_square (k : ℤ) : (∃ b : ℤ, (x + b)^2 = x^2 + 8 * x + k) -> k = 16 := by
  sorry

end polynomial_perfect_square_l51_51944


namespace positional_relationship_l51_51590

-- Defining the concepts of parallelism, containment, and positional relationships
structure Line -- subtype for a Line
structure Plane -- subtype for a Plane

-- Definitions and Conditions
def is_parallel_to (l : Line) (p : Plane) : Prop := sorry  -- A line being parallel to a plane
def is_contained_in (l : Line) (p : Plane) : Prop := sorry  -- A line being contained within a plane
def are_skew (l₁ l₂ : Line) : Prop := sorry  -- Two lines being skew
def are_parallel (l₁ l₂ : Line) : Prop := sorry  -- Two lines being parallel

-- Given conditions
variables (a b : Line) (α : Plane)
axiom Ha : is_parallel_to a α
axiom Hb : is_contained_in b α

-- The theorem to be proved
theorem positional_relationship (a b : Line) (α : Plane) 
  (Ha : is_parallel_to a α) 
  (Hb : is_contained_in b α) : 
  (are_skew a b ∨ are_parallel a b) :=
sorry

end positional_relationship_l51_51590


namespace count_ways_to_exhaust_black_matches_l51_51861

theorem count_ways_to_exhaust_black_matches 
  (n r g : ℕ) 
  (h_r_le_n : r ≤ n) 
  (h_g_le_n : g ≤ n) 
  (h_r_ge_0 : 0 ≤ r) 
  (h_g_ge_0 : 0 ≤ g) 
  (h_n_ge_0 : 0 < n) :
  ∃ ways : ℕ, ways = (Nat.factorial (3 * n - r - g - 1)) / (Nat.factorial (n - 1) * Nat.factorial (n - r) * Nat.factorial (n - g)) :=
by
  sorry

end count_ways_to_exhaust_black_matches_l51_51861


namespace count_3_digit_numbers_divisible_by_13_l51_51136

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51136


namespace increasing_order_magnitudes_l51_51453

variable (x : ℝ)

noncomputable def y := x^x
noncomputable def z := x^(x^x)

theorem increasing_order_magnitudes (h1 : 1 < x) (h2 : x < 1.1) : x < y x ∧ y x < z x :=
by
  have h3 : y x = x^x := rfl
  have h4 : z x = x^(x^x) := rfl
  sorry

end increasing_order_magnitudes_l51_51453


namespace bajazet_winning_strategy_l51_51415

-- Define the polynomial P with place holder coefficients a, b, c (assuming they are real numbers)
def P (a b c : ℝ) (x : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + 1

-- The statement that regardless of how Alcina plays, Bajazet can ensure that P has a real root.
theorem bajazet_winning_strategy :
  ∃ (a b c : ℝ), ∃ (x : ℝ), P a b c x = 0 :=
sorry

end bajazet_winning_strategy_l51_51415


namespace union_complement_eq_l51_51454

open Set

-- Condition definitions
def U : Set ℝ := univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Theorem statement (what we want to prove)
theorem union_complement_eq :
  A ∪ compl B = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
by
  sorry

end union_complement_eq_l51_51454


namespace largest_among_abc_l51_51307

variable {a b c : ℝ}

theorem largest_among_abc 
  (hn1 : a < 0) 
  (hn2 : b < 0) 
  (hn3 : c < 0) 
  (h : (c / (a + b)) < (a / (b + c)) ∧ (a / (b + c)) < (b / (c + a))) : c > a ∧ c > b :=
by
  sorry

end largest_among_abc_l51_51307


namespace diagonals_sum_pentagon_inscribed_in_circle_l51_51960

theorem diagonals_sum_pentagon_inscribed_in_circle
  (FG HI GH IJ FJ : ℝ)
  (h1 : FG = 4)
  (h2 : HI = 4)
  (h3 : GH = 11)
  (h4 : IJ = 11)
  (h5 : FJ = 15) :
  3 * FJ + (FJ^2 - 121) / 4 + (FJ^2 - 16) / 11 = 80 := by {
  sorry
}

end diagonals_sum_pentagon_inscribed_in_circle_l51_51960


namespace tonya_needs_to_eat_more_l51_51374

-- Define the conditions in the problem
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Define a function to calculate hamburgers given ounces
def hamburgers_eaten (ounces : ℕ) (ounce_per_hamburger : ℕ) : ℕ :=
  ounces / ounce_per_hamburger

-- State the theorem
theorem tonya_needs_to_eat_more (ounces_per_hamburger ounces_eaten_last_year : ℕ) :
  hamburgers_eaten ounces_eaten_last_year ounces_per_hamburger + 1 = 22 := by
  sorry

end tonya_needs_to_eat_more_l51_51374


namespace count_3_digit_numbers_divisible_by_13_l51_51065

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51065


namespace shiela_neighbors_l51_51211

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) (neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : drawings_per_neighbor = 9) : neighbors = total_drawings / drawings_per_neighbor :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end shiela_neighbors_l51_51211


namespace value_of_trig_expr_l51_51653

theorem value_of_trig_expr : 2 * Real.cos (Real.pi / 12) ^ 2 + 1 = 2 + Real.sqrt 3 / 2 :=
by
  sorry

end value_of_trig_expr_l51_51653


namespace geometric_sequence_a1_cannot_be_2_l51_51732

theorem geometric_sequence_a1_cannot_be_2
  (a : ℕ → ℕ)
  (q : ℕ)
  (h1 : 2 * a 2 + a 3 = a 4)
  (h2 : (a 2 + 1) * (a 3 + 1) = a 5 - 1)
  (h3 : ∀ n, a (n + 1) = a n * q) :
  a 1 ≠ 2 :=
by sorry

end geometric_sequence_a1_cannot_be_2_l51_51732


namespace sum_nonnegative_reals_l51_51175

variable {x y z : ℝ}

theorem sum_nonnegative_reals (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 24) : 
  x + y + z = 10 := 
by sorry

end sum_nonnegative_reals_l51_51175


namespace find_largest_number_l51_51649

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
sorry

end find_largest_number_l51_51649


namespace christmas_bonus_remainder_l51_51840

theorem christmas_bonus_remainder (X : ℕ) (h : X % 5 = 2) : (3 * X) % 5 = 1 :=
by
  sorry

end christmas_bonus_remainder_l51_51840


namespace calculate_expression_l51_51272

theorem calculate_expression : |1 - Real.sqrt 3| - (Real.sqrt 3 - 1)^0 = Real.sqrt 3 - 2 := by
  sorry

end calculate_expression_l51_51272


namespace markese_earnings_l51_51795

-- Define the conditions
def earnings_relation (E M : ℕ) : Prop :=
  M = E - 5 ∧ M + E = 37

-- The theorem to prove
theorem markese_earnings (E M : ℕ) (h : earnings_relation E M) : M = 16 :=
by
  sorry

end markese_earnings_l51_51795


namespace certain_number_l51_51589

theorem certain_number (x : ℝ) (h : 7125 / x = 5700) : x = 1.25 := 
sorry

end certain_number_l51_51589


namespace points_per_round_l51_51177

-- Definitions based on conditions
def final_points (jane_points : ℕ) : Prop := jane_points = 60
def lost_points (jane_lost : ℕ) : Prop := jane_lost = 20
def rounds_played (jane_rounds : ℕ) : Prop := jane_rounds = 8

-- The theorem we want to prove
theorem points_per_round (jane_points jane_lost jane_rounds points_per_round : ℕ) 
  (h1 : final_points jane_points) 
  (h2 : lost_points jane_lost) 
  (h3 : rounds_played jane_rounds) : 
  points_per_round = ((jane_points + jane_lost) / jane_rounds) := 
sorry

end points_per_round_l51_51177


namespace area_enclosed_by_sin_l51_51636

/-- The area of the figure enclosed by the curve y = sin(x), the lines x = -π/3, x = π/2, and the x-axis is 3/2. -/
theorem area_enclosed_by_sin (x y : ℝ) (h : y = Real.sin x) (a b : ℝ) 
(h1 : a = -Real.pi / 3) (h2 : b = Real.pi / 2) :
  ∫ x in a..b, |Real.sin x| = 3 / 2 := 
sorry

end area_enclosed_by_sin_l51_51636


namespace quadratic_equation_with_roots_sum_and_difference_l51_51886

theorem quadratic_equation_with_roots_sum_and_difference (p q : ℚ)
  (h1 : p + q = 10)
  (h2 : abs (p - q) = 2) :
  (Polynomial.eval₂ (RingHom.id ℚ) p (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C 24) = 0) ∧
  (Polynomial.eval₂ (RingHom.id ℚ) q (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C 24) = 0) :=
by sorry

end quadratic_equation_with_roots_sum_and_difference_l51_51886


namespace count_3_digit_numbers_divisible_by_13_l51_51138

def smallest_3_digit_divisible_by_13 : ℕ := 104
def largest_3_digit_divisible_by_13 : ℕ := 988

theorem count_3_digit_numbers_divisible_by_13 :
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / 13 + 1 = 69 :=
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51138


namespace rectangle_area_stage_8_l51_51937

def square_side_length : ℕ := 4
def stage_count : ℕ := 8

-- The function to compute the area of one square
def square_area (side_length: ℕ) : ℕ :=
  side_length * side_length

-- The function to compute the total area at a given stage
def total_area_at_stage (side_length: ℕ) (stages: ℕ) : ℕ :=
  stages * (square_area side_length)

theorem rectangle_area_stage_8 :
  total_area_at_stage square_side_length stage_count = 128 :=
  by
    sorry

end rectangle_area_stage_8_l51_51937


namespace koschei_coins_l51_51328

theorem koschei_coins :
  ∃ a : ℕ, a % 10 = 7 ∧ a % 12 = 9 ∧ 300 ≤ a ∧ a ≤ 400 ∧ a = 357 :=
by
  sorry

end koschei_coins_l51_51328


namespace amount_paid_correct_l51_51216

-- Defining the conditions and constants
def hourly_rate : ℕ := 60
def hours_per_day : ℕ := 3
def total_days : ℕ := 14

-- The proof statement
theorem amount_paid_correct : hourly_rate * hours_per_day * total_days = 2520 := by
  sorry

end amount_paid_correct_l51_51216


namespace num_3_digit_div_by_13_l51_51155

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l51_51155


namespace rectangle_perimeter_l51_51263

theorem rectangle_perimeter (breadth length : ℝ) (h1 : length = 3 * breadth) (h2 : length * breadth = 147) : 2 * length + 2 * breadth = 56 :=
by
  sorry

end rectangle_perimeter_l51_51263


namespace quadratic_root_condition_l51_51596

theorem quadratic_root_condition (k : ℝ) :
  (∀ (x : ℝ), x^2 + k * x + 4 * k^2 - 3 = 0 → ∃ x1 x2 : ℝ, x1 + x2 = (-k) ∧ x1 * x2 = 4 * k^2 - 3 ∧ x1 + x2 = x1 * x2) →
  k = 3 / 4 :=
by
  sorry

end quadratic_root_condition_l51_51596


namespace average_minutes_correct_l51_51701

noncomputable def average_minutes_run_per_day : ℚ :=
  let f (fifth_graders : ℕ) : ℚ := (48 * (4 * fifth_graders) + 30 * (2 * fifth_graders) + 10 * fifth_graders) / (4 * fifth_graders + 2 * fifth_graders + fifth_graders)
  f 1

theorem average_minutes_correct :
  average_minutes_run_per_day = 88 / 7 :=
by
  sorry

end average_minutes_correct_l51_51701


namespace find_height_of_cuboid_l51_51655

-- Definitions and given conditions
def length : ℕ := 22
def width : ℕ := 30
def total_edges : ℕ := 224

-- Proof statement
theorem find_height_of_cuboid (h : ℕ) (H : 4 * length + 4 * width + 4 * h = total_edges) : h = 4 :=
by
  sorry

end find_height_of_cuboid_l51_51655


namespace cubic_identity_l51_51749

variable (a b c : ℝ)
variable (h1 : a + b + c = 13)
variable (h2 : ab + ac + bc = 30)

theorem cubic_identity : a^3 + b^3 + c^3 - 3 * a * b * c = 1027 :=
by 
  sorry

end cubic_identity_l51_51749


namespace simplify_expression_l51_51497

theorem simplify_expression (x : ℝ) : (3 * x)^4 + 3 * x * x^3 + 2 * x^5 = 84 * x^4 + 2 * x^5 := by
    sorry

end simplify_expression_l51_51497


namespace arithmetic_mean_of_sixty_integers_starting_from_3_l51_51270

def arithmetic_mean_of_sequence (a d n : ℕ) : ℚ :=
  let a_n := a + (n - 1) * d
  let S_n := n * (a + a_n) / 2
  S_n / n

theorem arithmetic_mean_of_sixty_integers_starting_from_3 : arithmetic_mean_of_sequence 3 1 60 = 32.5 :=
by 
  sorry

end arithmetic_mean_of_sixty_integers_starting_from_3_l51_51270


namespace speeds_correct_l51_51726

-- Definitions for conditions
def distance (A B : Type) := 40 -- given distance between A and B is 40 km
def start_time_pedestrian : Real := 4 -- pedestrian starts at 4:00 AM
def start_time_cyclist : Real := 7 + (20 / 60) -- cyclist starts at 7:20 AM
def midpoint_distance : Real := 20 -- the midpoint distance where cyclist catches up with pedestrian is 20 km

noncomputable def speeds (x y : Real) : Prop :=
  let t_catch_up := (20 - (10 / 3) * x) / (y - x) in -- time taken by the cyclist to catch up
  let t_total := (10 / 3) + t_catch_up + 1 in -- total time for pedestrian until meeting second cyclist
  4.5 = t_total ∧ -- total time in hours from 4:00 AM to 8:30 AM
  10 * x * (y - x) + 60 * x - 10 * x^2 = 60 * y - 60 * x ∧ -- initial condition simplification step
  y = 6 * x -- relationship between speeds based on derived equations

-- The proposition to prove
theorem speeds_correct : ∃ x y : Real, speeds x y ∧ x = 5 ∧ y = 30 :=
by
  sorry

end speeds_correct_l51_51726


namespace geom_sequence_50th_term_l51_51606

theorem geom_sequence_50th_term (a a_2 : ℤ) (n : ℕ) (r : ℤ) (h1 : a = 8) (h2 : a_2 = -16) (h3 : r = a_2 / a) (h4 : n = 50) :
  a * r^(n-1) = -8 * 2^49 :=
by
  sorry

end geom_sequence_50th_term_l51_51606


namespace vector_addition_correct_dot_product_correct_l51_51744

-- Define the two vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

-- Define the expected results
def a_plus_b_expected : ℝ × ℝ := (4, 3)
def a_dot_b_expected : ℝ := 5

-- Prove the sum of vectors a and b
theorem vector_addition_correct : a + b = a_plus_b_expected := by
  sorry

-- Prove the dot product of vectors a and b
theorem dot_product_correct : a.1 * b.1 + a.2 * b.2 = a_dot_b_expected := by
  sorry

end vector_addition_correct_dot_product_correct_l51_51744


namespace num_factors_180_l51_51928

-- Conditions: The prime factorization of 180
def fact180 : ℕ := 180
def fact180_prime_decomp : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Definition of counting the number of factors from prime factorization
def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc p => acc * (p.snd + 1)) 1

-- Theorem statement: The number of positive factors of 180 is 18 
theorem num_factors_180 : number_of_factors fact180_prime_decomp = 18 := 
by
  sorry

end num_factors_180_l51_51928


namespace ratio_of_c_and_d_l51_51433

theorem ratio_of_c_and_d (x y c d : ℝ) (hd : d ≠ 0) 
  (h1 : 3 * x + 2 * y = c) 
  (h2 : 4 * y - 6 * x = d) : c / d = -1 / 3 := 
sorry

end ratio_of_c_and_d_l51_51433


namespace juice_spilled_l51_51255

def initial_amount := 1.0
def Youngin_drank := 0.1
def Narin_drank := Youngin_drank + 0.2
def remaining_amount := 0.3

theorem juice_spilled :
  initial_amount - (Youngin_drank + Narin_drank) - remaining_amount = 0.3 :=
by
  sorry

end juice_spilled_l51_51255


namespace inequality_ineqs_l51_51577

theorem inequality_ineqs (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_cond : x * y + y * z + z * x = 1) :
  (27 / 4) * (x + y) * (y + z) * (z + x) 
  ≥ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2
  ∧ 
  (Real.sqrt (x + y) + Real.sqrt (y + z) + Real.sqrt (z + x)) ^ 2 
  ≥ 
  6 * Real.sqrt 3 := by
  sorry

end inequality_ineqs_l51_51577


namespace incorrect_conclusions_l51_51626

theorem incorrect_conclusions :
  let p := (∀ x y : ℝ, x * y ≠ 6 → x ≠ 2 ∨ y ≠ 3)
  let q := (2, 1) ∈ { p : ℝ × ℝ | p.2 = 2 * p.1 - 3 }
  (p ∨ ¬q) = false ∧ (¬p ∨ q) = false ∧ (p ∧ ¬q) = false :=
by
  sorry

end incorrect_conclusions_l51_51626


namespace min_occupied_seats_l51_51267

theorem min_occupied_seats (n : ℕ) (h_n : n = 150) : 
  ∃ k : ℕ, k = 37 ∧ ∀ (occupied : Finset ℕ), 
    occupied.card < k → ∃ i : ℕ, i ∉ occupied ∧ ∀ j : ℕ, j ∈ occupied → j + 1 ≠ i ∧ j - 1 ≠ i :=
by
  sorry

end min_occupied_seats_l51_51267


namespace charles_remaining_skittles_l51_51273

def c : ℕ := 25
def d : ℕ := 7
def remaining_skittles : ℕ := c - d

theorem charles_remaining_skittles : remaining_skittles = 18 := by
  sorry

end charles_remaining_skittles_l51_51273


namespace max_value_g_f_less_than_e_x_div_x_sq_l51_51574

noncomputable def f (x : ℝ) := Real.log (x + 1)
noncomputable def g (x : ℝ) := f x - x / 4 - 1

theorem max_value_g : ∃ x, x = 3 ∧ g x = 2 * Real.log 2 - 7 / 4 := by
  sorry

theorem f_less_than_e_x_div_x_sq (x : ℝ) (hx : x > 0) : f x < (Real.exp x - 1) / x ^ 2 := by
  sorry

end max_value_g_f_less_than_e_x_div_x_sq_l51_51574


namespace koschei_coin_count_l51_51334

theorem koschei_coin_count (a : ℕ) :
  (a % 10 = 7) ∧
  (a % 12 = 9) ∧
  (300 ≤ a ∧ a ≤ 400) →
  a = 357 :=
sorry

end koschei_coin_count_l51_51334


namespace maximize_profit_l51_51405

variables (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ)

-- Definitions for the conditions
def nonneg_x := (0 ≤ x)
def nonneg_y := (0 ≤ y)
def constraint1 := (a1 * x + a2 * y ≤ c1)
def constraint2 := (b1 * x + b2 * y ≤ c2)
def profit := (z = d1 * x + d2 * y)

-- Proof of constraints and profit condition
theorem maximize_profit (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ) :
    nonneg_x x ∧ nonneg_y y ∧ constraint1 a1 a2 c1 x y ∧ constraint2 b1 b2 c2 x y → profit d1 d2 x y z :=
by
  sorry

end maximize_profit_l51_51405


namespace line_intersects_circle_l51_51733

theorem line_intersects_circle
  (a b r : ℝ)
  (r_nonzero : r ≠ 0)
  (h_outside : a^2 + b^2 > r^2) :
  ∃ x y : ℝ, (x^2 + y^2 = r^2) ∧ (a * x + b * y = r^2) :=
sorry

end line_intersects_circle_l51_51733


namespace three_digit_numbers_divisible_by_13_l51_51144

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l51_51144


namespace strategy_for_antonio_l51_51266

-- We define the concept of 'winning' and 'losing' positions
def winning_position (m n : ℕ) : Prop :=
  ¬ (m % 2 = 0 ∧ n % 2 = 0)

-- Now create the main theorem
theorem strategy_for_antonio (m n : ℕ) : winning_position m n ↔ 
  (¬(m % 2 = 0 ∧ n % 2 = 0)) :=
by
  unfold winning_position
  sorry

end strategy_for_antonio_l51_51266


namespace net_profit_expression_and_break_even_point_l51_51753

-- Definitions based on the conditions in a)
def investment : ℝ := 600000
def initial_expense : ℝ := 80000
def expense_increase : ℝ := 20000
def annual_income : ℝ := 260000

-- Define the net profit function as given in the solution
def net_profit (n : ℕ) : ℝ :=
  - (n : ℝ)^2 + 19 * n - 60

-- Statement about the function and where the dealer starts making profit
theorem net_profit_expression_and_break_even_point :
  net_profit n = - (n : ℝ)^2 + 19 * n - 60 ∧ ∃ n ≥ 5, net_profit n > 0 :=
sorry

end net_profit_expression_and_break_even_point_l51_51753


namespace num_factors_180_l51_51925

theorem num_factors_180 : 
  ∃ (n : ℕ), n = 180 ∧ prime_factorization n = [(2, 2), (3, 2), (5, 1)] ∧ positive_factors n = 18 :=
by
  sorry

end num_factors_180_l51_51925


namespace solve_for_x_l51_51632

variable (x : ℝ)

theorem solve_for_x (h : 5 * x - 3 = 17) : x = 4 := sorry

end solve_for_x_l51_51632


namespace speed_of_sound_l51_51858

theorem speed_of_sound (time_heard : ℕ) (time_occured : ℕ) (distance : ℝ) : 
  time_heard = 30 * 60 + 20 → 
  time_occured = 30 * 60 → 
  distance = 6600 → 
  (distance / ((time_heard - time_occured) / 3600)) / 3600 = 330 :=
by 
  intros h1 h2 h3
  sorry

end speed_of_sound_l51_51858


namespace bob_paid_24_percent_of_SRP_l51_51233

theorem bob_paid_24_percent_of_SRP
  (P : ℝ) -- Suggested Retail Price (SRP)
  (MP : ℝ) -- Marked Price (MP)
  (price_bob_paid : ℝ) -- Price Bob Paid
  (h1 : MP = 0.60 * P) -- Condition 1: MP is 60% of SRP
  (h2 : price_bob_paid = 0.40 * MP) -- Condition 2: Bob paid 40% of the MP
  : (price_bob_paid / P) * 100 = 24 := -- Bob paid 24% of the SRP
by
  sorry

end bob_paid_24_percent_of_SRP_l51_51233


namespace total_distance_walked_l51_51981

-- Define the conditions
def walking_rate : ℝ := 4
def time_before_break : ℝ := 2
def time_after_break : ℝ := 0.5

-- Define the required theorem
theorem total_distance_walked : 
  walking_rate * time_before_break + walking_rate * time_after_break = 10 := 
sorry

end total_distance_walked_l51_51981


namespace markese_earnings_l51_51796

-- Define the conditions
def earnings_relation (E M : ℕ) : Prop :=
  M = E - 5 ∧ M + E = 37

-- The theorem to prove
theorem markese_earnings (E M : ℕ) (h : earnings_relation E M) : M = 16 :=
by
  sorry

end markese_earnings_l51_51796


namespace smallest_nat_satisfies_conditions_l51_51692

theorem smallest_nat_satisfies_conditions : 
  ∃ x : ℕ, (∃ m : ℤ, x + 13 = 5 * m) ∧ (∃ n : ℤ, x - 13 = 6 * n) ∧ x = 37 := by
  sorry

end smallest_nat_satisfies_conditions_l51_51692


namespace water_consumption_comparison_l51_51021

-- Define the given conditions
def waterConsumptionWest : ℝ := 21428
def waterConsumptionNonWest : ℝ := 26848.55
def waterConsumptionRussia : ℝ := 302790.13

-- Theorem statement to prove that the water consumption per person matches the given values
theorem water_consumption_comparison :
  waterConsumptionWest = 21428 ∧
  waterConsumptionNonWest = 26848.55 ∧
  waterConsumptionRussia = 302790.13 :=
by
  -- Sorry to skip the proof
  sorry

end water_consumption_comparison_l51_51021


namespace min_value_x_plus_inv_x_l51_51049

theorem min_value_x_plus_inv_x (x : ℝ) (hx : x > 0) : ∃ y, (y = x + 1/x) ∧ (∀ z, z = x + 1/x → z ≥ 2) :=
by
  sorry

end min_value_x_plus_inv_x_l51_51049


namespace solve_for_t_l51_51633

theorem solve_for_t (t : ℝ) (ht : (t^2 - 3*t - 70) / (t - 10) = 7 / (t + 4)) : 
  t = -3 := sorry

end solve_for_t_l51_51633


namespace hyperbola_real_axis_length_l51_51817

theorem hyperbola_real_axis_length :
  (∃ a : ℝ, (∀ x y : ℝ, (x^2 / 9 - y^2 = 1) → (2 * a = 6))) :=
sorry

end hyperbola_real_axis_length_l51_51817


namespace count_3_digit_multiples_of_13_l51_51098

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l51_51098


namespace count_3_digit_numbers_divisible_by_13_l51_51068

-- Definition of the problem's conditions
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem count_3_digit_numbers_divisible_by_13 : 
  ∃ n, n = 69 ∧ 
    ∀ k, (k ≥ smallest_3_digit ∧ k ≤ largest_3_digit) ∧ divisible_by_13(k) ↔ k ∈ (List.range' 104 (988-104+1)).filter divisible_by_13 := 
by
  sorry

end count_3_digit_numbers_divisible_by_13_l51_51068


namespace sequence_properties_l51_51444

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d : ℤ} {q : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a n = a 1 + (n - 1) * d

def geometric_sequence (b : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n, b n = b 1 * q^(n - 1)

theorem sequence_properties
  (ha : arithmetic_sequence a d)
  (hb : geometric_sequence b q)
  (h1 : 2 * a 5 - a 3 = 3)
  (h2 : b 2 = 1)
  (h3 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (q = 2 ∨ q = -2) :=
by
  sorry

end sequence_properties_l51_51444


namespace exists_positive_integers_abc_l51_51447

theorem exists_positive_integers_abc (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_m_gt_one : 1 < m) (h_n_gt_one : 1 < n) :
  ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ m^a = 1 + n^b * c ∧ Nat.gcd c n = 1 :=
by
  sorry

end exists_positive_integers_abc_l51_51447


namespace find_k_l51_51564

theorem find_k (k : ℝ) (h : ∀ x y : ℝ, (x, y) = (-2, -1) → y = k * x + 2) : k = 3 / 2 :=
sorry

end find_k_l51_51564


namespace hydrochloric_acid_moles_l51_51581

theorem hydrochloric_acid_moles (amyl_alcohol moles_required : ℕ) 
  (h_ratio : amyl_alcohol = moles_required) 
  (h_balanced : amyl_alcohol = 3) :
  moles_required = 3 :=
by
  sorry

end hydrochloric_acid_moles_l51_51581


namespace net_displacement_east_of_A_total_fuel_consumed_l51_51856

def distances : List Int := [22, -3, 4, -2, -8, -17, -2, 12, 7, -5]
def fuel_consumption_per_km : ℝ := 0.07

theorem net_displacement_east_of_A :
  List.sum distances = 8 := by
  sorry

theorem total_fuel_consumed :
  List.sum (distances.map Int.natAbs) * fuel_consumption_per_km = 5.74 := by
  sorry

end net_displacement_east_of_A_total_fuel_consumed_l51_51856


namespace isabella_jumped_farthest_l51_51805

-- defining the jumping distances
def ricciana_jump : ℕ := 4
def margarita_jump : ℕ := 2 * ricciana_jump - 1
def isabella_jump : ℕ := ricciana_jump + 3 

-- defining the total distances
def ricciana_total : ℕ := 20 + ricciana_jump
def margarita_total : ℕ := 18 + margarita_jump
def isabella_total : ℕ := 22 + isabella_jump

-- stating the theorem
theorem isabella_jumped_farthest : isabella_total = 29 :=
by sorry

end isabella_jumped_farthest_l51_51805


namespace license_plate_combinations_l51_51580

theorem license_plate_combinations :
  let num_consonants := 21
  let num_vowels := 5
  let num_digits := 10
  num_consonants^2 * num_vowels^2 * num_digits = 110250 :=
by
  let num_consonants := 21
  let num_vowels := 5
  let num_digits := 10
  sorry

end license_plate_combinations_l51_51580


namespace find_speeds_l51_51729

noncomputable def speed_pedestrian := 5
noncomputable def speed_cyclist := 30

def distance_AB := 40
def starting_time_pedestrian := 4 -- In hours (24-hour format)
def starting_time_cyclist_1 := 7 + 20 / 60 -- 7:20 AM in hours
def halfway_distance := distance_AB / 2
def midpoint_meeting_time := 1 -- Time (in hours) after the first meeting
def starting_time_cyclist_2 := 8 + 30 / 60 -- 8:30 AM in hours

theorem find_speeds (x y : ℝ) (hx : x = speed_pedestrian) (hy : y = speed_cyclist) :
  let time_to_halfway := halfway_distance / x in
  let cyclist_time := (midpoint_meeting_time + time_to_halfway) in
  distance_AB = 
    cyclist_time * y + 
    time_to_halfway * x + 
    (midpoint_meeting_time - 1) * x :=
    x = speed_pedestrian ∧ y = speed_cyclist :=
begin
  sorry
end

end find_speeds_l51_51729


namespace prove_tangency_l51_51198

noncomputable def parabola (p : ℝ) (hp : p > 0) : set (ℝ × ℝ) :=
  { z | z.snd^2 = 2 * p * z.fst }

structure Focus (p : ℝ) (hp : p > 0) : Type :=
(point : ℝ × ℝ)
(is_focus : point = (p / 2, 0))

structure PointOutsideParabola (p : ℝ) (hp : p > 0) : Type :=
(point : ℝ × ℝ)
(is_outside : ¬∃ x ∈ parabola p hp, point = x ∧ point.snd = 0)

structure TangentPoints (p : ℝ) (hp : p > 0) (P : PointOutsideParabola p hp) : Type :=
(A B : ℝ × ℝ)
(tangentA : A ∈ parabola p hp)
(tangentB : B ∈ parabola p hp)
(tangent_condition_A : ∃ l : line ℝ, l.contains P.point ∧ l.contains A)
(tangent_condition_B : ∃ l : line ℝ, l.contains P.point ∧ l.contains B)

structure IntersectionY (Q : ℝ × ℝ) : Type :=
(C D : ℝ × ℝ)
(intersects_y_axis_C : C.fst = 0)
(intersects_y_axis_D : D.fst = 0)
(line_QA : line ℝ)
(line_QB : line ℝ)
(intersection_C : ∃ A, tangents_to_A line_QA A ∧ P.tangentA)
(intersection_D : ∃ B, tangents_to_B line_QB B ∧ P.tangentB)

structure CircumcenterQAB (Q : PointOutsideParabola p hp) (T : TangentPoints p hp Q) : Type :=
(M : ℝ × ℝ)
(is_circumcenter: ∃ circumcircle : circle ℝ, circumcenter_of_triangle M Q.point T.A T.B)

theorem prove_tangency 
  (p : ℝ) (hp : p > 0)
  (F : Focus p hp)
  (Q : PointOutsideParabola p hp)
  (T : TangentPoints p hp Q)
  (I : IntersectionY Q.point)
  (CQC : CircumcenterQAB Q T)
  : tangent_to_circumcircle F.point CQC.M I.C I.D :=
sorry

end prove_tangency_l51_51198


namespace markese_earnings_16_l51_51797

theorem markese_earnings_16 (E M : ℕ) (h1 : M = E - 5) (h2 : E + M = 37) : M = 16 :=
by
  sorry

end markese_earnings_16_l51_51797


namespace solve_profession_arrangement_l51_51897

inductive Profession
| architect
| barista
| veterinarian
| guitarist

inductive Person
| Andrey
| Boris
| Vyacheslav
| Gennady

open Profession
open Person

structure SeatingArrangement :=
(seat1 : Person)
(seat2 : Person)
(seat3 : Person)
(seat4 : Person)

structure Assignment :=
(profession : Person → Profession)

def correct_arrangement_and_profession (sa : SeatingArrangement) (asgmt : Assignment) : Prop :=
(sa.seat1 ≠ sa.seat2 ∧ sa.seat1 ≠ sa.seat3 ∧ sa.seat1 ≠ sa.seat4 ∧
 sa.seat2 ≠ sa.seat3 ∧ sa.seat2 ≠ sa.seat4 ∧ sa.seat3 ≠ sa.seat4) ∧
((asgmt profession sa.seat1 = barista) ∧ (asgmt profession sa.seat2 = architect) ∧ 
 (asgmt profession sa.seat3 = veterinarian) ∧ (asgmt profession sa.seat4 = guitarist)) ∧
(sa.seat3 = Andrey) ∧ 
(sa.seat2 = Boris) ∧ 
(sa.seat4 = Vyacheslav)

theorem solve_profession_arrangement : 
  ∃ (sa : SeatingArrangement) (asgmt : Assignment), correct_arrangement_and_profession sa asgmt :=
sorry

end solve_profession_arrangement_l51_51897


namespace M_subset_P_l51_51970

universe u

-- Definitions of the sets
def U : Set ℝ := {x : ℝ | True}
def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x^2 > 1}

-- Proof statement
theorem M_subset_P : M ⊆ P := by
  sorry

end M_subset_P_l51_51970


namespace combined_percentage_tennis_is_31_l51_51645

-- Define the number of students at North High School
def students_north : ℕ := 1800

-- Define the number of students at South Elementary School
def students_south : ℕ := 2200

-- Define the percentage of students who prefer tennis at North High School
def percentage_tennis_north : ℚ := 25/100

-- Define the percentage of students who prefer tennis at South Elementary School
def percentage_tennis_south : ℚ := 35/100

-- Calculate the number of students who prefer tennis at North High School
def tennis_students_north : ℚ := students_north * percentage_tennis_north

-- Calculate the number of students who prefer tennis at South Elementary School
def tennis_students_south : ℚ := students_south * percentage_tennis_south

-- Calculate the total number of students who prefer tennis in both schools
def total_tennis_students : ℚ := tennis_students_north + tennis_students_south

-- Calculate the total number of students in both schools
def total_students : ℚ := students_north + students_south

-- Calculate the combined percentage of students who prefer tennis
def combined_percentage_tennis : ℚ := (total_tennis_students / total_students) * 100

-- Main statement to prove
theorem combined_percentage_tennis_is_31 :
  round combined_percentage_tennis = 31 := by sorry

end combined_percentage_tennis_is_31_l51_51645


namespace inverse_of_11_mod_1021_l51_51424

theorem inverse_of_11_mod_1021 : ∃ x : ℕ, x < 1021 ∧ 11 * x ≡ 1 [MOD 1021] := by
  use 557
  -- We leave the proof as an exercise.
  sorry

end inverse_of_11_mod_1021_l51_51424


namespace purely_imaginary_iff_l51_51571

noncomputable def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0

theorem purely_imaginary_iff (a : ℝ) :
  isPurelyImaginary (Complex.mk ((a * (a + 2)) / (a - 1)) (a ^ 2 + 2 * a - 3))
  ↔ a = 0 ∨ a = -2 := by
  sorry

end purely_imaginary_iff_l51_51571


namespace product_of_real_roots_eq_one_l51_51822

theorem product_of_real_roots_eq_one:
  ∀ x : ℝ, x ^ Real.log x = Real.exp 1 → (x = Real.exp 1 ∨ x = Real.exp (-1)) →
  x * (if x = Real.exp 1 then Real.exp (-1) else Real.exp 1) = 1 :=
by sorry

end product_of_real_roots_eq_one_l51_51822


namespace main_theorem_l51_51904

variable {a b c : ℝ}

noncomputable def inequality_1 (a b c : ℝ) : Prop :=
  a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b)

noncomputable def inequality_2 (a b c : ℝ) : Prop :=
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ a^2 / (b * c) + b^2 / (c * a) + c^2 / (a * b)

theorem main_theorem (h : a < 0 ∧ b < 0 ∧ c < 0) :
  inequality_1 a b c ∧ inequality_2 a b c := by sorry

end main_theorem_l51_51904


namespace speed_ratio_is_2_l51_51984

def distance_to_work : ℝ := 20
def total_hours_on_road : ℝ := 6
def speed_back_home : ℝ := 10

theorem speed_ratio_is_2 :
  (∃ v : ℝ, (20 / v) + (20 / 10) = 6) → (10 = 2 * v) :=
by sorry

end speed_ratio_is_2_l51_51984


namespace three_digit_numbers_divisible_by_13_l51_51147

theorem three_digit_numbers_divisible_by_13 : (finset.filter (λ n, n % 13 = 0) (finset.Icc 100 999)).card = 69 :=
sorry

end three_digit_numbers_divisible_by_13_l51_51147


namespace three_digit_numbers_divisible_by_13_count_l51_51133

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l51_51133


namespace total_distance_covered_l51_51991

def teams_data : List (String × Nat × Nat) :=
  [("Green Bay High", 5, 150), 
   ("Blue Ridge Middle", 7, 200),
   ("Sunset Valley Elementary", 4, 100),
   ("Riverbend Prep", 6, 250)]

theorem total_distance_covered (team : String) (members relays : Nat) :
  (team, members, relays) ∈ teams_data →
    (team = "Green Bay High" → members * relays = 750) ∧
    (team = "Blue Ridge Middle" → members * relays = 1400) ∧
    (team = "Sunset Valley Elementary" → members * relays = 400) ∧
    (team = "Riverbend Prep" → members * relays = 1500) :=
  by
    intros; sorry -- Proof omitted

end total_distance_covered_l51_51991


namespace number_of_moles_of_HCl_l51_51885

-- Defining the chemical equation relationship
def reaction_relation (HCl NaHCO3 NaCl H2O CO2 : ℕ) : Prop :=
  H2O = HCl ∧ H2O = NaHCO3

-- Conditions
def conditions (HCl NaHCO3 H2O : ℕ) : Prop :=
  NaHCO3 = 3 ∧ H2O = 3

-- Theorem statement proving the number of moles of HCl given the conditions
theorem number_of_moles_of_HCl (HCl NaHCO3 NaCl H2O CO2 : ℕ) 
  (h1 : reaction_relation HCl NaHCO3 NaCl H2O CO2) 
  (h2 : conditions HCl NaHCO3 H2O) :
  HCl = 3 :=
sorry

end number_of_moles_of_HCl_l51_51885


namespace probability_xiao_ming_chooses_king_of_sky_l51_51710

theorem probability_xiao_ming_chooses_king_of_sky :
  let choices := ["Life is Unfamiliar", "King of the Sky", "Prosecution Storm"]
  in Probability (Xiao Ming chooses "King of the Sky") = 1/3 :=
by sorry

end probability_xiao_ming_chooses_king_of_sky_l51_51710


namespace difference_is_693_l51_51665

noncomputable def one_tenth_of_seven_thousand : ℕ := 1 / 10 * 7000
noncomputable def one_tenth_percent_of_seven_thousand : ℕ := (1 / 10 / 100) * 7000
noncomputable def difference : ℕ := one_tenth_of_seven_thousand - one_tenth_percent_of_seven_thousand

theorem difference_is_693 :
  difference = 693 :=
by
  sorry

end difference_is_693_l51_51665


namespace mary_fruits_left_l51_51341

theorem mary_fruits_left (apples_initial : ℕ) (oranges_initial : ℕ) (blueberries_initial : ℕ)
                         (ate_apples : ℕ) (ate_oranges : ℕ) (ate_blueberries : ℕ) :
  apples_initial = 14 → oranges_initial = 9 → blueberries_initial = 6 → 
  ate_apples = 1 → ate_oranges = 1 → ate_blueberries = 1 → 
  (apples_initial - ate_apples) + (oranges_initial - ate_oranges) + (blueberries_initial - ate_blueberries) = 26 :=
by
  intros
  simp [*]
  sorry

end mary_fruits_left_l51_51341


namespace tonya_needs_to_eat_more_l51_51373

-- Define the conditions in the problem
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Define a function to calculate hamburgers given ounces
def hamburgers_eaten (ounces : ℕ) (ounce_per_hamburger : ℕ) : ℕ :=
  ounces / ounce_per_hamburger

-- State the theorem
theorem tonya_needs_to_eat_more (ounces_per_hamburger ounces_eaten_last_year : ℕ) :
  hamburgers_eaten ounces_eaten_last_year ounces_per_hamburger + 1 = 22 := by
  sorry

end tonya_needs_to_eat_more_l51_51373


namespace base6_divisible_by_13_l51_51293

theorem base6_divisible_by_13 (d : ℕ) (h : d < 6) : 13 ∣ (435 + 42 * d) ↔ d = 5 := 
by
  -- Proof implementation will go here, but is currently omitted
  sorry

end base6_divisible_by_13_l51_51293


namespace count_numbers_with_remainder_7_dividing_65_l51_51579

theorem count_numbers_with_remainder_7_dividing_65 : 
  (∃ n : ℕ, n > 7 ∧ n ∣ 58 ∧ 65 % n = 7) ∧ 
  (∀ m : ℕ, m > 7 ∧ m ∣ 58 ∧ 65 % m = 7 → m = 29 ∨ m = 58) :=
sorry

end count_numbers_with_remainder_7_dividing_65_l51_51579


namespace selection_probability_equal_l51_51899

theorem selection_probability_equal :
  let n := 2012
  let eliminated := 12
  let remaining := n - eliminated
  let selected := 50
  let probability := (remaining / n) * (selected / remaining)
  probability = 25 / 1006 :=
by
  sorry

end selection_probability_equal_l51_51899


namespace elaine_rent_percentage_l51_51672

theorem elaine_rent_percentage (E : ℝ) (hE : E > 0) :
  let rent_last_year := 0.20 * E
  let earnings_this_year := 1.25 * E
  let rent_this_year := 0.30 * earnings_this_year
  (rent_this_year / rent_last_year) * 100 = 187.5 :=
by
  sorry

end elaine_rent_percentage_l51_51672


namespace count_3_digit_multiples_of_13_l51_51097

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l51_51097


namespace finite_inf_n_rephinado_primes_l51_51380

noncomputable def delta := (1 + Real.sqrt 5) / 2 + 1

def is_nth_residue_mod (p n : ℕ) (a : ℕ) : Prop :=
  ∃ x : ℕ, x^n % p = a % p

def is_n_rephinado (p n : ℕ) : Prop :=
  n ∣ (p - 1) ∧ ∀ a, 1 ≤ a ∧ a ≤ Nat.floor (Real.sqrt (p ^ (1 / delta))) → is_nth_residue_mod p n a

theorem finite_inf_n_rephinado_primes :
  ∀ n : ℕ, ¬(∃ S : Set ℕ, S.infinite ∧ ∀ p ∈ S, is_n_rephinado p n) :=
sorry

end finite_inf_n_rephinado_primes_l51_51380


namespace final_price_of_bicycle_l51_51399

def original_price : ℝ := 200
def first_discount_rate : ℝ := 0.40
def second_discount_rate : ℝ := 0.25

theorem final_price_of_bicycle :
  let first_sale_price := original_price - (first_discount_rate * original_price)
  let final_sale_price := first_sale_price - (second_discount_rate * first_sale_price)
  final_sale_price = 90 := by
  sorry

end final_price_of_bicycle_l51_51399


namespace fifteenth_digit_sum_l51_51511

theorem fifteenth_digit_sum (d₁ d₂ : ℕ → ℕ) 
  (h₁ : ∀ n, d₁ n = if n = 0 then 1 else 0) 
  (h₂ : ∀ n, d₂ n = if n % 2 = 0 then 0 else 9) :
  let sum_digit := λ n, (d₁ n + d₂ n) % 10 in
  sum_digit 14 = 1 :=
by 
  sorry

end fifteenth_digit_sum_l51_51511


namespace area_at_stage_8_l51_51935

theorem area_at_stage_8 
  (side_length : ℕ)
  (stage : ℕ)
  (num_squares : ℕ)
  (square_area : ℕ) 
  (total_area : ℕ) 
  (h1 : side_length = 4) 
  (h2 : stage = 8) 
  (h3 : num_squares = stage) 
  (h4 : square_area = side_length * side_length) 
  (h5 : total_area = num_squares * square_area) :
  total_area = 128 :=
sorry

end area_at_stage_8_l51_51935


namespace tonya_hamburgers_to_beat_winner_l51_51376

-- Given conditions
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Calculate the number of hamburgers eaten last year
def hamburgers_eaten_last_year : ℕ := ounces_eaten_last_year / ounces_per_hamburger

-- Prove the number of hamburgers Tonya needs to eat to beat last year's winner
theorem tonya_hamburgers_to_beat_winner : 
  hamburgers_eaten_last_year + 1 = 22 :=
by
  -- It remains to be proven
  sorry

end tonya_hamburgers_to_beat_winner_l51_51376


namespace max_coach_handshakes_l51_51868

-- Define the problem variables and conditions
noncomputable def coach_max_handshakes (total_handshakes : ℕ) := 
  ∃ (n k : ℕ), nat.choose n 2 + k = total_handshakes ∧ k = 0

-- Statement with total handshakes set to 465
theorem max_coach_handshakes : coach_max_handshakes 465 :=
sorry

end max_coach_handshakes_l51_51868


namespace sufficient_but_not_necessary_l51_51195

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1/2 → 2 * x^2 + x - 1 > 0) ∧ ¬(2 * x^2 + x - 1 > 0 → x > 1 / 2) := 
by
  sorry

end sufficient_but_not_necessary_l51_51195


namespace total_votes_l51_51471

theorem total_votes (V : ℕ) 
  (h1 : V * 45 / 100 + V * 25 / 100 + V * 15 / 100 + 180 + 50 = V) : 
  V = 1533 := 
by
  sorry

end total_votes_l51_51471


namespace prob_both_successful_prob_at_least_one_successful_l51_51254

variables (P_A P_B : ℚ)
variables (h1 : P_A = 1 / 2)
variables (h2 : P_B = 2 / 5)

/-- Prove that the probability that both A and B score in one shot each is 1 / 5. -/
theorem prob_both_successful (P_A P_B : ℚ) (h1 : P_A = 1 / 2) (h2 : P_B = 2 / 5) :
  P_A * P_B = 1 / 5 :=
by sorry

variables (P_A_miss P_B_miss : ℚ)
variables (h3 : P_A_miss = 1 / 2)
variables (h4 : P_B_miss = 3 / 5)

/-- Prove that the probability that at least one shot is successful is 7 / 10. -/
theorem prob_at_least_one_successful (P_A_miss P_B_miss : ℚ) (h3 : P_A_miss = 1 / 2) (h4 : P_B_miss = 3 / 5) :
  1 - P_A_miss * P_B_miss = 7 / 10 :=
by sorry

end prob_both_successful_prob_at_least_one_successful_l51_51254


namespace rhombus_perimeter_l51_51360

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * (Nat.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 68 :=
by
  sorry

end rhombus_perimeter_l51_51360


namespace min_value_at_neg7_l51_51385

noncomputable def f (x : ℝ) : ℝ := x^2 + 14 * x + 24

theorem min_value_at_neg7 : ∀ x : ℝ, f (-7) ≤ f x :=
by
  sorry

end min_value_at_neg7_l51_51385


namespace evaluate_ceiling_expression_l51_51281

theorem evaluate_ceiling_expression:
  (Int.ceil ((23 : ℚ) / 9 - Int.ceil ((35 : ℚ) / 23)))
  / (Int.ceil ((35 : ℚ) / 9 + Int.ceil ((9 * 23 : ℚ) / 35))) = 1 / 12 := by
  sorry

end evaluate_ceiling_expression_l51_51281


namespace distance_problem_l51_51532

-- Define the problem
theorem distance_problem
  (x y : ℝ)
  (h1 : x + y = 21)
  (h2 : x / 60 + 21 / 60 = 10 / 60 + y / 4) :
  x = 19 ∧ y = 2 :=
by
  sorry

end distance_problem_l51_51532


namespace employed_females_percentage_l51_51776

def P_total : ℝ := 0.64
def P_males : ℝ := 0.46

theorem employed_females_percentage : 
  ((P_total - P_males) / P_total) * 100 = 28.125 :=
by
  sorry

end employed_females_percentage_l51_51776


namespace count_three_digit_numbers_divisible_by_13_l51_51127

-- Define a function that checks if a number is a three-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function that checks if a number is divisible by 13
def divisible_by_13 (n : ℕ) : Prop :=
  n % 13 = 0

-- Define the main theorem
theorem count_three_digit_numbers_divisible_by_13 :
  { n : ℕ | is_three_digit n ∧ divisible_by_13 n }.to_finset.card = 69 := 
sorry

end count_three_digit_numbers_divisible_by_13_l51_51127


namespace positive_3_digit_numbers_divisible_by_13_count_l51_51080

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l51_51080


namespace donuts_Niraek_covers_l51_51006

/- Define the radii of the donut holes -/
def radius_Niraek : ℕ := 5
def radius_Theo : ℕ := 9
def radius_Akshaj : ℕ := 10
def radius_Lily : ℕ := 7

/- Define the surface areas of the donut holes -/
def surface_area (r : ℕ) : ℕ := 4 * r * r

/- Compute the surface areas -/
def sa_Niraek := surface_area radius_Niraek
def sa_Theo := surface_area radius_Theo
def sa_Akshaj := surface_area radius_Akshaj
def sa_Lily := surface_area radius_Lily

/- Define a function to compute the LCM of a list of natural numbers -/
def lcm_of_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

/- Compute the lcm of the surface areas -/
def lcm_surface_areas := lcm_of_list [sa_Niraek, sa_Theo, sa_Akshaj, sa_Lily]

/- Compute the answer -/
def num_donuts_Niraek_covers := lcm_surface_areas / sa_Niraek

/- Prove the statement -/
theorem donuts_Niraek_covers : num_donuts_Niraek_covers = 63504 :=
by
  /- Skipping the proof for now -/
  sorry

end donuts_Niraek_covers_l51_51006


namespace exist_ai_for_xij_l51_51804

theorem exist_ai_for_xij (n : ℕ) (x : Fin n → Fin n → ℝ)
  (h : ∀ i j k : Fin n, x i j + x j k + x k i = 0) :
  ∃ a : Fin n → ℝ, ∀ i j : Fin n, x i j = a i - a j :=
by
  sorry

end exist_ai_for_xij_l51_51804


namespace problem_l51_51043

theorem problem (f : ℝ → ℝ) (h : ∀ x, (x - 3) * (deriv f x) ≤ 0) : 
  f 0 + f 6 ≤ 2 * f 3 := 
sorry

end problem_l51_51043


namespace find_f_and_q_l51_51451

theorem find_f_and_q (m : ℤ) (q : ℝ) :
  (∀ x > 0, (x : ℝ)^(-m^2 + 2*m + 3) = (x : ℝ)^4) ∧
  (∀ x ∈ [-1, 1], 2 * (x^2) - 8 * x + q - 1 > 0) →
  q > 7 :=
by
  sorry

end find_f_and_q_l51_51451


namespace inequality_abc_l51_51617

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 :=
by
  sorry

end inequality_abc_l51_51617


namespace final_temp_fahrenheit_correct_l51_51595

noncomputable def initial_temp_celsius : ℝ := 50
noncomputable def conversion_c_to_f (c: ℝ) : ℝ := (c * 9 / 5) + 32
noncomputable def final_temp_celsius := initial_temp_celsius / 2

theorem final_temp_fahrenheit_correct : conversion_c_to_f final_temp_celsius = 77 :=
  by sorry

end final_temp_fahrenheit_correct_l51_51595


namespace trig_identity_l51_51438

theorem trig_identity (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) : Real.cos (α + 5 * π / 12) = -1 / 3 :=
by
  sorry

end trig_identity_l51_51438


namespace bicycles_wheels_l51_51508

theorem bicycles_wheels (b : ℕ) (h1 : 3 * b + 4 * 3 + 7 * 1 = 25) : b = 2 :=
sorry

end bicycles_wheels_l51_51508


namespace find_m_l51_51416

theorem find_m (m : ℝ) (h1 : m^2 - 3 * m + 2 = 0) (h2 : m ≠ 1) : m = 2 :=
sorry

end find_m_l51_51416


namespace PQ_length_l51_51953

theorem PQ_length (BC AD : ℝ) (angle_A angle_D : ℝ) (P Q : ℝ) 
  (H1 : BC = 700) (H2 : AD = 1400) (H3 : angle_A = 45) (H4 : angle_D = 45) 
  (mid_BC : P = BC / 2) (mid_AD : Q = AD / 2) :
  abs (Q - P) = 350 :=
by
  sorry

end PQ_length_l51_51953


namespace greatest_multiple_of_5_and_6_less_than_800_l51_51516

theorem greatest_multiple_of_5_and_6_less_than_800 : 
  ∃ n : ℕ, n < 800 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 800 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
sorry

end greatest_multiple_of_5_and_6_less_than_800_l51_51516


namespace sum_of_fractions_l51_51831

theorem sum_of_fractions : (3/7 : ℚ) + (5/14 : ℚ) = 11/14 :=
by
  sorry

end sum_of_fractions_l51_51831


namespace number_of_students_and_average_output_l51_51667

theorem number_of_students_and_average_output 
  (total_potatoes : ℕ)
  (days : ℕ)
  (x y : ℕ) 
  (h1 : total_potatoes = 45715) 
  (h2 : days = 5)
  (h3 : x * y * days = total_potatoes) : 
  x = 41 ∧ y = 223 :=
by
  sorry

end number_of_students_and_average_output_l51_51667


namespace functional_relationship_l51_51299

-- Define the conditions
def directlyProportional (y x k : ℝ) : Prop :=
  y + 6 = k * (x + 1)

def specificCondition1 (x y : ℝ) : Prop :=
  x = 3 ∧ y = 2

-- State the theorem
theorem functional_relationship (k : ℝ) :
  (∀ x y, directlyProportional y x k) →
  specificCondition1 3 2 →
  ∀ x, ∃ y, y = 2 * x - 4 :=
by
  intro directProp
  intro specCond
  sorry

end functional_relationship_l51_51299


namespace monthly_income_of_labourer_l51_51221

variable (I : ℕ) -- Monthly income

-- Conditions: 
def condition1 := (85 * 6) - (6 * I) -- A boolean expression depicting the labourer fell into debt
def condition2 := (60 * 4) + (85 * 6 - 6 * I) + 30 -- Total income covers debt and saving 30

-- Statement to be proven
theorem monthly_income_of_labourer : 
  ∃ I : ℕ, condition1 I = 0 ∧ condition2 I = 4 * I → I = 78 :=
by
  sorry

end monthly_income_of_labourer_l51_51221


namespace equilateral_triangle_l51_51976

namespace TriangleEquilateral

-- Define the structure of a triangle and given conditions
structure Triangle :=
  (A B C : ℝ)  -- vertices
  (angleA : ℝ) -- angle at vertex A
  (sideBC : ℝ) -- length of side BC
  (perimeter : ℝ)  -- perimeter of the triangle

-- Define the proof problem
theorem equilateral_triangle (T : Triangle) (h1 : T.angleA = 60)
  (h2 : T.sideBC = T.perimeter / 3) : 
  T.A = T.B ∧ T.B = T.C ∧ T.A = T.C ∧ T.A = T.B ∧ T.B = T.C ∧ T.A = T.C :=
  sorry

end TriangleEquilateral

end equilateral_triangle_l51_51976


namespace count_3_digit_multiples_of_13_l51_51096

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l51_51096


namespace intersection_eq_l51_51297

def A : Set ℝ := {x : ℝ | (x - 2) / (x + 3) ≤ 0 }
def B : Set ℝ := {x : ℝ | x ≤ 1 }

theorem intersection_eq : A ∩ B = {x : ℝ | -3 < x ∧ x ≤ 1 } :=
sorry

end intersection_eq_l51_51297


namespace height_ratio_l51_51411

theorem height_ratio (C : ℝ) (h_o : ℝ) (V_s : ℝ) (h_s : ℝ) (r : ℝ) :
  C = 18 * π →
  h_o = 20 →
  V_s = 270 * π →
  C = 2 * π * r →
  V_s = 1 / 3 * π * r^2 * h_s →
  h_s / h_o = 1 / 2 :=
by
  sorry

end height_ratio_l51_51411


namespace seating_profession_solution_l51_51894

inductive Profession
| architect
| barista
| veterinarian
| guitarist

open Profession

inductive Friend
| Andrey
| Boris
| Vyacheslav
| Gennady

open Friend

structure SeatArrangement :=
(seat1: Friend)
(seat2: Friend)
(seat3: Friend)
(seat4: Friend)

structure ProfessionAssignment :=
(Andrey_profession: Profession)
(Boris_profession: Profession)
(Vyacheslav_profession: Profession)
(Gennady_profession: Profession)

noncomputable def correct_assignment (seats: SeatArrangement) : ProfessionAssignment :=
{ Andrey_profession := veterinarian,
  Boris_profession := architect,
  Vyacheslav_profession := guitarist,
  Gennady_profession := barista }

theorem seating_profession_solution
  (seats: SeatArrangement)
  (cond1 : ∀ f, f ∈ [seats.seat2, seats.seat3] ↔ ∃ p, p ∈ [architect, guitarist])
  (cond2 : seats.seat2 = Boris)
  (cond3 : (seats.seat3 = Vyacheslav ∧ seats.seat1 = Andrey) ∨ (seats.seat4 = Vyacheslav ∧ seats.seat2 = Andrey))
  (cond4 : (seats.seat1 = Andrey ∨ seats.seat2 = Andrey) ∧ Andrey ∉ [seats.seat1, seats.seat4])
  (cond5 : seats.seat1 ≠ Vyacheslav ∧ (seats.seat2 ≠ Gennady ∨ seats.seat3 ≠ Gennady)) :
  correct_assignment seats = 
  { Andrey_profession := veterinarian,
    Boris_profession := architect,
    Vyacheslav_profession := guitarist,
    Gennady_profession := barista } :=
sorry

end seating_profession_solution_l51_51894


namespace num_factors_180_l51_51924

theorem num_factors_180 : 
  ∃ (n : ℕ), n = 180 ∧ prime_factorization n = [(2, 2), (3, 2), (5, 1)] ∧ positive_factors n = 18 :=
by
  sorry

end num_factors_180_l51_51924


namespace number_of_factors_180_l51_51920

theorem number_of_factors_180 : 
  let factors180 (n : ℕ) := 180 
  in factors180 180 = (2 + 1) * (2 + 1) * (1 + 1) => factors180 180 = 18 := 
by
  sorry

end number_of_factors_180_l51_51920


namespace students_enjoy_both_music_and_sports_l51_51602

theorem students_enjoy_both_music_and_sports :
  ∀ (T M S N B : ℕ), T = 55 → M = 35 → S = 45 → N = 4 → B = M + S - (T - N) → B = 29 :=
by
  intros T M S N B hT hM hS hN hB
  rw [hT, hM, hS, hN] at hB
  exact hB

end students_enjoy_both_music_and_sports_l51_51602


namespace dot_product_square_ABCD_l51_51765

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l51_51765


namespace vaccine_codes_l51_51319

theorem vaccine_codes (vaccines : List ℕ) :
  vaccines = [785, 567, 199, 507, 175] :=
  by
  sorry

end vaccine_codes_l51_51319


namespace diagonals_sum_pentagon_inscribed_in_circle_l51_51961

theorem diagonals_sum_pentagon_inscribed_in_circle
  (FG HI GH IJ FJ : ℝ)
  (h1 : FG = 4)
  (h2 : HI = 4)
  (h3 : GH = 11)
  (h4 : IJ = 11)
  (h5 : FJ = 15) :
  3 * FJ + (FJ^2 - 121) / 4 + (FJ^2 - 16) / 11 = 80 := by {
  sorry
}

end diagonals_sum_pentagon_inscribed_in_circle_l51_51961


namespace number_of_3_digit_divisible_by_13_l51_51070

theorem number_of_3_digit_divisible_by_13 : ∃ n : ℕ, n = 69 ∧ 
    (∀ k : ℕ, (100 ≤ k ∧ k < 1000 ∧ k % 13 = 0) ↔ (∃ j : ℕ, j < n ∧ k = 104 + j * 13)) :=
by
  use 69
  split
  { exact rfl }
  { intro k
    split
    { intro h
      obtain ⟨l, h1, h2⟩ := (exists_eq_mul_right_of_dvd h.2)
      have : l = k / 13 := by
        rw [← h2, nat.mul_div_cancel h.2]
      rw this at h.1
      simp at h1; sorry
    }
    { intro h
      obtain ⟨j, j_lt, rfl⟩ := h
      split
      { sorry }
      { sorry }
    }
  }
  sorry

end number_of_3_digit_divisible_by_13_l51_51070


namespace number_of_pink_cookies_l51_51624

def total_cookies : ℕ := 86
def red_cookies : ℕ := 36

def pink_cookies (total red : ℕ) : ℕ := total - red

theorem number_of_pink_cookies : pink_cookies total_cookies red_cookies = 50 :=
by
  sorry

end number_of_pink_cookies_l51_51624


namespace sachin_rahul_age_ratio_l51_51210

theorem sachin_rahul_age_ratio 
(S_age : ℕ) 
(R_age : ℕ) 
(h1 : R_age = S_age + 4) 
(h2 : S_age = 14) : 
S_age / Int.gcd S_age R_age = 7 ∧ R_age / Int.gcd S_age R_age = 9 := 
by 
sorry

end sachin_rahul_age_ratio_l51_51210


namespace proposition_1_proposition_2_proposition_3_proposition_4_l51_51836

theorem proposition_1 : ∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0 := sorry

theorem proposition_2 : ¬ (∀ x ∈ ({-1, 0, 1} : Set ℤ), 2 * x + 1 > 0) := sorry

theorem proposition_3 : ∃ x : ℕ, x^2 ≤ x := sorry

theorem proposition_4 : ∃ x : ℕ, x ∣ 29 := sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l51_51836


namespace theta_plus_2phi_l51_51443

theorem theta_plus_2phi (θ φ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (hφ : 0 < φ ∧ φ < π / 2)
  (h_tan_θ : Real.tan θ = 1 / 7) (h_sin_φ : Real.sin φ = 1 / Real.sqrt 10) :
  θ + 2 * φ = π / 4 := 
sorry

end theta_plus_2phi_l51_51443


namespace cone_height_l51_51945

theorem cone_height (r h : ℝ) (π : ℝ) (Hπ : Real.pi = π) (slant_height : ℝ) (lateral_area : ℝ) (base_area : ℝ) 
  (H1 : slant_height = 2) 
  (H2 : lateral_area = 2 * π * r) 
  (H3 : base_area = π * r^2) 
  (H4 : lateral_area = 4 * base_area) 
  (H5 : r^2 + h^2 = slant_height^2) 
  : h = π / 2 := by 
sorry

end cone_height_l51_51945


namespace derek_books_ratio_l51_51707

theorem derek_books_ratio :
  ∃ (T : ℝ), 960 - T - (1/4) * (960 - T) = 360 ∧ T / 960 = 1 / 2 :=
by
  sorry

end derek_books_ratio_l51_51707


namespace divide_number_l51_51278

theorem divide_number (x : ℝ) (h : 0.3 * x = 0.2 * (80 - x) + 10) : min x (80 - x) = 28 := 
by 
  sorry

end divide_number_l51_51278


namespace find_total_amount_l51_51850

noncomputable def total_amount (A T yearly_income : ℝ) : Prop :=
  0.05 * A + 0.06 * (T - A) = yearly_income

theorem find_total_amount :
  ∃ T : ℝ, total_amount 1600 T 140 ∧ T = 2600 :=
sorry

end find_total_amount_l51_51850


namespace expression_is_five_l51_51717

-- Define the expression
def given_expression : ℤ := abs (abs (-abs (-2 + 1) - 2) + 2)

-- Prove that the expression equals 5
theorem expression_is_five : given_expression = 5 :=
by
  -- We skip the proof for now
  sorry

end expression_is_five_l51_51717


namespace dot_product_EC_ED_l51_51766

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l51_51766


namespace function_evaluation_l51_51740

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 + 2 * x :=
by
  sorry

end function_evaluation_l51_51740


namespace min_value_abs_expression_l51_51582

theorem min_value_abs_expression {p x : ℝ} (hp1 : 0 < p) (hp2 : p < 15) (hx1 : p ≤ x) (hx2 : x ≤ 15) :
  |x - p| + |x - 15| + |x - p - 15| = 15 :=
sorry

end min_value_abs_expression_l51_51582


namespace triangle_side_lengths_l51_51750

theorem triangle_side_lengths (A B C : ℝ) (a b c : ℝ) 
  (hcosA : Real.cos A = 1/4)
  (ha : a = 4)
  (hbc_sum : b + c = 6)
  (hbc_order : b < c) :
  b = 2 ∧ c = 4 := by
  sorry

end triangle_side_lengths_l51_51750


namespace arithmetic_expression_eq2016_l51_51208

theorem arithmetic_expression_eq2016 :
  (1 / 8 : ℚ) * (1 / 9) * (1 / 28) = 1 / 2016 ∨ ((1 / 8 - 1 / 9) * (1 / 28) = 1 / 2016) := 
by sorry

end arithmetic_expression_eq2016_l51_51208


namespace num_3_digit_div_by_13_l51_51154

theorem num_3_digit_div_by_13 : 
  ∃ (n : ℕ), n = 69 ∧ ∀ (x : ℕ), 100 ≤ x ∧ x ≤ 999 ∧ x % 13 = 0 ↔ x = 13 * (ceiling (100 / 13)) + 13 * (x - ceiling (100 / 13) - 1) :=
sorry

end num_3_digit_div_by_13_l51_51154


namespace not_product_of_two_primes_l51_51967

theorem not_product_of_two_primes (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : ∃ n : ℕ, a^3 + b^3 = n^2) :
  ¬ (∃ p q : ℕ, p ≠ q ∧ Prime p ∧ Prime q ∧ a + b = p * q) :=
by
  sorry

end not_product_of_two_primes_l51_51967


namespace find_c_l51_51459

theorem find_c (c : ℝ) (h : ∃ a : ℝ, x^2 - 50 * x + c = (x - a)^2) : c = 625 :=
  by
  sorry

end find_c_l51_51459


namespace find_a_b_find_k_l51_51913

/-- The mathematical problem given the conditions and required proofs -/
noncomputable def g (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1 + b

noncomputable def f (a b x : ℝ) : ℝ := g a b (abs x)

theorem find_a_b (h₁ : ∀ x ∈ Icc 2 4, g a b x ≤ 9 ∧ g a b x ≥ 1) (pos_a : 0 < a) :
  a = 1 ∧ b = 0 :=
sorry

theorem find_k (pos_a : 0 < a) (k : ℝ) :
  f 1 0 (Real.log2 k) > f 1 0 2 ↔ k > 4 ∨ (0 < k ∧ k < 1/4) :=
sorry

end find_a_b_find_k_l51_51913


namespace set_subset_l51_51742

-- Define the sets M and N
def M := {x : ℝ | abs x ≤ 1}
def N := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x ≤ 0}

-- The mathematical statement to be proved
theorem set_subset : N ⊆ M := sorry

end set_subset_l51_51742


namespace count_three_digit_numbers_divisible_by_13_l51_51082

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l51_51082


namespace arithmetic_to_geometric_progression_l51_51209

theorem arithmetic_to_geometric_progression (x y z : ℝ) 
  (hAP : 2 * y^2 - y * x = z^2) : 
  z^2 = y * (2 * y - x) := 
  by 
  sorry

end arithmetic_to_geometric_progression_l51_51209


namespace flashlight_distance_difference_l51_51379

/--
Veronica's flashlight can be seen from 1000 feet. Freddie's flashlight can be seen from a distance
three times that of Veronica's flashlight. Velma's flashlight can be seen from a distance 2000 feet
less than 5 times Freddie's flashlight distance. We want to prove that Velma's flashlight can be seen 
12000 feet farther than Veronica's flashlight.
-/
theorem flashlight_distance_difference :
  let v_d := 1000
  let f_d := 3 * v_d
  let V_d := 5 * f_d - 2000
  V_d - v_d = 12000 := by
    sorry

end flashlight_distance_difference_l51_51379


namespace numberOfHandshakes_is_correct_l51_51828

noncomputable def numberOfHandshakes : ℕ :=
  let gremlins := 30
  let imps := 20
  let friendlyImps := 5
  let gremlinHandshakes := gremlins * (gremlins - 1) / 2
  let impGremlinHandshakes := imps * gremlins
  let friendlyImpHandshakes := friendlyImps * (friendlyImps - 1) / 2
  gremlinHandshakes + impGremlinHandshakes + friendlyImpHandshakes

theorem numberOfHandshakes_is_correct : numberOfHandshakes = 1045 := by
  sorry

end numberOfHandshakes_is_correct_l51_51828


namespace contrapositive_proposition_contrapositive_equiv_l51_51225

theorem contrapositive_proposition (x : ℝ) (h : -1 < x ∧ x < 1) : (x^2 < 1) :=
sorry

theorem contrapositive_equiv (x : ℝ) (h : x^2 ≥ 1) : x ≥ 1 ∨ x ≤ -1 :=
sorry

end contrapositive_proposition_contrapositive_equiv_l51_51225


namespace area_union_square_circle_l51_51538

theorem area_union_square_circle (side_length: ℝ) (radius: ℝ) (h1: side_length = 12) (h2: radius = 12):
  let square_area := side_length^2
  let circle_area := π * radius^2
  let union_area := if radius >= side_length / real.sqrt 2 then circle_area
                    else square_area + circle_area - overlapping_area radius square_area
  union_area = 144 * π := 
by
  -- Let the side length of the square and the radius of the circle be given as 12.
  have side_length_12: side_length = 12 := h1
  have radius_12: radius = 12 := h2
  -- Calculate the area of the square.
  have square_area_144: square_area = 144 := 
    by rw [side_length_12, pow_two]; exact rfl
  -- Calculate the area of the circle.
  have circle_area_144pi: circle_area = 144 * π := 
    by rw [radius_12, pow_two, mul_comm]; exact rfl
  -- The circle completely covers the square, so the area of the union is the area of the circle.
  rw [if_pos] -- because radius >= side_length / real.sqrt 2, 12 >= 12 / sqrt 2 is obviously true
  exact circle_area_144pi
  -- skip proof of overlapping_area function because we don't need it in this case
  sorry

end area_union_square_circle_l51_51538


namespace union_of_A_and_B_l51_51339

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem union_of_A_and_B :
  (A ∪ B) = {1, 2, 3, 4, 5, 7} := 
by
  sorry

end union_of_A_and_B_l51_51339


namespace intersection_of_M_and_P_l51_51452

def M : Set ℝ := { x | x^2 = x }
def P : Set ℝ := { x | |x - 1| = 1 }

theorem intersection_of_M_and_P : M ∩ P = {0} := by
  sorry

end intersection_of_M_and_P_l51_51452


namespace find_pq_l51_51703

noncomputable def area_of_triangle (p q : ℝ) : ℝ := 1/2 * (12 / p) * (12 / q)

theorem find_pq (p q : ℝ) (hp : p > 0) (hq : q > 0) (harea : area_of_triangle p q = 12) : p * q = 6 := 
by
  sorry

end find_pq_l51_51703


namespace koschei_coins_l51_51331

theorem koschei_coins :
  ∃ a : ℕ, (a % 10 = 7) ∧ (a % 12 = 9) ∧ (300 ≤ a) ∧ (a ≤ 400) ∧ a = 357 :=
by 
  sorry

end koschei_coins_l51_51331


namespace mary_fruits_left_l51_51343

theorem mary_fruits_left (apples_initial oranges_initial blueberries_initial : ℕ)
  (apples_eaten oranges_eaten blueberries_eaten : ℕ) :
  apples_initial = 14 →
  oranges_initial = 9 →
  blueberries_initial = 6 →
  apples_eaten = 1 →
  oranges_eaten = 1 →
  blueberries_eaten = 1 →
  (apples_initial - apples_eaten) + (oranges_initial - oranges_eaten) + (blueberries_initial - blueberries_eaten) = 26 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end mary_fruits_left_l51_51343


namespace number_of_solutions_l51_51430

-- Define the equation
def equation (x : ℝ) : Prop := (3 * x^2 - 15 * x) / (x^2 - 7 * x + 10) = x - 4

-- State the problem with conditions and conclusion
theorem number_of_solutions : (∀ x : ℝ, x ≠ 2 ∧ x ≠ 5 → equation x) ↔ (∃ x1 x2 : ℝ, x1 ≠ 2 ∧ x1 ≠ 5 ∧ x2 ≠ 2 ∧ x2 ≠ 5 ∧ equation x1 ∧ equation x2) :=
by
  sorry

end number_of_solutions_l51_51430


namespace trip_time_difference_l51_51412

theorem trip_time_difference (speed distance1 distance2 : ℕ) (h1 : speed > 0) (h2 : distance2 > distance1) 
  (h3 : speed = 60) (h4 : distance1 = 540) (h5 : distance2 = 570) : 
  (distance2 - distance1) / speed * 60 = 30 := 
by
  sorry

end trip_time_difference_l51_51412


namespace koschei_coins_l51_51330

theorem koschei_coins :
  ∃ a : ℕ, (a % 10 = 7) ∧ (a % 12 = 9) ∧ (300 ≤ a) ∧ (a ≤ 400) ∧ a = 357 :=
by 
  sorry

end koschei_coins_l51_51330


namespace paint_room_together_l51_51015

variable (t : ℚ)
variable (Doug_rate : ℚ := 1/5)
variable (Dave_rate : ℚ := 1/7)
variable (Diana_rate : ℚ := 1/6)
variable (Combined_rate : ℚ := Doug_rate + Dave_rate + Diana_rate)
variable (break_time : ℚ := 2)

theorem paint_room_together:
  Combined_rate * (t - break_time) = 1 :=
sorry

end paint_room_together_l51_51015


namespace expected_value_product_l51_51983

/-- 
  The expected value of the product of two three-digit numbers M and N, 
  where the digits 1, 2, 3, 4, 5, and 6 are chosen randomly without replacement 
  to form M = ABC and N = DEF, is 143745.
-/
theorem expected_value_product : 
  (∃ (M N : ℕ), M ∈ {100 * A + 10 * B + C | A B C ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C} ∧ 
                      N ∈ {100 * D + 10 * E + F | D E F ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) ∧ D ≠ E ∧ E ≠ F ∧ D ≠ F}) → 
    ∑ (M, N : ℕ), (M * N) / (6.choose 3 * (6-3).choose 3) = 143745 :=
by 
  sorry

end expected_value_product_l51_51983


namespace largest_prime_factor_of_1729_l51_51998

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l51_51998


namespace total_handshakes_l51_51543

section Handshakes

-- Define the total number of players
def total_players : ℕ := 4 + 6

-- Define the number of players in 2 and 3 player teams
def num_2player_teams : ℕ := 2
def num_3player_teams : ℕ := 2

-- Define the number of players per 2 player team and 3 player team
def players_per_2player_team : ℕ := 2
def players_per_3player_team : ℕ := 3

-- Define the total number of players in 2 player teams and in 3 player teams
def total_2player_team_players : ℕ := num_2player_teams * players_per_2player_team
def total_3player_team_players : ℕ := num_3player_teams * players_per_3player_team

-- Calculate handshakes
def handshakes (total_2player : ℕ) (total_3player : ℕ) : ℕ :=
  let h1 := total_2player * (total_players - players_per_2player_team) / 2
  let h2 := total_3player * (total_players - players_per_3player_team) / 2
  h1 + h2

-- Prove the total number of handshakes
theorem total_handshakes : handshakes total_2player_team_players total_3player_team_players = 37 :=
by
  have h1 := total_2player_team_players * (total_players - players_per_2player_team) / 2
  have h2 := total_3player_team_players * (total_players - players_per_3player_team) / 2
  have h_total := h1 + h2
  sorry

end Handshakes

end total_handshakes_l51_51543


namespace num_4digit_numbers_divisible_by_5_l51_51457

theorem num_4digit_numbers_divisible_by_5 : 
  (#{ n : ℕ | n ≥ 1000 ∧ n ≤ 9999 ∧ n % 5 = 0 }.finite.to_finset.card) = 1800 :=
by
  sorry

end num_4digit_numbers_divisible_by_5_l51_51457


namespace transform_polynomial_l51_51586

variable (x z : ℝ)

theorem transform_polynomial (h1 : z = x - 1 / x) (h2 : x^4 - 3 * x^3 - 2 * x^2 + 3 * x + 1 = 0) :
  x^2 * (z^2 - 3 * z) = 0 :=
sorry

end transform_polynomial_l51_51586


namespace angle_CDE_proof_l51_51472

theorem angle_CDE_proof
    (A B C D E : Type)
    (angle_A angle_B angle_C : ℝ)
    (angle_AEB : ℝ)
    (angle_BED : ℝ)
    (angle_BDE : ℝ) :
    angle_A = 90 ∧
    angle_B = 90 ∧
    angle_C = 90 ∧
    angle_AEB = 50 ∧
    angle_BED = 2 * angle_BDE →
    ∃ angle_CDE : ℝ, angle_CDE = 70 :=
by
  sorry

end angle_CDE_proof_l51_51472


namespace num_pos_3_digit_div_by_13_l51_51167

theorem num_pos_3_digit_div_by_13 : 
  ∃ n : ℕ, n = 69 ∧ ∀ x : ℕ, x ∈ (range 1000).filter (λ x, x >= 100 ∧ x % 13 = 0) ↔ x ∈ (list.range (n + 1)).map (λ k, 104 + 13 * k) :=
by sorry

end num_pos_3_digit_div_by_13_l51_51167


namespace circles_intersect_condition_l51_51566

theorem circles_intersect_condition (a : ℝ) (ha : a > 0) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + y^2 = 16) ↔ 3 < a ∧ a < 5 :=
by sorry

end circles_intersect_condition_l51_51566


namespace denote_loss_of_300_dollars_l51_51592

-- Define the concept of financial transactions
def denote_gain (amount : Int) : Int := amount
def denote_loss (amount : Int) : Int := -amount

-- The condition given in the problem
def earn_500_dollars_is_500 := denote_gain 500 = 500

-- The assertion we need to prove
theorem denote_loss_of_300_dollars : denote_loss 300 = -300 := 
by 
  sorry

end denote_loss_of_300_dollars_l51_51592


namespace katie_earnings_l51_51780

theorem katie_earnings 
  (bead_necklaces : ℕ)
  (gem_stone_necklaces : ℕ)
  (bead_cost : ℕ)
  (gem_stone_cost : ℕ)
  (h1 : bead_necklaces = 4)
  (h2 : gem_stone_necklaces = 3)
  (h3 : bead_cost = 5)
  (h4 : gem_stone_cost = 8) :
  (bead_necklaces * bead_cost + gem_stone_necklaces * gem_stone_cost = 44) :=
by
  sorry

end katie_earnings_l51_51780


namespace minimum_value_occurs_at_4_l51_51549

noncomputable def minimum_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, f x ≤ f y

def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 15

theorem minimum_value_occurs_at_4 :
  minimum_value_at quadratic_expression 4 :=
sorry

end minimum_value_occurs_at_4_l51_51549


namespace exp_inequality_l51_51491

theorem exp_inequality (n : ℕ) (h : 0 < n) : 2 ≤ (1 + 1 / (n : ℝ)) ^ n ∧ (1 + 1 / (n : ℝ)) ^ n < 3 :=
sorry

end exp_inequality_l51_51491


namespace ring_matching_possible_iff_odd_l51_51527

theorem ring_matching_possible_iff_odd (n : ℕ) (hn : n ≥ 3) :
  (∃ f : ℕ → ℕ, (∀ k : ℕ, k < n → ∃ j : ℕ, j < n ∧ f (j + k) % n = k % n) ↔ Odd n) :=
sorry

end ring_matching_possible_iff_odd_l51_51527


namespace bobs_improvement_percentage_l51_51395

-- Define the conditions
def bobs_time_minutes := 10
def bobs_time_seconds := 40
def sisters_time_minutes := 10
def sisters_time_seconds := 8

-- Convert minutes and seconds to total seconds
def bobs_total_time_seconds := bobs_time_minutes * 60 + bobs_time_seconds
def sisters_total_time_seconds := sisters_time_minutes * 60 + sisters_time_seconds

-- Define the improvement needed and calculate the percentage improvement
def improvement_needed := bobs_total_time_seconds - sisters_total_time_seconds
def percentage_improvement := (improvement_needed / bobs_total_time_seconds) * 100

-- The lean statement to prove
theorem bobs_improvement_percentage : percentage_improvement = 5 := by
  sorry

end bobs_improvement_percentage_l51_51395


namespace trigonometric_solution_l51_51668

theorem trigonometric_solution (x : Real) :
  (2 * Real.sin x * Real.cos (3 * Real.pi / 2 + x) 
  - 3 * Real.sin (Real.pi - x) * Real.cos x 
  + Real.sin (Real.pi / 2 + x) * Real.cos x = 0) ↔ 
  (∃ k : Int, x = Real.arctan ((3 + Real.sqrt 17) / -4) + k * Real.pi) ∨ 
  (∃ n : Int, x = Real.arctan ((3 - Real.sqrt 17) / -4) + n * Real.pi) :=
sorry

end trigonometric_solution_l51_51668


namespace system_of_inequalities_solution_l51_51035

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end system_of_inequalities_solution_l51_51035


namespace arithmetic_sequence_sum_l51_51774

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (h_arith : is_arithmetic_sequence a)
  (h_condition : a 2 + a 6 = 37) : 
  a 1 + a 3 + a 5 + a 7 = 74 :=
  sorry

end arithmetic_sequence_sum_l51_51774


namespace problem_solution_l51_51056

theorem problem_solution :
  ∃ (b₂ b₃ b₄ b₅ b₆ b₇ : ℤ),
    (0 ≤ b₂ ∧ b₂ < 2) ∧
    (0 ≤ b₃ ∧ b₃ < 3) ∧
    (0 ≤ b₄ ∧ b₄ < 4) ∧
    (0 ≤ b₅ ∧ b₅ < 5) ∧
    (0 ≤ b₆ ∧ b₆ < 6) ∧
    (0 ≤ b₇ ∧ b₇ < 8) ∧
    (6 / 7 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040) ∧
    (b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11) :=
sorry

end problem_solution_l51_51056


namespace merchant_profit_percentage_l51_51258

noncomputable def cost_price : ℝ := 100
noncomputable def marked_up_price : ℝ := cost_price + (0.75 * cost_price)
noncomputable def discount : ℝ := 0.30 * marked_up_price
noncomputable def selling_price : ℝ := marked_up_price - discount
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem merchant_profit_percentage :
  profit_percentage = 22.5 :=
by
  sorry

end merchant_profit_percentage_l51_51258


namespace find_value_in_table_l51_51382

theorem find_value_in_table :
  let W := 'W'
  let L := 'L'
  let Q := 'Q'
  let table := [
    [W, '?', Q],
    [L, Q, W],
    [Q, W, L]
  ]
  table[0][1] = L :=
by
  sorry

end find_value_in_table_l51_51382


namespace system_solution_l51_51354
-- importing the Mathlib library

-- define the problem with necessary conditions
theorem system_solution (x y : ℝ → ℝ) (x0 y0 : ℝ) 
    (h1 : ∀ t, deriv x t = y t) 
    (h2 : ∀ t, deriv y t = -x t) 
    (h3 : x 0 = x0)
    (h4 : y 0 = y0):
    (∀ t, x t = x0 * Real.cos t + y0 * Real.sin t) ∧ (∀ t, y t = -x0 * Real.sin t + y0 * Real.cos t) ∧ (∀ t, (x t)^2 + (y t)^2 = x0^2 + y0^2) := 
by 
    sorry

end system_solution_l51_51354


namespace dawn_monthly_savings_l51_51547

variable (annual_income : ℕ)
variable (months : ℕ)
variable (tax_deduction_percent : ℚ)
variable (variable_expense_percent : ℚ)
variable (savings_percent : ℚ)

def calculate_monthly_savings (annual_income months : ℕ) 
    (tax_deduction_percent variable_expense_percent savings_percent : ℚ) : ℚ :=
  let monthly_income := (annual_income : ℚ) / months;
  let after_tax_income := monthly_income * (1 - tax_deduction_percent);
  let after_expenses_income := after_tax_income * (1 - variable_expense_percent);
  after_expenses_income * savings_percent

theorem dawn_monthly_savings : 
    calculate_monthly_savings 48000 12 0.20 0.30 0.10 = 224 := 
  by 
    sorry

end dawn_monthly_savings_l51_51547


namespace count_three_digit_numbers_divisible_by_13_l51_51086

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end count_three_digit_numbers_divisible_by_13_l51_51086


namespace cost_per_package_l51_51344

theorem cost_per_package
  (parents : ℕ)
  (brothers : ℕ)
  (spouses_per_brother : ℕ)
  (children_per_brother : ℕ)
  (total_cost : ℕ)
  (num_packages : ℕ)
  (h1 : parents = 2)
  (h2 : brothers = 3)
  (h3 : spouses_per_brother = 1)
  (h4 : children_per_brother = 2)
  (h5 : total_cost = 70)
  (h6 : num_packages = parents + brothers + brothers * spouses_per_brother + brothers * children_per_brother) :
  total_cost / num_packages = 5 :=
by
  -- Proof goes here
  sorry

end cost_per_package_l51_51344


namespace find_k_l51_51593

theorem find_k (k a : ℤ)
  (h₁ : 49 + k = a^2)
  (h₂ : 361 + k = (a + 2)^2)
  (h₃ : 784 + k = (a + 4)^2) :
  k = 6035 :=
by sorry

end find_k_l51_51593


namespace num_factors_180_l51_51929

-- Conditions: The prime factorization of 180
def fact180 : ℕ := 180
def fact180_prime_decomp : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

-- Definition of counting the number of factors from prime factorization
def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (fun acc p => acc * (p.snd + 1)) 1

-- Theorem statement: The number of positive factors of 180 is 18 
theorem num_factors_180 : number_of_factors fact180_prime_decomp = 18 := 
by
  sorry

end num_factors_180_l51_51929


namespace card_probability_l51_51657

theorem card_probability :
  let totalCards := 52
  let kings := 4
  let jacks := 4
  let queens := 4
  let firstCardKing := kings / totalCards
  let secondCardJack := jacks / (totalCards - 1)
  let thirdCardQueen := queens / (totalCards - 2)
  (firstCardKing * secondCardJack * thirdCardQueen) = (8 / 16575) :=
by
  sorry

end card_probability_l51_51657


namespace min_value_expression_l51_51314

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (H : 1 / a + 1 / b = 1) :
  ∃ c : ℝ, (∀ (a b : ℝ), 0 < a → 0 < b → 1 / a + 1 / b = 1 → c ≤ 4 / (a - 1) + 9 / (b - 1)) ∧ (c = 6) :=
by
  sorry

end min_value_expression_l51_51314


namespace problem_1_problem_2_problem_3_l51_51055

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5 / 2}

theorem problem_1 : A ∩ B = {x | -1 < x ∧ x < 2} := sorry

theorem problem_2 : compl B ∪ P = {x | x ≤ 0 ∨ x ≥ 5 / 2} := sorry

theorem problem_3 : (A ∩ B) ∩ compl P = {x | 0 < x ∧ x < 2} := sorry

end problem_1_problem_2_problem_3_l51_51055


namespace american_summits_more_water_l51_51041

-- Definitions based on the conditions
def FosterFarmsChickens := 45
def AmericanSummitsWater := 2 * FosterFarmsChickens
def HormelChickens := 3 * FosterFarmsChickens
def BoudinButchersChickens := HormelChickens / 3
def TotalItems := 375
def ItemsByFourCompanies := FosterFarmsChickens + AmericanSummitsWater + HormelChickens + BoudinButchersChickens
def DelMonteWater := TotalItems - ItemsByFourCompanies
def WaterDifference := AmericanSummitsWater - DelMonteWater

theorem american_summits_more_water : WaterDifference = 30 := by
  sorry

end american_summits_more_water_l51_51041


namespace third_side_of_triangle_l51_51346

theorem third_side_of_triangle
  (A B C : Type)
  (a b c : ℝ)
  (ha : a = 6)
  (hb : b = 18)
  (angle_B angle_C : ℝ)
  (hangle : angle_B = 3 * angle_C)
  (cos_C : ℝ)
  (hcos_C : cos_C = Real.sqrt(2 / 3))
  (sin_C : ℝ)
  (hsin_C : sin_C = Real.sqrt(3) / 3)
  (hcos_eq : cos_C = (a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)) :
  c = 33 * Real.sqrt(6) :=
sorry

end third_side_of_triangle_l51_51346


namespace count_three_digit_div_by_13_l51_51105

def smallest_three_digit : ℤ := 100
def largest_three_digit : ℤ := 999
def divisor : ℤ := 13

theorem count_three_digit_div_by_13 : 
  let count := ((largest_three_digit / divisor) - 
                (smallest_three_digit + divisor - 1) / divisor + 1) in 
  count = 69 := by sorry

end count_three_digit_div_by_13_l51_51105


namespace three_digit_numbers_divisible_by_13_count_l51_51131

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l51_51131


namespace colleague_typing_time_l51_51203

theorem colleague_typing_time (T : ℝ) : 
  (∀ me_time : ℝ, (me_time = 180) →
  (∀ my_speed my_colleague_speed : ℝ, (my_speed = T / me_time) →
  (my_colleague_speed = 4 * my_speed) →
  (T / my_colleague_speed = 45))) :=
  sorry

end colleague_typing_time_l51_51203


namespace number_of_factors_180_l51_51921

theorem number_of_factors_180 : 
  let factors180 (n : ℕ) := 180 
  in factors180 180 = (2 + 1) * (2 + 1) * (1 + 1) => factors180 180 = 18 := 
by
  sorry

end number_of_factors_180_l51_51921


namespace graph_through_origin_y_increases_with_x_not_pass_third_quadrant_l51_51439

-- Part 1: Prove that if the graph passes through the origin, then m ≠ 2/3 and n = 1
theorem graph_through_origin {m n : ℝ} : 
  (3 * m - 2 ≠ 0) → (1 - n = 0) ↔ (m ≠ 2/3 ∧ n = 1) :=
by sorry

-- Part 2: Prove that if y increases as x increases, then m > 2/3 and n is any real number
theorem y_increases_with_x {m n : ℝ} : 
  (3 * m - 2 > 0) ↔ (m > 2/3 ∧ ∀ n : ℝ, True) :=
by sorry

-- Part 3: Prove that if the graph does not pass through the third quadrant, then m < 2/3 and n ≤ 1
theorem not_pass_third_quadrant {m n : ℝ} : 
  (3 * m - 2 < 0) ∧ (1 - n ≥ 0) ↔ (m < 2/3 ∧ n ≤ 1) :=
by sorry

end graph_through_origin_y_increases_with_x_not_pass_third_quadrant_l51_51439


namespace count_three_digit_numbers_divisible_by_13_l51_51153

-- Definition of a 3-digit number
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Definition of divisibility by 13
def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

-- Definition of a 3-digit number divisible by 13
def is_valid_number_13 (n : ℕ) : Prop :=
  is_three_digit n ∧ is_divisible_by_13 n

-- The theorem stating the problem and the condition
theorem count_three_digit_numbers_divisible_by_13 : 
  {n : ℕ | is_valid_number_13 n}.to_finset.card = 69 :=
sorry

end count_three_digit_numbers_divisible_by_13_l51_51153


namespace popsicles_eaten_l51_51487

theorem popsicles_eaten (total_time : ℕ) (interval : ℕ) (p : ℕ)
  (h_total_time : total_time = 6 * 60)
  (h_interval : interval = 20) :
  p = total_time / interval :=
sorry

end popsicles_eaten_l51_51487


namespace proof_problem_l51_51442

open Matrix

variables (v u : Fin 3 → ℝ)

def cross_product (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![
    a 1 * b 2 - a 2 * b 1,
    a 2 * b 0 - a 0 * b 2,
    a 0 * b 1 - a 1 * b 0
  ]

def scalar_mult (c : ℝ) (a : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![
    c * a 0,
    c * a 1,
    c * a 2
  ]

def vector_add (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![
    a 0 + b 0,
    a 1 + b 1,
    a 2 + b 2
  ]

def vector_sub (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![
    a 0 - b 0,
    a 1 - b 1,
    a 2 - b 2
  ]

variables (cross_vu : Fin 3 → ℝ)
variables (w : Fin 3 → ℝ)

-- Given conditions
def given_conditions : Prop :=
  cross_product v u = cross_vu ∧ w = vector_add (scalar_mult 2 v) (scalar_mult 3 u)

-- Proof statement
theorem proof_problem (h : given_conditions v u cross_vu w) :
  cross_product (vector_sub (scalar_mult 2 v) u) w = ![24, -8, 16] :=
by
  sorry

end proof_problem_l51_51442


namespace calculate_difference_l51_51830

theorem calculate_difference :
  let a := 3.56
  let b := 2.1
  let c := 1.5
  a - (b * c) = 0.41 :=
by
  let a := 3.56
  let b := 2.1
  let c := 1.5
  show a - (b * c) = 0.41
  sorry

end calculate_difference_l51_51830


namespace specified_time_correct_l51_51409

theorem specified_time_correct (x : ℝ) (h1 : 900.0 = dist) (h2 : slow_time = x + 1) 
  (h3 : fast_time = x - 3) (h4 : fast_speed = 2 * slow_speed) 
  (dist : ℝ := 900.0) (slow_speed : ℝ := 900.0 / (x + 1)) (fast_speed : ℝ := 900.0 / (x - 3)) 
  (slow_time fast_time : ℝ) :
  2 * slow_speed = fast_speed :=
by
  sorry

end specified_time_correct_l51_51409
