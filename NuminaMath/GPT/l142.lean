import Mathlib

namespace exists_larger_integer_l142_142461

theorem exists_larger_integer (a b : Nat) (h1 : b > a) (h2 : b - a = 5) (h3 : a * b = 88) :
  b = 11 :=
sorry

end exists_larger_integer_l142_142461


namespace minimum_value_frac_l142_142338

theorem minimum_value_frac (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p + q + r = 2) :
  (p + q) / (p * q * r) ≥ 9 :=
sorry

end minimum_value_frac_l142_142338


namespace second_bucket_capacity_l142_142693

-- Define the initial conditions as given in the problem.
def tank_capacity : ℕ := 48
def bucket1_capacity : ℕ := 4

-- Define the number of times the 4-liter bucket is used.
def bucket1_uses : ℕ := tank_capacity / bucket1_capacity

-- Define a condition related to bucket uses.
def buckets_use_relation (x : ℕ) : Prop :=
  bucket1_uses = (tank_capacity / x) - 4

-- Formulate the theorem that states the capacity of the second bucket.
theorem second_bucket_capacity (x : ℕ) (h : buckets_use_relation x) : x = 3 :=
by {
  sorry
}

end second_bucket_capacity_l142_142693


namespace cone_base_radius_l142_142312

theorem cone_base_radius (R : ℝ) (theta : ℝ) (radius : ℝ) (hR : R = 30) (hTheta : theta = 120) :
    2 * Real.pi * radius = (theta / 360) * 2 * Real.pi * R → radius = 10 :=
by
  intros h
  sorry

end cone_base_radius_l142_142312


namespace interest_rate_is_10_perc_l142_142677

noncomputable def interest_rate (P : ℝ) (R : ℝ) (T : ℝ := 2) : ℝ := (P * R * T) / 100

theorem interest_rate_is_10_perc (P : ℝ) : 
  (interest_rate P 10) = P / 5 :=
by
  sorry

end interest_rate_is_10_perc_l142_142677


namespace volume_uncovered_is_correct_l142_142060

-- Define the volumes of the shoebox and the objects
def volume_shoebox : ℕ := 12 * 6 * 4
def volume_object1 : ℕ := 5 * 3 * 1
def volume_object2 : ℕ := 2 * 2 * 3
def volume_object3 : ℕ := 4 * 2 * 4

-- Define the total volume of the objects
def total_volume_objects : ℕ := volume_object1 + volume_object2 + volume_object3

-- Define the volume left uncovered
def volume_uncovered : ℕ := volume_shoebox - total_volume_objects

-- Prove that the volume left uncovered is 229 cubic inches
theorem volume_uncovered_is_correct : volume_uncovered = 229 := by
  -- This is where the proof would be written
  sorry

end volume_uncovered_is_correct_l142_142060


namespace acute_angle_inclination_range_l142_142735

/-- 
For the line passing through points P(1-a, 1+a) and Q(3, 2a), 
prove that the range of the real number a such that the line has an acute angle of inclination is (-∞, 1) ∪ (1, 4).
-/
theorem acute_angle_inclination_range (a : ℝ) : 
  (a < 1 ∨ (1 < a ∧ a < 4)) ↔ (0 < (a - 1) / (4 - a)) :=
sorry

end acute_angle_inclination_range_l142_142735


namespace part1_part2_part3_l142_142065

noncomputable def y1 (x : ℝ) : ℝ := 0.1 * x + 15
noncomputable def y2 (x : ℝ) : ℝ := 0.15 * x

-- Prove that the functions are as described
theorem part1 : ∀ x : ℝ, y1 x = 0.1 * x + 15 ∧ y2 x = 0.15 * x :=
by sorry

-- Prove that x = 300 results in equal charges for Packages A and B
theorem part2 : y1 300 = y2 300 :=
by sorry

-- Prove that Package A is more cost-effective when x > 300
theorem part3 : ∀ x : ℝ, x > 300 → y1 x < y2 x :=
by sorry

end part1_part2_part3_l142_142065


namespace value_of_f3_f10_l142_142209

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic_condition (x : ℝ) : f (x + 4) = f x + f 2
axiom f_at_one : f 1 = 4

theorem value_of_f3_f10 : f 3 + f 10 = 4 := sorry

end value_of_f3_f10_l142_142209


namespace remainder_13_plus_y_l142_142015

theorem remainder_13_plus_y :
  (∃ y : ℕ, (0 < y) ∧ (7 * y ≡ 1 [MOD 31])) → (∃ y : ℕ, (13 + y ≡ 22 [MOD 31])) :=
by 
  sorry

end remainder_13_plus_y_l142_142015


namespace sequence_n_5_l142_142669

theorem sequence_n_5 (a : ℤ) (n : ℕ → ℤ) 
  (h1 : ∀ i > 1, n i = 2 * n (i - 1) + a)
  (h2 : n 2 = 5)
  (h3 : n 8 = 257) : n 5 = 33 :=
by
  sorry

end sequence_n_5_l142_142669


namespace root_in_interval_l142_142207

noncomputable def f (x : ℝ) := Real.log x + x - 2

theorem root_in_interval : ∃ c ∈ Set.Ioo 1 2, f c = 0 := 
sorry

end root_in_interval_l142_142207


namespace sum_fifth_powers_l142_142491

theorem sum_fifth_powers (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^5 + b^5 + c^5 = 98 / 6 := 
by 
  sorry

end sum_fifth_powers_l142_142491


namespace line_equation_through_point_parallel_to_lines_l142_142415

theorem line_equation_through_point_parallel_to_lines (L L1 L2 : ℝ → ℝ → Prop) :
  (∀ x, L1 x (y: ℝ) ↔ 3 * x + y - 6 = 0) →
  (∀ x, L2 x (y: ℝ) ↔ 3 * x + y + 3 = 0) →
  (L 1 0) →
  (∀ x1 y1 x2 y2, L1 x1 y1 → L1 x2 y2 → (y2 - y1) / (x2 - x1) = -3) →
  ∃ A B C, (A = 1 ∧ B = -3 ∧ C = -3) ∧ (∀ x y, L x y ↔ A * x + B * y + C = 0) :=
by sorry

end line_equation_through_point_parallel_to_lines_l142_142415


namespace value_of_expression_when_x_eq_4_l142_142048

theorem value_of_expression_when_x_eq_4 : (3 * 4 + 4)^2 = 256 := by
  sorry

end value_of_expression_when_x_eq_4_l142_142048


namespace prob_of_drawing_one_red_ball_distribution_of_X_l142_142794

-- Definitions for conditions
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def total_balls : ℕ := red_balls + white_balls
def balls_drawn : ℕ := 3

-- Combinations 
noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Probabilities
noncomputable def prob_ex_one_red_ball : ℚ :=
  (combination red_balls 1 * combination white_balls 2) / combination total_balls balls_drawn

noncomputable def prob_X_0 : ℚ := (combination white_balls 3) / combination total_balls balls_drawn
noncomputable def prob_X_1 : ℚ := prob_ex_one_red_ball
noncomputable def prob_X_2 : ℚ := (combination red_balls 2 * combination white_balls 1) / combination total_balls balls_drawn

-- Theorem statements
theorem prob_of_drawing_one_red_ball : prob_ex_one_red_ball = 3/5 := by
  sorry

theorem distribution_of_X : prob_X_0 = 1/10 ∧ prob_X_1 = 3/5 ∧ prob_X_2 = 3/10 := by
  sorry

end prob_of_drawing_one_red_ball_distribution_of_X_l142_142794


namespace find_c_l142_142725

theorem find_c (x c : ℤ) (h1 : 3 * x + 9 = 0) (h2 : c * x - 5 = -11) : c = 2 := by
  have x_eq : x = -3 := by
    linarith
  subst x_eq
  have c_eq : c = 2 := by
    linarith
  exact c_eq

end find_c_l142_142725


namespace find_a_l142_142803

theorem find_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 20) 
  (h3 : (4254253 % 53^1 - a) % 17 = 0): 
  a = 3 := 
sorry

end find_a_l142_142803


namespace intersection_A_B_l142_142508

def A (x : ℝ) : Prop := x^2 - 3 * x < 0
def B (x : ℝ) : Prop := x > 2

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l142_142508


namespace total_players_ground_l142_142464

-- Define the number of players for each type of sport
def c : ℕ := 10
def h : ℕ := 12
def f : ℕ := 16
def s : ℕ := 13

-- Statement of the problem to prove that the total number of players is 51
theorem total_players_ground : c + h + f + s = 51 :=
by
  -- proof will be added later
  sorry

end total_players_ground_l142_142464


namespace sequence_general_formula_l142_142520

theorem sequence_general_formula {a : ℕ → ℕ} 
  (h₁ : a 1 = 2) 
  (h₂ : ∀ n : ℕ, a (n + 1) = 2 * a n + 3 * 5 ^ n) 
  : ∀ n : ℕ, a n = 5 ^ n - 3 * 2 ^ (n - 1) :=
sorry

end sequence_general_formula_l142_142520


namespace can_capacity_l142_142523

theorem can_capacity (x : ℝ) (milk water : ℝ) (full_capacity : ℝ) : 
  5 * x = milk ∧ 
  3 * x = water ∧ 
  full_capacity = milk + water + 8 ∧ 
  (milk + 8) / water = 2 → 
  full_capacity = 72 := 
sorry

end can_capacity_l142_142523


namespace harkamal_payment_l142_142437

noncomputable def calculate_total_cost : ℝ :=
  let price_grapes := 8 * 70
  let price_mangoes := 9 * 45
  let price_apples := 5 * 30
  let price_strawberries := 3 * 100
  let price_oranges := 10 * 40
  let price_kiwis := 6 * 60
  let total_grapes_and_apples := price_grapes + price_apples
  let discount_grapes_and_apples := 0.10 * total_grapes_and_apples
  let total_oranges_and_kiwis := price_oranges + price_kiwis
  let discount_oranges_and_kiwis := 0.05 * total_oranges_and_kiwis
  let total_mangoes_and_strawberries := price_mangoes + price_strawberries
  let tax_mangoes_and_strawberries := 0.12 * total_mangoes_and_strawberries
  let total_amount := price_grapes + price_mangoes + price_apples + price_strawberries + price_oranges + price_kiwis
  total_amount - discount_grapes_and_apples - discount_oranges_and_kiwis + tax_mangoes_and_strawberries

theorem harkamal_payment : calculate_total_cost = 2150.6 :=
by
  sorry

end harkamal_payment_l142_142437


namespace num_mystery_shelves_l142_142325

def num_books_per_shelf : ℕ := 9
def num_picture_shelves : ℕ := 2
def total_books : ℕ := 72
def num_books_from_picture_shelves : ℕ := num_picture_shelves * num_books_per_shelf
def num_books_from_mystery_shelves : ℕ := total_books - num_books_from_picture_shelves

theorem num_mystery_shelves :
  num_books_from_mystery_shelves / num_books_per_shelf = 6 := by
sorry

end num_mystery_shelves_l142_142325


namespace eight_girls_circle_least_distance_l142_142224

theorem eight_girls_circle_least_distance :
  let r := 50
  let num_girls := 8
  let total_distance := (8 * (3 * (r * Real.sqrt 2) + 2 * (2 * r)))
  total_distance = 1200 * Real.sqrt 2 + 1600 :=
by
  sorry

end eight_girls_circle_least_distance_l142_142224


namespace expression_bounds_l142_142331

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) (hw : 0 ≤ w) (hw1 : w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by
  sorry

end expression_bounds_l142_142331


namespace system_a_l142_142946

theorem system_a (x y z : ℝ) (h1 : x + y + z = 6) (h2 : 1/x + 1/y + 1/z = 11/6) (h3 : x*y + y*z + z*x = 11) :
  x = 1 ∧ y = 2 ∧ z = 3 ∨ x = 1 ∧ y = 3 ∧ z = 2 ∨ x = 2 ∧ y = 1 ∧ z = 3 ∨ x = 2 ∧ y = 3 ∧ z = 1 ∨ x = 3 ∧ y = 1 ∧ z = 2 ∨ x = 3 ∧ y = 2 ∧ z = 1 :=
sorry

end system_a_l142_142946


namespace simplify_expression_l142_142640

variable (a b : ℤ)

theorem simplify_expression : 
  (50 * a + 130 * b) + (21 * a + 64 * b) - (30 * a + 115 * b) - 2 * (10 * a - 25 * b) = 21 * a + 129 * b := 
by
  sorry

end simplify_expression_l142_142640


namespace volume_of_pyramid_l142_142307

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

end volume_of_pyramid_l142_142307


namespace additional_days_when_selling_5_goats_l142_142764

variables (G D F X : ℕ)

def total_feed (num_goats days : ℕ) := G * num_goats * days

theorem additional_days_when_selling_5_goats
  (h1 : total_feed G 20 D = F)
  (h2 : total_feed G 15 (D + X) = F)
  (h3 : total_feed G 30 (D - 3) = F):
  X = 9 :=
by
  -- the exact proof is omitted and presented as 'sorry'
  sorry

end additional_days_when_selling_5_goats_l142_142764


namespace ratio_of_areas_l142_142718

theorem ratio_of_areas 
  (lenA : ℕ) (brdA : ℕ) (lenB : ℕ) (brdB : ℕ)
  (h_lenA : lenA = 48) 
  (h_brdA : brdA = 30)
  (h_lenB : lenB = 60) 
  (h_brdB : brdB = 35) :
  (lenA * brdA : ℚ) / (lenB * brdB) = 24 / 35 :=
by
  sorry

end ratio_of_areas_l142_142718


namespace john_receives_more_l142_142821

noncomputable def partnership_difference (investment_john : ℝ) (investment_mike : ℝ) (profit : ℝ) : ℝ :=
  let total_investment := investment_john + investment_mike
  let one_third_profit := profit / 3
  let two_third_profit := 2 * profit / 3
  let john_effort_share := one_third_profit / 2
  let mike_effort_share := one_third_profit / 2
  let ratio_john := investment_john / total_investment
  let ratio_mike := investment_mike / total_investment
  let john_investment_share := ratio_john * two_third_profit
  let mike_investment_share := ratio_mike * two_third_profit
  let john_total := john_effort_share + john_investment_share
  let mike_total := mike_effort_share + mike_investment_share
  john_total - mike_total

theorem john_receives_more (investment_john investment_mike profit : ℝ)
  (h_john : investment_john = 700)
  (h_mike : investment_mike = 300)
  (h_profit : profit = 3000.0000000000005) :
  partnership_difference investment_john investment_mike profit = 800.0000000000001 := 
sorry

end john_receives_more_l142_142821


namespace PlatformC_location_l142_142551

noncomputable def PlatformA : ℝ := 9
noncomputable def PlatformB : ℝ := 1 / 3
noncomputable def PlatformC : ℝ := 7
noncomputable def AB := PlatformA - PlatformB
noncomputable def AC := PlatformA - PlatformC

theorem PlatformC_location :
  AB = (13 / 3) * AC → PlatformC = 7 :=
by
  intro h
  simp [AB, AC, PlatformA, PlatformB, PlatformC] at h
  sorry

end PlatformC_location_l142_142551


namespace quadratic_function_value_l142_142310

theorem quadratic_function_value
  (p q r : ℝ)
  (h1 : p + q + r = 3)
  (h2 : 4 * p + 2 * q + r = 12) :
  p + q + 3 * r = -5 :=
by
  sorry

end quadratic_function_value_l142_142310


namespace Norbs_age_l142_142175

def guesses : List ℕ := [24, 28, 30, 32, 36, 38, 41, 44, 47, 49]

def is_prime (n : ℕ) : Prop := Nat.Prime n

def two_off_by_one (n : ℕ) (guesses : List ℕ) : Prop := 
  (n - 1 ∈ guesses) ∧ (n + 1 ∈ guesses)

def at_least_half_too_low (n : ℕ) (guesses : List ℕ) : Prop := 
  (guesses.filter (· < n)).length ≥ guesses.length / 2

theorem Norbs_age : 
  ∃ x, is_prime x ∧ two_off_by_one x guesses ∧ at_least_half_too_low x guesses ∧ x = 37 := 
by 
  sorry

end Norbs_age_l142_142175


namespace center_of_symmetry_l142_142416

theorem center_of_symmetry (k : ℤ) : ∀ (k : ℤ), ∃ x : ℝ, 
  (x = (k * Real.pi / 6 - Real.pi / 9) ∨ x = - (Real.pi / 18)) → False :=
by
  sorry

end center_of_symmetry_l142_142416


namespace can_use_bisection_method_l142_142192

noncomputable def f1 (x : ℝ) : ℝ := x^2
noncomputable def f2 (x : ℝ) : ℝ := x⁻¹
noncomputable def f3 (x : ℝ) : ℝ := abs x
noncomputable def f4 (x : ℝ) : ℝ := x^3

theorem can_use_bisection_method : ∃ (a b : ℝ), a < b ∧ (f4 a) * (f4 b) < 0 := 
sorry

end can_use_bisection_method_l142_142192


namespace polynomial_expansion_proof_l142_142636

variable (z : ℤ)

-- Define the polynomials p and q
noncomputable def p (z : ℤ) : ℤ := 3 * z^2 - 4 * z + 1
noncomputable def q (z : ℤ) : ℤ := 2 * z^3 + 3 * z^2 - 5 * z + 2

-- Define the expanded polynomial
noncomputable def expanded (z : ℤ) : ℤ :=
  6 * z^5 + z^4 - 25 * z^3 + 29 * z^2 - 13 * z + 2

-- The goal is to prove the equivalence of (p * q) == expanded 
theorem polynomial_expansion_proof :
  (p z) * (q z) = expanded z :=
by
  sorry

end polynomial_expansion_proof_l142_142636


namespace water_formed_l142_142686

theorem water_formed (n_HCl : ℕ) (n_CaCO3: ℕ) (n_H2O: ℕ) 
  (balance_eqn: ∀ (n : ℕ), 
    (2 * n_CaCO3) ≤ n_HCl ∧
    n_H2O = n_CaCO3 ):
  n_HCl = 4 ∧ n_CaCO3 = 2 → n_H2O = 2 :=
by
  intros h0
  obtain ⟨h1, h2⟩ := h0
  sorry

end water_formed_l142_142686


namespace quad_inequality_necessary_but_not_sufficient_l142_142646

def quad_inequality (x : ℝ) : Prop := x^2 - x - 6 > 0
def less_than_negative_five (x : ℝ) : Prop := x < -5

theorem quad_inequality_necessary_but_not_sufficient :
  (∀ x : ℝ, less_than_negative_five x → quad_inequality x) ∧ 
  (∃ x : ℝ, quad_inequality x ∧ ¬ less_than_negative_five x) :=
by
  sorry

end quad_inequality_necessary_but_not_sufficient_l142_142646


namespace lemonade_glasses_l142_142424

def lemons_total : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_glasses : lemons_total / lemons_per_glass = 9 := by
  sorry

end lemonade_glasses_l142_142424


namespace base_conversion_l142_142817

def baseThreeToBaseTen (n : List ℕ) : ℕ :=
  n.reverse.enumFrom 0 |>.map (λ ⟨i, d⟩ => d * 3^i) |>.sum

def baseTenToBaseFive (n : ℕ) : List ℕ :=
  let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else aux (n / 5) ((n % 5) :: acc)
  aux n []

theorem base_conversion (baseThreeNum : List ℕ) (baseTenNum : ℕ) (baseFiveNum : List ℕ) :
  baseThreeNum = [2, 0, 1, 2, 1] →
  baseTenNum = 178 →
  baseFiveNum = [1, 2, 0, 3] →
  baseThreeToBaseTen baseThreeNum = baseTenNum ∧ baseTenToBaseFive baseTenNum = baseFiveNum :=
by
  intros h1 h2 h3
  unfold baseThreeToBaseTen
  unfold baseTenToBaseFive
  sorry

end base_conversion_l142_142817


namespace snail_returns_l142_142497

noncomputable def snail_path : Type := ℕ → ℝ × ℝ

def snail_condition (snail : snail_path) (speed : ℝ) : Prop :=
  ∀ n : ℕ, n % 4 = 0 → snail (n + 4) = snail n

theorem snail_returns (snail : snail_path) (speed : ℝ) (h1 : ∀ n m : ℕ, n ≠ m → snail n ≠ snail m)
    (h2 : snail_condition snail speed) :
  ∃ t : ℕ, t > 0 ∧ t % 4 = 0 ∧ snail t = snail 0 := 
sorry

end snail_returns_l142_142497


namespace total_oranges_is_correct_l142_142572

/-- Define the number of boxes and the number of oranges per box -/
def boxes : ℕ := 7
def oranges_per_box : ℕ := 6

/-- Prove that the total number of oranges is 42 -/
theorem total_oranges_is_correct : boxes * oranges_per_box = 42 := 
by 
  sorry

end total_oranges_is_correct_l142_142572


namespace student_avg_greater_actual_avg_l142_142587

theorem student_avg_greater_actual_avg
  (x y z : ℝ)
  (hxy : x < y)
  (hyz : y < z) :
  (x + y + 2 * z) / 4 > (x + y + z) / 3 := by
  sorry

end student_avg_greater_actual_avg_l142_142587


namespace circle_radius_of_square_perimeter_eq_area_l142_142937

theorem circle_radius_of_square_perimeter_eq_area (r : ℝ) (s : ℝ) (h1 : 2 * r = s) (h2 : 4 * s = 8 * r) (h3 : π * r ^ 2 = 8 * r) : r = 8 / π := by
  sorry

end circle_radius_of_square_perimeter_eq_area_l142_142937


namespace subscription_difference_is_4000_l142_142873

-- Given definitions
def total_subscription (A B C : ℕ) : Prop :=
  A + B + C = 50000

def subscription_B (x : ℕ) : ℕ :=
  x + 5000

def subscription_A (x y : ℕ) : ℕ :=
  x + 5000 + y

def profit_ratio (profit_C total_profit x : ℕ) : Prop :=
  (profit_C : ℚ) / total_profit = (x : ℚ) / 50000

-- Prove that A subscribed Rs. 4,000 more than B
theorem subscription_difference_is_4000 (x y : ℕ)
  (h1 : total_subscription (subscription_A x y) (subscription_B x) x)
  (h2 : profit_ratio 8400 35000 x) :
  y = 4000 :=
sorry

end subscription_difference_is_4000_l142_142873


namespace probability_no_shaded_rectangle_l142_142663

theorem probability_no_shaded_rectangle :
  let n := (1002 * 1001) / 2
  let m := 501 * 501
  (1 - (m / n) = 500 / 1001) := sorry

end probability_no_shaded_rectangle_l142_142663


namespace roots_of_quadratic_l142_142649

theorem roots_of_quadratic (x : ℝ) : 3 * (x - 3) = (x - 3) ^ 2 → x = 3 ∨ x = 6 :=
by
  intro h
  sorry

end roots_of_quadratic_l142_142649


namespace fraction_identity_l142_142387

theorem fraction_identity (a b : ℝ) (hb : b ≠ 0) (h : a / b = 3 / 2) : (a + b) / b = 2.5 :=
by
  sorry

end fraction_identity_l142_142387


namespace basketball_game_count_l142_142218

noncomputable def total_games_played (teams games_each_opp : ℕ) : ℕ :=
  (teams * (teams - 1) / 2) * games_each_opp

theorem basketball_game_count (n : ℕ) (g : ℕ) (h_n : n = 10) (h_g : g = 4) : total_games_played n g = 180 :=
by
  -- Use 'h_n' and 'h_g' as hypotheses
  rw [h_n, h_g]
  show (10 * 9 / 2) * 4 = 180
  sorry

end basketball_game_count_l142_142218


namespace cars_in_parking_lot_l142_142921

theorem cars_in_parking_lot (initial_cars left_cars entered_cars : ℕ) (h1 : initial_cars = 80)
(h2 : left_cars = 13) (h3 : entered_cars = left_cars + 5) : 
initial_cars - left_cars + entered_cars = 85 :=
by
  rw [h1, h2, h3]
  sorry

end cars_in_parking_lot_l142_142921


namespace fraction_of_students_participated_l142_142017

theorem fraction_of_students_participated (total_students : ℕ) (did_not_participate : ℕ)
  (h_total : total_students = 39) (h_did_not_participate : did_not_participate = 26) :
  (total_students - did_not_participate) / total_students = 1 / 3 :=
by
  sorry

end fraction_of_students_participated_l142_142017


namespace find_a₃_l142_142848

variable (a₁ a₂ a₃ a₄ a₅ : ℝ)
variable (S₅ : ℝ) (a_seq : ℕ → ℝ)

-- Define the conditions for arithmetic sequence and given sum
def is_arithmetic_sequence (a_seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_seq (n+1) - a_seq n = a_seq 1 - a_seq 0

axiom sum_first_five_terms (S₅ : ℝ) (hS : S₅ = 20) : 
  S₅ = (5 * (a₁ + a₅)) / 2

-- Main theorem we need to prove
theorem find_a₃ (hS₅ : S₅ = 20) (h_seq : is_arithmetic_sequence a_seq) :
  (∃ (a₃ : ℝ), a₃ = 4) :=
sorry

end find_a₃_l142_142848


namespace tara_additional_stamps_l142_142517

def stamps_needed (current_stamps total_stamps : Nat) : Nat :=
  if total_stamps % 9 == 0 then 0 else 9 - (total_stamps % 9)

theorem tara_additional_stamps :
  stamps_needed 38 45 = 7 := by
  sorry

end tara_additional_stamps_l142_142517


namespace frog_reaches_top_l142_142722

theorem frog_reaches_top (x : ℕ) (h1 : ∀ d ≤ x - 1, 3 * d + 5 ≥ 50) : x = 16 := by
  sorry

end frog_reaches_top_l142_142722


namespace angle_ratio_l142_142090

theorem angle_ratio (A B C : ℝ) (hA : A = 60) (hB : B = 80) (h_sum : A + B + C = 180) : B / C = 2 := by
  sorry

end angle_ratio_l142_142090


namespace total_sugar_in_all_candy_l142_142351

-- definitions based on the conditions
def chocolateBars : ℕ := 14
def sugarPerChocolateBar : ℕ := 10
def lollipopSugar : ℕ := 37

-- proof statement
theorem total_sugar_in_all_candy :
  (chocolateBars * sugarPerChocolateBar + lollipopSugar) = 177 := 
by
  sorry

end total_sugar_in_all_candy_l142_142351


namespace geometric_sequence_k_value_l142_142084

theorem geometric_sequence_k_value (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ)
  (hS : ∀ n, S n = k + 3^n)
  (h_geom : ∀ n, a (n+1) = S (n+1) - S n)
  (h_geo_seq : ∀ n, a (n+2) / a (n+1) = a (n+1) / a n) :
  k = -1 := by
  sorry

end geometric_sequence_k_value_l142_142084


namespace complex_root_problem_l142_142922

theorem complex_root_problem (z : ℂ) :
  z^2 - 3*z = 10 - 6*Complex.I ↔
  z = 5.5 - 0.75 * Complex.I ∨
  z = -2.5 + 0.75 * Complex.I ∨
  z = 3.5 - 1.5 * Complex.I ∨
  z = -0.5 + 1.5 * Complex.I :=
sorry

end complex_root_problem_l142_142922


namespace quadruples_solution_l142_142242

noncomputable
def valid_quadruples (x1 x2 x3 x4 : ℝ) : Prop :=
  (x1 + x2 * x3 * x4 = 2) ∧
  (x2 + x1 * x3 * x4 = 2) ∧
  (x3 + x1 * x2 * x4 = 2) ∧
  (x4 + x1 * x2 * x3 = 2) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0) ∧ (x3 ≠ 0) ∧ (x4 ≠ 0)

theorem quadruples_solution (x1 x2 x3 x4 : ℝ) :
  valid_quadruples x1 x2 x3 x4 ↔ 
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) := 
by sorry

end quadruples_solution_l142_142242


namespace ellipse_tangent_line_l142_142140

theorem ellipse_tangent_line (m : ℝ) : 
  (∀ (x y : ℝ), (x ^ 2 / 4) + (y ^ 2 / m) = 1 → (y = mx + 2)) → m = 1 :=
by sorry

end ellipse_tangent_line_l142_142140


namespace product_ab_l142_142530

theorem product_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end product_ab_l142_142530


namespace smallest_c_for_f_inverse_l142_142939

noncomputable def f (x : ℝ) : ℝ := (x - 3)^2 - 4

theorem smallest_c_for_f_inverse :
  ∃ c : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≥ c → x₂ ≥ c → f x₁ = f x₂ → x₁ = x₂) ∧ (∀ d : ℝ, d < c → ∃ x₁ x₂ : ℝ, x₁ ≥ d ∧ x₂ ≥ d ∧ f x₁ = f x₂ ∧ x₁ ≠ x₂) ∧ c = 3 :=
by
  sorry

end smallest_c_for_f_inverse_l142_142939


namespace arithmetic_geometric_mean_l142_142931

variable (x y : ℝ)

theorem arithmetic_geometric_mean (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) :
  x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_l142_142931


namespace manufacturing_section_degrees_l142_142131

variable (percentage_manufacturing : ℝ) (total_degrees : ℝ)

theorem manufacturing_section_degrees
  (h1 : percentage_manufacturing = 0.40)
  (h2 : total_degrees = 360) :
  percentage_manufacturing * total_degrees = 144 := 
by 
  sorry

end manufacturing_section_degrees_l142_142131


namespace effective_annual_rate_l142_142751

theorem effective_annual_rate (i : ℚ) (n : ℕ) (h_i : i = 0.16) (h_n : n = 2) :
  (1 + i / n) ^ n - 1 = 0.1664 :=
by {
  sorry
}

end effective_annual_rate_l142_142751


namespace stock_exchange_total_l142_142653

theorem stock_exchange_total (L H : ℕ) 
  (h1 : H = 1080) 
  (h2 : H = 6 * L / 5) : 
  (L + H = 1980) :=
by {
  -- L and H are given as natural numbers
  -- h1: H = 1080
  -- h2: H = 1.20L -> H = 6L/5 as Lean does not handle floating point well directly in integers.
  sorry
}

end stock_exchange_total_l142_142653


namespace point_slope_intersection_lines_l142_142098

theorem point_slope_intersection_lines : 
  ∀ s : ℝ, ∃ x y : ℝ, 2*x - 3*y = 8*s + 6 ∧ x + 2*y = 3*s - 1 ∧ y = -((2*x)/25 + 182/175) := 
sorry

end point_slope_intersection_lines_l142_142098


namespace initial_fish_count_l142_142820

theorem initial_fish_count (x : ℕ) (h1 : x + 47 = 69) : x = 22 :=
by
  sorry

end initial_fish_count_l142_142820


namespace am_gm_example_l142_142289

variable {x y z : ℝ}

theorem am_gm_example (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / y + y / z + z / x + y / x + z / y + x / z ≥ 6 :=
sorry

end am_gm_example_l142_142289


namespace remainder_seven_times_quotient_l142_142380

theorem remainder_seven_times_quotient (n : ℕ) : 
  (∃ q r : ℕ, n = 23 * q + r ∧ r = 7 * q ∧ 0 ≤ r ∧ r < 23) ↔ (n = 30 ∨ n = 60 ∨ n = 90) :=
by 
  sorry

end remainder_seven_times_quotient_l142_142380


namespace point_of_tangency_is_correct_l142_142368

theorem point_of_tangency_is_correct : 
  (∃ (x y : ℝ), y = x^2 + 20 * x + 63 ∧ x = y^2 + 56 * y + 875 ∧ x = -19 / 2 ∧ y = -55 / 2) :=
by
  sorry

end point_of_tangency_is_correct_l142_142368


namespace gcd_exponential_identity_l142_142141

theorem gcd_exponential_identity (a b : ℕ) :
  Nat.gcd (2^a - 1) (2^b - 1) = 2^(Nat.gcd a b) - 1 := sorry

end gcd_exponential_identity_l142_142141


namespace average_of_11_results_l142_142800

theorem average_of_11_results 
  (S1: ℝ) (S2: ℝ) (fifth_result: ℝ) -- Define the variables
  (h1: S1 / 5 = 49)                -- sum of the first 5 results
  (h2: S2 / 7 = 52)                -- sum of the last 7 results
  (h3: fifth_result = 147)         -- the fifth result 
  : (S1 + S2 - fifth_result) / 11 = 42 := -- statement of the problem
by
  sorry

end average_of_11_results_l142_142800


namespace solution_set_of_abs_x_minus_1_lt_1_l142_142885

theorem solution_set_of_abs_x_minus_1_lt_1 : {x : ℝ | |x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_abs_x_minus_1_lt_1_l142_142885


namespace geometric_arithmetic_sequence_common_ratio_l142_142322

theorem geometric_arithmetic_sequence_common_ratio (a_1 a_2 a_3 q : ℝ) 
  (h1 : a_2 = a_1 * q) 
  (h2 : a_3 = a_1 * q^2)
  (h3 : 2 * a_3 = a_1 + a_2) : (q = 1) ∨ (q = -1) :=
by
  sorry

end geometric_arithmetic_sequence_common_ratio_l142_142322


namespace volume_of_solid_of_revolution_l142_142305

theorem volume_of_solid_of_revolution (a : ℝ) : 
  let h := a / 2
  let r := (Real.sqrt 3 / 2) * a
  2 * (1 / 3) * π * r^2 * h = (π * a^3) / 4 :=
by
  sorry

end volume_of_solid_of_revolution_l142_142305


namespace total_weight_correct_l142_142672

def Marco_strawberry_weight : ℕ := 15
def Dad_strawberry_weight : ℕ := 22
def total_strawberry_weight : ℕ := Marco_strawberry_weight + Dad_strawberry_weight

theorem total_weight_correct :
  total_strawberry_weight = 37 :=
by
  sorry

end total_weight_correct_l142_142672


namespace car_distance_l142_142477

theorem car_distance (t : ℚ) (s : ℚ) (d : ℚ) 
(h1 : t = 2 + 2 / 5) 
(h2 : s = 260) 
(h3 : d = s * t) : 
d = 624 := by
  sorry

end car_distance_l142_142477


namespace final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r_l142_142064

variable (k r s N : ℝ)
variable (h_pos_k : 0 < k)
variable (h_pos_r : 0 < r)
variable (h_pos_s : 0 < s)
variable (h_pos_N : 0 < N)
variable (h_r_lt_80 : r < 80)

theorem final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r :
  N * (1 + k / 100) * (1 - r / 100) + 10 * s > N ↔ k > 100 * r / (100 - r) :=
sorry

end final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r_l142_142064


namespace area_of_largest_medallion_is_314_l142_142727

noncomputable def largest_medallion_area_in_square (side: ℝ) (π: ℝ) : ℝ :=
  let diameter := side
  let radius := diameter / 2
  let area := π * radius^2
  area

theorem area_of_largest_medallion_is_314 :
  largest_medallion_area_in_square 20 3.14 = 314 := 
  sorry

end area_of_largest_medallion_is_314_l142_142727


namespace coeff_x3_in_expansion_of_x_plus_1_50_l142_142899

theorem coeff_x3_in_expansion_of_x_plus_1_50 :
  (Finset.range 51).sum (λ k => Nat.choose 50 k * (1 : ℕ) ^ (50 - k) * k ^ 3) = 19600 := by
  sorry

end coeff_x3_in_expansion_of_x_plus_1_50_l142_142899


namespace no_max_value_if_odd_and_symmetric_l142_142469

variable (f : ℝ → ℝ)

-- Definitions:
def domain_is_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f x
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_symmetric_about_1_1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 - x) = 2 - f x

-- The theorem stating that under the given conditions there is no maximum value.
theorem no_max_value_if_odd_and_symmetric :
  domain_is_R f → is_odd_function f → is_symmetric_about_1_1 f → ¬∃ M : ℝ, ∀ x : ℝ, f x ≤ M := by
  sorry

end no_max_value_if_odd_and_symmetric_l142_142469


namespace dodecahedron_diagonals_l142_142825

-- Define a structure representing a dodecahedron with its properties
structure Dodecahedron where
  faces : Nat
  vertices : Nat
  faces_meeting_at_each_vertex : Nat

-- Concretely define a dodecahedron based on the given problem properties
def dodecahedron_example : Dodecahedron :=
  { faces := 12,
    vertices := 20,
    faces_meeting_at_each_vertex := 3 }

-- Lean statement to prove the number of interior diagonals in a dodecahedron
theorem dodecahedron_diagonals (d : Dodecahedron) (h : d = dodecahedron_example) : 
  (d.vertices * (d.vertices - d.faces_meeting_at_each_vertex) / 2) = 160 := by
  rw [h]
  -- Even though we skip the proof, Lean should recognize the transformation
  sorry

end dodecahedron_diagonals_l142_142825


namespace sixty_percent_of_40_greater_than_four_fifths_of_25_l142_142595

theorem sixty_percent_of_40_greater_than_four_fifths_of_25 :
  (60 / 100 : ℝ) * 40 - (4 / 5 : ℝ) * 25 = 4 := by
  sorry

end sixty_percent_of_40_greater_than_four_fifths_of_25_l142_142595


namespace max_price_per_unit_l142_142747

-- Define the conditions
def original_price : ℝ := 25
def original_sales_volume : ℕ := 80000
def price_increase_effect (t : ℝ) : ℝ := 2000 * (t - original_price)
def new_sales_volume (t : ℝ) : ℝ := 130 - 2 * t

-- Define the condition for revenue
def revenue_condition (t : ℝ) : Prop :=
  t * new_sales_volume t ≥ original_price * original_sales_volume

-- Statement to prove the maximum price per unit
theorem max_price_per_unit : ∀ t : ℝ, revenue_condition t → t ≤ 40 := sorry

end max_price_per_unit_l142_142747


namespace polynomial_solutions_l142_142861

theorem polynomial_solutions (P : Polynomial ℝ) :
  (∀ x : ℝ, P.eval x * P.eval (x + 1) = P.eval (x^2 - x + 3)) →
  (P = 0 ∨ ∃ n : ℕ, P = (Polynomial.C 1) * (Polynomial.X^2 - 2 * Polynomial.X + 3)^n) :=
by
  sorry

end polynomial_solutions_l142_142861


namespace choose_team_captains_l142_142633

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem choose_team_captains :
  let total_members := 15
  let shortlisted := 5
  let regular := total_members - shortlisted
  binom total_members 4 - binom regular 4 = 1155 :=
by
  sorry

end choose_team_captains_l142_142633


namespace price_after_discount_eq_cost_price_l142_142709

theorem price_after_discount_eq_cost_price (m : Real) :
  let selling_price_before_discount := 1.25 * m
  let price_after_discount := 0.80 * selling_price_before_discount
  price_after_discount = m :=
by
  let selling_price_before_discount := 1.25 * m
  let price_after_discount := 0.80 * selling_price_before_discount
  sorry

end price_after_discount_eq_cost_price_l142_142709


namespace correct_method_eliminates_y_l142_142395

def eliminate_y_condition1 (x y : ℝ) : Prop :=
  5 * x + 2 * y = 20

def eliminate_y_condition2 (x y : ℝ) : Prop :=
  4 * x - y = 8

theorem correct_method_eliminates_y (x y : ℝ) :
  eliminate_y_condition1 x y ∧ eliminate_y_condition2 x y →
  5 * x + 2 * y + 2 * (4 * x - y) = 36 :=
by
  sorry

end correct_method_eliminates_y_l142_142395


namespace tan_C_value_l142_142295

theorem tan_C_value (A B C : ℝ)
  (h_cos_A : Real.cos A = 4/5)
  (h_tan_A_minus_B : Real.tan (A - B) = -1/2) :
  Real.tan C = 11/2 :=
sorry

end tan_C_value_l142_142295


namespace quadratic_discriminant_l142_142536

-- Define the quadratic equation coefficients
def a : ℤ := 5
def b : ℤ := -11
def c : ℤ := 2

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

-- assert the discriminant for given coefficients
theorem quadratic_discriminant : discriminant a b c = 81 :=
by
  sorry

end quadratic_discriminant_l142_142536


namespace angle_triple_complement_l142_142354

theorem angle_triple_complement (x : ℝ) (h : x = 3 * (90 - x)) : x = 67.5 := by
  sorry

end angle_triple_complement_l142_142354


namespace prime_p_prime_p₁₀_prime_p₁₄_l142_142487

theorem prime_p_prime_p₁₀_prime_p₁₄ (p : ℕ) (h₀p : Nat.Prime p) 
  (h₁ : Nat.Prime (p + 10)) (h₂ : Nat.Prime (p + 14)) : p = 3 := by
  sorry

end prime_p_prime_p₁₀_prime_p₁₄_l142_142487


namespace price_of_sundae_l142_142670

theorem price_of_sundae (total_ice_cream_bars : ℕ) (total_sundae_price : ℝ)
                        (total_price : ℝ) (price_per_ice_cream_bar : ℝ) (num_ice_cream_bars : ℕ) (num_sundaes : ℕ)
                        (h1 : total_ice_cream_bars = num_ice_cream_bars)
                        (h2 : total_price = 200)
                        (h3 : price_per_ice_cream_bar = 0.40)
                        (h4 : num_ice_cream_bars = 200)
                        (h5 : num_sundaes = 200)
                        (h6 : total_ice_cream_bars * price_per_ice_cream_bar + total_sundae_price = total_price) :
  total_sundae_price / num_sundaes = 0.60 :=
sorry

end price_of_sundae_l142_142670


namespace function_passes_through_fixed_point_l142_142996

variables {a : ℝ}

/-- Given the function f(x) = a^(x-1) (a > 0 and a ≠ 1), prove that the function always passes through the point (1, 1) -/
theorem function_passes_through_fixed_point (h1 : a > 0) (h2 : a ≠ 1) :
  (a^(1-1) = 1) :=
by
  sorry

end function_passes_through_fixed_point_l142_142996


namespace neither_sufficient_nor_necessary_l142_142340

theorem neither_sufficient_nor_necessary (a b : ℝ) (h : a^2 > b^2) : 
  ¬(a > b) ∨ ¬(b > a) := sorry

end neither_sufficient_nor_necessary_l142_142340


namespace dot_product_equilateral_l142_142446

-- Define the conditions for the equilateral triangle ABC
variable {A B C : ℝ}

noncomputable def equilateral_triangle (A B C : ℝ) := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ |A - B| = 1 ∧ |B - C| = 1 ∧ |C - A| = 1

-- Define the dot product of the vectors AB and BC
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved
theorem dot_product_equilateral (A B C : ℝ) (h : equilateral_triangle A B C) : 
  dot_product (B - A, 0) (C - B, 0) = -1 / 2 :=
sorry

end dot_product_equilateral_l142_142446


namespace side_length_of_S2_l142_142531

-- Define our context and the statements we need to work with
theorem side_length_of_S2
  (r s : ℕ)
  (h1 : 2 * r + s = 2450)
  (h2 : 2 * r + 3 * s = 4000) : 
  s = 775 :=
sorry

end side_length_of_S2_l142_142531


namespace symmetric_line_equation_l142_142085

theorem symmetric_line_equation : 
  ∀ (P : ℝ × ℝ) (L : ℝ × ℝ × ℝ), 
  P = (1, 1) → 
  L = (2, 3, -6) → 
  (∃ (a b c : ℝ), a * 1 + b * 1 + c = 0 → a * x + b * y + c = 0 ↔ 2 * x + 3 * y - 4 = 0) 
:= 
sorry

end symmetric_line_equation_l142_142085


namespace tangent_line_at_one_f_gt_one_l142_142235

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x * Real.log x + (2 * Real.exp (x - 1)) / x

theorem tangent_line_at_one : 
  let y := f 1 + (Real.exp 1) * (x - 1)
  y = Real.exp (1 : ℝ) * (x - 1) + 2 := 
sorry

theorem f_gt_one (x : ℝ) (hx : 0 < x) : f x > 1 := 
sorry

end tangent_line_at_one_f_gt_one_l142_142235


namespace inverse_of_B_squared_l142_142422

noncomputable def B_inv : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -3, 0], ![0, -1, 0], ![0, 0, 5]]

theorem inverse_of_B_squared :
  (B_inv * B_inv) = ![![4, -3, 0], ![0, 1, 0], ![0, 0, 25]] := by
  sorry

end inverse_of_B_squared_l142_142422


namespace problem_l142_142744

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * (Real.sqrt 3) * Real.cos x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let dot_product := (a x).fst * (b x).fst + (a x).snd * (b x).snd
  let magnitude_square_b := (b x).fst ^ 2 + (b x).snd ^ 2
  dot_product + magnitude_square_b

theorem problem :
  (∀ x, f x = 5 * Real.sin (2 * x + Real.pi / 6) + 7 / 2) ∧
  (∃ T, T = Real.pi) ∧ 
  (∃ x, f x = 17 / 2) ∧ 
  (∃ x, f x = -3 / 2) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 6), 0 ≤ x ∧ x ≤ Real.pi / 6) ∧
  (∀ x ∈ Set.Icc (2 * Real.pi / 3) Real.pi, (2 * Real.pi / 3) ≤ x ∧ x ≤ Real.pi)
:= by
  sorry

end problem_l142_142744


namespace janet_better_condition_count_l142_142249

noncomputable def janet_initial := 10
noncomputable def janet_sells := 6
noncomputable def janet_remaining := janet_initial - janet_sells
noncomputable def brother_gives := 2 * janet_remaining
noncomputable def janet_after_brother := janet_remaining + brother_gives
noncomputable def janet_total := 24

theorem janet_better_condition_count : 
  janet_total - janet_after_brother = 12 := by
  sorry

end janet_better_condition_count_l142_142249


namespace volume_of_regular_triangular_pyramid_l142_142593

noncomputable def pyramid_volume (R φ : ℝ) : ℝ :=
  (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)

theorem volume_of_regular_triangular_pyramid (R φ : ℝ) 
  (cond1 : R > 0)
  (cond2: 0 < φ ∧ φ < π) :
  ∃ V, V = pyramid_volume R φ := by
    use (8 / 27) * R^3 * (Real.sin (φ / 2))^2 * (1 + 2 * Real.cos φ)
    sorry

end volume_of_regular_triangular_pyramid_l142_142593


namespace jared_sent_in_november_l142_142139

noncomputable def text_messages (n : ℕ) : ℕ :=
  match n with
  | 0 => 1  -- November
  | 1 => 2  -- December
  | 2 => 4  -- January
  | 3 => 8  -- February
  | 4 => 16 -- March
  | _ => 0

theorem jared_sent_in_november : text_messages 0 = 1 :=
sorry

end jared_sent_in_november_l142_142139


namespace beyonce_album_songs_l142_142455

theorem beyonce_album_songs
  (singles : ℕ)
  (album1_songs album2_songs album3_songs total_songs : ℕ)
  (h1 : singles = 5)
  (h2 : album1_songs = 15)
  (h3 : album2_songs = 15)
  (h4 : total_songs = 55) :
  album3_songs = 20 :=
by
  sorry

end beyonce_album_songs_l142_142455


namespace pyramid_base_length_of_tangent_hemisphere_l142_142202

noncomputable def pyramid_base_side_length (radius height : ℝ) (tangent : ℝ → ℝ → Prop) : ℝ := sorry

theorem pyramid_base_length_of_tangent_hemisphere 
(r h : ℝ) (tangent : ℝ → ℝ → Prop) (tangent_property : ∀ x y, tangent x y → y = 0) 
(h_radius : r = 3) (h_height : h = 9) 
(tangent_conditions : tangent r h → tangent r h) : 
  pyramid_base_side_length r h tangent = 9 :=
sorry

end pyramid_base_length_of_tangent_hemisphere_l142_142202


namespace part1_part2_part3_l142_142965

-- Part (1)
theorem part1 (m : ℝ) : (2 * m - 3) * (5 - 3 * m) = -6 * m^2 + 19 * m - 15 :=
  sorry

-- Part (2)
theorem part2 (a b : ℝ) : (3 * a^3) ^ 2 * (2 * b^2) ^ 3 / (6 * a * b) ^ 2 = 2 * a^4 * b^4 :=
  sorry

-- Part (3)
theorem part3 (a b : ℝ) : (a - b) * (a^2 + a * b + b^2) = a^3 - b^3 :=
  sorry

end part1_part2_part3_l142_142965


namespace moles_of_HCl_combined_eq_one_l142_142594

-- Defining the chemical species involved in the reaction
def NaHCO3 : Type := Nat
def HCl : Type := Nat
def NaCl : Type := Nat
def H2O : Type := Nat
def CO2 : Type := Nat

-- Defining the balanced chemical equation as a condition
def reaction (n_NaHCO3 n_HCl n_NaCl n_H2O n_CO2 : Nat) : Prop :=
  n_NaHCO3 + n_HCl = n_NaCl + n_H2O + n_CO2

-- Given conditions
def one_mole_of_NaHCO3 : Nat := 1
def one_mole_of_NaCl_produced : Nat := 1

-- Proof problem
theorem moles_of_HCl_combined_eq_one :
  ∃ (n_HCl : Nat), reaction one_mole_of_NaHCO3 n_HCl one_mole_of_NaCl_produced 1 1 ∧ n_HCl = 1 := 
by
  sorry

end moles_of_HCl_combined_eq_one_l142_142594


namespace range_of_abs_function_l142_142603

theorem range_of_abs_function : ∀ (y : ℝ), (∃ (x : ℝ), y = |x + 5| - |x - 3|) ↔ y ∈ Set.Icc (-8) 8 :=
by
  sorry

end range_of_abs_function_l142_142603


namespace students_neither_cs_nor_robotics_l142_142897

theorem students_neither_cs_nor_robotics
  (total_students : ℕ)
  (cs_students : ℕ)
  (robotics_students : ℕ)
  (both_cs_and_robotics : ℕ)
  (H1 : total_students = 150)
  (H2 : cs_students = 90)
  (H3 : robotics_students = 70)
  (H4 : both_cs_and_robotics = 20) :
  (total_students - (cs_students + robotics_students - both_cs_and_robotics)) = 10 :=
by
  sorry

end students_neither_cs_nor_robotics_l142_142897


namespace intersection_at_one_point_l142_142454

theorem intersection_at_one_point (b : ℝ) :
  (∃ x₀ : ℝ, bx^2 + 7*x₀ + 4 = 0 ∧ (7)^2 - 4*b*4 = 0) →
  b = 49 / 16 :=
by
  sorry

end intersection_at_one_point_l142_142454


namespace hoursWorkedPerDay_l142_142601

-- Define the conditions
def widgetsPerHour := 20
def daysPerWeek := 5
def totalWidgetsPerWeek := 800

-- Theorem statement
theorem hoursWorkedPerDay : (totalWidgetsPerWeek / widgetsPerHour) / daysPerWeek = 8 := 
  sorry

end hoursWorkedPerDay_l142_142601


namespace min_dominos_in_2x2_l142_142421

/-- A 100 × 100 square is divided into 2 × 2 squares.
Then it is divided into dominos (rectangles 1 × 2 and 2 × 1).
Prove that the minimum number of dominos within the 2 × 2 squares is 100. -/
theorem min_dominos_in_2x2 (N : ℕ) (hN : N = 100) :
  ∃ d : ℕ, d = 100 :=
sorry

end min_dominos_in_2x2_l142_142421


namespace sum_of_their_ages_now_l142_142070

variable (Nacho Divya : ℕ)

-- Conditions
def divya_current_age := 5
def nacho_in_5_years := 3 * (divya_current_age + 5)

-- Definition to determine current age of Nacho
def nacho_current_age := nacho_in_5_years - 5

-- Sum of current ages
def sum_of_ages := divya_current_age + nacho_current_age

-- Theorem to prove the sum of their ages now is 30
theorem sum_of_their_ages_now : sum_of_ages = 30 :=
by
  sorry

end sum_of_their_ages_now_l142_142070


namespace range_of_a_l142_142700

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ (e^x + 1) * (a * x + 2 * a - 2) < 2) → a < 4 / 3 :=
by
  sorry

end range_of_a_l142_142700


namespace division_of_converted_values_l142_142088

theorem division_of_converted_values 
  (h : 144 * 177 = 25488) : 
  254.88 / 0.177 = 1440 := by
  sorry

end division_of_converted_values_l142_142088


namespace not_divisible_67_l142_142862

theorem not_divisible_67
  (x y : ℕ)
  (hx : ¬ (67 ∣ x))
  (hy : ¬ (67 ∣ y))
  (h : (7 * x + 32 * y) % 67 = 0)
  : (10 * x + 17 * y + 1) % 67 ≠ 0 := sorry

end not_divisible_67_l142_142862


namespace chloes_test_scores_l142_142302

theorem chloes_test_scores :
  ∃ (scores : List ℕ),
  scores = [93, 92, 86, 82, 79, 78] ∧
  (List.take 4 scores).sum = 339 ∧
  scores.sum / 6 = 85 ∧
  List.Nodup scores ∧
  ∀ score ∈ scores, score < 95 :=
by
  sorry

end chloes_test_scores_l142_142302


namespace time_on_wednesday_is_40_minutes_l142_142111

def hours_to_minutes (h : ℚ) : ℚ := h * 60

def time_monday : ℚ := hours_to_minutes (3 / 4)
def time_tuesday : ℚ := hours_to_minutes (1 / 2)
def time_wednesday (w : ℚ) : ℚ := w
def time_thursday : ℚ := hours_to_minutes (5 / 6)
def time_friday : ℚ := 75
def total_time : ℚ := hours_to_minutes 4

theorem time_on_wednesday_is_40_minutes (w : ℚ) 
    (h1 : time_monday = 45) 
    (h2 : time_tuesday = 30) 
    (h3 : time_thursday = 50) 
    (h4 : time_friday = 75)
    (h5 : total_time = 240) 
    (h6 : total_time = time_monday + time_tuesday + time_wednesday w + time_thursday + time_friday) 
    : w = 40 := 
by 
  sorry

end time_on_wednesday_is_40_minutes_l142_142111


namespace farm_area_l142_142522

theorem farm_area
  (b : ℕ) (l : ℕ) (d : ℕ)
  (h_b : b = 30)
  (h_cost : 15 * (l + b + d) = 1800)
  (h_pythagorean : d^2 = l^2 + b^2) :
  l * b = 1200 :=
by
  sorry

end farm_area_l142_142522


namespace undefined_denominator_values_l142_142254

theorem undefined_denominator_values (a : ℝ) : a = 3 ∨ a = -3 ↔ ∃ b : ℝ, (a - b) * (a + b) = 0 := by
  sorry

end undefined_denominator_values_l142_142254


namespace pause_point_l142_142658

-- Definitions
def total_movie_length := 60 -- In minutes
def remaining_time := 30 -- In minutes

-- Theorem stating the pause point in the movie
theorem pause_point : total_movie_length - remaining_time = 30 := by
  -- This is the original solution in mathematical terms, omitted in lean statement.
  -- total_movie_length - remaining_time = 60 - 30 = 30
  sorry

end pause_point_l142_142658


namespace find_functional_solution_l142_142675

theorem find_functional_solution (c : ℝ) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) :
  ∀ x : ℝ, f x = x ^ 3 + c * x := by
  sorry

end find_functional_solution_l142_142675


namespace find_B_l142_142027

theorem find_B (A B : ℕ) (h : 5 * 100 + 10 * A + 8 - (B * 100 + 14) = 364) : B = 2 :=
sorry

end find_B_l142_142027


namespace find_integer_divisible_by_18_and_square_root_in_range_l142_142023

theorem find_integer_divisible_by_18_and_square_root_in_range :
  ∃ x : ℕ, 28 < Real.sqrt x ∧ Real.sqrt x < 28.2 ∧ 18 ∣ x ∧ x = 792 :=
by
  sorry

end find_integer_divisible_by_18_and_square_root_in_range_l142_142023


namespace carlos_laundry_time_l142_142101

def washing_time1 := 30
def washing_time2 := 45
def washing_time3 := 40
def washing_time4 := 50
def washing_time5 := 35
def drying_time1 := 85
def drying_time2 := 95

def total_laundry_time := washing_time1 + washing_time2 + washing_time3 + washing_time4 + washing_time5 + drying_time1 + drying_time2

theorem carlos_laundry_time : total_laundry_time = 380 :=
by
  sorry

end carlos_laundry_time_l142_142101


namespace ages_correct_l142_142011

-- Let A be Anya's age and P be Petya's age
def anya_age : ℕ := 4
def petya_age : ℕ := 12

-- The conditions
def condition1 (A P : ℕ) : Prop := P = 3 * A
def condition2 (A P : ℕ) : Prop := P - A = 8

-- The statement to be proven
theorem ages_correct : condition1 anya_age petya_age ∧ condition2 anya_age petya_age :=
by
  unfold condition1 condition2 anya_age petya_age -- Reveal the definitions
  have h1 : petya_age = 3 * anya_age := by
    sorry
  have h2 : petya_age - anya_age = 8 := by
    sorry
  exact ⟨h1, h2⟩ -- Combine both conditions into a single conjunction

end ages_correct_l142_142011


namespace average_of_three_l142_142119

theorem average_of_three {a b c d e : ℚ}
    (h1 : (a + b + c + d + e) / 5 = 12)
    (h2 : (d + e) / 2 = 24) :
    (a + b + c) / 3 = 4 := by
  sorry

end average_of_three_l142_142119


namespace sum_quotient_reciprocal_eq_one_point_thirty_five_l142_142103

theorem sum_quotient_reciprocal_eq_one_point_thirty_five (x y : ℝ)
  (h1 : x + y = 45) (h2 : x * y = 500) : x / y + 1 / x + 1 / y = 1.35 := by
  -- Proof details would go here
  sorry

end sum_quotient_reciprocal_eq_one_point_thirty_five_l142_142103


namespace train_car_speed_ratio_l142_142910

theorem train_car_speed_ratio
  (distance_bus : ℕ) (time_bus : ℕ) (distance_car : ℕ) (time_car : ℕ)
  (speed_bus := distance_bus / time_bus)
  (speed_train := speed_bus / (3 / 4))
  (speed_car := distance_car / time_car)
  (ratio := (speed_train : ℚ) / (speed_car : ℚ))
  (h1 : distance_bus = 480)
  (h2 : time_bus = 8)
  (h3 : distance_car = 450)
  (h4 : time_car = 6) :
  ratio = 16 / 15 :=
by
  sorry

end train_car_speed_ratio_l142_142910


namespace two_digit_integer_eq_55_l142_142845

theorem two_digit_integer_eq_55
  (c : ℕ)
  (h1 : c / 10 + c % 10 = 10)
  (h2 : (c / 10) * (c % 10) = 25) :
  c = 55 :=
  sorry

end two_digit_integer_eq_55_l142_142845


namespace thirty_five_power_identity_l142_142509

theorem thirty_five_power_identity (m n : ℕ) : 
  let P := 5^m 
  let Q := 7^n 
  35^(m*n) = P^n * Q^m :=
by 
  sorry

end thirty_five_power_identity_l142_142509


namespace lindsey_squat_weight_l142_142336

-- Define the conditions
def num_bands : ℕ := 2
def resistance_per_band : ℤ := 5
def dumbbell_weight : ℤ := 10

-- Define the weight Lindsay will squat
def total_weight : ℤ := num_bands * resistance_per_band + dumbbell_weight

-- State the theorem
theorem lindsey_squat_weight : total_weight = 20 :=
by
  sorry

end lindsey_squat_weight_l142_142336


namespace supplies_total_cost_l142_142831

-- Definitions based on conditions in a)
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def cost_of_baking_soda : ℕ := 1
def students_count : ℕ := 23

-- The main theorem to prove
theorem supplies_total_cost :
  cost_of_bow * students_count + cost_of_vinegar * students_count + cost_of_baking_soda * students_count = 184 :=
by
  sorry

end supplies_total_cost_l142_142831


namespace boxes_with_nothing_l142_142333

theorem boxes_with_nothing (h_total : 15 = total_boxes)
    (h_pencils : 9 = pencil_boxes)
    (h_pens : 5 = pen_boxes)
    (h_both_pens_and_pencils : 3 = both_pen_and_pencil_boxes)
    (h_markers : 4 = marker_boxes)
    (h_both_markers_and_pencils : 2 = both_marker_and_pencil_boxes)
    (h_no_markers_and_pens : no_marker_and_pen_boxes = 0)
    (h_no_all_three_items : no_all_three_items = 0) :
    ∃ (neither_boxes : ℕ), neither_boxes = 2 :=
by
  sorry

end boxes_with_nothing_l142_142333


namespace students_failed_l142_142462

theorem students_failed (total_students : ℕ) (percent_A : ℚ) (fraction_BC : ℚ) (students_A : ℕ)
  (students_remaining : ℕ) (students_BC : ℕ) (students_failed : ℕ)
  (h1 : total_students = 32) (h2 : percent_A = 0.25) (h3 : fraction_BC = 0.25)
  (h4 : students_A = total_students * percent_A)
  (h5 : students_remaining = total_students - students_A)
  (h6 : students_BC = students_remaining * fraction_BC)
  (h7 : students_failed = total_students - students_A - students_BC) :
  students_failed = 18 :=
sorry

end students_failed_l142_142462


namespace number_of_cups_needed_to_fill_container_l142_142878

theorem number_of_cups_needed_to_fill_container (container_capacity cup_capacity : ℕ) (h1 : container_capacity = 640) (h2 : cup_capacity = 120) : 
  (container_capacity + cup_capacity - 1) / cup_capacity = 6 :=
by
  sorry

end number_of_cups_needed_to_fill_container_l142_142878


namespace more_stable_scores_l142_142384

-- Define the variances for Student A and Student B
def variance_A : ℝ := 38
def variance_B : ℝ := 15

-- Formulate the theorem
theorem more_stable_scores : variance_A > variance_B → "B" = "B" :=
by
  intro h
  sorry

end more_stable_scores_l142_142384


namespace area_of_rectangle_l142_142077

theorem area_of_rectangle (w l : ℝ) (h₁ : w = l / 3) (h₂ : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end area_of_rectangle_l142_142077


namespace discriminant_of_quadratic_l142_142963

def a := 5
def b := 5 + 1/5
def c := 1/5
def discriminant (a b c : ℚ) := b^2 - 4 * a * c

theorem discriminant_of_quadratic :
  discriminant a b c = 576 / 25 :=
by
  sorry

end discriminant_of_quadratic_l142_142963


namespace canadian_olympiad_2008_inequality_l142_142263

variable (a b c : ℝ)
variables (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c)
variable (sum_abc : a + b + c = 1)

theorem canadian_olympiad_2008_inequality :
  (ab / ((b + c) * (c + a))) + (bc / ((c + a) * (a + b))) + (ca / ((a + b) * (b + c))) ≥ 3 / 4 :=
sorry

end canadian_olympiad_2008_inequality_l142_142263


namespace total_cost_is_660_l142_142268

def total_material_cost : ℝ :=
  let velvet_area := (12 * 4) * 3
  let velvet_cost := velvet_area * 3
  let silk_cost := 2 * 6
  let lace_cost := 5 * 2 * 10
  let bodice_cost := silk_cost + lace_cost
  let satin_area := 2.5 * 1.5
  let satin_cost := satin_area * 4
  let leather_area := 1 * 1.5 * 2
  let leather_cost := leather_area * 5
  let wool_area := 5 * 2
  let wool_cost := wool_area * 8
  let ribbon_cost := 3 * 2
  velvet_cost + bodice_cost + satin_cost + leather_cost + wool_cost + ribbon_cost

theorem total_cost_is_660 : total_material_cost = 660 := by
  sorry

end total_cost_is_660_l142_142268


namespace cos_formula_of_tan_l142_142110

theorem cos_formula_of_tan (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi) :
  Real.cos (5 * Real.pi / 2 + 2 * α) = -4 / 5 := 
  sorry

end cos_formula_of_tan_l142_142110


namespace removed_term_sequence_l142_142402

theorem removed_term_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) (k : ℕ) :
  (∀ n, S n = 2 * n^2 - n) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (S 21 - a k = 40 * 20) →
  a k = 4 * k - 3 →
  k = 16 :=
by
  intros hs ha h_avg h_ak
  sorry

end removed_term_sequence_l142_142402


namespace xiaoliang_prob_correct_l142_142541

def initial_box_setup : List (Nat × Nat) := [(1, 2), (2, 2), (3, 2), (4, 2)]

def xiaoming_draw : List Nat := [1, 1, 3]

def remaining_balls_after_xiaoming : List (Nat × Nat) := [(1, 0), (2, 2), (3, 1), (4, 2)]

def remaining_ball_count (balls : List (Nat × Nat)) : Nat :=
  balls.foldl (λ acc ⟨_, count⟩ => acc + count) 0

theorem xiaoliang_prob_correct :
  (1 : ℚ) / (remaining_ball_count remaining_balls_after_xiaoming) = 1 / 5 :=
by
  sorry

end xiaoliang_prob_correct_l142_142541


namespace dolls_given_to_girls_correct_l142_142703

-- Define the total number of toys given
def total_toys_given : ℕ := 403

-- Define the number of toy cars given to boys
def toy_cars_given_to_boys : ℕ := 134

-- Define the number of dolls given to girls
def dolls_given_to_girls : ℕ := total_toys_given - toy_cars_given_to_boys

-- State the theorem to prove the number of dolls given to girls
theorem dolls_given_to_girls_correct : dolls_given_to_girls = 269 := by
  sorry

end dolls_given_to_girls_correct_l142_142703


namespace find_middle_and_oldest_sons_l142_142604

-- Defining the conditions
def youngest_age : ℕ := 2
def father_age : ℕ := 33
def father_age_in_12_years : ℕ := father_age + 12
def youngest_age_in_12_years : ℕ := youngest_age + 12

-- Lean theorem statement to find the ages of the middle and oldest sons
theorem find_middle_and_oldest_sons (y z : ℕ) (h1 : father_age_in_12_years = (youngest_age_in_12_years + 12 + y + 12 + z + 12)) :
  y = 3 ∧ z = 4 :=
sorry

end find_middle_and_oldest_sons_l142_142604


namespace faster_train_length_l142_142345

theorem faster_train_length
  (speed_faster : ℝ)
  (speed_slower : ℝ)
  (time_to_cross : ℝ)
  (relative_speed_limit: ℝ)
  (h1 : speed_faster = 108 * 1000 / 3600)
  (h2: speed_slower = 36 * 1000 / 3600)
  (h3: time_to_cross = 17)
  (h4: relative_speed_limit = 2) :
  (speed_faster - speed_slower) * time_to_cross = 340 := 
sorry

end faster_train_length_l142_142345


namespace line_canonical_form_l142_142163

theorem line_canonical_form :
  (∀ x y z : ℝ, 4 * x + y - 3 * z + 2 = 0 → 2 * x - y + z - 8 = 0 ↔
    ∃ t : ℝ, x = 1 + -2 * t ∧ y = -6 + -10 * t ∧ z = -6 * t) :=
by
  sorry

end line_canonical_form_l142_142163


namespace find_shaun_age_l142_142138

def current_ages (K G S : ℕ) :=
  K + 4 = 2 * (G + 4) ∧
  S + 8 = 2 * (K + 8) ∧
  S + 12 = 3 * (G + 12)

theorem find_shaun_age (K G S : ℕ) (h : current_ages K G S) : S = 48 :=
  by
    sorry

end find_shaun_age_l142_142138


namespace otimes_evaluation_l142_142481

def otimes (a b : ℝ) : ℝ := a * b + a - b

theorem otimes_evaluation (a b : ℝ) : 
  otimes a b + otimes (b - a) b = b^2 - b := 
  by
  sorry

end otimes_evaluation_l142_142481


namespace sum_of_primes_between_20_and_40_l142_142760

theorem sum_of_primes_between_20_and_40 : 
  (23 + 29 + 31 + 37) = 120 := 
by
  -- Proof goes here
sorry

end sum_of_primes_between_20_and_40_l142_142760


namespace marble_probability_l142_142443

theorem marble_probability :
  let total_ways := (Nat.choose 6 4)
  let favorable_ways := 
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1) +
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1) +
    (Nat.choose 2 2) * (Nat.choose 2 1) * (Nat.choose 2 1)
  let probability := (favorable_ways : ℚ) / total_ways
  probability = 4 / 5 := by
  sorry

end marble_probability_l142_142443


namespace most_suitable_sampling_method_l142_142153

/-- A unit has 28 elderly people, 54 middle-aged people, and 81 young people. 
    A sample of 36 people needs to be drawn in a way that accounts for age.
    The most suitable method for drawing a sample is to exclude one elderly person first,
    then use stratified sampling. -/
theorem most_suitable_sampling_method 
  (elderly : ℕ) (middle_aged : ℕ) (young : ℕ) (sample_size : ℕ) (suitable_method : String)
  (condition1 : elderly = 28) 
  (condition2 : middle_aged = 54) 
  (condition3 : young = 81) 
  (condition4 : sample_size = 36) 
  (condition5 : suitable_method = "Exclude one elderly person first, then stratify sampling") : 
  suitable_method = "Exclude one elderly person first, then stratify sampling" := 
by sorry

end most_suitable_sampling_method_l142_142153


namespace g_at_10_is_300_l142_142279

-- Define the function g and the given condition about g
def g: ℕ → ℤ := sorry

axiom g_cond (m n: ℕ) (h: m ≥ n): g (m + n) + g (m - n) = 2 * g m + 3 * g n
axiom g_1: g 1 = 3

-- Statement to be proved
theorem g_at_10_is_300 : g 10 = 300 := by
  sorry

end g_at_10_is_300_l142_142279


namespace original_numbers_placement_l142_142061

-- Define each letter stands for a given number
def A : ℕ := 1
def B : ℕ := 3
def C : ℕ := 2
def D : ℕ := 5
def E : ℕ := 6
def F : ℕ := 4

-- Conditions provided
def white_triangle_condition (x y z : ℕ) : Prop :=
x + y = z

-- Main problem reformulated as theorem
theorem original_numbers_placement :
  (A = 1) ∧ (B = 3) ∧ (C = 2) ∧ (D = 5) ∧ (E = 6) ∧ (F = 4) :=
sorry

end original_numbers_placement_l142_142061


namespace thirteen_percent_greater_than_80_l142_142837

theorem thirteen_percent_greater_than_80 (x : ℝ) (h : x = 1.13 * 80) : x = 90.4 :=
sorry

end thirteen_percent_greater_than_80_l142_142837


namespace function_monotonicity_l142_142950

theorem function_monotonicity (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 1 → (3 * x^2 + a) < 0) ∧ 
  (∀ x, 1 < x → (3 * x^2 + a) > 0) → 
  (a = -3 ∧ ∃ b : ℝ, true) :=
by {
  sorry
}

end function_monotonicity_l142_142950


namespace part1_intersection_when_a_is_zero_part2_range_of_a_l142_142984

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x < a + 5}

-- Part (1): When a = 0, find A ∩ B
theorem part1_intersection_when_a_is_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

-- Part (2): If A ∪ B = A, find the range of values for a
theorem part2_range_of_a (a : ℝ) :
  (A ∪ B a = A) → (0 < a ∧ a ≤ 1) ∨ (6 ≤ a) :=
sorry

end part1_intersection_when_a_is_zero_part2_range_of_a_l142_142984


namespace cattle_selling_price_per_pound_correct_l142_142903

def purchase_price : ℝ := 40000
def cattle_count : ℕ := 100
def feed_cost_percentage : ℝ := 0.20
def weight_per_head : ℕ := 1000
def profit : ℝ := 112000

noncomputable def total_feed_cost : ℝ := purchase_price * feed_cost_percentage
noncomputable def total_cost : ℝ := purchase_price + total_feed_cost
noncomputable def total_revenue : ℝ := total_cost + profit
def total_weight : ℕ := cattle_count * weight_per_head
noncomputable def selling_price_per_pound : ℝ := total_revenue / total_weight

theorem cattle_selling_price_per_pound_correct :
  selling_price_per_pound = 1.60 := by
  sorry

end cattle_selling_price_per_pound_correct_l142_142903


namespace mike_earnings_l142_142919

theorem mike_earnings :
  let total_games := 16
  let non_working_games := 8
  let price_per_game := 7
  let working_games := total_games - non_working_games
  let earnings := working_games * price_per_game
  earnings = 56 := 
by
  sorry

end mike_earnings_l142_142919


namespace matrix_B_power_103_l142_142326

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem matrix_B_power_103 :
  B ^ 103 = B :=
by
  sorry

end matrix_B_power_103_l142_142326


namespace function_increasing_l142_142660

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) :=
  ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2

theorem function_increasing {f : ℝ → ℝ}
  (H : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2) :
  is_monotonically_increasing f :=
by
  sorry

end function_increasing_l142_142660


namespace sin_90_deg_l142_142227

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l142_142227


namespace marble_ratio_l142_142379

-- Let Allison, Angela, and Albert have some number of marbles denoted by variables.
variable (Albert Angela Allison : ℕ)

-- Given conditions.
axiom h1 : Angela = Allison + 8
axiom h2 : Allison = 28
axiom h3 : Albert + Allison = 136

-- Prove that the ratio of the number of marbles Albert has to the number of marbles Angela has is 3.
theorem marble_ratio : Albert / Angela = 3 := by
  sorry

end marble_ratio_l142_142379


namespace sum_of_integer_solutions_l142_142220

theorem sum_of_integer_solutions (n_values : List ℤ) : 
  (∀ n ∈ n_values, ∃ (k : ℤ), 2 * n - 3 = k ∧ k ∣ 18) → (n_values.sum = 11) := 
by
  sorry

end sum_of_integer_solutions_l142_142220


namespace ratio_of_shaded_to_white_l142_142281

theorem ratio_of_shaded_to_white (A : ℝ) : 
  let shaded_area := 5 * A
  let unshaded_area := 3 * A
  shaded_area / unshaded_area = 5 / 3 := by
  sorry

end ratio_of_shaded_to_white_l142_142281


namespace elizabeth_fruits_l142_142953

def total_fruits (initial_bananas initial_apples initial_grapes eaten_bananas eaten_apples eaten_grapes : Nat) : Nat :=
  let bananas_left := initial_bananas - eaten_bananas
  let apples_left := initial_apples - eaten_apples
  let grapes_left := initial_grapes - eaten_grapes
  bananas_left + apples_left + grapes_left

theorem elizabeth_fruits : total_fruits 12 7 19 4 2 10 = 22 := by
  sorry

end elizabeth_fruits_l142_142953


namespace driver_total_distance_is_148_l142_142079

-- Definitions of the distances traveled according to the given conditions
def distance_MWF : ℕ := 12 * 3
def total_distance_MWF : ℕ := distance_MWF * 3
def distance_T : ℕ := 9 * 5 / 2  -- using ℕ for 2.5 hours as 5/2
def distance_Th : ℕ := 7 * 5 / 2

-- Statement of the total distance calculation
def total_distance_week : ℕ :=
  total_distance_MWF + distance_T + distance_Th

-- Theorem stating the total distance traveled during the week
theorem driver_total_distance_is_148 : total_distance_week = 148 := by
  sorry

end driver_total_distance_is_148_l142_142079


namespace Thabo_books_problem_l142_142573

theorem Thabo_books_problem 
  (P F : ℕ)
  (H1 : 180 = F + P + 30)
  (H2 : F = 2 * P)
  (H3 : P > 30) :
  P - 30 = 20 := 
sorry

end Thabo_books_problem_l142_142573


namespace find_base_k_l142_142929

-- Define the conversion condition as a polynomial equation.
def base_conversion (k : ℤ) : Prop := k^2 + 3*k + 2 = 42

-- State the theorem to be proven: given the conversion condition, k = 5.
theorem find_base_k (k : ℤ) (h : base_conversion k) : k = 5 :=
by
  sorry

end find_base_k_l142_142929


namespace evaluate_fg_of_8_l142_142309

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 11

theorem evaluate_fg_of_8 : f (g 8) = 211 :=
by
  sorry

end evaluate_fg_of_8_l142_142309


namespace simplify_fraction_l142_142600

variable {a b c : ℝ}

theorem simplify_fraction (h : a + b + c ≠ 0) :
  (a^2 + 3*a*b + b^2 - c^2) / (a^2 + 3*a*c + c^2 - b^2) = (a + b - c) / (a - b + c) := 
by
  sorry

end simplify_fraction_l142_142600


namespace probability_final_marble_red_l142_142348

theorem probability_final_marble_red :
  let P_wA := 5/8
  let P_bA := 3/8
  let P_gB_p1 := 6/10
  let P_rB_p2 := 4/10
  let P_gC_p1 := 4/7
  let P_rC_p2 := 3/7
  let P_wr_b_g := P_wA * P_gB_p1 * P_rB_p2
  let P_blk_g_red := P_bA * P_gC_p1 * P_rC_p2
  (P_wr_b_g + P_blk_g_red) = 79/980 :=
by {
  let P_wA := 5/8
  let P_bA := 3/8
  let P_gB_p1 := 6/10
  let P_rB_p2 := 4/10
  let P_gC_p1 := 4/7
  let P_rC_p2 := 3/7
  let P_wr_b_g := P_wA * P_gB_p1 * P_rB_p2
  let P_blk_g_red := P_bA * P_gC_p1 * P_rC_p2
  show (P_wr_b_g + P_blk_g_red) = 79/980
  sorry
}

end probability_final_marble_red_l142_142348


namespace trapezium_distance_parallel_sides_l142_142230

theorem trapezium_distance_parallel_sides (a b A : ℝ) (h : ℝ) (h1 : a = 20) (h2 : b = 18) (h3 : A = 380) :
  A = (1 / 2) * (a + b) * h → h = 20 :=
by
  intro h4
  rw [h1, h2, h3] at h4
  sorry

end trapezium_distance_parallel_sides_l142_142230


namespace number_of_spiders_l142_142040

theorem number_of_spiders (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 32) (h2 : legs_per_spider = 8) :
  total_legs / legs_per_spider = 4 := by
  sorry

end number_of_spiders_l142_142040


namespace samuel_faster_than_sarah_l142_142637

-- Definitions based on the conditions
def time_samuel : ℝ := 30
def time_sarah : ℝ := 1.3 * 60

-- The theorem to prove that Samuel finished his homework 48 minutes faster than Sarah
theorem samuel_faster_than_sarah : (time_sarah - time_samuel) = 48 := by
  sorry

end samuel_faster_than_sarah_l142_142637


namespace initial_discount_l142_142271

theorem initial_discount (total_amount price_after_initial_discount additional_disc_percent : ℝ)
  (H1 : total_amount = 1000)
  (H2 : price_after_initial_discount = total_amount - 280)
  (H3 : additional_disc_percent = 0.20) :
  let additional_discount := additional_disc_percent * price_after_initial_discount
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let total_discount := total_amount - price_after_additional_discount
  let initial_discount := total_discount - additional_discount
  initial_discount = 280 := by
  sorry

end initial_discount_l142_142271


namespace total_number_of_cottages_is_100_l142_142278

noncomputable def total_cottages
    (x : ℕ) (n : ℕ) 
    (h1 : 2 * x = number_of_two_room_cottages)
    (h2 : n * x = number_of_three_room_cottages)
    (h3 : 3 * (n * x) = 2 * x + 25) 
    (h4 : x + 2 * x + n * x ≥ 70) : ℕ :=
x + 2 * x + n * x

theorem total_number_of_cottages_is_100 
    (x n : ℕ)
    (h1 : 2 * x = number_of_two_room_cottages)
    (h2 : n * x = number_of_three_room_cottages)
    (h3 : 3 * (n * x) = 2 * x + 25)
    (h4 : x + 2 * x + n * x ≥ 70)
    (h5 : ∃ m : ℕ, m = (x + 2 * x + n * x)) :
  total_cottages x n h1 h2 h3 h4 = 100 :=
by
  sorry

end total_number_of_cottages_is_100_l142_142278


namespace unique_digit_sum_l142_142109

theorem unique_digit_sum (A B C D : ℕ) (h1 : A + B + C + D = 20) (h2 : B + A + 1 = 11) (uniq : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D)) : D = 8 :=
sorry

end unique_digit_sum_l142_142109


namespace find_number_l142_142311

theorem find_number (x : ℝ) (h : 3034 - x / 20.04 = 2984) : x = 1002 :=
sorry

end find_number_l142_142311


namespace sweetsies_remainder_l142_142701

-- Each definition used in Lean 4 statement should be directly from the conditions in a)
def number_of_sweetsies_in_one_bag (m : ℕ): Prop :=
  m % 8 = 5

theorem sweetsies_remainder (m : ℕ) (h : number_of_sweetsies_in_one_bag m) : 
  (4 * m) % 8 = 4 := by
  -- Proof will be provided here.
  sorry

end sweetsies_remainder_l142_142701


namespace truck_travel_distance_l142_142719

variable (d1 d2 g1 g2 : ℝ)
variable (rate : ℝ)

-- Define the conditions
axiom condition1 : d1 = 300
axiom condition2 : g1 = 10
axiom condition3 : rate = d1 / g1
axiom condition4 : g2 = 15

-- Define the goal
theorem truck_travel_distance : d2 = rate * g2 := by
  -- axiom assumption placeholder
  exact sorry

end truck_travel_distance_l142_142719


namespace ratio_of_square_sides_l142_142389

theorem ratio_of_square_sides
  (a b : ℝ) 
  (h1 : ∃ square1 : ℝ, square1 = 2 * a)
  (h2 : ∃ square2 : ℝ, square2 = 2 * b)
  (h3 : a ^ 2 - 4 * a * b - 5 * b ^ 2 = 0) :
  2 * a / 2 * b = 5 :=
by
  sorry

end ratio_of_square_sides_l142_142389


namespace speed_with_stream_l142_142888

-- Definitions for the conditions in part a
def Vm : ℕ := 8  -- Speed of the man in still water (in km/h)
def Vs : ℕ := Vm - 4  -- Speed of the stream (in km/h), derived from man's speed against the stream

-- The statement to prove the man's speed with the stream
theorem speed_with_stream : Vm + Vs = 12 := by sorry

end speed_with_stream_l142_142888


namespace kim_average_increase_l142_142616

noncomputable def avg (scores : List ℚ) : ℚ :=
  (scores.sum) / (scores.length)

theorem kim_average_increase :
  let scores_initial := [85, 89, 90, 92]  -- Initial scores
  let score_fifth := 95  -- Fifth score
  let original_average := avg scores_initial
  let new_average := avg (scores_initial ++ [score_fifth])
  new_average - original_average = 1.2 := by
  let scores_initial : List ℚ := [85, 89, 90, 92]
  let score_fifth : ℚ := 95
  let original_average : ℚ := avg scores_initial
  let new_average : ℚ := avg (scores_initial ++ [score_fifth])
  have : new_average - original_average = 1.2 := sorry
  exact this

end kim_average_increase_l142_142616


namespace maxwell_age_l142_142812

theorem maxwell_age (M : ℕ) (h1 : ∃ n : ℕ, n = M + 2) (h2 : ∃ k : ℕ, k = 4) (h3 : (M + 2) = 2 * 4) : M = 6 :=
sorry

end maxwell_age_l142_142812


namespace commute_times_l142_142286

theorem commute_times (x y : ℝ) 
  (h1 : x + y = 20)
  (h2 : (x - 10)^2 + (y - 10)^2 = 8) : |x - y| = 4 := 
sorry

end commute_times_l142_142286


namespace prove_RoseHasMoney_l142_142151
noncomputable def RoseHasMoney : Prop :=
  let cost_of_paintbrush := 2.40
  let cost_of_paints := 9.20
  let cost_of_easel := 6.50
  let total_cost := cost_of_paintbrush + cost_of_paints + cost_of_easel
  let additional_money_needed := 11
  let money_rose_has := total_cost - additional_money_needed
  money_rose_has = 7.10

theorem prove_RoseHasMoney : RoseHasMoney :=
  sorry

end prove_RoseHasMoney_l142_142151


namespace discount_percentage_l142_142590

variable {P P_b P_s : ℝ}
variable {D : ℝ}

theorem discount_percentage (P_s_eq_bought : P_s = 1.60 * P_b)
  (P_s_eq_original : P_s = 1.52 * P)
  (P_b_eq_discount : P_b = P * (1 - D)) :
  D = 0.05 := by
sorry

end discount_percentage_l142_142590


namespace probability_green_l142_142178

def total_marbles : ℕ := 100

def P_white : ℚ := 1 / 4

def P_red_or_blue : ℚ := 0.55

def P_sum : ℚ := 1

theorem probability_green :
  P_sum = P_white + P_red_or_blue + P_green →
  P_green = 0.2 :=
sorry

end probability_green_l142_142178


namespace divisibility_l142_142741

theorem divisibility {n A B k : ℤ} (h_n : n = 1000 * B + A) (h_k : k = A - B) :
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) :=
by
  sorry

end divisibility_l142_142741


namespace polar_r_eq_3_is_circle_l142_142858

theorem polar_r_eq_3_is_circle :
  ∀ θ : ℝ, ∃ x y : ℝ, (x, y) = (3 * Real.cos θ, 3 * Real.sin θ) ∧ x^2 + y^2 = 9 :=
by
  sorry

end polar_r_eq_3_is_circle_l142_142858


namespace circle_equation_range_l142_142383

theorem circle_equation_range (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + a + 1 = 0) → a < 4 := 
by 
  sorry

end circle_equation_range_l142_142383


namespace find_k_value_l142_142152

theorem find_k_value (x k : ℝ) (h : x = -3) (h_eq : k * (x - 2) - 4 = k - 2 * x) : k = -5/3 := by
  sorry

end find_k_value_l142_142152


namespace total_dolls_combined_l142_142643

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

end total_dolls_combined_l142_142643


namespace megatech_basic_astrophysics_degrees_l142_142444

def budget_allocation (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants: ℕ) :=
  100 - (microphotonics + home_electronics + food_additives + gm_microorganisms + industrial_lubricants)

noncomputable def degrees_for_astrophysics (percentage: ℕ) :=
  (percentage * 360) / 100

theorem megatech_basic_astrophysics_degrees (microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants: ℕ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 10 →
  gm_microorganisms = 29 →
  industrial_lubricants = 8 →
  degrees_for_astrophysics (budget_allocation microphotonics home_electronics food_additives gm_microorganisms industrial_lubricants) = 54 :=
by
  sorry

end megatech_basic_astrophysics_degrees_l142_142444


namespace x_add_inv_ge_two_l142_142961

theorem x_add_inv_ge_two (x : ℝ) (hx : x > 0) : x + (1 / x) ≥ 2 :=
by
  sorry

end x_add_inv_ge_two_l142_142961


namespace integer_not_natural_l142_142613

theorem integer_not_natural (n : ℕ) (a : ℝ) (b : ℝ) (x y z : ℝ) 
  (h₁ : x = (1 + a) ^ n) 
  (h₂ : y = (1 - a) ^ n) 
  (h₃ : z = a): 
  ∃ k : ℤ, (x - y) / z = ↑k ∧ (k < 0 ∨ k ≠ 0) :=
by 
  sorry

end integer_not_natural_l142_142613


namespace no_real_solutions_for_equation_l142_142542

theorem no_real_solutions_for_equation:
  ∀ x : ℝ, (3 * x / (x^2 + 2 * x + 4) + 4 * x / (x^2 - 4 * x + 5) = 1) →
  (¬(∃ x : ℝ, 3 * x / (x^2 + 2 * x + 4) + 4 * x / (x^2 - 4 * x + 5) = 1)) :=
by
  sorry

end no_real_solutions_for_equation_l142_142542


namespace center_of_circle_in_second_quadrant_l142_142612

theorem center_of_circle_in_second_quadrant (a : ℝ) (h : a > 12) :
  ∃ x y : ℝ, x^2 + y^2 + a * x - 2 * a * y + a^2 + 3 * a = 0 ∧ (-a / 2, a).2 > 0 ∧ (-a / 2, a).1 < 0 :=
by
  sorry

end center_of_circle_in_second_quadrant_l142_142612


namespace prove_total_payment_l142_142115

-- Define the conditions under which the problem is set
def monthly_subscription_cost : ℝ := 14
def split_ratio : ℝ := 0.5
def months_in_year : ℕ := 12

-- Define the target amount to prove
def total_payment_after_one_year : ℝ := 84

-- Theorem statement
theorem prove_total_payment
  (h1: monthly_subscription_cost = 14)
  (h2: split_ratio = 0.5)
  (h3: months_in_year = 12) :
  monthly_subscription_cost * split_ratio * months_in_year = total_payment_after_one_year := 
  by
  sorry

end prove_total_payment_l142_142115


namespace minimum_value_l142_142632

theorem minimum_value (a_n : ℕ → ℤ) (h : ∀ n, a_n n = n^2 - 8 * n + 5) : ∃ n, a_n n = -11 :=
by
  sorry

end minimum_value_l142_142632


namespace four_digit_numbers_with_8_or_3_l142_142394

theorem four_digit_numbers_with_8_or_3 :
  let total_four_digit_numbers := 9000
  let without_8_or_3_first := 7
  let without_8_or_3_rest := 8
  let numbers_without_8_or_3 := without_8_or_3_first * without_8_or_3_rest^3
  total_four_digit_numbers - numbers_without_8_or_3 = 5416 :=
by
  let total_four_digit_numbers := 9000
  let without_8_or_3_first := 7
  let without_8_or_3_rest := 8
  let numbers_without_8_or_3 := without_8_or_3_first * without_8_or_3_rest^3
  sorry

end four_digit_numbers_with_8_or_3_l142_142394


namespace find_b_value_l142_142880

theorem find_b_value (f : ℝ → ℝ) (f_inv : ℝ → ℝ) (b : ℝ) :
  (∀ x, f x = 1 / (3 * x + b)) →
  (∀ x, f_inv x = (2 - 3 * x) / (3 * x)) →
  b = -3 :=
by
  intros h1 h2
  sorry

end find_b_value_l142_142880


namespace general_term_formula_sum_first_n_terms_l142_142167

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (hS3 : S 3 = a 2 + 10 * a 1)
    (ha5 : a 5 = 9) : ∀ n, a n = 3^(n-2) := 
by
  sorry

theorem sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (hS3 : S 3 = a 2 + 10 * a 1)
    (ha5 : a 5 = 9) : ∀ n, S n = (3^(n-2)) / 2 - 1 / 18 := 
by
  sorry

end general_term_formula_sum_first_n_terms_l142_142167


namespace no_solution_for_x_l142_142323

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, (1 / (x - 4)) + (m / (x + 4)) ≠ ((m + 3) / (x^2 - 16))) ↔ (m = -1 ∨ m = 5 ∨ m = -1 / 3) :=
sorry

end no_solution_for_x_l142_142323


namespace point_on_number_line_l142_142814

theorem point_on_number_line (a : ℤ) (h : abs (a + 3) = 4) : a = 1 ∨ a = -7 := 
sorry

end point_on_number_line_l142_142814


namespace computation_result_l142_142802

theorem computation_result : 8 * (2 / 17) * 34 * (1 / 4) = 8 := by
  sorry

end computation_result_l142_142802


namespace Ferris_wheel_ticket_cost_l142_142206

theorem Ferris_wheel_ticket_cost
  (cost_rc : ℕ) (rides_rc : ℕ) (cost_c : ℕ) (rides_c : ℕ) (total_tickets : ℕ) (rides_fw : ℕ)
  (H1 : cost_rc = 4) (H2 : rides_rc = 3) (H3 : cost_c = 4) (H4 : rides_c = 2) (H5 : total_tickets = 21) (H6 : rides_fw = 1) :
  21 - (3 * 4 + 2 * 4) = 1 :=
by
  sorry

end Ferris_wheel_ticket_cost_l142_142206


namespace jonah_first_intermission_lemonade_l142_142104

theorem jonah_first_intermission_lemonade :
  ∀ (l1 l2 l3 l_total : ℝ)
  (h1 : l2 = 0.42)
  (h2 : l3 = 0.25)
  (h3 : l_total = 0.92)
  (h4 : l_total = l1 + l2 + l3),
  l1 = 0.25 :=
by sorry

end jonah_first_intermission_lemonade_l142_142104


namespace profit_percentage_of_cp_is_75_percent_of_sp_l142_142704

/-- If the cost price (CP) is 75% of the selling price (SP), then the profit percentage is 33.33% -/
theorem profit_percentage_of_cp_is_75_percent_of_sp (SP : ℝ) (h : SP > 0) (CP : ℝ) (hCP : CP = 0.75 * SP) :
  (SP - CP) / CP * 100 = 33.33 :=
by
  sorry

end profit_percentage_of_cp_is_75_percent_of_sp_l142_142704


namespace geom_sequence_general_formula_l142_142419

theorem geom_sequence_general_formula :
  ∃ (a : ℕ → ℝ) (a₁ q : ℝ), 
  (∀ n, a n = a₁ * q ^ n ∧ abs (q) < 1 ∧ ∑' i, a i = 3 ∧ ∑' i, (a i)^2 = (9 / 2)) →
  (∀ n, a n = 2 * ((1 / 3) ^ (n - 1))) :=
by sorry

end geom_sequence_general_formula_l142_142419


namespace smallest_number_of_rectangles_l142_142169

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l142_142169


namespace find_y_l142_142599

-- Definitions for the given conditions
def angle_sum_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

def right_triangle (A B : ℝ) : Prop :=
  A + B = 90

-- The main theorem to prove
theorem find_y 
  (angle_ABC : ℝ)
  (angle_BAC : ℝ)
  (angle_DCE : ℝ)
  (h1 : angle_ABC = 70)
  (h2 : angle_BAC = 50)
  (h3 : right_triangle angle_DCE 30)
  : 30 = 30 :=
sorry

end find_y_l142_142599


namespace a_minus_b_eq_neg_9_or_neg_1_l142_142754

theorem a_minus_b_eq_neg_9_or_neg_1 (a b : ℝ) (h₁ : |a| = 5) (h₂ : |b| = 4) (h₃ : a + b < 0) :
  a - b = -9 ∨ a - b = -1 :=
by
  sorry

end a_minus_b_eq_neg_9_or_neg_1_l142_142754


namespace ratio_proof_l142_142213

theorem ratio_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + y) / (x - 4 * y) = 3) :
    (x + 4 * y) / (4 * x - y) = 9 / 53 :=
  sorry

end ratio_proof_l142_142213


namespace remainder_of_m_div_5_l142_142962

theorem remainder_of_m_div_5 (m n : ℕ) (h1 : m = 15 * n - 1) (h2 : n > 0) : m % 5 = 4 :=
sorry

end remainder_of_m_div_5_l142_142962


namespace box_weight_without_balls_l142_142838

theorem box_weight_without_balls :
  let number_of_balls := 30
  let weight_per_ball := 0.36
  let total_weight_with_balls := 11.26
  let total_weight_of_balls := number_of_balls * weight_per_ball
  let weight_of_box := total_weight_with_balls - total_weight_of_balls
  weight_of_box = 0.46 :=
by 
  sorry

end box_weight_without_balls_l142_142838


namespace snow_white_seven_piles_l142_142886

def split_pile_action (piles : List ℕ) : Prop :=
  ∃ pile1 pile2, pile1 > 0 ∧ pile2 > 0 ∧ pile1 + pile2 + 1 ∈ piles

theorem snow_white_seven_piles :
  ∃ piles : List ℕ, piles.length = 7 ∧ ∀ pile ∈ piles, pile = 3 :=
sorry

end snow_white_seven_piles_l142_142886


namespace min_value_of_m_l142_142738

def ellipse (x y : ℝ) := (y^2 / 16) + (x^2 / 9) = 1
def line (x y m : ℝ) := y = x + m
def shortest_distance (d : ℝ) := d = Real.sqrt 2

theorem min_value_of_m :
  ∃ (m : ℝ), (∀ (x y : ℝ), ellipse x y → ∃ d, shortest_distance d ∧ line x y m) 
  ∧ ∀ m', m' < m → ¬(∃ (x y : ℝ), ellipse x y ∧ ∃ d, shortest_distance d ∧ line x y m') :=
sorry

end min_value_of_m_l142_142738


namespace tan_4530_l142_142344

noncomputable def tan_of_angle (deg : ℝ) : ℝ := Real.tan (deg * Real.pi / 180)

theorem tan_4530 : tan_of_angle 4530 = -1 / Real.sqrt 3 := sorry

end tan_4530_l142_142344


namespace desired_cost_per_pound_l142_142690

/-- 
Let $p_1 = 8$, $w_1 = 25$, $p_2 = 5$, and $w_2 = 50$ represent the prices and weights of two types of candies.
Calculate the desired cost per pound $p_m$ of the mixture.
-/
theorem desired_cost_per_pound 
  (p1 : ℝ) (w1 : ℝ) (p2 : ℝ) (w2 : ℝ) (p_m : ℝ) 
  (h1 : p1 = 8) (h2 : w1 = 25) (h3 : p2 = 5) (h4 : w2 = 50) :
  p_m = (p1 * w1 + p2 * w2) / (w1 + w2) → p_m = 6 :=
by 
  intros
  sorry

end desired_cost_per_pound_l142_142690


namespace batsman_average_after_17th_inning_l142_142774

theorem batsman_average_after_17th_inning 
    (A : ℕ)  -- assuming A (the average before the 17th inning) is a natural number
    (h₁ : 16 * A + 85 = 17 * (A + 3)) : 
    A + 3 = 37 := by
  sorry

end batsman_average_after_17th_inning_l142_142774


namespace cricket_run_rate_l142_142412

theorem cricket_run_rate (x : ℝ) (hx : 3.2 * x + 6.25 * 40 = 282) : x = 10 :=
by sorry

end cricket_run_rate_l142_142412


namespace solve_for_x_l142_142194

theorem solve_for_x : ∀ (x : ℝ), (2 * x + 3) / 5 = 11 → x = 26 :=
by {
  sorry
}

end solve_for_x_l142_142194


namespace product_of_solutions_is_zero_l142_142923

theorem product_of_solutions_is_zero :
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) -> x = 0)) -> true :=
by
  sorry

end product_of_solutions_is_zero_l142_142923


namespace dorothy_and_jemma_sales_l142_142073

theorem dorothy_and_jemma_sales :
  ∀ (frames_sold_by_jemma price_per_frame_jemma : ℕ)
  (price_per_frame_dorothy frames_sold_by_dorothy : ℚ)
  (total_sales_jemma total_sales_dorothy total_sales : ℚ),
  price_per_frame_jemma = 5 →
  frames_sold_by_jemma = 400 →
  price_per_frame_dorothy = price_per_frame_jemma / 2 →
  frames_sold_by_jemma = 2 * frames_sold_by_dorothy →
  total_sales_jemma = frames_sold_by_jemma * price_per_frame_jemma →
  total_sales_dorothy = frames_sold_by_dorothy * price_per_frame_dorothy →
  total_sales = total_sales_jemma + total_sales_dorothy →
  total_sales = 2500 := by
  sorry

end dorothy_and_jemma_sales_l142_142073


namespace max_value_of_n_l142_142212

theorem max_value_of_n (A B : ℤ) (h1 : A * B = 48) : 
  ∃ n, (∀ n', (∃ A' B', (A' * B' = 48) ∧ (n' = 2 * B' + 3 * A')) → n' ≤ n) ∧ n = 99 :=
by
  sorry

end max_value_of_n_l142_142212


namespace ordered_pair_count_l142_142579

theorem ordered_pair_count :
  (∃ (bc : ℕ × ℕ), bc.1 > 0 ∧ bc.2 > 0 ∧ bc.1 ^ 4 - 4 * bc.2 ≤ 0 ∧ bc.2 ^ 4 - 4 * bc.1 ≤ 0) ∧
  ∀ (bc1 bc2 : ℕ × ℕ),
    bc1 ≠ bc2 →
    bc1.1 > 0 ∧ bc1.2 > 0 ∧ bc1.1 ^ 4 - 4 * bc1.2 ≤ 0 ∧ bc1.2 ^ 4 - 4 * bc1.1 ≤ 0 →
    bc2.1 > 0 ∧ bc2.2 > 0 ∧ bc2.1 ^ 4 - 4 * bc2.2 ≤ 0 ∧ bc2.2 ^ 4 - 4 * bc2.1 ≤ 0 →
    false
:=
sorry

end ordered_pair_count_l142_142579


namespace speed_in_km_per_hr_l142_142914

noncomputable def side : ℝ := 40
noncomputable def time : ℝ := 64

-- Theorem statement
theorem speed_in_km_per_hr (side : ℝ) (time : ℝ) (h₁ : side = 40) (h₂ : time = 64) : 
  (4 * side * 3600) / (time * 1000) = 9 := by
  rw [h₁, h₂]
  sorry

end speed_in_km_per_hr_l142_142914


namespace tank_capacity_l142_142645

theorem tank_capacity :
  ∃ T : ℝ, (5/8) * T + 12 = (11/16) * T ∧ T = 192 :=
sorry

end tank_capacity_l142_142645


namespace sin_neg_seven_pi_over_three_correct_l142_142239

noncomputable def sin_neg_seven_pi_over_three : Prop :=
  (Real.sin (-7 * Real.pi / 3) = - (Real.sqrt 3 / 2))

theorem sin_neg_seven_pi_over_three_correct : sin_neg_seven_pi_over_three := 
by
  sorry

end sin_neg_seven_pi_over_three_correct_l142_142239


namespace cube_construction_possible_l142_142916

theorem cube_construction_possible (n : ℕ) : (∃ k : ℕ, n = 12 * k) ↔ ∃ V : ℕ, (n ^ 3) = 12 * V := by
sorry

end cube_construction_possible_l142_142916


namespace triangle_side_length_l142_142510

theorem triangle_side_length 
  (a b c : ℝ) 
  (cosA : ℝ) 
  (h1: a = Real.sqrt 5) 
  (h2: c = 2) 
  (h3: cosA = 2 / 3) 
  (h4: a^2 = b^2 + c^2 - 2 * b * c * cosA) : 
  b = 3 := 
by 
  sorry

end triangle_side_length_l142_142510


namespace smallest_b_l142_142238

theorem smallest_b (b: ℕ) (h1: b > 3) (h2: ∃ n: ℕ, n^3 = 2 * b + 3) : b = 12 :=
sorry

end smallest_b_l142_142238


namespace gift_distribution_l142_142038

theorem gift_distribution :
  let bags := [1, 2, 3, 4, 5]
  let num_people := 4
  ∃ d: ℕ, d = 96 := by
  -- Proof to be completed
  sorry

end gift_distribution_l142_142038


namespace eugene_total_cost_l142_142796

variable (TshirtCost PantCost ShoeCost : ℕ)
variable (NumTshirts NumPants NumShoes Discount : ℕ)

theorem eugene_total_cost
  (hTshirtCost : TshirtCost = 20)
  (hPantCost : PantCost = 80)
  (hShoeCost : ShoeCost = 150)
  (hNumTshirts : NumTshirts = 4)
  (hNumPants : NumPants = 3)
  (hNumShoes : NumShoes = 2)
  (hDiscount : Discount = 10) :
  TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes - (TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes) * Discount / 100 = 558 := by
  sorry

end eugene_total_cost_l142_142796


namespace simplify_expression_l142_142706

theorem simplify_expression (α : ℝ) :
  (1 + 2 * Real.sin (2 * α) * Real.cos (2 * α) - (2 * Real.cos (2 * α)^2 - 1)) /
  (1 + 2 * Real.sin (2 * α) * Real.cos (2 * α) + (2 * Real.cos (2 * α)^2 - 1)) = Real.tan (2 * α) :=
by
  sorry

end simplify_expression_l142_142706


namespace pete_and_raymond_spent_together_l142_142689

    def value_nickel : ℕ := 5
    def value_dime : ℕ := 10
    def value_quarter : ℕ := 25

    def pete_nickels_spent : ℕ := 4
    def pete_dimes_spent : ℕ := 3
    def pete_quarters_spent : ℕ := 2

    def raymond_initial : ℕ := 250
    def raymond_nickels_left : ℕ := 5
    def raymond_dimes_left : ℕ := 7
    def raymond_quarters_left : ℕ := 4
    
    def total_spent : ℕ := 155

    theorem pete_and_raymond_spent_together :
      (pete_nickels_spent * value_nickel + pete_dimes_spent * value_dime + pete_quarters_spent * value_quarter)
      + (raymond_initial - (raymond_nickels_left * value_nickel + raymond_dimes_left * value_dime + raymond_quarters_left * value_quarter))
      = total_spent :=
      by
        sorry
    
end pete_and_raymond_spent_together_l142_142689


namespace number_of_persons_l142_142982

-- Definitions of the given conditions
def average : ℕ := 15
def average_5 : ℕ := 14
def sum_5 : ℕ := 5 * average_5
def average_9 : ℕ := 16
def sum_9 : ℕ := 9 * average_9
def age_15th : ℕ := 41
def total_sum : ℕ := sum_5 + sum_9 + age_15th

-- The main theorem stating the equivalence
theorem number_of_persons (N : ℕ) (h_average : average * N = total_sum) : N = 17 :=
by
  -- Proof goes here
  sorry

end number_of_persons_l142_142982


namespace budget_allocation_genetically_modified_microorganisms_l142_142969

theorem budget_allocation_genetically_modified_microorganisms :
  let microphotonics := 14
  let home_electronics := 19
  let food_additives := 10
  let industrial_lubricants := 8
  let total_percentage := 100
  let basic_astrophysics_percentage := 25
  let known_percentage := microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics_percentage
  let genetically_modified_microorganisms := total_percentage - known_percentage
  genetically_modified_microorganisms = 24 := 
by
  sorry

end budget_allocation_genetically_modified_microorganisms_l142_142969


namespace three_digit_number_problem_l142_142197

theorem three_digit_number_problem (c d : ℕ) (h1 : 400 + c*10 + 1 = 786 - (300 + d*10 + 5)) (h2 : (300 + d*10 + 5) % 7 = 0) : c + d = 8 := 
sorry

end three_digit_number_problem_l142_142197


namespace sufficient_but_not_necessary_l142_142403

variable (x y : ℝ)

theorem sufficient_but_not_necessary (x_gt_y_gt_zero : x > y ∧ y > 0) : (x / y > 1) :=
by
  sorry

end sufficient_but_not_necessary_l142_142403


namespace customer_count_l142_142529

theorem customer_count :
  let initial_customers := 13
  let customers_after_first_leave := initial_customers - 5
  let customers_after_new_arrival := customers_after_first_leave + 4
  let customers_after_group_join := customers_after_new_arrival + 8
  let final_customers := customers_after_group_join - 6
  final_customers = 14 :=
by
  sorry

end customer_count_l142_142529


namespace mixed_number_calculation_l142_142112

theorem mixed_number_calculation :
  (481 + 1/6) + (265 + 1/12) + (904 + 1/20) - (184 + 29/30) - (160 + 41/42) - (703 + 55/56) = 603 + 3/8 := 
sorry

end mixed_number_calculation_l142_142112


namespace recurrent_sequence_solution_l142_142511

theorem recurrent_sequence_solution (a : ℕ → ℕ) : 
  (a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + 2^n) →
  (∀ n, n ≥ 1 → a n = (2 * n - 1) * 2^(n - 1)) :=
by
  sorry

end recurrent_sequence_solution_l142_142511


namespace equilateral_triangle_iff_l142_142130

theorem equilateral_triangle_iff (a b c : ℝ) :
  a^2 + b^2 + c^2 = a*b + b*c + c*a ↔ a = b ∧ b = c :=
sorry

end equilateral_triangle_iff_l142_142130


namespace intersection_A_B_l142_142201

open Set

-- Conditions given in the problem
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Statement to prove, no proof needed
theorem intersection_A_B : A ∩ B = {1, 2} := 
sorry

end intersection_A_B_l142_142201


namespace total_boys_in_class_l142_142994

/-- 
  Given 
    - n + 1 positions in a circle, where n is the number of boys and 1 position for the teacher.
    - The boy at the 6th position is exactly opposite to the boy at the 16th position.
  Prove that the total number of boys in the class is 20.
-/
theorem total_boys_in_class (n : ℕ) (h1 : n + 1 > 16) (h2 : (6 + 10) % (n + 1) = 16):
  n = 20 := 
by 
  sorry

end total_boys_in_class_l142_142994


namespace find_p_q_l142_142783

theorem find_p_q (p q : ℤ)
  (h : (5 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 5) = 20 * d^4 + 11 * d^3 - 45 * d^2 - 20 * d + 25) :
  p + q = 3 :=
sorry

end find_p_q_l142_142783


namespace problem_solution_l142_142294

noncomputable def f (x : ℝ) := 2 * Real.sin x + x^3 + 1

theorem problem_solution (a : ℝ) (h : f a = 3) : f (-a) = -1 := by
  sorry

end problem_solution_l142_142294


namespace toms_animal_robots_l142_142301

theorem toms_animal_robots (h : ∀ (m t : ℕ), t = 2 * m) (hmichael : 8 = m) : ∃ t, t = 16 := 
by
  sorry

end toms_animal_robots_l142_142301


namespace problem_statement_l142_142941

theorem problem_statement (a b c x : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0)
  (eq1 : (a * x^4 / b * c)^3 = x^3)
  (sum_eq : a + b + c = 9) :
  (x = 1 ∨ x = -1) ∧ a = 1 ∧ b = 4 ∧ c = 4 :=
by
  sorry

end problem_statement_l142_142941


namespace part1_l142_142977

theorem part1 (P Q R : Polynomial ℝ) : 
  ¬ ∃ (P Q R : Polynomial ℝ), (∀ x y z : ℝ, (x - y + 1)^3 * P.eval x + (y - z - 1)^3 * Q.eval y + (z - 2 * x + 1)^3 * R.eval z = 1) := sorry

end part1_l142_142977


namespace part1_part2_l142_142377
open Real

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (hm : m > 1) : ∃ x : ℝ, f x = 4 / (m - 1) + m :=
by
  sorry

end part1_part2_l142_142377


namespace tangent_line_ellipse_l142_142033

variable {a b x x0 y y0 : ℝ}

theorem tangent_line_ellipse (h : a * x0^2 + b * y0^2 = 1) :
  a * x0 * x + b * y0 * y = 1 :=
sorry

end tangent_line_ellipse_l142_142033


namespace find_c_l142_142546

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 8)) : c = 17 / 3 := 
by
  -- Add the necessary assumptions and let Lean verify these assumptions.
  have b_eq : 3 * b = 8 := sorry
  have b_val : b = 8 / 3 := sorry
  have h_coeff : c = b + 3 := sorry
  exact h_coeff.trans (by rw [b_val]; norm_num)

end find_c_l142_142546


namespace abs_opposite_sign_eq_sum_l142_142896

theorem abs_opposite_sign_eq_sum (a b : ℤ) (h : (|a + 1| * |b + 2| < 0)) : a + b = -3 :=
sorry

end abs_opposite_sign_eq_sum_l142_142896


namespace other_continents_passengers_l142_142549

def passengers_from_other_continents (T N_A E A As : ℕ) : ℕ := T - (N_A + E + A + As)

theorem other_continents_passengers :
  passengers_from_other_continents 108 (108 / 12) (108 / 4) (108 / 9) (108 / 6) = 42 :=
by
  -- Proof goes here
  sorry

end other_continents_passengers_l142_142549


namespace even_function_properties_l142_142317

theorem even_function_properties 
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_increasing : ∀ x y : ℝ, 5 ≤ x ∧ x ≤ y ∧ y ≤ 7 → f x ≤ f y)
  (h_min_value : ∀ x : ℝ, 5 ≤ x ∧ x ≤ 7 → 6 ≤ f x) :
  (∀ x y : ℝ, -7 ≤ x ∧ x ≤ y ∧ y ≤ -5 → f y ≤ f x) ∧ (∀ x : ℝ, -7 ≤ x ∧ x ≤ -5 → 6 ≤ f x) :=
by
  sorry

end even_function_properties_l142_142317


namespace james_drove_75_miles_l142_142435

noncomputable def james_total_distance : ℝ :=
  let speed1 := 30  -- mph
  let time1 := 0.5  -- hours
  let speed2 := 2 * speed1
  let time2 := 2 * time1
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  distance1 + distance2

theorem james_drove_75_miles : james_total_distance = 75 := by 
  sorry

end james_drove_75_miles_l142_142435


namespace tangent_neg_five_pi_six_eq_one_over_sqrt_three_l142_142826

noncomputable def tangent_neg_five_pi_six : Real :=
  Real.tan (-5 * Real.pi / 6)

theorem tangent_neg_five_pi_six_eq_one_over_sqrt_three :
  tangent_neg_five_pi_six = 1 / Real.sqrt 3 := by
  sorry

end tangent_neg_five_pi_six_eq_one_over_sqrt_three_l142_142826


namespace closed_under_all_operations_l142_142726

structure sqrt2_num where
  re : ℚ
  im : ℚ

namespace sqrt2_num

def add (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re + y.re, x.im + y.im⟩

def subtract (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re - y.re, x.im - y.im⟩

def multiply (x y : sqrt2_num) : sqrt2_num :=
  ⟨x.re * y.re + 2 * x.im * y.im, x.re * y.im + x.im * y.re⟩

def divide (x y : sqrt2_num) : sqrt2_num :=
  let denom := y.re^2 - 2 * y.im^2
  ⟨(x.re * y.re - 2 * x.im * y.im) / denom, (x.im * y.re - x.re * y.im) / denom⟩

theorem closed_under_all_operations (a b c d : ℚ) :
  ∃ (e f : ℚ), 
    add ⟨a, b⟩ ⟨c, d⟩ = ⟨e, f⟩ ∧ 
    ∃ (g h : ℚ), 
    subtract ⟨a, b⟩ ⟨c, d⟩ = ⟨g, h⟩ ∧ 
    ∃ (i j : ℚ), 
    multiply ⟨a, b⟩ ⟨c, d⟩ = ⟨i, j⟩ ∧ 
    ∃ (k l : ℚ), 
    divide ⟨a, b⟩ ⟨c, d⟩ = ⟨k, l⟩ := by
  sorry

end sqrt2_num

end closed_under_all_operations_l142_142726


namespace find_m_if_f_monotonic_l142_142154

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ :=
  4 * x^3 + m * x^2 + (m - 3) * x + n

def is_monotonically_increasing_on_ℝ (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≤ x2 → f x1 ≤ f x2

theorem find_m_if_f_monotonic (m n : ℝ)
  (h : is_monotonically_increasing_on_ℝ (f m n)) :
  m = 6 :=
sorry

end find_m_if_f_monotonic_l142_142154


namespace john_speed_l142_142870

def johns_speed (race_distance_miles next_fastest_guy_time_min won_by_min : ℕ) : ℕ :=
    let john_time_min := next_fastest_guy_time_min - won_by_min
    let john_time_hr := john_time_min / 60
    race_distance_miles / john_time_hr

theorem john_speed (race_distance_miles next_fastest_guy_time_min won_by_min : ℕ)
    (h1 : race_distance_miles = 5) (h2 : next_fastest_guy_time_min = 23) (h3 : won_by_min = 3) : 
    johns_speed race_distance_miles next_fastest_guy_time_min won_by_min = 15 := 
by
    sorry

end john_speed_l142_142870


namespace general_term_arithmetic_sum_terms_geometric_l142_142707

section ArithmeticSequence

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Conditions for Part 1
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) : Prop :=
  S 5 - S 2 = 195 ∧ d = -2 ∧
  ∀ n, S n = n * (a 1 + (n - 1) * (d / 2))

-- Prove the general term formula for the sequence {a_n}
theorem general_term_arithmetic (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) 
    (h : sum_arithmetic_sequence S a d) : 
    ∀ n, a n = -2 * n + 73 :=
sorry

end ArithmeticSequence


section GeometricSequence

variables {b : ℕ → ℝ} {n : ℕ} {T : ℕ → ℝ} {a : ℕ → ℝ}

-- Conditions for Part 2
def sum_geometric_sequence (b : ℕ → ℝ) (T : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 1 = 13 ∧ b 2 = 65 ∧ a 4 = 65

-- Prove the sum of the first n terms for the sequence {b_n}
theorem sum_terms_geometric (b : ℕ → ℝ) (T : ℕ → ℝ) (a : ℕ → ℝ)
    (h : sum_geometric_sequence b T a) : 
    ∀ n, T n = 13 * (5^n - 1) / 4 :=
sorry

end GeometricSequence

end general_term_arithmetic_sum_terms_geometric_l142_142707


namespace log_base_9_of_x_cubed_is_3_l142_142842

theorem log_base_9_of_x_cubed_is_3 
  (x : Real) 
  (hx : x = 9.000000000000002) : 
  Real.logb 9 (x^3) = 3 := 
by 
  sorry

end log_base_9_of_x_cubed_is_3_l142_142842


namespace password_probability_l142_142096

theorem password_probability : 
  (5/10) * (51/52) * (9/10) = 459 / 1040 := by
  sorry

end password_probability_l142_142096


namespace positive_triple_l142_142006

theorem positive_triple
  (a b c : ℝ)
  (h1 : a + b + c > 0)
  (h2 : ab + bc + ca > 0)
  (h3 : abc > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_triple_l142_142006


namespace necessary_but_not_sufficient_l142_142753

theorem necessary_but_not_sufficient (x y : ℝ) : 
  (x < 0 ∨ y < 0) → x + y < 0 :=
sorry

end necessary_but_not_sufficient_l142_142753


namespace problem_equiv_l142_142457

theorem problem_equiv {a : ℤ} : (a^2 ≡ 9 [ZMOD 10]) ↔ (a ≡ 3 [ZMOD 10] ∨ a ≡ -3 [ZMOD 10] ∨ a ≡ 7 [ZMOD 10] ∨ a ≡ -7 [ZMOD 10]) :=
sorry

end problem_equiv_l142_142457


namespace myOperation_identity_l142_142756

variable {R : Type*} [LinearOrderedField R]

def myOperation (a b : R) : R := (a - b) ^ 2

theorem myOperation_identity (x y : R) : myOperation ((x - y) ^ 2) ((y - x) ^ 2) = 0 := 
by 
  sorry

end myOperation_identity_l142_142756


namespace correctPairsAreSkating_l142_142089

def Friend := String
def Brother := String

structure SkatingPair where
  gentleman : Friend
  lady : Friend

-- Define the list of friends with their brothers
def friends : List Friend := ["Lyusya Egorova", "Olya Petrova", "Inna Krymova", "Anya Vorobyova"]
def brothers : List Brother := ["Andrey Egorov", "Serezha Petrov", "Dima Krymov", "Yura Vorobyov"]

-- Condition: The skating pairs such that gentlemen are taller than ladies and no one skates with their sibling
noncomputable def skatingPairs : List SkatingPair :=
  [ {gentleman := "Yura Vorobyov", lady := "Lyusya Egorova"},
    {gentleman := "Andrey Egorov", lady := "Olya Petrova"},
    {gentleman := "Serezha Petrov", lady := "Inna Krymova"},
    {gentleman := "Dima Krymov", lady := "Anya Vorobyova"} ]

-- Proving that the pairs are exactly as specified.
theorem correctPairsAreSkating :
  skatingPairs = 
    [ {gentleman := "Yura Vorobyov", lady := "Lyusya Egorova"},
      {gentleman := "Andrey Egorov", lady := "Olya Petrova"},
      {gentleman := "Serezha Petrov", lady := "Inna Krymova"},
      {gentleman := "Dima Krymov", lady := "Anya Vorobyova"} ] :=
by
  sorry

end correctPairsAreSkating_l142_142089


namespace random_event_is_eventA_l142_142804

-- Definitions of conditions
def eventA : Prop := true  -- Tossing a coin and it lands either heads up or tails up is a random event
def eventB : Prop := (∀ (a b : ℝ), (b * a = b * a))  -- The area of a rectangle with sides of length a and b is ab is a certain event
def eventC : Prop := ∃ (defective_items : ℕ), (defective_items / 100 = 10 / 100)  -- Drawing 2 defective items from 100 parts with 10% defective parts is uncertain
def eventD : Prop := false -- Scoring 105 points in a regular 100-point system exam is an impossible event

-- The proof problem statement
theorem random_event_is_eventA : eventA ∧ ¬eventB ∧ ¬eventC ∧ ¬eventD := 
sorry

end random_event_is_eventA_l142_142804


namespace first_operation_result_l142_142360

def pattern (x y : ℕ) : ℕ :=
  if (x, y) = (3, 7) then 27
  else if (x, y) = (4, 5) then 32
  else if (x, y) = (5, 8) then 60
  else if (x, y) = (6, 7) then 72
  else if (x, y) = (7, 8) then 98
  else 26

theorem first_operation_result : pattern 2 3 = 26 := by
  sorry

end first_operation_result_l142_142360


namespace min_reciprocal_sum_l142_142854

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x) + (1/y) = 3 + 2 * Real.sqrt 2 :=
sorry

end min_reciprocal_sum_l142_142854


namespace rectangle_side_ratio_square_l142_142217

noncomputable def ratio_square (a b : ℝ) : ℝ :=
(a / b) ^ 2

theorem rectangle_side_ratio_square (a b : ℝ) (h : (a - b) / (a + b) = 1 / 3) : 
  ratio_square a b = 4 := by
  sorry

end rectangle_side_ratio_square_l142_142217


namespace miles_run_on_tuesday_l142_142205

-- Defining the distances run on specific days
def distance_monday : ℝ := 4.2
def distance_wednesday : ℝ := 3.6
def distance_thursday : ℝ := 4.4

-- Average distance run on each of the days Terese runs
def average_distance : ℝ := 4
-- Number of days Terese runs
def running_days : ℕ := 4

-- Defining the total distance calculated using the average distance and number of days
def total_distance : ℝ := average_distance * running_days

-- Defining the total distance run on Monday, Wednesday, and Thursday
def total_other_days : ℝ := distance_monday + distance_wednesday + distance_thursday

-- The distance run on Tuesday can be defined as the difference between the total distance and the total distance on other days
theorem miles_run_on_tuesday : 
  total_distance - total_other_days = 3.8 :=
by
  sorry

end miles_run_on_tuesday_l142_142205


namespace line_circle_intersect_l142_142696

theorem line_circle_intersect {a : ℝ} :
  ∃ P : ℝ × ℝ, (P.1, P.2) = (-2, 0) ∧ (a * P.1 - P.2 + 2 * a = 0) ∧ (P.1^2 + P.2^2 < 9) :=
by
  use (-2, 0)
  sorry

end line_circle_intersect_l142_142696


namespace range_of_a_l142_142788

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2)*x^2 + (2 * a - 1) * x + 6 > 0 ↔ x = 3 → true) ∧
  (∀ x : ℝ, (a - 2)*x^2 + (2 * a - 1) * x + 6 > 0 ↔ x = 5 → false) →
  1 < a ∧ a ≤ 7 / 5 :=
by
  sorry

end range_of_a_l142_142788


namespace cookies_ratio_l142_142872

theorem cookies_ratio (total_cookies sells_mr_stone brock_buys left_cookies katy_buys : ℕ)
  (h1 : total_cookies = 5 * 12)
  (h2 : sells_mr_stone = 2 * 12)
  (h3 : brock_buys = 7)
  (h4 : left_cookies = 15)
  (h5 : total_cookies - sells_mr_stone - brock_buys - left_cookies = katy_buys) :
  katy_buys / brock_buys = 2 :=
by sorry

end cookies_ratio_l142_142872


namespace range_of_m_l142_142045

noncomputable def f : ℝ → ℝ := sorry

lemma function_symmetric {x : ℝ} : f (2 + x) = f (-x) := sorry

lemma f_decreasing_on_pos_halfline {x y : ℝ} (hx : x ≥ 1) (hy : y ≥ 1) (hxy : x < y) : f x ≥ f y := sorry

theorem range_of_m {m : ℝ} (h : f (1 - m) < f m) : m > (1 / 2) := sorry

end range_of_m_l142_142045


namespace fewer_trombone_than_trumpet_l142_142183

theorem fewer_trombone_than_trumpet 
  (flute_players : ℕ)
  (trumpet_players : ℕ)
  (trombone_players : ℕ)
  (drummers : ℕ)
  (clarinet_players : ℕ)
  (french_horn_players : ℕ)
  (total_members : ℕ) :
  flute_players = 5 →
  trumpet_players = 3 * flute_players →
  clarinet_players = 2 * flute_players →
  drummers = trombone_players + 11 →
  french_horn_players = trombone_players + 3 →
  total_members = flute_players + clarinet_players + trumpet_players + trombone_players + drummers + french_horn_players →
  total_members = 65 →
  trombone_players = 7 ∧ trumpet_players - trombone_players = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3] at h6
  sorry

end fewer_trombone_than_trumpet_l142_142183


namespace f_prime_neg1_l142_142723

def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

def f' (a b c x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem f_prime_neg1 (a b c : ℝ) (h : f' a b c 1 = 2) : f' a b c (-1) = -2 :=
by
  sorry

end f_prime_neg1_l142_142723


namespace range_of_m_l142_142303

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (mx-1)*(x-2) > 0 ↔ (1/m < x ∧ x < 2)) → m < 0 :=
by
  sorry

end range_of_m_l142_142303


namespace eight_bags_weight_l142_142353

theorem eight_bags_weight
  (bags_weight : ℕ → ℕ)
  (h1 : bags_weight 12 = 24) :
  bags_weight 8 = 16 :=
  sorry

end eight_bags_weight_l142_142353


namespace find_number_of_books_l142_142773

-- Define the constants and equation based on the conditions
def price_paid_per_book : ℕ := 11
def price_sold_per_book : ℕ := 25
def total_difference : ℕ := 210

def books_equation (x : ℕ) : Prop :=
  (price_sold_per_book * x) - (price_paid_per_book * x) = total_difference

-- The theorem statement that needs to be proved
theorem find_number_of_books (x : ℕ) (h : books_equation x) : 
  x = 15 :=
sorry

end find_number_of_books_l142_142773


namespace quadratic_roots_range_l142_142716

theorem quadratic_roots_range (k : ℝ) : (x^2 - 6*x + k = 0) → k < 9 := 
by
  sorry

end quadratic_roots_range_l142_142716


namespace probability_of_five_dice_all_same_l142_142866

theorem probability_of_five_dice_all_same : 
  (6 / (6 ^ 5) = 1 / 1296) :=
by
  sorry

end probability_of_five_dice_all_same_l142_142866


namespace find_positive_integer_l142_142834

theorem find_positive_integer (n : ℕ) (h1 : 100 % n = 3) (h2 : 197 % n = 3) : n = 97 := 
sorry

end find_positive_integer_l142_142834


namespace sin_inequality_solution_set_l142_142563

theorem sin_inequality_solution_set : 
  {x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x < - Real.sqrt 3 / 2} =
  {x : ℝ | (4 * Real.pi / 3) < x ∧ x < (5 * Real.pi / 3)} := by
  sorry

end sin_inequality_solution_set_l142_142563


namespace solution_set_abs_inequality_l142_142507

theorem solution_set_abs_inequality : {x : ℝ | |x - 2| < 1} = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

end solution_set_abs_inequality_l142_142507


namespace steve_fraction_of_day_in_school_l142_142795

theorem steve_fraction_of_day_in_school :
  let total_hours : ℕ := 24
  let sleep_fraction : ℚ := 1 / 3
  let assignment_fraction : ℚ := 1 / 12
  let family_hours : ℕ := 10
  let sleep_hours : ℚ := sleep_fraction * total_hours
  let assignment_hours : ℚ := assignment_fraction * total_hours
  let accounted_hours : ℚ := sleep_hours + assignment_hours + family_hours
  let school_hours : ℚ := total_hours - accounted_hours
  (school_hours / total_hours) = (1 / 6) :=
by
  let total_hours : ℕ := 24
  let sleep_fraction : ℚ := 1 / 3
  let assignment_fraction : ℚ := 1 / 12
  let family_hours : ℕ := 10
  let sleep_hours : ℚ := sleep_fraction * total_hours
  let assignment_hours : ℚ := assignment_fraction * total_hours
  let accounted_hours : ℚ := sleep_hours + assignment_hours + family_hours
  let school_hours : ℚ := total_hours - accounted_hours
  have : (school_hours / total_hours) = (1 / 6) := sorry
  exact this

end steve_fraction_of_day_in_school_l142_142795


namespace largest_sum_of_watch_digits_l142_142343

theorem largest_sum_of_watch_digits : ∃ s : ℕ, s = 23 ∧ 
  (∀ h m : ℕ, h < 24 → m < 60 → s ≤ (h / 10 + h % 10 + m / 10 + m % 10)) :=
by
  sorry

end largest_sum_of_watch_digits_l142_142343


namespace find_value_l142_142830

variable (a b c : Int)

-- Conditions from the problem
axiom abs_a_eq_two : |a| = 2
axiom b_eq_neg_seven : b = -7
axiom neg_c_eq_neg_five : -c = -5

-- Proof problem
theorem find_value : a^2 + (-b) + (-c) = 6 := by
  sorry

end find_value_l142_142830


namespace area_difference_l142_142656

theorem area_difference (radius1 radius2 : ℝ) (pi : ℝ) (h1 : radius1 = 15) (h2 : radius2 = 14 / 2) :
  pi * radius1 ^ 2 - pi * radius2 ^ 2 = 176 * pi :=
by 
  sorry

end area_difference_l142_142656


namespace samantha_coins_value_l142_142041

theorem samantha_coins_value (n d : ℕ) (h1 : n + d = 25) 
    (original_value : ℕ := 250 - 5 * n) 
    (swapped_value : ℕ := 125 + 5 * n)
    (h2 : swapped_value = original_value + 100) : original_value = 140 := 
by
  sorry

end samantha_coins_value_l142_142041


namespace sum_possible_distances_l142_142053

theorem sum_possible_distances {A B : ℝ} (hAB : |A - B| = 2) (hA : |A| = 3) : 
  (if A = 3 then |B + 2| + |B - 2| else |B + 4| + |B - 4|) = 12 :=
by
  sorry

end sum_possible_distances_l142_142053


namespace triangles_form_even_square_l142_142680

-- Given conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def triangle_area (b h : ℕ) : ℚ :=
  (b * h) / 2

-- Statement of the problem
theorem triangles_form_even_square (n : ℕ) :
  (∀ t : Fin n, is_right_triangle 3 4 5 ∧ triangle_area 3 4 = 6) →
  (∃ a : ℕ, a^2 = 6 * n) →
  Even n :=
by
  sorry

end triangles_form_even_square_l142_142680


namespace focus_coordinates_correct_l142_142954
noncomputable def ellipse_focus : Real × Real :=
  let center : Real × Real := (4, -1)
  let a : Real := 4
  let b : Real := 1.5
  let c : Real := Real.sqrt (a^2 - b^2)
  (center.1 + c, center.2)

theorem focus_coordinates_correct : 
  ellipse_focus = (7.708, -1) := 
by 
  sorry

end focus_coordinates_correct_l142_142954


namespace students_answered_both_correctly_l142_142361

theorem students_answered_both_correctly
  (enrolled : ℕ)
  (did_not_take_test : ℕ)
  (answered_q1_correctly : ℕ)
  (answered_q2_correctly : ℕ)
  (total_students_answered_both : ℕ) :
  enrolled = 29 →
  did_not_take_test = 5 →
  answered_q1_correctly = 19 →
  answered_q2_correctly = 24 →
  total_students_answered_both = 19 :=
by
  intros
  sorry

end students_answered_both_correctly_l142_142361


namespace bob_grade_is_35_l142_142372

variable (J : ℕ) (S : ℕ) (B : ℕ)

-- Define Jenny's grade, Jason's grade based on Jenny's, and Bob's grade based on Jason's
def jennyGrade := 95
def jasonGrade := J - 25
def bobGrade := S / 2

-- Theorem to prove Bob's grade is 35 given the conditions
theorem bob_grade_is_35 (h1 : J = 95) (h2 : S = J - 25) (h3 : B = S / 2) : B = 35 :=
by
  -- Placeholder for the proof
  sorry

end bob_grade_is_35_l142_142372


namespace tangent_sufficient_but_not_necessary_condition_l142_142578

noncomputable def tangent_condition (m : ℝ) : Prop :=
  let line := fun (x y : ℝ) => x + y - m = 0
  let circle := fun (x y : ℝ) => (x - 1) ^ 2 + (y - 1) ^ 2 = 2
  ∃ (x y: ℝ), line x y ∧ circle x y -- A line and circle are tangent if they intersect exactly at one point

theorem tangent_sufficient_but_not_necessary_condition (m : ℝ) :
  (tangent_condition m) ↔ (m = 0 ∨ m = 4) := by
  sorry

end tangent_sufficient_but_not_necessary_condition_l142_142578


namespace small_bottles_sold_percentage_l142_142149

theorem small_bottles_sold_percentage
  (small_bottles : ℕ) (big_bottles : ℕ) (percent_sold_big_bottles : ℝ)
  (remaining_bottles : ℕ) (percent_sold_small_bottles : ℝ) :
  small_bottles = 6000 ∧
  big_bottles = 14000 ∧
  percent_sold_big_bottles = 0.23 ∧
  remaining_bottles = 15580 ∧ 
  percent_sold_small_bottles / 100 * 6000 + 0.23 * 14000 + remaining_bottles = small_bottles + big_bottles →
  percent_sold_small_bottles = 37 := 
by
  intros
  exact sorry

end small_bottles_sold_percentage_l142_142149


namespace reduced_price_per_kg_of_oil_l142_142679

/-- The reduced price per kg of oil is approximately Rs. 48 -
given a 30% reduction in price and the ability to buy 5 kgs more
for Rs. 800. -/
theorem reduced_price_per_kg_of_oil
  (P R : ℝ)
  (h1 : R = 0.70 * P)
  (h2 : 800 / R = (800 / P) + 5) : 
  R = 48 :=
sorry

end reduced_price_per_kg_of_oil_l142_142679


namespace ratio_of_ages_l142_142620

theorem ratio_of_ages (x m : ℕ) 
  (mother_current_age : ℕ := 41) 
  (daughter_current_age : ℕ := 23) 
  (age_diff : ℕ := mother_current_age - daughter_current_age) 
  (eq : (mother_current_age - x) = m * (daughter_current_age - x)) : 
  (41 - x) / (23 - x) = m :=
by
  -- Proof not required
  sorry

end ratio_of_ages_l142_142620


namespace tan_alpha_sol_expr_sol_l142_142833

noncomputable def tan_half_alpha (α : ℝ) : ℝ := 2

noncomputable def tan_alpha_from_half (α : ℝ) : ℝ := 
  let tan_half := tan_half_alpha α
  2 * tan_half / (1 - tan_half * tan_half)

theorem tan_alpha_sol (α : ℝ) (h : tan_half_alpha α = 2) : tan_alpha_from_half α = -4 / 3 := by
  sorry

noncomputable def expr_eval (α : ℝ) : ℝ :=
  let tan_α := tan_alpha_from_half α
  let sin_α := tan_α / Real.sqrt (1 + tan_α * tan_α)
  let cos_α := 1 / Real.sqrt (1 + tan_α * tan_α)
  (6 * sin_α + cos_α) / (3 * sin_α - 2 * cos_α)

theorem expr_sol (α : ℝ) (h : tan_half_alpha α = 2) : expr_eval α = 7 / 6 := by
  sorry

end tan_alpha_sol_expr_sol_l142_142833


namespace transformed_graph_area_l142_142586

theorem transformed_graph_area (g : ℝ → ℝ) (a b : ℝ)
  (h_area_g : ∫ x in a..b, g x = 15) :
  ∫ x in a..b, 2 * g (x + 3) = 30 := 
sorry

end transformed_graph_area_l142_142586


namespace trig_identity_simplification_l142_142093

theorem trig_identity_simplification (θ : ℝ) (hθ : θ = 15 * Real.pi / 180) :
  (Real.sqrt 3 / 2 - Real.sqrt 3 * (Real.sin θ) ^ 2) = 3 / 4 := 
by sorry

end trig_identity_simplification_l142_142093


namespace sum_values_l142_142470

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (x + 2) = -f x
axiom value_at_one : f 1 = 8

theorem sum_values :
  f 2008 + f 2009 + f 2010 = 8 :=
sorry

end sum_values_l142_142470


namespace intersection_point_l142_142160

theorem intersection_point :
  ∃ (x y : ℝ), (2 * x + 3 * y + 8 = 0) ∧ (x - y - 1 = 0) ∧ (x = -1) ∧ (y = -2) := 
by
  sorry

end intersection_point_l142_142160


namespace rowing_upstream_speed_l142_142571

-- Define the speed of the man in still water
def V_m : ℝ := 45

-- Define the speed of the man rowing downstream
def V_downstream : ℝ := 65

-- Define the speed of the stream
def V_s : ℝ := V_downstream - V_m

-- Define the speed of the man rowing upstream
def V_upstream : ℝ := V_m - V_s

-- Prove that the speed of the man rowing upstream is 25 kmph
theorem rowing_upstream_speed :
  V_upstream = 25 := by
  sorry

end rowing_upstream_speed_l142_142571


namespace train_people_count_l142_142247

theorem train_people_count :
  let initial := 48
  let after_first_stop := initial - 13 + 5
  let after_second_stop := after_first_stop - 9 + 10 - 2
  let after_third_stop := after_second_stop - 7 + 4 - 3
  let after_fourth_stop := after_third_stop - 16 + 7 - 5
  let after_fifth_stop := after_fourth_stop - 8 + 15
  after_fifth_stop = 26 := sorry

end train_people_count_l142_142247


namespace distance_to_place_l142_142188

theorem distance_to_place (rowing_speed still_water : ℝ) (downstream_speed : ℝ)
                         (upstream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  rowing_speed = 10 → downstream_speed = 2 → upstream_speed = 3 →
  total_time = 10 → distance = 44.21 → 
  (distance / (rowing_speed + downstream_speed) + distance / (rowing_speed - upstream_speed)) = 10 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3]
  field_simp
  sorry

end distance_to_place_l142_142188


namespace total_weight_of_settings_l142_142142

-- Define the problem conditions
def weight_silverware_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def weight_plate_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Define the calculations
def total_settings_needed : ℕ :=
  (tables * settings_per_table) + backup_settings

def weight_silverware_per_setting : ℕ :=
  pieces_per_setting * weight_silverware_per_piece

def weight_plates_per_setting : ℕ :=
  plates_per_setting * weight_plate_per_piece

def total_weight_per_setting : ℕ :=
  weight_silverware_per_setting + weight_plates_per_setting

def total_weight_all_settings : ℕ :=
  total_settings_needed * total_weight_per_setting

-- Prove the solution
theorem total_weight_of_settings :
  total_weight_all_settings = 5040 :=
sorry

end total_weight_of_settings_l142_142142


namespace jean_spots_on_sides_l142_142133

variables (total_spots upper_torso_spots back_hindquarters_spots side_spots : ℕ)

def half (x : ℕ) := x / 2
def third (x : ℕ) := x / 3

-- Given conditions
axiom h1 : upper_torso_spots = 30
axiom h2 : upper_torso_spots = half total_spots
axiom h3 : back_hindquarters_spots = third total_spots
axiom h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots

-- Theorem to prove
theorem jean_spots_on_sides (h1 : upper_torso_spots = 30)
    (h2 : upper_torso_spots = half total_spots)
    (h3 : back_hindquarters_spots = third total_spots)
    (h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots) :
    side_spots = 10 := by
  sorry

end jean_spots_on_sides_l142_142133


namespace a2020_lt_inv_2020_l142_142157

theorem a2020_lt_inv_2020 (a : ℕ → ℝ) (ha0 : a 0 > 0) 
    (h_rec : ∀ n, a (n + 1) = a n / Real.sqrt (1 + 2020 * a n ^ 2)) :
    a 2020 < 1 / 2020 :=
sorry

end a2020_lt_inv_2020_l142_142157


namespace sum_of_odd_coefficients_l142_142547

theorem sum_of_odd_coefficients (a : ℝ) (h : (a + 1) * 16 = 32) : a = 3 :=
by
  sorry

end sum_of_odd_coefficients_l142_142547


namespace greatest_three_digit_multiple_of_17_l142_142034

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l142_142034


namespace smallest_integer_solution_l142_142755

theorem smallest_integer_solution (x : ℤ) : 
  (10 * x * x - 40 * x + 36 = 0) → x = 2 :=
sorry

end smallest_integer_solution_l142_142755


namespace fraction_of_sum_after_6_years_l142_142165

-- Define the principal amount, rate, and time period as given in the conditions
def P : ℝ := 1
def R : ℝ := 0.02777777777777779
def T : ℕ := 6

-- Definition of the Simple Interest calculation
def simple_interest (P R : ℝ) (T : ℕ) : ℝ :=
  P * R * T

-- Definition of the total amount after 6 years
def total_amount (P SI : ℝ) : ℝ :=
  P + SI

-- The main theorem to prove
theorem fraction_of_sum_after_6_years :
  total_amount P (simple_interest P R T) = 1.1666666666666667 :=
by
  sorry

end fraction_of_sum_after_6_years_l142_142165


namespace AdultsNotWearingBlue_l142_142193

theorem AdultsNotWearingBlue (number_of_children : ℕ) (number_of_adults : ℕ) (adults_who_wore_blue : ℕ) :
  number_of_children = 45 → 
  number_of_adults = number_of_children / 3 → 
  adults_who_wore_blue = number_of_adults / 3 → 
  number_of_adults - adults_who_wore_blue = 10 :=
by
  sorry

end AdultsNotWearingBlue_l142_142193


namespace probability_excellent_probability_good_or_better_l142_142373

noncomputable def total_selections : ℕ := 10
noncomputable def total_excellent_selections : ℕ := 1
noncomputable def total_good_or_better_selections : ℕ := 7
noncomputable def P_excellent : ℚ := 1 / 10
noncomputable def P_good_or_better : ℚ := 7 / 10

theorem probability_excellent (total_selections total_excellent_selections : ℕ) :
  (total_excellent_selections : ℚ) / total_selections = 1 / 10 := by
  sorry

theorem probability_good_or_better (total_selections total_good_or_better_selections : ℕ) :
  (total_good_or_better_selections : ℚ) / total_selections = 7 / 10 := by
  sorry

end probability_excellent_probability_good_or_better_l142_142373


namespace compare_values_l142_142610

variable (f : ℝ → ℝ)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

noncomputable def a : ℝ := f 1
noncomputable def b : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def c : ℝ := f ((Real.log 3 / Real.log 2) - 1)

theorem compare_values (h_log1 : Real.log 3 / Real.log 0.5 < -1) 
                       (h_log2 : 0 < (Real.log 3 / Real.log 2) - 1 ∧ (Real.log 3 / Real.log 2) - 1 < 1) : 
  b < a ∧ a < c :=
by
  sorry

end compare_values_l142_142610


namespace prime_divides_factorial_difference_l142_142757

theorem prime_divides_factorial_difference (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_five : p ≥ 5) : 
  p^5 ∣ (Nat.factorial p - p) := by
  sorry

end prime_divides_factorial_difference_l142_142757


namespace arithmetic_sequence_diff_l142_142662

theorem arithmetic_sequence_diff (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 7 = a 3 + 4 * d) :
  a 2008 - a 2000 = 8 * d :=
by
  sorry

end arithmetic_sequence_diff_l142_142662


namespace son_is_four_times_younger_l142_142123

-- Given Conditions
def son_age : ℕ := 9
def dad_age : ℕ := 36
def age_difference : ℕ := dad_age - son_age -- Ensure the difference in ages

-- The proof problem
theorem son_is_four_times_younger : dad_age / son_age = 4 :=
by
  -- Ensure the conditions are correct and consistent.
  have h1 : dad_age = 36 := rfl
  have h2 : son_age = 9 := rfl
  have h3 : dad_age - son_age = 27 := rfl
  sorry

end son_is_four_times_younger_l142_142123


namespace common_root_sum_k_l142_142280

theorem common_root_sum_k :
  (∃ x : ℝ, (x^2 - 4 * x + 3 = 0) ∧ (x^2 - 6 * x + k = 0)) → 
  (∃ (k₁ k₂ : ℝ), (k₁ = 5) ∧ (k₂ = 9) ∧ (k₁ + k₂ = 14)) :=
by
  sorry

end common_root_sum_k_l142_142280


namespace prime_division_or_divisibility_l142_142819

open Nat

theorem prime_division_or_divisibility (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) (hodd : Odd p) (hd : p ∣ q^r + 1) :
    (2 * r ∣ p - 1) ∨ (p ∣ q^2 - 1) := 
sorry

end prime_division_or_divisibility_l142_142819


namespace soccer_tournament_games_l142_142215

-- Define the single-elimination tournament problem
def single_elimination_games (teams : ℕ) : ℕ :=
  teams - 1

-- Define the specific problem instance
def teams := 20

-- State the theorem
theorem soccer_tournament_games : single_elimination_games teams = 19 :=
  sorry

end soccer_tournament_games_l142_142215


namespace movie_theater_attendance_l142_142598

theorem movie_theater_attendance : 
  let total_seats := 750
  let empty_seats := 218
  let people := total_seats - empty_seats
  people = 532 :=
by
  sorry

end movie_theater_attendance_l142_142598


namespace workshop_employees_l142_142759

theorem workshop_employees (x y : ℕ) 
  (H1 : (x + y) - ((1 / 2) * x + (1 / 3) * y + (1 / 3) * x + (1 / 2) * y) = 120)
  (H2 : (1 / 2) * x + (1 / 3) * y = (1 / 7) * ((1 / 3) * x + (1 / 2) * y) + (1 / 3) * x + (1 / 2) * y) : 
  x = 480 ∧ y = 240 := 
by
  sorry

end workshop_employees_l142_142759


namespace digit_B_for_divisibility_by_9_l142_142489

theorem digit_B_for_divisibility_by_9 :
  ∃! (B : ℕ), B < 10 ∧ (5 + B + B + 3) % 9 = 0 :=
by
  sorry

end digit_B_for_divisibility_by_9_l142_142489


namespace cos_E_floor_1000_l142_142906

theorem cos_E_floor_1000 {EF GH FG EH : ℝ} {E G : ℝ} (h1 : EF = 200) (h2 : GH = 200) (h3 : FG + EH = 380) (h4 : E = G) (h5 : EH ≠ FG) :
  ∃ (cE : ℝ), cE = 11/16 ∧ ⌊ 1000 * cE ⌋ = 687 :=
by sorry

end cos_E_floor_1000_l142_142906


namespace elsa_emma_spending_ratio_l142_142400

theorem elsa_emma_spending_ratio
  (E : ℝ)
  (h_emma : ∃ (x : ℝ), x = 58)
  (h_elizabeth : ∃ (y : ℝ), y = 4 * E)
  (h_total : 58 + E + 4 * E = 638) :
  E / 58 = 2 :=
by
  sorry

end elsa_emma_spending_ratio_l142_142400


namespace series_solution_l142_142544

theorem series_solution (r : ℝ) (h : (r^3 - r^2 + (1 / 4) * r - 1 = 0) ∧ r > 0) :
  (∑' (n : ℕ), (n + 1) * r^(3 * (n + 1))) = 16 * r :=
by
  sorry

end series_solution_l142_142544


namespace part_1_part_2_l142_142020

noncomputable def f (a m x : ℝ) := a ^ m / x

theorem part_1 (a : ℝ) (m : ℝ) (H1 : a > 1) (H2 : ∀ x, x ∈ Set.Icc a (2*a) → f a m x ∈ Set.Icc (a^2) (a^3)) :
  a = 2 :=
sorry

theorem part_2 (t : ℝ) (s : ℝ) (H1 : ∀ x, x ∈ Set.Icc 0 s → (x + t) ^ 2 + 2 * (x + t) ≤ 3 * x) :
  s ∈ Set.Ioc 0 5 :=
sorry

end part_1_part_2_l142_142020


namespace inequality_solution_l142_142225

theorem inequality_solution (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x + 1/2 > 0) ↔ (0 ≤ m ∧ m < 2) :=
by
  sorry

end inequality_solution_l142_142225


namespace solve_problem_l142_142068

noncomputable def problem_statement : Prop :=
  ∀ (a b c : ℕ),
    (a ≤ b) →
    (b ≤ c) →
    Nat.gcd (Nat.gcd a b) c = 1 →
    (a^2 * b) ∣ (a^3 + b^3 + c^3) →
    (b^2 * c) ∣ (a^3 + b^3 + c^3) →
    (c^2 * a) ∣ (a^3 + b^3 + c^3) →
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 2 ∧ c = 3)

-- Here we declare the main theorem but skip the proof.
theorem solve_problem : problem_statement :=
by sorry

end solve_problem_l142_142068


namespace negation_of_proposition_l142_142504

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x ≤ 0 ∧ x^2 ≥ 0) ↔ ∀ x : ℝ, x ≤ 0 → x^2 < 0 :=
by
  sorry

end negation_of_proposition_l142_142504


namespace lines_parallel_iff_a_eq_1_l142_142924

theorem lines_parallel_iff_a_eq_1 (x y a : ℝ) :
    (a = 1 ↔ ∃ k : ℝ, ∀ x y : ℝ, a*x + y - 1 = k*(x + a*y + 1)) :=
sorry

end lines_parallel_iff_a_eq_1_l142_142924


namespace min_value_frac_l142_142046

theorem min_value_frac (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  ∃ (min : ℝ), min = 9 / 2 ∧ (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 2 → 4 / x + 1 / y ≥ min) :=
by
  sorry

end min_value_frac_l142_142046


namespace calculation1_calculation2_calculation3_calculation4_l142_142638

theorem calculation1 : 72 * 54 + 28 * 54 = 5400 := 
by sorry

theorem calculation2 : 60 * 25 * 8 = 12000 := 
by sorry

theorem calculation3 : 2790 / (250 * 12 - 2910) = 31 := 
by sorry

theorem calculation4 : (100 - 1456 / 26) * 78 = 3432 := 
by sorry

end calculation1_calculation2_calculation3_calculation4_l142_142638


namespace min_value_frac_sum_l142_142944

theorem min_value_frac_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) : 
  ∃ m, ∀ x y, 0 < x → 0 < y → 2 * x + y = 1 → m ≤ (1/x + 1/y) ∧ (1/x + 1/y) = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_frac_sum_l142_142944


namespace taxi_trip_miles_l142_142081

theorem taxi_trip_miles 
  (initial_fee : ℝ := 2.35)
  (additional_charge : ℝ := 0.35)
  (segment_length : ℝ := 2/5)
  (total_charge : ℝ := 5.50) :
  ∃ (miles : ℝ), total_charge = initial_fee + additional_charge * (miles / segment_length) ∧ miles = 3.6 :=
by
  sorry

end taxi_trip_miles_l142_142081


namespace number_of_valid_sets_l142_142879

open Set

variable {α : Type} (a b : α)

def is_valid_set (M : Set α) : Prop := M ∪ {a} = {a, b}

theorem number_of_valid_sets (a b : α) : (∃! M : Set α, is_valid_set a b M) := 
sorry

end number_of_valid_sets_l142_142879


namespace solve_complex_addition_l142_142557

def complex_addition_problem : Prop :=
  let B := Complex.mk 3 (-2)
  let Q := Complex.mk (-5) 1
  let R := Complex.mk 1 (-2)
  let T := Complex.mk 4 3
  B - Q + R + T = Complex.mk 13 (-2)

theorem solve_complex_addition : complex_addition_problem := by
  sorry

end solve_complex_addition_l142_142557


namespace brittany_second_test_grade_l142_142730

theorem brittany_second_test_grade
  (first_test_grade second_test_grade : ℕ) 
  (average_after_second : ℕ)
  (h1 : first_test_grade = 78)
  (h2 : average_after_second = 81) 
  (h3 : (first_test_grade + second_test_grade) / 2 = average_after_second) :
  second_test_grade = 84 :=
by
  sorry

end brittany_second_test_grade_l142_142730


namespace arithmetic_sequence_sum_l142_142647

variable (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ)

def S₁₀ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ) : ℕ :=
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀

theorem arithmetic_sequence_sum (h : S₁₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ = 120) :
  a₁ + a₁₀ = 24 :=
by
  sorry

end arithmetic_sequence_sum_l142_142647


namespace sum_of_largest_two_l142_142869

-- Define the three numbers
def a := 10
def b := 11
def c := 12

-- Define the sum of the largest and the next largest numbers
def sum_of_largest_two_numbers (x y z : ℕ) : ℕ :=
  if x >= y ∧ y >= z then x + y
  else if x >= z ∧ z >= y then x + z
  else if y >= x ∧ x >= z then y + x
  else if y >= z ∧ z >= x then y + z
  else if z >= x ∧ x >= y then z + x
  else z + y

-- State the theorem to prove
theorem sum_of_largest_two (x y z : ℕ) : sum_of_largest_two_numbers x y z = 23 :=
by
  sorry

end sum_of_largest_two_l142_142869


namespace v_2015_eq_2_l142_142911

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 4
  | 4 => 1
  | 5 => 2
  | _ => 0  -- assuming g(x) = 0 for other values, though not used here

def v : ℕ → ℕ
| 0     => 3
| (n+1) => g (v n)

theorem v_2015_eq_2 : v 2015 = 2 :=
by
  sorry

end v_2015_eq_2_l142_142911


namespace mittens_in_each_box_l142_142780

theorem mittens_in_each_box (boxes scarves_per_box total_clothing : ℕ) (h1 : boxes = 8) (h2 : scarves_per_box = 4) (h3 : total_clothing = 80) :
  ∃ (mittens_per_box : ℕ), mittens_per_box = 6 :=
by
  let total_scarves := boxes * scarves_per_box
  let total_mittens := total_clothing - total_scarves
  let mittens_per_box := total_mittens / boxes
  use mittens_per_box
  sorry

end mittens_in_each_box_l142_142780


namespace ratio_values_l142_142589

theorem ratio_values (x y z : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) 
  (h₀ : (x + y) / z = (y + z) / x) (h₀' : (y + z) / x = (z + x) / y) :
  ∃ a : ℝ, a = -1 ∨ a = 8 :=
sorry

end ratio_values_l142_142589


namespace number_of_red_balls_l142_142500

theorem number_of_red_balls (x : ℕ) (h₀ : 4 > 0) (h₁ : (x : ℝ) / (x + 4) = 0.6) : x = 6 :=
sorry

end number_of_red_balls_l142_142500


namespace emily_sixth_score_l142_142418

theorem emily_sixth_score:
  ∀ (s₁ s₂ s₃ s₄ s₅ sᵣ : ℕ),
  s₁ = 88 →
  s₂ = 90 →
  s₃ = 85 →
  s₄ = 92 →
  s₅ = 97 →
  (s₁ + s₂ + s₃ + s₄ + s₅ + sᵣ) / 6 = 91 →
  sᵣ = 94 :=
by intros s₁ s₂ s₃ s₄ s₅ sᵣ h₁ h₂ h₃ h₄ h₅ h₆;
   rw [h₁, h₂, h₃, h₄, h₅] at h₆;
   sorry

end emily_sixth_score_l142_142418


namespace calculate_value_l142_142505

theorem calculate_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x = 1 / y) (hzy : z = 1 / y) : 
  (x + 1 / x) * (z - 1 / z) = 4 := 
by 
  -- Proof omitted, this is just the statement
  sorry

end calculate_value_l142_142505


namespace rich_total_distance_l142_142148

-- Define the given conditions 
def distance_house_to_sidewalk := 20
def distance_down_road := 200
def total_distance_so_far := distance_house_to_sidewalk + distance_down_road
def distance_left_turn := 2 * total_distance_so_far
def distance_to_intersection := total_distance_so_far + distance_left_turn
def distance_half := distance_to_intersection / 2
def total_distance_one_way := distance_to_intersection + distance_half

-- Define the theorem to be proven 
theorem rich_total_distance : total_distance_one_way * 2 = 1980 :=
by 
  -- This line is to complete the 'prove' demand of the theorem
  sorry

end rich_total_distance_l142_142148


namespace tangent_line_circle_l142_142846

theorem tangent_line_circle (a : ℝ) :
  (∀ (x y : ℝ), 4 * x - 3 * y = 0 → x^2 + y^2 - 2 * x + a * y + 1 = 0) →
  a = -1 ∨ a = 4 :=
sorry

end tangent_line_circle_l142_142846


namespace red_balls_count_l142_142292

theorem red_balls_count (white_balls_ratio : ℕ) (red_balls_ratio : ℕ) (total_white_balls : ℕ)
  (h_ratio : white_balls_ratio = 3 ∧ red_balls_ratio = 2)
  (h_white_balls : total_white_balls = 9) :
  ∃ (total_red_balls : ℕ), total_red_balls = 6 :=
by
  sorry

end red_balls_count_l142_142292


namespace fraction_equivalence_l142_142082

theorem fraction_equivalence : 
  (∀ (a b : ℕ), (a ≠ 0 ∧ b ≠ 0) → (15 * b = 25 * a ↔ a = 3 ∧ b = 5)) ∧
  (15 * 4 ≠ 25 * 3) ∧
  (15 * 3 ≠ 25 * 2) ∧
  (15 * 2 ≠ 25 * 1) ∧
  (15 * 7 ≠ 25 * 5) :=
by
  sorry

end fraction_equivalence_l142_142082


namespace carpet_width_l142_142745

theorem carpet_width
  (carpet_percentage : ℝ)
  (living_room_area : ℝ)
  (carpet_length : ℝ) :
  carpet_percentage = 0.30 →
  living_room_area = 120 →
  carpet_length = 9 →
  carpet_percentage * living_room_area / carpet_length = 4 :=
by
  sorry

end carpet_width_l142_142745


namespace smallest_sum_of_bases_l142_142763

theorem smallest_sum_of_bases :
  ∃ (c d : ℕ), 8 * c + 9 = 9 * d + 8 ∧ c + d = 19 := 
by
  sorry

end smallest_sum_of_bases_l142_142763


namespace shirt_cost_l142_142967

theorem shirt_cost (S : ℝ) (hats_cost jeans_cost total_cost : ℝ)
  (h_hats : hats_cost = 4)
  (h_jeans : jeans_cost = 10)
  (h_total : total_cost = 51)
  (h_eq : 3 * S + 2 * jeans_cost + 4 * hats_cost = total_cost) :
  S = 5 :=
by
  -- The main proof will be provided here
  sorry

end shirt_cost_l142_142967


namespace topsoil_cost_correct_l142_142411

noncomputable def topsoilCost (price_per_cubic_foot : ℝ) (yard_to_foot : ℝ) (discount_threshold : ℝ) (discount_rate : ℝ) (volume_in_yards : ℝ) : ℝ :=
  let volume_in_feet := volume_in_yards * yard_to_foot
  let cost_without_discount := volume_in_feet * price_per_cubic_foot
  if volume_in_feet > discount_threshold then
    cost_without_discount * (1 - discount_rate)
  else
    cost_without_discount

theorem topsoil_cost_correct:
  topsoilCost 8 27 100 0.10 7 = 1360.8 :=
by
  sorry

end topsoil_cost_correct_l142_142411


namespace determine_B_l142_142786

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (h1 : (A ∪ B)ᶜ = {1})
variable (h2 : A ∩ Bᶜ = {3})

theorem determine_B : B = {2, 4, 5} :=
by
  sorry

end determine_B_l142_142786


namespace sum_q_p_values_l142_142269

def p (x : ℤ) : ℤ := x^2 - 4

def q (x : ℤ) : ℤ := -abs x

theorem sum_q_p_values : 
  (q (p (-3)) + q (p (-2)) + q (p (-1)) + q (p (0)) + q (p (1)) + q (p (2)) + q (p (3))) = -20 :=
by
  sorry

end sum_q_p_values_l142_142269


namespace binomial_expansion_example_l142_142180

theorem binomial_expansion_example :
  57^3 + 3 * (57^2) * 4 + 3 * 57 * (4^2) + 4^3 = 226981 :=
by
  -- The proof would go here, using the steps outlined.
  sorry

end binomial_expansion_example_l142_142180


namespace pyramid_section_rhombus_l142_142792

structure Pyramid (A B C D : Type) := (point : Type)

def is_parallel (l1 l2 : ℝ) : Prop :=
  ∀ (m n : ℝ), m * l1 = n * l2

def is_parallelogram (K L M N : Type) : Prop :=
  sorry

def is_rhombus (K L M N : Type) : Prop :=
  sorry

noncomputable def side_length_rhombus (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

/-- Prove that the section of pyramid ABCD with a plane parallel to edges AC and BD is a parallelogram,
and under certain conditions, this parallelogram is a rhombus. Find the side of this rhombus given AC = a and BD = b. -/
theorem pyramid_section_rhombus (A B C D K L M N : Type) (a b : ℝ) :
  is_parallel AC BD →
  is_parallelogram K L M N →
  is_rhombus K L M N →
  side_length_rhombus a b = (a * b) / (a + b) :=
by
  sorry

end pyramid_section_rhombus_l142_142792


namespace jose_profit_share_l142_142117

theorem jose_profit_share (investment_tom : ℕ) (months_tom : ℕ) 
                         (investment_jose : ℕ) (months_jose : ℕ) 
                         (total_profit : ℕ) :
                         investment_tom = 30000 →
                         months_tom = 12 →
                         investment_jose = 45000 →
                         months_jose = 10 →
                         total_profit = 63000 →
                         (investment_jose * months_jose / 
                         (investment_tom * months_tom + investment_jose * months_jose)) * total_profit = 35000 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  norm_num
  sorry

end jose_profit_share_l142_142117


namespace ratio_of_juice_to_bread_l142_142501

variable (total_money : ℕ) (money_left : ℕ) (cost_bread : ℕ) (cost_butter : ℕ) (cost_juice : ℕ)

def compute_ratio (total_money money_left cost_bread cost_butter cost_juice : ℕ) : ℕ :=
  cost_juice / cost_bread

theorem ratio_of_juice_to_bread :
  total_money = 15 →
  money_left = 6 →
  cost_bread = 2 →
  cost_butter = 3 →
  total_money - money_left - (cost_bread + cost_butter) = cost_juice →
  compute_ratio total_money money_left cost_bread cost_butter cost_juice = 2 :=
by
  intros
  sorry

end ratio_of_juice_to_bread_l142_142501


namespace find_m_value_l142_142665

theorem find_m_value (m : ℝ) 
  (first_term : ℝ := 18) (second_term : ℝ := 6)
  (second_term_2 : ℝ := 6 + m) 
  (S1 : ℝ := first_term / (1 - second_term / first_term))
  (S2 : ℝ := first_term / (1 - second_term_2 / first_term))
  (eq_sum : S2 = 3 * S1) :
  m = 8 := by
  sorry

end find_m_value_l142_142665


namespace new_class_mean_l142_142746

theorem new_class_mean (n1 n2 : ℕ) (mean1 mean2 : ℝ) (h1 : n1 = 45) (h2 : n2 = 5) (h3 : mean1 = 0.85) (h4 : mean2 = 0.90) : 
(n1 + n2 = 50) → 
((n1 * mean1 + n2 * mean2) / (n1 + n2) = 0.855) := 
by
  intro total_students
  sorry

end new_class_mean_l142_142746


namespace min_value_of_a_plus_b_l142_142667

theorem min_value_of_a_plus_b 
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_eq : 1 / a + 2 / b = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_of_a_plus_b_l142_142667


namespace sum_is_ten_l142_142071

variable (x y : ℝ) (S : ℝ)

-- Conditions
def condition1 : Prop := x + y = S
def condition2 : Prop := x = 25 / y
def condition3 : Prop := x^2 + y^2 = 50

-- Theorem
theorem sum_is_ten (h1 : condition1 x y S) (h2 : condition2 x y) (h3 : condition3 x y) : S = 10 :=
sorry

end sum_is_ten_l142_142071


namespace add_solution_y_to_solution_x_l142_142364

theorem add_solution_y_to_solution_x
  (x_volume : ℝ) (x_percent : ℝ) (y_percent : ℝ) (desired_percent : ℝ) (final_volume : ℝ)
  (x_alcohol : ℝ := x_volume * x_percent / 100) (y : ℝ := final_volume - x_volume) :
  (x_percent = 10) → (y_percent = 30) → (desired_percent = 15) → (x_volume = 300) →
  (final_volume = 300 + y) →
  ((x_alcohol + y * y_percent / 100) / final_volume = desired_percent / 100) →
  y = 100 := by
    intros h1 h2 h3 h4 h5 h6
    sorry

end add_solution_y_to_solution_x_l142_142364


namespace track_length_l142_142296

theorem track_length (L : ℕ)
  (h1 : ∃ B S : ℕ, B = 120 ∧ (L - B) = S ∧ (S + 200) - B = (L + 80) - B)
  (h2 : L + 80 = 440 - L) : L = 180 := 
  by
    sorry

end track_length_l142_142296


namespace hyperbola_equation_l142_142618

theorem hyperbola_equation (a b k : ℝ) (p : ℝ × ℝ) (h_asymptotes : b = 3 * a)
  (h_hyperbola_passes_point : p = (2, -3 * Real.sqrt 3)) (h_hyperbola : ∀ x y, x^2 - (y^2 / (3 * a)^2) = k) :
  ∃ k, k = 1 :=
by
  -- Given the point p and asymptotes, we should prove k = 1.
  sorry

end hyperbola_equation_l142_142618


namespace inequality_positive_reals_l142_142765

theorem inequality_positive_reals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / a^2 + 1 / b^2 + 8 * a * b ≥ 8) ∧ (1 / a^2 + 1 / b^2 + 8 * a * b = 8 → a = b ∧ a = 1/2) :=
by
  sorry

end inequality_positive_reals_l142_142765


namespace absolute_value_is_four_l142_142970

-- Given condition: the absolute value of a number equals 4
theorem absolute_value_is_four (x : ℝ) : abs x = 4 → (x = 4 ∨ x = -4) :=
by
  sorry

end absolute_value_is_four_l142_142970


namespace sqrt_1708249_eq_1307_l142_142228

theorem sqrt_1708249_eq_1307 :
  ∃ (n : ℕ), n * n = 1708249 ∧ n = 1307 :=
sorry

end sqrt_1708249_eq_1307_l142_142228


namespace how_many_tickets_left_l142_142739

-- Define the conditions
def tickets_from_whack_a_mole : ℕ := 32
def tickets_from_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- Define the total tickets won by Tom
def total_tickets : ℕ := tickets_from_whack_a_mole + tickets_from_skee_ball

-- State the theorem to be proved: how many tickets Tom has left
theorem how_many_tickets_left : total_tickets - tickets_spent_on_hat = 50 := by
  sorry

end how_many_tickets_left_l142_142739


namespace remainder_of_173_mod_13_l142_142010

theorem remainder_of_173_mod_13 : ∀ (m : ℤ), 173 = 8 * m + 5 → 173 < 180 → 173 % 13 = 4 :=
by
  intro m hm h
  sorry

end remainder_of_173_mod_13_l142_142010


namespace solve_for_x_l142_142143

-- Definitions based on provided conditions
variables (x : ℝ) -- defining x as a real number
def condition : Prop := 0.25 * x = 0.15 * 1600 - 15

-- The theorem stating that x equals 900 given the condition
theorem solve_for_x (h : condition x) : x = 900 :=
by
  sorry

end solve_for_x_l142_142143


namespace tetrahedron_min_green_edges_l142_142118

theorem tetrahedron_min_green_edges : 
  ∃ (green_edges : Finset (Fin 6)), 
  (∀ face : Finset (Fin 6), face.card = 3 → ∃ edge ∈ face, edge ∈ green_edges) ∧ green_edges.card = 3 :=
by sorry

end tetrahedron_min_green_edges_l142_142118


namespace solve_system_equations_l142_142485

-- Define the hypotheses of the problem
variables {a x y : ℝ}
variables (h1 : (0 < a) ∧ (a ≠ 1))
variables (h2 : (0 < x))
variables (h3 : (0 < y))
variables (eq1 : (log a x + log a y - 2) * log 18 a = 1)
variables (eq2 : 2 * x + y - 20 * a = 0)

-- State the theorem to be proved
theorem solve_system_equations :
  (x = a ∧ y = 18 * a) ∨ (x = 9 * a ∧ y = 2 * a) := by
  sorry

end solve_system_equations_l142_142485


namespace at_least_one_alarm_rings_on_time_l142_142484

-- Definitions for the problem
def prob_A : ℝ := 0.5
def prob_B : ℝ := 0.6

def prob_not_A : ℝ := 1 - prob_A
def prob_not_B : ℝ := 1 - prob_B
def prob_neither_A_nor_B : ℝ := prob_not_A * prob_not_B
def prob_at_least_one : ℝ := 1 - prob_neither_A_nor_B

-- Final statement
theorem at_least_one_alarm_rings_on_time : prob_at_least_one = 0.8 :=
by sorry

end at_least_one_alarm_rings_on_time_l142_142484


namespace expression_is_integer_expression_modulo_3_l142_142918

theorem expression_is_integer (n : ℕ) (hn : n > 0) : 
  ∃ (k : ℤ), (n^3 + (3/2) * n^2 + (1/2) * n - 1) = k := 
sorry

theorem expression_modulo_3 (n : ℕ) (hn : n > 0) : 
  (n^3 + (3/2) * n^2 + (1/2) * n - 1) % 3 = 2 :=
sorry

end expression_is_integer_expression_modulo_3_l142_142918


namespace find_a_plus_b_l142_142287

theorem find_a_plus_b (a b : ℝ) : (3 = 1/3 * 1 + a) → (1 = 1/3 * 3 + b) → a + b = 8/3 :=
by
  intros h1 h2
  sorry

end find_a_plus_b_l142_142287


namespace distance_between_points_l142_142986

theorem distance_between_points :
  let point1 := (2, -3)
  let point2 := (8, 9)
  dist point1 point2 = 6 * Real.sqrt 5 :=
by
  sorry

end distance_between_points_l142_142986


namespace vector_subtraction_l142_142243

def a : ℝ × ℝ × ℝ := (1, -2, 1)
def b : ℝ × ℝ × ℝ := (1, 0, 2)

theorem vector_subtraction : a - b = (0, -2, -1) := 
by 
  unfold a b
  simp
  sorry

end vector_subtraction_l142_142243


namespace petroleum_crude_oil_problem_l142_142436

variables (x y : ℝ)

theorem petroleum_crude_oil_problem (h1 : x + y = 50)
  (h2 : 0.25 * x + 0.75 * y = 27.5) : y = 30 :=
by
  -- Proof would go here
  sorry

end petroleum_crude_oil_problem_l142_142436


namespace maximum_guaranteed_money_l142_142928

theorem maximum_guaranteed_money (board_width board_height tromino_width tromino_height guaranteed_rubles : ℕ) 
  (h_board_width : board_width = 21) 
  (h_board_height : board_height = 20)
  (h_tromino_width : tromino_width = 3) 
  (h_tromino_height : tromino_height = 1)
  (h_guaranteed_rubles : guaranteed_rubles = 14) :
  true := by
  sorry

end maximum_guaranteed_money_l142_142928


namespace table_area_l142_142641

theorem table_area (A : ℝ) (runner_total : ℝ) (cover_percentage : ℝ) (double_layer : ℝ) (triple_layer : ℝ) :
  runner_total = 208 ∧
  cover_percentage = 0.80 ∧
  double_layer = 24 ∧
  triple_layer = 22 →
  A = 260 :=
by
  sorry

end table_area_l142_142641


namespace triangle_angle_B_l142_142308

theorem triangle_angle_B (a b c A B C : ℝ) 
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) : 
  B = 3 * Real.pi / 10 := 
sorry

end triangle_angle_B_l142_142308


namespace Bruce_paid_correct_amount_l142_142799

def grape_kg := 9
def grape_price_per_kg := 70
def mango_kg := 7
def mango_price_per_kg := 55
def orange_kg := 5
def orange_price_per_kg := 45
def apple_kg := 3
def apple_price_per_kg := 80

def total_cost := grape_kg * grape_price_per_kg + 
                  mango_kg * mango_price_per_kg + 
                  orange_kg * orange_price_per_kg + 
                  apple_kg * apple_price_per_kg

theorem Bruce_paid_correct_amount : total_cost = 1480 := by
  sorry

end Bruce_paid_correct_amount_l142_142799


namespace odd_function_sum_l142_142186

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_sum :
  (∀ x, f x = -f (-x)) ∧ 
  (∀ x y (hx : 3 ≤ x) (hy : y ≤ 7), x < y → f x < f y) ∧ 
  ( ∃ x, 3 ≤ x ∧ x ≤ 6 ∧ f x = 8) ∧ 
  ( ∃ x, 3 ≤ x ∧ x ≤ 6 ∧ f x = -1) →
  (2 * f (-6) + f (-3) = -15) :=
by
  intros
  sorry

end odd_function_sum_l142_142186


namespace vikas_rank_among_boys_l142_142543

def vikas_rank_overall := 9
def tanvi_rank_overall := 17
def girls_between := 2
def vikas_rank_top_boys := 4
def vikas_rank_bottom_overall := 18

theorem vikas_rank_among_boys (vikas_rank_overall tanvi_rank_overall girls_between vikas_rank_top_boys vikas_rank_bottom_overall : ℕ) :
  vikas_rank_top_boys = 4 := by
  sorry

end vikas_rank_among_boys_l142_142543


namespace discriminant_eq_M_l142_142874

theorem discriminant_eq_M (a b c x0 : ℝ) (h1: a ≠ 0) (h2: a * x0^2 + b * x0 + c = 0) :
  (b^2 - 4 * a * c) = (2 * a * x0 + b)^2 :=
by
  sorry

end discriminant_eq_M_l142_142874


namespace selling_price_of_book_l142_142851

theorem selling_price_of_book (SP : ℝ) (CP : ℝ := 200) :
  (SP - CP) = (340 - CP) + 0.05 * CP → SP = 350 :=
by {
  sorry
}

end selling_price_of_book_l142_142851


namespace solve_x_eqns_solve_y_eqns_l142_142057

theorem solve_x_eqns : ∀ x : ℝ, 2 * x^2 = 8 * x ↔ (x = 0 ∨ x = 4) :=
by
  intro x
  sorry

theorem solve_y_eqns : ∀ y : ℝ, y^2 - 10 * y - 1 = 0 ↔ (y = 5 + Real.sqrt 26 ∨ y = 5 - Real.sqrt 26) :=
by
  intro y
  sorry

end solve_x_eqns_solve_y_eqns_l142_142057


namespace shopkeeper_sold_articles_l142_142688

theorem shopkeeper_sold_articles (C : ℝ) (N : ℕ) 
  (h1 : (35 * C = N * C + (1/6) * (N * C))) : 
  N = 30 :=
by
  sorry

end shopkeeper_sold_articles_l142_142688


namespace value_of_f_l142_142283

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x 
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b

theorem value_of_f'_at_1 (a b : ℝ)
  (h₁ : f a b 0 = 1)
  (h₂ : f' (a := a) (b := b) 0 = 0) :
  f' (a := a) (b := b) 1 = Real.exp 1 - 1 :=
by
  sorry

end value_of_f_l142_142283


namespace ratio_50kg_to_05tons_not_100_to_1_l142_142031

theorem ratio_50kg_to_05tons_not_100_to_1 (weight1 : ℕ) (weight2 : ℕ) (r : ℕ) 
  (h1 : weight1 = 50) (h2 : weight2 = 500) (h3 : r = 100) : ¬ (weight1 * r = weight2) := 
by
  sorry

end ratio_50kg_to_05tons_not_100_to_1_l142_142031


namespace length_sixth_episode_l142_142342

def length_first_episode : ℕ := 58
def length_second_episode : ℕ := 62
def length_third_episode : ℕ := 65
def length_fourth_episode : ℕ := 71
def length_fifth_episode : ℕ := 79
def total_viewing_time : ℕ := 450

theorem length_sixth_episode :
  length_first_episode + length_second_episode + length_third_episode + length_fourth_episode + length_fifth_episode + 115 = total_viewing_time := by
  sorry

end length_sixth_episode_l142_142342


namespace parabola_equation_l142_142857

theorem parabola_equation (p : ℝ) (h : 0 < p) (Fₓ : ℝ) (Tₓ Tᵧ : ℝ) (Mₓ Mᵧ : ℝ)
  (eq_parabola : ∀ (y x : ℝ), y^2 = 2 * p * x → (y, x) = (Tᵧ, Tₓ))
  (F : (Fₓ, 0) = (p / 2, 0))
  (T_on_C : (Tᵧ, Tₓ) ∈ {(y, x) | y^2 = 2 * p * x})
  (FT_dist : dist (Fₓ, 0) (Tₓ, Tᵧ) = 5 / 2)
  (M : (Mₓ, Mᵧ) = (0, 1))
  (MF_MT_perp : ((Mᵧ - 0) / (Mₓ - Fₓ)) * ((Tᵧ - Mᵧ) / (Tₓ - Mᵧ)) = -1) :
  y^2 = 2 * x ∨ y^2 = 8 * x := 
sorry

end parabola_equation_l142_142857


namespace num_pieces_l142_142427

theorem num_pieces (total_length : ℝ) (piece_length : ℝ) 
  (h1: total_length = 253.75) (h2: piece_length = 0.425) :
  ⌊total_length / piece_length⌋ = 597 :=
by
  rw [h1, h2]
  sorry

end num_pieces_l142_142427


namespace smallest_positive_value_l142_142472

theorem smallest_positive_value (c d : ℤ) (h : c^2 > d^2) : 
  ∃ m > 0, m = (c^2 + d^2) / (c^2 - d^2) + (c^2 - d^2) / (c^2 + d^2) ∧ m = 2 :=
by
  sorry

end smallest_positive_value_l142_142472


namespace domain_of_function_l142_142452

def valid_domain (x : ℝ) : Prop :=
  x ≤ 3 ∧ x ≠ 0

theorem domain_of_function (x : ℝ) (h₀ : 3 - x ≥ 0) (h₁ : x ≠ 0) : valid_domain x :=
by
  sorry

end domain_of_function_l142_142452


namespace eval_expression_l142_142828

theorem eval_expression : (-1)^45 + 2^(3^2 + 5^2 - 4^2) = 262143 := by
  sorry

end eval_expression_l142_142828


namespace length_of_CD_l142_142347

theorem length_of_CD {L : ℝ} (h₁ : 16 * Real.pi * L + (256 / 3) * Real.pi = 432 * Real.pi) :
  L = (50 / 3) :=
by
  sorry

end length_of_CD_l142_142347


namespace domain_of_f_3x_minus_1_domain_of_f_l142_142843

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

end domain_of_f_3x_minus_1_domain_of_f_l142_142843


namespace inequality_proof_l142_142655

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / b + b^2 / c + c^2 / a) ≥ 3 * (a^3 + b^3 + c^3) / (a^2 + b^2 + c^2) := 
sorry

end inequality_proof_l142_142655


namespace find_value_of_a_l142_142737

theorem find_value_of_a
  (a : ℝ)
  (h : (a + 3) * 2 * (-2 / 3) = -4) :
  a = -3 :=
sorry

end find_value_of_a_l142_142737


namespace total_students_in_class_l142_142431

-- Define the initial conditions
def num_students_in_row (a b: Nat) : Nat := a + 1 + b
def num_lines : Nat := 3
noncomputable def students_in_row : Nat := num_students_in_row 2 5 

-- Theorem to prove the total number of students in the class
theorem total_students_in_class : students_in_row * num_lines = 24 :=
by
  sorry

end total_students_in_class_l142_142431


namespace max_sum_x_y_l142_142936

theorem max_sum_x_y {x y a b : ℝ} 
  (hx : 0 < x) (hy : 0 < y) (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a * x + b * y = 1) : 
  x + y ≤ 2 :=
sorry

end max_sum_x_y_l142_142936


namespace sum_abc_eq_neg_ten_thirds_l142_142357

variable (a b c d y : ℝ)

-- Define the conditions
def condition_1 : Prop := a + 2 = y
def condition_2 : Prop := b + 3 = y
def condition_3 : Prop := c + 4 = y
def condition_4 : Prop := d + 5 = y
def condition_5 : Prop := a + b + c + d + 6 = y

-- State the theorem
theorem sum_abc_eq_neg_ten_thirds
    (h1 : condition_1 a y)
    (h2 : condition_2 b y)
    (h3 : condition_3 c y)
    (h4 : condition_4 d y)
    (h5 : condition_5 a b c d y) :
    a + b + c + d = -10 / 3 :=
sorry

end sum_abc_eq_neg_ten_thirds_l142_142357


namespace pair_d_are_equal_l142_142740

theorem pair_d_are_equal : -(2 ^ 3) = (-2) ^ 3 :=
by
  -- Detailed proof steps go here, but are omitted for this task.
  sorry

end pair_d_are_equal_l142_142740


namespace cakes_left_l142_142091

def initial_cakes : ℕ := 62
def additional_cakes : ℕ := 149
def cakes_sold : ℕ := 144

theorem cakes_left : (initial_cakes + additional_cakes) - cakes_sold = 67 :=
by
  sorry

end cakes_left_l142_142091


namespace total_distance_fourth_fifth_days_l142_142787

theorem total_distance_fourth_fifth_days (d : ℕ) (total_distance : ℕ) (n : ℕ) (q : ℚ) 
  (S_6 : d * (1 - q^6) / (1 - q) = 378) (ratio : q = 1/2) (n_six : n = 6) : 
  (d * q^3) + (d * q^4) = 36 :=
by 
  sorry

end total_distance_fourth_fifth_days_l142_142787


namespace circle_tangent_to_y_axis_l142_142315

/-- The relationship between the circle with the focal radius |PF| of the parabola y^2 = 2px (where p > 0)
as its diameter and the y-axis -/
theorem circle_tangent_to_y_axis
  (p : ℝ) (hp : p > 0)
  (x1 y1 : ℝ)
  (focus : ℝ × ℝ := (p / 2, 0))
  (P : ℝ × ℝ := (x1, y1))
  (center : ℝ × ℝ := ((2 * x1 + p) / 4, y1 / 2))
  (radius : ℝ := (2 * x1 + p) / 4) :
  -- proof that the circle with PF as its diameter is tangent to the y-axis
  ∃ k : ℝ, k = radius ∧ (center.1 = k) :=
sorry

end circle_tangent_to_y_axis_l142_142315


namespace polynomial_geometric_roots_k_value_l142_142300

theorem polynomial_geometric_roots_k_value 
    (j k : ℝ)
    (h : ∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 0 ∧ 
      (∀ u v : ℝ, (u = a ∨ u = a * r ∨ u = a * r^2 ∨ u = a * r^3) →
        (v = a ∨ v = a * r ∨ v = a * r^2 ∨ v = a * r^3) →
        u ≠ v) ∧ 
      (a + a * r + a * r^2 + a * r^3 = 0) ∧
      (a^4 * r^6 = 900)) :
  k = -900 :=
sorry

end polynomial_geometric_roots_k_value_l142_142300


namespace lucy_needs_more_distance_l142_142850

noncomputable def mary_distance : ℝ := (3 / 8) * 24
noncomputable def edna_distance : ℝ := (2 / 3) * mary_distance
noncomputable def lucy_distance : ℝ := (5 / 6) * edna_distance

theorem lucy_needs_more_distance :
  mary_distance - lucy_distance = 4 := by
  sorry

end lucy_needs_more_distance_l142_142850


namespace total_bread_amt_l142_142577

-- Define the conditions
variables (bread_dinner bread_lunch bread_breakfast total_bread : ℕ)
axiom bread_dinner_amt : bread_dinner = 240
axiom dinner_lunch_ratio : bread_dinner = 8 * bread_lunch
axiom dinner_breakfast_ratio : bread_dinner = 6 * bread_breakfast

-- The proof statement
theorem total_bread_amt : total_bread = bread_dinner + bread_lunch + bread_breakfast → total_bread = 310 :=
by
  -- Use the axioms and the given conditions to derive the statement
  sorry

end total_bread_amt_l142_142577


namespace smallest_sum_of_two_perfect_squares_l142_142582

theorem smallest_sum_of_two_perfect_squares (x y : ℕ) (h : x^2 - y^2 = 143) :
  x + y = 13 ∧ x - y = 11 → x^2 + y^2 = 145 :=
by
  -- Add this placeholder "sorry" to skip the proof, as required.
  sorry

end smallest_sum_of_two_perfect_squares_l142_142582


namespace cost_price_per_meter_of_cloth_l142_142732

theorem cost_price_per_meter_of_cloth 
  (total_meters : ℕ)
  (selling_price : ℝ)
  (profit_per_meter : ℝ) 
  (total_profit : ℝ)
  (cp_45 : ℝ)
  (cp_per_meter: ℝ) :
  total_meters = 45 →
  selling_price = 4500 →
  profit_per_meter = 14 →
  total_profit = profit_per_meter * total_meters →
  cp_45 = selling_price - total_profit →
  cp_per_meter = cp_45 / total_meters →
  cp_per_meter = 86 :=
by
  -- your proof here
  sorry

end cost_price_per_meter_of_cloth_l142_142732


namespace condition_sufficient_not_necessary_monotonicity_l142_142479

theorem condition_sufficient_not_necessary_monotonicity
  (f : ℝ → ℝ) (a : ℝ) (h_def : ∀ x, f x = 2^(abs (x - a))) :
  (∀ x > 1, x - a ≥ 0) → (∀ x y, (x > 1) ∧ (y > 1) ∧ (x ≤ y) → f x ≤ f y) ∧
  (∃ a, a ≤ 1 ∧ (∀ x > 1, x - a ≥ 0) ∧ (∀ x y, (x > 1) ∧ (y > 1) ∧ (x ≤ y) → f x ≤ f y)) :=
by
  sorry

end condition_sufficient_not_necessary_monotonicity_l142_142479


namespace sodium_bicarbonate_moles_l142_142116

theorem sodium_bicarbonate_moles (HCl NaHCO3 CO2 : ℕ) (h1 : HCl = 1) (h2 : CO2 = 1) :
  NaHCO3 = 1 :=
by sorry

end sodium_bicarbonate_moles_l142_142116


namespace age_of_replaced_person_is_46_l142_142934

variable (age_of_replaced_person : ℕ)
variable (new_person_age : ℕ := 16)
variable (decrease_in_age_per_person : ℕ := 3)
variable (number_of_people : ℕ := 10)

theorem age_of_replaced_person_is_46 :
  age_of_replaced_person - new_person_age = decrease_in_age_per_person * number_of_people → 
  age_of_replaced_person = 46 :=
by
  sorry

end age_of_replaced_person_is_46_l142_142934


namespace tangent_line_ellipse_l142_142979

theorem tangent_line_ellipse (a b x y x₀ y₀ : ℝ) (h : a > 0) (hb : b > 0) (ha_gt_hb : a > b) 
(h_on_ellipse : (x₀^2 / a^2) + (y₀^2 / b^2) = 1) :
    (x₀ * x / a^2) + (y₀ * y / b^2) = 1 := 
sorry

end tangent_line_ellipse_l142_142979


namespace digit_1035_is_2_l142_142406

noncomputable def sequence_digits (n : ℕ) : ℕ :=
  -- Convert the sequence of numbers from 1 to n to digits and return a specific position.
  sorry

theorem digit_1035_is_2 : sequence_digits 500 = 2 :=
  sorry

end digit_1035_is_2_l142_142406


namespace algebraic_expression_value_l142_142449

-- Define the given condition
def condition (a b : ℝ) : Prop := a + b - 2 = 0

-- State the theorem to prove the algebraic expression value
theorem algebraic_expression_value (a b : ℝ) (h : condition a b) : a^2 - b^2 + 4 * b = 4 := by
  sorry

end algebraic_expression_value_l142_142449


namespace minimum_value_expression_l142_142514

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ( (3*a*b - 6*b + a*(1-a))^2 + (9*b^2 + 2*a + 3*b*(1-a))^2 ) / (a^2 + 9*b^2) ≥ 4 :=
sorry

end minimum_value_expression_l142_142514


namespace Ed_more_marbles_than_Doug_l142_142051

-- Definitions based on conditions
def Ed_marbles_initial : ℕ := 45
def Doug_loss : ℕ := 11
def Doug_marbles_initial : ℕ := Ed_marbles_initial - 10
def Doug_marbles_after_loss : ℕ := Doug_marbles_initial - Doug_loss

-- Theorem statement
theorem Ed_more_marbles_than_Doug :
  Ed_marbles_initial - Doug_marbles_after_loss = 21 :=
by
  -- Proof would go here
  sorry

end Ed_more_marbles_than_Doug_l142_142051


namespace non_trivial_solution_exists_l142_142999

theorem non_trivial_solution_exists (a b c : ℤ) (p : ℕ) [Fact (Nat.Prime p)] :
  ∃ x y z : ℤ, (a * x^2 + b * y^2 + c * z^2) % p = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :=
sorry

end non_trivial_solution_exists_l142_142999


namespace midpoint_trajectory_of_intersecting_line_l142_142399

theorem midpoint_trajectory_of_intersecting_line 
    (h₁ : ∀ x y, x^2 + 2 * y^2 = 4) 
    (h₂ : ∀ M: ℝ × ℝ, M = (4, 6)) :
    ∃ x y, (x-2)^2 / 22 + (y-3)^2 / 11 = 1 :=
sorry

end midpoint_trajectory_of_intersecting_line_l142_142399


namespace greatest_value_exprD_l142_142608

-- Conditions
def a : ℚ := 2
def b : ℚ := 5

-- Expressions
def exprA := a / b
def exprB := b / a
def exprC := a - b
def exprD := b - a
def exprE := (1/2 : ℚ) * a

-- Proof problem statement
theorem greatest_value_exprD : exprD = 3 ∧ exprD > exprA ∧ exprD > exprB ∧ exprD > exprC ∧ exprD > exprE := sorry

end greatest_value_exprD_l142_142608


namespace distribute_marbles_correct_l142_142855

def distribute_marbles (total_marbles : Nat) (num_boys : Nat) : Nat :=
  total_marbles / num_boys

theorem distribute_marbles_correct :
  distribute_marbles 20 2 = 10 := 
by 
  sorry

end distribute_marbles_correct_l142_142855


namespace case1_BL_case2_BL_l142_142580

variable (AD BD BL AL : ℝ)

theorem case1_BL
  (h₁ : AD = 6)
  (h₂ : BD = 12 * Real.sqrt 3)
  (h₃ : AB = 6 * Real.sqrt 13)
  (hADBL : AD / BD = AL / BL)
  (h4 : BL = 2 * AL)
  : BL = 16 * Real.sqrt 3 - 12 := by
  sorry

theorem case2_BL
  (h₁ : AD = 6)
  (h₂ : BD = 12 * Real.sqrt 6)
  (h₃ : AB = 30)
  (hADBL : AD / BD = AL / BL)
  (h4 : BL = 4 * AL)
  : BL = (16 * Real.sqrt 6 - 6) / 5 := by
  sorry

end case1_BL_case2_BL_l142_142580


namespace circle_radius_l142_142907

-- Define the general equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 4 * y = 0

-- Prove the radius of the circle given by the equation is √5
theorem circle_radius :
  (∀ x y : ℝ, circle_eq x y) →
  (∃ r : ℝ, r = Real.sqrt 5) :=
by
  sorry

end circle_radius_l142_142907


namespace lcm_36_125_l142_142246

-- Define the prime factorizations
def factorization_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
def factorization_125 : List (ℕ × ℕ) := [(5, 3)]

-- Least common multiple definition
noncomputable def my_lcm (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

-- Theorem to prove
theorem lcm_36_125 : my_lcm 36 125 = 4500 :=
by
  sorry

end lcm_36_125_l142_142246


namespace find_x_l142_142811

theorem find_x (x : ℕ) : (x % 7 = 0) ∧ (x^2 > 200) ∧ (x < 30) ↔ (x = 21 ∨ x = 28) :=
by
  sorry

end find_x_l142_142811


namespace curves_intersect_on_x_axis_l142_142231

theorem curves_intersect_on_x_axis (t θ a : ℝ) (h : a > 0) :
  (∃ t, (t + 1, 1 - 2 * t).snd = 0) →
  (∃ θ, (a * Real.cos θ, 3 * Real.cos θ).snd = 0) →
  (t + 1 = a * Real.cos θ) →
  a = 3 / 2 :=
by
  intro h1 h2 h3
  sorry

end curves_intersect_on_x_axis_l142_142231


namespace rabbits_in_cage_l142_142324

theorem rabbits_in_cage (heads legs : ℝ) (total_heads : heads = 40) 
  (condition : legs = 8 + 10 * (2 * (heads - rabbits))) :
  ∃ rabbits : ℝ, rabbits = 33 :=
by
  sorry

end rabbits_in_cage_l142_142324


namespace even_integer_operations_l142_142685

theorem even_integer_operations (k : ℤ) (a : ℤ) (h : a = 2 * k) :
  (a * 5) % 2 = 0 ∧ (a ^ 2) % 2 = 0 ∧ (a ^ 3) % 2 = 0 :=
by
  sorry

end even_integer_operations_l142_142685


namespace first_train_cross_time_l142_142844

noncomputable def length_first_train : ℝ := 800
noncomputable def speed_first_train_kmph : ℝ := 120
noncomputable def length_second_train : ℝ := 1000
noncomputable def speed_second_train_kmph : ℝ := 80
noncomputable def length_third_train : ℝ := 600
noncomputable def speed_third_train_kmph : ℝ := 150

noncomputable def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

noncomputable def speed_first_train_mps : ℝ := speed_kmph_to_mps speed_first_train_kmph
noncomputable def speed_second_train_mps : ℝ := speed_kmph_to_mps speed_second_train_kmph
noncomputable def speed_third_train_mps : ℝ := speed_kmph_to_mps speed_third_train_kmph

noncomputable def relative_speed_same_direction : ℝ := speed_first_train_mps - speed_second_train_mps
noncomputable def relative_speed_opposite_direction : ℝ := speed_first_train_mps + speed_third_train_mps

noncomputable def time_to_cross_second_train : ℝ := (length_first_train + length_second_train) / relative_speed_same_direction
noncomputable def time_to_cross_third_train : ℝ := (length_first_train + length_third_train) / relative_speed_opposite_direction

noncomputable def total_time_to_cross : ℝ := time_to_cross_second_train + time_to_cross_third_train

theorem first_train_cross_time : total_time_to_cross = 180.67 := by
  sorry

end first_train_cross_time_l142_142844


namespace candidate_B_valid_votes_l142_142650

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

end candidate_B_valid_votes_l142_142650


namespace find_sisters_dolls_l142_142836

variable (H S : ℕ)

-- Conditions
def hannah_has_5_times_sisters_dolls : Prop :=
  H = 5 * S

def total_dolls_is_48 : Prop :=
  H + S = 48

-- Question: Prove S = 8
theorem find_sisters_dolls (h1 : hannah_has_5_times_sisters_dolls H S) (h2 : total_dolls_is_48 H S) : S = 8 :=
sorry

end find_sisters_dolls_l142_142836


namespace find_XY_sum_in_base10_l142_142144

def base8_addition_step1 (X : ℕ) : Prop :=
  X + 5 = 9

def base8_addition_step2 (Y X : ℕ) : Prop :=
  Y + 3 = X

theorem find_XY_sum_in_base10 (X Y : ℕ) (h1 : base8_addition_step1 X) (h2 : base8_addition_step2 Y X) :
  X + Y = 5 :=
by
  sorry

end find_XY_sum_in_base10_l142_142144


namespace find_softball_players_l142_142793

def total_players : ℕ := 51
def cricket_players : ℕ := 10
def hockey_players : ℕ := 12
def football_players : ℕ := 16

def softball_players : ℕ := total_players - (cricket_players + hockey_players + football_players)

theorem find_softball_players : softball_players = 13 := 
by {
  sorry
}

end find_softball_players_l142_142793


namespace length_of_largest_square_l142_142545

-- Define the conditions of the problem
def side_length_of_shaded_square : ℕ := 10
def side_length_of_largest_square : ℕ := 24

-- The statement to prove
theorem length_of_largest_square (x : ℕ) (h1 : x = side_length_of_shaded_square) : 
  4 * x = side_length_of_largest_square :=
  by
  -- Insert the proof here
  sorry

end length_of_largest_square_l142_142545


namespace b_investment_l142_142298

noncomputable def B_share := 880
noncomputable def A_share := 560
noncomputable def A_investment := 7000
noncomputable def C_investment := 18000
noncomputable def total_investment (B: ℝ) := A_investment + B + C_investment

theorem b_investment (B : ℝ) (P : ℝ)
    (h1 : 7000 / total_investment B * P = A_share)
    (h2 : B / total_investment B * P = B_share) : B = 8000 :=
by
  sorry

end b_investment_l142_142298


namespace average_salary_all_workers_l142_142987

-- Define the given conditions as constants
def num_technicians : ℕ := 7
def avg_salary_technicians : ℕ := 12000

def num_workers_total : ℕ := 21
def num_workers_remaining := num_workers_total - num_technicians
def avg_salary_remaining_workers : ℕ := 6000

-- Define the statement we need to prove
theorem average_salary_all_workers :
  let total_salary_technicians := num_technicians * avg_salary_technicians
  let total_salary_remaining_workers := num_workers_remaining * avg_salary_remaining_workers
  let total_salary_all_workers := total_salary_technicians + total_salary_remaining_workers
  let avg_salary_all_workers := total_salary_all_workers / num_workers_total
  avg_salary_all_workers = 8000 :=
by
  sorry

end average_salary_all_workers_l142_142987


namespace completing_the_square_l142_142274

theorem completing_the_square (x : ℝ) :
  x^2 + 4 * x + 1 = 0 ↔ (x + 2)^2 = 3 :=
by
  sorry

end completing_the_square_l142_142274


namespace angle_R_in_triangle_l142_142702

theorem angle_R_in_triangle (P Q R : ℝ) 
  (hP : P = 90)
  (hQ : Q = 4 * R - 10)
  (angle_sum : P + Q + R = 180) 
  : R = 20 := by 
sorry

end angle_R_in_triangle_l142_142702


namespace max_tied_teams_for_most_wins_l142_142894

-- Definitions based on conditions
def num_teams : ℕ := 7
def total_games_played : ℕ := num_teams * (num_teams - 1) / 2

-- Proposition stating the problem and the expected answer
theorem max_tied_teams_for_most_wins : 
  (∀ (t : ℕ), t ≤ num_teams → ∃ w : ℕ, t * w = total_games_played / num_teams) → 
  t = 7 :=
by
  sorry

end max_tied_teams_for_most_wins_l142_142894


namespace haley_marbles_l142_142532

theorem haley_marbles (boys marbles_per_boy : ℕ) (h1: boys = 5) (h2: marbles_per_boy = 7) : boys * marbles_per_boy = 35 := 
by 
  sorry

end haley_marbles_l142_142532


namespace problem_1_problem_2_l142_142468

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Problem 1
theorem problem_1 (x : ℝ) : (∀ x, f x (-2) > 5) ↔ (x < -4 / 3 ∨ x > 2) :=
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (a : ℝ) : (∀ x, f x a ≤ a * |x + 3|) → (a ≥ 1 / 2) :=
  sorry

end problem_1_problem_2_l142_142468


namespace find_discount_percentage_l142_142568

noncomputable def discount_percentage (P B S : ℝ) (H1 : B = P * (1 - D / 100)) (H2 : S = B * 1.5) (H3 : S - P = P * 0.19999999999999996) : ℝ :=
D

theorem find_discount_percentage (P B S : ℝ) (H1 : B = P * (1 - (60 / 100))) (H2 : S = B * 1.5) (H3 : S - P = P * 0.19999999999999996) : 
  discount_percentage P B S H1 H2 H3 = 60 := sorry

end find_discount_percentage_l142_142568


namespace solve_a_perpendicular_l142_142777

theorem solve_a_perpendicular (a : ℝ) : 
  ((2 * a + 5) * (2 - a) + (a - 2) * (a + 3) = 0) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end solve_a_perpendicular_l142_142777


namespace proof_l142_142822

noncomputable def problem : Prop :=
  let a := 1
  let b := 2
  let angleC := 60 * Real.pi / 180 -- convert degrees to radians
  let cosC := Real.cos angleC
  let sinC := Real.sin angleC
  let c_squared := a^2 + b^2 - 2 * a * b * cosC
  let c := Real.sqrt c_squared
  let area := 0.5 * a * b * sinC
  c = Real.sqrt 3 ∧ area = Real.sqrt 3 / 2

theorem proof : problem :=
by
  sorry

end proof_l142_142822


namespace pies_sold_in_a_week_l142_142537

theorem pies_sold_in_a_week : 
  let Monday := 8
  let Tuesday := 12
  let Wednesday := 14
  let Thursday := 20
  let Friday := 20
  let Saturday := 20
  let Sunday := 20
  Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday = 114 :=
by 
  let Monday := 8
  let Tuesday := 12
  let Wednesday := 14
  let Thursday := 20
  let Friday := 20
  let Saturday := 20
  let Sunday := 20
  have h1 : Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday = 8 + 12 + 14 + 20 + 20 + 20 + 20 := by rfl
  have h2 : 8 + 12 + 14 + 20 + 20 + 20 + 20 = 114 := by norm_num
  exact h1.trans h2

end pies_sold_in_a_week_l142_142537


namespace greatest_number_zero_l142_142801

-- Define the condition (inequality)
def inequality (x : ℤ) : Prop :=
  3 * x + 2 < 5 - 2 * x

-- Define the property of being the greatest whole number satisfying the inequality
def greatest_whole_number (x : ℤ) : Prop :=
  inequality x ∧ (∀ y : ℤ, inequality y → y ≤ x)

-- The main theorem stating the greatest whole number satisfying the inequality is 0
theorem greatest_number_zero : greatest_whole_number 0 :=
by
  sorry

end greatest_number_zero_l142_142801


namespace repeating_decimal_sum_l142_142072

theorem repeating_decimal_sum :
  let x := (0.3333333333333333 : ℚ) -- 0.\overline{3}
  let y := (0.0707070707070707 : ℚ) -- 0.\overline{07}
  let z := (0.008008008008008 : ℚ)  -- 0.\overline{008}
  x + y + z = 418 / 999 := by
sorry

end repeating_decimal_sum_l142_142072


namespace distance_between_Stockholm_and_Malmoe_l142_142533

noncomputable def actualDistanceGivenMapDistanceAndScale (mapDistance : ℕ) (scale : ℕ) : ℕ :=
  mapDistance * scale

theorem distance_between_Stockholm_and_Malmoe (mapDistance : ℕ) (scale : ℕ) :
  mapDistance = 150 → scale = 20 → actualDistanceGivenMapDistanceAndScale mapDistance scale = 3000 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end distance_between_Stockholm_and_Malmoe_l142_142533


namespace calculate_expr_l142_142625

theorem calculate_expr : (125 : ℝ)^(2/3) * 2 = 50 := sorry

end calculate_expr_l142_142625


namespace prove_value_of_expression_l142_142784

theorem prove_value_of_expression (x y a b : ℝ)
    (h1 : x = 2) 
    (h2 : y = 1)
    (h3 : 2 * a + b = 5)
    (h4 : a + 2 * b = 1) : 
    3 - a - b = 1 := 
by
    -- Skipping proof
    sorry

end prove_value_of_expression_l142_142784


namespace sum_of_consecutive_integers_product_336_l142_142018

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y : ℕ), x * (x + 1) = 336 ∧ (y - 1) * y * (y + 1) = 336 ∧ x + (x + 1) + (y - 1) + y + (y + 1) = 54 :=
by
  -- The formal proof would go here
  sorry

end sum_of_consecutive_integers_product_336_l142_142018


namespace arthur_first_day_spending_l142_142666

-- Define the costs of hamburgers and hot dogs.
variable (H D : ℝ)
-- Given conditions
axiom hot_dog_cost : D = 1
axiom second_day_purchase : 2 * H + 3 * D = 7

-- Goal: How much did Arthur spend on the first day?
-- We need to verify that 3H + 4D = 10
theorem arthur_first_day_spending : 3 * H + 4 * D = 10 :=
by
  -- Validating given conditions
  have h1 := hot_dog_cost
  have h2 := second_day_purchase
  -- Insert proof here
  sorry

end arthur_first_day_spending_l142_142666


namespace min_tiles_needed_l142_142150

theorem min_tiles_needed : 
  ∀ (tile_length tile_width region_length region_width: ℕ),
  tile_length = 5 → 
  tile_width = 6 → 
  region_length = 3 * 12 → 
  region_width = 4 * 12 → 
  (region_length * region_width) / (tile_length * tile_width) ≤ 58 :=
by
  intros tile_length tile_width region_length region_width h_tile_length h_tile_width h_region_length h_region_width
  sorry

end min_tiles_needed_l142_142150


namespace period_fraction_sum_nines_l142_142358

theorem period_fraction_sum_nines (q : ℕ) (p : ℕ) (N N1 N2 : ℕ) (n : ℕ) (t : ℕ) 
  (hq_prime : Nat.Prime q) (hq_gt_5 : q > 5) (hp_lt_q : p < q)
  (ht_eq_2n : t = 2 * n) (h_period : 10^t ≡ 1 [MOD q])
  (hN_eq_concat : (N = N1 * 10^n + N2) ∧ (N % 10^n = N2))
  : N1 + N2 = (10^n - 1) := 
sorry

end period_fraction_sum_nines_l142_142358


namespace max_not_divisible_by_3_l142_142306

theorem max_not_divisible_by_3 (a b c d e f : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) (h7 : 3 ∣ (a * b * c * d * e * f)) : 
  ∃ x y z u v, ((x = a ∧ y = b ∧ z = c ∧ u = d ∧ v = e) ∨ (x = a ∧ y = b ∧ z = c ∧ u = d ∧ v = f) ∨ (x = a ∧ y = b ∧ z = c ∧ u = e ∧ v = f) ∨ (x = a ∧ y = b ∧ z = d ∧ u = e ∧ v = f) ∨ (x = a ∧ y = c ∧ z = d ∧ u = e ∧ v = f) ∨ (x = b ∧ y = c ∧ z = d ∧ u = e ∧ v = f)) ∧ (¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z) ∧ ¬ (3 ∣ u) ∧ ¬ (3 ∣ v)) :=
sorry

end max_not_divisible_by_3_l142_142306


namespace original_ratio_l142_142410

theorem original_ratio (x y : ℤ)
  (h1 : y = 48)
  (h2 : (x + 12) * 2 = y) :
  x * 4 = y := sorry

end original_ratio_l142_142410


namespace min_value_of_x2_plus_y2_min_value_of_reciprocal_sum_l142_142839

namespace MathProof

-- Definitions and conditions
variables {x y : ℝ}
axiom x_pos : x > 0
axiom y_pos : y > 0
axiom sum_eq_one : x + y = 1

-- Problem Statement 1: Prove the minimum value of x^2 + y^2 is 1/2
theorem min_value_of_x2_plus_y2 : ∃ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) ∧ (x^2 + y^2 = 1/2) :=
by
  sorry

-- Problem Statement 2: Prove the minimum value of 1/x + 1/y + 1/(xy) is 6
theorem min_value_of_reciprocal_sum : ∃ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) ∧ ((1/x + 1/y + 1/(x*y)) = 6) :=
by
  sorry

end MathProof

end min_value_of_x2_plus_y2_min_value_of_reciprocal_sum_l142_142839


namespace servings_correct_l142_142442

-- Define the pieces of popcorn in a serving
def pieces_per_serving := 30

-- Define the pieces of popcorn Jared can eat
def jared_pieces := 90

-- Define the pieces of popcorn each friend can eat
def friend_pieces := 60

-- Define the number of friends
def friends := 3

-- Calculate total pieces eaten by friends
def total_friend_pieces := friends * friend_pieces

-- Calculate total pieces eaten by everyone
def total_pieces := jared_pieces + total_friend_pieces

-- Calculate the number of servings needed
def servings_needed := total_pieces / pieces_per_serving

theorem servings_correct : servings_needed = 9 :=
by
  sorry

end servings_correct_l142_142442


namespace train_crossing_time_l142_142463

-- Definitions of the given conditions
def length_of_train : ℝ := 110
def speed_of_train_kmph : ℝ := 72
def length_of_bridge : ℝ := 175

noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600

noncomputable def total_distance : ℝ := length_of_train + length_of_bridge

noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross_bridge = 14.25 := 
sorry

end train_crossing_time_l142_142463


namespace Liam_homework_assignments_l142_142981

theorem Liam_homework_assignments : 
  let assignments_needed (points : ℕ) : ℕ := match points with
    | 0     => 0
    | n+1 =>
        if n+1 <= 4 then 1
        else (4 + (((n+1) - 1)/4 - 1))

  30 <= 4 + 8 + 12 + 16 + 20 + 24 + 28 + 16 → ((λ points => List.sum (List.map assignments_needed (List.range points))) 30) = 128 :=
by
  sorry

end Liam_homework_assignments_l142_142981


namespace number_of_valid_trapezoids_l142_142465

noncomputable def calculate_number_of_trapezoids : ℕ :=
  let rows_1 := 7
  let rows_2 := 9
  let unit_spacing := 1
  let height := 2
  -- Here, we should encode the actual combinatorial calculation as per the problem solution
  -- but for the Lean 4 statement, we will provide the correct answer directly.
  361

theorem number_of_valid_trapezoids :
  calculate_number_of_trapezoids = 361 :=
sorry

end number_of_valid_trapezoids_l142_142465


namespace maximize_profit_l142_142997

variable (k : ℚ) -- Proportional constant for deposits
variable (x : ℚ) -- Annual interest rate paid to depositors
variable (D : ℚ) -- Total amount of deposits

-- Define the condition for the total amount of deposits
def deposits (x : ℚ) : ℚ := k * x^2

-- Define the profit function
def profit (x : ℚ) : ℚ := 0.045 * k * x^2 - k * x^3

-- Define the derivative of the profit function
def profit_derivative (x : ℚ) : ℚ := 3 * k * x * (0.03 - x)

-- Statement that x = 0.03 maximizes the bank's profit
theorem maximize_profit : ∃ x, x = 0.03 ∧ (∀ y, profit_derivative y = 0 → x = y) :=
by
  sorry

end maximize_profit_l142_142997


namespace system_solutions_l142_142985

theorem system_solutions : 
  ∃ (x y z t : ℝ), 
    (x * y - t^2 = 9) ∧ 
    (x^2 + y^2 + z^2 = 18) ∧ 
    ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ 
     (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) :=
by {
  sorry
}

end system_solutions_l142_142985


namespace locus_of_circumcenter_l142_142947

theorem locus_of_circumcenter (θ : ℝ) :
  let M := (3, 3 * Real.tan (θ - Real.pi / 3))
  let N := (3, 3 * Real.tan θ)
  let C := (3 / 2, 3 / 2 * Real.tan θ)
  ∃ (x y : ℝ), (x - 4) ^ 2 / 4 - y ^ 2 / 12 = 1 :=
by
  sorry

end locus_of_circumcenter_l142_142947


namespace next_wednesday_l142_142226
open Nat

/-- Prove that the next year after 2010 when April 16 falls on a Wednesday is 2014,
    given the conditions:
    1. 2010 is a non-leap year.
    2. The day advances by 1 day for a non-leap year and 2 days for a leap year.
    3. April 16, 2010 was a Friday. -/
theorem next_wednesday (initial_year : ℕ) (initial_day : String) (target_day : String) : 
  (initial_year = 2010) ∧
  (initial_day = "Friday") ∧ 
  (target_day = "Wednesday") →
  2014 = 2010 + 4 :=
by
  sorry

end next_wednesday_l142_142226


namespace max_value_sqrt_expression_l142_142003

noncomputable def expression_max_value (a b: ℝ) : ℝ :=
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b))

theorem max_value_sqrt_expression : 
  ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 → expression_max_value a b ≤ 1 :=
by
  intros a b h
  sorry

end max_value_sqrt_expression_l142_142003


namespace incorrect_statement_proof_l142_142126

-- Define the conditions as assumptions
def inductive_reasoning_correct : Prop := ∀ (P : Prop), ¬(P → P)
def analogical_reasoning_correct : Prop := ∀ (P Q : Prop), ¬(P → Q)
def reasoning_by_plausibility_correct : Prop := ∀ (P : Prop), ¬(P → P)

-- Define the incorrect statement to be proven
def inductive_reasoning_incorrect_statement : Prop := 
  ¬ (∀ (P Q : Prop), ¬(P ↔ Q))

-- The theorem to be proven
theorem incorrect_statement_proof 
  (h1 : inductive_reasoning_correct)
  (h2 : analogical_reasoning_correct)
  (h3 : reasoning_by_plausibility_correct) : inductive_reasoning_incorrect_statement :=
sorry

end incorrect_statement_proof_l142_142126


namespace cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles_l142_142609

/- Definitions -/
def is_isosceles_right_triangle (triangle : Type) (a b c : ℝ) (angleA angleB angleC : ℝ) : Prop :=
  -- A triangle is isosceles right triangle if it has two equal angles of 45 degrees and a right angle of 90 degrees
  a = b ∧ angleA = 45 ∧ angleB = 45 ∧ angleC = 90

/- Main Problem Statement -/
theorem cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles
  (T1 T2 : Type) (a1 b1 c1 a2 b2 c2 : ℝ) 
  (angleA1 angleB1 angleC1 angleA2 angleB2 angleC2 : ℝ) :
  is_isosceles_right_triangle T1 a1 b1 c1 angleA1 angleB1 angleC1 →
  is_isosceles_right_triangle T2 a2 b2 c2 angleA2 angleB2 angleC2 →
  ¬ (∃ (a b c : ℝ), a = b ∧ b = c ∧ a = c ∧ (a + b + c = 180)) :=
by
  intros hT1 hT2
  intro h
  sorry

end cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles_l142_142609


namespace max_value_of_expression_l142_142146

theorem max_value_of_expression (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h_sum : x + y + z = 3) 
  (h_order : x ≥ y ∧ y ≥ z) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 :=
sorry

end max_value_of_expression_l142_142146


namespace greatest_award_correct_l142_142177

-- Definitions and constants
def total_prize : ℕ := 600
def num_winners : ℕ := 15
def min_award : ℕ := 15
def prize_fraction_num : ℕ := 2
def prize_fraction_den : ℕ := 5
def winners_fraction_num : ℕ := 3
def winners_fraction_den : ℕ := 5

-- Conditions (translated and simplified)
def num_specific_winners : ℕ := (winners_fraction_num * num_winners) / winners_fraction_den
def specific_prize : ℕ := (prize_fraction_num * total_prize) / prize_fraction_den
def remaining_winners : ℕ := num_winners - num_specific_winners
def min_total_award_remaining : ℕ := remaining_winners * min_award
def remaining_prize : ℕ := total_prize - min_total_award_remaining
def min_award_specific : ℕ := num_specific_winners - 1
def sum_min_awards_specific : ℕ := min_award_specific * min_award

-- Correct answer
def greatest_award : ℕ := remaining_prize - sum_min_awards_specific

-- Theorem statement (Proof skipped with sorry)
theorem greatest_award_correct :
  greatest_award = 390 := sorry

end greatest_award_correct_l142_142177


namespace non_increasing_condition_l142_142561

variable {a b : ℝ} (f : ℝ → ℝ)

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem non_increasing_condition (h₀ : ∀ x y, a ≤ x → x < y → y ≤ b → ¬ (f x > f y)) :
  ¬ increasing_on_interval f a b :=
by
  intro h1
  have : ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y := h1
  exact sorry

end non_increasing_condition_l142_142561


namespace dave_paid_3_more_than_doug_l142_142257

theorem dave_paid_3_more_than_doug :
  let total_slices := 10
  let plain_pizza_cost := 10
  let anchovy_fee := 3
  let total_cost := plain_pizza_cost + anchovy_fee
  let cost_per_slice := total_cost / total_slices
  let slices_with_anchovies := total_slices / 3
  let dave_slices := slices_with_anchovies + 2
  let doug_slices := total_slices - dave_slices
  let doug_pay := doug_slices * plain_pizza_cost / total_slices
  let dave_pay := total_cost - doug_pay
  dave_pay - doug_pay = 3 :=
by
  sorry

end dave_paid_3_more_than_doug_l142_142257


namespace nonempty_solution_set_range_l142_142884

theorem nonempty_solution_set_range (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 := sorry

end nonempty_solution_set_range_l142_142884


namespace hall_length_l142_142692

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

end hall_length_l142_142692


namespace student_sums_l142_142860

theorem student_sums (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 48) : y = 36 :=
by
  sorry

end student_sums_l142_142860


namespace no_natural_n_such_that_6n2_plus_5n_is_power_of_2_l142_142805

theorem no_natural_n_such_that_6n2_plus_5n_is_power_of_2 :
  ¬ ∃ n : ℕ, ∃ k : ℕ, 6 * n^2 + 5 * n = 2^k :=
by
  sorry

end no_natural_n_such_that_6n2_plus_5n_is_power_of_2_l142_142805


namespace asymptotes_of_hyperbola_l142_142749

theorem asymptotes_of_hyperbola : 
  (∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1 → y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intro x y h
  sorry

end asymptotes_of_hyperbola_l142_142749


namespace total_profit_for_the_month_l142_142028

theorem total_profit_for_the_month (mean_profit_month : ℕ) (num_days_month : ℕ)
(mean_profit_first15 : ℕ) (num_days_first15 : ℕ) 
(mean_profit_last15 : ℕ) (num_days_last15 : ℕ) 
(h1 : mean_profit_month = 350) (h2 : num_days_month = 30) 
(h3 : mean_profit_first15 = 285) (h4 : num_days_first15 = 15) 
(h5 : mean_profit_last15 = 415) (h6 : num_days_last15 = 15) : 
(mean_profit_first15 * num_days_first15 + mean_profit_last15 * num_days_last15) = 10500 := by
  sorry

end total_profit_for_the_month_l142_142028


namespace equilibrium_temperature_l142_142032

-- Initial conditions for heat capacities and masses
variables (c_B c_W m_B m_W : ℝ) (h : c_W * m_W = 3 * c_B * m_B)

-- Initial temperatures
def T_W_initial := 100
def T_B_initial := 20
def T_f_initial := 80

-- Final equilibrium temperature after second block is added
def final_temp := 68

theorem equilibrium_temperature (t : ℝ)
  (h_first_eq : c_W * m_W * (T_W_initial - T_f_initial) = c_B * m_B * (T_f_initial - T_B_initial))
  (h_second_eq : c_W * m_W * (T_f_initial - t) + c_B * m_B * (T_f_initial - t) = c_B * m_B * (t - T_B_initial)) :
  t = final_temp :=
by 
  sorry

end equilibrium_temperature_l142_142032


namespace triangle_obtuse_of_eccentricities_l142_142299

noncomputable def is_obtuse_triangle (a b m : ℝ) : Prop :=
  a^2 + b^2 - m^2 < 0

theorem triangle_obtuse_of_eccentricities (a b m : ℝ) (ha : a > 0) (hm : m > b) (hb : b > 0)
  (ecc_cond : (Real.sqrt (a^2 + b^2) / a) * (Real.sqrt (m^2 - b^2) / m) > 1) :
  is_obtuse_triangle a b m := 
sorry

end triangle_obtuse_of_eccentricities_l142_142299


namespace feta_price_calculation_l142_142172

noncomputable def feta_price_per_pound (sandwiches_price : ℝ) (sandwiches_count : ℕ) 
  (salami_price : ℝ) (brie_factor : ℝ) (olive_price_per_pound : ℝ) 
  (olive_weight : ℝ) (bread_price : ℝ) (total_spent : ℝ)
  (feta_weight : ℝ) :=
  (total_spent - (sandwiches_count * sandwiches_price + salami_price + brie_factor * salami_price + olive_price_per_pound * olive_weight + bread_price)) / feta_weight

theorem feta_price_calculation : 
  feta_price_per_pound 7.75 2 4.00 3 10.00 0.25 2.00 40.00 0.5 = 8.00 := 
by
  sorry

end feta_price_calculation_l142_142172


namespace water_tank_capacity_l142_142420

theorem water_tank_capacity :
  ∃ (x : ℝ), 0.9 * x - 0.4 * x = 30 → x = 60 :=
by
  sorry

end water_tank_capacity_l142_142420


namespace infinite_chain_resistance_l142_142901

variables (R_0 R_X : ℝ)
def infinite_chain_resistance_condition (R_0 : ℝ) (R_X : ℝ) : Prop :=
  R_X = R_0 + (R_0 * R_X) / (R_0 + R_X)

theorem infinite_chain_resistance (R_0 : ℝ) (h : R_0 = 50) :
  ∃ R_X, infinite_chain_resistance_condition R_0 R_X ∧ R_X = (R_0 * (1 + Real.sqrt 5)) / 2 :=
  sorry

end infinite_chain_resistance_l142_142901


namespace find_n_l142_142260

def sum_for (x : ℕ) : ℕ :=
  if x > 1 then (List.range (2*x)).sum else 0

theorem find_n (n : ℕ) (h : n * (sum_for 4) = 360) : n = 10 :=
by
  sorry

end find_n_l142_142260


namespace smallest_internal_angle_l142_142236

theorem smallest_internal_angle (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : α = 2 * β) (h2 : α = 3 * γ)
  (h3 : α + β + γ = π) :
  α = π / 6 :=
by
  sorry

end smallest_internal_angle_l142_142236


namespace cost_of_TOP_book_l142_142591

theorem cost_of_TOP_book (T : ℝ) (h1 : T = 8)
  (abc_cost : ℝ := 23)
  (top_books_sold : ℝ := 13)
  (abc_books_sold : ℝ := 4)
  (earnings_difference : ℝ := 12)
  (h2 : top_books_sold * T - abc_books_sold * abc_cost = earnings_difference) :
  T = 8 := 
by
  sorry

end cost_of_TOP_book_l142_142591


namespace triangle_side_lengths_exist_l142_142024

theorem triangle_side_lengths_exist :
  ∃ (a b c : ℕ), a ≥ b ∧ b ≥ c ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ abc = 2 * (a - 1) * (b - 1) * (c - 1) ∧
  ((a, b, c) = (8, 7, 3) ∨ (a, b, c) = (6, 5, 4)) :=
by sorry

end triangle_side_lengths_exist_l142_142024


namespace part1_part2_l142_142385

noncomputable def f (x a : ℝ) : ℝ := |x - 1| - 2 * |x + a|
noncomputable def g (x b : ℝ) : ℝ := 0.5 * x + b

theorem part1 (a : ℝ) (h : a = 1/2) : 
  { x : ℝ | f x a ≤ 0 } = { x : ℝ | x ≤ -2 ∨ x ≥ 0 } :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ -1) (h2 : ∀ x, g x b ≥ f x a) : 
  2 * b - 3 * a > 2 :=
sorry

end part1_part2_l142_142385


namespace fraction_multiplication_l142_142346

theorem fraction_multiplication : (1 / 2) * (1 / 3) * (1 / 6) * 108 = 3 := by
  sorry

end fraction_multiplication_l142_142346


namespace rectangle_area_given_conditions_l142_142035

theorem rectangle_area_given_conditions
  (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by sorry

end rectangle_area_given_conditions_l142_142035


namespace exactly_one_wins_at_most_two_win_l142_142733

def prob_A : ℚ := 4 / 5 
def prob_B : ℚ := 3 / 5 
def prob_C : ℚ := 7 / 10

theorem exactly_one_wins :
  (prob_A * (1 - prob_B) * (1 - prob_C) + 
   (1 - prob_A) * prob_B * (1 - prob_C) + 
   (1 - prob_A) * (1 - prob_B) * prob_C) = 47 / 250 := 
by sorry

theorem at_most_two_win :
  (1 - (prob_A * prob_B * prob_C)) = 83 / 125 :=
by sorry

end exactly_one_wins_at_most_two_win_l142_142733


namespace symmetric_points_y_axis_l142_142328

theorem symmetric_points_y_axis :
  ∀ (m n : ℝ), (m + 4 = 0) → (n = 3) → (m + n) ^ 2023 = -1 :=
by
  intros m n Hm Hn
  sorry

end symmetric_points_y_axis_l142_142328


namespace dawn_hours_l142_142009

-- Define the conditions
def pedestrian_walked_from_A_to_B (x : ℕ) : Prop :=
  x > 0

def pedestrian_walked_from_B_to_A (x : ℕ) : Prop :=
  x > 0

def met_at_noon (x : ℕ) : Prop :=
  x > 0

def arrived_at_B_at_4pm (x : ℕ) : Prop :=
  x > 0

def arrived_at_A_at_9pm (x : ℕ) : Prop :=
  x > 0

-- Define the theorem to prove
theorem dawn_hours (x : ℕ) :
  pedestrian_walked_from_A_to_B x ∧ 
  pedestrian_walked_from_B_to_A x ∧
  met_at_noon x ∧ 
  arrived_at_B_at_4pm x ∧ 
  arrived_at_A_at_9pm x → 
  x = 6 := 
sorry

end dawn_hours_l142_142009


namespace solve_x_for_collinear_and_same_direction_l142_142166

-- Define vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (-1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (-x, 2)

-- Define the conditions for collinearity and same direction
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k • b.1, k • b.2)

def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (k • b.1, k • b.2)

theorem solve_x_for_collinear_and_same_direction
  (x : ℝ)
  (h_collinear : collinear (vector_a x) (vector_b x))
  (h_same_direction : same_direction (vector_a x) (vector_b x)) :
  x = Real.sqrt 2 :=
sorry

end solve_x_for_collinear_and_same_direction_l142_142166


namespace y_run_time_l142_142715

theorem y_run_time (t : ℕ) (h_avg : (t + 26) / 2 = 42) : t = 58 :=
by
  sorry

end y_run_time_l142_142715


namespace convert_quadratic_l142_142661

theorem convert_quadratic (x : ℝ) :
  (1 + 3 * x) * (x - 3) = 2 * x ^ 2 + 1 ↔ x ^ 2 - 8 * x - 4 = 0 := 
by sorry

end convert_quadratic_l142_142661


namespace gcf_of_24_and_16_l142_142335

theorem gcf_of_24_and_16 :
  let n := 24
  let lcm := 48
  gcd n 16 = 8 :=
by
  sorry

end gcf_of_24_and_16_l142_142335


namespace length_of_LN_l142_142216

theorem length_of_LN 
  (sinN : ℝ)
  (LM LN : ℝ)
  (h1 : sinN = 3 / 5)
  (h2 : LM = 20)
  (h3 : sinN = LM / LN) :
  LN = 100 / 3 :=
by
  sorry

end length_of_LN_l142_142216


namespace money_made_l142_142474

-- Define the conditions
def cost_per_bar := 4
def total_bars := 8
def bars_sold := total_bars - 3

-- We need to show that the money made is $20
theorem money_made :
  bars_sold * cost_per_bar = 20 := 
by
  sorry

end money_made_l142_142474


namespace find_n_satisfying_conditions_l142_142241

noncomputable def exists_set_satisfying_conditions (n : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card = n ∧
  (∀ x ∈ S, x < 2^(n-1)) ∧
  ∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ B → A ≠ ∅ → B ≠ ∅ → A.sum id ≠ B.sum id

theorem find_n_satisfying_conditions : ∀ n : ℕ, (n ≥ 4) ↔ exists_set_satisfying_conditions n :=
sorry

end find_n_satisfying_conditions_l142_142241


namespace carol_weight_l142_142063

variable (a c : ℝ)

-- Conditions based on the problem statement
def combined_weight : Prop := a + c = 280
def weight_difference : Prop := c - a = c / 3

theorem carol_weight (h1 : combined_weight a c) (h2 : weight_difference a c) : c = 168 :=
by
  -- Proof goes here
  sorry

end carol_weight_l142_142063


namespace hypotenuse_is_18_point_8_l142_142428

def hypotenuse_of_right_triangle (a b c : ℝ) : Prop :=
  a + b + c = 40 ∧ (1/2) * a * b = 24 ∧ a^2 + b^2 = c^2

theorem hypotenuse_is_18_point_8 (a b c : ℝ) (h : hypotenuse_of_right_triangle a b c) : c = 18.8 :=
  sorry

end hypotenuse_is_18_point_8_l142_142428


namespace A_wins_B_no_more_than_two_throws_C_treats_after_two_throws_C_treats_exactly_two_days_l142_142525

def prob_A_wins_B_one_throw : ℚ := 1 / 3
def prob_tie_one_throw : ℚ := 1 / 3
def prob_A_wins_B_no_more_2_throws : ℚ := 4 / 9

def prob_C_treats_two_throws : ℚ := 2 / 9

def prob_C_treats_exactly_2_days_out_of_3 : ℚ := 28 / 243

theorem A_wins_B_no_more_than_two_throws (P1 : ℚ := prob_A_wins_B_one_throw) (P2 : ℚ := prob_tie_one_throw) :
  P1 + P2 * P1 = prob_A_wins_B_no_more_2_throws := 
by
  sorry

theorem C_treats_after_two_throws : prob_tie_one_throw ^ 2 = prob_C_treats_two_throws :=
by
  sorry

theorem C_treats_exactly_two_days (n : ℕ := 3) (k : ℕ := 2) (p_success : ℚ := prob_C_treats_two_throws) :
  (n.choose k) * (p_success ^ k) * ((1 - p_success) ^ (n - k)) = prob_C_treats_exactly_2_days_out_of_3 :=
by
  sorry

end A_wins_B_no_more_than_two_throws_C_treats_after_two_throws_C_treats_exactly_two_days_l142_142525


namespace sum_in_base5_correct_l142_142697

-- Define numbers in base 5
def n1 : ℕ := 231
def n2 : ℕ := 414
def n3 : ℕ := 123

-- Function to convert a number from base 5 to base 10
def base5_to_base10(n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100)
  d0 * 1 + d1 * 5 + d2 * 25

-- Convert the given numbers from base 5 to base 10
def n1_base10 : ℕ := base5_to_base10 n1
def n2_base10 : ℕ := base5_to_base10 n2
def n3_base10 : ℕ := base5_to_base10 n3

-- Base 10 sum
def sum_base10 : ℕ := n1_base10 + n2_base10 + n3_base10

-- Function to convert a number from base 10 to base 5
def base10_to_base5(n : ℕ) : ℕ :=
  let d0 := n % 5
  let d1 := (n / 5) % 5
  let d2 := (n / 25) % 5
  let d3 := (n / 125)
  d0 * 1 + d1 * 10 + d2 * 100 + d3 * 1000

-- Convert the sum from base 10 to base 5
def sum_base5 : ℕ := base10_to_base5 sum_base10

-- The theorem to prove the sum in base 5 is 1323_5
theorem sum_in_base5_correct : sum_base5 = 1323 := by
  -- Proof steps would go here, but we insert sorry to skip it
  sorry

end sum_in_base5_correct_l142_142697


namespace alcohol_to_water_ratio_l142_142674

theorem alcohol_to_water_ratio (V p q : ℝ) (hV : V > 0) (hp : p > 0) (hq : q > 0) :
  let alcohol_first_jar := (p / (p + 1)) * V
  let water_first_jar   := (1 / (p + 1)) * V
  let alcohol_second_jar := (2 * q / (q + 1)) * V
  let water_second_jar   := (2 / (q + 1)) * V
  let total_alcohol := alcohol_first_jar + alcohol_second_jar
  let total_water := water_first_jar + water_second_jar
  (total_alcohol / total_water) = ((p * (q + 1) + 2 * p + 2 * q) / (q + 1 + 2 * p + 2)) :=
by
  sorry

end alcohol_to_water_ratio_l142_142674


namespace area_ratio_S_T_l142_142570

open Set

def T : Set (ℝ × ℝ × ℝ) := {p | let (x, y, z) := p; x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 1}

def supports (p q : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  let (a, b, c) := q
  (x ≥ a ∧ y ≥ b) ∨ (x ≥ a ∧ z ≥ c) ∨ (y ≥ b ∧ z ≥ c)

def S : Set (ℝ × ℝ × ℝ) := {p ∈ T | supports p (1/4, 1/4, 1/2)}

theorem area_ratio_S_T : ∃ k : ℝ, k = 3 / 4 ∧
  ∃ (area_T area_S : ℝ), area_T ≠ 0 ∧ (area_S / area_T = k) := sorry

end area_ratio_S_T_l142_142570


namespace total_oranges_over_four_days_l142_142871

def jeremy_oranges_monday := 100
def jeremy_oranges_tuesday (B: ℕ) := 3 * jeremy_oranges_monday
def jeremy_oranges_wednesday (B: ℕ) (C: ℕ) := 2 * (jeremy_oranges_monday + B)
def jeremy_oranges_thursday := 70
def brother_oranges_tuesday := 3 * jeremy_oranges_monday - jeremy_oranges_monday -- This is B from Tuesday
def cousin_oranges_wednesday (B: ℕ) (C: ℕ) := 2 * (jeremy_oranges_monday + B) - (jeremy_oranges_monday + B)

theorem total_oranges_over_four_days (B: ℕ) (C: ℕ)
        (B_equals_tuesday: B = brother_oranges_tuesday)
        (J_plus_B_equals_300 : jeremy_oranges_tuesday B = 300)
        (J_plus_B_plus_C_equals_600 : jeremy_oranges_wednesday B C = 600)
        (J_thursday_is_70 : jeremy_oranges_thursday = 70)
        (B_thursday_is_B : B = brother_oranges_tuesday):
    100 + 300 + 600 + 270 = 1270 := by
        sorry

end total_oranges_over_four_days_l142_142871


namespace scientific_notation_of_virus_diameter_l142_142267

theorem scientific_notation_of_virus_diameter :
  0.00000012 = 1.2 * 10 ^ (-7) :=
by
  sorry

end scientific_notation_of_virus_diameter_l142_142267


namespace initial_notebooks_l142_142441

variable (a n : ℕ)
variable (h1 : n = 13 * a + 8)
variable (h2 : n = 15 * a)

theorem initial_notebooks : n = 60 := by
  -- additional details within the proof
  sorry

end initial_notebooks_l142_142441


namespace tan_beta_of_tan_alpha_and_tan_alpha_plus_beta_l142_142493

theorem tan_beta_of_tan_alpha_and_tan_alpha_plus_beta (α β : ℝ)
  (h1 : Real.tan α = 2)
  (h2 : Real.tan (α + β) = 1 / 5) :
  Real.tan β = -9 / 7 :=
sorry

end tan_beta_of_tan_alpha_and_tan_alpha_plus_beta_l142_142493


namespace intersecting_chords_second_length_l142_142892

theorem intersecting_chords_second_length (a b : ℕ) (k : ℕ) 
  (h_a : a = 12) (h_b : b = 18) (h_ratio : k ^ 2 = (a * b) / 24) 
  (x y : ℕ) (h_x : x = 3 * k) (h_y : y = 8 * k) :
  x + y = 33 :=
by
  sorry

end intersecting_chords_second_length_l142_142892


namespace otimes_2_3_eq_23_l142_142397

-- Define the new operation
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- The proof statement
theorem otimes_2_3_eq_23 : otimes 2 3 = 23 := 
  by 
  sorry

end otimes_2_3_eq_23_l142_142397


namespace parking_lot_perimeter_l142_142998

theorem parking_lot_perimeter (a b : ℝ) 
  (h_diag : a^2 + b^2 = 784) 
  (h_area : a * b = 180) : 
  2 * (a + b) = 68 := 
by 
  sorry

end parking_lot_perimeter_l142_142998


namespace work_efficiency_ratio_l142_142099

theorem work_efficiency_ratio (a b k : ℝ) (ha : a = k * b) (hb : b = 1/15)
  (hab : a + b = 1/5) : k = 2 :=
by sorry

end work_efficiency_ratio_l142_142099


namespace fraction_filled_l142_142909

-- Definitions for the given conditions
variables (x C : ℝ) (h₁ : 20 * x / 3 = 25 * C / 5) 

-- The goal is to show that x / C = 3 / 4
theorem fraction_filled (h₁ : 20 * x / 3 = 25 * C / 5) : x / C = 3 / 4 :=
by sorry

end fraction_filled_l142_142909


namespace complex_sum_real_imag_l142_142980

theorem complex_sum_real_imag : 
  (Complex.re ((Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I))) + 
  Complex.im ((Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I)))) = 3/2 := 
by sorry

end complex_sum_real_imag_l142_142980


namespace find_number_eq_seven_point_five_l142_142125

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end find_number_eq_seven_point_five_l142_142125


namespace polynomial_bound_implies_l142_142029

theorem polynomial_bound_implies :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2) :=
by
  sorry

end polynomial_bound_implies_l142_142029


namespace double_root_polynomial_l142_142908

theorem double_root_polynomial (b4 b3 b2 b1 : ℤ) (s : ℤ) :
  (Polynomial.eval s (Polynomial.C 1 * Polynomial.X^5 + Polynomial.C b4 * Polynomial.X^4 + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C 24) = 0)
  ∧ (Polynomial.eval s (Polynomial.derivative (Polynomial.C 1 * Polynomial.X^5 + Polynomial.C b4 * Polynomial.X^4 + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C 24)) = 0)
  → s = 1 ∨ s = -1 ∨ s = 2 ∨ s = -2 :=
by
  sorry

end double_root_polynomial_l142_142908


namespace area_of_isosceles_right_triangle_l142_142758

def is_isosceles_right_triangle (X Y Z : Type*) : Prop :=
∃ (XY YZ XZ : ℝ), XY = 6.000000000000001 ∧ XY > YZ ∧ YZ = XZ ∧ XY = YZ * Real.sqrt 2

theorem area_of_isosceles_right_triangle
  {X Y Z : Type*}
  (h : is_isosceles_right_triangle X Y Z) :
  ∃ A : ℝ, A = 9.000000000000002 :=
by
  sorry

end area_of_isosceles_right_triangle_l142_142758


namespace solution_set_inequality_l142_142293

theorem solution_set_inequality (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x + 2 > 0) ↔ m ∈ Set.Ico 0 8 := by
  sorry

end solution_set_inequality_l142_142293


namespace compute_series_l142_142244

noncomputable def sum_series (c d : ℝ) : ℝ :=
  ∑' n, 1 / ((n-1) * d - (n-2) * c) / (n * d - (n-1) * c)

theorem compute_series (c d : ℝ) (hc_pos : 0 < c) (hd_pos : 0 < d) (hcd : d < c) : 
  sum_series c d = 1 / ((d - c) * c) :=
sorry

end compute_series_l142_142244


namespace least_xy_value_l142_142355

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (1/x : ℚ) + 1/(2*y) = 1/8) :
  xy ≥ 128 :=
sorry

end least_xy_value_l142_142355


namespace negation_correct_l142_142378

theorem negation_correct (x : ℝ) : -(3 * x - 2) = -3 * x + 2 := 
by sorry

end negation_correct_l142_142378


namespace isosceles_triangle_legs_length_l142_142407

theorem isosceles_triangle_legs_length 
  (P : ℝ) (base : ℝ) (leg_length : ℝ) 
  (hp : P = 26) 
  (hb : base = 11) 
  (hP : P = 2 * leg_length + base) : 
  leg_length = 7.5 := 
by 
  sorry

end isosceles_triangle_legs_length_l142_142407


namespace calculate_mirror_area_l142_142597

def outer_frame_width : ℝ := 65
def outer_frame_height : ℝ := 85
def frame_width : ℝ := 15

def mirror_width : ℝ := outer_frame_width - 2 * frame_width
def mirror_height : ℝ := outer_frame_height - 2 * frame_width
def mirror_area : ℝ := mirror_width * mirror_height

theorem calculate_mirror_area : mirror_area = 1925 := by
  sorry

end calculate_mirror_area_l142_142597


namespace length_of_platform_l142_142766

variable (L : ℕ)

theorem length_of_platform
  (train_length : ℕ)
  (time_cross_post : ℕ)
  (time_cross_platform : ℕ)
  (train_length_eq : train_length = 300)
  (time_cross_post_eq : time_cross_post = 18)
  (time_cross_platform_eq : time_cross_platform = 39)
  : L = 350 := sorry

end length_of_platform_l142_142766


namespace total_cupcakes_baked_l142_142502

-- Conditions
def morning_cupcakes : ℕ := 20
def afternoon_cupcakes : ℕ := morning_cupcakes + 15

-- Goal
theorem total_cupcakes_baked :
  (morning_cupcakes + afternoon_cupcakes) = 55 :=
by
  sorry

end total_cupcakes_baked_l142_142502


namespace line_eq_l142_142480

theorem line_eq (m b : ℝ) 
  (h_slope : m = (4 + 2) / (3 - 1)) 
  (h_point : -2 = m * 1 + b) :
  m + b = -2 :=
by
  sorry

end line_eq_l142_142480


namespace reading_time_difference_l142_142927

theorem reading_time_difference :
  let xanthia_reading_speed := 100 -- pages per hour
  let molly_reading_speed := 50 -- pages per hour
  let book_pages := 225
  let xanthia_time := book_pages / xanthia_reading_speed
  let molly_time := book_pages / molly_reading_speed
  let difference_in_hours := molly_time - xanthia_time
  let difference_in_minutes := difference_in_hours * 60
  difference_in_minutes = 135 := by
  sorry

end reading_time_difference_l142_142927


namespace reeya_average_l142_142026

theorem reeya_average (s1 s2 s3 s4 s5 : ℕ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 82) (h5 : s5 = 85) :
  (s1 + s2 + s3 + s4 + s5) / 5 = 75 := by
  sorry

end reeya_average_l142_142026


namespace sequence_B_is_arithmetic_l142_142791

-- Definitions of the sequences
def S_n (n : ℕ) : ℕ := 2*n + 1

-- Theorem statement
theorem sequence_B_is_arithmetic : ∀ n : ℕ, S_n (n + 1) - S_n n = 2 :=
by
  intro n
  sorry

end sequence_B_is_arithmetic_l142_142791


namespace product_of_a_l142_142200

theorem product_of_a : 
  (∃ a b : ℝ, (3 * a - 5)^2 + (a - 5 - (-2))^2 = (3 * Real.sqrt 13)^2 ∧ 
    (a * b = -8.32)) :=
by 
  sorry

end product_of_a_l142_142200


namespace average_class_size_l142_142875

theorem average_class_size 
  (three_year_olds : ℕ := 13)
  (four_year_olds : ℕ := 20)
  (five_year_olds : ℕ := 15)
  (six_year_olds : ℕ := 22) : 
  ((three_year_olds + four_year_olds + five_year_olds + six_year_olds) / 2) = 35 := 
by
  sorry

end average_class_size_l142_142875


namespace larger_fraction_l142_142253

theorem larger_fraction :
  (22222222221 : ℚ) / 22222222223 > (33333333331 : ℚ) / 33333333334 := by sorry

end larger_fraction_l142_142253


namespace employees_without_increase_l142_142113

-- Define the constants and conditions
def total_employees : ℕ := 480
def salary_increase_percentage : ℕ := 10
def travel_allowance_increase_percentage : ℕ := 20

-- Define the calculations derived from conditions
def employees_with_salary_increase : ℕ := (salary_increase_percentage * total_employees) / 100
def employees_with_travel_allowance_increase : ℕ := (travel_allowance_increase_percentage * total_employees) / 100

-- Total employees who got increases assuming no overlap
def employees_with_increases : ℕ := employees_with_salary_increase + employees_with_travel_allowance_increase

-- The proof statement
theorem employees_without_increase :
  total_employees - employees_with_increases = 336 := by
  sorry

end employees_without_increase_l142_142113


namespace max_mn_value_l142_142895

noncomputable def vector_max_sum (OA OB : ℝ) (m n : ℝ) : Prop :=
  (OA * OA = 4 ∧ OB * OB = 4 ∧ OA * OB = 2) →
  ((m * OA + n * OB) * (m * OA + n * OB) = 4) →
  (m + n ≤ 2 * Real.sqrt 3 / 3)

-- Here's the statement for the maximum value problem
theorem max_mn_value {m n : ℝ} (h1 : m > 0) (h2 : n > 0) :
  vector_max_sum 2 2 m n :=
sorry

end max_mn_value_l142_142895


namespace find_a_for_even_function_l142_142973

theorem find_a_for_even_function (a : ℝ) : 
  (∀ x : ℝ, (x + a) * (x - 4) = ((-x) + a) * ((-x) - 4)) → a = 4 :=
by sorry

end find_a_for_even_function_l142_142973


namespace final_values_comparison_l142_142259

theorem final_values_comparison :
  let AA_initial : ℝ := 100
  let BB_initial : ℝ := 100
  let CC_initial : ℝ := 100
  let AA_year1 := AA_initial * 1.20
  let BB_year1 := BB_initial * 0.75
  let CC_year1 := CC_initial
  let AA_year2 := AA_year1 * 0.80
  let BB_year2 := BB_year1 * 1.25
  let CC_year2 := CC_year1
  AA_year2 = 96 ∧ BB_year2 = 93.75 ∧ CC_year2 = 100 ∧ BB_year2 < AA_year2 ∧ AA_year2 < CC_year2 :=
by {
  -- Definitions from conditions
  let AA_initial : ℝ := 100;
  let BB_initial : ℝ := 100;
  let CC_initial : ℝ := 100;
  let AA_year1 := AA_initial * 1.20;
  let BB_year1 := BB_initial * 0.75;
  let CC_year1 := CC_initial;
  let AA_year2 := AA_year1 * 0.80;
  let BB_year2 := BB_year1 * 1.25;
  let CC_year2 := CC_year1;

  -- Use sorry to skip the actual proof
  sorry
}

end final_values_comparison_l142_142259


namespace burgers_per_day_l142_142094

def calories_per_burger : ℝ := 20
def total_calories_after_two_days : ℝ := 120

theorem burgers_per_day :
  total_calories_after_two_days / (2 * calories_per_burger) = 3 := 
by
  sorry

end burgers_per_day_l142_142094


namespace prove_inequality_l142_142521

theorem prove_inequality
  (a : ℕ → ℕ) -- Define a sequence of natural numbers
  (h_initial : a 1 > a 0) -- Initial condition
  (h_recurrence : ∀ n ≥ 2, a n = 3 * a (n - 1) - 2 * a (n - 2)) -- Recurrence relation
  : a 100 > 2^99 := by
  sorry -- Proof placeholder

end prove_inequality_l142_142521


namespace total_cost_is_72_l142_142392

-- Definitions based on conditions
def adults (total_people : ℕ) (kids : ℕ) : ℕ := total_people - kids
def cost_per_adult_meal (cost_per_meal : ℕ) (adults : ℕ) : ℕ := cost_per_meal * adults
def total_cost (total_people : ℕ) (kids : ℕ) (cost_per_meal : ℕ) : ℕ := 
  cost_per_adult_meal cost_per_meal (adults total_people kids)

-- Given values
def total_people := 11
def kids := 2
def cost_per_meal := 8

-- Theorem statement
theorem total_cost_is_72 : total_cost total_people kids cost_per_meal = 72 := by
  sorry

end total_cost_is_72_l142_142392


namespace monotone_f_range_l142_142527

noncomputable def f (a x : ℝ) : ℝ :=
  x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_f_range (a : ℝ) :
  (∀ x : ℝ, (1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x) ≥ 0) ↔ (-1 / 3 ≤ a ∧ a ≤ 1 / 3) := 
sorry

end monotone_f_range_l142_142527


namespace sum_series_eq_two_l142_142695

theorem sum_series_eq_two : ∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1)) = 2 :=
sorry

end sum_series_eq_two_l142_142695


namespace units_digit_of_calculation_l142_142993

-- Base definitions for units digits of given numbers
def units_digit (n : ℕ) : ℕ := n % 10

-- Main statement to prove
theorem units_digit_of_calculation : 
  units_digit ((25 ^ 3 + 17 ^ 3) * 12 ^ 2) = 2 :=
by
  -- This is where the proof would go, but it's omitted as requested
  sorry

end units_digit_of_calculation_l142_142993


namespace skylar_current_age_l142_142558

noncomputable def skylar_age_now (donation_start_age : ℕ) (annual_donation total_donation : ℕ) : ℕ :=
  donation_start_age + total_donation / annual_donation

theorem skylar_current_age : skylar_age_now 13 5000 105000 = 34 := by
  -- Proof follows from the conditions
  sorry

end skylar_current_age_l142_142558


namespace area_and_perimeter_l142_142058

-- Given a rectangle R with length l and width w
variables (l w : ℝ)
-- Define the area of R
def area_R : ℝ := l * w

-- Define a smaller rectangle that is cut out, with an area A_cut
variables (A_cut : ℝ)
-- Define the area of the resulting figure S
def area_S : ℝ := area_R l w - A_cut

-- Define the perimeter of R
def perimeter_R : ℝ := 2 * l + 2 * w

-- perimeter_R remains the same after cutting out the smaller rectangle
theorem area_and_perimeter (h_cut : 0 < A_cut) (h_cut_le : A_cut ≤ area_R l w) : 
  (area_S l w A_cut < area_R l w) ∧ (perimeter_R l w = perimeter_R l w) :=
by
  sorry

end area_and_perimeter_l142_142058


namespace sector_area_l142_142651

theorem sector_area (n : ℝ) (r : ℝ) (h₁ : n = 120) (h₂ : r = 4) : 
  (n * Real.pi * r^2 / 360) = (16 * Real.pi / 3) :=
by 
  sorry

end sector_area_l142_142651


namespace greatest_possible_value_of_x_l142_142314

theorem greatest_possible_value_of_x (x : ℕ) (H : Nat.lcm (Nat.lcm x 12) 18 = 108) : x ≤ 108 := sorry

end greatest_possible_value_of_x_l142_142314


namespace min_PM_PN_l142_142519

noncomputable def C1 (x y : ℝ) : Prop := (x + 6)^2 + (y - 5)^2 = 4
noncomputable def C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

theorem min_PM_PN : ∀ (P M N : ℝ × ℝ),
  P.2 = 0 ∧ C1 M.1 M.2 ∧ C2 N.1 N.2 → (|P.1 - M.1| + (P.1 - N.1)^2 + (P.2 - N.2)^2).sqrt = 7 := by
  sorry

end min_PM_PN_l142_142519


namespace find_integer_cube_sum_l142_142699

-- Define the problem in Lean
theorem find_integer_cube_sum : ∃ n : ℤ, n^3 = (n-1)^3 + (n-2)^3 + (n-3)^3 := by
  use 6
  sorry

end find_integer_cube_sum_l142_142699


namespace salt_fraction_l142_142100

variables {a x : ℝ}

-- First condition: the shortfall in salt the first time
def shortfall_first (a x : ℝ) : ℝ := a - x

-- Second condition: the shortfall in salt the second time
def shortfall_second (a x : ℝ) : ℝ := a - 2 * x

-- Third condition: relationship given by the problem
axiom condition : shortfall_first a x = 2 * shortfall_second a x

-- Prove fraction of necessary salt added the first time is 1/3
theorem salt_fraction (a x : ℝ) (h : shortfall_first a x = 2 * shortfall_second a x) : x = a / 3 :=
by
  sorry

end salt_fraction_l142_142100


namespace find_constant_l142_142900

-- Definitions based on the conditions provided
variable (f : ℕ → ℕ)
variable (c : ℕ)

-- Given conditions
def f_1_eq_0 : f 1 = 0 := sorry
def functional_equation (m n : ℕ) : f (m + n) = f m + f n + c * (m * n - 1) := sorry
def f_17_eq_4832 : f 17 = 4832 := sorry

-- The mathematically equivalent proof problem
theorem find_constant : c = 4 := 
sorry

end find_constant_l142_142900


namespace number_of_multiples_in_range_l142_142554

-- Definitions based on given conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def in_range (x lower upper : ℕ) : Prop := lower ≤ x ∧ x ≤ upper

def lcm_18_24_30 := ((2^3) * (3^2) * 5) -- LCM of 18, 24, and 30

-- Main theorem statement
theorem number_of_multiples_in_range : 
  (∃ a b c : ℕ, in_range a 2000 3000 ∧ is_multiple_of a lcm_18_24_30 ∧ 
                in_range b 2000 3000 ∧ is_multiple_of b lcm_18_24_30 ∧ 
                in_range c 2000 3000 ∧ is_multiple_of c lcm_18_24_30 ∧
                a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                ∀ z, in_range z 2000 3000 ∧ is_multiple_of z lcm_18_24_30 → z = a ∨ z = b ∨ z = c) := sorry

end number_of_multiples_in_range_l142_142554


namespace problem1_problem2_l142_142401

-- Given conditions
def A : Set ℝ := { x | x^2 - 2 * x - 15 > 0 }
def B : Set ℝ := { x | x < 6 }
def p (m : ℝ) : Prop := m ∈ A
def q (m : ℝ) : Prop := m ∈ B

-- Statements to prove
theorem problem1 (m : ℝ) : p m → m ∈ { x | x < -3 } ∪ { x | x > 5 } :=
sorry

theorem problem2 (m : ℝ) : (p m ∨ q m) ∧ (p m ∧ q m) → m ∈ { x | x < -3 } :=
sorry

end problem1_problem2_l142_142401


namespace ellipse_foci_distance_l142_142938

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∃ (a b : ℝ), (a = 6) ∧ (b = 3) ∧ distance_between_foci a b = 6 * Real.sqrt 3 :=
by
  sorry

end ellipse_foci_distance_l142_142938


namespace sandy_marks_l142_142912

def marks_each_correct_sum : ℕ := 3

theorem sandy_marks (x : ℕ) 
  (total_attempts : ℕ := 30)
  (correct_sums : ℕ := 23)
  (marks_per_incorrect_sum : ℕ := 2)
  (total_marks_obtained : ℕ := 55)
  (incorrect_sums : ℕ := total_attempts - correct_sums)
  (lost_marks : ℕ := incorrect_sums * marks_per_incorrect_sum) :
  (correct_sums * x - lost_marks = total_marks_obtained) -> x = marks_each_correct_sum :=
by
  sorry

end sandy_marks_l142_142912


namespace value_of_x_l142_142852

theorem value_of_x : 
  ∀ (x : ℕ), x = (2011^2 + 2011) / 2011 → x = 2012 :=
by
  intro x
  intro h
  sorry

end value_of_x_l142_142852


namespace div_eq_of_scaled_div_eq_l142_142490

theorem div_eq_of_scaled_div_eq (h : 29.94 / 1.45 = 17.7) : 2994 / 14.5 = 17.7 := 
by
  sorry

end div_eq_of_scaled_div_eq_l142_142490


namespace weight_of_new_student_l142_142513

-- Definitions from conditions
def total_weight_19 : ℝ := 19 * 15
def total_weight_20 : ℝ := 20 * 14.9

-- Theorem to prove the weight of the new student
theorem weight_of_new_student : (total_weight_20 - total_weight_19) = 13 := by
  sorry

end weight_of_new_student_l142_142513


namespace consecutive_odd_integers_l142_142813

theorem consecutive_odd_integers (n : ℤ) (h : (n - 2) + (n + 2) = 130) : n = 65 :=
sorry

end consecutive_odd_integers_l142_142813


namespace single_discount_equivalence_l142_142630

variable (p : ℝ) (d1 d2 d3 : ℝ)

def apply_discount (price discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_multiple_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem single_discount_equivalence :
  p = 1200 →
  d1 = 0.15 →
  d2 = 0.10 →
  d3 = 0.05 →
  let final_price_multiple := apply_multiple_discounts p [d1, d2, d3]
  let single_discount := (p - final_price_multiple) / p
  single_discount = 0.27325 :=
by
  intros h1 h2 h3 h4
  let final_price_multiple := apply_multiple_discounts p [d1, d2, d3]
  let single_discount := (p - final_price_multiple) / p
  sorry

end single_discount_equivalence_l142_142630


namespace weight_of_one_pencil_l142_142273

theorem weight_of_one_pencil (total_weight : ℝ) (num_pencils : ℕ) (H : total_weight = 141.5) (H' : num_pencils = 5) : (total_weight / num_pencils) = 28.3 :=
by sorry

end weight_of_one_pencil_l142_142273


namespace remainder_p_x_minus_2_l142_142840

def p (x : ℝ) := x^5 + 2 * x^2 + 3

theorem remainder_p_x_minus_2 : p 2 = 43 := 
by
  sorry

end remainder_p_x_minus_2_l142_142840


namespace problem_l142_142553

theorem problem (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = -2) :
  x + (x ^ 3 / y ^ 2) + (y ^ 3 / x ^ 2) + y = 440 := by
  sorry

end problem_l142_142553


namespace determine_truth_tellers_min_questions_to_determine_truth_tellers_l142_142913

variables (n k : ℕ)
variables (h_n_pos : 0 < n) (h_k_pos : 0 < k) (h_k_le_n : k ≤ n)

theorem determine_truth_tellers (h : k % 2 = 0) : 
  ∃ m : ℕ, m = n :=
  sorry

theorem min_questions_to_determine_truth_tellers :
  ∃ m : ℕ, m = n :=
  sorry

end determine_truth_tellers_min_questions_to_determine_truth_tellers_l142_142913


namespace outdoor_section_length_l142_142592

theorem outdoor_section_length (W : ℝ) (A : ℝ) (hW : W = 4) (hA : A = 24) : ∃ L : ℝ, A = W * L ∧ L = 6 := 
by
  use 6
  sorry

end outdoor_section_length_l142_142592


namespace consumer_installment_credit_l142_142988

theorem consumer_installment_credit (C : ℝ) (A : ℝ) (h1 : A = 0.36 * C) 
    (h2 : 75 = A / 2) : C = 416.67 :=
by
  sorry

end consumer_installment_credit_l142_142988


namespace f_f_f_three_l142_142681

def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 1

theorem f_f_f_three : f (f (f 3)) = 43 :=
by
  -- Introduction of definitions and further necessary steps here are skipped
  sorry

end f_f_f_three_l142_142681


namespace rebecca_marbles_l142_142548

theorem rebecca_marbles (M : ℕ) (h1 : 20 = M + 14) : M = 6 :=
by
  sorry

end rebecca_marbles_l142_142548


namespace division_neg4_by_2_l142_142769

theorem division_neg4_by_2 : (-4) / 2 = -2 := sorry

end division_neg4_by_2_l142_142769


namespace exponent_subtraction_l142_142121

theorem exponent_subtraction (a : ℝ) (m n : ℝ) (hm : a^m = 3) (hn : a^n = 5) : a^(m-n) = 3 / 5 := 
  sorry

end exponent_subtraction_l142_142121


namespace rectangle_dimension_area_l142_142370

theorem rectangle_dimension_area (x : ℝ) 
  (h_dim : (3 * x - 5) * (x + 7) = 14 * x - 35) : 
  x = 0 :=
by
  sorry

end rectangle_dimension_area_l142_142370


namespace longest_possible_height_l142_142471

theorem longest_possible_height (a b c : ℕ) (ha : a = 3 * c) (hb : b * 4 = 12 * c) (h_tri : a - c < b) (h_unequal : ¬(a = c)) :
  ∃ x : ℕ, (4 < x ∧ x < 6) ∧ x = 5 :=
by
  sorry

end longest_possible_height_l142_142471


namespace choose_amber_bronze_cells_l142_142369

theorem choose_amber_bronze_cells (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (grid : Fin (a+b+1) × Fin (a+b+1) → Prop) 
  (amber_cells : ℕ) (h_amber_cells : amber_cells ≥ a^2 + a * b - b)
  (bronze_cells : ℕ) (h_bronze_cells : bronze_cells ≥ b^2 + b * a - a):
  ∃ (amber_choice : Fin (a+b+1) → Fin (a+b+1)), 
    ∃ (bronze_choice : Fin (a+b+1) → Fin (a+b+1)), 
    amber_choice ≠ bronze_choice ∧ 
    (∀ i j, i ≠ j → grid (amber_choice i) ≠ grid (bronze_choice j)) :=
sorry

end choose_amber_bronze_cells_l142_142369


namespace sufficient_but_not_necessary_condition_l142_142955

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (0 ≤ x ∧ x ≤ 1) → |x| ≤ 1 :=
by sorry

end sufficient_but_not_necessary_condition_l142_142955


namespace find_a_l142_142602

theorem find_a (a x1 x2 : ℝ)
  (h1: 4 * x1 ^ 2 - 4 * (a + 2) * x1 + a ^ 2 + 11 = 0)
  (h2: 4 * x2 ^ 2 - 4 * (a + 2) * x2 + a ^ 2 + 11 = 0)
  (h3: x1 - x2 = 3) : a = 4 := sorry

end find_a_l142_142602


namespace cannot_lie_on_line_l142_142002

theorem cannot_lie_on_line (m b : ℝ) (h : m * b < 0) : ¬ (0 = m * (-2022) + b) := 
  by
  sorry

end cannot_lie_on_line_l142_142002


namespace christmas_distribution_l142_142958

theorem christmas_distribution :
  ∃ (n x : ℕ), 
    (240 + 120 + 1 = 361) ∧
    (n * x = 361) ∧
    (n = 19) ∧
    (x = 19) ∧
    ∃ (a b : ℕ), (a + b = 19) ∧ (a * 5 + b * 6 = 100) :=
by
  sorry

end christmas_distribution_l142_142958


namespace sum_of_solutions_comparison_l142_142062

variable (a a' b b' c c' : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0)

theorem sum_of_solutions_comparison :
  ( (c - b) / a > (c' - b') / a' ) ↔ ( (c'-b') / a' < (c-b) / a ) :=
by sorry

end sum_of_solutions_comparison_l142_142062


namespace value_of_expression_l142_142185

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 4 = 22 :=
by
  sorry

end value_of_expression_l142_142185


namespace quadratic_trinomial_with_integral_roots_l142_142971

theorem quadratic_trinomial_with_integral_roots (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (∃ x : ℤ, a * x^2 + b * x + c = 0) ∧ 
  (∃ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
  (∃ x : ℤ, (a + 2) * x^2 + (b + 2) * x + (c + 2) = 0) :=
sorry

end quadratic_trinomial_with_integral_roots_l142_142971


namespace initial_amount_spent_l142_142429

theorem initial_amount_spent
    (X : ℕ) -- initial amount of money to spend
    (sets_purchased : ℕ := 250) -- total sets purchased
    (sets_cost_20 : ℕ := 178) -- sets that cost $20 each
    (price_per_set : ℕ := 20) -- price of each set that cost $20
    (remaining_sets : ℕ := sets_purchased - sets_cost_20) -- remaining sets
    (spent_all : (X = sets_cost_20 * price_per_set + remaining_sets * 0)) -- spent all money, remaining sets assumed free to simplify as the exact price is not given or necessary
    : X = 3560 :=
    by
    sorry

end initial_amount_spent_l142_142429


namespace arithmetic_sequence_inequality_l142_142627

variable {α : Type*} [OrderedRing α]

theorem arithmetic_sequence_inequality 
  (a : ℕ → α) (d : α) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_pos : ∀ n, a n > 0)
  (h_d_ne_zero : d ≠ 0) : 
  a 0 * a 7 < a 3 * a 4 := 
by
  sorry

end arithmetic_sequence_inequality_l142_142627


namespace largest_x_plus_y_l142_142713

theorem largest_x_plus_y (x y : ℝ) (h1 : 5 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) : x + y ≤ 18 / 7 :=
by
  sorry

end largest_x_plus_y_l142_142713


namespace find_length_of_rod_l142_142223

-- Constants representing the given conditions
def weight_6m_rod : ℝ := 6.1
def length_6m_rod : ℝ := 6
def weight_unknown_rod : ℝ := 12.2

-- Proof statement ensuring the length of the rod that weighs 12.2 kg is 12 meters
theorem find_length_of_rod (L : ℝ) (h : weight_6m_rod / length_6m_rod = weight_unknown_rod / L) : 
  L = 12 := by
  sorry

end find_length_of_rod_l142_142223


namespace no_real_roots_implies_negative_l142_142262

theorem no_real_roots_implies_negative (m : ℝ) : (¬ ∃ x : ℝ, x^2 = m) → m < 0 :=
sorry

end no_real_roots_implies_negative_l142_142262


namespace Bruce_grape_purchase_l142_142255

theorem Bruce_grape_purchase
  (G : ℕ)
  (total_paid : ℕ)
  (cost_per_kg_grapes : ℕ)
  (kg_mangoes : ℕ)
  (cost_per_kg_mangoes : ℕ)
  (total_mango_cost : ℕ)
  (total_grape_cost : ℕ)
  (total_amount : ℕ)
  (h1 : cost_per_kg_grapes = 70)
  (h2 : kg_mangoes = 10)
  (h3 : cost_per_kg_mangoes = 55)
  (h4 : total_paid = 1110)
  (h5 : total_mango_cost = kg_mangoes * cost_per_kg_mangoes)
  (h6 : total_grape_cost = G * cost_per_kg_grapes)
  (h7 : total_amount = total_mango_cost + total_grape_cost)
  (h8 : total_amount = total_paid) :
  G = 8 := by
  sorry

end Bruce_grape_purchase_l142_142255


namespace area_of_inscribed_square_l142_142771

theorem area_of_inscribed_square (XY YZ : ℝ) (hXY : XY = 18) (hYZ : YZ = 30) :
  ∃ (s : ℝ), s^2 = 540 :=
by
  sorry

end area_of_inscribed_square_l142_142771


namespace quadratic_real_solutions_l142_142789

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) := 
sorry

end quadratic_real_solutions_l142_142789


namespace probability_non_edge_unit_square_l142_142752

theorem probability_non_edge_unit_square : 
  let total_squares := 100
  let perimeter_squares := 36
  let non_perimeter_squares := total_squares - perimeter_squares
  let probability := (non_perimeter_squares : ℚ) / total_squares
  probability = 16 / 25 :=
by
  sorry

end probability_non_edge_unit_square_l142_142752


namespace purple_valley_skirts_l142_142440

def AzureValley : ℕ := 60

def SeafoamValley (A : ℕ) : ℕ := (2 * A) / 3

def PurpleValley (S : ℕ) : ℕ := S / 4

theorem purple_valley_skirts :
  PurpleValley (SeafoamValley AzureValley) = 10 :=
by
  sorry

end purple_valley_skirts_l142_142440


namespace quadratic_function_passing_origin_l142_142815

theorem quadratic_function_passing_origin (a : ℝ) (h : ∃ x y, y = ax^2 + x + a * (a - 2) ∧ (x, y) = (0, 0)) : a = 2 := by
  sorry

end quadratic_function_passing_origin_l142_142815


namespace multiplier_of_reciprocal_l142_142776

theorem multiplier_of_reciprocal (x m : ℝ) (h1 : x = 7) (h2 : x - 4 = m * (1 / x)) : m = 21 :=
by
  sorry

end multiplier_of_reciprocal_l142_142776


namespace A_sub_B_value_l142_142863

def A : ℕ := 1000 * 1 + 100 * 16 + 10 * 28
def B : ℕ := 355 + 245 * 3

theorem A_sub_B_value : A - B = 1790 := by
  sorry

end A_sub_B_value_l142_142863


namespace intersection_eq_two_l142_142614

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

theorem intersection_eq_two : A ∩ B = {2} := by
  sorry

end intersection_eq_two_l142_142614


namespace number_of_littering_citations_l142_142482

variable (L D P : ℕ)
variable (h1 : L = D)
variable (h2 : P = 2 * (L + D))
variable (h3 : L + D + P = 24)

theorem number_of_littering_citations : L = 4 :=
by
  sorry

end number_of_littering_citations_l142_142482


namespace arithmetic_sequence_product_l142_142448

theorem arithmetic_sequence_product (a : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_a4a5 : a 3 * a 4 = 24) :
  a 2 * a 5 = 16 :=
sorry

end arithmetic_sequence_product_l142_142448


namespace minimum_omega_l142_142676

theorem minimum_omega (ω : ℝ) (h_pos : ω > 0) :
  (∃ k : ℤ, ω * (3 * π / 4) - ω * (π / 4) = k * π) → ω = 2 :=
by
  sorry

end minimum_omega_l142_142676


namespace initial_butterfat_percentage_l142_142565

theorem initial_butterfat_percentage (P : ℝ) :
  let initial_butterfat := (P / 100) * 1000
  let removed_butterfat := (23 / 100) * 50
  let remaining_volume := 1000 - 50
  let desired_butterfat := (3 / 100) * remaining_volume
  initial_butterfat - removed_butterfat = desired_butterfat
→ P = 4 :=
by
  intros
  let initial_butterfat := (P / 100) * 1000
  let removed_butterfat := (23 / 100) * 50
  let remaining_volume := 1000 - 50
  let desired_butterfat := (3 / 100) * remaining_volume
  sorry

end initial_butterfat_percentage_l142_142565


namespace largest_consecutive_odd_integers_sum_255_l142_142270

theorem largest_consecutive_odd_integers_sum_255 : 
  ∃ (n : ℤ), (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 255) ∧ (n + 8 = 55) :=
by
  sorry

end largest_consecutive_odd_integers_sum_255_l142_142270


namespace more_pairs_B_than_A_l142_142585

theorem more_pairs_B_than_A :
    let pairs_per_box := 20
    let boxes_A := 8
    let pairs_A := boxes_A * pairs_per_box
    let pairs_B := 5 * pairs_A
    let more_pairs := pairs_B - pairs_A
    more_pairs = 640
:= by
    sorry

end more_pairs_B_than_A_l142_142585


namespace natural_number_sum_of_coprimes_l142_142976

theorem natural_number_sum_of_coprimes (n : ℕ) (h : n ≥ 2) : ∃ a b : ℕ, n = a + b ∧ Nat.gcd a b = 1 :=
by
  use (n - 1), 1
  sorry

end natural_number_sum_of_coprimes_l142_142976


namespace circle_area_l142_142883

/--
Given the polar equation of a circle r = -4 * cos θ + 8 * sin θ,
prove that the area of the circle is 20π.
-/
theorem circle_area (θ : ℝ) (r : ℝ) (cos : ℝ → ℝ) (sin : ℝ → ℝ) 
  (h_eq : ∀ θ : ℝ, r = -4 * cos θ + 8 * sin θ) : 
  ∃ A : ℝ, A = 20 * Real.pi :=
by
  sorry

end circle_area_l142_142883


namespace maximum_value_2a_plus_b_l142_142528

variable (a b : ℝ)

theorem maximum_value_2a_plus_b (h : 4 * a^2 + b^2 + a * b = 1) : 2 * a + b ≤ 2 * Real.sqrt (10) / 5 :=
by sorry

end maximum_value_2a_plus_b_l142_142528


namespace sum_of_sequences_l142_142721

-- Definition of the problem conditions
def seq1 := [2, 12, 22, 32, 42]
def seq2 := [10, 20, 30, 40, 50]
def sum_seq1 := 2 + 12 + 22 + 32 + 42
def sum_seq2 := 10 + 20 + 30 + 40 + 50

-- Lean statement of the problem
theorem sum_of_sequences :
  sum_seq1 + sum_seq2 = 260 := by
  sorry

end sum_of_sequences_l142_142721


namespace intersection_property_l142_142566

def universal_set : Set ℝ := Set.univ

def M : Set ℝ := {-1, 1, 2, 4}

def N : Set ℝ := {x : ℝ | x > 2}

theorem intersection_property : (M ∩ N) = {4} := by
  sorry

end intersection_property_l142_142566


namespace smallest_circle_area_l142_142408

noncomputable def function_y (x : ℝ) : ℝ := 6 / x - 4 * x / 3

theorem smallest_circle_area :
  ∃ r : ℝ, (∀ x : ℝ, r * r = x^2 + (function_y x)^2) → r^2 * π = 4 * π :=
sorry

end smallest_circle_area_l142_142408


namespace inequality_order_l142_142926

theorem inequality_order (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) : 
  b > (a^4 - b^4) / (a - b) ∧ (a^4 - b^4) / (a - b) > (a + b) / 2 ∧ (a + b) / 2 > 2 * a * b :=
by 
  sorry

end inequality_order_l142_142926


namespace find_product_of_roots_l142_142494

namespace ProductRoots

variables {k m : ℝ} {x1 x2 : ℝ}

theorem find_product_of_roots (h1 : x1 ≠ x2) 
    (hx1 : 5 * x1 ^ 2 - k * x1 = m) 
    (hx2 : 5 * x2 ^ 2 - k * x2 = m) : x1 * x2 = -m / 5 :=
sorry

end ProductRoots

end find_product_of_roots_l142_142494


namespace not_p_suff_not_q_l142_142097

theorem not_p_suff_not_q (x : ℝ) :
  ¬(|x| ≥ 1) → ¬(x^2 + x - 6 ≥ 0) :=
sorry

end not_p_suff_not_q_l142_142097


namespace symmetry_y_axis_B_l142_142750

def point_A : ℝ × ℝ := (-1, 2)

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-(p.1), p.2)

theorem symmetry_y_axis_B :
  symmetric_point point_A = (1, 2) :=
by
  -- proof is omitted
  sorry

end symmetry_y_axis_B_l142_142750


namespace equal_sundays_tuesdays_l142_142624

theorem equal_sundays_tuesdays (days_in_month : ℕ) (week_days : ℕ) (extra_days : ℕ) :
  days_in_month = 30 → week_days = 7 → extra_days = 2 → 
  ∃ n, n = 3 ∧ ∀ start_day : ℕ, start_day = 3 ∨ start_day = 4 ∨ start_day = 5 :=
by sorry

end equal_sundays_tuesdays_l142_142624


namespace triangle_is_right_triangle_l142_142391

theorem triangle_is_right_triangle (a b c : ℕ) (h_ratio : a = 3 * (36 / 12)) (h_perimeter : 3 * (36 / 12) + 4 * (36 / 12) + 5 * (36 / 12) = 36) :
  a^2 + b^2 = c^2 :=
by
  -- sorry for skipping the proof.
  sorry

end triangle_is_right_triangle_l142_142391


namespace impossible_to_achieve_25_percent_grape_juice_l142_142191

theorem impossible_to_achieve_25_percent_grape_juice (x y : ℝ) 
  (h1 : ∀ a b : ℝ, (8 / (8 + 32) = 2 / 10) → (6 / (6 + 24) = 2 / 10))
  (h2 : (8 * x + 6 * y) / (40 * x + 30 * y) = 1 / 4) : false :=
by
  sorry

end impossible_to_achieve_25_percent_grape_juice_l142_142191


namespace polynomial_root_fraction_l142_142087

theorem polynomial_root_fraction (p q r s : ℝ) (h : p ≠ 0) 
    (h1 : p * 4^3 + q * 4^2 + r * 4 + s = 0)
    (h2 : p * (-3)^3 + q * (-3)^2 + r * (-3) + s = 0) : 
    (q + r) / p = -13 :=
by
  sorry

end polynomial_root_fraction_l142_142087


namespace Eunji_higher_than_Yoojung_l142_142330

-- Define floors for Yoojung and Eunji
def Yoojung_floor: ℕ := 17
def Eunji_floor: ℕ := 25

-- Assert that Eunji lives on a higher floor than Yoojung
theorem Eunji_higher_than_Yoojung : Eunji_floor > Yoojung_floor :=
  by
    sorry

end Eunji_higher_than_Yoojung_l142_142330


namespace leak_drain_time_l142_142762

noncomputable def pump_rate : ℚ := 1/2
noncomputable def leak_empty_rate : ℚ := 1 / (1 / pump_rate - 5/11)

theorem leak_drain_time :
  let pump_rate := 1/2
  let combined_rate := 5/11
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate = 22 :=
  by
    -- Definition of pump rate
    let pump_rate := 1/2
    -- Definition of combined rate
    let combined_rate := 5/11
    -- Definition of leak rate
    let leak_rate := pump_rate - combined_rate
    -- Calculate leak drain time
    show 1 / leak_rate = 22
    sorry

end leak_drain_time_l142_142762


namespace inscribed_square_area_after_cutting_l142_142945

theorem inscribed_square_area_after_cutting :
  let original_side := 5
  let cut_side := 1
  let remaining_side := original_side - 2 * cut_side
  let largest_inscribed_square_area := remaining_side ^ 2
  largest_inscribed_square_area = 9 :=
by
  let original_side := 5
  let cut_side := 1
  let remaining_side := original_side - 2 * cut_side
  let largest_inscribed_square_area := remaining_side ^ 2
  show largest_inscribed_square_area = 9
  sorry

end inscribed_square_area_after_cutting_l142_142945


namespace geometric_sequence_sum_of_first_five_l142_142948

theorem geometric_sequence_sum_of_first_five :
  (∃ (a : ℕ → ℝ) (r : ℝ),
    (∀ n, n > 0 → a n > 0) ∧
    a 2 = 2 ∧
    a 4 = 8 ∧
    r = 2 ∧
    a 1 = 1 ∧
    a 3 = a 1 * r^2 ∧
    a 5 = a 1 * r^4 ∧
    (a 1 + a 2 + a 3 + a 4 + a 5 = 31)
  ) :=
sorry

end geometric_sequence_sum_of_first_five_l142_142948


namespace ellie_needs_25ml_of_oil_l142_142425

theorem ellie_needs_25ml_of_oil 
  (oil_per_wheel : ℕ) 
  (number_of_wheels : ℕ) 
  (other_parts_oil : ℕ) 
  (total_oil : ℕ)
  (h1 : oil_per_wheel = 10)
  (h2 : number_of_wheels = 2)
  (h3 : other_parts_oil = 5)
  (h4 : total_oil = oil_per_wheel * number_of_wheels + other_parts_oil) : 
  total_oil = 25 :=
  sorry

end ellie_needs_25ml_of_oil_l142_142425


namespace trig_identity_problem_l142_142021

theorem trig_identity_problem
  (x : ℝ) (a b c : ℕ)
  (h1 : 0 < x ∧ x < (Real.pi / 2))
  (h2 : Real.sin x - Real.cos x = Real.pi / 4)
  (h3 : Real.tan x + 1 / Real.tan x = (a : ℝ) / (b - Real.pi^c)) :
  a + b + c = 50 :=
sorry

end trig_identity_problem_l142_142021


namespace value_of_x_l142_142438

theorem value_of_x (x : ℝ) (h : (10 - x)^2 = x^2 + 4) : x = 24 / 5 :=
by
  sorry

end value_of_x_l142_142438


namespace flowerbed_seeds_l142_142145

theorem flowerbed_seeds (n_fbeds n_seeds_per_fbed total_seeds : ℕ)
    (h1 : n_fbeds = 8)
    (h2 : n_seeds_per_fbed = 4) :
    total_seeds = n_fbeds * n_seeds_per_fbed := by
  sorry

end flowerbed_seeds_l142_142145


namespace find_width_of_floor_l142_142698

variable (w : ℝ) -- width of the floor

theorem find_width_of_floor (h1 : w - 4 > 0) (h2 : 10 - 4 > 0) 
                            (area_rug : (10 - 4) * (w - 4) = 24) : w = 8 :=
by
  sorry

end find_width_of_floor_l142_142698


namespace probability_defective_unit_l142_142534

theorem probability_defective_unit 
  (T : ℝ)
  (machine_a_output : ℝ := 0.4 * T)
  (machine_b_output : ℝ := 0.6 * T)
  (machine_a_defective_rate : ℝ := 9 / 1000)
  (machine_b_defective_rate : ℝ := 1 / 50)
  (total_defective_units : ℝ := (machine_a_output * machine_a_defective_rate) + (machine_b_output * machine_b_defective_rate))
  (probability_defective : ℝ := total_defective_units / T) :
  probability_defective = 0.0156 :=
by
  sorry

end probability_defective_unit_l142_142534


namespace units_digit_p_plus_5_l142_142204

theorem units_digit_p_plus_5 (p : ℕ) (h1 : p % 2 = 0) (h2 : p % 10 = 6) (h3 : (p^3 % 10) - (p^2 % 10) = 0) : (p + 5) % 10 = 1 :=
by
  sorry

end units_digit_p_plus_5_l142_142204


namespace crucian_carps_heavier_l142_142772

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

end crucian_carps_heavier_l142_142772


namespace sum_of_number_and_square_eq_132_l142_142734

theorem sum_of_number_and_square_eq_132 (x : ℝ) (h : x + x^2 = 132) : x = 11 ∨ x = -12 :=
by
  sorry

end sum_of_number_and_square_eq_132_l142_142734


namespace sum_of_cubes_l142_142779

theorem sum_of_cubes (x y : ℂ) (h1 : x + y = 1) (h2 : x * y = 1) : x^3 + y^3 = -2 := 
by 
  sorry

end sum_of_cubes_l142_142779


namespace probability_blackboard_empty_k_l142_142208

-- Define the conditions for the problem
def Ben_blackboard_empty_probability (n : ℕ) : ℚ :=
  if h : n = 2013 then (2 * (2013 / 3) + 1) / 2^(2013 / 3 * 2) else 0 / 1

-- Define the theorem that Ben's blackboard is empty after 2013 flips, and determine k
theorem probability_blackboard_empty_k :
  ∃ (u v k : ℕ), Ben_blackboard_empty_probability 2013 = (2 * u + 1) / (2^k * (2 * v + 1)) ∧ k = 1336 :=
by sorry

end probability_blackboard_empty_k_l142_142208


namespace find_value_of_M_l142_142245

theorem find_value_of_M (a b M : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = M) (h4 : ∀ x y : ℝ, x > 0 → y > 0 → (x + y = M) → x * y ≤ (M^2) / 4) (h5 : ∀ x y : ℝ, x > 0 → y > 0 → (x + y = M) → x * y = 2) :
  M = 2 * Real.sqrt 2 :=
by
  sorry

end find_value_of_M_l142_142245


namespace fraction_zero_implies_x_zero_l142_142232

theorem fraction_zero_implies_x_zero (x : ℝ) (h : x / (2 * x - 1) = 0) : x = 0 := 
by {
  sorry
}

end fraction_zero_implies_x_zero_l142_142232


namespace rectangle_dimensions_l142_142290

theorem rectangle_dimensions (w l : ℚ) (h1 : 2 * l + 2 * w = 2 * l * w) (h2 : l = 3 * w) :
  w = 4 / 3 ∧ l = 4 :=
by
  sorry

end rectangle_dimensions_l142_142290


namespace garden_width_is_correct_l142_142932

noncomputable def width_of_garden : ℝ :=
  let w := 12 -- We will define the width to be 12 as the final correct answer.
  w

theorem garden_width_is_correct (h_length : ∀ {w : ℝ}, 3 * w = 432 / w) : width_of_garden = 12 := by
  sorry

end garden_width_is_correct_l142_142932


namespace alternating_sign_max_pos_l142_142708

theorem alternating_sign_max_pos (x : ℕ → ℝ) 
  (h_nonzero : ∀ n, 1 ≤ n ∧ n ≤ 2022 → x n ≠ 0)
  (h_condition : ∀ k, 1 ≤ k ∧ k ≤ 2022 → x k + (1 / x (k + 1)) < 0)
  (h_periodic : x 2023 = x 1) :
  ∃ m, m = 1011 ∧ ( ∀ n, 1 ≤ n ∧ n ≤ 2022 → x n > 0 → n ≤ m ∧ m ≤ 2022 ) := 
sorry

end alternating_sign_max_pos_l142_142708


namespace line_equation_l142_142132

/-
Given points M(2, 3) and N(4, -5), and a line l passes through the 
point P(1, 2). Prove that the line l has equal distances from points 
M and N if and only if its equation is either 4x + y - 6 = 0 or 
3x + 2y - 7 = 0.
-/

theorem line_equation (M N P : ℝ × ℝ)
(hM : M = (2, 3))
(hN : N = (4, -5))
(hP : P = (1, 2))
(l : ℝ → ℝ → Prop)
(h_l : ∀ x y, l x y ↔ (4 * x + y - 6 = 0 ∨ 3 * x + 2 * y - 7 = 0))
: ∀ (dM dN : ℝ), 
(∀ x y , l x y → (x = 1) → (y = 2) ∧ (|M.1 - x| + |M.2 - y| = |N.1 - x| + |N.2 - y|)) :=
sorry

end line_equation_l142_142132


namespace downstream_speed_l142_142128

def Vm : ℝ := 31  -- speed in still water
def Vu : ℝ := 25  -- speed upstream
def Vs := Vm - Vu  -- speed of stream

theorem downstream_speed : Vm + Vs = 37 := 
by
  sorry

end downstream_speed_l142_142128


namespace polynomial_root_transformation_l142_142108

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem polynomial_root_transformation :
  let P (a b c d e : ℝ) (x : ℂ) := (x^6 : ℂ) + (a : ℂ) * x^5 + (b : ℂ) * x^4 + (c : ℂ) * x^3 + (d : ℂ) * x^2 + (e : ℂ) * x + 4096
  (∀ r : ℂ, P 0 0 0 0 0 r = 0 → P 0 0 0 0 0 (ω * r) = 0) →
  ∃ a b c d e : ℝ, ∃ p : ℕ, p = 3 := sorry

end polynomial_root_transformation_l142_142108


namespace inequality_solution_l142_142054

theorem inequality_solution (x : ℝ) :
  (-1 : ℝ) < (x^2 - 14*x + 11) / (x^2 - 2*x + 3) ∧
  (x^2 - 14*x + 11) / (x^2 - 2*x + 3) < (1 : ℝ) ↔
  (2/3 < x ∧ x < 1) ∨ (7 < x) :=
by
  sorry

end inequality_solution_l142_142054


namespace trivia_team_total_points_l142_142381

/-- Given the points scored by the 5 members who showed up in a trivia team game,
    prove that the total points scored by the team is 29. -/
theorem trivia_team_total_points 
  (points_first : ℕ := 5) 
  (points_second : ℕ := 9) 
  (points_third : ℕ := 7) 
  (points_fourth : ℕ := 5) 
  (points_fifth : ℕ := 3) 
  (total_points : ℕ := points_first + points_second + points_third + points_fourth + points_fifth) :
  total_points = 29 :=
by
  sorry

end trivia_team_total_points_l142_142381


namespace strawberry_pancakes_l142_142066

theorem strawberry_pancakes (total blueberry banana chocolate : ℕ) (h_total : total = 150) (h_blueberry : blueberry = 45) (h_banana : banana = 60) (h_chocolate : chocolate = 25) :
  total - (blueberry + banana + chocolate) = 20 :=
by
  sorry

end strawberry_pancakes_l142_142066


namespace fliers_sent_afternoon_fraction_l142_142562

-- Definitions of given conditions
def total_fliers : ℕ := 2000
def fliers_morning_fraction : ℚ := 1 / 10
def remaining_fliers_next_day : ℕ := 1350

-- Helper definitions based on conditions
def fliers_sent_morning := total_fliers * fliers_morning_fraction
def fliers_after_morning := total_fliers - fliers_sent_morning
def fliers_sent_afternoon := fliers_after_morning - remaining_fliers_next_day

-- Theorem stating the required proof
theorem fliers_sent_afternoon_fraction :
  fliers_sent_afternoon / fliers_after_morning = 1 / 4 :=
sorry

end fliers_sent_afternoon_fraction_l142_142562


namespace cos_alpha_condition_l142_142785

theorem cos_alpha_condition (k : ℤ) (α : ℝ) :
  (α = 2 * k * Real.pi - Real.pi / 4 -> Real.cos α = Real.sqrt 2 / 2) ∧
  (Real.cos α = Real.sqrt 2 / 2 -> ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 4 ∨ α = 2 * k * Real.pi - Real.pi / 4) :=
by
  sorry

end cos_alpha_condition_l142_142785


namespace softball_players_l142_142198

theorem softball_players (cricket hockey football total : ℕ) (h1 : cricket = 12) (h2 : hockey = 17) (h3 : football = 11) (h4 : total = 50) : 
  total - (cricket + hockey + football) = 10 :=
by
  sorry

end softball_players_l142_142198


namespace probability_of_AB_not_selected_l142_142316

-- The definition for the probability of not selecting both A and B 
def probability_not_selected : ℚ :=
  let total_ways := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial (4 - 2))
  let favorable_ways := 1 -- Only the selection of C and D
  favorable_ways / total_ways

-- The theorem stating the desired probability
theorem probability_of_AB_not_selected : probability_not_selected = 1 / 6 :=
by
  sorry

end probability_of_AB_not_selected_l142_142316


namespace length_of_AB_l142_142321

-- Define the problem variables
variables (AB CD : ℝ)
variables (h : ℝ)

-- Define the conditions
def ratio_condition (AB CD : ℝ) : Prop :=
  AB / CD = 7 / 3

def length_condition (AB CD : ℝ) : Prop :=
  AB + CD = 210

-- Lean statement combining the conditions and the final result
theorem length_of_AB (h : ℝ) (AB CD : ℝ) (h_ratio : ratio_condition AB CD) (h_length : length_condition AB CD) : 
  AB = 147 :=
by
  -- Definitions and proof would go here
  sorry

end length_of_AB_l142_142321


namespace log_cut_piece_weight_l142_142559

-- Defining the conditions

def log_length : ℕ := 20
def half_log_length : ℕ := log_length / 2
def weight_per_foot : ℕ := 150

-- The main theorem stating the problem
theorem log_cut_piece_weight : (half_log_length * weight_per_foot) = 1500 := 
by 
  sorry

end log_cut_piece_weight_l142_142559


namespace geometric_progression_ratio_l142_142129

theorem geometric_progression_ratio (r : ℕ) (h : 4 + 4 * r + 4 * r^2 + 4 * r^3 = 60) : r = 2 :=
by
  sorry

end geometric_progression_ratio_l142_142129


namespace arithmetic_sequence_prop_l142_142019

theorem arithmetic_sequence_prop (a1 d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 6 > S 7) (h2 : S 7 > S 5)
  (hSn : ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d) :
  (d < 0) ∧ (S 11 > 0) ∧ (|a1 + 5 * d| > |a1 + 6 * d|) := 
by
  sorry

end arithmetic_sequence_prop_l142_142019


namespace tailor_trim_length_l142_142648

theorem tailor_trim_length (x : ℕ) : 
  (18 - x) * 15 = 120 → x = 10 := 
by
  sorry

end tailor_trim_length_l142_142648


namespace perfect_square_sequence_l142_142526

theorem perfect_square_sequence (k : ℤ) (y : ℕ → ℤ) :
  (y 1 = 1) ∧ (y 2 = 1) ∧
  (∀ n : ℕ, y (n + 2) = (4 * k - 5) * y (n + 1) - y n + 4 - 2 * k) →
  (∀ n ≥ 1, ∃ m : ℤ, y n = m^2) ↔ (k = 1 ∨ k = 3) :=
sorry

end perfect_square_sequence_l142_142526


namespace find_roots_l142_142925

noncomputable def P (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem find_roots : {x : ℝ | P x = 0} = {-1, 1, 2} :=
by
  sorry

end find_roots_l142_142925


namespace weight_of_oil_per_ml_l142_142810

variable (w : ℝ)  -- Weight of the oil per ml
variable (total_volume : ℝ := 150)  -- Bowl volume
variable (oil_fraction : ℝ := 2/3)  -- Fraction of oil
variable (vinegar_fraction : ℝ := 1/3)  -- Fraction of vinegar
variable (vinegar_density : ℝ := 4)  -- Vinegar density in g/ml
variable (total_weight : ℝ := 700)  -- Total weight in grams

theorem weight_of_oil_per_ml :
  (total_volume * oil_fraction * w) + (total_volume * vinegar_fraction * vinegar_density) = total_weight →
  w = 5 := by
  sorry

end weight_of_oil_per_ml_l142_142810


namespace alice_favorite_number_l142_142277

def is_multiple (x y : ℕ) : Prop := ∃ k : ℕ, k * y = x
def digit_sum (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem alice_favorite_number 
  (n : ℕ) 
  (h1 : 90 ≤ n ∧ n ≤ 150) 
  (h2 : is_multiple n 13) 
  (h3 : ¬ is_multiple n 4) 
  (h4 : is_multiple (digit_sum n) 4) : 
  n = 143 := 
by 
  sorry

end alice_favorite_number_l142_142277


namespace volume_common_part_equal_quarter_volume_each_cone_l142_142990

theorem volume_common_part_equal_quarter_volume_each_cone
  (r h : ℝ) (V_cone : ℝ)
  (h_cone_volume : V_cone = (1 / 3) * π * r^2 * h) :
  ∃ V_common, V_common = (1 / 4) * V_cone :=
by
  -- Main structure of the proof skipped
  sorry

end volume_common_part_equal_quarter_volume_each_cone_l142_142990


namespace readers_of_science_fiction_l142_142606

variable (Total S L B : Nat)

theorem readers_of_science_fiction 
  (h1 : Total = 400) 
  (h2 : L = 230) 
  (h3 : B = 80) 
  (h4 : Total = S + L - B) : 
  S = 250 := 
by
  sorry

end readers_of_science_fiction_l142_142606


namespace ratio_of_efficiencies_l142_142917

-- Definitions of efficiencies
def efficiency (time : ℕ) : ℚ := 1 / time

-- Conditions:
def E_C : ℚ := efficiency 20
def E_D : ℚ := efficiency 30
def E_A : ℚ := efficiency 18
def E_B : ℚ := 1 / 36 -- Placeholder for efficiency of B to complete the statement

-- The proof goal
theorem ratio_of_efficiencies (h1 : E_A + E_B = E_C + E_D) : E_A / E_B = 2 :=
by
  -- Placeholder to structure the format, the proof will be constructed here
  sorry

end ratio_of_efficiencies_l142_142917


namespace domain_shift_l142_142569

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the domain of f
def domain_f : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

-- State the problem in Lean
theorem domain_shift (hf : ∀ x, f x ∈ domain_f) : 
    { x | 1 ≤ x ∧ x ≤ 2 } = { x | ∃ y, y ∈ domain_f ∧ x = y + 1 } :=
by
  sorry

end domain_shift_l142_142569


namespace largest_n_unique_k_l142_142881

theorem largest_n_unique_k :
  ∃ (n : ℕ), (∀ (k1 k2 : ℕ), 
    (9 / 17 < n / (n + k1) → n / (n + k1) < 8 / 15 → 9 / 17 < n / (n + k2) → n / (n + k2) < 8 / 15 → k1 = k2) ∧ 
    n = 72) :=
sorry

end largest_n_unique_k_l142_142881


namespace expression_evaluation_l142_142426

theorem expression_evaluation : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 :=
by
  sorry

end expression_evaluation_l142_142426


namespace total_time_correct_l142_142959

-- Conditions
def minutes_per_story : Nat := 7
def weeks : Nat := 20

-- Total time calculation
def total_minutes : Nat := minutes_per_story * weeks

-- Conversion to hours and minutes
def total_hours : Nat := total_minutes / 60
def remaining_minutes : Nat := total_minutes % 60

-- The proof problem
theorem total_time_correct :
  total_minutes = 140 ∧ total_hours = 2 ∧ remaining_minutes = 20 := by
  sorry

end total_time_correct_l142_142959


namespace total_value_of_coins_l142_142405

theorem total_value_of_coins (q d : ℕ) (total_value original_value swapped_value : ℚ)
  (h1 : q + d = 30)
  (h2 : total_value = 4.50)
  (h3 : original_value = 25 * q + 10 * d)
  (h4 : swapped_value = 10 * q + 25 * d)
  (h5 : swapped_value = original_value + 1.50) :
  total_value = original_value / 100 :=
sorry

end total_value_of_coins_l142_142405


namespace simplify_exponent_multiplication_l142_142107

theorem simplify_exponent_multiplication :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_multiplication_l142_142107


namespace problem_part_1_problem_part_2_l142_142978

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f (x + 1) a + g x

-- Problem Part (1)
theorem problem_part_1 (a : ℝ) (h_pos : 0 < a) :
  (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 :=
sorry

-- Problem Part (2)
theorem problem_part_2 (a : ℝ) (h_cond : ∀ x, 0 ≤ x → h x a ≥ 1) :
  a ≤ 2 :=
sorry

end problem_part_1_problem_part_2_l142_142978


namespace find_a_c_l142_142352

theorem find_a_c (a c : ℝ) (h1 : a + c = 35) (h2 : a < c)
  (h3 : ∀ x : ℝ, a * x^2 + 30 * x + c = 0 → ∃! x, a * x^2 + 30 * x + c = 0) :
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) :=
by
  sorry

end find_a_c_l142_142352


namespace cupcakes_difference_l142_142728

theorem cupcakes_difference (h : ℕ) (betty_rate : ℕ) (dora_rate : ℕ) (betty_break : ℕ) 
  (cupcakes_difference : ℕ) 
  (H₁ : betty_rate = 10) 
  (H₂ : dora_rate = 8) 
  (H₃ : betty_break = 2) 
  (H₄ : cupcakes_difference = 10) : 
  8 * h - 10 * (h - 2) = 10 → h = 5 :=
by
  intro H
  sorry

end cupcakes_difference_l142_142728


namespace max_value_of_PQ_l142_142731

noncomputable def maxDistance (P Q : ℝ × ℝ) : ℝ :=
  let dist (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)
  let O1 : ℝ × ℝ := (0, 4)
  dist P Q

theorem max_value_of_PQ:
  ∀ (P Q : ℝ × ℝ),
    (P.1 ^ 2 + (P.2 - 4) ^ 2 = 1) →
    (Q.1 ^ 2 / 9 + Q.2 ^ 2 = 1) →
    maxDistance P Q ≤ 1 + 3 * Real.sqrt 3 :=
by
  sorry

end max_value_of_PQ_l142_142731


namespace min_value_b_l142_142989

noncomputable def f (x a : ℝ) := 3 * x^2 - 4 * a * x
noncomputable def g (x a b : ℝ) := 2 * a^2 * Real.log x - b
noncomputable def f' (x a : ℝ) := 6 * x - 4 * a
noncomputable def g' (x a : ℝ) := 2 * a^2 / x

theorem min_value_b (a : ℝ) (h_a : a > 0) :
  ∃ (b : ℝ), ∃ (x₀ : ℝ), 
  (f x₀ a = g x₀ a b ∧ f' x₀ a = g' x₀ a) ∧ 
  ∀ (b' : ℝ), (∀ (x' : ℝ), (f x' a = g x' a b' ∧ f' x' a = g' x' a) → b' ≥ -1 / Real.exp 2) := 
sorry

end min_value_b_l142_142989


namespace train_length_calculation_l142_142359

def speed_km_per_hr : ℝ := 60
def time_sec : ℝ := 9
def length_of_train : ℝ := 150

theorem train_length_calculation :
  (speed_km_per_hr * 1000 / 3600) * time_sec = length_of_train := by
  sorry

end train_length_calculation_l142_142359


namespace find_cube_difference_l142_142459

theorem find_cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := 
by
  sorry

end find_cube_difference_l142_142459


namespace total_cost_of_video_games_l142_142782

theorem total_cost_of_video_games :
  let cost_football_game := 14.02
  let cost_strategy_game := 9.46
  let cost_batman_game := 12.04
  let total_cost := cost_football_game + cost_strategy_game + cost_batman_game
  total_cost = 35.52 :=
by
  -- Proof goes here
  sorry

end total_cost_of_video_games_l142_142782


namespace greatest_divisor_four_consecutive_l142_142196

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l142_142196


namespace complex_division_correct_l142_142007

theorem complex_division_correct : (3 - 1 * Complex.I) / (1 + Complex.I) = 1 - 2 * Complex.I := 
by
  sorry

end complex_division_correct_l142_142007


namespace find_inscribed_circle_area_l142_142889

noncomputable def inscribed_circle_area (length : ℝ) (breadth : ℝ) : ℝ :=
  let perimeter_rectangle := 2 * (length + breadth)
  let side_square := perimeter_rectangle / 4
  let radius_circle := side_square / 2
  Real.pi * radius_circle^2

theorem find_inscribed_circle_area :
  inscribed_circle_area 36 28 = 804.25 := by
  sorry

end find_inscribed_circle_area_l142_142889


namespace rhombus_obtuse_angle_l142_142016

theorem rhombus_obtuse_angle (perimeter height : ℝ) (h_perimeter : perimeter = 8) (h_height : height = 1) : 
  ∃ θ : ℝ, θ = 150 :=
by
  sorry

end rhombus_obtuse_angle_l142_142016


namespace intervals_of_positivity_l142_142189

theorem intervals_of_positivity :
  {x : ℝ | (x + 1) * (x - 1) * (x - 2) > 0} = {x : ℝ | (-1 < x ∧ x < 1) ∨ (2 < x)} :=
by
  sorry

end intervals_of_positivity_l142_142189


namespace arithmetic_sequence_product_l142_142576

noncomputable def b (n : ℕ) : ℤ := sorry -- define the arithmetic sequence

theorem arithmetic_sequence_product (d : ℤ) 
  (h_seq : ∀ n, b (n + 1) = b n + d)
  (h_inc : ∀ m n, m < n → b m < b n)
  (h_prod : b 4 * b 5 = 30) :
  b 3 * b 6 = -1652 ∨ b 3 * b 6 = -308 ∨ b 3 * b 6 = -68 ∨ b 3 * b 6 = 28 := 
sorry

end arithmetic_sequence_product_l142_142576


namespace max_area_of_backyard_l142_142524

theorem max_area_of_backyard (fence_length : ℕ) (h1 : fence_length = 500) 
  (l w : ℕ) (h2 : l = 2 * w) (h3 : l + 2 * w = fence_length) : 
  l * w = 31250 := 
by
  sorry

end max_area_of_backyard_l142_142524


namespace part1_part2_l142_142887

variable (α : Real)
-- Condition
axiom tan_neg_alpha : Real.tan (-α) = -2

-- Question 1
theorem part1 : ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α)) = 3 := 
by
  sorry

-- Question 2
theorem part2 : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end part1_part2_l142_142887


namespace arithmetic_sequence_problem_l142_142790

variable (a : ℕ → ℤ) -- The arithmetic sequence as a function from natural numbers to integers
variable (S : ℕ → ℤ) -- Sum of the first n terms of the sequence

-- Conditions
variable (h1 : S 8 = 4 * a 3) -- Sum of the first 8 terms is 4 times the third term
variable (h2 : a 7 = -2)      -- The seventh term is -2

-- Proven Goal
theorem arithmetic_sequence_problem : a 9 = -6 := 
by sorry -- This is a placeholder for the proof

end arithmetic_sequence_problem_l142_142790


namespace solve_for_x_l142_142258

theorem solve_for_x : (2 / 5 : ℚ) - (1 / 7) = 1 / (35 / 9) :=
by
  sorry

end solve_for_x_l142_142258


namespace extremum_range_k_l142_142515

noncomputable def f (x k : Real) : Real :=
  Real.exp x / x + k * (Real.log x - x)

/-- 
For the function f(x) = (exp(x) / x) + k * (log(x) - x), if x = 1 is the only extremum point, 
then k is in the interval (-∞, e].
-/
theorem extremum_range_k (k : Real) : 
  (∀ x : Real, (0 < x) → (f x k ≤ f 1 k)) → 
  k ≤ Real.exp 1 :=
sorry

end extremum_range_k_l142_142515


namespace low_degree_polys_condition_l142_142567

theorem low_degree_polys_condition :
  ∃ (f : Polynomial ℤ), ∃ (g : Polynomial ℤ), ∃ (h : Polynomial ℤ),
    (f = Polynomial.X ^ 3 + Polynomial.X ^ 2 + Polynomial.X + 1 ∨
          f = Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + 2 * Polynomial.X + 2 ∨
          f = 2 * Polynomial.X ^ 3 + Polynomial.X ^ 2 + 2 * Polynomial.X + 1 ∨
          f = 2 * Polynomial.X ^ 3 + 2 * Polynomial.X ^ 2 + Polynomial.X + 2) ∧
          f ^ 4 + 2 * f + 2 = (Polynomial.X ^ 4 + 2 * Polynomial.X ^ 2 + 2) * g + 3 * h := 
sorry

end low_degree_polys_condition_l142_142567


namespace sqrt_sum_of_roots_l142_142386

theorem sqrt_sum_of_roots :
  (36 + 14 * Real.sqrt 6 + 14 * Real.sqrt 5 + 6 * Real.sqrt 30).sqrt
  = (Real.sqrt 15 + Real.sqrt 10 + Real.sqrt 8 + Real.sqrt 3) :=
by
  sorry

end sqrt_sum_of_roots_l142_142386


namespace investment_doubling_time_l142_142506

theorem investment_doubling_time :
  ∀ (r : ℝ) (initial_investment future_investment : ℝ),
  r = 8 →
  initial_investment = 5000 →
  future_investment = 20000 →
  (future_investment = initial_investment * 2 ^ (70 / r * 2)) →
  70 / r * 2 = 17.5 :=
by
  intros r initial_investment future_investment h_r h_initial h_future h_double
  sorry

end investment_doubling_time_l142_142506


namespace solution_set_f_geq_3_range_of_a_for_f_geq_abs_a_minus_4_l142_142229

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- Proof Problem 1 Statement:
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≥ 1} :=
sorry

-- Proof Problem 2 Statement:
theorem range_of_a_for_f_geq_abs_a_minus_4 (a : ℝ) :
  (∃ x : ℝ, f x ≥ |a - 4|) ↔ -1 ≤ a ∧ a ≤ 9 :=
sorry

end solution_set_f_geq_3_range_of_a_for_f_geq_abs_a_minus_4_l142_142229


namespace value_of_x_plus_y_l142_142902

theorem value_of_x_plus_y (x y : ℤ) (hx : x = -3) (hy : |y| = 5) : x + y = 2 ∨ x + y = -8 := by
  sorry

end value_of_x_plus_y_l142_142902


namespace sequence_form_l142_142233

theorem sequence_form {a : ℕ → ℚ} (h_eq : ∀ n : ℕ, a n * x ^ 2 - a (n + 1) * x + 1 = 0) 
  (h_roots : ∀ α β : ℚ, 6 * α - 2 * α * β + 6 * β = 3 ) (h_a1 : a 1 = 7 / 6) :
  ∀ n : ℕ, a n = (1 / 2) ^ n + 2 / 3 :=
by
  sorry

end sequence_form_l142_142233


namespace carli_charlie_flute_ratio_l142_142770

theorem carli_charlie_flute_ratio :
  let charlie_flutes := 1
  let charlie_horns := 2
  let charlie_harps := 1
  let carli_horns := charlie_horns / 2
  let total_instruments := 7
  ∃ (carli_flutes : ℕ), 
    (charlie_flutes + charlie_horns + charlie_harps + carli_flutes + carli_horns = total_instruments) ∧ 
    (carli_flutes / charlie_flutes = 2) :=
by
  sorry

end carli_charlie_flute_ratio_l142_142770


namespace max_stories_on_odd_pages_l142_142056

theorem max_stories_on_odd_pages 
    (stories : Fin 30 -> Fin 31) 
    (h_unique : Function.Injective stories) 
    (h_bounds : ∀ i, stories i < 31)
    : ∃ n, n = 23 ∧ ∃ f : Fin n -> Fin 30, ∀ j, f j % 2 = 1 := 
sorry

end max_stories_on_odd_pages_l142_142056


namespace appetizer_cost_per_person_l142_142240

theorem appetizer_cost_per_person
    (cost_per_bag: ℕ)
    (num_bags: ℕ)
    (cost_creme_fraiche: ℕ)
    (cost_caviar: ℕ)
    (num_people: ℕ)
    (h1: cost_per_bag = 1)
    (h2: num_bags = 3)
    (h3: cost_creme_fraiche = 5)
    (h4: cost_caviar = 73)
    (h5: num_people = 3):
    (cost_per_bag * num_bags + cost_creme_fraiche + cost_caviar) / num_people = 27 := 
  by
    sorry

end appetizer_cost_per_person_l142_142240


namespace solve_for_x_l142_142682

theorem solve_for_x (x y z : ℝ) 
  (h1 : x * y + 3 * x + 2 * y = 12) 
  (h2 : y * z + 5 * y + 3 * z = 15) 
  (h3 : x * z + 5 * x + 4 * z = 40) :
  x = 4 :=
by
  sorry

end solve_for_x_l142_142682


namespace johns_original_number_l142_142952

def switch_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * units + tens

theorem johns_original_number :
  ∃ x : ℕ, (10 ≤ x ∧ x < 100) ∧ (∃ y : ℕ, y = 5 * x + 13 ∧ 82 ≤ switch_digits y ∧ switch_digits y ≤ 86 ∧ x = 11) :=
by
  sorry

end johns_original_number_l142_142952


namespace tan_subtraction_l142_142975

theorem tan_subtraction (α β : ℝ) (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_subtraction_l142_142975


namespace standard_deviation_is_2_l142_142966

noncomputable def dataset := [51, 54, 55, 57, 53]

noncomputable def mean (l : List ℝ) : ℝ :=
  ((l.sum : ℝ) / (l.length : ℝ))

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  ((l.map (λ x => (x - m)^2)).sum : ℝ) / (l.length : ℝ)

noncomputable def std_dev (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_is_2 :
  mean dataset = 54 →
  std_dev dataset = 2 := by
  intro h_mean
  sorry

end standard_deviation_is_2_l142_142966


namespace smallest_number_l142_142137

-- Definitions based on the conditions given in the problem
def satisfies_conditions (b : ℕ) : Prop :=
  b % 5 = 2 ∧ b % 4 = 3 ∧ b % 7 = 1

-- Lean proof statement
theorem smallest_number (b : ℕ) : satisfies_conditions b → b = 87 :=
sorry

end smallest_number_l142_142137


namespace exists_quadratic_sequence_l142_142767

theorem exists_quadratic_sequence (b c : ℤ) : ∃ n : ℕ, ∃ (a : ℕ → ℤ), (a 0 = b) ∧ (a n = c) ∧ ∀ i : ℕ, 1 ≤ i → i ≤ n → |a i - a (i - 1)| = i ^ 2 := 
sorry

end exists_quadratic_sequence_l142_142767


namespace scientific_notation_32000000_l142_142615

def scientific_notation (n : ℕ) : String := sorry

theorem scientific_notation_32000000 :
  scientific_notation 32000000 = "3.2 × 10^7" :=
sorry

end scientific_notation_32000000_l142_142615


namespace hall_length_width_difference_l142_142743

theorem hall_length_width_difference (L W : ℝ) 
  (h1 : W = 1 / 2 * L) 
  (h2 : L * W = 128) : 
  L - W = 8 :=
by
  sorry

end hall_length_width_difference_l142_142743


namespace alex_initial_silk_l142_142124

theorem alex_initial_silk (m_per_dress : ℕ) (m_per_friend : ℕ) (num_friends : ℕ) (num_dresses : ℕ) (initial_silk : ℕ) :
  m_per_dress = 5 ∧ m_per_friend = 20 ∧ num_friends = 5 ∧ num_dresses = 100 ∧ 
  (initial_silk - (num_friends * m_per_friend)) / m_per_dress * m_per_dress = num_dresses * m_per_dress → 
  initial_silk = 600 :=
by
  intros
  sorry

end alex_initial_silk_l142_142124


namespace rationalization_correct_l142_142748

noncomputable def rationalize_denominator : Prop :=
  (∃ (a b : ℝ), a = (12:ℝ).sqrt + (5:ℝ).sqrt ∧ b = (3:ℝ).sqrt + (5:ℝ).sqrt ∧
                (a / b) = (((15:ℝ).sqrt - 1) / 2))

theorem rationalization_correct : rationalize_denominator :=
by {
  sorry
}

end rationalization_correct_l142_142748


namespace intersecting_lines_k_value_l142_142083

theorem intersecting_lines_k_value :
  ∃ k : ℚ, (∀ x y : ℚ, y = 3 * x + 12 ∧ y = -5 * x - 7 → y = 2 * x + k) → k = 77 / 8 :=
sorry

end intersecting_lines_k_value_l142_142083


namespace difference_of_numbers_l142_142841

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 12390) (h2 : b = 2 * a + 18) : b - a = 4142 :=
by {
  sorry
}

end difference_of_numbers_l142_142841


namespace sum_of_x_y_l142_142350

theorem sum_of_x_y (x y : ℕ) (x_square_condition : ∃ x, ∃ n : ℕ, 450 * x = n^2)
                   (y_cube_condition : ∃ y, ∃ m : ℕ, 450 * y = m^3) :
                   x = 2 ∧ y = 4 → x + y = 6 := 
sorry

end sum_of_x_y_l142_142350


namespace bella_more_than_max_l142_142371

noncomputable def num_students : ℕ := 10
noncomputable def bananas_eaten_by_bella : ℕ := 7
noncomputable def bananas_eaten_by_max : ℕ := 1

theorem bella_more_than_max : 
  bananas_eaten_by_bella - bananas_eaten_by_max = 6 :=
by
  sorry

end bella_more_than_max_l142_142371


namespace probability_heads_9_or_more_12_flips_l142_142687

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l142_142687


namespace cassie_nails_l142_142488

def num_dogs : ℕ := 4
def nails_per_dog_leg : ℕ := 4
def legs_per_dog : ℕ := 4
def num_parrots : ℕ := 8
def claws_per_parrot_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def extra_claws : ℕ := 1

def total_nails_to_cut : ℕ :=
  num_dogs * nails_per_dog_leg * legs_per_dog +
  num_parrots * claws_per_parrot_leg * legs_per_parrot + extra_claws

theorem cassie_nails : total_nails_to_cut = 113 :=
  by sorry

end cassie_nails_l142_142488


namespace min_val_f_l142_142179

noncomputable def f (x : ℝ) : ℝ :=
  4 / (x - 2) + x

theorem min_val_f (x : ℝ) (h : x > 2) : ∃ y, y = f x ∧ y ≥ 6 :=
by {
  sorry
}

end min_val_f_l142_142179


namespace seeds_in_each_flower_bed_l142_142877

theorem seeds_in_each_flower_bed (total_seeds : ℕ) (flower_beds : ℕ) (h1 : total_seeds = 54) (h2 : flower_beds = 9) : total_seeds / flower_beds = 6 :=
by
  sorry

end seeds_in_each_flower_bed_l142_142877


namespace part1_part2_l142_142162

def f (a : ℝ) (x : ℝ) : ℝ := a * |x - 2| + x
def g (x : ℝ) : ℝ := |x - 2| - |2 * x - 3| + x

theorem part1 (a : ℝ) : (∀ x, f a x ≤ f a 2) ↔ a ≤ -1 :=
by sorry

theorem part2 (x : ℝ) : f 1 x < |2 * x - 3| ↔ x > 0.5 :=
by sorry

end part1_part2_l142_142162


namespace expected_number_of_different_faces_l142_142876

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l142_142876


namespace range_a_l142_142382

def A : Set ℝ :=
  {x | x^2 + 5 * x + 6 ≤ 0}

def B : Set ℝ :=
  {x | -3 ≤ x ∧ x ≤ 5}

def C (a : ℝ) : Set ℝ :=
  {x | a < x ∧ x < a + 1}

theorem range_a (a : ℝ) : ((A ∪ B) ∩ C a = ∅) → (a ≥ 5 ∨ a ≤ -4) :=
  sorry

end range_a_l142_142382


namespace closest_perfect_square_to_1042_is_1024_l142_142334

theorem closest_perfect_square_to_1042_is_1024 :
  ∀ n : ℕ, (n = 32 ∨ n = 33) → ((1042 - n^2 = 18) ↔ n = 32):=
by
  intros n hn
  cases hn
  case inl h32 => sorry
  case inr h33 => sorry

end closest_perfect_square_to_1042_is_1024_l142_142334


namespace complex_magnitude_pow_eight_l142_142453

theorem complex_magnitude_pow_eight :
  (Complex.abs ((2/5 : ℂ) + (7/5 : ℂ) * Complex.I))^8 = 7890481 / 390625 := 
by
  sorry

end complex_magnitude_pow_eight_l142_142453


namespace rug_floor_coverage_l142_142127

/-- A rectangular rug with side lengths of 2 feet and 7 feet is placed on an irregularly-shaped floor composed of a square with an area of 36 square feet and a right triangle adjacent to one of the square's sides, with leg lengths of 6 feet and 4 feet. If the surface of the rug does not extend beyond the area of the floor, then the fraction of the area of the floor that is not covered by the rug is 17/24. -/
theorem rug_floor_coverage : (48 - 14) / 48 = 17 / 24 :=
by
  -- proof goes here
  sorry

end rug_floor_coverage_l142_142127


namespace find_k_l142_142898

theorem find_k : ∃ k : ℕ, 32 / k = 4 ∧ k = 8 := 
sorry

end find_k_l142_142898


namespace business_value_l142_142824

-- Define the conditions
variable (V : ℝ) -- Total value of the business
variable (man_shares : ℝ := (2/3) * V) -- Man's share in the business
variable (sold_shares_value : ℝ := (3/4) * man_shares) -- Value of sold shares
variable (sale_price : ℝ := 45000) -- Price the shares were sold for

-- State the theorem to be proven
theorem business_value (h : (3/4) * (2/3) * V = 45000) : V = 90000 := by
  sorry

end business_value_l142_142824


namespace curve_is_circle_l142_142080

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b r : ℝ), (r > 0) ∧ ((x + a)^2 + (y + b)^2 = r^2) :=
by
  sorry

end curve_is_circle_l142_142080


namespace coeff_of_x_pow_4_in_expansion_l142_142668

theorem coeff_of_x_pow_4_in_expansion : 
  (∃ c : ℤ, c = (-1)^3 * Nat.choose 8 3 ∧ c = -56) :=
by
  sorry

end coeff_of_x_pow_4_in_expansion_l142_142668


namespace Delta_15_xDelta_eq_neg_15_l142_142920

-- Definitions of the operations based on conditions
def xDelta (x : ℝ) : ℝ := 9 - x
def Delta (x : ℝ) : ℝ := x - 9

-- Statement that we need to prove
theorem Delta_15_xDelta_eq_neg_15 : Delta (xDelta 15) = -15 :=
by
  -- The proof will go here
  sorry

end Delta_15_xDelta_eq_neg_15_l142_142920


namespace tangent_line_eqn_extreme_values_l142_142798

/-- The tangent line to the function f at (0, 5) -/
theorem tangent_line_eqn (f : ℝ → ℝ) (h : ∀ x, f x = (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 - 2 * x + 5) :
  (∃ k b, (∀ x, f x = k * x + b) ∧ k = -2 ∧ b = 5) ∧ (2 * 0 + 5 - 5 = 0) := by
  sorry

/-- The function f has a local maximum at x = -1 and a local minimum at x = 2 -/
theorem extreme_values (f : ℝ → ℝ) (h : ∀ x, f x = (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 - 2 * x + 5) :
  (∃ x₁ x₂, x₁ = -1 ∧ f x₁ = 37 / 6 ∧ x₂ = 2 ∧ f x₂ = 5 / 3) := by
  sorry

end tangent_line_eqn_extreme_values_l142_142798


namespace calculate_xy_yz_zx_l142_142074

variable (x y z : ℝ)

theorem calculate_xy_yz_zx (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h1 : x^2 + x * y + y^2 = 75)
    (h2 : y^2 + y * z + z^2 = 49)
    (h3 : z^2 + z * x + x^2 = 124) : 
    x * y + y * z + z * x = 70 :=
sorry

end calculate_xy_yz_zx_l142_142074


namespace cookies_in_fridge_l142_142642

theorem cookies_in_fridge (total_baked : ℕ) (cookies_Tim : ℕ) (cookies_Mike : ℕ) (cookies_Sarah : ℕ) (cookies_Anna : ℕ)
  (h_total_baked : total_baked = 1024)
  (h_cookies_Tim : cookies_Tim = 48)
  (h_cookies_Mike : cookies_Mike = 58)
  (h_cookies_Sarah : cookies_Sarah = 78)
  (h_cookies_Anna : cookies_Anna = (2 * (cookies_Tim + cookies_Mike)) - (cookies_Sarah / 2)) :
  total_baked - (cookies_Tim + cookies_Mike + cookies_Sarah + cookies_Anna) = 667 := by
sorry

end cookies_in_fridge_l142_142642


namespace three_times_first_number_minus_second_value_l142_142628

theorem three_times_first_number_minus_second_value (x y : ℕ) 
  (h1 : x + y = 48) 
  (h2 : y = 17) : 
  3 * x - y = 76 := 
by 
  sorry

end three_times_first_number_minus_second_value_l142_142628


namespace range_of_a_l142_142275

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : 3 * a ≥ 1) (h3 : 4 * a ≤ 3 / 2) : 
  (1 / 3) ≤ a ∧ a ≤ (3 / 8) :=
by
  sorry

end range_of_a_l142_142275


namespace cucumber_weight_evaporation_l142_142136

theorem cucumber_weight_evaporation :
  let w_99 := 50
  let p_99 := 0.99
  let evap_99 := 0.01
  let w_98 := 30
  let p_98 := 0.98
  let evap_98 := 0.02
  let w_97 := 20
  let p_97 := 0.97
  let evap_97 := 0.03

  let initial_water_99 := p_99 * w_99
  let dry_matter_99 := w_99 - initial_water_99
  let evaporated_water_99 := evap_99 * initial_water_99
  let new_weight_99 := (initial_water_99 - evaporated_water_99) + dry_matter_99

  let initial_water_98 := p_98 * w_98
  let dry_matter_98 := w_98 - initial_water_98
  let evaporated_water_98 := evap_98 * initial_water_98
  let new_weight_98 := (initial_water_98 - evaporated_water_98) + dry_matter_98

  let initial_water_97 := p_97 * w_97
  let dry_matter_97 := w_97 - initial_water_97
  let evaporated_water_97 := evap_97 * initial_water_97
  let new_weight_97 := (initial_water_97 - evaporated_water_97) + dry_matter_97

  let total_new_weight := new_weight_99 + new_weight_98 + new_weight_97
  total_new_weight = 98.335 :=
 by
  sorry

end cucumber_weight_evaporation_l142_142136


namespace solve_for_x_l142_142849

-- Define the variables and conditions based on the problem statement
def equation (x : ℚ) := 5 * x - 3 * (x + 2) = 450 - 9 * (x - 4)

-- State the theorem to be proved, including the condition and the result
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = 44.72727272727273 := by
  sorry  -- The proof is omitted

end solve_for_x_l142_142849


namespace div_add_fraction_l142_142318

theorem div_add_fraction : (3 / 7) / 4 + 2 = 59 / 28 :=
by
  sorry

end div_add_fraction_l142_142318


namespace paint_mixer_days_l142_142516

/-- Making an equal number of drums of paint each day, a paint mixer takes three days to make 18 drums of paint.
    We want to determine how many days it will take for him to make 360 drums of paint. -/
theorem paint_mixer_days (n : ℕ) (h1 : n > 0) 
  (h2 : 3 * n = 18) : 
  360 / n = 60 := by
  sorry

end paint_mixer_days_l142_142516


namespace op_dot_of_10_5_l142_142476

-- Define the operation \odot
def op_dot (a b : ℕ) : ℕ := a + (2 * a) / b

-- Theorem stating that 10 \odot 5 = 14
theorem op_dot_of_10_5 : op_dot 10 5 = 14 :=
by
  sorry

end op_dot_of_10_5_l142_142476


namespace common_difference_arithmetic_sequence_l142_142540

noncomputable def first_term : ℕ := 5
noncomputable def last_term : ℕ := 50
noncomputable def sum_terms : ℕ := 275

theorem common_difference_arithmetic_sequence :
  ∃ d n, (last_term = first_term + (n - 1) * d) ∧ (sum_terms = n * (first_term + last_term) / 2) ∧ d = 5 :=
  sorry

end common_difference_arithmetic_sequence_l142_142540


namespace abc_equality_l142_142327

theorem abc_equality (a b c : ℕ) (h1 : b = a^2 - a) (h2 : c = b^2 - b) (h3 : a = c^2 - c) : 
  a = 2 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end abc_equality_l142_142327


namespace tan_double_angle_l142_142864

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f_derivative_def (x : ℝ) : ℝ := 3 * f x

theorem tan_double_angle (x : ℝ) (h : f_derivative_def x = Real.cos x - Real.sin x) : 
  Real.tan (2 * x) = -4 / 3 :=
by
  sorry

end tan_double_angle_l142_142864


namespace even_positive_factors_count_l142_142956

theorem even_positive_factors_count (n : ℕ) (h : n = 2^4 * 3^3 * 7) : 
  ∃ k : ℕ, k = 32 := 
by
  sorry

end even_positive_factors_count_l142_142956


namespace pink_tulips_l142_142797

theorem pink_tulips (total_tulips : ℕ)
    (blue_ratio : ℚ) (red_ratio : ℚ)
    (h_total : total_tulips = 56)
    (h_blue_ratio : blue_ratio = 3/8)
    (h_red_ratio : red_ratio = 3/7) :
    ∃ pink_tulips : ℕ, pink_tulips = total_tulips - ((blue_ratio * total_tulips) + (red_ratio * total_tulips)) ∧ pink_tulips = 11 := by
  sorry

end pink_tulips_l142_142797


namespace magician_red_marbles_taken_l142_142012

theorem magician_red_marbles_taken:
  ∃ R : ℕ, (20 - R) + (30 - 4 * R) = 35 ∧ R = 3 :=
by
  sorry

end magician_red_marbles_taken_l142_142012


namespace cone_base_circumference_l142_142105

theorem cone_base_circumference
  (V : ℝ) (h : ℝ) (C : ℝ)
  (volume_eq : V = 18 * Real.pi)
  (height_eq : h = 3) :
  C = 6 * Real.sqrt 2 * Real.pi :=
sorry

end cone_base_circumference_l142_142105


namespace probability_both_blue_buttons_l142_142248

theorem probability_both_blue_buttons :
  let initial_red_C := 6
  let initial_blue_C := 12
  let initial_total_C := initial_red_C + initial_blue_C
  let remaining_fraction_C := 2 / 3
  let remaining_total_C := initial_total_C * remaining_fraction_C
  let removed_buttons := initial_total_C - remaining_total_C
  let removed_red := removed_buttons / 2
  let removed_blue := removed_buttons / 2
  let remaining_blue_C := initial_blue_C - removed_blue
  let total_remaining_C := remaining_total_C
  let probability_blue_C := remaining_blue_C / total_remaining_C
  let probability_blue_D := removed_blue / removed_buttons
  probability_blue_C * probability_blue_D = 3 / 8 :=
by
  sorry

end probability_both_blue_buttons_l142_142248


namespace arithmetic_sum_l142_142367

theorem arithmetic_sum (a₁ an n : ℕ) (h₁ : a₁ = 5) (h₂ : an = 32) (h₃ : n = 10) :
  (n * (a₁ + an)) / 2 = 185 :=
by
  sorry

end arithmetic_sum_l142_142367


namespace cars_on_river_road_l142_142332

-- Define the number of buses and cars
variables (B C : ℕ)

-- Given conditions
def ratio_condition : Prop := (B : ℚ) / C = 1 / 17
def fewer_buses_condition : Prop := B = C - 80

-- Problem statement
theorem cars_on_river_road (h_ratio : ratio_condition B C) (h_fewer : fewer_buses_condition B C) : C = 85 :=
by
  sorry

end cars_on_river_road_l142_142332


namespace point_of_tangency_l142_142552

theorem point_of_tangency : 
    ∃ (m n : ℝ), 
    (∀ x : ℝ, x ≠ 0 → n = 1 / m ∧ (-1 / m^2) = (n - 2) / (m - 0)) ∧ 
    m = 1 ∧ n = 1 :=
by
  sorry

end point_of_tangency_l142_142552


namespace range_of_m_l142_142705

variable (f : Real → Real)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom decreasing_function : ∀ x y, x < y → -1 < x ∧ y < 1 → f x > f y
axiom domain : ∀ x, -1 < x ∧ x < 1 → true

-- The statement to be proved
theorem range_of_m (m : Real) : 
  f (1 - m) + f (1 - m^2) < 0 → 0 < m → m < 1 :=
by
  sorry

end range_of_m_l142_142705


namespace arithmetic_sequence_k_l142_142683

theorem arithmetic_sequence_k :
  ∀ (a : ℕ → ℤ) (d : ℤ) (k : ℕ),
  d ≠ 0 →
  (∀ n : ℕ, a n = a 0 + n * d) →
  a 0 = 0 →
  a k = a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 →
  k = 22 :=
by
  intros a d k hdnz h_arith h_a1_zero h_ak_sum
  sorry

end arithmetic_sequence_k_l142_142683


namespace correct_option_l142_142503

theorem correct_option : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := sorry

end correct_option_l142_142503


namespace fraction_of_cookies_with_nuts_l142_142434

theorem fraction_of_cookies_with_nuts
  (nuts_per_cookie : ℤ)
  (total_cookies : ℤ)
  (total_nuts : ℤ)
  (h1 : nuts_per_cookie = 2)
  (h2 : total_cookies = 60)
  (h3 : total_nuts = 72) :
  (total_nuts / nuts_per_cookie) / total_cookies = 3 / 5 := by
  sorry

end fraction_of_cookies_with_nuts_l142_142434


namespace max_volume_of_open_top_box_l142_142694

noncomputable def box_max_volume (x : ℝ) : ℝ :=
  (10 - 2 * x) * (16 - 2 * x) * x

theorem max_volume_of_open_top_box : ∃ x : ℝ, 0 < x ∧ x < 5 ∧ box_max_volume x = 144 :=
by
  sorry

end max_volume_of_open_top_box_l142_142694


namespace crayons_count_l142_142184

def crayons_per_box : ℕ := 8
def number_of_boxes : ℕ := 10
def total_crayons : ℕ := crayons_per_box * number_of_boxes

theorem crayons_count : total_crayons = 80 := by
  sorry

end crayons_count_l142_142184


namespace geometric_series_squares_sum_l142_142076

theorem geometric_series_squares_sum (a : ℝ) (r : ℝ) (h : -1 < r ∧ r < 1) :
  (∑' n : ℕ, (a * r^n)^2) = a^2 / (1 - r^2) :=
by sorry

end geometric_series_squares_sum_l142_142076


namespace range_of_b_in_acute_triangle_l142_142714

variable {a b c : ℝ}

theorem range_of_b_in_acute_triangle (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_acute : (a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (c^2 + a^2 > b^2))
  (h_arith_seq : ∃ d : ℝ, 0 ≤ d ∧ a = b - d ∧ c = b + d)
  (h_sum_squares : a^2 + b^2 + c^2 = 21) :
  (2 * Real.sqrt 42) / 5 < b ∧ b ≤ Real.sqrt 7 :=
sorry

end range_of_b_in_acute_triangle_l142_142714


namespace arithmetic_sequence_a20_l142_142337

theorem arithmetic_sequence_a20 :
  (∀ n : ℕ, n > 0 → ∃ a : ℕ → ℕ, a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n + 2)) → 
  (∃ a : ℕ → ℕ, a 20 = 39) :=
by
  sorry

end arithmetic_sequence_a20_l142_142337


namespace cannot_fit_480_pictures_l142_142832

theorem cannot_fit_480_pictures 
  (A_capacity : ℕ) (B_capacity : ℕ) (C_capacity : ℕ) 
  (n_A : ℕ) (n_B : ℕ) (n_C : ℕ) 
  (total_pictures : ℕ) : 
  A_capacity = 12 → B_capacity = 18 → C_capacity = 24 → 
  n_A = 6 → n_B = 4 → n_C = 3 → 
  total_pictures = 480 → 
  A_capacity * n_A + B_capacity * n_B + C_capacity * n_C < total_pictures :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end cannot_fit_480_pictures_l142_142832


namespace remainder_zero_when_x_divided_by_y_l142_142251

theorem remainder_zero_when_x_divided_by_y :
  ∀ (x y : ℝ), 
    0 < x ∧ 0 < y ∧ x / y = 6.12 ∧ y = 49.99999999999996 → 
      x % y = 0 := by
  sorry

end remainder_zero_when_x_divided_by_y_l142_142251


namespace net_population_increase_per_day_l142_142092

def birth_rate : Nat := 4
def death_rate : Nat := 2
def seconds_per_day : Nat := 24 * 60 * 60

theorem net_population_increase_per_day : 
  (birth_rate - death_rate) * (seconds_per_day / 2) = 86400 := by
  sorry

end net_population_increase_per_day_l142_142092


namespace salary_of_A_l142_142155

-- Given:
-- A + B = 6000
-- A's savings = 0.05A
-- B's savings = 0.15B
-- A's savings = B's savings

theorem salary_of_A (A B : ℝ) (h1 : A + B = 6000) (h2 : 0.05 * A = 0.15 * B) :
  A = 4500 :=
sorry

end salary_of_A_l142_142155


namespace problem_1_problem_2_l142_142657

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |3 * x - 2|

theorem problem_1 {a b : ℝ} (h : ∀ x, f x ≤ 5 → -4 * a / 5 ≤ x ∧ x ≤ 3 * b / 5) : 
  a = 1 ∧ b = 2 :=
sorry

theorem problem_2 {a b m : ℝ} (h1 : a = 1) (h2 : b = 2) (h3 : ∀ x, |x - a| + |x + b| ≥ m^2 - 3 * m + 5) :
  ∃ m, m = 2 :=
sorry

end problem_1_problem_2_l142_142657


namespace angle_E_in_quadrilateral_l142_142445

theorem angle_E_in_quadrilateral (E F G H : ℝ) 
  (h1 : E = 5 * H)
  (h2 : E = 4 * G)
  (h3 : E = (5/3) * F)
  (h_sum : E + F + G + H = 360) : 
  E = 131 := by 
  sorry

end angle_E_in_quadrilateral_l142_142445


namespace expected_value_of_groups_l142_142047

noncomputable def expectedNumberOfGroups (k m : ℕ) : ℝ :=
  1 + (2 * k * m) / (k + m)

theorem expected_value_of_groups (k m : ℕ) :
  k > 0 → m > 0 → expectedNumberOfGroups k m = 1 + 2 * k * m / (k + m) :=
by
  intros
  unfold expectedNumberOfGroups
  sorry

end expected_value_of_groups_l142_142047


namespace abc_equality_l142_142671

noncomputable def abc_value (a b c : ℝ) : ℝ := (11 + Real.sqrt 117) / 2

theorem abc_equality (a b c : ℝ) (h1 : a + 1/b = 5) (h2 : b + 1/c = 2) (h3 : (c + 1/a)^2 = 4) :
  a * b * c = abc_value a b c := 
sorry

end abc_equality_l142_142671


namespace minimum_value_of_f_l142_142512

noncomputable def f (x : ℝ) : ℝ := 2 * x + (3 * x) / (x^2 + 3) + (2 * x * (x + 5)) / (x^2 + 5) + (3 * (x + 3)) / (x * (x^2 + 5))

theorem minimum_value_of_f : ∃ a : ℝ, a > 0 ∧ (∀ x > 0, f x ≥ 7) ∧ (f a = 7) :=
by
  sorry

end minimum_value_of_f_l142_142512


namespace multiply_polynomials_l142_142159

variable {x y z : ℝ}

theorem multiply_polynomials :
  (3 * x^4 - 4 * y^3 - 6 * z^2) * (9 * x^8 + 16 * y^6 + 36 * z^4 + 12 * x^4 * y^3 + 18 * x^4 * z^2 + 24 * y^3 * z^2)
  = 27 * x^12 - 64 * y^9 - 216 * z^6 - 216 * x^4 * y^3 * z^2 := by {
  sorry
}

end multiply_polynomials_l142_142159


namespace gcd_fact_plus_two_l142_142173

theorem gcd_fact_plus_two (n m : ℕ) (h1 : n = 6) (h2 : m = 8) :
  Nat.gcd (n.factorial + 2) (m.factorial + 2) = 2 :=
  sorry

end gcd_fact_plus_two_l142_142173


namespace largest_possible_d_l142_142720

theorem largest_possible_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 := 
sorry

end largest_possible_d_l142_142720


namespace geometric_sequence_third_term_l142_142095

theorem geometric_sequence_third_term (a b c d : ℕ) (r : ℕ) 
  (h₁ : d * r = 81) 
  (h₂ : 81 * r = 243) 
  (h₃ : r = 3) : c = 27 :=
by
  -- Insert proof here
  sorry

end geometric_sequence_third_term_l142_142095


namespace boys_assigned_l142_142581

theorem boys_assigned (B G : ℕ) (h1 : B + G = 18) (h2 : B = G - 2) : B = 8 :=
sorry

end boys_assigned_l142_142581


namespace set_inclusion_l142_142588

-- Definitions based on given conditions
def setA (x : ℝ) : Prop := 0 < x ∧ x < 2
def setB (x : ℝ) : Prop := x > 0

-- Statement of the proof problem
theorem set_inclusion : ∀ x, setA x → setB x :=
by
  intros x h
  sorry

end set_inclusion_l142_142588


namespace exists_a_b_l142_142575

theorem exists_a_b (r : Fin 5 → ℝ) : ∃ (i j : Fin 5), i ≠ j ∧ 0 ≤ (r i - r j) / (1 + r i * r j) ∧ (r i - r j) / (1 + r i * r j) ≤ 1 :=
by
  sorry

end exists_a_b_l142_142575


namespace bamboo_consumption_correct_l142_142349

-- Define the daily bamboo consumption for adult and baby pandas
def adult_daily_bamboo : ℕ := 138
def baby_daily_bamboo : ℕ := 50

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the total bamboo consumed by an adult panda in a week
def adult_weekly_bamboo := adult_daily_bamboo * days_in_week

-- Define the total bamboo consumed by a baby panda in a week
def baby_weekly_bamboo := baby_daily_bamboo * days_in_week

-- Define the total bamboo consumed by both pandas in a week
def total_bamboo_consumed := adult_weekly_bamboo + baby_weekly_bamboo

-- The theorem states that the total bamboo consumption in a week is 1316 pounds
theorem bamboo_consumption_correct : total_bamboo_consumed = 1316 := by
  sorry

end bamboo_consumption_correct_l142_142349


namespace speed_of_first_part_l142_142560

theorem speed_of_first_part (v : ℝ) (h1 : v > 0)
  (h_total_distance : 50 = 25 + 25)
  (h_average_speed : 44 = 50 / ((25 / v) + (25 / 33))) :
  v = 66 :=
by sorry

end speed_of_first_part_l142_142560


namespace incorrect_statement_A_l142_142291

-- Definitions for the conditions
def conditionA (x : ℝ) : Prop := -3 * x > 9
def conditionB (x : ℝ) : Prop := 2 * x - 1 < 0
def conditionC (x : ℤ) : Prop := x < 10
def conditionD (x : ℤ) : Prop := x < 2

-- Formal theorem statement
theorem incorrect_statement_A : ¬ (∀ x : ℝ, conditionA x ↔ x < -3) :=
by 
  sorry

end incorrect_statement_A_l142_142291


namespace size_of_smaller_package_l142_142362

theorem size_of_smaller_package
  (total_coffee : ℕ)
  (n_ten_ounce_packages : ℕ)
  (extra_five_ounce_packages : ℕ)
  (size_smaller_package : ℕ)
  (h1 : total_coffee = 115)
  (h2 : size_smaller_package = 5)
  (h3 : n_ten_ounce_packages = 7)
  (h4 : extra_five_ounce_packages = 2)
  (h5 : total_coffee = n_ten_ounce_packages * 10 + (n_ten_ounce_packages + extra_five_ounce_packages) * size_smaller_package) :
  size_smaller_package = 5 :=
by 
  sorry

end size_of_smaller_package_l142_142362


namespace count_integers_l142_142122

def Q (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 9) * (x - 16) * (x - 25) * (x - 36) * (x - 49) * (x - 64) * (x - 81)

theorem count_integers (Q_le_0 : ∀ n : ℤ, Q n ≤ 0 → ∃ k : ℕ, k = 53) : ∃ k : ℕ, k = 53 := by
  sorry

end count_integers_l142_142122


namespace total_people_surveyed_l142_142030

theorem total_people_surveyed (x y : ℝ) (h1 : 0.536 * x = 30) (h2 : 0.794 * y = x) : y = 71 :=
by
  sorry

end total_people_surveyed_l142_142030


namespace percentage_boys_not_attended_college_l142_142272

/-
Define the constants and given conditions.
-/
def number_of_boys : ℕ := 300
def number_of_girls : ℕ := 240
def total_students : ℕ := number_of_boys + number_of_girls
def percentage_class_attended_college : ℝ := 0.70
def percentage_girls_not_attended_college : ℝ := 0.30

/-
The proof problem statement: 
Prove the percentage of the boys class that did not attend college.
-/
theorem percentage_boys_not_attended_college :
  let students_attended_college := percentage_class_attended_college * total_students
  let not_attended_college_students := total_students - students_attended_college
  let not_attended_college_girls := percentage_girls_not_attended_college * number_of_girls
  let not_attended_college_boys := not_attended_college_students - not_attended_college_girls
  let percentage_boys_not_attended_college := (not_attended_college_boys / number_of_boys) * 100
  percentage_boys_not_attended_college = 30 := by
  sorry

end percentage_boys_not_attended_college_l142_142272


namespace hypotenuse_of_45_45_90_triangle_l142_142644

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end hypotenuse_of_45_45_90_triangle_l142_142644


namespace find_three_digit_number_l142_142266

theorem find_three_digit_number : 
  ∃ x : ℕ, (x >= 100 ∧ x < 1000) ∧ (2 * x = 3 * x - 108) :=
by
  have h : ∀ x : ℕ, 100 ≤ x → x < 1000 → 2 * x = 3 * x - 108 → x = 108 := sorry
  exact ⟨108, by sorry⟩

end find_three_digit_number_l142_142266


namespace circumscribed_sphere_radius_l142_142659

theorem circumscribed_sphere_radius (a b R : ℝ) (ha : a > 0) (hb : b > 0) :
  R = b^2 / (2 * (Real.sqrt (b^2 - a^2))) :=
sorry

end circumscribed_sphere_radius_l142_142659


namespace percentage_of_female_students_l142_142203

theorem percentage_of_female_students {F : ℝ} (h1 : 200 > 0): ((200 * (F / 100)) * 0.5 * 0.5 = 30) → (F = 60) :=
by
  sorry

end percentage_of_female_students_l142_142203


namespace annual_rent_per_square_foot_l142_142134

theorem annual_rent_per_square_foot 
  (monthly_rent : ℕ) 
  (length : ℕ) 
  (width : ℕ) 
  (area : ℕ)
  (annual_rent : ℕ) : 
  monthly_rent = 3600 → 
  length = 18 → 
  width = 20 → 
  area = length * width → 
  annual_rent = monthly_rent * 12 → 
  annual_rent / area = 120 :=
by
  sorry

end annual_rent_per_square_foot_l142_142134


namespace probability_at_least_5_heads_l142_142319

def fair_coin_probability_at_least_5_heads : ℚ :=
  (Nat.choose 7 5 + Nat.choose 7 6 + Nat.choose 7 7) / 2^7

theorem probability_at_least_5_heads :
  fair_coin_probability_at_least_5_heads = 29 / 128 := 
  by
    sorry

end probability_at_least_5_heads_l142_142319


namespace probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l142_142221

noncomputable def total_outcomes : ℕ := Nat.choose 6 2

noncomputable def prob_both_boys : ℚ := (Nat.choose 4 2 : ℚ) / total_outcomes

noncomputable def prob_exactly_one_girl : ℚ := ((Nat.choose 4 1) * (Nat.choose 2 1) : ℚ) / total_outcomes

noncomputable def prob_at_least_one_girl : ℚ := 1 - prob_both_boys

theorem probability_both_boys : prob_both_boys = 2 / 5 := by sorry
theorem probability_exactly_one_girl : prob_exactly_one_girl = 8 / 15 := by sorry
theorem probability_at_least_one_girl : prob_at_least_one_girl = 3 / 5 := by sorry

end probability_both_boys_probability_exactly_one_girl_probability_at_least_one_girl_l142_142221


namespace markup_is_correct_l142_142496

def purchase_price : ℝ := 48
def overhead_percent : ℝ := 0.25
def net_profit : ℝ := 12

def overhead_cost := overhead_percent * purchase_price
def total_cost := purchase_price + overhead_cost
def selling_price := total_cost + net_profit
def markup := selling_price - purchase_price

theorem markup_is_correct : markup = 24 := by sorry

end markup_is_correct_l142_142496


namespace Olivia_pays_4_dollars_l142_142868

-- Definitions based on the conditions
def quarters_chips : ℕ := 4
def quarters_soda : ℕ := 12
def conversion_rate : ℕ := 4

-- Prove that the total dollars Olivia pays is 4
theorem Olivia_pays_4_dollars (h1 : quarters_chips = 4) (h2 : quarters_soda = 12) (h3 : conversion_rate = 4) : 
  (quarters_chips + quarters_soda) / conversion_rate = 4 :=
by
  -- skipping the proof
  sorry

end Olivia_pays_4_dollars_l142_142868


namespace buy_tshirts_l142_142761

theorem buy_tshirts
  (P T : ℕ)
  (h1 : 3 * P + 6 * T = 1500)
  (h2 : P + 12 * T = 1500)
  (budget : ℕ)
  (budget_eq : budget = 800) :
  (budget / T) = 8 := by
  sorry

end buy_tshirts_l142_142761


namespace gwen_money_remaining_l142_142356

def gwen_money (initial : ℝ) (spent1 : ℝ) (earned : ℝ) (spent2 : ℝ) : ℝ :=
  initial - spent1 + earned - spent2

theorem gwen_money_remaining :
  gwen_money 5 3.25 1.5 0.7 = 2.55 :=
by
  sorry

end gwen_money_remaining_l142_142356


namespace surface_area_of_box_l142_142049

def cube_edge_length : ℕ := 1
def cubes_required : ℕ := 12

theorem surface_area_of_box (l w h : ℕ) (h1 : l * w * h = cubes_required / cube_edge_length ^ 3) :
  (2 * (l * w + w * h + h * l) = 32 ∨ 2 * (l * w + w * h + h * l) = 38 ∨ 2 * (l * w + w * h + h * l) = 40) :=
  sorry

end surface_area_of_box_l142_142049


namespace SavingsInequality_l142_142691

theorem SavingsInequality (n : ℕ) : 52 + 15 * n > 70 + 12 * n := 
by sorry

end SavingsInequality_l142_142691


namespace sum_of_first_five_multiples_of_15_l142_142574

theorem sum_of_first_five_multiples_of_15 : (15 + 30 + 45 + 60 + 75) = 225 :=
by sorry

end sum_of_first_five_multiples_of_15_l142_142574


namespace price_of_adult_ticket_l142_142413

theorem price_of_adult_ticket
  (price_child : ℤ)
  (price_adult : ℤ)
  (num_adults : ℤ)
  (num_children : ℤ)
  (total_amount : ℤ)
  (h1 : price_adult = 2 * price_child)
  (h2 : num_adults = 400)
  (h3 : num_children = 200)
  (h4 : total_amount = 16000) :
  num_adults * price_adult + num_children * price_child = total_amount → price_adult = 32 := by
    sorry

end price_of_adult_ticket_l142_142413


namespace total_time_is_11_l142_142942

-- Define the times each person spent in the pool
def Jerry_time : Nat := 3
def Elaine_time : Nat := 2 * Jerry_time
def George_time : Nat := Elaine_time / 3
def Kramer_time : Nat := 0

-- Define the total time spent in the pool by all friends
def total_time : Nat := Jerry_time + Elaine_time + George_time + Kramer_time

-- Prove that the total time is 11 minutes
theorem total_time_is_11 : total_time = 11 := sorry

end total_time_is_11_l142_142942


namespace single_shot_decrease_l142_142075

theorem single_shot_decrease (S : ℝ) (r1 r2 r3 : ℝ) (h1 : r1 = 0.05) (h2 : r2 = 0.10) (h3 : r3 = 0.15) :
  (1 - (1 - r1) * (1 - r2) * (1 - r3)) * 100 = 27.325 := 
by
  sorry

end single_shot_decrease_l142_142075


namespace matrix_system_solution_range_l142_142329

theorem matrix_system_solution_range (m : ℝ) :
  (∃ x y: ℝ, 
    (m * x + y = m + 1) ∧ 
    (x + m * y = 2 * m)) ↔ m ≠ -1 :=
by
  sorry

end matrix_system_solution_range_l142_142329


namespace range_of_x_l142_142460

theorem range_of_x (a b c x : ℝ) (h1 : a^2 + 2 * b^2 + 3 * c^2 = 6) (h2 : a + 2 * b + 3 * c > |x + 1|) : -7 < x ∧ x < 5 :=
by
  sorry

end range_of_x_l142_142460


namespace drying_time_correct_l142_142390

theorem drying_time_correct :
  let short_haired_dog_drying_time := 10
  let full_haired_dog_drying_time := 2 * short_haired_dog_drying_time
  let num_short_haired_dogs := 6
  let num_full_haired_dogs := 9
  let total_short_haired_dogs_time := num_short_haired_dogs * short_haired_dog_drying_time
  let total_full_haired_dogs_time := num_full_haired_dogs * full_haired_dog_drying_time
  let total_drying_time_in_minutes := total_short_haired_dogs_time + total_full_haired_dogs_time
  let total_drying_time_in_hours := total_drying_time_in_minutes / 60
  total_drying_time_in_hours = 4 := 
by
  sorry

end drying_time_correct_l142_142390


namespace min_value_expression_l142_142069

theorem min_value_expression (x : ℝ) (h : x ≠ -7) : 
  ∃ y, y = 1 ∧ ∀ z, z = (2 * x ^ 2 + 98) / ((x + 7) ^ 2) → y ≤ z := 
sorry

end min_value_expression_l142_142069


namespace compare_fractions_l142_142626

theorem compare_fractions : (6/29 : ℚ) < (8/25 : ℚ) ∧ (8/25 : ℚ) < (11/31 : ℚ):=
by
  have h1 : (6/29 : ℚ) < (8/25 : ℚ) := sorry
  have h2 : (8/25 : ℚ) < (11/31 : ℚ) := sorry
  exact ⟨h1, h2⟩

end compare_fractions_l142_142626


namespace perimeter_eq_120_plus_2_sqrt_1298_l142_142086

noncomputable def total_perimeter_of_two_quadrilaterals (AB BC CD : ℝ) (AC : ℝ := Real.sqrt (AB ^ 2 + BC ^ 2)) (AD : ℝ := Real.sqrt (AC ^ 2 + CD ^ 2)) : ℝ :=
2 * (AB + BC + CD + AD)

theorem perimeter_eq_120_plus_2_sqrt_1298 (hAB : AB = 15) (hBC : BC = 28) (hCD : CD = 17) :
  total_perimeter_of_two_quadrilaterals 15 28 17 = 120 + 2 * Real.sqrt 1298 :=
by
  sorry

end perimeter_eq_120_plus_2_sqrt_1298_l142_142086


namespace centroid_of_triangle_PQR_positions_l142_142433

-- Define the basic setup
def square_side_length : ℕ := 12
def total_points : ℕ := 48

-- Define the centroid calculation condition
def centroid_positions_count : ℕ :=
  let side_segments := square_side_length
  let points_per_edge := total_points / 4
  let possible_positions_per_side := points_per_edge - 1
  (possible_positions_per_side * possible_positions_per_side)

/-- Proof statement: Proving the number of possible positions for the centroid of triangle PQR 
    formed by any three non-collinear points out of the 48 points on the perimeter of the square. --/
theorem centroid_of_triangle_PQR_positions : centroid_positions_count = 121 := 
  sorry

end centroid_of_triangle_PQR_positions_l142_142433


namespace find_the_added_number_l142_142025

theorem find_the_added_number (n : ℤ) : (1 + n) / (3 + n) = 3 / 4 → n = 5 :=
  sorry

end find_the_added_number_l142_142025


namespace solution_set_of_inequality_l142_142102

theorem solution_set_of_inequality {x : ℝ} : 
  (|2 * x - 1| - |x - 2| < 0) → (-1 < x ∧ x < 1) :=
by
  sorry

end solution_set_of_inequality_l142_142102


namespace sufficient_but_not_necessary_condition_l142_142778

variable (a b x y : ℝ)

theorem sufficient_but_not_necessary_condition (ha : a > 0) (hb : b > 0) :
  ((x > a ∧ y > b) → (x + y > a + b ∧ x * y > a * b)) ∧
  ¬((x + y > a + b ∧ x * y > a * b) → (x > a ∧ y > b)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l142_142778


namespace parallel_lines_l142_142005

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, (ax + 2 * y + a = 0 ∧ 3 * a * x + (a - 1) * y + 7 = 0) →
    - (a / 2) = - (3 * a / (a - 1))) ↔ (a = 0 ∨ a = 7) :=
by
  sorry

end parallel_lines_l142_142005


namespace total_worth_of_stock_l142_142042

theorem total_worth_of_stock :
  let cost_expensive := 10
  let cost_cheaper := 3.5
  let total_modules := 11
  let cheaper_modules := 10
  let expensive_modules := total_modules - cheaper_modules
  let worth_cheaper_modules := cheaper_modules * cost_cheaper
  let worth_expensive_module := expensive_modules * cost_expensive 
  worth_cheaper_modules + worth_expensive_module = 45 := by
  sorry

end total_worth_of_stock_l142_142042


namespace valid_number_count_is_300_l142_142823

-- Define the set of digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6]

-- Define the set of odd digits
def odd_digits : List ℕ := [1, 3, 5]

-- Define a function to count valid four-digit numbers
noncomputable def count_valid_numbers : ℕ :=
  (odd_digits.length * (digits.length - 2) * (digits.length - 2) * (digits.length - 3))

-- State the theorem
theorem valid_number_count_is_300 : count_valid_numbers = 300 :=
  sorry

end valid_number_count_is_300_l142_142823


namespace resulting_polygon_has_30_sides_l142_142458

def polygon_sides : ℕ := 3 + 4 + 5 + 6 + 7 + 8 + 9 - 6 * 2

theorem resulting_polygon_has_30_sides : polygon_sides = 30 := by
  sorry

end resulting_polygon_has_30_sides_l142_142458


namespace sets_of_laces_needed_l142_142288

-- Define the conditions as constants
def teams := 4
def members_per_team := 10
def pairs_per_member := 2
def skates_per_pair := 2
def sets_of_laces_per_skate := 3

-- Formulate and state the theorem to be proven
theorem sets_of_laces_needed : 
  sets_of_laces_per_skate * (teams * members_per_team * (pairs_per_member * skates_per_pair)) = 480 :=
by sorry

end sets_of_laces_needed_l142_142288


namespace total_beats_together_in_week_l142_142374

theorem total_beats_together_in_week :
  let samantha_beats_per_min := 250
  let samantha_hours_per_day := 3
  let michael_beats_per_min := 180
  let michael_hours_per_day := 2.5
  let days_per_week := 5

  let samantha_beats_per_day := samantha_beats_per_min * 60 * samantha_hours_per_day
  let samantha_beats_per_week := samantha_beats_per_day * days_per_week
  let michael_beats_per_day := michael_beats_per_min * 60 * michael_hours_per_day
  let michael_beats_per_week := michael_beats_per_day * days_per_week
  let total_beats_per_week := samantha_beats_per_week + michael_beats_per_week

  total_beats_per_week = 360000 := 
by
  -- The proof will go here
  sorry

end total_beats_together_in_week_l142_142374


namespace trips_needed_to_fill_pool_l142_142000

def caleb_gallons_per_trip : ℕ := 7
def cynthia_gallons_per_trip : ℕ := 8
def pool_capacity : ℕ := 105

theorem trips_needed_to_fill_pool : (pool_capacity / (caleb_gallons_per_trip + cynthia_gallons_per_trip) = 7) :=
by
  sorry

end trips_needed_to_fill_pool_l142_142000


namespace number_of_people_l142_142450

-- Conditions
def cost_oysters : ℤ := 3 * 15
def cost_shrimp : ℤ := 2 * 14
def cost_clams : ℤ := 2 * 135 / 10  -- Using integers for better precision
def total_cost : ℤ := cost_oysters + cost_shrimp + cost_clams
def amount_owed_each_person : ℤ := 25

-- Goal
theorem number_of_people (number_of_people : ℤ) : total_cost = number_of_people * amount_owed_each_person → number_of_people = 4 := by
  -- Proof to be completed here.
  sorry

end number_of_people_l142_142450


namespace area_ratio_triangle_MNO_XYZ_l142_142256

noncomputable def triangle_area_ratio (XY YZ XZ p q r : ℝ) : ℝ := sorry

theorem area_ratio_triangle_MNO_XYZ : 
  ∀ (p q r: ℝ),
  p > 0 → q > 0 → r > 0 →
  p + q + r = 3 / 4 →
  p ^ 2 + q ^ 2 + r ^ 2 = 1 / 2 →
  triangle_area_ratio 12 16 20 p q r = 9 / 32 :=
sorry

end area_ratio_triangle_MNO_XYZ_l142_142256


namespace find_t_value_l142_142052

theorem find_t_value (t : ℝ) (h1 : (t - 6) * (2 * t - 5) = (2 * t - 8) * (t - 5)) : t = 10 :=
sorry

end find_t_value_l142_142052


namespace election_result_l142_142176

theorem election_result (total_votes : ℕ) (invalid_vote_percentage valid_vote_percentage : ℚ) 
  (candidate_A_percentage : ℚ) (hv: valid_vote_percentage = 1 - invalid_vote_percentage) 
  (ht: total_votes = 560000) 
  (hi: invalid_vote_percentage = 0.15) 
  (hc: candidate_A_percentage = 0.80) : 
  (candidate_A_percentage * valid_vote_percentage * total_votes = 380800) :=
by 
  sorry

end election_result_l142_142176


namespace fred_earnings_l142_142388
noncomputable def start := 111
noncomputable def now := 115
noncomputable def earnings := now - start

theorem fred_earnings : earnings = 4 :=
by
  sorry

end fred_earnings_l142_142388


namespace intersection_points_l142_142712

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := -f x
noncomputable def h (x : ℝ) : ℝ := f (-x)

theorem intersection_points :
  let a := 2
  let b := 1
  10 * a + b = 21 :=
by
  sorry

end intersection_points_l142_142712


namespace percent_calculation_l142_142673

theorem percent_calculation (y : ℝ) : (0.3 * 0.7 * y - 0.1 * y) = 0.11 * y ∧ (0.11 * y / y * 100 = 11) := by
  sorry

end percent_calculation_l142_142673


namespace factorization_sum_l142_142022

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, x^2 + 9 * x + 20 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) :
  a + b + c = 21 :=
by
  sorry

end factorization_sum_l142_142022


namespace height_at_10inches_l142_142935

theorem height_at_10inches 
  (a : ℚ)
  (h : 20 = (- (4 / 125) * 25 ^ 2 + 20))
  (span_eq : 50 = 50)
  (height_eq : 20 = 20)
  (y_eq : ∀ x : ℚ, - (4 / 125) * x ^ 2 + 20 = 16.8) :
  (- (4 / 125) * 10 ^ 2 + 20) = 16.8 :=
by
  sorry

end height_at_10inches_l142_142935


namespace convince_jury_l142_142276

-- Define predicates for being a criminal, normal man, guilty, or a knight
def Criminal : Prop := sorry
def NormalMan : Prop := sorry
def Guilty : Prop := sorry
def Knight : Prop := sorry

-- Define your status
variable (you : Prop)

-- Assumptions as per given conditions
axiom criminal_not_normal_man : Criminal → ¬NormalMan
axiom you_not_guilty : ¬Guilty
axiom you_not_knight : ¬Knight

-- The statement to prove
theorem convince_jury : ¬Guilty ∧ ¬Knight := by
  exact And.intro you_not_guilty you_not_knight

end convince_jury_l142_142276


namespace amy_required_hours_per_week_l142_142859

variable (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_pay : ℕ) 
variable (pay_raise_percent : ℕ) (school_year_weeks : ℕ) (required_school_year_pay : ℕ)

def summer_hours_total := summer_hours_per_week * summer_weeks
def summer_hourly_pay := summer_pay / summer_hours_total
def new_hourly_pay := summer_hourly_pay + (summer_hourly_pay / 10)  -- 10% pay raise
def total_needed_hours := required_school_year_pay / new_hourly_pay
def required_hours_per_week := total_needed_hours / school_year_weeks

theorem amy_required_hours_per_week :
  summer_hours_per_week = 40 →
  summer_weeks = 12 →
  summer_pay = 4800 →
  pay_raise_percent = 10 →
  school_year_weeks = 36 →
  required_school_year_pay = 7200 →
  required_hours_per_week = 18 := sorry

end amy_required_hours_per_week_l142_142859


namespace average_of_remaining_two_numbers_l142_142001

theorem average_of_remaining_two_numbers (S a₁ a₂ a₃ a₄ : ℝ)
    (h₁ : S / 6 = 3.95)
    (h₂ : (a₁ + a₂) / 2 = 3.8)
    (h₃ : (a₃ + a₄) / 2 = 3.85) :
    (S - (a₁ + a₂ + a₃ + a₄)) / 2 = 4.2 := 
sorry

end average_of_remaining_two_numbers_l142_142001


namespace correct_population_statement_l142_142891

def correct_statement :=
  "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population."

def sample_size : ℕ := 500

def is_correct (statement : String) : Prop :=
  statement = correct_statement

theorem correct_population_statement (scores : Fin 500 → ℝ) :
  is_correct "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population." :=
by
  sorry

end correct_population_statement_l142_142891


namespace problem_solution_l142_142729

def expr := 1 + 1 / (1 + 1 / (1 + 1))
def answer : ℚ := 5 / 3

theorem problem_solution : expr = answer :=
by
  sorry

end problem_solution_l142_142729


namespace ratio_of_cereal_boxes_l142_142933

variable (F : ℕ) (S : ℕ) (T : ℕ) (k : ℚ)

def boxes_cereal : Prop :=
  F = 14 ∧
  F + S + T = 33 ∧
  S = k * (F : ℚ) ∧
  S = T - 5 → 
  S / F = 1 / 2

theorem ratio_of_cereal_boxes (F S T : ℕ) (k : ℚ) : 
  boxes_cereal F S T k :=
by
  sorry

end ratio_of_cereal_boxes_l142_142933


namespace find_rate_l142_142282

def simple_interest_rate (P A T : ℕ) : ℕ :=
  ((A - P) * 100) / (P * T)

theorem find_rate :
  simple_interest_rate 750 1200 5 = 12 :=
by
  -- This is the statement of equality we need to prove
  sorry

end find_rate_l142_142282


namespace sequence_a100_l142_142164

theorem sequence_a100 :
  ∃ a : ℕ → ℕ, (a 1 = 1) ∧ (∀ m n : ℕ, 0 < m → 0 < n → a (n + m) = a n + a m + n * m) ∧ (a 100 = 5050) :=
by
  sorry

end sequence_a100_l142_142164


namespace minimum_perimeter_l142_142320

noncomputable def minimum_perimeter_triangle (l m n : ℕ) : ℕ :=
  l + m + n

theorem minimum_perimeter :
  ∀ (l m n : ℕ),
    (l > m) → (m > n) → 
    ((∃ k : ℕ, 10^4 ∣ 3^l - 3^m + k * 10^4) ∧ (∃ k : ℕ, 10^4 ∣ 3^m - 3^n + k * 10^4) ∧ (∃ k : ℕ, 10^4 ∣ 3^l - 3^n + k * 10^4)) →
    minimum_perimeter_triangle l m n = 3003 :=
by
  intros l m n hlm hmn hmod
  sorry

end minimum_perimeter_l142_142320


namespace polynomial_inequality_l142_142960

theorem polynomial_inequality (f : ℝ → ℝ) (h1 : f 0 = 1)
    (h2 : ∀ (x y : ℝ), f (x - y) + f x ≥ 2 * x^2 - 2 * x * y + y^2 + 2 * x - y + 2) :
    f = λ x => x^2 + x + 1 := by
  sorry

end polynomial_inequality_l142_142960


namespace service_center_location_l142_142313

def serviceCenterMilepost (x3 x10 : ℕ) (r : ℚ) : ℚ :=
  x3 + r * (x10 - x3)

theorem service_center_location :
  (serviceCenterMilepost 50 170 (2/3) : ℚ) = 130 :=
by
  -- placeholder for the actual proof
  sorry

end service_center_location_l142_142313


namespace division_identity_l142_142478

theorem division_identity
  (x y : ℕ)
  (h1 : x = 7)
  (h2 : y = 2)
  : (x^3 + y^3) / (x^2 - x * y + y^2) = 9 :=
by
  sorry

end division_identity_l142_142478


namespace rectangle_length_reduction_l142_142867

theorem rectangle_length_reduction (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_length := L * (1 - 10 / 100)
  let new_width := W * (10 / 9)
  (new_length * new_width = L * W) → 
  x = 10 := by sorry

end rectangle_length_reduction_l142_142867


namespace areas_of_triangles_l142_142539

-- Define the condition that the gcd of a, b, and c is 1
def gcd_one (a b c : ℤ) : Prop := Int.gcd (Int.gcd a b) c = 1

-- Define the set of possible areas for triangles in E
def f_E : Set ℝ :=
  { area | ∃ (a b c : ℤ), gcd_one a b c ∧ area = (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) }

theorem areas_of_triangles : 
  f_E = { area | ∃ (a b c : ℤ), gcd_one a b c ∧ area = (1 / 2) * Real.sqrt (a^2 + b^2 + c^2) } :=
by {
  sorry
}

end areas_of_triangles_l142_142539


namespace cars_meet_time_l142_142742

theorem cars_meet_time (t : ℝ) (highway_length : ℝ) (speed_car1 : ℝ) (speed_car2 : ℝ)
  (h1 : highway_length = 105) (h2 : speed_car1 = 15) (h3 : speed_car2 = 20) :
  15 * t + 20 * t = 105 → t = 3 := by
  sorry

end cars_meet_time_l142_142742


namespace find_n_from_equation_l142_142473

theorem find_n_from_equation :
  ∃ n : ℕ, (3 * 3 * 5 * 5 * 7 * 9 = 3 * 3 * 7 * n * n) → n = 15 :=
by
  sorry

end find_n_from_equation_l142_142473


namespace set_D_is_empty_l142_142222

theorem set_D_is_empty :
  {x : ℝ | x^2 + 2 = 0} = ∅ :=
by {
  sorry
}

end set_D_is_empty_l142_142222


namespace calculate_g3_l142_142495

def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

theorem calculate_g3 : g 3 = 3 / 17 :=
by {
    -- Here we add the proof steps if necessary, but for now we use sorry
    sorry
}

end calculate_g3_l142_142495


namespace distance_from_point_to_x_axis_l142_142409

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

theorem distance_from_point_to_x_axis :
  let p := (-2, -Real.sqrt 5)
  distance_to_x_axis p = Real.sqrt 5 := by
  sorry

end distance_from_point_to_x_axis_l142_142409


namespace find_divisor_l142_142265

theorem find_divisor (x y : ℕ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 14) / y = 4) : y = 10 :=
sorry

end find_divisor_l142_142265


namespace determine_values_a_b_l142_142004

theorem determine_values_a_b (a b x : ℝ) (h₁ : x > 1)
  (h₂ : 3 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = (10 * (Real.log x)^2) / (Real.log a + Real.log b)) :
  b = a ^ ((5 + Real.sqrt 10) / 3) ∨ b = a ^ ((5 - Real.sqrt 10) / 3) :=
by sorry

end determine_values_a_b_l142_142004


namespace minimum_balls_same_color_minimum_balls_two_white_l142_142366

-- Define the number of black and white balls.
def num_black_balls : Nat := 100
def num_white_balls : Nat := 100

-- Problem 1: Ensure at least 2 balls of the same color.
theorem minimum_balls_same_color (n_black n_white : Nat) (h_black : n_black = num_black_balls) (h_white : n_white = num_white_balls) : 
  3 ≥ 2 :=
by
  sorry

-- Problem 2: Ensure at least 2 white balls.
theorem minimum_balls_two_white (n_black n_white : Nat) (h_black: n_black = num_black_balls) (h_white: n_white = num_white_balls) :
  102 ≥ 2 :=
by
  sorry

end minimum_balls_same_color_minimum_balls_two_white_l142_142366


namespace calculate_average_fish_caught_l142_142451

-- Definitions based on conditions
def Aang_fish : ℕ := 7
def Sokka_fish : ℕ := 5
def Toph_fish : ℕ := 12

-- Total fish and average calculation
def total_fish : ℕ := Aang_fish + Sokka_fish + Toph_fish
def number_of_people : ℕ := 3
def average_fish_per_person : ℕ := total_fish / number_of_people

-- Theorem to prove
theorem calculate_average_fish_caught : average_fish_per_person = 8 := 
by 
  -- Proof steps are skipped with 'sorry', but the statement is set up correctly
  sorry

end calculate_average_fish_caught_l142_142451


namespace find_second_offset_l142_142341

theorem find_second_offset 
  (diagonal : ℝ) (offset1 : ℝ) (area_quad : ℝ) (offset2 : ℝ)
  (h1 : diagonal = 20) (h2 : offset1 = 9) (h3 : area_quad = 150) :
  offset2 = 6 :=
by
  sorry

end find_second_offset_l142_142341


namespace sqrt_neg2023_squared_l142_142949

theorem sqrt_neg2023_squared : Real.sqrt ((-2023 : ℝ)^2) = 2023 :=
by
  sorry

end sqrt_neg2023_squared_l142_142949


namespace compute_value_l142_142114

theorem compute_value {a b : ℝ} 
  (h1 : ∀ x, (x + a) * (x + b) * (x + 12) = 0 → x ≠ -3 → x = -a ∨ x = -b ∨ x = -12)
  (h2 : ∀ x, (x + 2 * a) * (x + 3) * (x + 6) = 0 → x ≠ -b ∧ x ≠ -12 → x = -3) :
  100 * (3 / 2) + 6 = 156 :=
by
  sorry

end compute_value_l142_142114


namespace values_of_a_and_b_l142_142190

theorem values_of_a_and_b (a b : ℝ) : 
  (∀ x : ℝ, (x + a - 2 > 0 ∧ 2 * x - b - 1 < 0) ↔ (0 < x ∧ x < 1)) → (a = 2 ∧ b = 1) :=
by 
  sorry

end values_of_a_and_b_l142_142190


namespace odd_squares_diff_divisible_by_8_l142_142972

theorem odd_squares_diff_divisible_by_8 (m n : ℤ) (a b : ℤ) (hm : a = 2 * m + 1) (hn : b = 2 * n + 1) : (a^2 - b^2) % 8 = 0 := sorry

end odd_squares_diff_divisible_by_8_l142_142972


namespace triangle_cos_area_l142_142210

/-- In triangle ABC, with angles A, B, and C, opposite sides a, b, and c respectively, given the condition 
    a * cos C = (2 * b - c) * cos A, prove: 
    1. cos A = 1/2
    2. If a = 6 and b + c = 8, then the area of triangle ABC is 7 * sqrt 3 / 3 --/
theorem triangle_cos_area (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos C = (2 * b - c) * Real.cos A)
  (h2 : a = 6) (h3 : b + c = 8) :
  Real.cos A = 1 / 2 ∧ ∃ area : ℝ, area = 7 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end triangle_cos_area_l142_142210


namespace acid_solution_replaced_l142_142404

theorem acid_solution_replaced (P : ℝ) :
  (0.5 * 0.50 + 0.5 * P = 0.35) → P = 0.20 :=
by
  intro h
  sorry

end acid_solution_replaced_l142_142404


namespace total_fish_l142_142652

def LillyFish : ℕ := 10
def RosyFish : ℕ := 8
def MaxFish : ℕ := 15

theorem total_fish : LillyFish + RosyFish + MaxFish = 33 := by
  sorry

end total_fish_l142_142652


namespace tan_alpha_of_cos_alpha_l142_142930

theorem tan_alpha_of_cos_alpha (α : ℝ) (hα : 0 < α ∧ α < Real.pi) (h_cos : Real.cos α = -3/5) :
  Real.tan α = -4/3 :=
sorry

end tan_alpha_of_cos_alpha_l142_142930


namespace gcd_problem_l142_142417

def a := 47^11 + 1
def b := 47^11 + 47^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := 
by
  sorry

end gcd_problem_l142_142417


namespace total_value_of_item_l142_142195

variable {V : ℝ}

theorem total_value_of_item (h : 0.07 * (V - 1000) = 109.20) : V = 2560 := 
by
  sorry

end total_value_of_item_l142_142195


namespace find_a_plus_b_l142_142199

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x - 6

theorem find_a_plus_b (a b : ℝ) (h_cond : ∀ x : ℝ, h (f a b x) = 4 * x + 3) : a + b = 13 / 3 :=
by
  sorry

end find_a_plus_b_l142_142199


namespace valid_numbers_count_l142_142483

def count_valid_numbers (n : ℕ) : ℕ := 1 / 4 * (5^n + 2 * 3^n + 1)

theorem valid_numbers_count (n : ℕ) : count_valid_numbers n = (1 / 4) * (5^n + 2 * 3^n + 1) :=
by sorry

end valid_numbers_count_l142_142483


namespace martha_gingers_amount_l142_142285

theorem martha_gingers_amount (G : ℚ) (h : G = 0.43 * (G + 3)) : G = 2 := by
  sorry

end martha_gingers_amount_l142_142285


namespace solve_ab_eq_l142_142564

theorem solve_ab_eq (a b : ℕ) (h : a^b + a + b = b^a) : a = 5 ∧ b = 2 :=
sorry

end solve_ab_eq_l142_142564


namespace intersection_of_domains_l142_142806

def M (x : ℝ) : Prop := x < 1
def N (x : ℝ) : Prop := x > -1
def P (x : ℝ) : Prop := -1 < x ∧ x < 1

theorem intersection_of_domains : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | P x} :=
by
  sorry

end intersection_of_domains_l142_142806


namespace james_chore_time_l142_142974

-- Definitions for the conditions
def t_vacuum : ℕ := 3
def t_chores : ℕ := 3 * t_vacuum
def t_total : ℕ := t_vacuum + t_chores

-- Statement
theorem james_chore_time : t_total = 12 := by
  sorry

end james_chore_time_l142_142974


namespace bread_cost_l142_142158

theorem bread_cost
  (B : ℝ)
  (cost_peanut_butter : ℝ := 2)
  (initial_money : ℝ := 14)
  (money_leftover : ℝ := 5.25) :
  3 * B + cost_peanut_butter = (initial_money - money_leftover) → B = 2.25 :=
by
  sorry

end bread_cost_l142_142158


namespace smallest_n_for_107n_same_last_two_digits_l142_142284

theorem smallest_n_for_107n_same_last_two_digits :
  ∃ n : ℕ, n > 0 ∧ (107 * n) % 100 = n % 100 ∧ n = 50 :=
by {
  sorry
}

end smallest_n_for_107n_same_last_two_digits_l142_142284


namespace existence_of_committees_l142_142264

noncomputable def committeesExist : Prop :=
∃ (C : Fin 1990 → Fin 11 → Fin 3), 
  (∀ i j, i ≠ j → C i ≠ C j) ∧
  (∀ i j, i = j + 1 ∨ (i = 0 ∧ j = 1990 - 1) → ∃ k, C i k = C j k)

theorem existence_of_committees : committeesExist :=
sorry

end existence_of_committees_l142_142264


namespace sum_of_reciprocals_of_squares_l142_142629

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 3) :
  (1 / (a : ℚ)^2) + (1 / (b : ℚ)^2) = 10 / 9 :=
sorry

end sum_of_reciprocals_of_squares_l142_142629


namespace bus_stops_for_minutes_per_hour_l142_142375

theorem bus_stops_for_minutes_per_hour (speed_no_stops speed_with_stops : ℕ)
  (h1 : speed_no_stops = 60) (h2 : speed_with_stops = 45) : 
  (60 * (speed_no_stops - speed_with_stops) / speed_no_stops) = 15 :=
by
  sorry

end bus_stops_for_minutes_per_hour_l142_142375


namespace exists_segment_satisfying_condition_l142_142538

theorem exists_segment_satisfying_condition :
  ∃ (x₁ x₂ x₃ : ℚ) (f : ℚ → ℤ), x₃ = (x₁ + x₂) / 2 ∧ f x₁ + f x₂ ≤ 2 * f x₃ :=
sorry

end exists_segment_satisfying_condition_l142_142538


namespace find_angle_BEC_l142_142835

-- Constants and assumptions
def angle_A : ℝ := 45
def angle_D : ℝ := 50
def angle_F : ℝ := 55
def E_above_C : Prop := true  -- This is a placeholder to represent the condition that E is directly above C.

-- Definition of the problem
theorem find_angle_BEC (angle_A_eq : angle_A = 45) 
                      (angle_D_eq : angle_D = 50) 
                      (angle_F_eq : angle_F = 55)
                      (triangle_BEC_formed : Prop)
                      (E_directly_above_C : E_above_C) 
                      : ∃ (BEC : ℝ), BEC = 10 :=
by sorry

end find_angle_BEC_l142_142835


namespace problem_solution_l142_142106

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - x^2 + a * x - a

theorem problem_solution (x₀ x₁ a : ℝ) (h₁ : 3 * x₀^2 - 2 * x₀ + a = 0) (h₂ : f x₁ a = f x₀ a) (h₃ : x₁ ≠ x₀) : x₁ + 2 * x₀ = 1 :=
by
  sorry

end problem_solution_l142_142106


namespace parametric_equations_curveC2_minimum_distance_M_to_curveC_l142_142156

noncomputable def curveC1_param (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sin α)

def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (3 * x, 2 * y)

theorem parametric_equations_curveC2 (θ : ℝ) :
  scaling_transform (Real.cos θ) (Real.sin θ) = (3 * Real.cos θ, 2 * Real.sin θ) :=
sorry

noncomputable def curveC (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin θ + ρ * Real.cos θ = 10

noncomputable def distance_to_curveC (θ : ℝ) : ℝ :=
  abs (3 * Real.cos θ + 4 * Real.sin θ - 10) / Real.sqrt 5

theorem minimum_distance_M_to_curveC : 
  ∀ θ, distance_to_curveC θ >= Real.sqrt 5 :=
sorry

end parametric_equations_curveC2_minimum_distance_M_to_curveC_l142_142156


namespace determine_sanity_l142_142393

-- Defining the conditions for sanity based on responses to a specific question

-- Define possible responses
inductive Response
| ball : Response
| yes : Response

-- Define sanity based on logical interpretation of an illogical question
def is_sane (response : Response) : Prop :=
  response = Response.ball

-- The theorem stating asking the specific question determines sanity
theorem determine_sanity (response : Response) : is_sane response ↔ response = Response.ball :=
by
  sorry

end determine_sanity_l142_142393


namespace find_views_multiplier_l142_142161

theorem find_views_multiplier (M: ℝ) (h: 4000 * M + 50000 = 94000) : M = 11 :=
by
  sorry

end find_views_multiplier_l142_142161


namespace picnic_problem_l142_142211

theorem picnic_problem
  (M W C A : ℕ)
  (h1 : M + W + C = 240)
  (h2 : M = W + 80)
  (h3 : A = C + 80)
  (h4 : A = M + W) :
  M = 120 :=
by
  sorry

end picnic_problem_l142_142211


namespace base_8_digits_sum_l142_142775

theorem base_8_digits_sum
    (X Y Z : ℕ)
    (h1 : 1 ≤ X ∧ X < 8)
    (h2 : 1 ≤ Y ∧ Y < 8)
    (h3 : 1 ≤ Z ∧ Z < 8)
    (h4 : X ≠ Y)
    (h5 : Y ≠ Z)
    (h6 : Z ≠ X)
    (h7 : 8^2 * X + 8 * Y + Z + 8^2 * Y + 8 * Z + X + 8^2 * Z + 8 * X + Y = 8^3 * X + 8^2 * X + 8 * X) :
  Y + Z = 7 * X :=
by
  sorry

end base_8_digits_sum_l142_142775


namespace value_of_S_l142_142710

theorem value_of_S (x R S : ℝ) (h1 : x + 1/x = R) (h2 : R = 6) : x^3 + 1/x^3 = 198 :=
by
  sorry

end value_of_S_l142_142710


namespace necessary_but_not_sufficient_l142_142964

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 - 1 = 0) ↔ (x = -1 ∨ x = 1) ∧ (x - 1 = 0) → (x^2 - 1 = 0) ∧ ¬((x^2 - 1 = 0) → (x - 1 = 0)) := 
by sorry

end necessary_but_not_sufficient_l142_142964


namespace factorization_correct_l142_142535

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l142_142535


namespace pig_count_correct_l142_142043

def initial_pigs : ℝ := 64.0
def additional_pigs : ℝ := 86.0
def total_pigs : ℝ := 150.0

theorem pig_count_correct : initial_pigs + additional_pigs = total_pigs := by
  show 64.0 + 86.0 = 150.0
  sorry

end pig_count_correct_l142_142043


namespace x_squared_minus_y_squared_l142_142807

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9 / 13) (h2 : x - y = 5 / 13) : x^2 - y^2 = 45 / 169 := 
by 
  -- proof omitted 
  sorry

end x_squared_minus_y_squared_l142_142807


namespace geometric_series_common_ratio_l142_142808

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 500) (hS : S = 2500) (h_series : S = a / (1 - r)) : r = 4 / 5 :=
by
  sorry

end geometric_series_common_ratio_l142_142808


namespace shoe_size_ratio_l142_142147

theorem shoe_size_ratio (J A : ℕ) (hJ : J = 7) (hAJ : A + J = 21) : A / J = 2 :=
by
  -- Skipping the proof
  sorry

end shoe_size_ratio_l142_142147


namespace percentage_decrease_last_year_l142_142890

-- Define the percentage decrease last year
variable (x : ℝ)

-- Define the condition that expresses the stock price this year
def final_price_change (x : ℝ) : Prop :=
  (1 - x / 100) * 1.10 = 1 + 4.499999999999993 / 100

-- Theorem stating the percentage decrease
theorem percentage_decrease_last_year : final_price_change 5 := by
  sorry

end percentage_decrease_last_year_l142_142890


namespace prism_properties_sum_l142_142711

/-- Prove that the sum of the number of edges, corners, and faces of a rectangular box (prism) with dimensions 2 by 3 by 4 is 26. -/
theorem prism_properties_sum :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 := 
by
  -- Provided conditions and definitions
  let edges := 12
  let corners := 8
  let faces := 6
  -- Summing up these values
  exact rfl

end prism_properties_sum_l142_142711


namespace exists_directed_triangle_l142_142940

structure Tournament (V : Type) :=
  (edges : V → V → Prop)
  (complete : ∀ x y, x ≠ y → edges x y ∨ edges y x)
  (outdegree_at_least_one : ∀ x, ∃ y, edges x y)

theorem exists_directed_triangle {V : Type} [Fintype V] (T : Tournament V) :
  ∃ (a b c : V), T.edges a b ∧ T.edges b c ∧ T.edges c a := by
sorry

end exists_directed_triangle_l142_142940


namespace two_digit_subtraction_pattern_l142_142013

theorem two_digit_subtraction_pattern (a b : ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) :
  (10 * a + b) - (10 * b + a) = 9 * (a - b) := 
by
  sorry

end two_digit_subtraction_pattern_l142_142013


namespace minimize_prod_time_l142_142622

noncomputable def shortest_production_time
  (items : ℕ) 
  (workers : ℕ) 
  (shaping_time : ℕ) 
  (firing_time : ℕ) : ℕ := by
  sorry

-- The main theorem statement
theorem minimize_prod_time
  (items : ℕ := 75)
  (workers : ℕ := 13)
  (shaping_time : ℕ := 15)
  (drying_time : ℕ := 10)
  (firing_time : ℕ := 30)
  (optimal_time : ℕ := 325) :
  shortest_production_time items workers shaping_time firing_time = optimal_time := by
  sorry

end minimize_prod_time_l142_142622


namespace pigeonhole_6_points_3x4_l142_142904

theorem pigeonhole_6_points_3x4 :
  ∀ (points : Fin 6 → (ℝ × ℝ)), 
  (∀ i, 0 ≤ (points i).fst ∧ (points i).fst ≤ 4 ∧ 0 ≤ (points i).snd ∧ (points i).snd ≤ 3) →
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end pigeonhole_6_points_3x4_l142_142904


namespace fill_box_with_L_blocks_l142_142550

theorem fill_box_with_L_blocks (m n k : ℕ) 
  (hm : m > 1) (hn : n > 1) (hk : k > 1) (hk_div3 : k % 3 = 0) : 
  ∃ (fill : ℕ → ℕ → ℕ → Prop), fill m n k → True := 
by
  sorry

end fill_box_with_L_blocks_l142_142550


namespace solution_set_of_inequality_l142_142882

def f (x : ℝ) : ℝ := sorry
def f_prime (x : ℝ) : ℝ := sorry

theorem solution_set_of_inequality :
  (∀ x > 0, x^2 * f_prime x + 1 > 0) → 
  f 1 = 5 →
  { x : ℝ | 0 < x ∧ x < 1 } = { x : ℝ | 0 < x ∧ f x < 1 / x + 4 } :=
by 
  intros h1 h2 
  sorry

end solution_set_of_inequality_l142_142882


namespace find_constant_l142_142008

-- Define the relationship between Fahrenheit and Celsius
def temp_rel (c f k : ℝ) : Prop :=
  f = (9 / 5) * c + k

-- Temperature increases
def temp_increase (c1 c2 f1 f2 : ℝ) : Prop :=
  (f2 - f1 = 30) ∧ (c2 - c1 = 16.666666666666668)

-- Freezing point condition
def freezing_point (f : ℝ) : Prop :=
  f = 32

-- Main theorem to prove
theorem find_constant (k : ℝ) :
  ∃ (c1 c2 f1 f2: ℝ), temp_rel c1 f1 k ∧ temp_rel c2 f2 k ∧ 
  temp_increase c1 c2 f1 f2 ∧ freezing_point f1 → k = 32 :=
by sorry

end find_constant_l142_142008


namespace problem1_problem2_problem3_problem4_l142_142621

-- Problem 1
theorem problem1 : (-10 + (-5) - (-18)) = 3 := 
by
  sorry

-- Problem 2
theorem problem2 : (-80 * (-(4 / 5)) / (abs 16)) = -4 := 
by 
  sorry

-- Problem 3
theorem problem3 : ((1/2 - 5/9 + 5/6 - 7/12) * (-36)) = -7 := 
by 
  sorry

-- Problem 4
theorem problem4 : (- 3^2 * (-1/3)^2 +(-2)^2 / (- (2/3))^3) = -29 / 27 :=
by 
  sorry

end problem1_problem2_problem3_problem4_l142_142621


namespace vodka_shot_size_l142_142078

theorem vodka_shot_size (x : ℝ) (h1 : 8 / 2 = 4) (h2 : 4 * x = 2 * 3) : x = 1.5 :=
by
  sorry

end vodka_shot_size_l142_142078


namespace English_family_information_l142_142829

-- Define the statements given by the family members.
variables (father_statement : Prop)
          (mother_statement : Prop)
          (daughter_statement : Prop)

-- Conditions provided in the problem
variables (going_to_Spain : Prop)
          (coming_from_Newcastle : Prop)
          (stopped_in_Paris : Prop)

-- Define what each family member said
axiom Father : father_statement ↔ (going_to_Spain ∨ coming_from_Newcastle)
axiom Mother : mother_statement ↔ ((¬going_to_Spain ∧ coming_from_Newcastle) ∨ (stopped_in_Paris ∧ ¬going_to_Spain))
axiom Daughter : daughter_statement ↔ (¬coming_from_Newcastle ∨ stopped_in_Paris)

-- The final theorem to be proved:
theorem English_family_information : (¬going_to_Spain ∧ coming_from_Newcastle ∧ stopped_in_Paris) :=
by
  -- steps to prove the theorem should go here, but they are skipped with sorry
  sorry

end English_family_information_l142_142829


namespace calories_per_person_l142_142555

-- Definitions based on the conditions from a)
def oranges : ℕ := 5
def pieces_per_orange : ℕ := 8
def people : ℕ := 4
def calories_per_orange : ℝ := 80

-- Theorem based on the equivalent proof problem
theorem calories_per_person : 
    ((oranges * pieces_per_orange) / people) / pieces_per_orange * calories_per_orange = 100 := 
by
  sorry

end calories_per_person_l142_142555


namespace roses_per_flat_l142_142856

-- Conditions
def flats_petunias := 4
def petunias_per_flat := 8
def flats_roses := 3
def venus_flytraps := 2
def fertilizer_per_petunia := 8
def fertilizer_per_rose := 3
def fertilizer_per_venus_flytrap := 2
def total_fertilizer_needed := 314

-- Derived definitions
def total_petunias := flats_petunias * petunias_per_flat
def fertilizer_for_petunias := total_petunias * fertilizer_per_petunia
def fertilizer_for_venus_flytraps := venus_flytraps * fertilizer_per_venus_flytrap
def total_fertilizer_needed_roses := total_fertilizer_needed - (fertilizer_for_petunias + fertilizer_for_venus_flytraps)

-- Proof statement
theorem roses_per_flat :
  ∃ R : ℕ, flats_roses * R * fertilizer_per_rose = total_fertilizer_needed_roses ∧ R = 6 :=
by
  -- Proof goes here
  sorry

end roses_per_flat_l142_142856


namespace local_minimum_at_2_l142_142991

noncomputable def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem local_minimum_at_2 (m : ℝ) (h : 2 * (2 - m)^2 + 2 * 4 * (2 - m) = 0) : m = 6 :=
by
  sorry

end local_minimum_at_2_l142_142991


namespace common_root_of_two_equations_l142_142439

theorem common_root_of_two_equations (m x : ℝ) :
  (m * x - 1000 = 1001) ∧ (1001 * x = m - 1000 * x) → (m = 2001 ∨ m = -2001) :=
by
  sorry

end common_root_of_two_equations_l142_142439


namespace find_divisor_l142_142036

theorem find_divisor (D Q R Div : ℕ) (h1 : Q = 40) (h2 : R = 64) (h3 : Div = 2944) 
  (h4 : Div = (D * Q) + R) : D = 72 :=
by
  sorry

end find_divisor_l142_142036


namespace negative_correction_is_correct_l142_142816

-- Define the constants given in the problem
def gain_per_day : ℚ := 13 / 4
def set_time : ℚ := 8 -- 8 A.M. on April 10
def end_time : ℚ := 15 -- 3 P.M. on April 19
def days_passed : ℚ := 9

-- Calculate the total time in hours from 8 A.M. on April 10 to 3 P.M. on April 19
def total_hours_passed : ℚ := days_passed * 24 + (end_time - set_time)

-- Calculate the gain in time per hour
def gain_per_hour : ℚ := gain_per_day / 24

-- Calculate the total gained time over the total hours passed
def total_gain : ℚ := total_hours_passed * gain_per_hour

-- The negative correction m to be subtracted
def correction : ℚ := 2899 / 96

theorem negative_correction_is_correct :
  total_gain = correction :=
by
-- skipping the proof
sorry

end negative_correction_is_correct_l142_142816


namespace x_varies_as_sin_squared_l142_142611

variable {k j z : ℝ}
variable (x y : ℝ)

-- condition: x is proportional to y^2
def proportional_xy_square (x y : ℝ) (k : ℝ) : Prop :=
  x = k * y ^ 2

-- condition: y is proportional to sin(z)
def proportional_y_sin (y : ℝ) (j z : ℝ) : Prop :=
  y = j * Real.sin z

-- statement to prove: x is proportional to (sin(z))^2
theorem x_varies_as_sin_squared (k j z : ℝ) (x y : ℝ)
  (h1 : proportional_xy_square x y k)
  (h2 : proportional_y_sin y j z) :
  ∃ m, x = m * (Real.sin z) ^ 2 :=
by
  sorry

end x_varies_as_sin_squared_l142_142611


namespace find_number_l142_142995

theorem find_number (x : ℝ) (h : (4 / 3) * x = 48) : x = 36 :=
sorry

end find_number_l142_142995


namespace chloromethane_formation_l142_142466

variable (CH₄ Cl₂ CH₃Cl : Type)
variable (molesCH₄ molesCl₂ molesCH₃Cl : ℕ)

theorem chloromethane_formation 
  (h₁ : molesCH₄ = 3)
  (h₂ : molesCl₂ = 3)
  (reaction : CH₄ → Cl₂ → CH₃Cl)
  (one_to_one : ∀ (x y : ℕ), x = y → x = y): 
  molesCH₃Cl = 3 :=
by
  sorry

end chloromethane_formation_l142_142466


namespace problem_statement_l142_142365

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := 2 * x + 1

theorem problem_statement : f (g 5) - g (f 5) = 63 :=
by
  sorry

end problem_statement_l142_142365


namespace cone_lateral_area_l142_142250

theorem cone_lateral_area (cos_ASB : ℝ)
  (angle_SA_base : ℝ)
  (triangle_SAB_area : ℝ) :
  cos_ASB = 7 / 8 →
  angle_SA_base = 45 →
  triangle_SAB_area = 5 * Real.sqrt 15 →
  (lateral_area : ℝ) = 40 * Real.sqrt 2 * Real.pi :=
by
  intros h1 h2 h3
  sorry

end cone_lateral_area_l142_142250


namespace polar_to_rectangular_coordinates_l142_142664

noncomputable def rectangular_coordinates_from_polar (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_coordinates :
  rectangular_coordinates_from_polar 12 (5 * Real.pi / 4) = (-6 * Real.sqrt 2, -6 * Real.sqrt 2) :=
  sorry

end polar_to_rectangular_coordinates_l142_142664


namespace quarters_and_dimes_l142_142014

theorem quarters_and_dimes (n : ℕ) (qval : ℕ := 25) (dval : ℕ := 10) 
  (hq : 20 * qval + 10 * dval = 10 * qval + n * dval) : 
  n = 35 :=
by
  sorry

end quarters_and_dimes_l142_142014


namespace triangle_other_side_length_l142_142423

theorem triangle_other_side_length (a b : ℝ) (c : ℝ) (h_a : a = 3) (h_b : b = 4) (h_right_angle : c * c = a * a + b * b ∨ a * a = c * c + b * b):
  c = Real.sqrt 7 ∨ c = 5 :=
by
  sorry

end triangle_other_side_length_l142_142423


namespace pool_capacity_percentage_l142_142596

theorem pool_capacity_percentage
  (rate : ℕ := 60) -- cubic feet per minute
  (time : ℕ := 800) -- minutes
  (width : ℕ := 60) -- feet
  (length : ℕ := 100) -- feet
  (depth : ℕ := 10) -- feet
  : (rate * time * 100) / (width * length * depth) = 8 := by
{
  sorry
}

end pool_capacity_percentage_l142_142596


namespace yield_is_eight_percent_l142_142050

noncomputable def par_value : ℝ := 100
noncomputable def annual_dividend : ℝ := 0.12 * par_value
noncomputable def market_value : ℝ := 150
noncomputable def yield_percentage : ℝ := (annual_dividend / market_value) * 100

theorem yield_is_eight_percent : yield_percentage = 8 := 
by 
  sorry

end yield_is_eight_percent_l142_142050


namespace task_completion_days_l142_142865

theorem task_completion_days (a b c d : ℝ) 
    (h1 : 1/a + 1/b = 1/8)
    (h2 : 1/b + 1/c = 1/6)
    (h3 : 1/c + 1/d = 1/12) :
    1/a + 1/d = 1/24 :=
by
  sorry

end task_completion_days_l142_142865


namespace melanie_bought_books_l142_142492

-- Defining the initial number of books and final number of books
def initial_books : ℕ := 41
def final_books : ℕ := 87

-- Theorem stating that Melanie bought 46 books at the yard sale
theorem melanie_bought_books : (final_books - initial_books) = 46 := by
  sorry

end melanie_bought_books_l142_142492


namespace find_radius_of_circle_l142_142853

theorem find_radius_of_circle (C : ℝ) (h : C = 72 * Real.pi) : ∃ r : ℝ, 2 * Real.pi * r = C ∧ r = 36 :=
by
  sorry

end find_radius_of_circle_l142_142853


namespace verify_trees_in_other_row_l142_142943

-- Definition of a normal lemon tree lemon production per year
def normalLemonTreeProduction : ℕ := 60

-- Definition of the percentage increase in lemon production for specially engineered lemon trees
def percentageIncrease : ℕ := 50

-- Definition of lemon production for specially engineered lemon trees
def specialLemonTreeProduction : ℕ := normalLemonTreeProduction * (1 + percentageIncrease / 100)

-- Number of trees in one row of the grove
def treesInOneRow : ℕ := 50

-- Total lemon production in 5 years
def totalLemonProduction : ℕ := 675000

-- Number of years
def years : ℕ := 5

-- Total number of trees in the grove
def totalNumberOfTrees : ℕ := totalLemonProduction / (specialLemonTreeProduction * years)

-- Number of trees in the other row
def treesInOtherRow : ℕ := totalNumberOfTrees - treesInOneRow

-- Theorem: Verification of the number of trees in the other row
theorem verify_trees_in_other_row : treesInOtherRow = 1450 :=
  by
  -- Proof logic is omitted, leaving as sorry
  sorry

end verify_trees_in_other_row_l142_142943


namespace person_c_completion_time_l142_142234

def job_completion_days (Ra Rb Rc : ℚ) (total_earnings b_earnings : ℚ) : ℚ :=
  Rc

theorem person_c_completion_time (Ra Rb Rc : ℚ)
  (hRa : Ra = 1 / 6)
  (hRb : Rb = 1 / 8)
  (total_earnings : ℚ)
  (b_earnings : ℚ)
  (earnings_ratio : b_earnings / total_earnings = Rb / (Ra + Rb + Rc))
  : Rc = 1 / 12 :=
sorry

end person_c_completion_time_l142_142234


namespace wall_width_is_4_l142_142619

structure Wall where
  width : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

theorem wall_width_is_4 (h_eq_6w : ∀ (wall : Wall), wall.height = 6 * wall.width)
                        (l_eq_7h : ∀ (wall : Wall), wall.length = 7 * wall.height)
                        (volume_16128 : ∀ (wall : Wall), wall.volume = 16128) :
  ∃ (wall : Wall), wall.width = 4 :=
by
  sorry

end wall_width_is_4_l142_142619


namespace original_numbers_l142_142037

theorem original_numbers (a b c d : ℕ) (x : ℕ)
  (h1 : a + b + c + d = 45)
  (h2 : a + 2 = x)
  (h3 : b - 2 = x)
  (h4 : 2 * c = x)
  (h5 : d / 2 = x) : 
  (a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20) :=
sorry

end original_numbers_l142_142037


namespace work_completed_in_8_days_l142_142171

theorem work_completed_in_8_days 
  (A_complete : ℕ → Prop)
  (B_complete : ℕ → Prop)
  (C_complete : ℕ → Prop)
  (A_can_complete_in_10_days : A_complete 10)
  (B_can_complete_in_20_days : B_complete 20)
  (C_can_complete_in_30_days : C_complete 30)
  (A_leaves_5_days_before_completion : ∀ x : ℕ, x ≥ 5 → A_complete (x - 5))
  (C_leaves_3_days_before_completion : ∀ x : ℕ, x ≥ 3 → C_complete (x - 3)) :
  ∃ x : ℕ, x = 8 := sorry

end work_completed_in_8_days_l142_142171


namespace range_of_m_l142_142607

variable (a b c m y1 y2 y3 : Real)

-- Given points and the parabola equation
def on_parabola (x y a b c : Real) : Prop := y = a * x^2 + b * x + c

-- Conditions
variable (hP : on_parabola (-2) y1 a b c)
variable (hQ : on_parabola 4 y2 a b c)
variable (hM : on_parabola m y3 a b c)
variable (h_vertex : 2 * a * m + b = 0)
variable (h_y_order : y3 ≥ y2 ∧ y2 > y1)

-- Theorem to prove m > 1
theorem range_of_m : m > 1 :=
sorry

end range_of_m_l142_142607


namespace f_at_3_l142_142736

noncomputable def f : ℝ → ℝ := sorry

lemma periodic (f : ℝ → ℝ) : ∀ x : ℝ, f (x + 4) = f x := sorry

lemma odd_function (f : ℝ → ℝ) : ∀ x : ℝ, f (-x) + f x = 0 := sorry

lemma given_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = (x - 1)^2 := sorry

theorem f_at_3 : f 3 = 0 := 
by
  sorry

end f_at_3_l142_142736


namespace compute_fraction_l142_142456

theorem compute_fraction : 
  (2045^2 - 2030^2) / (2050^2 - 2025^2) = 3 / 5 :=
by
  sorry

end compute_fraction_l142_142456


namespace maximum_a_value_l142_142678

theorem maximum_a_value :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → -2022 ≤ (a + 1)*x^2 - (a + 1)*x + 2022 ∧ (a + 1)*x^2 - (a + 1)*x + 2022 ≤ 2022) →
  a ≤ 16175 := 
by {
  sorry
}

end maximum_a_value_l142_142678


namespace point_in_third_quadrant_l142_142768

theorem point_in_third_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : (-b < 0 ∧ a - 3 < 0) :=
by sorry

end point_in_third_quadrant_l142_142768


namespace total_distance_craig_walked_l142_142252

theorem total_distance_craig_walked :
  0.2 + 0.7 = 0.9 :=
by sorry

end total_distance_craig_walked_l142_142252


namespace distance_from_origin_l142_142219

theorem distance_from_origin (x y : ℝ) :
  (x, y) = (12, -5) →
  (0, 0) = (0, 0) →
  Real.sqrt ((x - 0)^2 + (y - 0)^2) = 13 :=
by
  -- Please note, the proof steps go here, but they are omitted as per instructions.
  -- Typically we'd use sorry to indicate the proof is missing.
  sorry

end distance_from_origin_l142_142219


namespace roots_equation_l142_142951

theorem roots_equation (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) : p + q = 69 :=
sorry

end roots_equation_l142_142951


namespace age_difference_proof_l142_142467

theorem age_difference_proof (A B C : ℕ) (h1 : B = 12) (h2 : B = 2 * C) (h3 : A + B + C = 32) :
  A - B = 2 :=
by
  sorry

end age_difference_proof_l142_142467


namespace min_expr_value_l142_142475

theorem min_expr_value (α β : ℝ) :
  ∃ (c : ℝ), c = 36 ∧ ((3 * Real.cos α + 4 * Real.sin β - 5)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = c) :=
sorry

end min_expr_value_l142_142475


namespace merchant_profit_percentage_l142_142684

theorem merchant_profit_percentage (C S : ℝ) (h : 24 * C = 16 * S) : ((S - C) / C) * 100 = 50 := by
  -- Adding "by" to denote beginning of proof section
  sorry  -- Proof is skipped

end merchant_profit_percentage_l142_142684


namespace polynomial_real_root_l142_142187

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^5 + a * x^4 - x^3 + a * x^2 + x + 1 = 0) ↔
  (a ∈ (Set.Iic (-1/2)) ∨ a ∈ (Set.Ici (1/2))) :=
by
  sorry

end polynomial_real_root_l142_142187


namespace fractional_eq_nonneg_solution_l142_142486

theorem fractional_eq_nonneg_solution 
  (m x : ℝ)
  (h1 : x ≠ 2)
  (h2 : x ≥ 0)
  (eq_fractional : m / (x - 2) + 1 = x / (2 - x)) :
  m ≤ 2 ∧ m ≠ -2 := 
  sorry

end fractional_eq_nonneg_solution_l142_142486


namespace will_3_point_shots_l142_142499

theorem will_3_point_shots :
  ∃ x y : ℕ, 3 * x + 2 * y = 26 ∧ x + y = 11 ∧ x = 4 :=
by
  sorry

end will_3_point_shots_l142_142499


namespace work_completion_time_l142_142717

-- Define the constants for work rates and times
def W : ℚ := 1
def P_rate : ℚ := W / 20
def Q_rate : ℚ := W / 12
def initial_days : ℚ := 4

-- Define the amount of work done by P in the initial 4 days
def work_done_initial : ℚ := initial_days * P_rate

-- Define the remaining work after initial 4 days
def remaining_work : ℚ := W - work_done_initial

-- Define the combined work rate of P and Q
def combined_rate : ℚ := P_rate + Q_rate

-- Define the time taken to complete the remaining work
def remaining_days : ℚ := remaining_work / combined_rate

-- Define the total time taken to complete the work
def total_days : ℚ := initial_days + remaining_days

-- The theorem to prove
theorem work_completion_time :
  total_days = 10 := 
by
  -- these term can be the calculation steps
  sorry

end work_completion_time_l142_142717


namespace fundraiser_price_per_item_l142_142174

theorem fundraiser_price_per_item
  (students_brownies : ℕ)
  (brownies_per_student : ℕ)
  (students_cookies : ℕ)
  (cookies_per_student : ℕ)
  (students_donuts : ℕ)
  (donuts_per_student : ℕ)
  (total_amount_raised : ℕ)
  (total_brownies : ℕ := students_brownies * brownies_per_student)
  (total_cookies : ℕ := students_cookies * cookies_per_student)
  (total_donuts : ℕ := students_donuts * donuts_per_student)
  (total_items : ℕ := total_brownies + total_cookies + total_donuts)
  (price_per_item : ℕ := total_amount_raised / total_items) :
  students_brownies = 30 →
  brownies_per_student = 12 →
  students_cookies = 20 →
  cookies_per_student = 24 →
  students_donuts = 15 →
  donuts_per_student = 12 →
  total_amount_raised = 2040 →
  price_per_item = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  sorry

end fundraiser_price_per_item_l142_142174


namespace population_of_seventh_village_l142_142724

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 980]

def average_population : ℕ := 1000

theorem population_of_seventh_village 
  (h1 : List.length village_populations = 6)
  (h2 : 1000 * 7 = 7000)
  (h3 : village_populations.sum = 5751) : 
  7000 - village_populations.sum = 1249 := 
by {
  -- h1 ensures there's exactly 6 villages in the list
  -- h2 calculates the total population of 7 villages assuming the average population
  -- h3 calculates the sum of populations in the given list of 6 villages
  -- our goal is to show that 7000 - village_populations.sum = 1249
  -- this will be simplified in the proof
  sorry
}

end population_of_seventh_village_l142_142724


namespace factor_expression_l142_142583

variable (y : ℝ)

theorem factor_expression : 
  6*y*(y + 2) + 15*(y + 2) + 12 = 3*(2*y + 5)*(y + 2) :=
sorry

end factor_expression_l142_142583


namespace joshua_final_bottle_caps_l142_142617

def initial_bottle_caps : ℕ := 150
def bought_bottle_caps : ℕ := 23
def given_away_bottle_caps : ℕ := 37

theorem joshua_final_bottle_caps : (initial_bottle_caps + bought_bottle_caps - given_away_bottle_caps) = 136 := by
  sorry

end joshua_final_bottle_caps_l142_142617


namespace speed_in_still_water_l142_142135

theorem speed_in_still_water (U D : ℝ) (hU : U = 15) (hD : D = 25) : (U + D) / 2 = 20 :=
by
  rw [hU, hD]
  norm_num

end speed_in_still_water_l142_142135


namespace quadratic_real_roots_condition_l142_142297

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + x + m = 0) → m ≤ 1/4 :=
by
  sorry

end quadratic_real_roots_condition_l142_142297


namespace difference_between_heads_and_feet_l142_142847

-- Definitions based on the conditions
def penguins := 30
def zebras := 22
def tigers := 8
def zookeepers := 12

-- Counting heads
def heads := penguins + zebras + tigers + zookeepers

-- Counting feet
def feet := (2 * penguins) + (4 * zebras) + (4 * tigers) + (2 * zookeepers)

-- Proving the difference between the number of feet and heads is 132
theorem difference_between_heads_and_feet : (feet - heads) = 132 :=
by
  sorry

end difference_between_heads_and_feet_l142_142847


namespace john_taller_than_lena_l142_142261

-- Define the heights of John, Lena, and Rebeca.
variables (J L R : ℕ)

-- Given conditions:
-- 1. John has a height of 152 cm
axiom john_height : J = 152

-- 2. John is 6 cm shorter than Rebeca
axiom john_shorter_rebeca : J = R - 6

-- 3. The height of Lena and Rebeca together is 295 cm
axiom lena_rebeca_together : L + R = 295

-- Prove that John is 15 cm taller than Lena
theorem john_taller_than_lena : (J - L) = 15 := by
  sorry

end john_taller_than_lena_l142_142261


namespace range_of_m_l142_142214

theorem range_of_m (m : ℝ) : 
  ((m - 1) * x^2 - 4 * x + 1 = 0) → 
  ((20 - 4 * m ≥ 0) ∧ (m ≠ 1)) :=
by
  sorry

end range_of_m_l142_142214


namespace equal_roots_quadratic_l142_142414

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

/--
If the quadratic equation 2x^2 - ax + 2 = 0 has two equal real roots,
then the value of a is ±4.
-/
theorem equal_roots_quadratic (a : ℝ) (h : quadratic_discriminant 2 (-a) 2 = 0) :
  a = 4 ∨ a = -4 :=
sorry

end equal_roots_quadratic_l142_142414


namespace workshop_workers_transfer_l142_142781

theorem workshop_workers_transfer (w d t : ℕ) (h_w : 63 ≤ w) (h_d : d ≤ 31) 
(h_prod : 1994 = 31 * w + t * (t + 1) / 2) : 
(d = 28 ∧ t = 10) ∨ (d = 30 ∧ t = 21) := sorry

end workshop_workers_transfer_l142_142781


namespace servings_per_day_l142_142635

-- Definitions based on the given problem conditions
def serving_size : ℚ := 0.5
def container_size : ℚ := 32 - 2 -- 1 quart is 32 ounces and the jar is 2 ounces less
def days_last : ℕ := 20

-- The theorem statement to prove
theorem servings_per_day (h1 : serving_size = 0.5) (h2 : container_size = 30) (h3 : days_last = 20) :
  (container_size / days_last) / serving_size = 3 :=
by
  sorry

end servings_per_day_l142_142635


namespace rad_to_deg_eq_l142_142992

theorem rad_to_deg_eq : (4 / 3) * 180 = 240 := by
  sorry

end rad_to_deg_eq_l142_142992


namespace problem_a_problem_b_problem_c_l142_142067

open Real

noncomputable def conditions (x : ℝ) := x >= 1 / 2

/-- 
a) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = \sqrt{2} \)
valid if and only if x in [1/2, 1].
-/
theorem problem_a (x : ℝ) (h : conditions x) :
  (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = sqrt 2) ↔ (1 / 2 ≤ x ∧ x ≤ 1) :=
  sorry

/-- 
b) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = 1 \)
has no solution.
-/
theorem problem_b (x : ℝ) (h : conditions x) :
  sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 1 → False :=
  sorry

/-- 
c) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = 2 \)
if and only if x = 3/2.
-/
theorem problem_c (x : ℝ) (h : conditions x) :
  (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 2) ↔ (x = 3 / 2) :=
  sorry

end problem_a_problem_b_problem_c_l142_142067


namespace total_towels_folded_in_one_hour_l142_142059

-- Define the conditions for folding rates and breaks of each person
def Jane_folding_rate (minutes : ℕ) : ℕ :=
  if minutes % 8 < 5 then 5 * (minutes / 8 + 1) else 5 * (minutes / 8)

def Kyla_folding_rate (minutes : ℕ) : ℕ :=
  if minutes < 30 then 12 * (minutes / 10 + 1) else 36 + 6 * ((minutes - 30) / 10)

def Anthony_folding_rate (minutes : ℕ) : ℕ :=
  if minutes <= 40 then 14 * (minutes / 20)
  else if minutes <= 50 then 28
  else 28 + 14 * ((minutes - 50) / 20)

def David_folding_rate (minutes : ℕ) : ℕ :=
  let sets := minutes / 15
  let additional := sets / 3
  4 * (sets - additional) + 5 * additional

-- Definitions are months passing given in the questions
def hours_fold_towels (minutes : ℕ) : ℕ :=
  Jane_folding_rate minutes + Kyla_folding_rate minutes + Anthony_folding_rate minutes + David_folding_rate minutes

theorem total_towels_folded_in_one_hour : hours_fold_towels 60 = 134 := sorry

end total_towels_folded_in_one_hour_l142_142059


namespace gcd_45_75_l142_142983

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l142_142983


namespace ratio_of_boys_to_girls_l142_142809

/-- 
  Given 200 girls and a total of 600 students in a college,
  the ratio of the number of boys to the number of girls is 2:1.
--/
theorem ratio_of_boys_to_girls 
  (num_girls : ℕ) (total_students : ℕ) (h_girls : num_girls = 200) 
  (h_total : total_students = 600) : 
  (total_students - num_girls) / num_girls = 2 :=
by
  sorry

end ratio_of_boys_to_girls_l142_142809


namespace ratio_students_l142_142556

theorem ratio_students
  (finley_students : ℕ)
  (johnson_students : ℕ)
  (h_finley : finley_students = 24)
  (h_johnson : johnson_students = 22)
  : (johnson_students : ℚ) / ((finley_students / 2 : ℕ) : ℚ) = 11 / 6 :=
by
  sorry

end ratio_students_l142_142556


namespace perpendicular_vectors_relation_l142_142044

theorem perpendicular_vectors_relation (a b : ℝ) (h : 3 * a - 7 * b = 0) : a = 7 * b / 3 :=
by
  sorry

end perpendicular_vectors_relation_l142_142044


namespace sqrt6_eq_l142_142968

theorem sqrt6_eq (r : Real) (h : r = Real.sqrt 2 + Real.sqrt 3) : Real.sqrt 6 = (r ^ 2 - 5) / 2 :=
by
  sorry

end sqrt6_eq_l142_142968


namespace inconsistent_b_positive_l142_142120

theorem inconsistent_b_positive
  (a b c : ℝ)
  (h_solution_set : ∀ x : ℝ, -2 < x ∧ x < 3 / 2 → ax^2 + bx + c > 0) :
  ¬ b > 0 :=
sorry

end inconsistent_b_positive_l142_142120


namespace find_range_of_f_l142_142915

noncomputable def f (x : ℝ) : ℝ := (Real.logb (1/2) x) ^ 2 - 2 * (Real.logb (1/2) x) + 4

theorem find_range_of_f :
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → 7 ≤ f x ∧ f x ≤ 12 :=
by
  sorry

end find_range_of_f_l142_142915


namespace cape_may_shark_sightings_l142_142055

def total_shark_sightings (D C : ℕ) : Prop :=
  D + C = 40

def cape_may_sightings (D C : ℕ) : Prop :=
  C = 2 * D - 8

theorem cape_may_shark_sightings : 
  ∃ (C D : ℕ), total_shark_sightings D C ∧ cape_may_sightings D C ∧ C = 24 :=
by
  sorry

end cape_may_shark_sightings_l142_142055


namespace farmer_land_l142_142818

theorem farmer_land (A : ℝ) (A_nonneg : A ≥ 0) (cleared_land : ℝ) 
  (soybeans wheat potatoes vegetables corn : ℝ) 
  (h_cleared : cleared_land = 0.95 * A) 
  (h_soybeans : soybeans = 0.35 * cleared_land) 
  (h_wheat : wheat = 0.40 * cleared_land) 
  (h_potatoes : potatoes = 0.15 * cleared_land) 
  (h_vegetables : vegetables = 0.08 * cleared_land) 
  (h_corn : corn = 630) 
  (cleared_sum : soybeans + wheat + potatoes + vegetables + corn = cleared_land) :
  A = 33158 := 
by 
  sorry

end farmer_land_l142_142818


namespace inequality_solution_equality_condition_l142_142893

theorem inequality_solution (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) (h3 : b < -1 ∨ b > 0) :
  (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b :=
sorry

theorem equality_condition (a b : ℝ) :
  (1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b :=
sorry

end inequality_solution_equality_condition_l142_142893


namespace line_of_intersection_in_standard_form_l142_142376

noncomputable def plane1 (x y z : ℝ) := 3 * x + 4 * y - 2 * z = 5
noncomputable def plane2 (x y z : ℝ) := 2 * x + 3 * y - z = 3

theorem line_of_intersection_in_standard_form :
  (∃ x y z : ℝ, plane1 x y z ∧ plane2 x y z ∧ (∀ t : ℝ, (x, y, z) = 
  (3 + 2 * t, -1 - t, t))) :=
by {
  sorry
}

end line_of_intersection_in_standard_form_l142_142376


namespace find_t_l142_142631

theorem find_t (t : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - t| + |5 - x|) (h2 : ∃ x, f x = 3) : t = 2 ∨ t = 8 :=
by
  sorry

end find_t_l142_142631


namespace min_value_z_l142_142304

theorem min_value_z (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  ∃ z_min, z_min = (x + 1 / x) * (y + 1 / y) ∧ z_min = 33 / 4 :=
sorry

end min_value_z_l142_142304


namespace min_games_required_l142_142905

-- Given condition: max_games ≤ 15
def max_games := 15

-- Theorem statement to prove: minimum number of games that must be played is 8
theorem min_games_required (n : ℕ) (h : n ≤ max_games) : n = 8 :=
sorry

end min_games_required_l142_142905


namespace solution_range_of_a_l142_142170

theorem solution_range_of_a (a : ℝ) (x y : ℝ) :
  3 * x + y = 1 + a → x + 3 * y = 3 → x + y < 2 → a < 4 :=
by
  sorry

end solution_range_of_a_l142_142170


namespace even_integers_diff_digits_200_to_800_l142_142430

theorem even_integers_diff_digits_200_to_800 :
  ∃ n : ℕ, n = 131 ∧ (∀ x : ℕ, 200 ≤ x ∧ x < 800 ∧ (x % 2 = 0) ∧ (∀ i j : ℕ, i ≠ j → (x / 10^i % 10) ≠ (x / 10^j % 10)) ↔ x < n) :=
sorry

end even_integers_diff_digits_200_to_800_l142_142430


namespace school_pays_570_l142_142396

theorem school_pays_570
  (price_per_model : ℕ := 100)
  (models_kindergarten : ℕ := 2)
  (models_elementary_multiple : ℕ := 2)
  (total_models : ℕ := models_kindergarten + models_elementary_multiple * models_kindergarten)
  (price_reduction : ℕ := if total_models > 5 then (price_per_model * 5 / 100) else 0)
  (reduced_price_per_model : ℕ := price_per_model - price_reduction) :
  2 * models_kindergarten * reduced_price_per_model = 570 :=
by
  -- Proof omitted
  sorry

end school_pays_570_l142_142396


namespace calculate_average_age_l142_142363

variables (k : ℕ) (female_to_male_ratio : ℚ) (avg_young_female : ℚ) (avg_old_female : ℚ) (avg_young_male : ℚ) (avg_old_male : ℚ)

theorem calculate_average_age 
  (h_ratio : female_to_male_ratio = 7/8)
  (h_avg_yf : avg_young_female = 26)
  (h_avg_of : avg_old_female = 42)
  (h_avg_ym : avg_young_male = 28)
  (h_avg_om : avg_old_male = 46) : 
  (534/15 : ℚ) = 36 :=
by sorry

end calculate_average_age_l142_142363


namespace quadratic_solve_l142_142168

theorem quadratic_solve (x : ℝ) : (x + 4)^2 = 5 * (x + 4) → x = -4 ∨ x = 1 :=
by sorry

end quadratic_solve_l142_142168


namespace house_value_l142_142957

open Nat

-- Define the conditions
variables (V x : ℕ)
variables (split_amount money_paid : ℕ)
variables (houses_brothers youngest_received : ℕ)
variables (y1 y2 : ℕ)

-- Hypotheses from the conditions
def conditions (V x split_amount money_paid houses_brothers youngest_received y1 y2 : ℕ) :=
  (split_amount = V / 5) ∧
  (houses_brothers = 3) ∧
  (money_paid = 2000) ∧
  (youngest_received = 3000) ∧
  (3 * houses_brothers * money_paid = 6000) ∧
  (y1 = youngest_received) ∧
  (y2 = youngest_received) ∧
  (3 * x + 6000 = V)

-- Main theorem stating the value of one house
theorem house_value (V x : ℕ) (split_amount money_paid houses_brothers youngest_received y1 y2: ℕ) :
  conditions V x split_amount money_paid houses_brothers youngest_received y1 y2 →
  x = 3000 :=
by
  intros
  simp [conditions] at *
  sorry

end house_value_l142_142957


namespace sequence_property_l142_142654

theorem sequence_property (k : ℝ) (h_k : 0 < k) (x : ℕ → ℝ)
  (h₀ : x 0 = 1)
  (h₁ : x 1 = 1 + k)
  (rec1 : ∀ n, x (2*n + 1) - x (2*n) = x (2*n) - x (2*n - 1))
  (rec2 : ∀ n, x (2*n) / x (2*n - 1) = x (2*n - 1) / x (2*n - 2)) :
  ∃ N, ∀ n ≥ N, x n > 1994 :=
by
  sorry

end sequence_property_l142_142654


namespace quadratic_vertex_ordinate_l142_142181

theorem quadratic_vertex_ordinate :
  let a := 2
  let b := -4
  let c := -1
  let vertex_x := -b / (2 * a)
  let vertex_y := a * vertex_x ^ 2 + b * vertex_x + c
  vertex_y = -3 :=
by
  sorry

end quadratic_vertex_ordinate_l142_142181


namespace determine_x_l142_142398

theorem determine_x : ∃ (x : ℕ), 
  (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ ¬(x > 7) ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37 ∧ ¬(2 * x ≥ 21) ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ x < 120 ∧ ¬(4 * x > 37) ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ ¬(x < 120) ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∧
  x = 10 :=
sorry

end determine_x_l142_142398


namespace inequality_condition_l142_142827

theorem inequality_condition
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 2015) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / Real.sqrt 2015 :=
by
  sorry

end inequality_condition_l142_142827


namespace parallel_vectors_m_eq_neg3_l142_142584

theorem parallel_vectors_m_eq_neg3
  (m : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (m + 1, -3))
  (h2 : b = (2, 3))
  (h3 : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  m = -3 := 
sorry

end parallel_vectors_m_eq_neg3_l142_142584


namespace range_of_a_l142_142634

noncomputable def f (x : ℝ) := (Real.log x) / x
noncomputable def g (x a : ℝ) := -Real.exp 1 * x^2 + a * x

theorem range_of_a (a : ℝ) : (∀ x1 : ℝ, ∃ x2 ∈ Set.Icc (1/3) 2, f x1 ≤ g x2 a) → 2 ≤ a :=
sorry

end range_of_a_l142_142634


namespace two_point_seven_five_as_fraction_l142_142447

theorem two_point_seven_five_as_fraction : 2.75 = 11 / 4 :=
by
  sorry

end two_point_seven_five_as_fraction_l142_142447


namespace correct_match_results_l142_142623

-- Define the teams in the league
inductive Team
| Scotland : Team
| England  : Team
| Wales    : Team
| Ireland  : Team

-- Define a match result for a pair of teams
structure MatchResult where
  team1 : Team
  team2 : Team
  goals1 : ℕ
  goals2 : ℕ

def scotland_vs_england : MatchResult := {
  team1 := Team.Scotland,
  team2 := Team.England,
  goals1 := 3,
  goals2 := 0
}

-- All possible match results
def england_vs_ireland : MatchResult := {
  team1 := Team.England,
  team2 := Team.Ireland,
  goals1 := 1,
  goals2 := 0
}

def wales_vs_england : MatchResult := {
  team1 := Team.Wales,
  team2 := Team.England,
  goals1 := 1,
  goals2 := 1
}

def wales_vs_ireland : MatchResult := {
  team1 := Team.Wales,
  team2 := Team.Ireland,
  goals1 := 2,
  goals2 := 1
}

def scotland_vs_ireland : MatchResult := {
  team1 := Team.Scotland,
  team2 := Team.Ireland,
  goals1 := 2,
  goals2 := 0
}

theorem correct_match_results : 
  (england_vs_ireland.goals1 = 1 ∧ england_vs_ireland.goals2 = 0) ∧
  (wales_vs_england.goals1 = 1 ∧ wales_vs_england.goals2 = 1) ∧
  (scotland_vs_england.goals1 = 3 ∧ scotland_vs_england.goals2 = 0) ∧
  (wales_vs_ireland.goals1 = 2 ∧ wales_vs_ireland.goals2 = 1) ∧
  (scotland_vs_ireland.goals1 = 2 ∧ scotland_vs_ireland.goals2 = 0) :=
by 
  sorry

end correct_match_results_l142_142623


namespace lineup_count_l142_142182

def total_players : ℕ := 15
def out_players : ℕ := 3  -- Alice, Max, and John
def lineup_size : ℕ := 6

-- Define the binomial coefficient in Lean
def binom (n k : ℕ) : ℕ :=
  if h : n ≥ k then
    Nat.choose n k
  else
    0

theorem lineup_count (total_players out_players lineup_size : ℕ) :
  let remaining_with_alice := total_players - out_players + 1 
  let remaining_without_alice := total_players - out_players + 1 
  let remaining_without_both := total_players - out_players 
  binom remaining_with_alice (lineup_size-1) + binom remaining_without_alice (lineup_size-1) + binom remaining_without_both lineup_size = 3498 :=
by
  sorry

end lineup_count_l142_142182


namespace solution_l142_142339

theorem solution (x y : ℝ) (h1 : x ≠ y) (h2 : x^2 - 2000 * x = y^2 - 2000 * y) : 
  x + y = 2000 := 
by 
  sorry

end solution_l142_142339


namespace arithmetic_sequence_a4_eight_l142_142518

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, a (n + m) = a n + m * (a 2 - a 1)

variable {a : ℕ → ℤ}

theorem arithmetic_sequence_a4_eight (h_arith_sequence : arithmetic_sequence a)
    (h_cond : a 3 + a 5 = 16) : a 4 = 8 :=
by
  sorry

end arithmetic_sequence_a4_eight_l142_142518


namespace students_failed_to_get_degree_l142_142237

/-- 
Out of 1,500 senior high school students, 70% passed their English exams,
80% passed their Mathematics exams, and 65% passed their Science exams.
To get their degree, a student must pass in all three subjects.
Assume independence of passing rates. This Lean proof shows that
the number of students who failed to get their degree is 954.
-/
theorem students_failed_to_get_degree :
  let total_students := 1500
  let p_english := 0.70
  let p_math := 0.80
  let p_science := 0.65
  let p_all_pass := p_english * p_math * p_science
  let students_all_pass := p_all_pass * total_students
  total_students - students_all_pass = 954 :=
by
  sorry

end students_failed_to_get_degree_l142_142237


namespace max_value_m_l142_142639

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem max_value_m (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c (x-4) = quadratic_function a b c (2-x))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → quadratic_function a b c x ≤ ( (x+1)/2 )^2)
  (h4 : ∀ x : ℝ, quadratic_function a b c x ≥ 0)
  (h_min : ∃ x : ℝ, quadratic_function a b c x = 0) :
  ∃ (m : ℝ), m > 1 ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → quadratic_function a b c (x+t) ≤ x) ∧ m = 9 := 
sorry

end max_value_m_l142_142639


namespace pens_purchased_is_30_l142_142039

def num_pens_purchased (cost_total: ℕ) 
                       (num_pencils: ℕ) 
                       (price_per_pencil: ℚ) 
                       (price_per_pen: ℚ)
                       (expected_pens: ℕ): Prop :=
   let cost_pencils := num_pencils * price_per_pencil
   let cost_pens := cost_total - cost_pencils
   let num_pens := cost_pens / price_per_pen
   num_pens = expected_pens

theorem pens_purchased_is_30 : num_pens_purchased 630 75 2.00 16 30 :=
by
  -- Unfold the definition manually if needed
  sorry

end pens_purchased_is_30_l142_142039


namespace find_c_l142_142605

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_c (c : ℝ) (h1 : f 1 = 1) (h2 : ∀ x y : ℝ, f (x + y) = f x + f y + 8 * x * y - c) (h3 : f 7 = 163) :
  c = 2 / 3 :=
sorry

end find_c_l142_142605


namespace option_a_is_correct_l142_142432

variable (a b : ℝ)
variable (ha : a < 0)
variable (hb : b < 0)
variable (hab : a < b)

theorem option_a_is_correct : (a < abs (3 * a + 2 * b) / 5) ∧ (abs (3 * a + 2 * b) / 5 < b) :=
by
  sorry

end option_a_is_correct_l142_142432


namespace solution_set_I_range_of_m_l142_142498

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |2 * x - 1|

theorem solution_set_I (x : ℝ) : f x < 8 ↔ -5 / 2 < x ∧ x < 3 / 2 :=
sorry

theorem range_of_m (m : ℝ) (h : ∃ x, f x ≤ |3 * m + 1|) : m ≤ -5 / 3 ∨ m ≥ 1 :=
sorry

end solution_set_I_range_of_m_l142_142498
