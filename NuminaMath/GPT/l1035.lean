import Mathlib

namespace NUMINAMATH_GPT_james_can_lift_546_pounds_l1035_103540

def initial_lift_20m : ℝ := 300
def increase_10m : ℝ := 0.30
def strap_increase : ℝ := 0.20
def additional_weight_20m : ℝ := 50
def final_lift_10m_with_straps : ℝ := 546

theorem james_can_lift_546_pounds :
  let initial_lift_10m := initial_lift_20m * (1 + increase_10m)
  let updated_lift_20m := initial_lift_20m + additional_weight_20m
  let ratio := initial_lift_10m / initial_lift_20m
  let updated_lift_10m := updated_lift_20m * ratio
  let lift_with_straps := updated_lift_10m * (1 + strap_increase)
  lift_with_straps = final_lift_10m_with_straps :=
by
  sorry

end NUMINAMATH_GPT_james_can_lift_546_pounds_l1035_103540


namespace NUMINAMATH_GPT_ideal_sleep_hours_l1035_103530

open Nat

theorem ideal_sleep_hours 
  (weeknight_sleep : Nat)
  (weekend_sleep : Nat)
  (sleep_deficit : Nat)
  (num_weeknights : Nat := 5)
  (num_weekend_nights : Nat := 2)
  (total_nights : Nat := 7) :
  weeknight_sleep = 5 →
  weekend_sleep = 6 →
  sleep_deficit = 19 →
  ((num_weeknights * weeknight_sleep + num_weekend_nights * weekend_sleep) + sleep_deficit) / total_nights = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_ideal_sleep_hours_l1035_103530


namespace NUMINAMATH_GPT_card_count_l1035_103597

theorem card_count (x y : ℕ) (h1 : x + y + 2 = 10) (h2 : 3 * x + 4 * y + 10 = 39) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_card_count_l1035_103597


namespace NUMINAMATH_GPT_sum_reciprocals_squares_l1035_103556

theorem sum_reciprocals_squares {a b : ℕ} (h1 : Nat.gcd a b = 1) (h2 : a * b = 11) :
  (1 / (a: ℚ)^2) + (1 / (b: ℚ)^2) = 122 / 121 := 
sorry

end NUMINAMATH_GPT_sum_reciprocals_squares_l1035_103556


namespace NUMINAMATH_GPT_cost_price_of_article_l1035_103533

noncomputable def cost_price (M : ℝ) : ℝ := 98.68 / 1.25

theorem cost_price_of_article (M : ℝ)
    (h1 : 0.95 * M = 98.68)
    (h2 : 98.68 = 1.25 * cost_price M) :
    cost_price M = 78.944 :=
by sorry

end NUMINAMATH_GPT_cost_price_of_article_l1035_103533


namespace NUMINAMATH_GPT_problem_part1_problem_part2_area_height_l1035_103526

theorem problem_part1 (x y : ℝ) (h : abs (x - 4 - 2 * Real.sqrt 2) + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0) : 
  x * y ^ 2 - x ^ 2 * y = -32 * Real.sqrt 2 := 
  sorry

theorem problem_part2_area_height (x y : ℝ) (h : abs (x - 4 - 2 * Real.sqrt 2) + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0) :
  let side_length := Real.sqrt 12
  let area := (1 / 2) * x * y
  let height := area / side_length
  area = 4 ∧ height = (2 * Real.sqrt 3) / 3 := 
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_area_height_l1035_103526


namespace NUMINAMATH_GPT_beginner_trigonometry_probability_l1035_103582

def BC := ℝ
def AC := ℝ
def IC := ℝ
def BT := ℝ
def AT := ℝ
def IT := ℝ
def T := 5000

theorem beginner_trigonometry_probability :
  ∀ (BC AC IC BT AT IT : ℝ),
  (BC + AC + IC = 0.60 * T) →
  (BT + AT + IT = 0.40 * T) →
  (BC + BT = 0.45 * T) →
  (AC + AT = 0.35 * T) →
  (IC + IT = 0.20 * T) →
  (BC = 1.25 * BT) →
  (IC + AC = 1.20 * (IT + AT)) →
  (BT / T = 1/5) :=
by
  intros
  sorry

end NUMINAMATH_GPT_beginner_trigonometry_probability_l1035_103582


namespace NUMINAMATH_GPT_moneyEarnedDuringHarvest_l1035_103536

-- Define the weekly earnings, duration of harvest, and weekly rent.
def weeklyEarnings : ℕ := 403
def durationOfHarvest : ℕ := 233
def weeklyRent : ℕ := 49

-- Define total earnings and total rent.
def totalEarnings : ℕ := weeklyEarnings * durationOfHarvest
def totalRent : ℕ := weeklyRent * durationOfHarvest

-- Calculate the money earned after rent.
def moneyEarnedAfterRent : ℕ := totalEarnings - totalRent

-- The theorem to prove.
theorem moneyEarnedDuringHarvest : moneyEarnedAfterRent = 82482 :=
  by
  sorry

end NUMINAMATH_GPT_moneyEarnedDuringHarvest_l1035_103536


namespace NUMINAMATH_GPT_complex_root_cubic_l1035_103590

theorem complex_root_cubic (a b q r : ℝ) (h_b_ne_zero : b ≠ 0)
  (h_root : (Polynomial.C a + Polynomial.C b * Polynomial.C I) * 
             (Polynomial.C a - Polynomial.C b * Polynomial.C I) * 
             (Polynomial.C (-2 * a)) 
             = Polynomial.X^3 + Polynomial.C q * Polynomial.X + Polynomial.C r) :
  q = b^2 - 3 * a^2 :=
sorry

end NUMINAMATH_GPT_complex_root_cubic_l1035_103590


namespace NUMINAMATH_GPT_mia_receives_chocolate_l1035_103557

-- Given conditions
def total_chocolate : ℚ := 72 / 7
def piles : ℕ := 6
def piles_to_Mia : ℕ := 2

-- Weight of one pile
def weight_of_one_pile (total_chocolate : ℚ) (piles : ℕ) := total_chocolate / piles

-- Total weight Mia receives
def mia_chocolate (weight_of_one_pile : ℚ) (piles_to_Mia : ℕ) := piles_to_Mia * weight_of_one_pile

theorem mia_receives_chocolate : mia_chocolate (weight_of_one_pile total_chocolate piles) piles_to_Mia = 24 / 7 :=
by
  sorry

end NUMINAMATH_GPT_mia_receives_chocolate_l1035_103557


namespace NUMINAMATH_GPT_f_neg_two_l1035_103558

noncomputable def f : ℝ → ℝ := sorry

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

variables (f_odd : is_odd_function f)
variables (f_two : f 2 = 2)

theorem f_neg_two : f (-2) = -2 :=
by
  -- Given that f is an odd function and f(2) = 2
  sorry

end NUMINAMATH_GPT_f_neg_two_l1035_103558


namespace NUMINAMATH_GPT_train_passing_time_l1035_103568

noncomputable def train_length : ℝ := 180
noncomputable def train_speed_km_hr : ℝ := 36
noncomputable def train_speed_m_s : ℝ := train_speed_km_hr * (1000 / 3600)

theorem train_passing_time : train_length / train_speed_m_s = 18 := by
  sorry

end NUMINAMATH_GPT_train_passing_time_l1035_103568


namespace NUMINAMATH_GPT_cyclist_speed_l1035_103578

theorem cyclist_speed
  (V : ℝ)
  (H1 : ∃ t_p : ℝ, V * t_p = 96 ∧ t_p = (96 / (V - 1)) - 2)
  (H2 : V > 1.25 * (V - 1)) :
  V = 16 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_speed_l1035_103578


namespace NUMINAMATH_GPT_john_savings_percentage_l1035_103562

theorem john_savings_percentage :
  ∀ (savings discounted_price total_price original_price : ℝ),
  savings = 4.5 →
  total_price = 49.5 →
  total_price = discounted_price * 1.10 →
  original_price = discounted_price + savings →
  (savings / original_price) * 100 = 9 := by
  intros
  sorry

end NUMINAMATH_GPT_john_savings_percentage_l1035_103562


namespace NUMINAMATH_GPT_find_b_plus_m_l1035_103501

-- Definitions of the constants and functions based on the given conditions.
variables (m b : ℝ)

-- The first line equation passing through (5, 8).
def line1 := 8 = m * 5 + 3

-- The second line equation passing through (5, 8).
def line2 := 8 = 4 * 5 + b

-- The goal statement we need to prove.
theorem find_b_plus_m (h1 : line1 m) (h2 : line2 b) : b + m = -11 :=
sorry

end NUMINAMATH_GPT_find_b_plus_m_l1035_103501


namespace NUMINAMATH_GPT_clotheslines_per_house_l1035_103565

/-- There are a total of 11 children and 20 adults.
Each child has 4 items of clothing on the clotheslines.
Each adult has 3 items of clothing on the clotheslines.
Each clothesline can hold 2 items of clothing.
All of the clotheslines are full.
There are 26 houses on the street.
Show that the number of clotheslines per house is 2. -/
theorem clotheslines_per_house :
  (11 * 4 + 20 * 3) / 2 / 26 = 2 :=
by
  sorry

end NUMINAMATH_GPT_clotheslines_per_house_l1035_103565


namespace NUMINAMATH_GPT_cheenu_speed_difference_l1035_103570

theorem cheenu_speed_difference :
  let cycling_time := 120 -- minutes
  let cycling_distance := 24 -- miles
  let jogging_time := 180 -- minutes
  let jogging_distance := 18 -- miles
  let cycling_speed := cycling_time / cycling_distance -- minutes per mile
  let jogging_speed := jogging_time / jogging_distance -- minutes per mile
  let speed_difference := jogging_speed - cycling_speed -- minutes per mile
  speed_difference = 5 := by sorry

end NUMINAMATH_GPT_cheenu_speed_difference_l1035_103570


namespace NUMINAMATH_GPT_total_students_in_class_l1035_103592

-- Definitions based on the conditions
def volleyball_participants : Nat := 22
def basketball_participants : Nat := 26
def both_participants : Nat := 4

-- The theorem statement
theorem total_students_in_class : volleyball_participants + basketball_participants - both_participants = 44 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1035_103592


namespace NUMINAMATH_GPT_tan_triple_angle_l1035_103585

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end NUMINAMATH_GPT_tan_triple_angle_l1035_103585


namespace NUMINAMATH_GPT_number_of_basketball_cards_l1035_103520

theorem number_of_basketball_cards 
  (B : ℕ) -- Number of basketball cards in each box
  (H1 : 4 * B = 40) -- Given condition from equation 4B = 40
  
  (H2 : 4 * B + 40 - 58 = 22) -- Given condition from the total number of cards

: B = 10 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_basketball_cards_l1035_103520


namespace NUMINAMATH_GPT_correct_operation_l1035_103595

theorem correct_operation (a : ℕ) :
  (a^2 * a^3 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(a^6 / a^2 = a^3) ∧ ¬(3 * a^2 - 2 * a = a^2) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1035_103595


namespace NUMINAMATH_GPT_total_pizza_order_cost_l1035_103569

def pizza_cost_per_pizza := 10
def topping_cost_per_topping := 1
def tip_amount := 5
def number_of_pizzas := 3
def number_of_toppings := 4

theorem total_pizza_order_cost : 
  (pizza_cost_per_pizza * number_of_pizzas + topping_cost_per_topping * number_of_toppings + tip_amount) = 39 := by
  sorry

end NUMINAMATH_GPT_total_pizza_order_cost_l1035_103569


namespace NUMINAMATH_GPT_distinct_colored_triangle_l1035_103503

open Finset

variables {n k : ℕ} (hn : 0 < n) (hk : 3 ≤ k)
variables (K : SimpleGraph (Fin n))
variables (color : Edge (Fin n) → Fin k)
variables (connected_subgraph : ∀ i : Fin k, ∀ u v : Fin n, u ≠ v → (∃ p : Walk (Fin n) u v, ∀ {e}, e ∈ p.edges → color e = i))

theorem distinct_colored_triangle :
  ∃ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  color (A, B) ≠ color (B, C) ∧
  color (B, C) ≠ color (C, A) ∧
  color (C, A) ≠ color (A, B) :=
sorry

end NUMINAMATH_GPT_distinct_colored_triangle_l1035_103503


namespace NUMINAMATH_GPT_ratio_traditionalists_progressives_l1035_103544

-- Define the given conditions
variables (T P C : ℝ)
variables (h1 : C = P + 4 * T)
variables (h2 : 4 * T = 0.75 * C)

-- State the theorem
theorem ratio_traditionalists_progressives (h1 : C = P + 4 * T) (h2 : 4 * T = 0.75 * C) : T / P = 3 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_traditionalists_progressives_l1035_103544


namespace NUMINAMATH_GPT_find_min_value_x_l1035_103535

theorem find_min_value_x (x y z : ℝ) (h1 : x + y + z = 6) (h2 : xy + xz + yz = 10) : 
  ∃ (x_min : ℝ), (∀ (x' : ℝ), (∀ y' z', x' + y' + z' = 6 ∧ x' * y' + x' * z' + y' * z' = 10 → x' ≥ x_min)) ∧ x_min = 2 / 3 :=
sorry

end NUMINAMATH_GPT_find_min_value_x_l1035_103535


namespace NUMINAMATH_GPT_collinear_probability_correct_l1035_103560

def number_of_dots := 25

def number_of_four_dot_combinations := Nat.choose number_of_dots 4

-- Calculate the different possibilities for collinear sets:
def horizontal_sets := 5 * 5
def vertical_sets := 5 * 5
def diagonal_sets := 2 + 2

def total_collinear_sets := horizontal_sets + vertical_sets + diagonal_sets

noncomputable def probability_collinear : ℚ :=
  total_collinear_sets / number_of_four_dot_combinations

theorem collinear_probability_correct :
  probability_collinear = 6 / 1415 :=
sorry

end NUMINAMATH_GPT_collinear_probability_correct_l1035_103560


namespace NUMINAMATH_GPT_product_of_base8_digits_of_8654_l1035_103507

theorem product_of_base8_digits_of_8654 : 
  let base10 := 8654
  let base8_rep := [2, 0, 7, 1, 6] -- Representing 8654(10) to 20716(8)
  (base8_rep.prod = 0) :=
  sorry

end NUMINAMATH_GPT_product_of_base8_digits_of_8654_l1035_103507


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1035_103522

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : 0 > a) 
(h2 : ∀ x : ℝ, (1 < x ∧ x < 2) ↔ (0 < ax^2 + bx + c)) : 
(∀ x : ℝ, (x < 1/2 ∨ 1 < x) ↔ (0 < 2*a*x^2 - 3*a*x + a)) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1035_103522


namespace NUMINAMATH_GPT_hanna_gives_roses_l1035_103534

-- Conditions as Lean definitions
def initial_budget : ℕ := 300
def price_jenna : ℕ := 2
def price_imma : ℕ := 3
def price_ravi : ℕ := 4
def price_leila : ℕ := 5

def roses_for_jenna (budget : ℕ) : ℕ :=
  budget / price_jenna * 1 / 3

def roses_for_imma (budget : ℕ) : ℕ :=
  budget / price_imma * 1 / 4

def roses_for_ravi (budget : ℕ) : ℕ :=
  budget / price_ravi * 1 / 6

def roses_for_leila (budget : ℕ) : ℕ :=
  budget / price_leila

-- Calculations based on conditions
def roses_jenna : ℕ := Nat.floor (50 * 1/3)
def roses_imma : ℕ := Nat.floor ((100 / price_imma) * 1 / 4)
def roses_ravi : ℕ := Nat.floor ((50 / price_ravi) * 1 / 6)
def roses_leila : ℕ := 50 / price_leila

-- Final statement to be proven
theorem hanna_gives_roses :
  roses_jenna + roses_imma + roses_ravi + roses_leila = 36 := by
  sorry

end NUMINAMATH_GPT_hanna_gives_roses_l1035_103534


namespace NUMINAMATH_GPT_constant_term_q_l1035_103529

theorem constant_term_q (p q r : Polynomial ℝ) 
  (hp_const : p.coeff 0 = 6) 
  (hr_const : (p * q).coeff 0 = -18) : q.coeff 0 = -3 :=
sorry

end NUMINAMATH_GPT_constant_term_q_l1035_103529


namespace NUMINAMATH_GPT_rect_area_correct_l1035_103550

-- Defining the function to calculate the area of a rectangle given the coordinates of its vertices
noncomputable def rect_area (x1 y1 x2 y2 x3 y3 x4 y4 : ℤ) : ℤ :=
  let length := abs (x2 - x1)
  let width := abs (y1 - y3)
  length * width

-- The vertices of the rectangle
def x1 : ℤ := -8
def y1 : ℤ := 1
def x2 : ℤ := 1
def y2 : ℤ := 1
def x3 : ℤ := 1
def y3 : ℤ := -7
def x4 : ℤ := -8
def y4 : ℤ := -7

-- Proving that the area of the rectangle is 72 square units
theorem rect_area_correct : rect_area x1 y1 x2 y2 x3 y3 x4 y4 = 72 := by
  sorry

end NUMINAMATH_GPT_rect_area_correct_l1035_103550


namespace NUMINAMATH_GPT_total_dress_designs_l1035_103581

theorem total_dress_designs:
  let colors := 5
  let patterns := 4
  let sleeve_lengths := 2
  colors * patterns * sleeve_lengths = 40 := 
by
  sorry

end NUMINAMATH_GPT_total_dress_designs_l1035_103581


namespace NUMINAMATH_GPT_sam_fish_count_l1035_103516

/-- Let S be the number of fish Sam has. -/
def num_fish_sam : ℕ := sorry

/-- Joe has 8 times as many fish as Sam, which gives 8S fish. -/
def num_fish_joe (S : ℕ) : ℕ := 8 * S

/-- Harry has 4 times as many fish as Joe, hence 32S fish. -/
def num_fish_harry (S : ℕ) : ℕ := 32 * S

/-- Harry has 224 fish. -/
def harry_fish : ℕ := 224

/-- Prove that Sam has 7 fish given the conditions above. -/
theorem sam_fish_count : num_fish_harry num_fish_sam = harry_fish → num_fish_sam = 7 := by
  sorry

end NUMINAMATH_GPT_sam_fish_count_l1035_103516


namespace NUMINAMATH_GPT_high_speed_train_equation_l1035_103542

theorem high_speed_train_equation (x : ℝ) (h1 : x > 0) : 
  700 / x - 700 / (2.8 * x) = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_high_speed_train_equation_l1035_103542


namespace NUMINAMATH_GPT_compute_ab_l1035_103538

theorem compute_ab (a b : ℝ)
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 867.75 := 
by
  sorry

end NUMINAMATH_GPT_compute_ab_l1035_103538


namespace NUMINAMATH_GPT_geo_seq_sum_l1035_103576

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geo_seq_sum (a : ℕ → ℝ) (h : geometric_sequence a) (h1 : a 0 + a 1 = 30) (h4 : a 3 + a 4 = 120) :
  a 6 + a 7 = 480 :=
sorry

end NUMINAMATH_GPT_geo_seq_sum_l1035_103576


namespace NUMINAMATH_GPT_inequality_solution_l1035_103543

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -(4 / 3) ∨ x > -(5 / 3)) := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1035_103543


namespace NUMINAMATH_GPT_emus_count_l1035_103512

theorem emus_count (E : ℕ) (heads : ℕ) (legs : ℕ) 
  (h_heads : ∀ e : ℕ, heads = e) 
  (h_legs : ∀ e : ℕ, legs = 2 * e)
  (h_total : heads + legs = 60) : 
  E = 20 :=
by sorry

end NUMINAMATH_GPT_emus_count_l1035_103512


namespace NUMINAMATH_GPT_john_share_l1035_103504

theorem john_share
  (total_amount : ℝ)
  (john_ratio jose_ratio binoy_ratio : ℝ)
  (total_amount_eq : total_amount = 6000)
  (ratios_eq : john_ratio = 2 ∧ jose_ratio = 4 ∧ binoy_ratio = 6) :
  (john_ratio / (john_ratio + jose_ratio + binoy_ratio)) * total_amount = 1000 :=
by
  -- Here we would derive the proof, but just use sorry for the moment.
  sorry

end NUMINAMATH_GPT_john_share_l1035_103504


namespace NUMINAMATH_GPT_verify_value_of_2a10_minus_a12_l1035_103553

-- Define the arithmetic sequence and the sum condition
variable {a : ℕ → ℝ}  -- arithmetic sequence
variable {a1 : ℝ}     -- the first term of the sequence
variable {d : ℝ}      -- the common difference of the sequence

-- Assume that the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + n * d

-- Assume the sum condition
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- The goal is to prove that 2 * a 10 - a 12 = 24
theorem verify_value_of_2a10_minus_a12 (h_arith : arithmetic_sequence a a1 d) (h_sum : sum_condition a) :
  2 * a 10 - a 12 = 24 :=
  sorry

end NUMINAMATH_GPT_verify_value_of_2a10_minus_a12_l1035_103553


namespace NUMINAMATH_GPT_inequality_proof_l1035_103559

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b + c) / (2 * a) + (c + a) / (2 * b) + (a + b) / (2 * c) ≥ (2 * a) / (b + c) + (2 * b) / (c + a) + (2 * c) / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1035_103559


namespace NUMINAMATH_GPT_complex_division_l1035_103510

theorem complex_division (z1 z2 : ℂ) (h1 : z1 = 1 + 1 * Complex.I) (h2 : z2 = 0 + 2 * Complex.I) :
  z2 / z1 = 1 + Complex.I :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l1035_103510


namespace NUMINAMATH_GPT_continuous_tape_length_l1035_103506

theorem continuous_tape_length :
  let num_sheets := 15
  let sheet_length_cm := 25
  let overlap_cm := 0.5 
  let total_length_without_overlap := num_sheets * sheet_length_cm
  let num_overlaps := num_sheets - 1
  let total_overlap_length := num_overlaps * overlap_cm
  let total_length_cm := total_length_without_overlap - total_overlap_length
  let total_length_m := total_length_cm / 100
  total_length_m = 3.68 := 
by {
  sorry
}

end NUMINAMATH_GPT_continuous_tape_length_l1035_103506


namespace NUMINAMATH_GPT_find_x_l1035_103555

def angle_sum_condition (x : ℝ) := 6 * x + 3 * x + x + x + 4 * x = 360

theorem find_x (x : ℝ) (h : angle_sum_condition x) : x = 24 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l1035_103555


namespace NUMINAMATH_GPT_max_omega_for_increasing_l1035_103547

noncomputable def sin_function (ω : ℕ) (x : ℝ) := Real.sin (ω * x + Real.pi / 6)

def is_monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem max_omega_for_increasing : ∀ (ω : ℕ), (0 < ω) →
  is_monotonically_increasing_on (sin_function ω) (Real.pi / 6) (Real.pi / 4) ↔ ω ≤ 9 :=
sorry

end NUMINAMATH_GPT_max_omega_for_increasing_l1035_103547


namespace NUMINAMATH_GPT_repeating_decimal_sum_l1035_103584

theorem repeating_decimal_sum (x : ℚ) (hx : x = 0.417) :
  let num := 46
  let denom := 111
  let sum := num + denom
  sum = 157 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l1035_103584


namespace NUMINAMATH_GPT_cacti_average_height_l1035_103588

variables {Cactus1 Cactus2 Cactus3 Cactus4 Cactus5 Cactus6 : ℕ}
variables (condition1 : Cactus1 = 14)
variables (condition3 : Cactus3 = 7)
variables (condition6 : Cactus6 = 28)
variables (condition2 : Cactus2 = 14)
variables (condition4 : Cactus4 = 14)
variables (condition5 : Cactus5 = 14)

theorem cacti_average_height : 
  (Cactus1 + Cactus2 + Cactus3 + Cactus4 + Cactus5 + Cactus6 : ℕ) = 91 → 
  (91 : ℝ) / 6 = (15.2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_cacti_average_height_l1035_103588


namespace NUMINAMATH_GPT_citric_acid_molecular_weight_l1035_103546

noncomputable def molecularWeightOfCitricAcid : ℝ :=
  let weight_C := 12.01
  let weight_H := 1.008
  let weight_O := 16.00
  let num_C := 6
  let num_H := 8
  let num_O := 7
  (num_C * weight_C) + (num_H * weight_H) + (num_O * weight_O)

theorem citric_acid_molecular_weight :
  molecularWeightOfCitricAcid = 192.124 :=
by
  -- the step-by-step proof will go here
  sorry

end NUMINAMATH_GPT_citric_acid_molecular_weight_l1035_103546


namespace NUMINAMATH_GPT_original_number_is_16_l1035_103508

theorem original_number_is_16 (x : ℤ) (h1 : 3 * (2 * x + 5) = 111) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_16_l1035_103508


namespace NUMINAMATH_GPT_inequality_proof_l1035_103518

noncomputable def a : ℝ := 1 + Real.tan (-0.2)
noncomputable def b : ℝ := Real.log (0.8 * Real.exp 1)
noncomputable def c : ℝ := 1 / Real.exp 0.2

theorem inequality_proof : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1035_103518


namespace NUMINAMATH_GPT_find_m_l1035_103500

theorem find_m (m : ℝ) (a b : ℝ) (r s : ℝ) (S1 S2 : ℝ)
  (h1 : a = 10)
  (h2 : b = 10)
  (h3 : 10 * r = 5)
  (h4 : S1 = 20)
  (h5 : 10 * s = 5 + m)
  (h6 : S2 = 100 / (5 - m))
  (h7 : S2 = 3 * S1) :
  m = 10 / 3 := by
  sorry

end NUMINAMATH_GPT_find_m_l1035_103500


namespace NUMINAMATH_GPT_triangle_inequality_inequality_l1035_103539

theorem triangle_inequality_inequality {a b c p q r : ℝ}
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b)
  (h4 : p + q + r = 0) :
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_inequality_l1035_103539


namespace NUMINAMATH_GPT_find_smallest_integer_y_l1035_103589

theorem find_smallest_integer_y : ∃ y : ℤ, (8 / 12 : ℚ) < (y / 15) ∧ ∀ z : ℤ, z < y → ¬ ((8 / 12 : ℚ) < (z / 15)) :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_integer_y_l1035_103589


namespace NUMINAMATH_GPT_probability_green_or_yellow_l1035_103571

def total_marbles (green yellow red blue : Nat) : Nat :=
  green + yellow + red + blue

def marble_probability (green yellow red blue : Nat) : Rat :=
  (green + yellow) / (total_marbles green yellow red blue)

theorem probability_green_or_yellow :
  let green := 4
  let yellow := 3
  let red := 4
  let blue := 2
  marble_probability green yellow red blue = 7 / 13 := by
  sorry

end NUMINAMATH_GPT_probability_green_or_yellow_l1035_103571


namespace NUMINAMATH_GPT_functional_equation_solution_l1035_103583

theorem functional_equation_solution :
  ∃ f : ℝ → ℝ,
  (f 1 = 1 ∧ (∀ x y : ℝ, f (x * y + f x) = x * f y + f x)) ∧ f (1/2) = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1035_103583


namespace NUMINAMATH_GPT_eggs_removed_l1035_103586

theorem eggs_removed (initial remaining : ℕ) (h1 : initial = 27) (h2 : remaining = 20) : initial - remaining = 7 :=
by
  sorry

end NUMINAMATH_GPT_eggs_removed_l1035_103586


namespace NUMINAMATH_GPT_solution_set_all_real_solution_set_empty_exists_at_least_one_solution_l1035_103593

-- Definitions for the inequality ax^2 - 2ax + 2a - 3 < 0
def quadratic_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 * a - 3

-- Requirement (1): The solution set is ℝ
theorem solution_set_all_real (a : ℝ) (h : a ≤ 0) : 
  ∀ x : ℝ, quadratic_expr a x < 0 :=
by sorry

-- Requirement (2): The solution set is ∅
theorem solution_set_empty (a : ℝ) (h : a ≥ 3) : 
  ¬∃ x : ℝ, quadratic_expr a x < 0 :=
by sorry

-- Requirement (3): There is at least one real solution
theorem exists_at_least_one_solution (a : ℝ) (h : a < 3) : 
  ∃ x : ℝ, quadratic_expr a x < 0 :=
by sorry

end NUMINAMATH_GPT_solution_set_all_real_solution_set_empty_exists_at_least_one_solution_l1035_103593


namespace NUMINAMATH_GPT_find_d_value_l1035_103548

open Nat

variable {PA BC PB : ℕ}
noncomputable def d (PA BC PB : ℕ) := PB

theorem find_d_value (h₁ : PA = 6) (h₂ : BC = 9) (h₃ : PB = d PA BC PB) : d PA BC PB = 3 := by
  sorry

end NUMINAMATH_GPT_find_d_value_l1035_103548


namespace NUMINAMATH_GPT_initial_number_proof_l1035_103549

-- Definitions for the given problem
def to_add : ℝ := 342.00000000007276
def multiple_of_412 (n : ℤ) : ℝ := 412 * n

-- The initial number
def initial_number : ℝ := 412 - to_add

-- The proof problem statement
theorem initial_number_proof (n : ℤ) (h : multiple_of_412 n = initial_number + to_add) : 
  ∃ x : ℝ, initial_number = x := 
sorry

end NUMINAMATH_GPT_initial_number_proof_l1035_103549


namespace NUMINAMATH_GPT_batsman_average_17th_innings_l1035_103579

theorem batsman_average_17th_innings:
  ∀ (A : ℝ), 
  (16 * A + 85 = 17 * (A + 3)) →
  (A + 3 = 37) :=
by
  intros A h
  sorry

end NUMINAMATH_GPT_batsman_average_17th_innings_l1035_103579


namespace NUMINAMATH_GPT_fraction_sum_of_roots_l1035_103587

theorem fraction_sum_of_roots (x1 x2 : ℝ) (h1 : 5 * x1^2 - 3 * x1 - 2 = 0) (h2 : 5 * x2^2 - 3 * x2 - 2 = 0) (hx : x1 ≠ x2) :
  (1 / x1 + 1 / x2 = -3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_of_roots_l1035_103587


namespace NUMINAMATH_GPT_negation_prob1_negation_prob2_negation_prob3_l1035_103541

-- Definitions and Conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def defines_const_func (f : ℝ → ℝ) (y : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, f x1 = f x2

-- Problem 1
theorem negation_prob1 : 
  (∃ n : ℕ, ∀ p : ℕ, is_prime p → p ≤ n) ↔ 
  ¬(∀ n : ℕ, ∃ p : ℕ, is_prime p ∧ n ≤ p) :=
sorry

-- Problem 2
theorem negation_prob2 : 
  (∃ n : ℤ, ∀ p : ℤ, n + p ≠ 0) ↔ 
  ¬(∀ n : ℤ, ∃! p : ℤ, n + p = 0) :=
sorry

-- Problem 3
theorem negation_prob3 : 
  (∀ y : ℝ, ¬defines_const_func (λ x => x * y) y) ↔ 
  ¬(∃ y : ℝ, defines_const_func (λ x => x * y) y) :=
sorry

end NUMINAMATH_GPT_negation_prob1_negation_prob2_negation_prob3_l1035_103541


namespace NUMINAMATH_GPT_oz_words_lost_l1035_103524

theorem oz_words_lost (letters : Fin 64) (forbidden_letter : Fin 64) (h_forbidden : forbidden_letter.val = 6) : 
  let one_letter_words := 64 
  let two_letter_words := 64 * 64
  let one_letter_lost := if letters = forbidden_letter then 1 else 0
  let two_letter_lost := (if letters = forbidden_letter then 64 else 0) + (if letters = forbidden_letter then 64 else 0) 
  1 + two_letter_lost = 129 :=
by
  sorry

end NUMINAMATH_GPT_oz_words_lost_l1035_103524


namespace NUMINAMATH_GPT_no_pos_reals_floor_prime_l1035_103523

open Real
open Nat

theorem no_pos_reals_floor_prime : 
  ∀ (a b : ℝ), (0 < a) → (0 < b) → ∃ n : ℕ, ¬ Prime (⌊a * n + b⌋) :=
by
  intro a b a_pos b_pos
  sorry

end NUMINAMATH_GPT_no_pos_reals_floor_prime_l1035_103523


namespace NUMINAMATH_GPT_expression_value_l1035_103574

/--
Prove that for a = 51 and b = 15, the expression (a + b)^2 - (a^2 + b^2) equals 1530.
-/
theorem expression_value (a b : ℕ) (h1 : a = 51) (h2 : b = 15) : (a + b)^2 - (a^2 + b^2) = 1530 := by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_expression_value_l1035_103574


namespace NUMINAMATH_GPT_miss_adamson_num_classes_l1035_103531

theorem miss_adamson_num_classes
  (students_per_class : ℕ)
  (sheets_per_student : ℕ)
  (total_sheets : ℕ)
  (h1 : students_per_class = 20)
  (h2 : sheets_per_student = 5)
  (h3 : total_sheets = 400) :
  let sheets_per_class := sheets_per_student * students_per_class
  let num_classes := total_sheets / sheets_per_class
  num_classes = 4 :=
by
  sorry

end NUMINAMATH_GPT_miss_adamson_num_classes_l1035_103531


namespace NUMINAMATH_GPT_contrapositive_true_l1035_103566

theorem contrapositive_true (q p : Prop) (h : q → p) : ¬p → ¬q :=
by sorry

end NUMINAMATH_GPT_contrapositive_true_l1035_103566


namespace NUMINAMATH_GPT_Nellie_legos_l1035_103567

def initial_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_legos : ℕ := 24

def remaining_legos : ℕ := initial_legos - lost_legos - given_legos

theorem Nellie_legos : remaining_legos = 299 := by
  sorry

end NUMINAMATH_GPT_Nellie_legos_l1035_103567


namespace NUMINAMATH_GPT_eval_ceil_sqrt_sum_l1035_103521

theorem eval_ceil_sqrt_sum :
  (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉) = 27 :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by sorry
  have h2 : 5 < Real.sqrt 33 ∧ Real.sqrt 33 < 6 := by sorry
  have h3 : 18 < Real.sqrt 333 ∧ Real.sqrt 333 < 19 := by sorry
  sorry

end NUMINAMATH_GPT_eval_ceil_sqrt_sum_l1035_103521


namespace NUMINAMATH_GPT_three_dice_probability_even_l1035_103552

/-- A die is represented by numbers from 1 to 6. -/
def die := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

/-- Define an event where three dice are thrown, and we check if their sum is even. -/
def three_dice_sum_even (d1 d2 d3 : die) : Prop :=
  (d1.val + d2.val + d3.val) % 2 = 0

/-- Define the probability that a single die shows an odd number. -/
def prob_odd := 1 / 2

/-- Define the probability that a single die shows an even number. -/
def prob_even := 1 / 2

/-- Define the total probability for the sum of three dice to be even. -/
def prob_sum_even : ℚ :=
  prob_even ^ 3 + (3 * prob_odd ^ 2 * prob_even)

theorem three_dice_probability_even :
  prob_sum_even = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_three_dice_probability_even_l1035_103552


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l1035_103575

theorem geometric_sequence_seventh_term (a r: ℤ) (h1 : a = 3) (h2 : a * r ^ 5 = 729) : a * r ^ 6 = 2187 :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l1035_103575


namespace NUMINAMATH_GPT_area_of_polygon_l1035_103545

-- Define the conditions
variables (n : ℕ) (s : ℝ)
-- Given that polygon has 32 sides.
def sides := 32
-- Each side is congruent, and the total perimeter is 64 units.
def perimeter := 64
-- Side length of each side
def side_length := perimeter / sides

-- Area of the polygon we need to prove
def target_area := 96

theorem area_of_polygon : side_length * side_length * sides = target_area := 
by {
  -- Here proof would come in reality, we'll skip it by sorry for now.
  sorry
}

end NUMINAMATH_GPT_area_of_polygon_l1035_103545


namespace NUMINAMATH_GPT_athletes_meet_second_time_at_l1035_103564

-- Define the conditions given in the problem
def distance_AB : ℕ := 110

def man_uphill_speed : ℕ := 3
def man_downhill_speed : ℕ := 5

def woman_uphill_speed : ℕ := 2
def woman_downhill_speed : ℕ := 3

-- Define the times for the athletes' round trips
def man_round_trip_time : ℚ := (distance_AB / man_uphill_speed) + (distance_AB / man_downhill_speed)
def woman_round_trip_time : ℚ := (distance_AB / woman_uphill_speed) + (distance_AB / woman_downhill_speed)

-- Lean statement for the proof
theorem athletes_meet_second_time_at :
  ∀ (t : ℚ), t = lcm (man_round_trip_time) (woman_round_trip_time) →
  ∃ d : ℚ, d = 330 / 7 := 
by sorry

end NUMINAMATH_GPT_athletes_meet_second_time_at_l1035_103564


namespace NUMINAMATH_GPT_log_sum_range_l1035_103599

theorem log_sum_range (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hx_ne_one : x ≠ 1) (hy_ne_one : y ≠ 1) :
  (Real.log y / Real.log x + Real.log x / Real.log y) ∈ Set.union (Set.Iic (-2)) (Set.Ici 2) :=
sorry

end NUMINAMATH_GPT_log_sum_range_l1035_103599


namespace NUMINAMATH_GPT_tangent_line_eq_a1_max_value_f_a_gt_1_div_5_l1035_103532

noncomputable def f (a b x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + b * x + 1)

noncomputable def f_prime (a b x : ℝ) : ℝ := 
  -Real.exp (-x) * (a * x^2 + b * x + 1) + Real.exp (-x) * (2 * a * x + b)

theorem tangent_line_eq_a1 (b : ℝ) (h : f_prime 1 b (-1) = 0) : 
  ∃ m q, m = 1 ∧ q = 1 ∧ ∀ y, y = f 1 b 0 + m * y := sorry

theorem max_value_f_a_gt_1_div_5 (a b : ℝ) 
  (h_gt : a > 1/5) 
  (h_fp_eq : f_prime a b (-1) = 0)
  (h_max : ∀ x, -1 ≤ x ∧ x ≤ 1 → f a b x ≤ 4 * Real.exp 1) : 
  a = (24 * Real.exp 2 - 9) / 15 ∧ b = (12 * Real.exp 2 - 2) / 5 := sorry

end NUMINAMATH_GPT_tangent_line_eq_a1_max_value_f_a_gt_1_div_5_l1035_103532


namespace NUMINAMATH_GPT_rachel_reading_homework_l1035_103537

theorem rachel_reading_homework (math_hw : ℕ) (additional_reading_hw : ℕ) (total_reading_hw : ℕ) 
  (h1 : math_hw = 8) (h2 : additional_reading_hw = 6) (h3 : total_reading_hw = math_hw + additional_reading_hw) :
  total_reading_hw = 14 :=
sorry

end NUMINAMATH_GPT_rachel_reading_homework_l1035_103537


namespace NUMINAMATH_GPT_P_plus_Q_eq_14_l1035_103554

variable (P Q : Nat)

-- Conditions:
axiom single_digit_P : P < 10
axiom single_digit_Q : Q < 10
axiom three_P_ends_7 : 3 * P % 10 = 7
axiom two_Q_ends_0 : 2 * Q % 10 = 0

theorem P_plus_Q_eq_14 : P + Q = 14 :=
by
  sorry

end NUMINAMATH_GPT_P_plus_Q_eq_14_l1035_103554


namespace NUMINAMATH_GPT_necessary_condition_l1035_103515

theorem necessary_condition (A B C D : Prop) (h1 : A > B → C < D) : A > B → C < D := by
  exact h1 -- This is just a placeholder for the actual hypothesis, a required assumption in our initial problem statement

end NUMINAMATH_GPT_necessary_condition_l1035_103515


namespace NUMINAMATH_GPT_lemon_pie_degrees_l1035_103573

-- Defining the constants
def total_students : ℕ := 45
def chocolate_pie : ℕ := 15
def apple_pie : ℕ := 10
def blueberry_pie : ℕ := 9

-- Defining the remaining students
def remaining_students := total_students - (chocolate_pie + apple_pie + blueberry_pie)

-- Half of the remaining students prefer cherry pie and half prefer lemon pie
def students_prefer_cherry : ℕ := remaining_students / 2
def students_prefer_lemon : ℕ := remaining_students / 2

-- Defining the degree measure function
def degrees (students : ℕ) := (students * 360) / total_students

-- Proof statement
theorem lemon_pie_degrees : degrees students_prefer_lemon = 48 := by
  sorry  -- proof omitted

end NUMINAMATH_GPT_lemon_pie_degrees_l1035_103573


namespace NUMINAMATH_GPT_number_of_clips_after_k_steps_l1035_103505

theorem number_of_clips_after_k_steps (k : ℕ) : 
  ∃ (c : ℕ), c = 2^(k-1) + 1 :=
by sorry

end NUMINAMATH_GPT_number_of_clips_after_k_steps_l1035_103505


namespace NUMINAMATH_GPT_seq_bounded_l1035_103563

def digit_product (n : ℕ) : ℕ :=
  n.digits 10 |>.prod

def a_seq (a : ℕ → ℕ) (m : ℕ) : Prop :=
  a 0 = m ∧ (∀ n, a (n + 1) = a n + digit_product (a n))

theorem seq_bounded (m : ℕ) : ∃ B, ∀ n, a_seq a m → a n < B :=
by sorry

end NUMINAMATH_GPT_seq_bounded_l1035_103563


namespace NUMINAMATH_GPT_min_value_prime_factorization_l1035_103502

/-- Let x and y be positive integers and assume 5 * x ^ 7 = 13 * y ^ 11.
  If x has a prime factorization of the form a ^ c * b ^ d, then the minimum possible value of a + b + c + d is 31. -/
theorem min_value_prime_factorization (x y a b c d : ℕ) (hx_pos : x > 0) (hy_pos: y > 0) (ha_pos : a > 0) (hb_pos : b > 0) (hc_pos: c > 0) (hd_pos: d > 0)
    (h_eq : 5 * x ^ 7 = 13 * y ^ 11) (h_fact : x = a^c * b^d) : a + b + c + d = 31 :=
by
  sorry

end NUMINAMATH_GPT_min_value_prime_factorization_l1035_103502


namespace NUMINAMATH_GPT_ratio_B_to_A_l1035_103513

-- Definitions for conditions
def w_B : ℕ := 275 -- weight of element B in grams
def w_X : ℕ := 330 -- total weight of compound X in grams

-- Statement to prove
theorem ratio_B_to_A : (w_B:ℚ) / (w_X - w_B) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_B_to_A_l1035_103513


namespace NUMINAMATH_GPT_monotonic_intervals_and_extreme_values_of_f_f_g_inequality_sum_of_x1_x2_l1035_103551

noncomputable def f (x : ℝ) := Real.exp x - (1 / 2) * x^2 - x - 1
noncomputable def f' (x : ℝ) := Real.exp x - x - 1
noncomputable def f'' (x : ℝ) := Real.exp x - 1
noncomputable def g (x : ℝ) := -f (-x)

-- Proof of (I)
theorem monotonic_intervals_and_extreme_values_of_f' :
  f' 0 = 0 ∧ (∀ x < 0, f'' x < 0 ∧ f' x > f' 0) ∧ (∀ x > 0, f'' x > 0 ∧ f' x > f' 0) := 
sorry

-- Proof of (II)
theorem f_g_inequality (x : ℝ) (hx : x > 0) : f x > g x :=
sorry

-- Proof of (III)
theorem sum_of_x1_x2 (x1 x2 : ℝ) (h : f x1 + f x2 = 0) (hne : x1 ≠ x2) : x1 + x2 < 0 := 
sorry

end NUMINAMATH_GPT_monotonic_intervals_and_extreme_values_of_f_f_g_inequality_sum_of_x1_x2_l1035_103551


namespace NUMINAMATH_GPT_max_value_ab_bc_cd_l1035_103511

theorem max_value_ab_bc_cd (a b c d : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0)
  (h_sum : a + b + c + d = 120) : ab + bc + cd ≤ 3600 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_value_ab_bc_cd_l1035_103511


namespace NUMINAMATH_GPT_f_positive_for_all_x_f_increasing_solution_set_inequality_l1035_103591

namespace ProofProblem

-- Define the function f and its properties
def f : ℝ → ℝ := sorry

axiom f_zero_ne_zero : f 0 ≠ 0
axiom f_one_eq_two : f 1 = 2
axiom f_pos_when_pos : ∀ x : ℝ, x > 0 → f x > 1
axiom f_add_mul : ∀ a b : ℝ, f (a + b) = f a * f b

-- Problem 1: Prove that f(x) > 0 for all x ∈ ℝ
theorem f_positive_for_all_x : ∀ x : ℝ, f x > 0 := sorry

-- Problem 2: Prove that f(x) is increasing on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

-- Problem 3: Find the solution set of the inequality f(3-2x) > 4
theorem solution_set_inequality : { x : ℝ | f (3 - 2 * x) > 4 } = { x | x < 1 / 2 } := sorry

end ProofProblem

end NUMINAMATH_GPT_f_positive_for_all_x_f_increasing_solution_set_inequality_l1035_103591


namespace NUMINAMATH_GPT_walnut_trees_planted_l1035_103580

-- Define the initial number of walnut trees
def initial_walnut_trees : ℕ := 22

-- Define the total number of walnut trees after planting
def total_walnut_trees_after : ℕ := 55

-- The Lean statement to prove the number of walnut trees planted today
theorem walnut_trees_planted : (total_walnut_trees_after - initial_walnut_trees = 33) :=
by
  sorry

end NUMINAMATH_GPT_walnut_trees_planted_l1035_103580


namespace NUMINAMATH_GPT_total_cookies_l1035_103598

def mona_cookies : ℕ := 20
def jasmine_cookies : ℕ := mona_cookies - 5
def rachel_cookies : ℕ := jasmine_cookies + 10

theorem total_cookies : mona_cookies + jasmine_cookies + rachel_cookies = 60 := 
by
  have h1 : jasmine_cookies = 15 := by sorry
  have h2 : rachel_cookies = 25 := by sorry
  have h3 : mona_cookies = 20 := by sorry
  sorry

end NUMINAMATH_GPT_total_cookies_l1035_103598


namespace NUMINAMATH_GPT_max_subset_count_l1035_103514

-- Define the problem conditions in Lean 4
def is_valid_subset (T : Finset ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬ (a + b) % 5 = 0

theorem max_subset_count :
  ∃ (T : Finset ℕ), (is_valid_subset T) ∧ T.card = 18 := by
  sorry

end NUMINAMATH_GPT_max_subset_count_l1035_103514


namespace NUMINAMATH_GPT_derek_history_test_l1035_103509

theorem derek_history_test :
  let ancient_questions := 20
  let medieval_questions := 25
  let modern_questions := 35
  let total_questions := ancient_questions + medieval_questions + modern_questions

  let derek_ancient_correct := 0.60 * ancient_questions
  let derek_medieval_correct := 0.56 * medieval_questions
  let derek_modern_correct := 0.70 * modern_questions

  let derek_total_correct := derek_ancient_correct + derek_medieval_correct + derek_modern_correct

  let passing_score := 0.65 * total_questions
  (derek_total_correct < passing_score) →
  passing_score - derek_total_correct = 2
  := by
  sorry

end NUMINAMATH_GPT_derek_history_test_l1035_103509


namespace NUMINAMATH_GPT_min_correct_answers_l1035_103519

theorem min_correct_answers (x : ℕ) : 
  (∃ x, 0 ≤ x ∧ x ≤ 20 ∧ 5 * x - (20 - x) ≥ 88) :=
sorry

end NUMINAMATH_GPT_min_correct_answers_l1035_103519


namespace NUMINAMATH_GPT_triangle_side_length_difference_l1035_103525

theorem triangle_side_length_difference :
  (∃ x : ℤ, 3 ≤ x ∧ x ≤ 17 ∧ ∀ a b c : ℤ, x + 8 > 10 ∧ x + 10 > 8 ∧ 8 + 10 > x) →
  (17 - 3 = 14) :=
by
  intros
  sorry

end NUMINAMATH_GPT_triangle_side_length_difference_l1035_103525


namespace NUMINAMATH_GPT_squared_difference_of_roots_l1035_103527

theorem squared_difference_of_roots:
  ∀ (Φ φ : ℝ), (∀ x : ℝ, x^2 = 2*x + 1 ↔ (x = Φ ∨ x = φ)) ∧ Φ ≠ φ → (Φ - φ)^2 = 8 :=
by
  intros Φ φ h
  sorry

end NUMINAMATH_GPT_squared_difference_of_roots_l1035_103527


namespace NUMINAMATH_GPT_polygon_sides_l1035_103594

theorem polygon_sides (x : ℝ) (hx : 0 < x) (h : x + 5 * x = 180) : 12 = 360 / x :=
by {
  -- Steps explaining: x should be the exterior angle then proof follows.
  sorry
}

end NUMINAMATH_GPT_polygon_sides_l1035_103594


namespace NUMINAMATH_GPT_find_c_d_l1035_103572

theorem find_c_d (C D : ℤ) (h1 : 3 * C - 4 * D = 18) (h2 : C = 2 * D - 5) :
  C = 28 ∧ D = 33 / 2 := by
sorry

end NUMINAMATH_GPT_find_c_d_l1035_103572


namespace NUMINAMATH_GPT_convert_to_rectangular_form_l1035_103561

noncomputable def rectangular_form (z : ℂ) : ℂ :=
  let e := Complex.exp (13 * Real.pi * Complex.I / 6)
  3 * e

theorem convert_to_rectangular_form :
  rectangular_form (3 * Complex.exp (13 * Real.pi * Complex.I / 6)) = (3 * (Complex.cos (Real.pi / 6)) + 3 * Complex.I * (Complex.sin (Real.pi / 6))) :=
by
  sorry

end NUMINAMATH_GPT_convert_to_rectangular_form_l1035_103561


namespace NUMINAMATH_GPT_sanghyeon_questions_l1035_103528

variable (S : ℕ)

theorem sanghyeon_questions (h1 : S + (S + 5) = 43) : S = 19 :=
by
    sorry

end NUMINAMATH_GPT_sanghyeon_questions_l1035_103528


namespace NUMINAMATH_GPT_smallest_marbles_l1035_103577

theorem smallest_marbles
  : ∃ n : ℕ, ((n % 8 = 5) ∧ (n % 7 = 2) ∧ (n = 37) ∧ (37 % 9 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_marbles_l1035_103577


namespace NUMINAMATH_GPT_problem_l1035_103596

noncomputable def f : ℝ → ℝ := sorry

theorem problem (x : ℝ) (h : ∀ x : ℝ, f (4 * x) = 4) : f (2 * x) = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1035_103596


namespace NUMINAMATH_GPT_max_electronic_thermometers_l1035_103517

-- Definitions
def budget : ℕ := 300
def price_mercury : ℕ := 3
def price_electronic : ℕ := 10
def total_students : ℕ := 53

-- The theorem statement
theorem max_electronic_thermometers : 
  (∃ x : ℕ, x ≤ total_students ∧ 10 * x + 3 * (total_students - x) ≤ budget ∧ 
            ∀ y : ℕ, y ≤ total_students ∧ 10 * y + 3 * (total_students - y) ≤ budget → y ≤ x) :=
sorry

end NUMINAMATH_GPT_max_electronic_thermometers_l1035_103517
