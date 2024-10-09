import Mathlib

namespace time_to_cross_pole_is_correct_l625_62534

-- Define the conversion factor to convert km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : ℕ) : ℕ := speed_km_per_hr * 1000 / 3600

-- Define the speed of the train in m/s
def train_speed_m_per_s : ℕ := km_per_hr_to_m_per_s 216

-- Define the length of the train
def train_length_m : ℕ := 480

-- Define the time to cross an electric pole
def time_to_cross_pole : ℕ := train_length_m / train_speed_m_per_s

-- Theorem stating that the computed time to cross the pole is 8 seconds
theorem time_to_cross_pole_is_correct :
  time_to_cross_pole = 8 := by
  sorry

end time_to_cross_pole_is_correct_l625_62534


namespace trig_inequality_l625_62577

theorem trig_inequality (theta : ℝ) (h1 : Real.pi / 4 < theta) (h2 : theta < Real.pi / 2) : 
  Real.cos theta < Real.sin theta ∧ Real.sin theta < Real.tan theta :=
sorry

end trig_inequality_l625_62577


namespace geom_sequence_50th_term_l625_62555

theorem geom_sequence_50th_term (a a_2 : ℤ) (n : ℕ) (r : ℤ) (h1 : a = 8) (h2 : a_2 = -16) (h3 : r = a_2 / a) (h4 : n = 50) :
  a * r^(n-1) = -8 * 2^49 :=
by
  sorry

end geom_sequence_50th_term_l625_62555


namespace sharon_trip_distance_l625_62540

noncomputable section

variable (x : ℝ)

def sharon_original_speed (x : ℝ) := x / 200

def sharon_reduced_speed (x : ℝ) := (x / 200) - 1 / 2

def time_before_traffic (x : ℝ) := (x / 2) / (sharon_original_speed x)

def time_after_traffic (x : ℝ) := (x / 2) / (sharon_reduced_speed x)

theorem sharon_trip_distance : 
  (time_before_traffic x) + (time_after_traffic x) = 300 → x = 200 := 
by
  sorry

end sharon_trip_distance_l625_62540


namespace value_range_a_l625_62559

theorem value_range_a (a : ℝ) :
  (∀ (x : ℝ), |x + 2| * |x - 3| ≥ 4 / (a - 1)) ↔ (a < 1 ∨ a = 3) :=
by
  sorry

end value_range_a_l625_62559


namespace jack_initial_yen_l625_62589

theorem jack_initial_yen 
  (pounds yen_per_pound euros pounds_per_euro total_yen : ℕ)
  (h₁ : pounds = 42)
  (h₂ : euros = 11)
  (h₃ : pounds_per_euro = 2)
  (h₄ : yen_per_pound = 100)
  (h₅ : total_yen = 9400) : 
  ∃ initial_yen : ℕ, initial_yen = 3000 :=
by
  sorry

end jack_initial_yen_l625_62589


namespace average_licks_l625_62579

theorem average_licks 
  (Dan_licks : ℕ := 58)
  (Michael_licks : ℕ := 63)
  (Sam_licks : ℕ := 70)
  (David_licks : ℕ := 70)
  (Lance_licks : ℕ := 39) :
  (Dan_licks + Michael_licks + Sam_licks + David_licks + Lance_licks) / 5 = 60 := 
sorry

end average_licks_l625_62579


namespace geometric_sequence_a3_q_l625_62598

theorem geometric_sequence_a3_q (a_5 a_4 a_3 a_2 a_1 : ℝ) (q : ℝ) :
  a_5 - a_1 = 15 →
  a_4 - a_2 = 6 →
  (q = 2 ∧ a_3 = 4) ∨ (q = 1/2 ∧ a_3 = -4) :=
by
  sorry

end geometric_sequence_a3_q_l625_62598


namespace find_pairs_l625_62500

noncomputable def diamond (a b : ℝ) : ℝ :=
  a^2 * b^2 - a^3 * b - a * b^3

theorem find_pairs (x y : ℝ) :
  diamond x y = diamond y x ↔
  x = 0 ∨ y = 0 ∨ x = y ∨ x = -y :=
by
  sorry

end find_pairs_l625_62500


namespace find_x_l625_62528

variables {x y z d e f : ℝ}
variables (h1 : xy / (x + 2 * y) = d)
variables (h2 : xz / (2 * x + z) = e)
variables (h3 : yz / (y + 2 * z) = f)

theorem find_x :
  x = 3 * d * e * f / (d * e - 2 * d * f + e * f) :=
sorry

end find_x_l625_62528


namespace joel_strawberries_area_l625_62505

-- Define the conditions
def garden_area : ℕ := 64
def fruit_fraction : ℚ := 1 / 2
def strawberry_fraction : ℚ := 1 / 4

-- Define the desired conclusion
def strawberries_area : ℕ := 8

-- State the theorem
theorem joel_strawberries_area 
  (H1 : garden_area = 64) 
  (H2 : fruit_fraction = 1 / 2) 
  (H3 : strawberry_fraction = 1 / 4)
  : garden_area * fruit_fraction * strawberry_fraction = strawberries_area := 
sorry

end joel_strawberries_area_l625_62505


namespace triangle_at_most_one_obtuse_angle_l625_62539

theorem triangle_at_most_one_obtuse_angle :
  (∀ (α β γ : ℝ), α + β + γ = 180 → α ≤ 90 ∨ β ≤ 90 ∨ γ ≤ 90) ↔
  ¬ (∃ (α β γ : ℝ), α + β + γ = 180 ∧ α > 90 ∧ β > 90) :=
by
  sorry

end triangle_at_most_one_obtuse_angle_l625_62539


namespace marble_box_l625_62507

theorem marble_box (T: ℕ) 
  (h_white: (1 / 6) * T = T / 6)
  (h_green: (1 / 5) * T = T / 5)
  (h_red_blue: (19 / 30) * T = 19 * T / 30)
  (h_sum: (T / 6) + (T / 5) + (19 * T / 30) = T): 
  ∃ k : ℕ, T = 30 * k ∧ k ≥ 1 :=
by
  sorry

end marble_box_l625_62507


namespace MeganMarkers_l625_62533

def initialMarkers : Nat := 217
def additionalMarkers : Nat := 109
def totalMarkers : Nat := initialMarkers + additionalMarkers

theorem MeganMarkers : totalMarkers = 326 := by
    sorry

end MeganMarkers_l625_62533


namespace jason_needs_201_grams_l625_62576

-- Define the conditions
def rectangular_patch_length : ℕ := 6
def rectangular_patch_width : ℕ := 7
def square_path_side_length : ℕ := 5
def sand_per_square_inch : ℕ := 3

-- Define the areas
def rectangular_patch_area : ℕ := rectangular_patch_length * rectangular_patch_width
def square_path_area : ℕ := square_path_side_length * square_path_side_length

-- Define the total area
def total_area : ℕ := rectangular_patch_area + square_path_area

-- Define the total sand needed
def total_sand_needed : ℕ := total_area * sand_per_square_inch

-- State the proof problem
theorem jason_needs_201_grams : total_sand_needed = 201 := by
    sorry

end jason_needs_201_grams_l625_62576


namespace grover_total_profit_is_15_l625_62523

theorem grover_total_profit_is_15 
  (boxes : ℕ) 
  (masks_per_box : ℕ) 
  (price_per_mask : ℝ) 
  (cost_of_boxes : ℝ) 
  (total_profit : ℝ)
  (hb : boxes = 3)
  (hm : masks_per_box = 20)
  (hp : price_per_mask = 0.5)
  (hc : cost_of_boxes = 15)
  (htotal : total_profit = (boxes * masks_per_box) * price_per_mask - cost_of_boxes) :
  total_profit = 15 :=
sorry

end grover_total_profit_is_15_l625_62523


namespace problem_statement_l625_62545

open Real

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * cos θ, sin θ)

theorem problem_statement (A B : ℝ × ℝ) 
  (θA θB : ℝ) 
  (hA : A = curve_C θA) 
  (hB : B = curve_C θB) 
  (h_perpendicular : θB = θA + π / 2) :
  (1 / (A.1 ^ 2 + A.2 ^ 2)) + (1 / (B.1 ^ 2 + B.2 ^ 2)) = 5 / 4 := by
  sorry

end problem_statement_l625_62545


namespace dina_dolls_count_l625_62557

-- Define the conditions
variable (Ivy_dolls : ℕ)
variable (Collectors_Ivy_dolls : ℕ := 20)
variable (Dina_dolls : ℕ)

-- Condition: Ivy has 2/3 of her dolls as collectors editions
def collectors_edition_condition : Prop := (2 / 3 : ℝ) * Ivy_dolls = Collectors_Ivy_dolls

-- Condition: Dina has twice as many dolls as Ivy
def dina_ivy_dolls_relationship : Prop := Dina_dolls = 2 * Ivy_dolls

-- Theorem: Prove that Dina has 60 dolls
theorem dina_dolls_count : collectors_edition_condition Ivy_dolls ∧ dina_ivy_dolls_relationship Ivy_dolls Dina_dolls → Dina_dolls = 60 := by
  sorry

end dina_dolls_count_l625_62557


namespace pats_password_length_l625_62535

-- Definitions based on conditions
def num_lowercase_letters := 8
def num_uppercase_numbers := num_lowercase_letters / 2
def num_symbols := 2

-- Translate the math proof problem to Lean 4 statement
theorem pats_password_length : 
  num_lowercase_letters + num_uppercase_numbers + num_symbols = 14 := by
  sorry

end pats_password_length_l625_62535


namespace parallel_line_eq_l625_62574

theorem parallel_line_eq (a b c : ℝ) (p1 p2 : ℝ) :
  (∃ m b1 b2, 3 * a + 6 * b * p1 = 12 ∧ p2 = - (1 / 2) * p1 + b1 ∧
    - (1 / 2) * p1 - m * p1 = b2) → 
    (∃ b', p2 = - (1 / 2) * p1 + b' ∧ b' = 0) := 
sorry

end parallel_line_eq_l625_62574


namespace discount_amount_l625_62567

def tshirt_cost : ℕ := 30
def backpack_cost : ℕ := 10
def blue_cap_cost : ℕ := 5
def total_spent : ℕ := 43

theorem discount_amount : (tshirt_cost + backpack_cost + blue_cap_cost) - total_spent = 2 := by
  sorry

end discount_amount_l625_62567


namespace triangle_minimum_perimeter_l625_62556

/--
In a triangle ABC where sides have integer lengths such that no two sides are equal, let ω be a circle with its center at the incenter of ΔABC. Suppose one excircle is tangent to AB and internally tangent to ω, while excircles tangent to AC and BC are externally tangent to ω.
Prove that the minimum possible perimeter of ΔABC is 12.
-/
theorem triangle_minimum_perimeter {a b c : ℕ} (h1 : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
    (h2 : ∀ (r rA rB rC s : ℝ),
      rA = r * s / (s - a) → rB = r * s / (s - b) → rC = r * s / (s - c) →
      r + rA = rB ∧ r + rA = rC) :
  a + b + c = 12 :=
sorry

end triangle_minimum_perimeter_l625_62556


namespace find_f3_l625_62512

theorem find_f3 (f : ℚ → ℚ)
  (h : ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + (3 * f x) / x = x^3) :
  f 3 = 7753 / 729 :=
sorry

end find_f3_l625_62512


namespace ordinary_eq_of_curve_l625_62587

theorem ordinary_eq_of_curve 
  (t : ℝ) (x : ℝ) (y : ℝ)
  (ht : t > 0) 
  (hx : x = Real.sqrt t - 1 / Real.sqrt t)
  (hy : y = 3 * (t + 1 / t)) :
  3 * x^2 - y + 6 = 0 ∧ y ≥ 6 :=
sorry

end ordinary_eq_of_curve_l625_62587


namespace sufficient_drivers_and_ivan_petrovich_departure_l625_62573

/--
One-way trip duration in minutes.
-/
def one_way_trip_duration : ℕ := 160

/--
Round trip duration in minutes is twice the one-way trip duration.
-/
def round_trip_duration : ℕ := 2 * one_way_trip_duration

/--
Rest duration after a trip in minutes.
-/
def rest_duration : ℕ := 60

/--
Departure times and rest periods for drivers A, B, C starting initially, 
with additional driver D.
-/
def driver_a_return_time : ℕ := 12 * 60 + 40  -- 12:40 in minutes
def driver_a_next_departure : ℕ := driver_a_return_time + rest_duration  -- 13:40 in minutes
def driver_d_departure : ℕ := 13 * 60 + 5  -- 13:05 in minutes

/--
A proof that 4 drivers are sufficient and Ivan Petrovich departs at 10:40 AM.
-/
theorem sufficient_drivers_and_ivan_petrovich_departure :
  (4 = 4) ∧ (10 * 60 + 40 = 10 * 60 + 40) :=
by
  sorry

end sufficient_drivers_and_ivan_petrovich_departure_l625_62573


namespace subtract_decimal_numbers_l625_62549

theorem subtract_decimal_numbers : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_decimal_numbers_l625_62549


namespace negation_of_proposition_l625_62531

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ ∃ x : ℝ, Real.exp x ≤ x^2 :=
by sorry

end negation_of_proposition_l625_62531


namespace sum_of_squares_expr_l625_62584

theorem sum_of_squares_expr : 
  (23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2) = 288 := 
by
  sorry

end sum_of_squares_expr_l625_62584


namespace range_of_b_l625_62519

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x - (1 / 2) * a * x^2 - 2 * x

theorem range_of_b (a x b : ℝ) (ha : -1 ≤ a) (ha' : a < 0) (hx : 0 < x) (hx' : x ≤ 1) 
  (h : f x a < b) : -3 / 2 < b := 
sorry

end range_of_b_l625_62519


namespace root_of_equation_value_l625_62595

theorem root_of_equation_value (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : 2 * m^2 - 4 * m + 5 = 11 := 
by
  sorry

end root_of_equation_value_l625_62595


namespace min_cards_needed_l625_62513

/-- 
On a table, there are five types of number cards: 1, 3, 5, 7, and 9, with 30 cards of each type. 
Prove that the minimum number of cards required to ensure that the sum of the drawn card numbers 
can represent all integers from 1 to 200 is 26.
-/
theorem min_cards_needed : ∀ (cards_1 cards_3 cards_5 cards_7 cards_9 : ℕ), 
  cards_1 = 30 → cards_3 = 30 → cards_5 = 30 → cards_7 = 30 → cards_9 = 30 → 
  ∃ n, (n = 26) ∧ 
  (∀ k, 1 ≤ k ∧ k ≤ 200 → 
    ∃ a b c d e, 
      a ≤ cards_1 ∧ b ≤ cards_3 ∧ c ≤ cards_5 ∧ d ≤ cards_7 ∧ e ≤ cards_9 ∧ 
      k = a * 1 + b * 3 + c * 5 + d * 7 + e * 9) :=
by {
  sorry
}

end min_cards_needed_l625_62513


namespace problem_statement_l625_62510

theorem problem_statement (m : ℝ) (h : m^2 - m - 2 = 0) : m^2 - m + 2023 = 2025 :=
sorry

end problem_statement_l625_62510


namespace average_six_conseq_ints_l625_62552

theorem average_six_conseq_ints (c d : ℝ) (h₁ : d = c + 2.5) :
  (d - 2 + d - 1 + d + d + 1 + d + 2 + d + 3) / 6 = c + 3 :=
by
  sorry

end average_six_conseq_ints_l625_62552


namespace simplify_expression_l625_62543

theorem simplify_expression : 
  (2 * Real.sqrt 7) / (Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 6) = 
  (2 * Real.sqrt 14 + 8 * Real.sqrt 21 + 2 * Real.sqrt 42 + 8 * Real.sqrt 63) / 23 :=
by
  sorry

end simplify_expression_l625_62543


namespace base7_65432_to_dec_is_16340_l625_62537

def base7_to_dec (n : ℕ) : ℕ :=
  6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0

theorem base7_65432_to_dec_is_16340 : base7_to_dec 65432 = 16340 :=
by
  sorry

end base7_65432_to_dec_is_16340_l625_62537


namespace krishan_money_l625_62551

theorem krishan_money 
  (R G K : ℝ)
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (h3 : R = 490) : K = 2890 :=
sorry

end krishan_money_l625_62551


namespace train_speed_l625_62529

theorem train_speed (len_train len_bridge time : ℝ)
  (h1 : len_train = 100)
  (h2 : len_bridge = 180)
  (h3 : time = 27.997760179185665) :
  (len_train + len_bridge) / time * 3.6 = 36 :=
by
  sorry

end train_speed_l625_62529


namespace minimum_value_expression_l625_62518

noncomputable def minimum_value (a b : ℝ) := (1 / (2 * |a|)) + (|a| / b)

theorem minimum_value_expression
  (a : ℝ) (b : ℝ) (h1 : a + b = 2) (h2 : b > 0) :
  ∃ (min_val : ℝ), min_val = 3 / 4 ∧ ∀ (a b : ℝ), a + b = 2 → b > 0 → minimum_value a b ≥ min_val :=
sorry

end minimum_value_expression_l625_62518


namespace trajectory_equation_find_m_value_l625_62591

def point (α : Type) := (α × α)
def fixed_points (α : Type) := point α

noncomputable def slopes (x y : ℝ) : ℝ := y / x

theorem trajectory_equation (x y : ℝ) (P : point ℝ) (A B : fixed_points ℝ)
  (k1 k2 : ℝ) (hk : k1 * k2 = -1/4) :
  A = (-2, 0) → B = (2, 0) →
  P = (x, y) → 
  slopes (x + 2) y * slopes (x - 2) y = -1/4 →
  (x^2 / 4) + y^2 = 1 :=
sorry

theorem find_m_value (m x₁ x₂ y₁ y₂ : ℝ) (k : ℝ) (hx : (4 * k^2) + 1 - m^2 > 0)
  (hroots_sum : x₁ + x₂ = -((8 * k * m) / ((4 * k^2) + 1)))
  (hroots_prod : x₁ * x₂ = (4 * m^2 - 4) / ((4 * k^2) + 1))
  (hperp : x₁ * x₂ + y₁ * y₂ = 0) :
  y₁ = k * x₁ + m → y₂ = k * x₂ + m →
  m^2 = 4/5 * (k^2 + 1) →
  m = 2 ∨ m = -2 :=
sorry

end trajectory_equation_find_m_value_l625_62591


namespace max_time_digit_sum_l625_62558

-- Define the conditions
def is_valid_time (h m : ℕ) : Prop :=
  (0 ≤ h ∧ h < 24) ∧ (0 ≤ m ∧ m < 60)

-- Define the function to calculate the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  n % 10 + n / 10

-- Define the function to calculate the sum of digits in the time display
def time_digit_sum (h m : ℕ) : ℕ :=
  digit_sum h + digit_sum m

-- The theorem to prove
theorem max_time_digit_sum : ∀ (h m : ℕ),
  is_valid_time h m → time_digit_sum h m ≤ 24 :=
by {
  sorry
}

end max_time_digit_sum_l625_62558


namespace rectangle_diagonals_equiv_positive_even_prime_equiv_l625_62544

-- Definitions based on problem statement (1)
def is_rectangle (q : Quadrilateral) : Prop := sorry -- "q is a rectangle"
def diagonals_equal_and_bisect (q : Quadrilateral) : Prop := sorry -- "the diagonals of q are equal and bisect each other"

-- Problem statement (1)
theorem rectangle_diagonals_equiv (q : Quadrilateral) :
  (is_rectangle q → diagonals_equal_and_bisect q) ∧
  (diagonals_equal_and_bisect q → is_rectangle q) ∧
  (¬ is_rectangle q → ¬ diagonals_equal_and_bisect q) ∧
  (¬ diagonals_equal_and_bisect q → ¬ is_rectangle q) :=
sorry

-- Definitions based on problem statement (2)
def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0
def is_prime (n : ℕ) : Prop := sorry -- "n is a prime number"

-- Problem statement (2)
theorem positive_even_prime_equiv (n : ℕ) :
  (is_positive_even n → ¬ is_prime n) ∧
  ((¬ is_prime n → is_positive_even n) = False) ∧
  ((¬ is_positive_even n → is_prime n) = False) ∧
  ((is_prime n → ¬ is_positive_even n) = False) :=
sorry

end rectangle_diagonals_equiv_positive_even_prime_equiv_l625_62544


namespace waiter_earnings_l625_62525

theorem waiter_earnings (total_customers : ℕ) (no_tip_customers : ℕ) (tip_per_customer : ℕ)
  (h1 : total_customers = 10)
  (h2 : no_tip_customers = 5)
  (h3 : tip_per_customer = 3) :
  (total_customers - no_tip_customers) * tip_per_customer = 15 :=
by sorry

end waiter_earnings_l625_62525


namespace tangent_line_eq_l625_62560

def perp_eq (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 - 1

theorem tangent_line_eq (x y : ℝ) (h1 : perp_eq x y) (h2 : y = curve x) : 
  ∃ (m : ℝ), y = -3 * x + m ∧ y = -3 * x - 2 := 
sorry

end tangent_line_eq_l625_62560


namespace valid_arrangement_after_removal_l625_62503

theorem valid_arrangement_after_removal (n : ℕ) (m : ℕ → ℕ) :
  (∀ i j, i ≠ j → m i ≠ m j → ¬ (i < n ∧ j < n))
  → (∀ i, i < n → m i ≥ m (i + 1))
  → ∃ (m' : ℕ → ℕ), (∀ i, i < n.pred → m' i = m (i + 1) - 1 ∨ m' i = m (i + 1))
    ∧ (∀ i, m' i ≥ m' (i + 1))
    ∧ (∀ i j, i ≠ j → i < n.pred → j < n.pred → ¬ (m' i = m' j ∧ m' i = m (i + 1))) := sorry

end valid_arrangement_after_removal_l625_62503


namespace regular_polygon_sides_l625_62508

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 12) : n = 30 := 
by
  sorry

end regular_polygon_sides_l625_62508


namespace maximum_pizzas_baked_on_Friday_l625_62590

def george_bakes := 
  let total_pizzas : ℕ := 1000
  let monday_pizzas := total_pizzas * 7 / 10
  let tuesday_pizzas := if monday_pizzas * 4 / 5 < monday_pizzas * 9 / 10 
                        then monday_pizzas * 4 / 5 
                        else monday_pizzas * 9 / 10
  let wednesday_pizzas := if tuesday_pizzas * 4 / 5 < tuesday_pizzas * 9 / 10 
                          then tuesday_pizzas * 4 / 5 
                          else tuesday_pizzas * 9 / 10
  let thursday_pizzas := if wednesday_pizzas * 4 / 5 < wednesday_pizzas * 9 / 10 
                         then wednesday_pizzas * 4 / 5 
                         else wednesday_pizzas * 9 / 10
  let friday_pizzas := if thursday_pizzas * 4 / 5 < thursday_pizzas * 9 / 10 
                       then thursday_pizzas * 4 / 5 
                       else thursday_pizzas * 9 / 10
  friday_pizzas

theorem maximum_pizzas_baked_on_Friday : george_bakes = 2 := by
  sorry

end maximum_pizzas_baked_on_Friday_l625_62590


namespace expression_equal_a_five_l625_62563

noncomputable def a : ℕ := sorry

theorem expression_equal_a_five (a : ℕ) : (a^4 * a) = a^5 := by
  sorry

end expression_equal_a_five_l625_62563


namespace total_teaching_time_l625_62585

def teaching_times :=
  let eduardo_math_time := 3 * 60
  let eduardo_science_time := 4 * 90
  let eduardo_history_time := 2 * 120
  let total_eduardo_time := eduardo_math_time + eduardo_science_time + eduardo_history_time

  let frankie_math_time := 2 * (3 * 60)
  let frankie_science_time := 2 * (4 * 90)
  let frankie_history_time := 2 * (2 * 120)
  let total_frankie_time := frankie_math_time + frankie_science_time + frankie_history_time

  let georgina_math_time := 3 * (3 * 80)
  let georgina_science_time := 3 * (4 * 100)
  let georgina_history_time := 3 * (2 * 150)
  let total_georgina_time := georgina_math_time + georgina_science_time + georgina_history_time

  total_eduardo_time + total_frankie_time + total_georgina_time

theorem total_teaching_time : teaching_times = 5160 := by
  -- calculations omitted
  sorry

end total_teaching_time_l625_62585


namespace f_neg_m_equals_neg_8_l625_62532

def f (x : ℝ) : ℝ := x^5 + x^3 + 1

theorem f_neg_m_equals_neg_8 (m : ℝ) (h : f m = 10) : f (-m) = -8 :=
by
  sorry

end f_neg_m_equals_neg_8_l625_62532


namespace num_of_integers_l625_62541

theorem num_of_integers (n : ℤ) (h : -1000 ≤ n ∧ n ≤ 1000) (h1 : 1 < 4 * n + 7) (h2 : 4 * n + 7 < 150) : 
  (∃ N : ℕ, N = 37) :=
by
  sorry

end num_of_integers_l625_62541


namespace sum_of_a_b_l625_62566

theorem sum_of_a_b (a b : ℝ) (h1 : a * b = 1) (h2 : (3 * a + 2 * b) * (3 * b + 2 * a) = 295) : a + b = 7 :=
by
  sorry

end sum_of_a_b_l625_62566


namespace last_digit_of_N_l625_62583

def sum_of_first_n_natural_numbers (N : ℕ) : ℕ :=
  N * (N + 1) / 2

theorem last_digit_of_N (N : ℕ) (h : sum_of_first_n_natural_numbers N = 3080) :
  N % 10 = 8 :=
by {
  sorry
}

end last_digit_of_N_l625_62583


namespace Jacob_has_48_graham_crackers_l625_62562

def marshmallows_initial := 6
def marshmallows_needed := 18
def marshmallows_total := marshmallows_initial + marshmallows_needed
def graham_crackers_per_smore := 2

def smores_total := marshmallows_total
def graham_crackers_total := smores_total * graham_crackers_per_smore

theorem Jacob_has_48_graham_crackers (h1 : marshmallows_initial = 6)
                                     (h2 : marshmallows_needed = 18)
                                     (h3 : graham_crackers_per_smore = 2)
                                     (h4 : marshmallows_total = marshmallows_initial + marshmallows_needed)
                                     (h5 : smores_total = marshmallows_total)
                                     (h6 : graham_crackers_total = smores_total * graham_crackers_per_smore) :
                                     graham_crackers_total = 48 :=
by
  sorry

end Jacob_has_48_graham_crackers_l625_62562


namespace range_of_m_l625_62522

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * m * x + m^2 - 1 = 0) → (-2 < x)) ↔ m > -1 :=
by
  sorry

end range_of_m_l625_62522


namespace expand_product_l625_62571

theorem expand_product (x : ℝ) :
  (2 * x^2 - 3 * x + 5) * (x^2 + 4 * x + 3) = 2 * x^4 + 5 * x^3 - x^2 + 11 * x + 15 :=
by
  -- Proof to be filled in
  sorry

end expand_product_l625_62571


namespace average_marks_l625_62597

variable (M P C : ℤ)

-- Conditions
axiom h1 : M + P = 50
axiom h2 : C = P + 20

-- Theorem statement
theorem average_marks : (M + C) / 2 = 35 := by
  sorry

end average_marks_l625_62597


namespace matching_times_l625_62592

noncomputable def chargeAtTime (t : Nat) : ℚ :=
  100 - t / 6

def isMatchingTime (hh mm : Nat) : Prop :=
  hh * 60 + mm = 100 - (hh * 60 + mm) / 6

theorem matching_times:
  isMatchingTime 4 52 ∨
  isMatchingTime 5 43 ∨
  isMatchingTime 6 35 ∨
  isMatchingTime 7 26 ∨
  isMatchingTime 9 9 :=
by
  repeat { sorry }

end matching_times_l625_62592


namespace combined_height_is_320_cm_l625_62553

-- Define Maria's height in inches
def Maria_height_in_inches : ℝ := 54

-- Define Ben's height in inches
def Ben_height_in_inches : ℝ := 72

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℝ := 2.54

-- Define the combined height of Maria and Ben in centimeters
def combined_height_in_cm : ℝ := (Maria_height_in_inches + Ben_height_in_inches) * inch_to_cm

-- State and prove that the combined height is 320.0 cm
theorem combined_height_is_320_cm : combined_height_in_cm = 320.0 := by
  sorry

end combined_height_is_320_cm_l625_62553


namespace value_of_sum_cubes_l625_62580

theorem value_of_sum_cubes (x : ℝ) (hx : x ≠ 0) (h : 47 = x^6 + (1 / x^6)) : (x^3 + (1 / x^3)) = 7 := 
by 
  sorry

end value_of_sum_cubes_l625_62580


namespace max_value_fraction_diff_l625_62502

noncomputable def max_fraction_diff (a b : ℝ) : ℝ :=
  1 / a - 1 / b

theorem max_value_fraction_diff (a b : ℝ) (ha : a > 0) (hb : b > 0) (hc : 4 * a - b ≥ 2) :
  max_fraction_diff a b ≤ 1 / 2 :=
by
  sorry

end max_value_fraction_diff_l625_62502


namespace jimin_yuna_difference_l625_62588

-- Definitions based on the conditions.
def seokjin_marbles : ℕ := 3
def yuna_marbles : ℕ := seokjin_marbles - 1
def jimin_marbles : ℕ := seokjin_marbles * 2

-- Theorem stating the problem we need to prove: the difference in marbles between Jimin and Yuna is 4.
theorem jimin_yuna_difference : jimin_marbles - yuna_marbles = 4 :=
by sorry

end jimin_yuna_difference_l625_62588


namespace number_of_schools_l625_62520

def yellow_balloons := 3414
def additional_black_balloons := 1762
def balloons_per_school := 859

def black_balloons := yellow_balloons + additional_black_balloons
def total_balloons := yellow_balloons + black_balloons

theorem number_of_schools : total_balloons / balloons_per_school = 10 :=
by
  sorry

end number_of_schools_l625_62520


namespace parts_per_day_system_l625_62514

variable (x y : ℕ)

def personA_parts_per_day (x : ℕ) : ℕ := x
def personB_parts_per_day (y : ℕ) : ℕ := y

-- First condition
def condition1 (x y : ℕ) : Prop :=
  6 * x = 5 * y

-- Second condition
def condition2 (x y : ℕ) : Prop :=
  30 + 4 * x = 4 * y - 10

theorem parts_per_day_system (x y : ℕ) :
  condition1 x y ∧ condition2 x y :=
sorry

end parts_per_day_system_l625_62514


namespace expected_final_set_size_l625_62547

noncomputable def final_expected_set_size : ℚ :=
  let n := 8
  let initial_size := 255
  let steps := initial_size - 1
  n * (2^7 / initial_size)

theorem expected_final_set_size :
  final_expected_set_size = 1024 / 255 :=
by
  sorry

end expected_final_set_size_l625_62547


namespace jacob_has_5_times_more_l625_62521

variable (A J D : ℕ)
variable (hA : A = 75)
variable (hAJ : A = J / 2)
variable (hD : D = 30)

theorem jacob_has_5_times_more (hA : A = 75) (hAJ : A = J / 2) (hD : D = 30) : J / D = 5 :=
sorry

end jacob_has_5_times_more_l625_62521


namespace digit_possibilities_for_mod4_count_possibilities_is_3_l625_62570

theorem digit_possibilities_for_mod4 (N : ℕ) (h : N < 10): 
  (80 + N) % 4 = 0 → N = 0 ∨ N = 4 ∨ N = 8 → true := 
by
  -- proof is not needed
  sorry

def count_possibilities : ℕ := 
  (if (80 + 0) % 4 = 0 then 1 else 0) +
  (if (80 + 1) % 4 = 0 then 1 else 0) +
  (if (80 + 2) % 4 = 0 then 1 else 0) +
  (if (80 + 3) % 4 = 0 then 1 else 0) +
  (if (80 + 4) % 4 = 0 then 1 else 0) +
  (if (80 + 5) % 4 = 0 then 1 else 0) +
  (if (80 + 6) % 4 = 0 then 1 else 0) +
  (if (80 + 7) % 4 = 0 then 1 else 0) +
  (if (80 + 8) % 4 = 0 then 1 else 0) +
  (if (80 + 9) % 4 = 0 then 1 else 0)

theorem count_possibilities_is_3: count_possibilities = 3 := 
by
  -- proof is not needed
  sorry

end digit_possibilities_for_mod4_count_possibilities_is_3_l625_62570


namespace meat_per_slice_is_22_l625_62515

noncomputable def piecesOfMeatPerSlice : ℕ :=
  let pepperoni := 30
  let ham := 2 * pepperoni
  let sausage := pepperoni + 12
  let totalMeat := pepperoni + ham + sausage
  let slices := 6
  totalMeat / slices

theorem meat_per_slice_is_22 : piecesOfMeatPerSlice = 22 :=
by
  -- Here would be the proof (not required in the task)
  sorry

end meat_per_slice_is_22_l625_62515


namespace chord_length_l625_62572

/-- Given two concentric circles with radii R and r, where the area of the annulus between them is 16π,
    a chord of the larger circle that is tangent to the smaller circle has a length of 8. -/
theorem chord_length {R r c : ℝ} 
  (h1 : R^2 - r^2 = 16)
  (h2 : (c / 2)^2 + r^2 = R^2) :
  c = 8 :=
by
  sorry

end chord_length_l625_62572


namespace perpendicular_d_to_BC_l625_62554

def vector := (ℝ × ℝ)

noncomputable def AB : vector := (1, 1)
noncomputable def AC : vector := (2, 3)

noncomputable def BC : vector := (AC.1 - AB.1, AC.2 - AB.2)

def is_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

noncomputable def d : vector := (-6, 3)

theorem perpendicular_d_to_BC : is_perpendicular d BC :=
by
  sorry

end perpendicular_d_to_BC_l625_62554


namespace melon_weights_l625_62550

-- We start by defining the weights of the individual melons.
variables {D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 : ℝ}

-- Define the weights of the given sets of three melons.
def W1 := D1 + D2 + D3
def W2 := D2 + D3 + D4
def W3 := D1 + D3 + D4
def W4 := D1 + D2 + D4
def W5 := D5 + D6 + D7
def W6 := D8 + D9 + D10

-- State the theorem to be proven.
theorem melon_weights (W1 W2 W3 W4 W5 W6 : ℝ) :
  (W1 + W2 + W3 + W4) / 3 + W5 + W6 = D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 :=
sorry 

end melon_weights_l625_62550


namespace depth_of_melted_ice_cream_l625_62586

theorem depth_of_melted_ice_cream (r_sphere r_cylinder : ℝ) (Vs : ℝ) (Vc : ℝ) :
  r_sphere = 3 →
  r_cylinder = 12 →
  Vs = (4 / 3) * Real.pi * r_sphere^3 →
  Vc = Real.pi * r_cylinder^2 * (1 / 4) →
  Vs = Vc →
  (1 / 4) = 1 / 4 := 
by
  intros hr_sphere hr_cylinder hVs hVc hVs_eq_Vc
  sorry

end depth_of_melted_ice_cream_l625_62586


namespace num_games_played_l625_62548

theorem num_games_played (n : ℕ) (h : n = 14) : (n.choose 2) = 91 :=
by
  sorry

end num_games_played_l625_62548


namespace traffic_flow_solution_l625_62506

noncomputable def traffic_flow_second_ring : ℕ := 10000
noncomputable def traffic_flow_third_ring (x : ℕ) : Prop := 3 * x - (x + 2000) = 2 * traffic_flow_second_ring

theorem traffic_flow_solution :
  ∃ (x : ℕ), traffic_flow_third_ring x ∧ (x = 11000) ∧ (x + 2000 = 13000) :=
by
  sorry

end traffic_flow_solution_l625_62506


namespace intersection_eq_0_l625_62538

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 3, 4}

theorem intersection_eq_0 : M ∩ N = {0} := by
  sorry

end intersection_eq_0_l625_62538


namespace elaine_earnings_l625_62561

variable (E P : ℝ)
variable (H1 : 0.30 * E * (1 + P / 100) = 2.025 * 0.20 * E)

theorem elaine_earnings : P = 35 :=
by
  -- We assume the conditions here and the proof is skipped by sorry.
  sorry

end elaine_earnings_l625_62561


namespace range_of_m_l625_62526

theorem range_of_m (m : ℝ) (y_P : ℝ) (h1 : -3 ≤ y_P) (h2 : y_P ≤ 0) :
  m = (2 + y_P) / 2 → -1 / 2 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l625_62526


namespace math_problem_l625_62575

-- Condition 1: The solution set of the inequality \(\frac{x-2}{ax+b} > 0\) is \((-1,2)\)
def solution_set_condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, (x > -1 ∧ x < 2) ↔ ((x - 2) * (a * x + b) > 0)

-- Condition 2: \(m\) is the geometric mean of \(a\) and \(b\)
def geometric_mean_condition (a b m : ℝ) : Prop :=
  a * b = m^2

-- The mathematical statement to prove: \(\frac{3m^{2}a}{a^{3}+2b^{3}} = 1\)
theorem math_problem (a b m : ℝ) (h1 : solution_set_condition a b) (h2 : geometric_mean_condition a b m) :
  3 * m^2 * a / (a^3 + 2 * b^3) = 1 :=
sorry

end math_problem_l625_62575


namespace find_number_l625_62569

theorem find_number (n : ℤ) 
  (h : (69842 * 69842 - n * n) / (69842 - n) = 100000) : 
  n = 30158 :=
sorry

end find_number_l625_62569


namespace number_of_routes_4x3_grid_l625_62516

def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem number_of_routes_4x3_grid : binomial_coefficient 7 4 = 35 := by
  sorry

end number_of_routes_4x3_grid_l625_62516


namespace parabola_min_value_roots_l625_62524

-- Lean definition encapsulating the problem conditions and conclusion
theorem parabola_min_value_roots (a b c : ℝ) 
  (h1 : ∀ x, (a * x^2 + b * x + c) ≥ 36)
  (hvc : (b^2 - 4 * a * c) = 0)
  (hx1 : (a * (-3)^2 + b * (-3) + c) = 0)
  (hx2 : (a * (5)^2 + b * 5 + c) = 0)
  : a + b + c = 36 := by
  sorry

end parabola_min_value_roots_l625_62524


namespace part1_part2_l625_62582

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x - m|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * f x m - f (x + m) m

theorem part1 (h : ∀ x, g x m ≥ -1) : m = 1 :=
  sorry

theorem part2 {a b m : ℝ} (ha : |a| < m) (hb : |b| < m) (a_ne_zero : a ≠ 0) (hm: m = 1) : 
  f (a * b) m > |a| * f (b / a) m :=
  sorry

end part1_part2_l625_62582


namespace bobby_candy_l625_62594

theorem bobby_candy (C G : ℕ) (H : C + G = 36) (Hchoc: (2/3 : ℚ) * C = 12) (Hgummy: (3/4 : ℚ) * G = 9) : 
  (1/3 : ℚ) * C + (1/4 : ℚ) * G = 9 :=
by
  sorry

end bobby_candy_l625_62594


namespace molecular_weight_of_compound_l625_62599

noncomputable def atomic_weight_carbon : ℝ := 12.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.008
noncomputable def atomic_weight_oxygen : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def num_oxygen_atoms : ℕ := 1

noncomputable def molecular_weight (num_C num_H num_O : ℕ) : ℝ :=
  (num_C * atomic_weight_carbon) + (num_H * atomic_weight_hydrogen) + (num_O * atomic_weight_oxygen)

theorem molecular_weight_of_compound :
  molecular_weight num_carbon_atoms num_hydrogen_atoms num_oxygen_atoms = 65.048 :=
by
  sorry

end molecular_weight_of_compound_l625_62599


namespace find_principal_l625_62546

variable (R P : ℝ)
variable (h1 : ∀ (R P : ℝ), (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400)

theorem find_principal (h1 : ∀ (R P : ℝ), (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400) :
  P = 800 := 
sorry

end find_principal_l625_62546


namespace range_of_x_l625_62568

theorem range_of_x 
  (x : ℝ)
  (h1 : 1 / x < 4) 
  (h2 : 1 / x > -6) 
  (h3 : x < 0) : 
  -1 / 6 < x ∧ x < 0 := 
by 
  sorry

end range_of_x_l625_62568


namespace gears_can_look_complete_l625_62517

theorem gears_can_look_complete (n : ℕ) (h1 : n = 14)
                                 (h2 : ∀ k, k = 4)
                                 (h3 : ∀ i, 0 ≤ i ∧ i < n) :
  ∃ j, 1 ≤ j ∧ j < n ∧ (∀ m1 m2, m1 ≠ m2 → ((m1 + j) % n) ≠ ((m2 + j) % n)) := 
sorry

end gears_can_look_complete_l625_62517


namespace upper_bound_expression_l625_62509

theorem upper_bound_expression (n : ℤ) (U : ℤ) :
  (∀ n, 4 * n + 7 > 1 ∧ 4 * n + 7 < U → ∃ k : ℤ, k = 50) →
  U = 204 :=
by
  sorry

end upper_bound_expression_l625_62509


namespace ratio_of_thermometers_to_hotwater_bottles_l625_62581

theorem ratio_of_thermometers_to_hotwater_bottles (T H : ℕ) (thermometer_price hotwater_bottle_price total_sales : ℕ) 
  (h1 : thermometer_price = 2) (h2 : hotwater_bottle_price = 6) (h3 : total_sales = 1200) (h4 : H = 60) 
  (h5 : total_sales = thermometer_price * T + hotwater_bottle_price * H) : 
  T / H = 7 :=
by
  sorry

end ratio_of_thermometers_to_hotwater_bottles_l625_62581


namespace store_cost_comparison_l625_62530

noncomputable def store_A_cost (x : ℕ) : ℝ := 1760 + 40 * x
noncomputable def store_B_cost (x : ℕ) : ℝ := 1920 + 32 * x

theorem store_cost_comparison (x : ℕ) (h : x > 16) :
  (x > 20 → store_B_cost x < store_A_cost x) ∧ (x < 20 → store_A_cost x < store_B_cost x) :=
by
  sorry

end store_cost_comparison_l625_62530


namespace servings_made_l625_62501

noncomputable def chickpeas_per_can := 16 -- ounces in one can
noncomputable def ounces_per_serving := 6 -- ounces needed per serving
noncomputable def total_cans := 8 -- total cans Thomas buys

theorem servings_made : (total_cans * chickpeas_per_can) / ounces_per_serving = 21 :=
by
  sorry

end servings_made_l625_62501


namespace two_digit_number_l625_62565

theorem two_digit_number (x y : ℕ) (h1 : x + y = 7) (h2 : (x + 2) + 10 * (y + 2) = 2 * (x + 10 * y) - 3) : (10 * y + x) = 25 :=
by
  sorry

end two_digit_number_l625_62565


namespace difference_of_squares_l625_62593

theorem difference_of_squares (n : ℕ) : (n+1)^2 - n^2 = 2*n + 1 :=
by
  sorry

end difference_of_squares_l625_62593


namespace incorrect_statement_for_function_l625_62542

theorem incorrect_statement_for_function (x : ℝ) (h : x > 0) : 
  ¬(∀ x₁ x₂ : ℝ, (x₁ > 0) → (x₂ > 0) → (x₁ < x₂) → (6 / x₁ < 6 / x₂)) := 
sorry

end incorrect_statement_for_function_l625_62542


namespace find_am_2n_l625_62527

-- Definition of the conditions
variables {a : ℝ} {m n : ℝ}
axiom am_eq_5 : a ^ m = 5
axiom an_eq_2 : a ^ n = 2

-- The statement we want to prove
theorem find_am_2n : a ^ (m - 2 * n) = 5 / 4 :=
by {
  sorry
}

end find_am_2n_l625_62527


namespace same_terminal_side_l625_62596

theorem same_terminal_side (α : ℝ) (k : ℤ) : 
  ∃ k : ℤ, α = k * 360 + 60 → α = -300 := 
by
  sorry

end same_terminal_side_l625_62596


namespace John_has_15_snakes_l625_62578

theorem John_has_15_snakes (S : ℕ)
  (H1 : ∀ M, M = 2 * S)
  (H2 : ∀ M L, L = M - 5)
  (H3 : ∀ L P, P = L + 8)
  (H4 : ∀ P D, D = P / 3)
  (H5 : S + (2 * S) + ((2 * S) - 5) + (((2 * S) - 5) + 8) + (((((2 * S) - 5) + 8) / 3)) = 114) :
  S = 15 :=
by sorry

end John_has_15_snakes_l625_62578


namespace problem_part1_problem_part2_l625_62511

theorem problem_part1
  (A : ℤ → ℤ → ℤ)
  (B : ℤ → ℤ → ℤ)
  (x y : ℤ)
  (hA : A x y = 2 * x ^ 2 + 4 * x * y - 2 * x - 3)
  (hB : B x y = -x^2 + x*y + 2) :
  3 * A x y - 2 * (A x y + 2 * B x y) = 6 * x ^ 2 - 2 * x - 11 := by
  sorry

theorem problem_part2
  (A : ℤ → ℤ → ℤ)
  (B : ℤ → ℤ → ℤ)
  (y : ℤ)
  (H : ∀ x, B x y + (1 / 2) * A x y = C) :
  y = 1 / 3 := by
  sorry

end problem_part1_problem_part2_l625_62511


namespace solve_fish_tank_problem_l625_62536

def fish_tank_problem : Prop :=
  ∃ (first_tank_fish second_tank_fish third_tank_fish : ℕ),
  first_tank_fish = 7 + 8 ∧
  second_tank_fish = 2 * first_tank_fish ∧
  third_tank_fish = 10 ∧
  (third_tank_fish : ℚ) / second_tank_fish = 1 / 3

theorem solve_fish_tank_problem : fish_tank_problem :=
by
  sorry

end solve_fish_tank_problem_l625_62536


namespace convert_decimal_to_fraction_l625_62504

theorem convert_decimal_to_fraction : (0.38 : ℚ) = 19 / 50 :=
by
  sorry

end convert_decimal_to_fraction_l625_62504


namespace find_PS_l625_62564

theorem find_PS 
    (P Q R S : Type)
    (PQ PR : ℝ)
    (h : ℝ) 
    (ratio_QS_SR : ℝ)
    (hyp1 : PQ = 13)
    (hyp2 : PR = 20)
    (hyp3 : ratio_QS_SR = 3/7) :
    h = Real.sqrt (117.025) :=
by
  -- Proof steps would go here, but we are just stating the theorem
  sorry

end find_PS_l625_62564
