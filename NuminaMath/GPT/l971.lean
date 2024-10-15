import Mathlib

namespace NUMINAMATH_GPT_lines_parallel_m_value_l971_97198

theorem lines_parallel_m_value (m : ℝ) : 
  (∀ (x y : ℝ), (x + 2 * m * y - 1 = 0) → ((m - 2) * x - m * y + 2 = 0)) → m = 3 / 2 :=
by
  -- placeholder for mathematical proof
  sorry

end NUMINAMATH_GPT_lines_parallel_m_value_l971_97198


namespace NUMINAMATH_GPT_difference_of_reciprocals_l971_97109

theorem difference_of_reciprocals (p q : ℝ) (hp : 3 / p = 6) (hq : 3 / q = 15) : p - q = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_reciprocals_l971_97109


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l971_97134

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 2 - 3) : 
  (1 - (3 / (m + 3))) / (m / (m^2 + 6 * m + 9)) = Real.sqrt 2 := 
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l971_97134


namespace NUMINAMATH_GPT_focus_of_parabola_l971_97190

theorem focus_of_parabola (a : ℝ) (h1 : a > 0)
  (h2 : ∀ x, y = 3 * x → 3 / a = 3) :
  ∃ (focus : ℝ × ℝ), focus = (0, 1 / 8) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l971_97190


namespace NUMINAMATH_GPT_triangle_cross_section_l971_97177

-- Definitions for the given conditions
inductive Solid
| Prism
| Pyramid
| Frustum
| Cylinder
| Cone
| TruncatedCone
| Sphere

-- The theorem statement of the proof problem
theorem triangle_cross_section (s : Solid) (cross_section_is_triangle : Prop) : 
  cross_section_is_triangle →
  (s = Solid.Prism ∨ s = Solid.Pyramid ∨ s = Solid.Frustum ∨ s = Solid.Cone) :=
sorry

end NUMINAMATH_GPT_triangle_cross_section_l971_97177


namespace NUMINAMATH_GPT_negation_abs_lt_one_l971_97118

theorem negation_abs_lt_one (x : ℝ) : (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_abs_lt_one_l971_97118


namespace NUMINAMATH_GPT_abs_eq_abs_iff_eq_frac_l971_97129

theorem abs_eq_abs_iff_eq_frac {x : ℚ} :
  |x - 3| = |x - 4| → x = 7 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_abs_eq_abs_iff_eq_frac_l971_97129


namespace NUMINAMATH_GPT_part1_part2_l971_97138

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := x^2 ≤ 5 * x - 4
def q (x a : ℝ) : Prop := x^2 - (a + 2) * x + 2 * a ≤ 0

-- Theorem statement for part (1)
theorem part1 (x : ℝ) (h : p x) : 1 ≤ x ∧ x ≤ 4 := 
by sorry

-- Theorem statement for part (2)
theorem part2 (a : ℝ) : 
  (∀ x, p x → q x a) ∧ (∃ x, p x) ∧ ¬ (∀ x, q x a → p x) → 1 ≤ a ∧ a ≤ 4 := 
by sorry

end NUMINAMATH_GPT_part1_part2_l971_97138


namespace NUMINAMATH_GPT_cleaner_steps_l971_97180

theorem cleaner_steps (a b c : ℕ) (h1 : a < 10 ∧ b < 10 ∧ c < 10) (h2 : 100 * a + 10 * b + c > 100 * c + 10 * b + a) (h3 : 100 * a + 10 * b + c + 100 * c + 10 * b + a = 746) :
  (100 * a + 10 * b + c) * 2 = 944 ∨ (100 * a + 10 * b + c) * 2 = 1142 :=
by
  sorry

end NUMINAMATH_GPT_cleaner_steps_l971_97180


namespace NUMINAMATH_GPT_smallest_positive_natural_number_l971_97148

theorem smallest_positive_natural_number (a b c d e : ℕ) 
    (h1 : a = 3) (h2 : b = 5) (h3 : c = 6) (h4 : d = 18) (h5 : e = 23) :
    ∃ (x y : ℕ), x = (e - a) / b - d / c ∨ x = e - d + b - c - a ∧ x = 1 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_natural_number_l971_97148


namespace NUMINAMATH_GPT_smallest_year_with_digit_sum_16_l971_97132

def sum_of_digits (n : Nat) : Nat :=
  let digits : List Nat := n.digits 10
  digits.foldl (· + ·) 0

theorem smallest_year_with_digit_sum_16 :
  ∃ (y : Nat), 2010 < y ∧ sum_of_digits y = 16 ∧
  (∀ (z : Nat), 2010 < z ∧ sum_of_digits z = 16 → z ≥ y) → y = 2059 :=
by
  sorry

end NUMINAMATH_GPT_smallest_year_with_digit_sum_16_l971_97132


namespace NUMINAMATH_GPT_find_y_l971_97122

/-- 
  Given: The sum of angles around a point is 360 degrees, 
  and those angles are: 6y, 3y, 4y, and 2y.
  Prove: y = 24 
-/ 
theorem find_y (y : ℕ) (h : 6 * y + 3 * y + 4 * y + 2 * y = 360) : y = 24 :=
sorry

end NUMINAMATH_GPT_find_y_l971_97122


namespace NUMINAMATH_GPT_percentage_both_correct_l971_97171

theorem percentage_both_correct (p1 p2 pn : ℝ) (h1 : p1 = 0.85) (h2 : p2 = 0.80) (h3 : pn = 0.05) :
  ∃ x, x = 0.70 ∧ x = p1 + p2 - 1 + pn := by
  sorry

end NUMINAMATH_GPT_percentage_both_correct_l971_97171


namespace NUMINAMATH_GPT_total_discount_is_58_percent_l971_97173

-- Definitions and conditions
def sale_discount : ℝ := 0.4
def coupon_discount : ℝ := 0.3

-- Given an original price, the sale discount price and coupon discount price
def sale_price (original_price : ℝ) : ℝ := (1 - sale_discount) * original_price
def final_price (original_price : ℝ) : ℝ := (1 - coupon_discount) * (sale_price original_price)

-- Theorem statement: final discount is 58%
theorem total_discount_is_58_percent (original_price : ℝ) : (original_price - final_price original_price) / original_price = 0.58 :=
by intros; sorry

end NUMINAMATH_GPT_total_discount_is_58_percent_l971_97173


namespace NUMINAMATH_GPT_ordered_pair_sqrt_l971_97185

/-- Problem statement: Given positive integers a and b such that a < b, prove that:
sqrt (1 + sqrt (40 + 24 * sqrt 5)) = sqrt a + sqrt b, if (a, b) = (1, 6). -/
theorem ordered_pair_sqrt (a b : ℕ) (h1 : a = 1) (h2 : b = 6) (h3 : a < b) :
  Real.sqrt (1 + Real.sqrt (40 + 24 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b :=
by
  sorry -- The proof is not required in this task.

end NUMINAMATH_GPT_ordered_pair_sqrt_l971_97185


namespace NUMINAMATH_GPT_total_legs_l971_97116

def animals_legs (dogs : Nat) (birds : Nat) (insects : Nat) : Nat :=
  (dogs * 4) + (birds * 2) + (insects * 6)

theorem total_legs :
  animals_legs 3 2 2 = 22 := by
  sorry

end NUMINAMATH_GPT_total_legs_l971_97116


namespace NUMINAMATH_GPT_triangle_area_sqrt2_div2_find_a_c_l971_97143

  -- Problem 1
  -- Prove the area of triangle ABC is sqrt(2)/2
  theorem triangle_area_sqrt2_div2 {a b c : ℝ} 
    (cond1 : a + (1 / a) = 4 * Real.cos (Real.arccos (a^2 + 1 - c^2) / (2 * a))) 
    (cond2 : b = 1) 
    (cond3 : Real.arcsin (1) = Real.pi / 2) : 
    (1 / 2) * 1 * Real.sqrt 2 = Real.sqrt 2 / 2 := sorry

  -- Problem 2
  -- Prove a = sqrt(7) and c = 2
  theorem find_a_c {a b c : ℝ} 
    (cond1 : a + (1 / a) = 4 * Real.cos (Real.arccos (a^2 + 1 - c^2) / (2 * a))) 
    (cond2 : b = 1) 
    (cond3 : (1 / 2) * a * Real.sin (Real.arcsin (Real.sqrt 3 / a)) = Real.sqrt 3 / 2) : 
    a = Real.sqrt 7 ∧ c = 2 := sorry

  
end NUMINAMATH_GPT_triangle_area_sqrt2_div2_find_a_c_l971_97143


namespace NUMINAMATH_GPT_axis_of_symmetry_l971_97170

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (5 - x)) : ∀ x : ℝ, f x = f (2 * 2.5 - x) :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l971_97170


namespace NUMINAMATH_GPT_sam_money_left_l971_97125

/- Definitions -/

def initial_dimes : ℕ := 38
def initial_quarters : ℕ := 12
def initial_nickels : ℕ := 25
def initial_pennies : ℕ := 30

def price_per_candy_bar_dimes : ℕ := 4
def price_per_candy_bar_nickels : ℕ := 2
def candy_bars_bought : ℕ := 5

def price_per_lollipop_nickels : ℕ := 6
def price_per_lollipop_pennies : ℕ := 10
def lollipops_bought : ℕ := 2

def price_per_bag_of_chips_quarters : ℕ := 1
def price_per_bag_of_chips_dimes : ℕ := 3
def price_per_bag_of_chips_pennies : ℕ := 5
def bags_of_chips_bought : ℕ := 3

/- Proof problem statement -/

theorem sam_money_left : 
  (initial_dimes * 10 + initial_quarters * 25 + initial_nickels * 5 + initial_pennies * 1) - 
  (
    candy_bars_bought * (price_per_candy_bar_dimes * 10 + price_per_candy_bar_nickels * 5) + 
    lollipops_bought * (price_per_lollipop_nickels * 5 + price_per_lollipop_pennies * 1) +
    bags_of_chips_bought * (price_per_bag_of_chips_quarters * 25 + price_per_bag_of_chips_dimes * 10 + price_per_bag_of_chips_pennies * 1)
  ) = 325 := 
sorry

end NUMINAMATH_GPT_sam_money_left_l971_97125


namespace NUMINAMATH_GPT_rhombus_area_l971_97142

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 11) (h2 : d2 = 16) : (d1 * d2) / 2 = 88 :=
by {
  -- substitution and proof are omitted, proof body would be provided here
  sorry
}

end NUMINAMATH_GPT_rhombus_area_l971_97142


namespace NUMINAMATH_GPT_cubes_sum_formula_l971_97131

theorem cubes_sum_formula (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 5) : a^3 + b^3 = 238 := 
by 
  sorry

end NUMINAMATH_GPT_cubes_sum_formula_l971_97131


namespace NUMINAMATH_GPT_total_votes_election_l971_97157

theorem total_votes_election (V : ℝ)
    (h1 : 0.55 * 0.8 * V + 2520 = 0.8 * V)
    (h2 : 0.36 > 0) :
    V = 7000 :=
  by
  sorry

end NUMINAMATH_GPT_total_votes_election_l971_97157


namespace NUMINAMATH_GPT_journey_total_distance_l971_97152

/--
Given:
- A person covers 3/5 of their journey by train.
- A person covers 7/20 of their journey by bus.
- A person covers 3/10 of their journey by bicycle.
- A person covers 1/50 of their journey by taxi.
- The rest of the journey (4.25 km) is covered by walking.

Prove:
  D = 15.74 km
where D is the total distance of the journey.
-/
theorem journey_total_distance :
  ∀ (D : ℝ), 3/5 * D + 7/20 * D + 3/10 * D + 1/50 * D + 4.25 = D → D = 15.74 :=
by
  intro D
  sorry

end NUMINAMATH_GPT_journey_total_distance_l971_97152


namespace NUMINAMATH_GPT_shots_cost_l971_97130

-- Define the conditions
def golden_retriever_pregnant_dogs : ℕ := 3
def golden_retriever_puppies_per_dog : ℕ := 4
def golden_retriever_shots_per_puppy : ℕ := 2
def golden_retriever_cost_per_shot : ℕ := 5

def german_shepherd_pregnant_dogs : ℕ := 2
def german_shepherd_puppies_per_dog : ℕ := 5
def german_shepherd_shots_per_puppy : ℕ := 3
def german_shepherd_cost_per_shot : ℕ := 8

def bulldog_pregnant_dogs : ℕ := 4
def bulldog_puppies_per_dog : ℕ := 3
def bulldog_shots_per_puppy : ℕ := 4
def bulldog_cost_per_shot : ℕ := 10

-- Define the total cost calculation
def total_puppies (dogs_per_breed puppies_per_dog : ℕ) : ℕ :=
  dogs_per_breed * puppies_per_dog

def total_shot_cost (puppies shots_per_puppy cost_per_shot : ℕ) : ℕ :=
  puppies * shots_per_puppy * cost_per_shot

def total_cost : ℕ :=
  let golden_retriever_puppies := total_puppies golden_retriever_pregnant_dogs golden_retriever_puppies_per_dog
  let german_shepherd_puppies := total_puppies german_shepherd_pregnant_dogs german_shepherd_puppies_per_dog
  let bulldog_puppies := total_puppies bulldog_pregnant_dogs bulldog_puppies_per_dog
  let golden_retriever_cost := total_shot_cost golden_retriever_puppies golden_retriever_shots_per_puppy golden_retriever_cost_per_shot
  let german_shepherd_cost := total_shot_cost german_shepherd_puppies german_shepherd_shots_per_puppy german_shepherd_cost_per_shot
  let bulldog_cost := total_shot_cost bulldog_puppies bulldog_shots_per_puppy bulldog_cost_per_shot
  golden_retriever_cost + german_shepherd_cost + bulldog_cost

-- Statement of the problem
theorem shots_cost (total_cost : ℕ) : total_cost = 840 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_shots_cost_l971_97130


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l971_97121

theorem hyperbola_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (e : ℝ) (he : e = Real.sqrt 3) (h_eq : e = Real.sqrt ((a^2 + b^2) / a^2)) :
  (∀ x : ℝ, y = x * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l971_97121


namespace NUMINAMATH_GPT_find_x_l971_97164

theorem find_x :
  ∃ x : Real, abs (x - 0.052) < 1e-3 ∧
  (0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + x^2 + 0.0035^2) = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l971_97164


namespace NUMINAMATH_GPT_six_digit_ababab_divisible_by_101_l971_97159

theorem six_digit_ababab_divisible_by_101 (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 0 ≤ b) (h₄ : b ≤ 9) :
  ∃ k : ℕ, 101 * k = 101010 * a + 10101 * b :=
sorry

end NUMINAMATH_GPT_six_digit_ababab_divisible_by_101_l971_97159


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_div_four_l971_97182

theorem tan_alpha_plus_pi_div_four (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_div_four_l971_97182


namespace NUMINAMATH_GPT_cube_convex_hull_half_volume_l971_97133

theorem cube_convex_hull_half_volume : 
  ∃ a : ℝ, 0 <= a ∧ a <= 1 ∧ 4 * (a^3) / 6 + 4 * ((1 - a)^3) / 6 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cube_convex_hull_half_volume_l971_97133


namespace NUMINAMATH_GPT_candy_bar_profit_l971_97179

theorem candy_bar_profit
  (bars_bought : ℕ)
  (cost_per_six : ℝ)
  (bars_sold : ℕ)
  (price_per_three : ℝ)
  (tax_rate : ℝ)
  (h1 : bars_bought = 800)
  (h2 : cost_per_six = 3)
  (h3 : bars_sold = 800)
  (h4 : price_per_three = 2)
  (h5 : tax_rate = 0.1) :
  let cost_per_bar := cost_per_six / 6
  let total_cost := bars_bought * cost_per_bar
  let price_per_bar := price_per_three / 3
  let total_revenue := bars_sold * price_per_bar
  let tax := tax_rate * total_revenue
  let after_tax_revenue := total_revenue - tax
  let profit_after_tax := after_tax_revenue - total_cost
  profit_after_tax = 80.02 := by
    sorry

end NUMINAMATH_GPT_candy_bar_profit_l971_97179


namespace NUMINAMATH_GPT_max_f1_l971_97169

-- Define the function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * b * x + a + 2 * b 

-- Define the condition 
def condition (a : ℝ) (b : ℝ) : Prop := f 0 a b = 4

-- State the theorem
theorem max_f1 (a b: ℝ) (h: condition a b) : 
  ∃ b_max, b_max = 1 ∧ ∀ b, f 1 a b ≤ 7 := 
sorry

end NUMINAMATH_GPT_max_f1_l971_97169


namespace NUMINAMATH_GPT_ordered_sum_ways_l971_97104

theorem ordered_sum_ways (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 2) : 
  ∃ (ways : ℕ), ways = 70 :=
by
  sorry

end NUMINAMATH_GPT_ordered_sum_ways_l971_97104


namespace NUMINAMATH_GPT_guard_team_size_l971_97192

theorem guard_team_size (b n s : ℕ) (h_total : b * s * n = 1001) (h_condition : s < n ∧ n < b) : s = 7 := 
by
  sorry

end NUMINAMATH_GPT_guard_team_size_l971_97192


namespace NUMINAMATH_GPT_biloca_path_proof_l971_97193

def diagonal_length := 5 -- Length of one diagonal as deduced from Pipoca's path
def tile_width := 3 -- Width of one tile as deduced from Tonica's path
def tile_length := 4 -- Length of one tile as deduced from Cotinha's path

def Biloca_path_length : ℝ :=
  3 * diagonal_length + 4 * tile_width + 2 * tile_length

theorem biloca_path_proof :
  Biloca_path_length = 43 :=
by
  sorry

end NUMINAMATH_GPT_biloca_path_proof_l971_97193


namespace NUMINAMATH_GPT_larger_number_225_l971_97189

theorem larger_number_225 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a - b = 120) 
  (h4 : Nat.lcm a b = 105 * Nat.gcd a b) : 
  max a b = 225 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_225_l971_97189


namespace NUMINAMATH_GPT_triangle_area_is_24_l971_97127

structure Point where
  x : ℝ
  y : ℝ

def distance_x (A B : Point) : ℝ :=
  abs (B.x - A.x)

def distance_y (A C : Point) : ℝ :=
  abs (C.y - A.y)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * distance_x A B * distance_y A C

noncomputable def A : Point := ⟨2, 2⟩
noncomputable def B : Point := ⟨8, 2⟩
noncomputable def C : Point := ⟨4, 10⟩

theorem triangle_area_is_24 : triangle_area A B C = 24 := 
  sorry

end NUMINAMATH_GPT_triangle_area_is_24_l971_97127


namespace NUMINAMATH_GPT_holiday_customers_l971_97181

-- Define the normal rate of customers entering the store (175 people/hour)
def normal_rate : ℕ := 175

-- Define the holiday rate of customers entering the store
def holiday_rate : ℕ := 2 * normal_rate

-- Define the duration for which we are calculating the total number of customers (8 hours)
def duration : ℕ := 8

-- Define the correct total number of customers (2800 people)
def correct_total_customers : ℕ := 2800

-- The theorem that asserts the total number of customers in 8 hours during the holiday season is 2800
theorem holiday_customers : holiday_rate * duration = correct_total_customers := by
  sorry

end NUMINAMATH_GPT_holiday_customers_l971_97181


namespace NUMINAMATH_GPT_sequence_general_term_l971_97107

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = n^2 + 1) :
  (∀ n, a n = if n = 1 then 2 else 2 * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l971_97107


namespace NUMINAMATH_GPT_circle_equation_l971_97166

theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (1, 0)
  let point : ℝ × ℝ := (1, -1)
  let radius : ℝ := dist center point
  dist center point = 1 → 
  (x - 1)^2 + y^2 = radius^2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_circle_equation_l971_97166


namespace NUMINAMATH_GPT_total_wheels_in_garage_l971_97149

def bicycles: Nat := 3
def tricycles: Nat := 4
def unicycles: Nat := 7

def wheels_per_bicycle: Nat := 2
def wheels_per_tricycle: Nat := 3
def wheels_per_unicycle: Nat := 1

theorem total_wheels_in_garage (bicycles tricycles unicycles wheels_per_bicycle wheels_per_tricycle wheels_per_unicycle : Nat) :
  bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle = 25 := by
  sorry

end NUMINAMATH_GPT_total_wheels_in_garage_l971_97149


namespace NUMINAMATH_GPT_max_value_a4_b2_c2_d2_l971_97105

theorem max_value_a4_b2_c2_d2
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = 10) :
  a^4 + b^2 + c^2 + d^2 ≤ 100 :=
sorry

end NUMINAMATH_GPT_max_value_a4_b2_c2_d2_l971_97105


namespace NUMINAMATH_GPT_parabola_vertex_l971_97154

theorem parabola_vertex :
  ∀ (x : ℝ), y = 2 * (x + 9)^2 - 3 → 
  (∃ h k, h = -9 ∧ k = -3 ∧ y = 2 * (x - h)^2 + k) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l971_97154


namespace NUMINAMATH_GPT_piper_gym_sessions_l971_97150

-- Define the conditions and the final statement as a theorem
theorem piper_gym_sessions (session_count : ℕ) (week_days : ℕ) (start_day : ℕ) 
  (alternate_day : ℕ) (skip_day : ℕ): (session_count = 35) ∧ (week_days = 7) ∧ 
  (start_day = 1) ∧ (alternate_day = 2) ∧ (skip_day = 7) → 
  (start_day + ((session_count - 1) / 3) * week_days + ((session_count - 1) % 3) * alternate_day) % week_days = 3 := 
by 
  sorry

end NUMINAMATH_GPT_piper_gym_sessions_l971_97150


namespace NUMINAMATH_GPT_find_value_a_prove_inequality_l971_97115

noncomputable def arithmetic_sequence (a : ℕ) (S : ℕ → ℕ) (a_n : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 2 → S n * S n = 3 * n ^ 2 * a_n n + S (n - 1) * S (n - 1) ∧ a_n n ≠ 0

theorem find_value_a {S : ℕ → ℕ} {a_n : ℕ → ℕ} :
  (∃ (a : ℕ), arithmetic_sequence a S a_n) → a = 3 :=
sorry

noncomputable def sequence_bn (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) :=
  ∀ n : ℕ, b_n n = 1 / ((a_n n - 1) * (a_n n + 2))

theorem prove_inequality {S : ℕ → ℕ} {a_n : ℕ → ℕ} {b_n : ℕ → ℕ} {T : ℕ → ℕ} :
  (∃ (a : ℕ), arithmetic_sequence a S a_n) →
  (sequence_bn a_n b_n) →
  ∀ n : ℕ, T n < 1 / 6 :=
sorry

end NUMINAMATH_GPT_find_value_a_prove_inequality_l971_97115


namespace NUMINAMATH_GPT_minimum_value_of_f_l971_97117

def f (x : ℝ) : ℝ := 5 * x^2 - 20 * x + 1357

theorem minimum_value_of_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) := 
by 
  use 1337
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l971_97117


namespace NUMINAMATH_GPT_find_a_minus_b_l971_97178

theorem find_a_minus_b (a b : ℝ) (h1: ∀ x : ℝ, (ax^2 + bx - 2 = 0 → x = -2 ∨ x = -1/4)) : (a - b = 5) :=
sorry

end NUMINAMATH_GPT_find_a_minus_b_l971_97178


namespace NUMINAMATH_GPT_second_player_wins_l971_97168

theorem second_player_wins 
  (pile1 : ℕ) (pile2 : ℕ) (pile3 : ℕ)
  (h1 : pile1 = 10) (h2 : pile2 = 15) (h3 : pile3 = 20) :
  (pile1 - 1) + (pile2 - 1) + (pile3 - 1) % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_second_player_wins_l971_97168


namespace NUMINAMATH_GPT_sin_cos_15_eq_1_over_4_l971_97124

theorem sin_cos_15_eq_1_over_4 : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_15_eq_1_over_4_l971_97124


namespace NUMINAMATH_GPT_tony_additional_degrees_l971_97155

-- Definitions for the conditions
def total_years : ℕ := 14
def science_degree_years : ℕ := 4
def physics_degree_years : ℕ := 2
def additional_degree_years : ℤ := total_years - (science_degree_years + physics_degree_years)
def each_additional_degree_years : ℕ := 4
def additional_degrees : ℤ := additional_degree_years / each_additional_degree_years

-- Theorem stating the problem and the answer
theorem tony_additional_degrees : additional_degrees = 2 :=
 by
     sorry

end NUMINAMATH_GPT_tony_additional_degrees_l971_97155


namespace NUMINAMATH_GPT_simplify_expression_l971_97194

theorem simplify_expression (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
    (x + 2 - 5 / (x - 2)) / ((x + 3) / (x - 2)) = x - 3 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l971_97194


namespace NUMINAMATH_GPT_saucepan_capacity_l971_97137

-- Define the conditions
variable (x : ℝ)
variable (h : 0.28 * x = 35)

-- State the theorem
theorem saucepan_capacity : x = 125 :=
by
  sorry

end NUMINAMATH_GPT_saucepan_capacity_l971_97137


namespace NUMINAMATH_GPT_three_rays_with_common_point_l971_97146

theorem three_rays_with_common_point (x y : ℝ) :
  (∃ (common : ℝ), ((5 = x - 1 ∧ y + 3 ≤ 5) ∨ 
                     (5 = y + 3 ∧ x - 1 ≤ 5) ∨ 
                     (x - 1 = y + 3 ∧ 5 ≤ x - 1 ∧ 5 ≤ y + 3)) 
  ↔ ((x = 6 ∧ y ≤ 2) ∨ (y = 2 ∧ x ≤ 6) ∨ (y = x - 4 ∧ x ≥ 6))) :=
sorry

end NUMINAMATH_GPT_three_rays_with_common_point_l971_97146


namespace NUMINAMATH_GPT_twenty_four_multiples_of_4_l971_97145

theorem twenty_four_multiples_of_4 {n : ℕ} : (n = 104) ↔ (∃ k : ℕ, k = 24 ∧ ∀ m : ℕ, (12 ≤ m ∧ m ≤ n) → ∃ t : ℕ, m = 12 + 4 * t ∧ 1 ≤ t ∧ t ≤ 24) := 
by
  sorry

end NUMINAMATH_GPT_twenty_four_multiples_of_4_l971_97145


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l971_97197

theorem arithmetic_sequence_sum :
  ∀ (x y : ℕ), (∀ (a b c : ℕ), b - a = c - b → c - b = 5) ∧
               (3 + 5 * 1 = 8) ∧
               (8 + 5 * 1 = 13) ∧
               (x + 5 * 1 = y) ∧
               (y + 5 * 1 = 33) →
               x + y = 51 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l971_97197


namespace NUMINAMATH_GPT_last_digit_sum_l971_97101

theorem last_digit_sum :
  (2^2 % 10 + 20^20 % 10 + 200^200 % 10 + 2006^2006 % 10) % 10 = 0 := 
by
  sorry

end NUMINAMATH_GPT_last_digit_sum_l971_97101


namespace NUMINAMATH_GPT_product_ge_half_l971_97183

theorem product_ge_half (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : 0 ≤ x3) (h_sum : x1 + x2 + x3 ≤ 1/2) :
  (1 - x1) * (1 - x2) * (1 - x3) ≥ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_product_ge_half_l971_97183


namespace NUMINAMATH_GPT_simson_line_properties_l971_97140

-- Given a triangle ABC
variables {A B C M P Q R H : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] 
variables [Inhabited M] [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited H]

-- Conditions
def is_point_on_circumcircle (A B C : Type) (M : Type) : Prop :=
sorry  -- formal definition that M is on the circumcircle of triangle ABC

def perpendicular_dropped_to_side (M : Type) (side : Type) (foot : Type) : Prop :=
sorry  -- formal definition of a perpendicular dropping from M to a side

def is_orthocenter (A B C H : Type) : Prop := 
sorry  -- formal definition that H is the orthocenter of triangle ABC

-- Proof Goal 1: The points P, Q, R are collinear (Simson line)
def simson_line (A B C M P Q R : Type) : Prop :=
sorry  -- formal definition and proof that P, Q, R are collinear

-- Proof Goal 2: The Simson line is equidistant from point M and the orthocenter H
def simson_line_equidistant (M H P Q R : Type) : Prop :=
sorry  -- formal definition and proof that Simson line is equidistant from M and H

-- Main theorem combining both proof goals
theorem simson_line_properties 
  (A B C M P Q R H : Type)
  (M_on_circumcircle : is_point_on_circumcircle A B C M)
  (perp_to_BC : perpendicular_dropped_to_side M (B × C) P)
  (perp_to_CA : perpendicular_dropped_to_side M (C × A) Q)
  (perp_to_AB : perpendicular_dropped_to_side M (A × B) R)
  (H_is_orthocenter : is_orthocenter A B C H) :
  simson_line A B C M P Q R ∧ simson_line_equidistant M H P Q R := 
by sorry

end NUMINAMATH_GPT_simson_line_properties_l971_97140


namespace NUMINAMATH_GPT_find_smallest_z_l971_97162

theorem find_smallest_z (x y z : ℤ) (h1 : 7 < x) (h2 : x < 9) (h3 : x < y) (h4 : y < z) 
  (h5 : y - x = 7) : z = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_z_l971_97162


namespace NUMINAMATH_GPT_hedge_cost_and_blocks_l971_97135

-- Define the costs of each type of block
def costA : Nat := 2
def costB : Nat := 3
def costC : Nat := 4

-- Define the number of each type of block per section
def blocksPerSectionA : Nat := 20
def blocksPerSectionB : Nat := 10
def blocksPerSectionC : Nat := 5

-- Define the number of sections
def sections : Nat := 8

-- Define the total cost calculation
def totalCost : Nat := sections * (blocksPerSectionA * costA + blocksPerSectionB * costB + blocksPerSectionC * costC)

-- Define the total number of each type of block used
def totalBlocksA : Nat := sections * blocksPerSectionA
def totalBlocksB : Nat := sections * blocksPerSectionB
def totalBlocksC : Nat := sections * blocksPerSectionC

-- State the theorem
theorem hedge_cost_and_blocks :
  totalCost = 720 ∧ totalBlocksA = 160 ∧ totalBlocksB = 80 ∧ totalBlocksC = 40 := by
  sorry

end NUMINAMATH_GPT_hedge_cost_and_blocks_l971_97135


namespace NUMINAMATH_GPT_perpendicular_vectors_l971_97174

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end NUMINAMATH_GPT_perpendicular_vectors_l971_97174


namespace NUMINAMATH_GPT_values_of_x_that_satisfy_gg_x_eq_g_x_l971_97110

noncomputable def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_that_satisfy_gg_x_eq_g_x :
  {x : ℝ | g (g x) = g x} = {0, 5, -2, 3} :=
by
  sorry

end NUMINAMATH_GPT_values_of_x_that_satisfy_gg_x_eq_g_x_l971_97110


namespace NUMINAMATH_GPT_probability_three_defective_phones_l971_97199

theorem probability_three_defective_phones :
  let total_smartphones := 380
  let defective_smartphones := 125
  let P_def_1 := (defective_smartphones : ℝ) / total_smartphones
  let P_def_2 := (defective_smartphones - 1 : ℝ) / (total_smartphones - 1)
  let P_def_3 := (defective_smartphones - 2 : ℝ) / (total_smartphones - 2)
  let P_all_three_def := P_def_1 * P_def_2 * P_def_3
  abs (P_all_three_def - 0.0351) < 0.001 := 
by
  sorry

end NUMINAMATH_GPT_probability_three_defective_phones_l971_97199


namespace NUMINAMATH_GPT_distance_to_weekend_class_l971_97123

theorem distance_to_weekend_class:
  ∃ d v : ℝ, (d = v * (1 / 2)) ∧ (d = (v + 10) * (3 / 10)) → d = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_weekend_class_l971_97123


namespace NUMINAMATH_GPT_initial_ratio_of_liquids_l971_97120

theorem initial_ratio_of_liquids (A B : ℕ) (H1 : A = 21)
  (H2 : 9 * A = 7 * (B + 9)) :
  A / B = 7 / 6 :=
sorry

end NUMINAMATH_GPT_initial_ratio_of_liquids_l971_97120


namespace NUMINAMATH_GPT_z_squared_in_second_quadrant_l971_97119
open Complex Real

noncomputable def z : ℂ := exp (π * I / 3)

theorem z_squared_in_second_quadrant : (z^2).re < 0 ∧ (z^2).im > 0 :=
by
  sorry

end NUMINAMATH_GPT_z_squared_in_second_quadrant_l971_97119


namespace NUMINAMATH_GPT_siblings_pizza_order_l971_97141

theorem siblings_pizza_order :
  let Alex := 1 / 6
  let Beth := 2 / 5
  let Cyril := 1 / 3
  let Dan := 1 - (Alex + Beth + Cyril)
  Dan > Alex ∧ Alex > Cyril ∧ Cyril > Beth := sorry

end NUMINAMATH_GPT_siblings_pizza_order_l971_97141


namespace NUMINAMATH_GPT_equivalent_single_discount_rate_l971_97128

-- Definitions based on conditions
def original_price : ℝ := 120
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.15
def combined_discount_rate : ℝ := 0.3625  -- This is the expected result

-- The proof problem statement
theorem equivalent_single_discount_rate :
  (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) = 
  (original_price * (1 - combined_discount_rate)) := 
sorry

end NUMINAMATH_GPT_equivalent_single_discount_rate_l971_97128


namespace NUMINAMATH_GPT_find_m_of_power_fn_and_increasing_l971_97187

theorem find_m_of_power_fn_and_increasing (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 5) * x^(m - 1) > 0) →
  m^2 - m - 5 = 1 →
  1 < m →
  m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_of_power_fn_and_increasing_l971_97187


namespace NUMINAMATH_GPT_total_miles_l971_97196

theorem total_miles (miles_Katarina miles_Harriet miles_Tomas miles_Tyler : ℕ)
  (hK : miles_Katarina = 51)
  (hH : miles_Harriet = 48)
  (hT : miles_Tomas = 48)
  (hTy : miles_Tyler = 48) :
  miles_Katarina + miles_Harriet + miles_Tomas + miles_Tyler = 195 :=
  by
    sorry

end NUMINAMATH_GPT_total_miles_l971_97196


namespace NUMINAMATH_GPT_find_b_l971_97106

-- Define the given hyperbola equation and conditions
def hyperbola (x y : ℝ) (b : ℝ) : Prop := x^2 - y^2 / b^2 = 1
def asymptote_line (x y : ℝ) : Prop := 2 * x - y = 0

-- State the theorem to prove
theorem find_b (b : ℝ) (hb : b > 0) :
    (∀ x y : ℝ, hyperbola x y b → asymptote_line x y) → b = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_b_l971_97106


namespace NUMINAMATH_GPT_solve_for_t_l971_97158

variable (f : ℝ → ℝ)
variable (x t : ℝ)

-- Conditions
def cond1 : Prop := ∀ x, f ((1 / 2) * x - 1) = 2 * x + 3
def cond2 : Prop := f t = 4

-- Theorem statement
theorem solve_for_t (h1 : cond1 f) (h2 : cond2 f t) : t = -3 / 4 := by
  sorry

end NUMINAMATH_GPT_solve_for_t_l971_97158


namespace NUMINAMATH_GPT_fruit_basket_apples_oranges_ratio_l971_97102

theorem fruit_basket_apples_oranges_ratio : 
  ∀ (apples oranges : ℕ), 
  apples = 15 ∧ (2 * apples / 3 + 2 * oranges / 3 = 50) → (apples = 15 ∧ oranges = 60) → apples / gcd apples oranges = 1 ∧ oranges / gcd apples oranges = 4 :=
by 
  intros apples oranges h1 h2
  have h_apples : apples = 15 := by exact h2.1
  have h_oranges : oranges = 60 := by exact h2.2
  rw [h_apples, h_oranges]
  sorry

end NUMINAMATH_GPT_fruit_basket_apples_oranges_ratio_l971_97102


namespace NUMINAMATH_GPT_triangle_area_proof_l971_97184

noncomputable def triangle_area (a b c C : ℝ) : ℝ := 0.5 * a * b * Real.sin C

theorem triangle_area_proof:
  ∀ (A B C a b c : ℝ),
  ¬ (C = π/2) ∧
  c = 1 ∧
  C = π/3 ∧
  Real.sin C + Real.sin (A - B) = 3 * Real.sin (2*B) →
  triangle_area a b c C = 3 * Real.sqrt 3 / 28 :=
by
  intros A B C a b c h
  sorry

end NUMINAMATH_GPT_triangle_area_proof_l971_97184


namespace NUMINAMATH_GPT_find_x_l971_97114

-- Conditions
def volume_condition (x : ℝ) (s : ℝ) : Prop := s^3 = 8 * x
def area_condition (x : ℝ) (s : ℝ) : Prop := 6 * s^2 = x / 2

-- Theorem to prove
theorem find_x (x s : ℝ) (h1 : volume_condition x s) (h2 : area_condition x s) : x = 110592 := sorry

end NUMINAMATH_GPT_find_x_l971_97114


namespace NUMINAMATH_GPT_sanoop_initial_tshirts_l971_97186

theorem sanoop_initial_tshirts (n : ℕ) (T : ℕ) 
(avg_initial : T = n * 526) 
(avg_remaining : T - 673 = (n - 1) * 505) 
(avg_returned : 673 = 673) : 
n = 8 := 
by 
  sorry

end NUMINAMATH_GPT_sanoop_initial_tshirts_l971_97186


namespace NUMINAMATH_GPT_milton_zoology_books_l971_97176

theorem milton_zoology_books
  (z b : ℕ)
  (h1 : z + b = 80)
  (h2 : b = 4 * z) :
  z = 16 :=
by sorry

end NUMINAMATH_GPT_milton_zoology_books_l971_97176


namespace NUMINAMATH_GPT_shaded_percentage_l971_97111

-- Definition for the six-by-six grid and total squares
def total_squares : ℕ := 36
def shaded_squares : ℕ := 16

-- Definition of the problem: to prove the percentage of shaded squares
theorem shaded_percentage : (shaded_squares : ℚ) / total_squares * 100 = 44.4 :=
by
  sorry

end NUMINAMATH_GPT_shaded_percentage_l971_97111


namespace NUMINAMATH_GPT_sqrt_nested_eq_five_l971_97126

theorem sqrt_nested_eq_five {x : ℝ} (h : x = Real.sqrt (15 + x)) : x = 5 :=
sorry

end NUMINAMATH_GPT_sqrt_nested_eq_five_l971_97126


namespace NUMINAMATH_GPT_square_side_length_l971_97172

theorem square_side_length
  (P : ℕ) (A : ℕ) (s : ℕ)
  (h1 : P = 44)
  (h2 : A = 121)
  (h3 : P = 4 * s)
  (h4 : A = s * s) :
  s = 11 :=
sorry

end NUMINAMATH_GPT_square_side_length_l971_97172


namespace NUMINAMATH_GPT_player_A_advantage_l971_97153

theorem player_A_advantage (B A : ℤ) (rolls : ℕ) (h : rolls = 36) 
  (game_conditions : ∀ (x : ℕ), (x % 2 = 1 → A = A + x ∧ B = B - x) ∧ 
                      (x % 2 = 0 ∧ x ≠ 2 → A = A - x ∧ B = B + x) ∧ 
                      (x = 2 → A = A ∧ B = B)) : 
  (36 * (1 / 18 : ℚ) = 2) :=
by {
  -- Mathematical proof will be filled here
  sorry
}

end NUMINAMATH_GPT_player_A_advantage_l971_97153


namespace NUMINAMATH_GPT_track_width_l971_97147

theorem track_width (r : ℝ) (h1 : 4 * π * r - 2 * π * r = 16 * π) (h2 : 2 * r = r + r) : 2 * r - r = 8 :=
by
  sorry

end NUMINAMATH_GPT_track_width_l971_97147


namespace NUMINAMATH_GPT_perpendicular_lines_m_l971_97100

theorem perpendicular_lines_m (m : ℝ) :
  (∀ (x y : ℝ), x - 2 * y + 5 = 0 → 2 * x + m * y - 6 = 0) →
  m = 1 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_m_l971_97100


namespace NUMINAMATH_GPT_max_marked_points_l971_97139

theorem max_marked_points (segments : ℕ) (ratio : ℚ) (h_segments : segments = 10) (h_ratio : ratio = 3 / 4) : 
  ∃ n, n ≤ (segments * 2 / 2) ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_max_marked_points_l971_97139


namespace NUMINAMATH_GPT_no_real_solution_l971_97144

-- Define the hypothesis: the sum of partial fractions
theorem no_real_solution : 
  ¬ ∃ x : ℝ, 
    (1 / ((x - 1) * (x - 3)) + 
     1 / ((x - 3) * (x - 5)) + 
     1 / ((x - 5) * (x - 7))) = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_no_real_solution_l971_97144


namespace NUMINAMATH_GPT_boat_downstream_distance_l971_97191

variable (speed_still_water : ℤ) (speed_stream : ℤ) (time_downstream : ℤ)

theorem boat_downstream_distance
    (h₁ : speed_still_water = 24)
    (h₂ : speed_stream = 4)
    (h₃ : time_downstream = 4) :
    (speed_still_water + speed_stream) * time_downstream = 112 := by
  sorry

end NUMINAMATH_GPT_boat_downstream_distance_l971_97191


namespace NUMINAMATH_GPT_parabola_directrix_l971_97163

theorem parabola_directrix (x y : ℝ) (h : y = 2 * x^2) : y = - (1 / 8) :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l971_97163


namespace NUMINAMATH_GPT_total_batteries_correct_l971_97156

-- Definitions of the number of batteries used in each category
def batteries_flashlight : ℕ := 2
def batteries_toys : ℕ := 15
def batteries_controllers : ℕ := 2

-- The total number of batteries used by Tom
def total_batteries : ℕ := batteries_flashlight + batteries_toys + batteries_controllers

-- The proof statement that needs to be proven
theorem total_batteries_correct : total_batteries = 19 := by
  sorry

end NUMINAMATH_GPT_total_batteries_correct_l971_97156


namespace NUMINAMATH_GPT_percent_of_y_eq_l971_97188

theorem percent_of_y_eq (y : ℝ) (h : y ≠ 0) : (0.3 * 0.7 * y) = (0.21 * y) := by
  sorry

end NUMINAMATH_GPT_percent_of_y_eq_l971_97188


namespace NUMINAMATH_GPT_ellipse_polar_inverse_sum_l971_97108

noncomputable def ellipse_equation (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 3 * Real.sin α)

theorem ellipse_polar_inverse_sum (A B : ℝ × ℝ)
  (hA : ∃ α₁, ellipse_equation α₁ = A)
  (hB : ∃ α₂, ellipse_equation α₂ = B)
  (hPerp : A.1 * B.1 + A.2 * B.2 = 0) :
  (1 / (A.1 ^ 2 + A.2 ^ 2) + 1 / (B.1 ^ 2 + B.2 ^ 2)) = 7 / 12 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_polar_inverse_sum_l971_97108


namespace NUMINAMATH_GPT_sister_granola_bars_l971_97167

-- Definitions based on conditions
def total_bars := 20
def chocolate_chip_bars := 8
def oat_honey_bars := 6
def peanut_butter_bars := 6

def greg_set_aside_chocolate := 3
def greg_set_aside_oat_honey := 2
def greg_set_aside_peanut_butter := 2

def final_chocolate_chip := chocolate_chip_bars - greg_set_aside_chocolate - 2  -- 2 traded away
def final_oat_honey := oat_honey_bars - greg_set_aside_oat_honey - 4           -- 4 traded away
def final_peanut_butter := peanut_butter_bars - greg_set_aside_peanut_butter

-- Final distribution to sisters
def older_sister_chocolate := 2.5 -- 2 whole bars + 1/2 bar
def younger_sister_peanut := 2.5  -- 2 whole bars + 1/2 bar

theorem sister_granola_bars :
  older_sister_chocolate = 2.5 ∧ younger_sister_peanut = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_sister_granola_bars_l971_97167


namespace NUMINAMATH_GPT_books_in_series_l971_97151

-- Define the number of movies
def M := 14

-- Define that the number of books is one more than the number of movies
def B := M + 1

-- Theorem statement to prove that the number of books is 15
theorem books_in_series : B = 15 :=
by
  sorry

end NUMINAMATH_GPT_books_in_series_l971_97151


namespace NUMINAMATH_GPT_add_base8_l971_97175

/-- Define the numbers in base 8 --/
def base8_add (a b : Nat) : Nat := 
  sorry

theorem add_base8 : base8_add 0o12 0o157 = 0o171 := 
  sorry

end NUMINAMATH_GPT_add_base8_l971_97175


namespace NUMINAMATH_GPT_proof_problem_l971_97136

-- Definitions
def is_factor (a b : ℕ) : Prop := ∃ k, b = a * k
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

-- Conditions
def condition_A : Prop := is_factor 4 24
def condition_B : Prop := is_divisor 19 152 ∧ ¬ is_divisor 19 96
def condition_E : Prop := is_factor 6 180

-- Proof problem statement
theorem proof_problem : condition_A ∧ condition_B ∧ condition_E :=
by sorry

end NUMINAMATH_GPT_proof_problem_l971_97136


namespace NUMINAMATH_GPT_at_least_one_hit_l971_97160

-- Introduce the predicates
variable (p q : Prop)

-- State the theorem
theorem at_least_one_hit : (¬ (¬ p ∧ ¬ q)) = (p ∨ q) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_hit_l971_97160


namespace NUMINAMATH_GPT_max_possible_ratio_squared_l971_97112

noncomputable def maxRatioSquared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) (h4 : ∃ x y, (0 ≤ x) ∧ (x < a) ∧ (0 ≤ y) ∧ (y < b) ∧ (a^2 + y^2 = b^2 + x^2) ∧ (b^2 + x^2 = (a - x)^2 + (b + y)^2)) : ℝ :=
  2

theorem max_possible_ratio_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) (h4 : ∃ x y, (0 ≤ x) ∧ (x < a) ∧ (0 ≤ y) ∧ (y < b) ∧ (a^2 + y^2 = b^2 + x^2) ∧ (b^2 + x^2 = (a - x)^2 + (b + y)^2)) : maxRatioSquared a b h1 h2 h3 h4 = 2 :=
sorry

end NUMINAMATH_GPT_max_possible_ratio_squared_l971_97112


namespace NUMINAMATH_GPT_min_value_of_quadratic_l971_97113

theorem min_value_of_quadratic (x : ℝ) : ∃ z : ℝ, z = 2 * x^2 + 16 * x + 40 ∧ z = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_of_quadratic_l971_97113


namespace NUMINAMATH_GPT_problem1_problem2_l971_97195

namespace ProofProblems

-- Problem 1: Prove the inequality
theorem problem1 (x : ℝ) (h : x + |2 * x - 1| < 3) : -2 < x ∧ x < 4 / 3 := 
sorry

-- Problem 2: Prove the value of x + y + z 
theorem problem2 (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2 * y + 3 * z = Real.sqrt 14) : 
  x + y + z = 3 * Real.sqrt 14 / 7 := 
sorry

end ProofProblems

end NUMINAMATH_GPT_problem1_problem2_l971_97195


namespace NUMINAMATH_GPT_ginger_distance_l971_97165

theorem ginger_distance : 
  ∀ (d : ℝ), (d / 4 - d / 6 = 1 / 16) → (d = 3 / 4) := 
by 
  intro d h
  sorry

end NUMINAMATH_GPT_ginger_distance_l971_97165


namespace NUMINAMATH_GPT_problem_statement_l971_97161

open Real

theorem problem_statement (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1/a) + (1/b) = 1) (hn_pos : 0 < n) :
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry -- proof to be provided

end NUMINAMATH_GPT_problem_statement_l971_97161


namespace NUMINAMATH_GPT_compute_result_l971_97103

theorem compute_result : (300000 * 200000) / 100000 = 600000 := by
  sorry

end NUMINAMATH_GPT_compute_result_l971_97103
