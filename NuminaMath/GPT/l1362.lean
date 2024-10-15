import Mathlib

namespace NUMINAMATH_GPT_vector_normalization_condition_l1362_136205

variables {a b : ℝ} -- Ensuring that Lean understands ℝ refers to real numbers and specifically vectors in ℝ before using it in the next parts.

-- Definitions of the vector variables
variables (a b : ℝ) (ab_non_zero : a ≠ 0 ∧ b ≠ 0)

-- Required statement
theorem vector_normalization_condition (a b : ℝ) 
(h₀ : a ≠ 0 ∧ b ≠ 0) :
  (a / abs a = b / abs b) ↔ (a = 2 * b) :=
sorry

end NUMINAMATH_GPT_vector_normalization_condition_l1362_136205


namespace NUMINAMATH_GPT_negation_of_proposition_l1362_136260

theorem negation_of_proposition : 
    (¬ (∀ x : ℝ, x^2 - 2 * |x| ≥ 0)) ↔ (∃ x : ℝ, x^2 - 2 * |x| < 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l1362_136260


namespace NUMINAMATH_GPT_Moscow1964_27th_MMO_l1362_136270

theorem Moscow1964_27th_MMO {a : ℤ} (h : ∀ k : ℤ, k ≠ 27 → ∃ m : ℤ, a - k^1964 = m * (27 - k)) : 
  a = 27^1964 :=
sorry

end NUMINAMATH_GPT_Moscow1964_27th_MMO_l1362_136270


namespace NUMINAMATH_GPT_eval_expression_l1362_136220

theorem eval_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / (a * b) - (a^2 + a * b) / (a^2 + b^2) = (a^4 + b^4 + a^2 * b^2 - a^2 * b - a * b^2) / (a * b * (a^2 + b^2)) :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1362_136220


namespace NUMINAMATH_GPT_largest_divisor_even_triplet_l1362_136209

theorem largest_divisor_even_triplet :
  ∀ (n : ℕ), 24 ∣ (2 * n) * (2 * n + 2) * (2 * n + 4) :=
by intros; sorry

end NUMINAMATH_GPT_largest_divisor_even_triplet_l1362_136209


namespace NUMINAMATH_GPT_count_integers_in_range_l1362_136265

theorem count_integers_in_range : 
  let lower_bound := -2.8
  let upper_bound := Real.pi
  let in_range (x : ℤ) := (lower_bound : ℝ) < (x : ℝ) ∧ (x : ℝ) ≤ upper_bound
  (Finset.filter in_range (Finset.Icc (Int.floor lower_bound) (Int.floor upper_bound))).card = 6 :=
by
  sorry

end NUMINAMATH_GPT_count_integers_in_range_l1362_136265


namespace NUMINAMATH_GPT_find_cost_per_batch_l1362_136273

noncomputable def cost_per_tire : ℝ := 8
noncomputable def selling_price_per_tire : ℝ := 20
noncomputable def profit_per_tire : ℝ := 10.5
noncomputable def number_of_tires : ℕ := 15000

noncomputable def total_cost (C : ℝ) : ℝ := C + cost_per_tire * number_of_tires
noncomputable def total_revenue : ℝ := selling_price_per_tire * number_of_tires
noncomputable def total_profit : ℝ := profit_per_tire * number_of_tires

theorem find_cost_per_batch (C : ℝ) :
  total_profit = total_revenue - total_cost C → C = 22500 := by
  sorry

end NUMINAMATH_GPT_find_cost_per_batch_l1362_136273


namespace NUMINAMATH_GPT_sequence_perfect_square_l1362_136217

theorem sequence_perfect_square (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) :
  ∃! n, ∃ k, a n = k ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_perfect_square_l1362_136217


namespace NUMINAMATH_GPT_ratio_of_place_values_l1362_136241

def thousands_place_value : ℝ := 1000
def tenths_place_value : ℝ := 0.1

theorem ratio_of_place_values : thousands_place_value / tenths_place_value = 10000 := by
  sorry

end NUMINAMATH_GPT_ratio_of_place_values_l1362_136241


namespace NUMINAMATH_GPT_find_a4_l1362_136238

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) * a (n - 1) = a n * a n

def given_sequence_conditions (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 2 + a 6 = 34 ∧ a 3 * a 5 = 64

-- Statement
theorem find_a4 (a : ℕ → ℝ) (h : given_sequence_conditions a) : a 4 = 8 :=
sorry

end NUMINAMATH_GPT_find_a4_l1362_136238


namespace NUMINAMATH_GPT_min_length_MN_l1362_136229

theorem min_length_MN (a b : ℝ) (H h : ℝ) (MN : ℝ) (midsegment_eq_4 : (a + b) / 2 = 4)
    (area_div_eq_half : (a + MN) / 2 * h = (MN + b) / 2 * H) : MN = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_length_MN_l1362_136229


namespace NUMINAMATH_GPT_lesser_number_l1362_136285

theorem lesser_number (x y : ℕ) (h1 : x + y = 58) (h2 : x - y = 6) : y = 26 :=
by
  sorry

end NUMINAMATH_GPT_lesser_number_l1362_136285


namespace NUMINAMATH_GPT_no_such_cuboid_exists_l1362_136267

theorem no_such_cuboid_exists (a b c : ℝ) :
  a + b + c = 12 ∧ ab + bc + ca = 1 ∧ abc = 12 → false :=
by
  sorry

end NUMINAMATH_GPT_no_such_cuboid_exists_l1362_136267


namespace NUMINAMATH_GPT_painting_price_decrease_l1362_136211

theorem painting_price_decrease (P : ℝ) (h1 : 1.10 * P - 0.935 * P = x * 1.10 * P) :
  x = 0.15 := by
  sorry

end NUMINAMATH_GPT_painting_price_decrease_l1362_136211


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1362_136206

theorem isosceles_triangle_base_length :
  ∃ (x y : ℝ), 
    ((x + x / 2 = 15 ∧ y + x / 2 = 6) ∨ (x + x / 2 = 6 ∧ y + x / 2 = 15)) ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1362_136206


namespace NUMINAMATH_GPT_find_pairs_gcd_lcm_l1362_136299

theorem find_pairs_gcd_lcm : 
  { (a, b) : ℕ × ℕ | Nat.gcd a b = 24 ∧ Nat.lcm a b = 360 } = {(24, 360), (72, 120)} := 
by
  sorry

end NUMINAMATH_GPT_find_pairs_gcd_lcm_l1362_136299


namespace NUMINAMATH_GPT_branches_count_eq_6_l1362_136266

theorem branches_count_eq_6 (x : ℕ) (h : 1 + x + x^2 = 43) : x = 6 :=
sorry

end NUMINAMATH_GPT_branches_count_eq_6_l1362_136266


namespace NUMINAMATH_GPT_cast_cost_l1362_136291

theorem cast_cost (C : ℝ) 
  (visit_cost : ℝ := 300)
  (insurance_coverage : ℝ := 0.60)
  (out_of_pocket_cost : ℝ := 200) :
  0.40 * (visit_cost + C) = out_of_pocket_cost → 
  C = 200 := by
  sorry

end NUMINAMATH_GPT_cast_cost_l1362_136291


namespace NUMINAMATH_GPT_route_down_distance_l1362_136297

-- Definitions
def rate_up : ℝ := 7
def time_up : ℝ := 2
def distance_up : ℝ := rate_up * time_up
def rate_down : ℝ := 1.5 * rate_up
def time_down : ℝ := time_up
def distance_down : ℝ := rate_down * time_down

-- Theorem
theorem route_down_distance : distance_down = 21 := by
  sorry

end NUMINAMATH_GPT_route_down_distance_l1362_136297


namespace NUMINAMATH_GPT_find_number_l1362_136232

noncomputable def number_divided_by_seven_is_five_fourteen (x : ℝ) : Prop :=
  x / 7 = 5 / 14

theorem find_number (x : ℝ) (h : number_divided_by_seven_is_five_fourteen x) : x = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1362_136232


namespace NUMINAMATH_GPT_Kim_total_hours_l1362_136278

-- Define the initial conditions
def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_class : ℕ := 1

-- The proof problem: Given the initial conditions, prove the total hours of classes per day is 6
theorem Kim_total_hours : (initial_classes - dropped_class) * hours_per_class = 6 := by
  sorry

end NUMINAMATH_GPT_Kim_total_hours_l1362_136278


namespace NUMINAMATH_GPT_sum_remainder_l1362_136227

theorem sum_remainder (p q r : ℕ) (hp : p % 15 = 11) (hq : q % 15 = 13) (hr : r % 15 = 14) : 
  (p + q + r) % 15 = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_remainder_l1362_136227


namespace NUMINAMATH_GPT_probability_of_one_pair_one_triplet_l1362_136212

-- Define the necessary conditions
def six_sided_die_rolls (n : ℕ) : ℕ := 6 ^ n

def successful_outcomes : ℕ :=
  6 * 20 * 5 * 3 * 4

def total_outcomes : ℕ :=
  six_sided_die_rolls 6

def probability_success : ℚ :=
  successful_outcomes / total_outcomes

-- The theorem we want to prove
theorem probability_of_one_pair_one_triplet :
  probability_success = 25/162 :=
sorry

end NUMINAMATH_GPT_probability_of_one_pair_one_triplet_l1362_136212


namespace NUMINAMATH_GPT_triangle_ABC_properties_l1362_136239

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem triangle_ABC_properties
  (xA xB xC : ℝ)
  (h_seq : xA < xB ∧ xB < xC ∧ 2 * xB = xA + xC)
  : (f xB + (f xA + f xC) / 2 > f ((xA + xC) / 2)) ∧ (f xA ≠ f xB ∧ f xB ≠ f xC) := 
sorry

end NUMINAMATH_GPT_triangle_ABC_properties_l1362_136239


namespace NUMINAMATH_GPT_length_of_MN_l1362_136225

noncomputable def curve_eq (α : ℝ) : ℝ × ℝ := (2 * Real.cos α + 1, 2 * Real.sin α)

noncomputable def line_eq (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem length_of_MN : ∀ (M N : ℝ × ℝ), 
  M ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2)^2 = 4} ∧
  N ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2)^2 = 4} ∧
  M ∈ {p : ℝ × ℝ | p.1 + p.2 = 2} ∧
  N ∈ {p : ℝ × ℝ | p.1 + p.2 = 2} →
  dist M N = Real.sqrt 14 :=
by
  sorry

end NUMINAMATH_GPT_length_of_MN_l1362_136225


namespace NUMINAMATH_GPT_work_completes_in_39_days_l1362_136218

theorem work_completes_in_39_days 
  (amit_days : ℕ := 15)  -- Amit can complete work in 15 days
  (ananthu_days : ℕ := 45)  -- Ananthu can complete work in 45 days
  (amit_worked_days : ℕ := 3)  -- Amit worked for 3 days
  : (amit_worked_days + ((4 / 5) / (1 / ananthu_days))) = 39 :=
by
  sorry

end NUMINAMATH_GPT_work_completes_in_39_days_l1362_136218


namespace NUMINAMATH_GPT_find_positive_integer_pairs_l1362_136251

theorem find_positive_integer_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a^2 = 3 * b^3) ↔ ∃ d : ℕ, 0 < d ∧ a = 18 * d^3 ∧ b = 6 * d^2 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integer_pairs_l1362_136251


namespace NUMINAMATH_GPT_find_inverse_sum_l1362_136210

variable {R : Type*} [OrderedRing R]

-- Define the function f and its inverse
variable (f : R → R)
variable (f_inv : R → R)

-- Conditions
axiom f_inverse : ∀ y, f (f_inv y) = y
axiom f_prop : ∀ x, f x + f (1 - x) = 2

-- The theorem we need to prove
theorem find_inverse_sum (x : R) : f_inv (x - 2) + f_inv (4 - x) = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_inverse_sum_l1362_136210


namespace NUMINAMATH_GPT_cone_height_l1362_136201

noncomputable def height_of_cone (r : ℝ) (n : ℕ) : ℝ :=
  let sector_circumference := (2 * Real.pi * r) / n
  let cone_base_radius := sector_circumference / (2 * Real.pi)
  Real.sqrt (r^2 - cone_base_radius^2)

theorem cone_height
  (r_original : ℝ)
  (n : ℕ)
  (h : r_original = 10)
  (hc : n = 4) :
  height_of_cone r_original n = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_cone_height_l1362_136201


namespace NUMINAMATH_GPT_articles_produced_l1362_136284

theorem articles_produced (a b c p q r : Nat) (h : a * b * c = abc) : p * q * r = pqr := sorry

end NUMINAMATH_GPT_articles_produced_l1362_136284


namespace NUMINAMATH_GPT_perimeter_triangle_ABC_l1362_136269

-- Define the conditions and statement
theorem perimeter_triangle_ABC 
  (r : ℝ) (AP PB altitude : ℝ) 
  (h1 : r = 30) 
  (h2 : AP = 26) 
  (h3 : PB = 32) 
  (h4 : altitude = 96) :
  (2 * (58 + 34.8)) = 185.6 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_triangle_ABC_l1362_136269


namespace NUMINAMATH_GPT_n_squared_plus_n_is_even_l1362_136271

theorem n_squared_plus_n_is_even (n : ℤ) : Even (n^2 + n) :=
by
  sorry

end NUMINAMATH_GPT_n_squared_plus_n_is_even_l1362_136271


namespace NUMINAMATH_GPT_gcd_of_1887_and_2091_is_51_l1362_136221

variable (a b : Nat)
variable (coefficient1 coefficient2 quotient1 quotient2 quotient3 remainder1 remainder2 : Nat)

def gcd_condition1 : Prop := (b = 1 * a + remainder1)
def gcd_condition2 : Prop := (a = quotient1 * remainder1 + remainder2)
def gcd_condition3 : Prop := (remainder1 = quotient2 * remainder2)

def numbers_1887_and_2091 : Prop := (a = 1887) ∧ (b = 2091)

theorem gcd_of_1887_and_2091_is_51 :
  numbers_1887_and_2091 a b ∧
  gcd_condition1 a b remainder1 ∧ 
  gcd_condition2 a remainder1 remainder2 quotient1 ∧ 
  gcd_condition3 remainder1 remainder2 quotient2 → 
  Nat.gcd 1887 2091 = 51 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_1887_and_2091_is_51_l1362_136221


namespace NUMINAMATH_GPT_triangle_inequality_proof_l1362_136202

theorem triangle_inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
    a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
    sorry

end NUMINAMATH_GPT_triangle_inequality_proof_l1362_136202


namespace NUMINAMATH_GPT_calculate_expression_l1362_136236

theorem calculate_expression : (4 + Real.sqrt 6) * (4 - Real.sqrt 6) = 10 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1362_136236


namespace NUMINAMATH_GPT_quadratic_has_one_real_solution_l1362_136213

theorem quadratic_has_one_real_solution (m : ℝ) : (∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → m = 6 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_one_real_solution_l1362_136213


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l1362_136222

theorem distance_between_parallel_lines (r d : ℝ) 
  (h1 : ∃ p1 p2 p3 : ℝ, p1 = 40 ∧ p2 = 40 ∧ p3 = 36) 
  (h2 : ∀ θ : ℝ, ∃ A B C D : ℝ → ℝ, 
    (A θ - B θ) = 40 ∧ (C θ - D θ) = 36) : d = 6 :=
sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l1362_136222


namespace NUMINAMATH_GPT_number_of_children_l1362_136293

theorem number_of_children (total_people : ℕ) (num_adults num_children : ℕ)
  (h1 : total_people = 42)
  (h2 : num_children = 2 * num_adults)
  (h3 : num_adults + num_children = total_people) :
  num_children = 28 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l1362_136293


namespace NUMINAMATH_GPT_product_of_roots_is_12_l1362_136233

theorem product_of_roots_is_12 :
  (81 ^ (1 / 4) * 8 ^ (1 / 3) * 4 ^ (1 / 2)) = 12 := by
  sorry

end NUMINAMATH_GPT_product_of_roots_is_12_l1362_136233


namespace NUMINAMATH_GPT_painting_ways_correct_l1362_136248

noncomputable def num_ways_to_paint : ℕ :=
  let red := 1
  let green_or_blue := 2
  let total_ways_case1 := red
  let total_ways_case2 := (green_or_blue ^ 4)
  let total_ways_case3 := green_or_blue ^ 3
  let total_ways_case4 := green_or_blue ^ 2
  let total_ways_case5 := green_or_blue
  let total_ways_case6 := red
  total_ways_case1 + total_ways_case2 + total_ways_case3 + total_ways_case4 + total_ways_case5 + total_ways_case6

theorem painting_ways_correct : num_ways_to_paint = 32 :=
  by
  sorry

end NUMINAMATH_GPT_painting_ways_correct_l1362_136248


namespace NUMINAMATH_GPT_strictly_increasing_arithmetic_seq_l1362_136250

theorem strictly_increasing_arithmetic_seq 
  (s : ℕ → ℕ) 
  (hs_incr : ∀ n, s n < s (n + 1)) 
  (hs_seq1 : ∃ D1, ∀ n, s (s n) = s (s 0) + n * D1) 
  (hs_seq2 : ∃ D2, ∀ n, s (s n + 1) = s (s 0 + 1) + n * D2) : 
  ∃ d, ∀ n, s (n + 1) = s n + d :=
sorry

end NUMINAMATH_GPT_strictly_increasing_arithmetic_seq_l1362_136250


namespace NUMINAMATH_GPT_hotel_friends_count_l1362_136288

theorem hotel_friends_count
  (n : ℕ)
  (friend_share extra friend_payment : ℕ)
  (h1 : 7 * 80 + friend_payment = 720)
  (h2 : friend_payment = friend_share + extra)
  (h3 : friend_payment = 160)
  (h4 : extra = 70)
  (h5 : friend_share = 90) :
  n = 8 :=
sorry

end NUMINAMATH_GPT_hotel_friends_count_l1362_136288


namespace NUMINAMATH_GPT_parabola_conditions_l1362_136275

theorem parabola_conditions 
  (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b = 2 * a) 
  (hc : c = -3 * a) 
  (hA : a * (-3)^2 + b * (-3) + c = 0) 
  (hB : a * (1)^2 + b * (1) + c = 0) : 
  (b^2 - 4 * a * c > 0) ∧ (3 * b + 2 * c = 0) :=
sorry

end NUMINAMATH_GPT_parabola_conditions_l1362_136275


namespace NUMINAMATH_GPT_quadratic_func_max_value_l1362_136252

theorem quadratic_func_max_value (b c x y : ℝ) (h1 : y = -x^2 + b * x + c)
(h1_x1 : (y = 0) → x = -1 ∨ x = 3) :
    -x^2 + 2 * x + 3 ≤ 4 :=
sorry

end NUMINAMATH_GPT_quadratic_func_max_value_l1362_136252


namespace NUMINAMATH_GPT_hypotenuse_length_l1362_136215

theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : a^2 + b^2 = c^2) : c = 13 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1362_136215


namespace NUMINAMATH_GPT_gcd_7_fact_10_fact_div_4_fact_eq_5040_l1362_136228

def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

noncomputable def quotient_fact (a b : ℕ) : ℕ := fact a / fact b

theorem gcd_7_fact_10_fact_div_4_fact_eq_5040 :
  Nat.gcd (fact 7) (quotient_fact 10 4) = 5040 := by
sorry

end NUMINAMATH_GPT_gcd_7_fact_10_fact_div_4_fact_eq_5040_l1362_136228


namespace NUMINAMATH_GPT_remainder_of_17_power_1801_mod_28_l1362_136264

theorem remainder_of_17_power_1801_mod_28 : (17 ^ 1801) % 28 = 17 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_17_power_1801_mod_28_l1362_136264


namespace NUMINAMATH_GPT_remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one_l1362_136280

theorem remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one :
  ((x - 1) ^ 2028) % (x^2 - x + 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one_l1362_136280


namespace NUMINAMATH_GPT_article_production_l1362_136296

-- Conditions
variables (x z : ℕ) (hx : 0 < x) (hz : 0 < z)
-- The given condition: x men working x hours a day for x days produce 2x^2 articles.
def articles_produced_x (x : ℕ) : ℕ := 2 * x^2

-- The question: the number of articles produced by z men working z hours a day for z days
def articles_produced_z (x z : ℕ) : ℕ := 2 * z^3 / x

-- Prove that the number of articles produced by z men working z hours a day for z days is 2 * (z^3) / x
theorem article_production (hx : 0 < x) (hz : 0 < z) :
  articles_produced_z x z = 2 * z^3 / x :=
sorry

end NUMINAMATH_GPT_article_production_l1362_136296


namespace NUMINAMATH_GPT_find_D_plus_E_plus_F_l1362_136253

noncomputable def g (x : ℝ) (D E F : ℝ) : ℝ := (x^2) / (D * x^2 + E * x + F)

theorem find_D_plus_E_plus_F (D E F : ℤ) 
  (h1 : ∀ x : ℝ, x > 3 → g x D E F > 0.3)
  (h2 : ∀ x : ℝ, ¬(D * x^2 + E * x + F = 0 ↔ (x = -3 ∨ x = 2))) :
  D + E + F = -8 :=
sorry

end NUMINAMATH_GPT_find_D_plus_E_plus_F_l1362_136253


namespace NUMINAMATH_GPT_color_stamps_sold_l1362_136279

theorem color_stamps_sold :
    let total_stamps : ℕ := 1102609
    let black_and_white_stamps : ℕ := 523776
    total_stamps - black_and_white_stamps = 578833 := 
by
  sorry

end NUMINAMATH_GPT_color_stamps_sold_l1362_136279


namespace NUMINAMATH_GPT_before_lunch_rush_customers_l1362_136292

def original_customers_before_lunch := 29
def added_customers_during_lunch := 20
def customers_no_tip := 34
def customers_tip := 15

theorem before_lunch_rush_customers : 
  original_customers_before_lunch + added_customers_during_lunch = customers_no_tip + customers_tip → 
  original_customers_before_lunch = 29 := 
by
  sorry

end NUMINAMATH_GPT_before_lunch_rush_customers_l1362_136292


namespace NUMINAMATH_GPT_expected_value_ball_draw_l1362_136207

noncomputable def E_xi : ℚ :=
  let prob_xi_2 := 3/5
  let prob_xi_3 := 3/10
  let prob_xi_4 := 1/10
  2 * prob_xi_2 + 3 * prob_xi_3 + 4 * prob_xi_4

theorem expected_value_ball_draw : E_xi = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_expected_value_ball_draw_l1362_136207


namespace NUMINAMATH_GPT_A_loses_240_l1362_136240

def initial_house_value : ℝ := 12000
def house_value_after_A_sells : ℝ := initial_house_value * 0.85
def house_value_after_B_sells_back : ℝ := house_value_after_A_sells * 1.2

theorem A_loses_240 : house_value_after_B_sells_back - initial_house_value = 240 := by
  sorry

end NUMINAMATH_GPT_A_loses_240_l1362_136240


namespace NUMINAMATH_GPT_calculate_PC_l1362_136295
noncomputable def ratio (a b : ℝ) : ℝ := a / b

theorem calculate_PC (AB BC CA PC PA : ℝ) (h1: AB = 6) (h2: BC = 10) (h3: CA = 8)
  (h4: ratio PC PA = ratio 8 6)
  (h5: ratio PA (PC + 10) = ratio 6 10) :
  PC = 40 :=
sorry

end NUMINAMATH_GPT_calculate_PC_l1362_136295


namespace NUMINAMATH_GPT_symmetric_points_sum_l1362_136298

variable {p q : ℤ}

theorem symmetric_points_sum (h1 : p = -6) (h2 : q = 2) : p + q = -4 := by
  sorry

end NUMINAMATH_GPT_symmetric_points_sum_l1362_136298


namespace NUMINAMATH_GPT_factorization_correct_l1362_136274

theorem factorization_correct : ∀ y : ℝ, y^2 - 4*y + 4 = (y - 2)^2 := by
  intro y
  sorry

end NUMINAMATH_GPT_factorization_correct_l1362_136274


namespace NUMINAMATH_GPT_election_max_k_1002_l1362_136255

/-- There are 2002 candidates initially. 
In each round, one candidate with the least number of votes is eliminated unless a candidate receives more than half the votes.
Determine the highest possible value of k if Ostap Bender is elected in the 1002nd round. -/
theorem election_max_k_1002 
  (number_of_candidates : ℕ)
  (number_of_rounds : ℕ)
  (k : ℕ)
  (h1 : number_of_candidates = 2002)
  (h2 : number_of_rounds = 1002)
  (h3 : k ≤ number_of_candidates - 1)
  (h4 : ∀ n : ℕ, n < number_of_rounds → (k + n) % (number_of_candidates - n) ≠ 0) : 
  k = 2001 := sorry

end NUMINAMATH_GPT_election_max_k_1002_l1362_136255


namespace NUMINAMATH_GPT_chairs_per_row_l1362_136256

/-- There are 10 rows of chairs, with the first row for awardees, the second and third rows for
    administrators and teachers, the last two rows for parents, and the remaining five rows for students.
    Given that 4/5 of the student seats are occupied, and there are 15 vacant seats among the students,
    proves that the number of chairs per row is 15. --/
theorem chairs_per_row (x : ℕ) (h1 : 10 = 1 + 1 + 1 + 5 + 2)
  (h2 : 4 / 5 * (5 * x) + 1 / 5 * (5 * x) = 5 * x)
  (h3 : 1 / 5 * (5 * x) = 15) : x = 15 :=
sorry

end NUMINAMATH_GPT_chairs_per_row_l1362_136256


namespace NUMINAMATH_GPT_find_sum_l1362_136226

-- Define the prime conditions
variables (P : ℝ) (SI15 SI12 : ℝ)

-- Assume conditions for the problem
axiom h1 : SI15 = P * 15 / 100 * 2
axiom h2 : SI12 = P * 12 / 100 * 2
axiom h3 : SI15 - SI12 = 840

-- Prove that P = 14000
theorem find_sum : P = 14000 :=
sorry

end NUMINAMATH_GPT_find_sum_l1362_136226


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1362_136223

noncomputable def U : Set ℤ := {-3, -1, 0, 1, 3}

noncomputable def A : Set ℤ := {x | x^2 - 2 * x - 3 = 0}

theorem complement_of_A_in_U : (U \ A) = {-3, 0, 1} :=
by sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1362_136223


namespace NUMINAMATH_GPT_number_of_candidates_is_three_l1362_136243

variable (votes : List ℕ) (totalVotes : ℕ)

def determineNumberOfCandidates (votes : List ℕ) (totalVotes : ℕ) : ℕ :=
  votes.length

theorem number_of_candidates_is_three (V : ℕ) 
  (h_votes : [2500, 5000, 20000].sum = V) 
  (h_percent : 20000 = 7273 / 10000 * V): 
  determineNumberOfCandidates [2500, 5000, 20000] V = 3 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_candidates_is_three_l1362_136243


namespace NUMINAMATH_GPT_price_of_second_tea_l1362_136246

theorem price_of_second_tea (P : ℝ) (h1 : 1 * 64 + 1 * P = 2 * 69) : P = 74 := 
by
  sorry

end NUMINAMATH_GPT_price_of_second_tea_l1362_136246


namespace NUMINAMATH_GPT_percentage_of_temporary_workers_l1362_136230

theorem percentage_of_temporary_workers (total_workers technicians non_technicians permanent_technicians permanent_non_technicians : ℕ) 
  (h1 : total_workers = 100)
  (h2 : technicians = total_workers / 2) 
  (h3 : non_technicians = total_workers / 2) 
  (h4 : permanent_technicians = technicians / 2) 
  (h5 : permanent_non_technicians = non_technicians / 2) :
  ((total_workers - (permanent_technicians + permanent_non_technicians)) / total_workers) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_temporary_workers_l1362_136230


namespace NUMINAMATH_GPT_lcm_fractions_l1362_136237

theorem lcm_fractions (x : ℕ) (hx : x > 0) :
  lcm (1 / (2 * x)) (lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (12 * x)))) = 1 / (12 * x) :=
sorry

end NUMINAMATH_GPT_lcm_fractions_l1362_136237


namespace NUMINAMATH_GPT_anna_phone_chargers_l1362_136263

theorem anna_phone_chargers (p l : ℕ) (h₁ : l = 5 * p) (h₂ : l + p = 24) : p = 4 :=
by
  sorry

end NUMINAMATH_GPT_anna_phone_chargers_l1362_136263


namespace NUMINAMATH_GPT_find_initial_marbles_l1362_136282

-- Definitions based on conditions
def loses_to_street (initial_marbles : ℕ) : ℕ := initial_marbles - (initial_marbles * 60 / 100)
def loses_to_sewer (marbles_after_street : ℕ) : ℕ := marbles_after_street / 2

-- The given number of marbles left
def remaining_marbles : ℕ := 20

-- Proof statement
theorem find_initial_marbles (initial_marbles : ℕ) : 
  loses_to_sewer (loses_to_street initial_marbles) = remaining_marbles -> 
  initial_marbles = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_marbles_l1362_136282


namespace NUMINAMATH_GPT_quadratic_has_real_solutions_iff_l1362_136242

theorem quadratic_has_real_solutions_iff (m : ℝ) :
  ∃ x y : ℝ, (y = m * x + 3) ∧ (y = (3 * m - 2) * x ^ 2 + 5) ↔ 
  (m ≤ 12 - 8 * Real.sqrt 2) ∨ (m ≥ 12 + 8 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_solutions_iff_l1362_136242


namespace NUMINAMATH_GPT_convex_polyhedron_in_inscribed_sphere_l1362_136294

-- Definitions based on conditions
variables (S c r : ℝ) (S' V R : ℝ)

-- The given relationship for a convex polygon.
def poly_relationship := S = (1 / 2) * c * r

-- The desired relationship for a convex polyhedron.
def polyhedron_relationship := V = (1 / 3) * S' * R

-- Proof statement
theorem convex_polyhedron_in_inscribed_sphere (S c r S' V R : ℝ) 
  (poly : S = (1 / 2) * c * r) : V = (1 / 3) * S' * R :=
sorry

end NUMINAMATH_GPT_convex_polyhedron_in_inscribed_sphere_l1362_136294


namespace NUMINAMATH_GPT_complex_z_power_l1362_136286

theorem complex_z_power:
  ∀ (z : ℂ), (z + 1/z = 2 * Real.cos (5 * Real.pi / 180)) →
  z^1000 + (1/z)^1000 = 2 * Real.cos (20 * Real.pi / 180) :=
by
  sorry

end NUMINAMATH_GPT_complex_z_power_l1362_136286


namespace NUMINAMATH_GPT_largest_consecutive_multiple_l1362_136289

theorem largest_consecutive_multiple (n : ℕ) (h : 3 * n + 3 * (n + 1) + 3 * (n + 2) = 117) : 3 * (n + 2) = 42 :=
sorry

end NUMINAMATH_GPT_largest_consecutive_multiple_l1362_136289


namespace NUMINAMATH_GPT_ratio_of_areas_l1362_136276
-- Define the conditions and the ratio to be proven
theorem ratio_of_areas (t r : ℝ) (h : 3 * t = 2 * π * r) : 
  (π^2 / 18) = (π^2 * r^2 / 9) / (2 * r^2) :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1362_136276


namespace NUMINAMATH_GPT_simplify_expr_l1362_136203

variable (a b : ℤ)  -- assuming a and b are elements of the ring ℤ

theorem simplify_expr : 105 * a - 38 * a + 27 * b - 12 * b = 67 * a + 15 * b := 
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1362_136203


namespace NUMINAMATH_GPT_parabola_intersection_prob_l1362_136272

noncomputable def prob_intersect_parabolas : ℚ :=
  57 / 64

theorem parabola_intersection_prob :
  ∀ (a b c d : ℤ), (1 ≤ a ∧ a ≤ 8) → (1 ≤ b ∧ b ≤ 8) →
  (1 ≤ c∧ c ≤ 8) → (1 ≤ d ∧ d ≤ 8) →
  prob_intersect_parabolas = 57 / 64 :=
by
  intros a b c d ha hb hc hd
  sorry

end NUMINAMATH_GPT_parabola_intersection_prob_l1362_136272


namespace NUMINAMATH_GPT_complex_problem_l1362_136287

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem complex_problem :
  (1 - z) * (1 - z^2) * (1 - z^3) * (1 - z^4) = 5 :=
by
  sorry

end NUMINAMATH_GPT_complex_problem_l1362_136287


namespace NUMINAMATH_GPT_negation_of_universal_l1362_136290

theorem negation_of_universal : (¬ ∀ x : ℝ, x^2 + 2 * x - 1 = 0) ↔ ∃ x : ℝ, x^2 + 2 * x - 1 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_l1362_136290


namespace NUMINAMATH_GPT_admin_fee_percentage_l1362_136259

noncomputable def percentage_deducted_for_admin_fees 
  (amt_johnson : ℕ) (amt_sutton : ℕ) (amt_rollin : ℕ)
  (amt_school : ℕ) (amt_after_deduction : ℕ) : ℚ :=
  ((amt_school - amt_after_deduction) * 100) / amt_school

theorem admin_fee_percentage : 
  ∃ (amt_johnson amt_sutton amt_rollin amt_school amt_after_deduction : ℕ),
  amt_johnson = 2300 ∧
  amt_johnson = 2 * amt_sutton ∧
  amt_sutton * 8 = amt_rollin ∧
  amt_rollin * 3 = amt_school ∧
  amt_after_deduction = 27048 ∧
  percentage_deducted_for_admin_fees amt_johnson amt_sutton amt_rollin amt_school amt_after_deduction = 2 :=
by
  sorry

end NUMINAMATH_GPT_admin_fee_percentage_l1362_136259


namespace NUMINAMATH_GPT_sin_trig_identity_l1362_136281

theorem sin_trig_identity (α : ℝ) (h : Real.sin (α - π/4) = 1/2) : Real.sin ((5 * π) / 4 - α) = 1/2 := 
by 
  sorry

end NUMINAMATH_GPT_sin_trig_identity_l1362_136281


namespace NUMINAMATH_GPT_projectile_height_reaches_49_l1362_136249

theorem projectile_height_reaches_49 (t : ℝ) :
  (∃ t : ℝ, 49 = -20 * t^2 + 100 * t) → t = 0.7 :=
by
  sorry

end NUMINAMATH_GPT_projectile_height_reaches_49_l1362_136249


namespace NUMINAMATH_GPT_residue_12_2040_mod_19_l1362_136231

theorem residue_12_2040_mod_19 :
  12^2040 % 19 = 7 := 
sorry

end NUMINAMATH_GPT_residue_12_2040_mod_19_l1362_136231


namespace NUMINAMATH_GPT_weeding_planting_support_l1362_136214

-- Definitions based on conditions
def initial_weeding := 31
def initial_planting := 18
def additional_support := 20

-- Let x be the number of people sent to support weeding.
variable (x : ℕ)

-- The equation to prove.
theorem weeding_planting_support :
  initial_weeding + x = 2 * (initial_planting + (additional_support - x)) :=
sorry

end NUMINAMATH_GPT_weeding_planting_support_l1362_136214


namespace NUMINAMATH_GPT_correct_multiplication_l1362_136244

theorem correct_multiplication :
  ∃ (n : ℕ), 98765 * n = 888885 ∧ (98765 * n = 867559827931 → n = 9) :=
by
  sorry

end NUMINAMATH_GPT_correct_multiplication_l1362_136244


namespace NUMINAMATH_GPT_intersection_C_U_M_N_l1362_136258

open Set

-- Define U, M and N
def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

-- Define complement C_U M in U
def C_U_M : Set ℕ := U \ M

-- The theorem to prove
theorem intersection_C_U_M_N : (C_U_M ∩ N) = {3} := by
  sorry

end NUMINAMATH_GPT_intersection_C_U_M_N_l1362_136258


namespace NUMINAMATH_GPT_range_of_b_l1362_136208

theorem range_of_b (b : ℝ) : (∃ x : ℝ, |x - 2| + |x - 5| < b) → b > 3 :=
by 
-- This is where the proof would go.
sorry

end NUMINAMATH_GPT_range_of_b_l1362_136208


namespace NUMINAMATH_GPT_find_constants_l1362_136261

theorem find_constants :
  ∃ (A B C : ℚ), 
  (A = 1 ∧ B = 4 ∧ C = 1) ∧ 
  (∀ x, x ≠ -1 → x ≠ 3/2 → x ≠ 2 → 
    (6 * x^2 - 13 * x + 6) / (2 * x^3 + 3 * x^2 - 11 * x - 6) = 
    (A / (x + 1) + B / (2 * x - 3) + C / (x - 2))) :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l1362_136261


namespace NUMINAMATH_GPT_fish_caught_l1362_136254

noncomputable def total_fish_caught (chris_trips : ℕ) (chris_fish_per_trip : ℕ) (brian_trips : ℕ) (brian_fish_per_trip : ℕ) : ℕ :=
  chris_trips * chris_fish_per_trip + brian_trips * brian_fish_per_trip

theorem fish_caught (chris_trips : ℕ) (brian_factor : ℕ) (brian_fish_per_trip : ℕ) (ratio_numerator : ℕ) (ratio_denominator : ℕ) :
  chris_trips = 10 → brian_factor = 2 → brian_fish_per_trip = 400 → ratio_numerator = 3 → ratio_denominator = 5 →
  total_fish_caught chris_trips (brian_fish_per_trip * ratio_denominator / ratio_numerator) (chris_trips * brian_factor) brian_fish_per_trip = 14660 :=
by
  intros h_chris_trips h_brian_factor h_brian_fish_per_trip h_ratio_numer h_ratio_denom
  rw [h_chris_trips, h_brian_factor, h_brian_fish_per_trip, h_ratio_numer, h_ratio_denom]
  -- adding actual arithmetic would resolve the statement correctly
  sorry

end NUMINAMATH_GPT_fish_caught_l1362_136254


namespace NUMINAMATH_GPT_yellow_tiles_count_l1362_136257

theorem yellow_tiles_count
  (total_tiles : ℕ)
  (yellow_tiles : ℕ)
  (blue_tiles : ℕ)
  (purple_tiles : ℕ)
  (white_tiles : ℕ)
  (h1 : total_tiles = 20)
  (h2 : blue_tiles = yellow_tiles + 1)
  (h3 : purple_tiles = 6)
  (h4 : white_tiles = 7)
  (h5 : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  yellow_tiles = 3 :=
by sorry

end NUMINAMATH_GPT_yellow_tiles_count_l1362_136257


namespace NUMINAMATH_GPT_probability_of_winning_l1362_136277

def probability_of_losing : ℚ := 3 / 7

theorem probability_of_winning (h : probability_of_losing + p = 1) : p = 4 / 7 :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_winning_l1362_136277


namespace NUMINAMATH_GPT_value_of_kaftan_l1362_136235

theorem value_of_kaftan (K : ℝ) (h : (7 / 12) * (12 + K) = 5 + K) : K = 4.8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_kaftan_l1362_136235


namespace NUMINAMATH_GPT_parabola_expression_l1362_136268

theorem parabola_expression 
  (a b : ℝ) 
  (h : 9 = a * (-2)^2 + b * (-2) + 5) : 
  2 * a - b + 6 = 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_expression_l1362_136268


namespace NUMINAMATH_GPT_term_addition_k_to_kplus1_l1362_136219

theorem term_addition_k_to_kplus1 (k : ℕ) : 
  (2 * k + 2) + (2 * k + 3) = 4 * k + 5 := 
sorry

end NUMINAMATH_GPT_term_addition_k_to_kplus1_l1362_136219


namespace NUMINAMATH_GPT_x_varies_as_nth_power_of_z_l1362_136216

theorem x_varies_as_nth_power_of_z 
  (k j z : ℝ) 
  (h1 : ∃ y : ℝ, x = k * y^4 ∧ y = j * z^(1/3)) : 
  ∃ m : ℝ, x = m * z^(4/3) := 
 sorry

end NUMINAMATH_GPT_x_varies_as_nth_power_of_z_l1362_136216


namespace NUMINAMATH_GPT_mean_score_is_74_l1362_136283

theorem mean_score_is_74 (M SD : ℝ) 
  (h1 : 58 = M - 2 * SD) 
  (h2 : 98 = M + 3 * SD) : 
  M = 74 := 
by 
  -- problem statement without solving steps
  sorry

end NUMINAMATH_GPT_mean_score_is_74_l1362_136283


namespace NUMINAMATH_GPT_men_who_wore_glasses_l1362_136262

theorem men_who_wore_glasses (total_people : ℕ) (women_ratio men_with_glasses_ratio : ℚ)  
  (h_total : total_people = 1260) 
  (h_women_ratio : women_ratio = 7 / 18)
  (h_men_with_glasses_ratio : men_with_glasses_ratio = 6 / 11)
  : ∃ (men_with_glasses : ℕ), men_with_glasses = 420 := 
by
  sorry

end NUMINAMATH_GPT_men_who_wore_glasses_l1362_136262


namespace NUMINAMATH_GPT_doug_fires_l1362_136247

theorem doug_fires (D : ℝ) (Kai_fires : ℝ) (Eli_fires : ℝ) 
    (hKai : Kai_fires = 3 * D)
    (hEli : Eli_fires = 1.5 * D)
    (hTotal : D + Kai_fires + Eli_fires = 110) : 
  D = 20 := 
by
  sorry

end NUMINAMATH_GPT_doug_fires_l1362_136247


namespace NUMINAMATH_GPT_daniella_lap_time_l1362_136245

theorem daniella_lap_time
  (T_T : ℕ) (H_TT : T_T = 56)
  (meet_time : ℕ) (H_meet : meet_time = 24) :
  ∃ T_D : ℕ, T_D = 42 :=
by
  sorry

end NUMINAMATH_GPT_daniella_lap_time_l1362_136245


namespace NUMINAMATH_GPT_abs_neg_2023_eq_2023_neg_one_pow_2023_eq_neg_one_l1362_136204

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 :=
sorry

theorem neg_one_pow_2023_eq_neg_one : (-1 : ℤ) ^ 2023 = -1 :=
sorry

end NUMINAMATH_GPT_abs_neg_2023_eq_2023_neg_one_pow_2023_eq_neg_one_l1362_136204


namespace NUMINAMATH_GPT_final_total_cost_is_12_70_l1362_136234

-- Definitions and conditions
def sandwich_count : ℕ := 2
def sandwich_cost_per_unit : ℝ := 2.45

def soda_count : ℕ := 4
def soda_cost_per_unit : ℝ := 0.87

def chips_count : ℕ := 3
def chips_cost_per_unit : ℝ := 1.29

def sandwich_discount : ℝ := 0.10
def sales_tax : ℝ := 0.08

-- Final price after discount and tax
noncomputable def total_cost : ℝ :=
  let sandwiches_total := sandwich_count * sandwich_cost_per_unit
  let discounted_sandwiches := sandwiches_total * (1 - sandwich_discount)
  let sodas_total := soda_count * soda_cost_per_unit
  let chips_total := chips_count * chips_cost_per_unit
  let subtotal := discounted_sandwiches + sodas_total + chips_total
  let final_total := subtotal * (1 + sales_tax)
  final_total

theorem final_total_cost_is_12_70 : total_cost = 12.70 :=
by 
  sorry

end NUMINAMATH_GPT_final_total_cost_is_12_70_l1362_136234


namespace NUMINAMATH_GPT_line1_line2_line3_l1362_136200

-- Line 1: Through (-1, 3), parallel to x - 2y + 3 = 0.
theorem line1 (x y : ℝ) : (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 3) →
                              (x - 2 * y + 7 = 0) :=
by sorry

-- Line 2: Through (3, 4), perpendicular to 3x - y + 2 = 0.
theorem line2 (x y : ℝ) : (3 * x - y + 2 = 0) ∧ (x = 3) ∧ (y = 4) →
                              (x + 3 * y - 15 = 0) :=
by sorry

-- Line 3: Through (1, 2), with equal intercepts on both axes.
theorem line3 (x y : ℝ) : (x = y) ∧ (x = 1) ∧ (y = 2) →
                              (x + y - 3 = 0) :=
by sorry

end NUMINAMATH_GPT_line1_line2_line3_l1362_136200


namespace NUMINAMATH_GPT_mod_3_pow_2040_eq_1_mod_5_l1362_136224

theorem mod_3_pow_2040_eq_1_mod_5 :
  (3 ^ 2040) % 5 = 1 := by
  -- Here the theorem states that the remainder of 3^2040 when divided by 5 is equal to 1
  sorry

end NUMINAMATH_GPT_mod_3_pow_2040_eq_1_mod_5_l1362_136224
