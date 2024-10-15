import Mathlib

namespace NUMINAMATH_GPT_find_equation_of_BC_l1701_170153

theorem find_equation_of_BC :
  ∃ (BC : ℝ → ℝ → Prop), 
  (∀ x y, (BC x y ↔ 2 * x - y + 5 = 0)) :=
sorry

end NUMINAMATH_GPT_find_equation_of_BC_l1701_170153


namespace NUMINAMATH_GPT_incorrect_average_initially_l1701_170173

theorem incorrect_average_initially (S : ℕ) :
  (S + 25) / 10 = 46 ↔ (S + 65) / 10 = 50 := by
  sorry

end NUMINAMATH_GPT_incorrect_average_initially_l1701_170173


namespace NUMINAMATH_GPT_ratio_eq_two_l1701_170130

theorem ratio_eq_two (a b c d : ℤ) (h1 : b * c + a * d = 1) (h2 : a * c + 2 * b * d = 1) : 
  (a^2 + c^2 : ℚ) / (b^2 + d^2) = 2 :=
sorry

end NUMINAMATH_GPT_ratio_eq_two_l1701_170130


namespace NUMINAMATH_GPT_point_lies_on_graph_l1701_170176

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 2|

theorem point_lies_on_graph (a : ℝ) : f (-a) = f (a) :=
by
  sorry

end NUMINAMATH_GPT_point_lies_on_graph_l1701_170176


namespace NUMINAMATH_GPT_coprime_count_l1701_170149

theorem coprime_count (n : ℕ) (h : n = 56700000) : 
  ∃ m, m = 12960000 ∧ ∀ i < n, Nat.gcd i n = 1 → i < m :=
by
  sorry

end NUMINAMATH_GPT_coprime_count_l1701_170149


namespace NUMINAMATH_GPT_minimum_employees_needed_l1701_170127

theorem minimum_employees_needed
  (n_W : ℕ) (n_A : ℕ) (n_S : ℕ)
  (n_WA : ℕ) (n_AS : ℕ) (n_SW : ℕ)
  (n_WAS : ℕ)
  (h_W : n_W = 115)
  (h_A : n_A = 92)
  (h_S : n_S = 60)
  (h_WA : n_WA = 32)
  (h_AS : n_AS = 20)
  (h_SW : n_SW = 10)
  (h_WAS : n_WAS = 5) :
  n_W + n_A + n_S - (n_WA - n_WAS) - (n_AS - n_WAS) - (n_SW - n_WAS) + 2 * n_WAS = 225 :=
by
  sorry

end NUMINAMATH_GPT_minimum_employees_needed_l1701_170127


namespace NUMINAMATH_GPT_functional_equation_solution_exists_l1701_170156

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution_exists (f : ℝ → ℝ) (h : ∀ x y, 0 < x → 0 < y → f x * f y = 2 * f (x + y * f x)) :
  ∃ c : ℝ, ∀ x, 0 < x → f x = x + c := 
sorry

end NUMINAMATH_GPT_functional_equation_solution_exists_l1701_170156


namespace NUMINAMATH_GPT_sum_of_two_consecutive_negative_integers_l1701_170182

theorem sum_of_two_consecutive_negative_integers (n : ℤ) (h : n * (n + 1) = 812) (h_neg : n < 0 ∧ (n + 1) < 0) : 
  n + (n + 1) = -57 :=
sorry

end NUMINAMATH_GPT_sum_of_two_consecutive_negative_integers_l1701_170182


namespace NUMINAMATH_GPT_find_total_cows_l1701_170112

-- Definitions as per the conditions
variables (D C L H : ℕ)

-- Condition 1: Total number of legs
def total_legs : ℕ := 2 * D + 4 * C

-- Condition 2: Total number of heads
def total_heads : ℕ := D + C

-- Condition 3: Legs are 28 more than twice the number of heads
def legs_heads_relation : Prop := total_legs D C = 2 * total_heads D C + 28

-- The theorem to prove
theorem find_total_cows (h : legs_heads_relation D C) : C = 14 :=
sorry

end NUMINAMATH_GPT_find_total_cows_l1701_170112


namespace NUMINAMATH_GPT_simplify_complex_fraction_l1701_170146

theorem simplify_complex_fraction :
  (⟨3, 5⟩ : ℂ) / (⟨-2, 7⟩ : ℂ) = (29 / 53) - (31 / 53) * I :=
by sorry

end NUMINAMATH_GPT_simplify_complex_fraction_l1701_170146


namespace NUMINAMATH_GPT_eight_p_plus_one_is_composite_l1701_170167

theorem eight_p_plus_one_is_composite (p : ℕ) (hp : Nat.Prime p) (h8p1 : Nat.Prime (8 * p - 1)) : ¬ Nat.Prime (8 * p + 1) :=
by
  sorry

end NUMINAMATH_GPT_eight_p_plus_one_is_composite_l1701_170167


namespace NUMINAMATH_GPT_original_number_of_movies_l1701_170113

theorem original_number_of_movies (x : ℕ) (dvd blu_ray : ℕ)
  (h1 : dvd = 17 * x)
  (h2 : blu_ray = 4 * x)
  (h3 : 17 * x / (4 * x - 4) = 9 / 2) :
  dvd + blu_ray = 378 := by
  sorry

end NUMINAMATH_GPT_original_number_of_movies_l1701_170113


namespace NUMINAMATH_GPT_chairs_left_after_selling_l1701_170143

-- Definitions based on conditions
def chairs_before_selling : ℕ := 15
def difference_after_selling : ℕ := 12

-- Theorem statement based on the question
theorem chairs_left_after_selling : (chairs_before_selling - 3 = difference_after_selling) → (chairs_before_selling - difference_after_selling = 3) := by
  intro h
  sorry

end NUMINAMATH_GPT_chairs_left_after_selling_l1701_170143


namespace NUMINAMATH_GPT_field_trip_people_per_bus_l1701_170164

def number_of_people_on_each_bus (vans buses people_per_van total_people : ℕ) : ℕ :=
  (total_people - (vans * people_per_van)) / buses

theorem field_trip_people_per_bus :
  let vans := 9
  let buses := 10
  let people_per_van := 8
  let total_people := 342
  number_of_people_on_each_bus vans buses people_per_van total_people = 27 :=
by
  sorry

end NUMINAMATH_GPT_field_trip_people_per_bus_l1701_170164


namespace NUMINAMATH_GPT_sasha_age_l1701_170108

theorem sasha_age :
  ∃ a : ℕ, 
    (M = 2 * a - 3) ∧
    (M = a + (a - 3)) ∧
    (a = 3) :=
by
  sorry

end NUMINAMATH_GPT_sasha_age_l1701_170108


namespace NUMINAMATH_GPT_range_of_a_l1701_170185

theorem range_of_a (a : ℝ) (h : ∃ α β : ℝ, (α + β = -(a^2 - 1)) ∧ (α * β = a - 2) ∧ (1 < α ∧ β < 1) ∨ (α < 1 ∧ 1 < β)) :
  -2 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1701_170185


namespace NUMINAMATH_GPT_domain_f_l1701_170178

open Real

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2) - 3

theorem domain_f :
  {x : ℝ | g x > 0} = {x : ℝ | x < 0 ∨ x > 3} :=
by 
  sorry

end NUMINAMATH_GPT_domain_f_l1701_170178


namespace NUMINAMATH_GPT_chase_cardinals_count_l1701_170125

variable (gabrielle_robins : Nat)
variable (gabrielle_cardinals : Nat)
variable (gabrielle_blue_jays : Nat)
variable (chase_robins : Nat)
variable (chase_blue_jays : Nat)
variable (chase_cardinals : Nat)

variable (gabrielle_total : Nat)
variable (chase_total : Nat)

variable (percent_more : Nat)

axiom gabrielle_robins_def : gabrielle_robins = 5
axiom gabrielle_cardinals_def : gabrielle_cardinals = 4
axiom gabrielle_blue_jays_def : gabrielle_blue_jays = 3

axiom chase_robins_def : chase_robins = 2
axiom chase_blue_jays_def : chase_blue_jays = 3

axiom gabrielle_total_def : gabrielle_total = gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
axiom chase_total_def : chase_total = chase_robins + chase_blue_jays + chase_cardinals
axiom percent_more_def : percent_more = 20

axiom gabrielle_more_birds : gabrielle_total = Nat.ceil ((chase_total * (100 + percent_more)) / 100)

theorem chase_cardinals_count : chase_cardinals = 5 := by sorry

end NUMINAMATH_GPT_chase_cardinals_count_l1701_170125


namespace NUMINAMATH_GPT_inversely_proportional_x_y_l1701_170166

noncomputable def k := 320

theorem inversely_proportional_x_y (x y : ℕ) (h1 : x * y = k) :
  (∀ x, y = 10 → x = 32) ↔ (x = 32) :=
by
  sorry

end NUMINAMATH_GPT_inversely_proportional_x_y_l1701_170166


namespace NUMINAMATH_GPT_sequence_general_formula_l1701_170133

theorem sequence_general_formula (a : ℕ → ℕ) :
  (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 4) ∧ (a 4 = 8) ∧ (a 5 = 16) → ∀ n : ℕ, n > 0 → a n = 2^(n-1) :=
by
  intros h n hn
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l1701_170133


namespace NUMINAMATH_GPT_right_triangle_leg_length_l1701_170132

theorem right_triangle_leg_length (a b c : ℕ) (h_c : c = 13) (h_a : a = 12) (h_pythagorean : a^2 + b^2 = c^2) :
  b = 5 := 
by {
  -- Provide a placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_right_triangle_leg_length_l1701_170132


namespace NUMINAMATH_GPT_centimeters_per_inch_l1701_170110

theorem centimeters_per_inch (miles_per_map_inch : ℝ) (cm_measured : ℝ) (approx_miles : ℝ) (miles_per_inch : ℝ) (inches_from_cm : ℝ) : 
  miles_per_map_inch = 16 →
  inches_from_cm = 18.503937007874015 →
  miles_per_map_inch = 24 / 1.5 →
  approx_miles = 296.06299212598424 →
  cm_measured = 47 →
  (cm_measured / inches_from_cm) = 2.54 :=
by
  sorry

end NUMINAMATH_GPT_centimeters_per_inch_l1701_170110


namespace NUMINAMATH_GPT_option_costs_more_cost_effective_x30_more_cost_effective_plan_x30_l1701_170148

def racket_price : ℕ := 80
def ball_price : ℕ := 20
def discount_rate : ℕ := 90

def option_1_cost (n_rackets : ℕ) : ℕ :=
  n_rackets * racket_price

def option_2_cost (n_rackets : ℕ) (n_balls : ℕ) : ℕ :=
  (discount_rate * (n_rackets * racket_price + n_balls * ball_price)) / 100

-- Part 1: Express in Algebraic Terms
theorem option_costs (n_rackets : ℕ) (n_balls : ℕ) :
  option_1_cost n_rackets = 1600 ∧ option_2_cost n_rackets n_balls = 1440 + 18 * n_balls := 
by
  sorry

-- Part 2: For x = 30, determine more cost-effective option
theorem more_cost_effective_x30 (x : ℕ) (h : x = 30) :
  option_1_cost 20 < option_2_cost 20 x := 
by
  sorry

-- Part 3: More cost-effective Plan for x = 30
theorem more_cost_effective_plan_x30 :
  1600 + (discount_rate * (10 * ball_price)) / 100 < option_2_cost 20 30 :=
by
  sorry

end NUMINAMATH_GPT_option_costs_more_cost_effective_x30_more_cost_effective_plan_x30_l1701_170148


namespace NUMINAMATH_GPT_find_a_l1701_170129

theorem find_a (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 2) (h3 : c^2 / a = 3) : 
  a = 12^(1/7 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1701_170129


namespace NUMINAMATH_GPT_total_bags_l1701_170186

theorem total_bags (people : ℕ) (bags_per_person : ℕ) (h_people : people = 4) (h_bags_per_person : bags_per_person = 8) : people * bags_per_person = 32 := by
  sorry

end NUMINAMATH_GPT_total_bags_l1701_170186


namespace NUMINAMATH_GPT_angle_bisector_triangle_inequality_l1701_170128

theorem angle_bisector_triangle_inequality (AB AC D BD CD x : ℝ) (hAB : AB = 10) (hCD : CD = 3) (h_angle_bisector : BD = 30 / x)
  (h_triangle_inequality_1 : x + (BD + CD) > AB)
  (h_triangle_inequality_2 : AB + (BD + CD) > x)
  (h_triangle_inequality_3 : AB + x > BD + CD) :
  (3 < x) ∧ (x < 15) ∧ (3 + 15 = (18 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_angle_bisector_triangle_inequality_l1701_170128


namespace NUMINAMATH_GPT_fraction_spent_toy_store_l1701_170124

noncomputable def weekly_allowance : ℚ := 2.25
noncomputable def arcade_fraction_spent : ℚ := 3 / 5
noncomputable def candy_store_spent : ℚ := 0.60

theorem fraction_spent_toy_store :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction_spent)
  let spent_toy_store := remaining_after_arcade - candy_store_spent
  spent_toy_store / remaining_after_arcade = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_spent_toy_store_l1701_170124


namespace NUMINAMATH_GPT_product_telescope_identity_l1701_170163

theorem product_telescope_identity :
  (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * (1 + (1 / 5)) * (1 + (1 / 6)) * (1 + (1 / 7)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_product_telescope_identity_l1701_170163


namespace NUMINAMATH_GPT_Dawn_hourly_earnings_l1701_170126

theorem Dawn_hourly_earnings :
  let t_per_painting := 2 
  let num_paintings := 12
  let total_earnings := 3600
  let total_time := t_per_painting * num_paintings
  let hourly_wage := total_earnings / total_time
  hourly_wage = 150 := by
  sorry

end NUMINAMATH_GPT_Dawn_hourly_earnings_l1701_170126


namespace NUMINAMATH_GPT_bread_cooling_time_l1701_170198

theorem bread_cooling_time 
  (dough_room_temp : ℕ := 60)   -- 1 hour in minutes
  (shape_dough : ℕ := 15)       -- 15 minutes
  (proof_dough : ℕ := 120)      -- 2 hours in minutes
  (bake_bread : ℕ := 30)        -- 30 minutes
  (start_time : ℕ := 2 * 60)    -- 2:00 am in minutes
  (end_time : ℕ := 6 * 60)      -- 6:00 am in minutes
  : (end_time - start_time) - (dough_room_temp + shape_dough + proof_dough + bake_bread) = 15 := 
  by
  sorry

end NUMINAMATH_GPT_bread_cooling_time_l1701_170198


namespace NUMINAMATH_GPT_prob_win_3_1_correct_l1701_170104

-- Defining the probability for winning a game
def prob_win_game : ℚ := 2 / 3

-- Defining the probability for losing a game
def prob_lose_game : ℚ := 1 - prob_win_game

-- A function to calculate the probability of winning the match with a 3:1 score
def prob_win_3_1 : ℚ :=
  let combinations := 3 -- Number of ways to lose exactly 1 game in the first 3 games (C_3^1)
  let win_prob := prob_win_game ^ 3 -- Probability for winning 3 games
  let lose_prob := prob_lose_game -- Probability for losing 1 game
  combinations * win_prob * lose_prob

-- The theorem that states the probability that player A wins with a score of 3:1
theorem prob_win_3_1_correct : prob_win_3_1 = 8 / 27 := by
  sorry

end NUMINAMATH_GPT_prob_win_3_1_correct_l1701_170104


namespace NUMINAMATH_GPT_unique_intersection_point_l1701_170161

theorem unique_intersection_point {c : ℝ} :
  (∀ x y : ℝ, y = |x - 20| + |x + 18| → y = x + c → (x = 20 ∧ y = 38)) ↔ c = 18 :=
by
  sorry

end NUMINAMATH_GPT_unique_intersection_point_l1701_170161


namespace NUMINAMATH_GPT_product_units_digit_mod_10_l1701_170123

theorem product_units_digit_mod_10
  (u1 u2 u3 : ℕ)
  (hu1 : u1 = 2583 % 10)
  (hu2 : u2 = 7462 % 10)
  (hu3 : u3 = 93215 % 10) :
  ((2583 * 7462 * 93215) % 10) = 0 :=
by
  have h_units1 : u1 = 3 := by sorry
  have h_units2 : u2 = 2 := by sorry
  have h_units3 : u3 = 5 := by sorry
  have h_produce_units : ((3 * 2 * 5) % 10) = 0 := by sorry
  exact h_produce_units

end NUMINAMATH_GPT_product_units_digit_mod_10_l1701_170123


namespace NUMINAMATH_GPT_log_tan_ratio_l1701_170105

noncomputable def sin_add (α β : ℝ) : ℝ := Real.sin (α + β)
noncomputable def sin_sub (α β : ℝ) : ℝ := Real.sin (α - β)
noncomputable def tan_ratio (α β : ℝ) : ℝ := Real.tan α / Real.tan β

theorem log_tan_ratio (α β : ℝ)
  (h1 : sin_add α β = 1 / 2)
  (h2 : sin_sub α β = 1 / 3) :
  Real.logb 5 (tan_ratio α β) = 1 := by
sorry

end NUMINAMATH_GPT_log_tan_ratio_l1701_170105


namespace NUMINAMATH_GPT_find_positive_integer_n_l1701_170174

theorem find_positive_integer_n (n : ℕ) (h₁ : 200 % n = 5) (h₂ : 395 % n = 5) : n = 13 :=
sorry

end NUMINAMATH_GPT_find_positive_integer_n_l1701_170174


namespace NUMINAMATH_GPT_area_ratio_parallelogram_to_triangle_l1701_170117

variables {A B C D R E : Type*}
variables (s_AB s_AD : ℝ)

-- Given AR = 2/3 AB and AE = 1/3 AD
axiom AR_proportion : s_AB > 0 → s_AB * (2/3) = s_AB
axiom AE_proportion : s_AD > 0 → s_AD * (1/3) = s_AD

-- Given the relationship, we need to prove
theorem area_ratio_parallelogram_to_triangle (hAB : s_AB > 0) (hAD : s_AD > 0) :
  ∃ (S_ABCD S_ARE : ℝ), S_ABCD / S_ARE = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_area_ratio_parallelogram_to_triangle_l1701_170117


namespace NUMINAMATH_GPT_cost_to_buy_450_candies_l1701_170180

-- Define a structure representing the problem conditions
structure CandyStore where
  candies_per_box : Nat
  regular_price : Nat
  discounted_price : Nat
  discount_threshold : Nat

-- Define parameters for this specific problem
def store : CandyStore :=
  { candies_per_box := 15,
    regular_price := 5,
    discounted_price := 4,
    discount_threshold := 10 }

-- Define the cost function with the given conditions
def cost (store : CandyStore) (candies : Nat) : Nat :=
  let boxes := candies / store.candies_per_box
  if boxes >= store.discount_threshold then
    boxes * store.discounted_price
  else
    boxes * store.regular_price

-- State the theorem we want to prove
theorem cost_to_buy_450_candies (store : CandyStore) (candies := 450) :
  store.candies_per_box = 15 →
  store.discounted_price = 4 →
  store.discount_threshold = 10 →
  cost store candies = 120 := by
  sorry

end NUMINAMATH_GPT_cost_to_buy_450_candies_l1701_170180


namespace NUMINAMATH_GPT_fractional_equation_m_value_l1701_170181

theorem fractional_equation_m_value {x m : ℝ} (hx : 0 < x) (h : 3 / (x - 4) = 1 - (x + m) / (4 - x))
: m = -1 := sorry

end NUMINAMATH_GPT_fractional_equation_m_value_l1701_170181


namespace NUMINAMATH_GPT_units_digit_x4_invx4_l1701_170172

theorem units_digit_x4_invx4 (x : ℝ) (h : x^2 - 12 * x + 1 = 0) : 
  (x^4 + (1 / x)^4) % 10 = 2 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_x4_invx4_l1701_170172


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1701_170190

theorem line_passes_through_fixed_point (p q : ℝ) (h : 3 * p - 2 * q = 1) :
  p * (-3 / 2) + 3 * (1 / 6) + q = 0 :=
by 
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1701_170190


namespace NUMINAMATH_GPT_polynomial_evaluation_l1701_170171

def f (x : ℝ) : ℝ := sorry

theorem polynomial_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 6 * x^2 + 2) :
  f (x^2 - 3) = x^4 - 2 * x^2 - 7 :=
sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1701_170171


namespace NUMINAMATH_GPT_new_volume_eq_7352_l1701_170162

variable (l w h : ℝ)

-- Given conditions
def volume_eq : Prop := l * w * h = 5184
def surface_area_eq : Prop := l * w + w * h + h * l = 972
def edge_sum_eq : Prop := l + w + h = 54

-- Question: New volume when dimensions are increased by two inches
def new_volume : ℝ := (l + 2) * (w + 2) * (h + 2)

-- Correct Answer: Prove that the new volume equals 7352
theorem new_volume_eq_7352 (h_vol : volume_eq l w h) (h_surf : surface_area_eq l w h) (h_edge : edge_sum_eq l w h) 
    : new_volume l w h = 7352 :=
by
  -- Proof omitted
  sorry

#check new_volume_eq_7352

end NUMINAMATH_GPT_new_volume_eq_7352_l1701_170162


namespace NUMINAMATH_GPT_cistern_length_is_correct_l1701_170141

-- Definitions for the conditions mentioned in the problem
def cistern_width : ℝ := 6
def water_depth : ℝ := 1.25
def wet_surface_area : ℝ := 83

-- The length of the cistern to be proven
def cistern_length : ℝ := 8

-- Theorem statement that length of the cistern must be 8 meters given the conditions
theorem cistern_length_is_correct :
  ∃ (L : ℝ), (wet_surface_area = (L * cistern_width) + (2 * L * water_depth) + (2 * cistern_width * water_depth)) ∧ L = cistern_length :=
  sorry

end NUMINAMATH_GPT_cistern_length_is_correct_l1701_170141


namespace NUMINAMATH_GPT_minimum_value_l1701_170169

noncomputable def problem_statement : Prop :=
  ∃ (a b : ℝ), (∃ (x : ℝ), x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) ∧ (a^2 + b^2 = 4 / 5)

-- This line states that the minimum possible value of a^2 + b^2, given the condition, is 4/5.
theorem minimum_value (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 :=
  sorry

end NUMINAMATH_GPT_minimum_value_l1701_170169


namespace NUMINAMATH_GPT_sum_of_numbers_of_large_cube_l1701_170160

def sum_faces_of_die := 1 + 2 + 3 + 4 + 5 + 6

def number_of_dice := 125

def number_of_faces_per_die := 6

def total_exposed_faces (side_length: ℕ) : ℕ := 6 * (side_length * side_length)

theorem sum_of_numbers_of_large_cube (side_length : ℕ) (dice_count : ℕ) 
    (sum_per_die : ℕ) (opposite_face_sum : ℕ) :
    dice_count = 125 →
    total_exposed_faces side_length = 150 →
    sum_per_die = 21 →
    (∀ f1 f2, (f1 + f2 = opposite_face_sum)) →
    dice_count * sum_per_die = 2625 →
    (210 ≤ dice_count * sum_per_die ∧ dice_count * sum_per_die ≤ 840) :=
by 
  intro h_dice_count
  intro h_exposed_faces
  intro h_sum_per_die
  intro h_opposite_faces
  intro h_total_sum
  sorry

end NUMINAMATH_GPT_sum_of_numbers_of_large_cube_l1701_170160


namespace NUMINAMATH_GPT_function_increasing_on_R_l1701_170175

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem function_increasing_on_R (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a ≤ f x₂ a) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_function_increasing_on_R_l1701_170175


namespace NUMINAMATH_GPT_plane_equation_l1701_170152

-- We will create a structure for 3D points to use in our problem
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the problem conditions and the equation we want to prove
def containsPoint (p: Point3D) : Prop := p.x = 1 ∧ p.y = 4 ∧ p.z = -8

def onLine (p: Point3D) : Prop := 
  ∃ t : ℝ, 
    (p.x = 4 * t + 2) ∧ 
    (p.y = - t - 1) ∧ 
    (p.z = 5 * t + 3)

def planeEq (p: Point3D) : Prop := 
  -4 * p.x + 2 * p.y - 5 * p.z + 3 = 0

-- Now state the theorem
theorem plane_equation (p: Point3D) : 
  containsPoint p ∨ onLine p → planeEq p := 
  sorry

end NUMINAMATH_GPT_plane_equation_l1701_170152


namespace NUMINAMATH_GPT_money_last_weeks_l1701_170157

-- Define the amounts of money earned and spent per week
def money_mowing : ℕ := 5
def money_weed_eating : ℕ := 58
def weekly_spending : ℕ := 7

-- Define the total money earned
def total_money : ℕ := money_mowing + money_weed_eating

-- Define the number of weeks the money will last
def weeks_last (total : ℕ) (weekly : ℕ) : ℕ := total / weekly

-- Theorem stating the number of weeks the money will last
theorem money_last_weeks : weeks_last total_money weekly_spending = 9 := by
  sorry

end NUMINAMATH_GPT_money_last_weeks_l1701_170157


namespace NUMINAMATH_GPT_cos_double_angle_l1701_170102

variable (θ : ℝ)

theorem cos_double_angle (h : Real.tan (θ + Real.pi / 4) = 3) : Real.cos (2 * θ) = 3 / 5 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l1701_170102


namespace NUMINAMATH_GPT_cubic_equation_roots_l1701_170193

theorem cubic_equation_roots :
  (∀ x : ℝ, (x^3 - 7*x^2 + 36 = 0) → (x = -2 ∨ x = 3 ∨ x = 6)) ∧
  ∃ (x1 x2 x3 : ℝ), (x1 * x2 = 18) ∧ (x1 * x2 * x3 = -36) :=
by
  sorry

end NUMINAMATH_GPT_cubic_equation_roots_l1701_170193


namespace NUMINAMATH_GPT_f_properties_l1701_170151

noncomputable def f : ℝ → ℝ := sorry -- we define f as a noncomputable function for generality 

-- Given conditions as Lean hypotheses
axiom functional_eq : ∀ x y : ℝ, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom not_always_zero : ¬(∀ x : ℝ, f x = 0)

-- The statement we need to prove
theorem f_properties : f 0 = 1 ∧ (∀ x : ℝ, f (-x) = f x) := 
  by 
    sorry

end NUMINAMATH_GPT_f_properties_l1701_170151


namespace NUMINAMATH_GPT_domain_of_f_l1701_170114

noncomputable def f (x : ℝ) : ℝ := (Real.tan (2 * x)) / Real.sqrt (x - x^2)

theorem domain_of_f :
  { x : ℝ | ∃ k : ℤ, 2*x ≠ k*π + π/2 ∧ x ∈ (Set.Ioo 0 (π/4) ∪ Set.Ioo (π/4) 1) } = 
  { x : ℝ | x ∈ Set.Ioo 0 (π/4) ∪ Set.Ioo (π/4) 1 } :=
sorry

end NUMINAMATH_GPT_domain_of_f_l1701_170114


namespace NUMINAMATH_GPT_distance_from_two_eq_three_l1701_170184

theorem distance_from_two_eq_three (x : ℝ) (h : |x - 2| = 3) : x = -1 ∨ x = 5 :=
sorry

end NUMINAMATH_GPT_distance_from_two_eq_three_l1701_170184


namespace NUMINAMATH_GPT_simplify_abs_expression_l1701_170142

theorem simplify_abs_expression (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) :
  |a - 2 * b + 5| + |-3 * a + 2 * b - 2| = 4 * a - 4 * b + 7 := by
  sorry

end NUMINAMATH_GPT_simplify_abs_expression_l1701_170142


namespace NUMINAMATH_GPT_solve_square_l1701_170119

theorem solve_square:
  ∃ (square: ℚ), 
    ((13/5) - ((17/2) - square) / (7/2)) / (1 / ((61/20) + (89/20))) = 2 → 
    square = 1/3 :=
  sorry

end NUMINAMATH_GPT_solve_square_l1701_170119


namespace NUMINAMATH_GPT_problem1_problem2_l1701_170183

-- Problem 1
theorem problem1 : (1 / 2) ^ (-2 : ℤ) - (Real.pi - Real.sqrt 5) ^ 0 - Real.sqrt 20 = 3 - 2 * Real.sqrt 5 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -1) : 
  ((x ^ 2 - 2 * x + 1) / (x ^ 2 - 1)) / ((x - 1) / (x ^ 2 + x)) = x :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1701_170183


namespace NUMINAMATH_GPT_foci_on_x_axis_l1701_170122

theorem foci_on_x_axis (k : ℝ) : (∃ a b : ℝ, ∀ x y : ℝ, (x^2)/(3 - k) + (y^2)/(1 + k) = 1) ↔ -1 < k ∧ k < 1 :=
by
  sorry

end NUMINAMATH_GPT_foci_on_x_axis_l1701_170122


namespace NUMINAMATH_GPT_coefficient_a_eq_2_l1701_170107

theorem coefficient_a_eq_2 (a : ℝ) (h : (a^3 * (4 : ℝ)) = 32) : a = 2 :=
by {
  -- Proof will need to be filled in here
  sorry
}

end NUMINAMATH_GPT_coefficient_a_eq_2_l1701_170107


namespace NUMINAMATH_GPT_solution_to_system_of_inequalities_l1701_170106

variable {x y : ℝ}

theorem solution_to_system_of_inequalities :
  11 * (-1/3 : ℝ)^2 + 8 * (-1/3 : ℝ) * (2/3 : ℝ) + 8 * (2/3 : ℝ)^2 ≤ 3 ∧
  (-1/3 : ℝ) - 4 * (2/3 : ℝ) ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_system_of_inequalities_l1701_170106


namespace NUMINAMATH_GPT_product_of_bc_l1701_170191

theorem product_of_bc (b c : ℤ) 
  (h : ∀ r, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) : b * c = 110 :=
sorry

end NUMINAMATH_GPT_product_of_bc_l1701_170191


namespace NUMINAMATH_GPT_isabella_houses_l1701_170118

theorem isabella_houses (green yellow red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  green + red = 160 := 
by sorry

end NUMINAMATH_GPT_isabella_houses_l1701_170118


namespace NUMINAMATH_GPT_ratio_of_playground_area_to_total_landscape_area_l1701_170199

theorem ratio_of_playground_area_to_total_landscape_area {B L : ℝ} 
    (h1 : L = 8 * B)
    (h2 : L = 240)
    (h3 : 1200 = (240 * B * L) / (240 * B)) :
    1200 / (240 * B) = 1 / 6 :=
sorry

end NUMINAMATH_GPT_ratio_of_playground_area_to_total_landscape_area_l1701_170199


namespace NUMINAMATH_GPT_vector_BC_l1701_170100

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

theorem vector_BC (BA CA BC : ℝ × ℝ) (BA_def : BA = (1, 2)) (CA_def : CA = (4, 5)) (BC_def : BC = vector_sub BA CA) : BC = (-3, -3) :=
by
  subst BA_def
  subst CA_def
  subst BC_def
  sorry

end NUMINAMATH_GPT_vector_BC_l1701_170100


namespace NUMINAMATH_GPT_production_in_three_minutes_l1701_170197

noncomputable def production_rate_per_machine (total_bottles : ℕ) (num_machines : ℕ) : ℕ :=
  total_bottles / num_machines

noncomputable def production_per_minute (machines : ℕ) (rate_per_machine : ℕ) : ℕ :=
  machines * rate_per_machine

noncomputable def total_production (production_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  production_per_minute * minutes

theorem production_in_three_minutes :
  ∀ (total_bottles : ℕ) (num_machines : ℕ) (machines : ℕ) (minutes : ℕ),
  total_bottles = 16 → num_machines = 4 → machines = 8 → minutes = 3 →
  total_production (production_per_minute machines (production_rate_per_machine total_bottles num_machines)) minutes = 96 :=
by
  intros total_bottles num_machines machines minutes h_total_bottles h_num_machines h_machines h_minutes
  sorry

end NUMINAMATH_GPT_production_in_three_minutes_l1701_170197


namespace NUMINAMATH_GPT_combinations_15_3_l1701_170111

def num_combinations (n k : ℕ) : ℕ := n.choose k

theorem combinations_15_3 :
  num_combinations 15 3 = 455 :=
sorry

end NUMINAMATH_GPT_combinations_15_3_l1701_170111


namespace NUMINAMATH_GPT_price_of_A_is_40_l1701_170179

theorem price_of_A_is_40
  (p_a p_b : ℕ)
  (h1 : p_a = 2 * p_b)
  (h2 : 400 / p_a = 400 / p_b - 10) : p_a = 40 := 
by
  sorry

end NUMINAMATH_GPT_price_of_A_is_40_l1701_170179


namespace NUMINAMATH_GPT_min_value_fraction_l1701_170165

theorem min_value_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b > 0) (h₃ : 2 * a + b = 1) : 
  ∃ x, x = 8 ∧ ∀ y, (y = (1 / a) + (2 / b)) → y ≥ x :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l1701_170165


namespace NUMINAMATH_GPT_boy_reaches_early_l1701_170189

theorem boy_reaches_early (usual_rate new_rate : ℝ) (Usual_Time New_Time : ℕ) 
  (Hrate : new_rate = 9/8 * usual_rate) (Htime : Usual_Time = 36) :
  New_Time = 32 → Usual_Time - New_Time = 4 :=
by
  intros
  subst_vars
  sorry

end NUMINAMATH_GPT_boy_reaches_early_l1701_170189


namespace NUMINAMATH_GPT_intersection_of_sets_l1701_170134

-- Definitions of sets A and B based on given conditions
def setA : Set ℤ := {x | x + 2 = 0}
def setB : Set ℤ := {x | x^2 - 4 = 0}

-- The theorem to prove the intersection of A and B
theorem intersection_of_sets : setA ∩ setB = {-2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1701_170134


namespace NUMINAMATH_GPT_log_cosine_range_l1701_170155

noncomputable def log_base_three (a : ℝ) : ℝ := Real.log a / Real.log 3

theorem log_cosine_range (x : ℝ) (hx : x ∈ Set.Ioo (Real.pi / 2) (7 * Real.pi / 6)) :
    ∃ y, y = log_base_three (1 - 2 * Real.cos x) ∧ y ∈ Set.Ioc 0 1 :=
by
  sorry

end NUMINAMATH_GPT_log_cosine_range_l1701_170155


namespace NUMINAMATH_GPT_f_monotonically_decreasing_range_of_a_tangent_intersection_l1701_170192

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + 2
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x

-- Part (I)
theorem f_monotonically_decreasing (a : ℝ) (x : ℝ) :
  (a > 0 → 0 < x ∧ x < (2 / 3) * a → f' x a < 0) ∧
  (a = 0 → ¬∃ x, f' x a < 0) ∧
  (a < 0 → (2 / 3) * a < x ∧ x < 0 → f' x a < 0) :=
sorry

-- Part (II)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f' x a ≥ abs x - 3 / 4) → (-1 ≤ a ∧ a ≤ 1) :=
sorry

-- Part (III)
theorem tangent_intersection (a : ℝ) :
  (a = 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ ∃ t : ℝ, (t - x1^3 - 2 = 3 * x1^2 * (2 - x1)) ∧
  (t - x2^3 - 2 = 3 * x2^2 * (2 - x2)) ∧ 2 ≤ t ∧ t ≤ 10 ∧
  ∀ t', (t' - x1^3 - 2 = 3 * x1^2 * (2 - x1)) ∧
  (t' - x2^3 - 2 = 3 * x2^2 * (2 - x2)) → t' ≤ 10) :=
sorry

end NUMINAMATH_GPT_f_monotonically_decreasing_range_of_a_tangent_intersection_l1701_170192


namespace NUMINAMATH_GPT_males_listen_l1701_170159

theorem males_listen (males_dont_listen females_listen total_listen total_dont_listen : ℕ) 
  (h1 : males_dont_listen = 70)
  (h2 : females_listen = 75)
  (h3 : total_listen = 180)
  (h4 : total_dont_listen = 120) :
  ∃ m, m = 105 :=
by {
  sorry
}

end NUMINAMATH_GPT_males_listen_l1701_170159


namespace NUMINAMATH_GPT_sitio_proof_l1701_170188

theorem sitio_proof :
  (∃ t : ℝ, t = 4 + 7 + 12 ∧ 
    (∃ f : ℝ, 
      (∃ s : ℝ, s = 6 + 5 + 10 ∧ t = 23 ∧ f = 23 - s) ∧
      f = 2) ∧
    (∃ cost_per_hectare : ℝ, cost_per_hectare = 2420 / (4 + 12) ∧ 
      (∃ saci_spent : ℝ, saci_spent = 6 * cost_per_hectare ∧ saci_spent = 1320))) :=
by sorry

end NUMINAMATH_GPT_sitio_proof_l1701_170188


namespace NUMINAMATH_GPT_transport_cost_6725_l1701_170103

variable (P : ℝ) (T : ℝ)

theorem transport_cost_6725
  (h1 : 0.80 * P = 17500)
  (h2 : 1.10 * P = 24475)
  (h3 : 17500 + T + 250 = 24475) :
  T = 6725 := 
sorry

end NUMINAMATH_GPT_transport_cost_6725_l1701_170103


namespace NUMINAMATH_GPT_complement_A_in_U_l1701_170138

-- Define the universal set as ℝ
def U : Set ℝ := Set.univ

-- Define the set A as given in the conditions
def A : Set ℝ := {y | ∃ x : ℝ, 2^(Real.log x) = y}

-- The main statement based on the conditions and the correct answer
theorem complement_A_in_U : (U \ A) = {y | y ≤ 0} := by
  sorry

end NUMINAMATH_GPT_complement_A_in_U_l1701_170138


namespace NUMINAMATH_GPT_shanghai_world_expo_l1701_170101

theorem shanghai_world_expo (n : ℕ) (total_cost : ℕ) 
  (H1 : total_cost = 4000)
  (H2 : n ≤ 30 → total_cost = n * 120)
  (H3 : n > 30 → total_cost = n * (120 - 2 * (n - 30)) ∧ (120 - 2 * (n - 30)) ≥ 90) :
  n = 40 := 
sorry

end NUMINAMATH_GPT_shanghai_world_expo_l1701_170101


namespace NUMINAMATH_GPT_letters_in_small_envelopes_l1701_170194

theorem letters_in_small_envelopes (total_letters : ℕ) (large_envelopes : ℕ) (letters_per_large_envelope : ℕ) (letters_in_small_envelopes : ℕ) :
  total_letters = 80 →
  large_envelopes = 30 →
  letters_per_large_envelope = 2 →
  letters_in_small_envelopes = total_letters - (large_envelopes * letters_per_large_envelope) →
  letters_in_small_envelopes = 20 :=
by
  intros ht hl he hs
  rw [ht, hl, he] at hs
  exact hs

#check letters_in_small_envelopes

end NUMINAMATH_GPT_letters_in_small_envelopes_l1701_170194


namespace NUMINAMATH_GPT_triangle_cosine_identity_l1701_170115

open Real

variables {A B C a b c : ℝ}

theorem triangle_cosine_identity (h : b = (a + c) / 2) : cos (A - C) + 4 * cos B = 3 :=
sorry

end NUMINAMATH_GPT_triangle_cosine_identity_l1701_170115


namespace NUMINAMATH_GPT_surface_area_spherical_segment_l1701_170147

-- Definitions based on given conditions
variables {R h : ℝ}

-- The theorem to be proven
theorem surface_area_spherical_segment (h_pos : 0 < h) (R_pos : 0 < R)
  (planes_not_intersect_sphere : h < 2 * R) :
  S = 2 * π * R * h := by
  sorry

end NUMINAMATH_GPT_surface_area_spherical_segment_l1701_170147


namespace NUMINAMATH_GPT_employees_original_number_l1701_170196

noncomputable def original_employees_approx (employees_remaining : ℝ) (reduction_percent : ℝ) : ℝ :=
  employees_remaining / (1 - reduction_percent)

theorem employees_original_number (employees_remaining : ℝ) (reduction_percent : ℝ) (original : ℝ) :
  employees_remaining = 462 → reduction_percent = 0.276 →
  abs (original_employees_approx employees_remaining reduction_percent - original) < 1 →
  original = 638 :=
by
  intros h_remaining h_reduction h_approx
  sorry

end NUMINAMATH_GPT_employees_original_number_l1701_170196


namespace NUMINAMATH_GPT_find_result_l1701_170168

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 4 * x - 3

theorem find_result : f (g 3) - g (f 3) = -6 := by
  sorry

end NUMINAMATH_GPT_find_result_l1701_170168


namespace NUMINAMATH_GPT_digits_problem_solution_l1701_170121

def digits_proof_problem (E F G H : ℕ) : Prop :=
  (E, F, G) = (5, 0, 5) → H = 0

theorem digits_problem_solution 
  (E F G H : ℕ)
  (h1 : F + E = E ∨ F + E = E + 10)
  (h2 : E ≠ 0)
  (h3 : E = 5)
  (h4 : 5 + G = H)
  (h5 : 5 - G = 0) :
  H = 0 := 
by {
  sorry -- proof goes here
}

end NUMINAMATH_GPT_digits_problem_solution_l1701_170121


namespace NUMINAMATH_GPT_solve_for_t_l1701_170158

variables (V0 V g a t S : ℝ)

-- Given conditions
def velocity_eq : Prop := V = (g + a) * t + V0
def displacement_eq : Prop := S = (1/2) * (g + a) * t^2 + V0 * t

-- The theorem to prove
theorem solve_for_t (h1 : velocity_eq V0 V g a t)
                    (h2 : displacement_eq V0 g a t S) :
  t = 2 * S / (V + V0) :=
sorry

end NUMINAMATH_GPT_solve_for_t_l1701_170158


namespace NUMINAMATH_GPT_tip_customers_count_l1701_170136

-- Definitions and given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def no_tip_customers : ℕ := 34

-- Total customers computation
def total_customers : ℕ := initial_customers + added_customers

-- Lean 4 statement for proof problem
theorem tip_customers_count : (total_customers - no_tip_customers) = 15 := by
  sorry

end NUMINAMATH_GPT_tip_customers_count_l1701_170136


namespace NUMINAMATH_GPT_max_discount_l1701_170177

variable (x : ℝ)

theorem max_discount (h1 : (1 + 0.8) * x = 360) : 360 - 1.2 * x = 120 := 
by
  sorry

end NUMINAMATH_GPT_max_discount_l1701_170177


namespace NUMINAMATH_GPT_solve_for_y_l1701_170116

theorem solve_for_y (y : ℝ) : (y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) → y = -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_y_l1701_170116


namespace NUMINAMATH_GPT_sum_of_squares_not_divisible_by_5_or_13_l1701_170137

-- Definition of the set T
def T (n : ℤ) : ℤ :=
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2

-- The theorem to prove
theorem sum_of_squares_not_divisible_by_5_or_13 (n : ℤ) :
  ¬ (T n % 5 = 0) ∧ ¬ (T n % 13 = 0) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_not_divisible_by_5_or_13_l1701_170137


namespace NUMINAMATH_GPT_problem_2_8_3_4_7_2_2_l1701_170145

theorem problem_2_8_3_4_7_2_2 : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end NUMINAMATH_GPT_problem_2_8_3_4_7_2_2_l1701_170145


namespace NUMINAMATH_GPT_man_l1701_170139

theorem man's_salary (S : ℝ)
  (h1 : S * (1/5 + 1/10 + 3/5) = 9/10 * S)
  (h2 : S - 9/10 * S = 14000) :
  S = 140000 :=
by
  sorry

end NUMINAMATH_GPT_man_l1701_170139


namespace NUMINAMATH_GPT_largest_three_digit_multiple_of_4_and_5_l1701_170131

theorem largest_three_digit_multiple_of_4_and_5 : 
  ∃ (n : ℕ), n < 1000 ∧ n ≥ 100 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n = 980 :=
by
  sorry

end NUMINAMATH_GPT_largest_three_digit_multiple_of_4_and_5_l1701_170131


namespace NUMINAMATH_GPT_not_possible_to_create_3_piles_l1701_170144

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end NUMINAMATH_GPT_not_possible_to_create_3_piles_l1701_170144


namespace NUMINAMATH_GPT_remainder_53_pow_10_div_8_l1701_170195

theorem remainder_53_pow_10_div_8 : (53^10) % 8 = 1 := 
by sorry

end NUMINAMATH_GPT_remainder_53_pow_10_div_8_l1701_170195


namespace NUMINAMATH_GPT_true_proposition_l1701_170135

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end NUMINAMATH_GPT_true_proposition_l1701_170135


namespace NUMINAMATH_GPT_find_n_in_geom_series_l1701_170109

noncomputable def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem find_n_in_geom_series :
  ∃ n : ℕ, geom_sum 1 (1/2) n = 31 / 16 :=
sorry

end NUMINAMATH_GPT_find_n_in_geom_series_l1701_170109


namespace NUMINAMATH_GPT_sum_int_values_l1701_170154

theorem sum_int_values (sum : ℤ) : 
  (∀ n : ℤ, (20 % (2 * n - 1) = 0) → sum = 2) :=
by
  sorry

end NUMINAMATH_GPT_sum_int_values_l1701_170154


namespace NUMINAMATH_GPT_percent_neither_condition_l1701_170150

namespace TeachersSurvey

variables (Total HighBloodPressure HeartTrouble Both: ℕ)

theorem percent_neither_condition :
  Total = 150 → HighBloodPressure = 90 → HeartTrouble = 50 → Both = 30 →
  (HighBloodPressure + HeartTrouble - Both) = 110 →
  ((Total - (HighBloodPressure + HeartTrouble - Both)) * 100 / Total) = 2667 / 100 :=
by
  intros hTotal hBP hHT hBoth hUnion
  sorry

end TeachersSurvey

end NUMINAMATH_GPT_percent_neither_condition_l1701_170150


namespace NUMINAMATH_GPT_probability_sum_less_than_16_l1701_170187

-- The number of possible outcomes when three six-sided dice are rolled
def total_outcomes : ℕ := 6 * 6 * 6

-- The number of favorable outcomes where the sum of the dice is less than 16
def favorable_outcomes : ℕ := (6 * 6 * 6) - (3 + 3 + 3 + 1)

-- The probability that the sum of the dice is less than 16
def probability_less_than_16 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_less_than_16 : probability_less_than_16 = 103 / 108 := 
by sorry

end NUMINAMATH_GPT_probability_sum_less_than_16_l1701_170187


namespace NUMINAMATH_GPT_probability_of_exactly_one_hitting_l1701_170120

variable (P_A_hitting B_A_hitting : ℝ)

theorem probability_of_exactly_one_hitting (hP_A : P_A_hitting = 0.6) (hP_B : B_A_hitting = 0.6) :
  ((P_A_hitting * (1 - B_A_hitting)) + ((1 - P_A_hitting) * B_A_hitting)) = 0.48 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_exactly_one_hitting_l1701_170120


namespace NUMINAMATH_GPT_binomial_coefficient_8_5_l1701_170170

theorem binomial_coefficient_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_8_5_l1701_170170


namespace NUMINAMATH_GPT_no_real_solution_for_inequality_l1701_170140

theorem no_real_solution_for_inequality :
  ¬ ∃ a : ℝ, ∃ x : ℝ, ∀ b : ℝ, |x^2 + 4*a*x + 5*a| ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_for_inequality_l1701_170140
