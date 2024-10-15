import Mathlib

namespace NUMINAMATH_GPT_imaginary_part_of_z_l339_33939

theorem imaginary_part_of_z {z : ℂ} (h : (1 + z) / I = 1 - z) : z.im = 1 := 
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l339_33939


namespace NUMINAMATH_GPT_sum_of_reciprocals_l339_33923

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + y = 3 * x * y) (h2 : x - y = 2) : (1/x + 1/y) = 4/3 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l339_33923


namespace NUMINAMATH_GPT_art_collection_total_cost_l339_33901

theorem art_collection_total_cost 
  (price_first_three : ℕ)
  (price_fourth : ℕ)
  (total_first_three : price_first_three * 3 = 45000)
  (price_fourth_cond : price_fourth = price_first_three + (price_first_three / 2)) :
  3 * price_first_three + price_fourth = 67500 :=
by
  sorry

end NUMINAMATH_GPT_art_collection_total_cost_l339_33901


namespace NUMINAMATH_GPT_find_number_l339_33927

theorem find_number (x : ℝ) (h: 9999 * x = 4690910862): x = 469.1 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l339_33927


namespace NUMINAMATH_GPT_largest_value_of_x_l339_33975

noncomputable def find_largest_x : ℝ :=
  let a := 10
  let b := 39
  let c := 18
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2 * a)
  let x2 := (-b - sqrt_discriminant) / (2 * a)
  if x1 > x2 then x1 else x2

theorem largest_value_of_x :
  ∃ x : ℝ, 3 * (10 * x^2 + 11 * x + 12) = x * (10 * x - 45) ∧
  x = find_largest_x := by
  exists find_largest_x
  sorry

end NUMINAMATH_GPT_largest_value_of_x_l339_33975


namespace NUMINAMATH_GPT_carrie_pays_94_l339_33976

theorem carrie_pays_94 :
  ∀ (num_shirts num_pants num_jackets : ℕ) (cost_shirt cost_pants cost_jacket : ℕ),
  num_shirts = 4 →
  cost_shirt = 8 →
  num_pants = 2 →
  cost_pants = 18 →
  num_jackets = 2 →
  cost_jacket = 60 →
  (cost_shirt * num_shirts + cost_pants * num_pants + cost_jacket * num_jackets) / 2 = 94 :=
by
  intros num_shirts num_pants num_jackets cost_shirt cost_pants cost_jacket
  sorry

end NUMINAMATH_GPT_carrie_pays_94_l339_33976


namespace NUMINAMATH_GPT_probability_of_specific_combination_l339_33930

def count_all_clothes : ℕ := 6 + 7 + 8 + 3
def choose4_out_of_24 : ℕ := Nat.choose 24 4
def choose1_shirt : ℕ := 6
def choose1_pair_shorts : ℕ := 7
def choose1_pair_socks : ℕ := 8
def choose1_hat : ℕ := 3
def favorable_outcomes : ℕ := choose1_shirt * choose1_pair_shorts * choose1_pair_socks * choose1_hat
def probability_of_combination : ℚ := favorable_outcomes / choose4_out_of_24

theorem probability_of_specific_combination :
  probability_of_combination = 144 / 1815 := by
sorry

end NUMINAMATH_GPT_probability_of_specific_combination_l339_33930


namespace NUMINAMATH_GPT_isosceles_triangle_area_l339_33909

theorem isosceles_triangle_area (x : ℤ) (h1 : x > 2) (h2 : x < 4) 
  (h3 : ∃ (a b : ℤ), a = x ∧ b = 8 - 2 * x ∧ a = b) :
  ∃ (area : ℝ), area = 2 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l339_33909


namespace NUMINAMATH_GPT_seventy_three_days_after_monday_is_thursday_l339_33981

def day_of_week : Nat → String
| 0 => "Monday"
| 1 => "Tuesday"
| 2 => "Wednesday"
| 3 => "Thursday"
| 4 => "Friday"
| 5 => "Saturday"
| _ => "Sunday"

theorem seventy_three_days_after_monday_is_thursday :
  day_of_week (73 % 7) = "Thursday" :=
by
  sorry

end NUMINAMATH_GPT_seventy_three_days_after_monday_is_thursday_l339_33981


namespace NUMINAMATH_GPT_more_than_half_millet_on_day_three_l339_33926

-- Definition of the initial conditions
def seeds_in_feeder (n: ℕ) : ℝ :=
  1 + n

def millet_amount (n: ℕ) : ℝ :=
  0.6 * (1 - (0.5)^n)

-- The theorem we want to prove
theorem more_than_half_millet_on_day_three :
  ∀ n, n = 3 → (millet_amount n) / (seeds_in_feeder n) > 0.5 :=
by
  intros n hn
  rw [hn, seeds_in_feeder, millet_amount]
  sorry

end NUMINAMATH_GPT_more_than_half_millet_on_day_three_l339_33926


namespace NUMINAMATH_GPT_difference_of_squares_l339_33991

theorem difference_of_squares : 
  let a := 625
  let b := 575
  (a^2 - b^2) = 60000 :=
by 
  let a := 625
  let b := 575
  sorry

end NUMINAMATH_GPT_difference_of_squares_l339_33991


namespace NUMINAMATH_GPT_determine_no_conditionals_l339_33943

def problem_requires_conditionals (n : ℕ) : Prop :=
  n = 3 ∨ n = 4

theorem determine_no_conditionals :
  problem_requires_conditionals 1 = false ∧
  problem_requires_conditionals 2 = false ∧
  problem_requires_conditionals 3 = true ∧
  problem_requires_conditionals 4 = true :=
by sorry

end NUMINAMATH_GPT_determine_no_conditionals_l339_33943


namespace NUMINAMATH_GPT_perpendicular_parallel_l339_33955

variables {a b : Line} {α : Plane}

-- Definition of perpendicular and parallel relations should be available
-- since their exact details were not provided, placeholder functions will be used for demonstration

-- Placeholder definitions for perpendicular and parallel (they should be accurately defined elsewhere)
def perp (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry

theorem perpendicular_parallel {a b : Line} {α : Plane}
    (a_perp_alpha : perp a α)
    (b_perp_alpha : perp b α)
    : parallel a b :=
sorry

end NUMINAMATH_GPT_perpendicular_parallel_l339_33955


namespace NUMINAMATH_GPT_find_a_l339_33941

theorem find_a (r s a : ℚ) (h₁ : 2 * r * s = 18) (h₂ : s^2 = 16) (h₃ : a = r^2) : 
  a = 81 / 16 := 
sorry

end NUMINAMATH_GPT_find_a_l339_33941


namespace NUMINAMATH_GPT_cd_e_value_l339_33994

theorem cd_e_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 195) (h2 : b * c * d = 65) 
  (h3 : d * e * f = 250) (h4 : (a * f) / (c * d) = 0.75) :
  c * d * e = 1000 := 
by
  sorry

end NUMINAMATH_GPT_cd_e_value_l339_33994


namespace NUMINAMATH_GPT_probability_of_three_draws_l339_33920

noncomputable def box_chips : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def valid_first_two_draws (a b : ℕ) : Prop :=
  a + b <= 7

def prob_three_draws_to_exceed_seven : ℚ :=
  1 / 6

theorem probability_of_three_draws :
  (∃ (draws : List ℕ), (draws.length = 3) ∧ (draws.sum > 7)
    ∧ (∀ x ∈ draws, x ∈ box_chips)
    ∧ (∀ (a b : ℕ), (a ∈ box_chips ∧ b ∈ box_chips) → valid_first_two_draws a b))
  → prob_three_draws_to_exceed_seven = 1 / 6 :=
sorry

end NUMINAMATH_GPT_probability_of_three_draws_l339_33920


namespace NUMINAMATH_GPT_volume_of_rock_correct_l339_33974

-- Define the initial conditions
def tank_length := 30
def tank_width := 20
def water_depth := 8
def water_level_rise := 4

-- Define the volume function for the rise in water level
def calculate_volume_of_rise (length: ℕ) (width: ℕ) (rise: ℕ) : ℕ :=
  length * width * rise

-- Define the target volume of the rock
def volume_of_rock := 2400

-- The theorem statement that the volume of the rock is 2400 cm³
theorem volume_of_rock_correct :
  calculate_volume_of_rise tank_length tank_width water_level_rise = volume_of_rock :=
by 
  sorry

end NUMINAMATH_GPT_volume_of_rock_correct_l339_33974


namespace NUMINAMATH_GPT_train_length_l339_33937

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h_speed : speed_kmph = 60) 
  (h_time : time_sec = 7.199424046076314) 
  (h_length : length_m = 120)
  : speed_kmph * (1000 / 3600) * time_sec = length_m :=
by 
  sorry

end NUMINAMATH_GPT_train_length_l339_33937


namespace NUMINAMATH_GPT_sin_cos_identity_l339_33925

theorem sin_cos_identity : (Real.sin (65 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) 
  - Real.cos (65 * Real.pi / 180) * Real.sin (35 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l339_33925


namespace NUMINAMATH_GPT_penny_exceeded_by_32_l339_33987

def bulk_price : ℤ := 5
def min_spend_before_tax : ℤ := 40
def tax_per_pound : ℤ := 1
def penny_payment : ℤ := 240

def total_cost_per_pound : ℤ := bulk_price + tax_per_pound

def min_pounds_for_min_spend : ℤ := min_spend_before_tax / bulk_price

def total_pounds_penny_bought : ℤ := penny_payment / total_cost_per_pound

def pounds_exceeded : ℤ := total_pounds_penny_bought - min_pounds_for_min_spend

theorem penny_exceeded_by_32 : pounds_exceeded = 32 := by
  sorry

end NUMINAMATH_GPT_penny_exceeded_by_32_l339_33987


namespace NUMINAMATH_GPT_non_zero_real_value_l339_33908

theorem non_zero_real_value (y : ℝ) (hy : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 :=
sorry

end NUMINAMATH_GPT_non_zero_real_value_l339_33908


namespace NUMINAMATH_GPT_intersection_of_sets_l339_33961

def set_M : Set ℝ := { x : ℝ | (x + 2) * (x - 1) < 0 }
def set_N : Set ℝ := { x : ℝ | x + 1 < 0 }
def intersection (A B : Set ℝ) : Set ℝ := { x : ℝ | x ∈ A ∧ x ∈ B }

theorem intersection_of_sets :
  intersection set_M set_N = { x : ℝ | -2 < x ∧ x < -1 } := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l339_33961


namespace NUMINAMATH_GPT_sum_mod_13_l339_33957

theorem sum_mod_13 (a b c d e : ℤ) (ha : a % 13 = 3) (hb : b % 13 = 5) (hc : c % 13 = 7) (hd : d % 13 = 9) (he : e % 13 = 11) :
  (a + b + c + d + e) % 13 = 9 :=
by
  -- The proof can be constructed here
  sorry

end NUMINAMATH_GPT_sum_mod_13_l339_33957


namespace NUMINAMATH_GPT_acute_triangle_l339_33912

-- Given the lengths of three line segments
def length1 : ℝ := 5
def length2 : ℝ := 6
def length3 : ℝ := 7

-- Conditions (C): The lengths of the three line segments
def triangle_inequality : Prop :=
  length1 + length2 > length3 ∧
  length1 + length3 > length2 ∧
  length2 + length3 > length1

-- Question (Q) and Answer (A): They form an acute triangle
theorem acute_triangle (h : triangle_inequality) : (length1^2 + length2^2 - length3^2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_acute_triangle_l339_33912


namespace NUMINAMATH_GPT_impossible_coins_l339_33988

theorem impossible_coins (p1 p2 : ℝ) (h1 : (1 - p1) * (1 - p2) = p1 * p2) (h2 : p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2) : false :=
  by sorry

end NUMINAMATH_GPT_impossible_coins_l339_33988


namespace NUMINAMATH_GPT_limit_leq_l339_33962

variables {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

theorem limit_leq {a_n b_n : ℕ → α} {a b : α}
  (ha : Filter.Tendsto a_n Filter.atTop (nhds a))
  (hb : Filter.Tendsto b_n Filter.atTop (nhds b))
  (h_leq : ∀ n, a_n n ≤ b_n n)
  : a ≤ b :=
by
  -- Proof will be constructed here
  sorry

end NUMINAMATH_GPT_limit_leq_l339_33962


namespace NUMINAMATH_GPT_find_number_l339_33964

theorem find_number (x : ℝ) (h : x - (3/5) * x = 56) : x = 140 :=
sorry

end NUMINAMATH_GPT_find_number_l339_33964


namespace NUMINAMATH_GPT_netCaloriesConsumedIs1082_l339_33916

-- Given conditions
def caloriesPerCandyBar : ℕ := 347
def candyBarsEatenInAWeek : ℕ := 6
def caloriesBurnedInAWeek : ℕ := 1000

-- Net calories calculation
def netCaloriesInAWeek (calsPerBar : ℕ) (barsPerWeek : ℕ) (calsBurned : ℕ) : ℕ :=
  calsPerBar * barsPerWeek - calsBurned

-- The theorem to prove
theorem netCaloriesConsumedIs1082 :
  netCaloriesInAWeek caloriesPerCandyBar candyBarsEatenInAWeek caloriesBurnedInAWeek = 1082 :=
by
  sorry

end NUMINAMATH_GPT_netCaloriesConsumedIs1082_l339_33916


namespace NUMINAMATH_GPT_system_of_equations_property_l339_33904

theorem system_of_equations_property (a x y : ℝ)
  (h1 : x + y = 1 - a)
  (h2 : x - y = 3 * a + 5)
  (h3 : 0 < x)
  (h4 : 0 ≤ y) :
  (a = -5 / 3 → x = y) ∧ 
  (a = -2 → x + y = 5 + a) ∧ 
  (0 < x ∧ x ≤ 1 → 2 ≤ y ∧ y < 4) := 
by
  sorry

end NUMINAMATH_GPT_system_of_equations_property_l339_33904


namespace NUMINAMATH_GPT_ratio_of_linear_combination_l339_33936

theorem ratio_of_linear_combination (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (3 * a + 2 * b) / (b + 4 * c) = 3 / 17 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_linear_combination_l339_33936


namespace NUMINAMATH_GPT_negation_statement_l339_33924

open Set

variable {S : Set ℝ}

theorem negation_statement (h : ∀ x ∈ S, 3 * x - 5 > 0) : ∃ x ∈ S, 3 * x - 5 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_statement_l339_33924


namespace NUMINAMATH_GPT_find_a_l339_33965

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x / (3 * x + 4)

theorem find_a (a : ℝ) (x : ℝ) (h : x ≠ -4 / 3) : (f a) (f a x) = x → a = -2 := by
  unfold f
  -- Remaining proof steps skipped
  sorry

end NUMINAMATH_GPT_find_a_l339_33965


namespace NUMINAMATH_GPT_highest_score_l339_33998

theorem highest_score (H L : ℕ) (avg total46 total44 runs46 runs44 : ℕ)
  (h1 : H - L = 150)
  (h2 : avg = 61)
  (h3 : total46 = 46)
  (h4 : runs46 = avg * total46)
  (h5 : runs46 = 2806)
  (h6 : total44 = 44)
  (h7 : runs44 = 58 * total44)
  (h8 : runs44 = 2552)
  (h9 : runs46 - runs44 = H + L) :
  H = 202 := by
  sorry

end NUMINAMATH_GPT_highest_score_l339_33998


namespace NUMINAMATH_GPT_matrix_determinant_equiv_l339_33959

variable {x y z w : ℝ}

theorem matrix_determinant_equiv (h : x * w - y * z = 7) :
    (x + 2 * z) * w - (y + 2 * w) * z = 7 :=
by
    sorry

end NUMINAMATH_GPT_matrix_determinant_equiv_l339_33959


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l339_33950

theorem geometric_sequence_fourth_term (a : ℝ) (r : ℝ) (h : a = 512) (h1 : a * r^5 = 125) :
  a * r^3 = 1536 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l339_33950


namespace NUMINAMATH_GPT_range_f_l339_33963

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x) ^ 3

theorem range_f : Set.range f = Set.Ioi 0 ∪ Set.Iio 0 := by
  sorry

end NUMINAMATH_GPT_range_f_l339_33963


namespace NUMINAMATH_GPT_rectangular_floor_problem_possibilities_l339_33903

theorem rectangular_floor_problem_possibilities :
  ∃ (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s → p.2 > p.1 ∧ p.2 % 3 = 0 ∧ (p.1 - 6) * (p.2 - 6) = 36) 
    ∧ s.card = 2 := 
sorry

end NUMINAMATH_GPT_rectangular_floor_problem_possibilities_l339_33903


namespace NUMINAMATH_GPT_indigo_restaurant_total_reviews_l339_33945

-- Define the number of 5-star reviews
def five_star_reviews : Nat := 6

-- Define the number of 4-star reviews
def four_star_reviews : Nat := 7

-- Define the number of 3-star reviews
def three_star_reviews : Nat := 4

-- Define the number of 2-star reviews
def two_star_reviews : Nat := 1

-- Define the total number of reviews
def total_reviews : Nat := five_star_reviews + four_star_reviews + three_star_reviews + two_star_reviews

-- Proof that the total number of customer reviews is 18
theorem indigo_restaurant_total_reviews : total_reviews = 18 :=
by
  -- Direct calculation
  sorry

end NUMINAMATH_GPT_indigo_restaurant_total_reviews_l339_33945


namespace NUMINAMATH_GPT_total_votes_l339_33954

theorem total_votes (total_votes : ℕ) (brenda_votes : ℕ) (fraction : ℚ) (h : brenda_votes = fraction * total_votes) (h_fraction : fraction = 1 / 5) (h_brenda : brenda_votes = 15) : 
  total_votes = 75 := 
by
  sorry

end NUMINAMATH_GPT_total_votes_l339_33954


namespace NUMINAMATH_GPT_length_CD_l339_33982

-- Definitions of the edge lengths provided in the problem
def edge_lengths : Set ℕ := {7, 13, 18, 27, 36, 41}

-- Assumption that AB = 41
def AB := 41
def BC : ℕ := 13
def AC : ℕ := 36

-- Main theorem to prove that CD = 13
theorem length_CD (AB BC AC : ℕ) (edges : Set ℕ) (hAB : AB = 41) (hedges : edges = edge_lengths) :
  ∃ (CD : ℕ), CD ∈ edges ∧ CD = 13 :=
by
  sorry

end NUMINAMATH_GPT_length_CD_l339_33982


namespace NUMINAMATH_GPT_least_number_to_add_for_divisibility_by_nine_l339_33956

theorem least_number_to_add_for_divisibility_by_nine : ∃ x : ℕ, (4499 + x) % 9 = 0 ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_for_divisibility_by_nine_l339_33956


namespace NUMINAMATH_GPT_parabola_and_hyperbola_focus_equal_l339_33933

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) :=
(2, 0)

noncomputable def hyperbola_focus : (ℝ × ℝ) :=
(2, 0)

theorem parabola_and_hyperbola_focus_equal
  (p : ℝ)
  (h_parabola : parabola_focus p = (2, 0))
  (h_hyperbola : hyperbola_focus = (2, 0)) :
  p = 4 := by
  sorry

end NUMINAMATH_GPT_parabola_and_hyperbola_focus_equal_l339_33933


namespace NUMINAMATH_GPT_blue_lights_count_l339_33948

def num_colored_lights := 350
def num_red_lights := 85
def num_yellow_lights := 112
def num_green_lights := 65
def num_blue_lights := num_colored_lights - (num_red_lights + num_yellow_lights + num_green_lights)

theorem blue_lights_count : num_blue_lights = 88 := by
  sorry

end NUMINAMATH_GPT_blue_lights_count_l339_33948


namespace NUMINAMATH_GPT_seating_impossible_l339_33992

theorem seating_impossible (reps : Fin 54 → Fin 27) : 
  ¬ ∃ (s : Fin 54 → Fin 54),
    (∀ i : Fin 27, ∃ a b : Fin 54, a ≠ b ∧ s a = i ∧ s b = i ∧ (b - a ≡ 10 [MOD 54] ∨ a - b ≡ 10 [MOD 54])) :=
sorry

end NUMINAMATH_GPT_seating_impossible_l339_33992


namespace NUMINAMATH_GPT_number_of_adults_attending_concert_l339_33972

-- We have to define the constants and conditions first.
variable (A C : ℕ)
variable (h1 : A + C = 578)
variable (h2 : 2 * A + 3 / 2 * C = 985)

-- Now we state the theorem that given these conditions, A is equal to 236.

theorem number_of_adults_attending_concert : A = 236 :=
by sorry

end NUMINAMATH_GPT_number_of_adults_attending_concert_l339_33972


namespace NUMINAMATH_GPT_new_volume_l339_33997

theorem new_volume (l w h : ℝ) 
  (h1 : l * w * h = 4320)
  (h2 : l * w + w * h + l * h = 852)
  (h3 : l + w + h = 52) :
  (l + 4) * (w + 4) * (h + 4) = 8624 := sorry

end NUMINAMATH_GPT_new_volume_l339_33997


namespace NUMINAMATH_GPT_math_proof_problem_l339_33902

noncomputable def find_value (a b c : ℝ) : ℝ :=
  (a^3 + b^3 + c^3) / (a * b * c * (a * b + a * c + b * c))

theorem math_proof_problem (a b c : ℝ)
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a * b + a * c + b * c ≠ 0) :
  find_value a b c = 3 :=
by 
  -- sorry is used as we are only asked to provide the theorem statement in Lean.
  sorry

end NUMINAMATH_GPT_math_proof_problem_l339_33902


namespace NUMINAMATH_GPT_find_s_l339_33918

theorem find_s (n r s c d : ℝ) (h1 : c^2 - n * c + 3 = 0) (h2 : d^2 - n * d + 3 = 0) 
  (h3 : (c + 1/d)^2 - r * (c + 1/d) + s = 0) (h4 : (d + 1/c)^2 - r * (d + 1/c) + s = 0) 
  (h5 : c * d = 3) : s = 16 / 3 := 
by
  sorry

end NUMINAMATH_GPT_find_s_l339_33918


namespace NUMINAMATH_GPT_area_of_WXYZ_l339_33970

structure Quadrilateral (α : Type _) :=
  (W : α) (X : α) (Y : α) (Z : α)
  (WZ ZW' WX XX' XY YY' YZ Z'W : ℝ)
  (area_WXYZ : ℝ)

theorem area_of_WXYZ' (WXYZ : Quadrilateral ℝ) 
  (h1 : WXYZ.WZ = 10) 
  (h2 : WXYZ.ZW' = 10)
  (h3 : WXYZ.WX = 6)
  (h4 : WXYZ.XX' = 6)
  (h5 : WXYZ.XY = 7)
  (h6 : WXYZ.YY' = 7)
  (h7 : WXYZ.YZ = 12)
  (h8 : WXYZ.Z'W = 12)
  (h9 : WXYZ.area_WXYZ = 15) : 
  ∃ area_WXZY' : ℝ, area_WXZY' = 45 :=
sorry

end NUMINAMATH_GPT_area_of_WXYZ_l339_33970


namespace NUMINAMATH_GPT_min_sum_4410_l339_33967

def min_sum (a b c d : ℕ) : ℕ := a + b + c + d

theorem min_sum_4410 :
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * b * c * d = 4410 ∧ min_sum a b c d = 69 :=
sorry

end NUMINAMATH_GPT_min_sum_4410_l339_33967


namespace NUMINAMATH_GPT_problem_statement_l339_33905

-- Define function f(x) given parameter m
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- Define even function condition
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Define the monotonic decreasing interval condition
def is_monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) :=
 ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≥ f y

theorem problem_statement :
  (∀ x : ℝ, f m x = f m (-x)) → is_monotonically_decreasing (f 0) {x | 0 < x} :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l339_33905


namespace NUMINAMATH_GPT_number_of_grandchildren_l339_33907

-- Definitions based on the conditions
def cards_per_grandkid := 2
def money_per_card := 80
def total_money_given_away := 480

-- Calculation of money each grandkid receives per year
def money_per_grandkid := cards_per_grandkid * money_per_card

-- The theorem we want to prove
theorem number_of_grandchildren :
  (total_money_given_away / money_per_grandkid) = 3 :=
by
  -- Placeholder for the proof
  sorry 

end NUMINAMATH_GPT_number_of_grandchildren_l339_33907


namespace NUMINAMATH_GPT_right_triangle_area_l339_33952

theorem right_triangle_area (a b c r : ℝ) (h1 : a = 15) (h2 : r = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_right : a ^ 2 + b ^ 2 = c ^ 2) (h_incircle : r = (a + b - c) / 2) : 
  1 / 2 * a * b = 60 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l339_33952


namespace NUMINAMATH_GPT_ring_toss_total_earnings_l339_33969

theorem ring_toss_total_earnings :
  let earnings_first_ring_day1 := 761
  let days_first_ring_day1 := 88
  let earnings_first_ring_day2 := 487
  let days_first_ring_day2 := 20
  let earnings_second_ring_day1 := 569
  let days_second_ring_day1 := 66
  let earnings_second_ring_day2 := 932
  let days_second_ring_day2 := 15

  let total_first_ring := (earnings_first_ring_day1 * days_first_ring_day1) + (earnings_first_ring_day2 * days_first_ring_day2)
  let total_second_ring := (earnings_second_ring_day1 * days_second_ring_day1) + (earnings_second_ring_day2 * days_second_ring_day2)
  let total_earnings := total_first_ring + total_second_ring

  total_earnings = 128242 :=
by
  sorry

end NUMINAMATH_GPT_ring_toss_total_earnings_l339_33969


namespace NUMINAMATH_GPT_Elberta_has_23_dollars_l339_33989

theorem Elberta_has_23_dollars (GrannySmith_has : ℕ := 72)
    (Anjou_has : ℕ := GrannySmith_has / 4)
    (Elberta_has : ℕ := Anjou_has + 5) : Elberta_has = 23 :=
by
  sorry

end NUMINAMATH_GPT_Elberta_has_23_dollars_l339_33989


namespace NUMINAMATH_GPT_sum_of_new_dimensions_l339_33932

theorem sum_of_new_dimensions (s : ℕ) (h₁ : s^2 = 36) (h₂ : s' = s - 1) : s' + s' + s' = 15 :=
sorry

end NUMINAMATH_GPT_sum_of_new_dimensions_l339_33932


namespace NUMINAMATH_GPT_two_numbers_and_sum_l339_33931

theorem two_numbers_and_sum (x y : ℕ) (hx : x * y = 18) (hy : x - y = 4) : x + y = 10 :=
sorry

end NUMINAMATH_GPT_two_numbers_and_sum_l339_33931


namespace NUMINAMATH_GPT_function_decreasing_interval_l339_33934

variable (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 * (a * x + b)

theorem function_decreasing_interval :
  (deriv (f a b) 2 = 0) ∧ (deriv (f a b) 1 = -3) →
  ∃ (a b : ℝ), (deriv (f a b) x < 0) ↔ (0 < x ∧ x < 2) := sorry

end NUMINAMATH_GPT_function_decreasing_interval_l339_33934


namespace NUMINAMATH_GPT_point_relationship_l339_33917

def quadratic_function (x : ℝ) (c : ℝ) : ℝ :=
  -(x - 1) ^ 2 + c

noncomputable def y1_def (c : ℝ) : ℝ := quadratic_function (-3) c
noncomputable def y2_def (c : ℝ) : ℝ := quadratic_function (-1) c
noncomputable def y3_def (c : ℝ) : ℝ := quadratic_function 5 c

theorem point_relationship (c : ℝ) :
  y2_def c > y1_def c ∧ y1_def c = y3_def c :=
by
  sorry

end NUMINAMATH_GPT_point_relationship_l339_33917


namespace NUMINAMATH_GPT_Monica_books_next_year_l339_33951

-- Definitions for conditions
def books_last_year : ℕ := 25
def books_this_year (bl_year: ℕ) : ℕ := 3 * bl_year
def books_next_year (bt_year: ℕ) : ℕ := 3 * bt_year + 7

-- Theorem statement
theorem Monica_books_next_year : books_next_year (books_this_year books_last_year) = 232 :=
by
  sorry

end NUMINAMATH_GPT_Monica_books_next_year_l339_33951


namespace NUMINAMATH_GPT_number_at_two_units_right_of_origin_l339_33946

theorem number_at_two_units_right_of_origin : 
  ∀ (n : ℝ), (n = 0) →
  ∀ (x : ℝ), (x = n + 2) →
  x = 2 := 
by
  sorry

end NUMINAMATH_GPT_number_at_two_units_right_of_origin_l339_33946


namespace NUMINAMATH_GPT_length_of_platform_l339_33929

theorem length_of_platform {train_length : ℕ} {time_to_cross_pole : ℕ} {time_to_cross_platform : ℕ} 
  (h1 : train_length = 300) 
  (h2 : time_to_cross_pole = 18) 
  (h3 : time_to_cross_platform = 45) : 
  ∃ platform_length : ℕ, platform_length = 450 :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_l339_33929


namespace NUMINAMATH_GPT_hyperbola_asymptotes_and_parabola_l339_33947

-- Definitions for hyperbola and parabola
noncomputable def hyperbola (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1
noncomputable def focus_of_hyperbola (focus : ℝ × ℝ) : Prop := focus = (5, 0)
noncomputable def asymptote_of_hyperbola (y x : ℝ) : Prop := y = (4 / 3) * x ∨ y = - (4 / 3) * x
noncomputable def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x

-- Main statement
theorem hyperbola_asymptotes_and_parabola :
  (∀ x y, hyperbola x y → asymptote_of_hyperbola y x) ∧
  (∀ y x, focus_of_hyperbola (5, 0) → parabola y x 10) :=
by
  -- To be proved
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_and_parabola_l339_33947


namespace NUMINAMATH_GPT_number_of_students_l339_33993

theorem number_of_students (n : ℕ) :
  n < 50 ∧ n % 6 = 4 ∧ n % 4 = 2 → n = 10 ∨ n = 22 ∨ n = 34 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_students_l339_33993


namespace NUMINAMATH_GPT_matt_skips_correctly_l339_33985

-- Definitions based on conditions
def skips_per_second := 3
def jumping_time_minutes := 10
def seconds_per_minute := 60
def total_jumping_seconds := jumping_time_minutes * seconds_per_minute
def expected_skips := total_jumping_seconds * skips_per_second

-- Proof statement
theorem matt_skips_correctly :
  expected_skips = 1800 :=
by
  sorry

end NUMINAMATH_GPT_matt_skips_correctly_l339_33985


namespace NUMINAMATH_GPT_find_abc_l339_33973

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.cos x + 3 * Real.sin x

theorem find_abc (a b c : ℝ) : 
  (∀ x : ℝ, a * f x + b * f (x - c) = 1) →
  (∃ n : ℤ, a = 1 / 2 ∧ b = 1 / 2 ∧ c = (2 * n + 1) * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_find_abc_l339_33973


namespace NUMINAMATH_GPT_total_legs_of_animals_l339_33935

def num_kangaroos := 23
def num_goats := 3 * num_kangaroos
def legs_per_kangaroo := 2
def legs_per_goat := 4

def total_legs := (num_kangaroos * legs_per_kangaroo) + (num_goats * legs_per_goat)

theorem total_legs_of_animals : total_legs = 322 := by
  sorry

end NUMINAMATH_GPT_total_legs_of_animals_l339_33935


namespace NUMINAMATH_GPT_other_number_is_7_l339_33984

-- Given conditions
variable (a b : ℤ)
variable (h1 : 2 * a + 3 * b = 110)
variable (h2 : a = 32 ∨ b = 32)

-- The proof goal
theorem other_number_is_7 : (a = 7 ∧ b = 32) ∨ (a = 32 ∧ b = 7) :=
by
  sorry

end NUMINAMATH_GPT_other_number_is_7_l339_33984


namespace NUMINAMATH_GPT_parallelogram_circumference_l339_33914

-- Defining the conditions
def isParallelogram (a b : ℕ) := a = 18 ∧ b = 12

-- The theorem statement to prove
theorem parallelogram_circumference (a b : ℕ) (h : isParallelogram a b) : 2 * (a + b) = 60 :=
  by
  -- Extract the conditions from hypothesis
  cases h with
  | intro hab' hab'' =>
    sorry

end NUMINAMATH_GPT_parallelogram_circumference_l339_33914


namespace NUMINAMATH_GPT_first_player_win_boards_l339_33995

-- Define what it means for a player to guarantee a win
def first_player_guarantees_win (n m : ℕ) : Prop :=
  ¬(n % 2 = 1 ∧ m % 2 = 1)

-- The main theorem that matches the math proof problem
theorem first_player_win_boards : (first_player_guarantees_win 6 7) ∧
                                  (first_player_guarantees_win 6 8) ∧
                                  (first_player_guarantees_win 7 8) ∧
                                  (first_player_guarantees_win 8 8) ∧
                                  ¬(first_player_guarantees_win 7 7) := 
by 
sorry

end NUMINAMATH_GPT_first_player_win_boards_l339_33995


namespace NUMINAMATH_GPT_money_spent_on_video_games_l339_33986

theorem money_spent_on_video_games :
  let total_money := 50
  let fraction_books := 1 / 4
  let fraction_snacks := 2 / 5
  let fraction_apps := 1 / 5
  let spent_books := fraction_books * total_money
  let spent_snacks := fraction_snacks * total_money
  let spent_apps := fraction_apps * total_money
  let spent_other := spent_books + spent_snacks + spent_apps
  let spent_video_games := total_money - spent_other
  spent_video_games = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_money_spent_on_video_games_l339_33986


namespace NUMINAMATH_GPT_translate_vertex_l339_33910

/-- Given points A and B and their translations, verify the translated coordinates of B --/
theorem translate_vertex (A A' B B' : ℝ × ℝ)
  (hA : A = (0, 2))
  (hA' : A' = (-1, 0))
  (hB : B = (2, -1))
  (h_translation : A' = (A.1 - 1, A.2 - 2)) :
  B' = (B.1 - 1, B.2 - 2) :=
by
  sorry

end NUMINAMATH_GPT_translate_vertex_l339_33910


namespace NUMINAMATH_GPT_compute_expression_l339_33922

theorem compute_expression : (6 + 10)^2 + (6^2 + 10^2 + 6 * 10) = 452 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l339_33922


namespace NUMINAMATH_GPT_total_pairs_of_shoes_l339_33911

-- Conditions as Definitions
def blue_shoes := 540
def purple_shoes := 355
def green_shoes := purple_shoes  -- The number of green shoes is equal to the number of purple shoes

-- The theorem we need to prove
theorem total_pairs_of_shoes : blue_shoes + green_shoes + purple_shoes = 1250 := by
  sorry

end NUMINAMATH_GPT_total_pairs_of_shoes_l339_33911


namespace NUMINAMATH_GPT_common_points_l339_33915

variable {R : Type*} [LinearOrderedField R]

def eq1 (x y : R) : Prop := x - y + 2 = 0
def eq2 (x y : R) : Prop := 3 * x + y - 4 = 0
def eq3 (x y : R) : Prop := x + y - 2 = 0
def eq4 (x y : R) : Prop := 2 * x - 5 * y + 7 = 0

theorem common_points : ∃ s : Finset (R × R), 
  (∀ p ∈ s, eq1 p.1 p.2 ∨ eq2 p.1 p.2) ∧ (∀ p ∈ s, eq3 p.1 p.2 ∨ eq4 p.1 p.2) ∧ s.card = 6 :=
by
  sorry

end NUMINAMATH_GPT_common_points_l339_33915


namespace NUMINAMATH_GPT_solve_for_y_l339_33960

theorem solve_for_y (y : ℝ) (h : 9 / y^3 = y / 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l339_33960


namespace NUMINAMATH_GPT_find_B_plus_C_l339_33990

theorem find_B_plus_C 
(A B C : ℕ)
(h1 : A ≠ B)
(h2 : B ≠ C)
(h3 : C ≠ A)
(h4 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
(h5 : A < 5 ∧ B < 5 ∧ C < 5)
(h6 : 25 * A + 5 * B + C + 25 * B + 5 * C + A + 25 * C + 5 * A + B = 125 * A + 25 * A + 5 * A) : 
B + C = 4 * A := by
  sorry

end NUMINAMATH_GPT_find_B_plus_C_l339_33990


namespace NUMINAMATH_GPT_domain_of_function_l339_33942

noncomputable def function_domain := {x : ℝ | x * (3 - x) ≥ 0 ∧ x - 1 ≥ 0 }

theorem domain_of_function: function_domain = {x : ℝ | 1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l339_33942


namespace NUMINAMATH_GPT_roots_of_equation_l339_33980

def operation (a b : ℝ) : ℝ := a^2 * b + a * b - 1

theorem roots_of_equation :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ operation x₁ 1 = 0 ∧ operation x₂ 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_equation_l339_33980


namespace NUMINAMATH_GPT_equivalent_operation_l339_33953

theorem equivalent_operation (x : ℚ) : (x * (2 / 5)) / (4 / 7) = x * (7 / 10) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_operation_l339_33953


namespace NUMINAMATH_GPT_perfect_square_trinomial_l339_33928

theorem perfect_square_trinomial {m : ℝ} :
  (∃ (a : ℝ), x^2 + 2 * m * x + 9 = (x + a)^2) → (m = 3 ∨ m = -3) :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l339_33928


namespace NUMINAMATH_GPT_product_of_solutions_abs_eq_40_l339_33949

theorem product_of_solutions_abs_eq_40 :
  (∃ x1 x2 : ℝ, (|3 * x1 - 5| = 40) ∧ (|3 * x2 - 5| = 40) ∧ ((x1 * x2) = -175)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_abs_eq_40_l339_33949


namespace NUMINAMATH_GPT_shelby_stars_yesterday_l339_33958

-- Define the number of stars earned yesterday
def stars_yesterday : ℕ := sorry

-- Condition 1: In all, Shelby earned 7 gold stars
def stars_total : ℕ := 7

-- Condition 2: Today, she earned 3 more gold stars
def stars_today : ℕ := 3

-- The proof statement that combines the conditions 
-- and question to the correct answer
theorem shelby_stars_yesterday (y : ℕ) (h1 : y + stars_today = stars_total) : y = 4 := 
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_shelby_stars_yesterday_l339_33958


namespace NUMINAMATH_GPT_maximum_capacity_of_smallest_barrel_l339_33919

theorem maximum_capacity_of_smallest_barrel : 
  ∃ (A B C D E F : ℕ), 
    8 ≤ A ∧ A ≤ 16 ∧
    8 ≤ B ∧ B ≤ 16 ∧
    8 ≤ C ∧ C ≤ 16 ∧
    8 ≤ D ∧ D ≤ 16 ∧
    8 ≤ E ∧ E ≤ 16 ∧
    8 ≤ F ∧ F ≤ 16 ∧
    (A + B + C + D + E + F = 72) ∧
    ((C + D) / 2 = 14) ∧ 
    (F = 11 ∨ F = 13) ∧
    (∀ (A' : ℕ), 8 ≤ A' ∧ A' ≤ 16 ∧
      ∃ (B' C' D' E' F' : ℕ), 
      8 ≤ B' ∧ B' ≤ 16 ∧
      8 ≤ C' ∧ C' ≤ 16 ∧
      8 ≤ D' ∧ D' ≤ 16 ∧
      8 ≤ E' ∧ E' ≤ 16 ∧
      8 ≤ F' ∧ F' ≤ 16 ∧
      (A' + B' + C' + D' + E' + F' = 72) ∧
      ((C' + D') / 2 = 14) ∧ 
      (F' = 11 ∨ F' = 13) → A' ≤ A ) :=
sorry

end NUMINAMATH_GPT_maximum_capacity_of_smallest_barrel_l339_33919


namespace NUMINAMATH_GPT_gcd_20020_11011_l339_33900

theorem gcd_20020_11011 : Nat.gcd 20020 11011 = 1001 := 
by
  sorry

end NUMINAMATH_GPT_gcd_20020_11011_l339_33900


namespace NUMINAMATH_GPT_total_legs_among_animals_l339_33944

def legs (chickens sheep grasshoppers spiders : Nat) (legs_chicken legs_sheep legs_grasshopper legs_spider : Nat) : Nat :=
  (chickens * legs_chicken) + (sheep * legs_sheep) + (grasshoppers * legs_grasshopper) + (spiders * legs_spider)

theorem total_legs_among_animals :
  let chickens := 7
  let sheep := 5
  let grasshoppers := 10
  let spiders := 3
  let legs_chicken := 2
  let legs_sheep := 4
  let legs_grasshopper := 6
  let legs_spider := 8
  legs chickens sheep grasshoppers spiders legs_chicken legs_sheep legs_grasshopper legs_spider = 118 :=
by
  sorry

end NUMINAMATH_GPT_total_legs_among_animals_l339_33944


namespace NUMINAMATH_GPT_total_coins_are_correct_l339_33996

-- Define the initial number of coins
def initial_dimes : Nat := 2
def initial_quarters : Nat := 6
def initial_nickels : Nat := 5

-- Define the additional coins given by Linda's mother
def additional_dimes : Nat := 2
def additional_quarters : Nat := 10
def additional_nickels : Nat := 2 * initial_nickels

-- Calculate the total number of each type of coin
def total_dimes : Nat := initial_dimes + additional_dimes
def total_quarters : Nat := initial_quarters + additional_quarters
def total_nickels : Nat := initial_nickels + additional_nickels

-- Total number of coins
def total_coins : Nat := total_dimes + total_quarters + total_nickels

-- Theorem to prove the total number of coins is 35
theorem total_coins_are_correct : total_coins = 35 := by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_total_coins_are_correct_l339_33996


namespace NUMINAMATH_GPT_can_determine_counterfeit_coin_l339_33999

/-- 
Given 101 coins where 50 are counterfeit and each counterfeit coin 
differs by 1 gram from the genuine ones, prove that Petya can 
determine if a given coin is counterfeit with a single weighing 
using a balance scale.
-/
theorem can_determine_counterfeit_coin :
  ∃ (coins : Fin 101 → ℤ), 
    (∃ i : Fin 101, (1 ≤ i ∧ i ≤ 50 → coins i = 1) ∧ (51 ≤ i ∧ i ≤ 101 → coins i = 0)) →
    (∃ (b : ℤ), (0 < b → b ∣ 1) ∧ (¬(0 < b → b ∣ 1) → coins 101 = b)) :=
by
  sorry

end NUMINAMATH_GPT_can_determine_counterfeit_coin_l339_33999


namespace NUMINAMATH_GPT_sum_of_digits_in_T_shape_35_l339_33977

-- Define the set of digits
def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the problem variables and conditions
theorem sum_of_digits_in_T_shape_35
  (a b c d e f g h : ℕ)
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
        d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
        e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
        f ≠ g ∧ f ≠ h ∧
        g ≠ h)
  (h2 : a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ 
        e ∈ digits ∧ f ∈ digits ∧ g ∈ digits ∧ h ∈ digits)
  (h3 : a + b + c + d = 26)
  (h4 : e + b + f + g + h = 20) :
  a + b + c + d + e + f + g + h = 35 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_in_T_shape_35_l339_33977


namespace NUMINAMATH_GPT_range_of_m_l339_33940

theorem range_of_m (m : ℝ) : 
  ((m + 3) * (m - 4) < 0) → 
  (m^2 - 4 * (m + 3) ≤ 0) → 
  (-2 ≤ m ∧ m < 4) :=
by 
  intro h1 h2
  sorry

end NUMINAMATH_GPT_range_of_m_l339_33940


namespace NUMINAMATH_GPT_triangle_identity_l339_33979

theorem triangle_identity (a b c : ℝ) (B: ℝ) (hB: B = 120) :
    a^2 + a * c + c^2 - b^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_triangle_identity_l339_33979


namespace NUMINAMATH_GPT_complex_imaginary_condition_l339_33906

theorem complex_imaginary_condition (m : ℝ) : (∀ m : ℝ, (m^2 - 3*m - 4 = 0) → (m^2 - 5*m - 6) ≠ 0) ↔ (m ≠ -1 ∧ m ≠ 6) :=
by
  sorry

end NUMINAMATH_GPT_complex_imaginary_condition_l339_33906


namespace NUMINAMATH_GPT_range_of_a_l339_33983

noncomputable def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def is_increasing_on_nonneg (f : ℝ → ℝ) :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem range_of_a
  {f : ℝ → ℝ}
  (hf_even : is_even f)
  (hf_increasing : is_increasing_on_nonneg f)
  (hf_inequality : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f (a * x + 1) ≤ f (x - 3)) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l339_33983


namespace NUMINAMATH_GPT_circle_inscribed_angles_l339_33968

theorem circle_inscribed_angles (O : Type) (circle : Set O) (A B C D E F G H I J K L : O) 
  (P : ℕ) (n : ℕ) (x_deg_sum y_deg_sum : ℝ)  
  (h1 : n = 12) 
  (h2 : x_deg_sum = 45) 
  (h3 : y_deg_sum = 75) :
  x_deg_sum + y_deg_sum = 120 :=
by
  /- Proof steps are not required -/
  apply sorry

end NUMINAMATH_GPT_circle_inscribed_angles_l339_33968


namespace NUMINAMATH_GPT_parrots_in_each_cage_l339_33913

theorem parrots_in_each_cage (P : ℕ) (h : 9 * P + 9 * 6 = 72) : P = 2 :=
sorry

end NUMINAMATH_GPT_parrots_in_each_cage_l339_33913


namespace NUMINAMATH_GPT_soda_mineral_cost_l339_33978

theorem soda_mineral_cost
  (x y : ℝ)
  (h1 : 2 * x + y = 7)
  (h2 : 4 * x + 3 * y = 16) :
  10 * x + 10 * y = 45 :=
  sorry

end NUMINAMATH_GPT_soda_mineral_cost_l339_33978


namespace NUMINAMATH_GPT_min_distance_equals_sqrt2_over_2_l339_33966

noncomputable def min_distance_from_point_to_line (m n : ℝ) : ℝ :=
  (|m + n + 10|) / Real.sqrt (1^2 + 1^2)

def circle_eq (m n : ℝ) : Prop :=
  (m - 1 / 2)^2 + (n - 1 / 2)^2 = 1 / 2

theorem min_distance_equals_sqrt2_over_2 (m n : ℝ) (h1 : circle_eq m n) :
  min_distance_from_point_to_line m n = 1 / (Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_min_distance_equals_sqrt2_over_2_l339_33966


namespace NUMINAMATH_GPT_find_other_polynomial_l339_33971

variables {a b c d : ℤ}

theorem find_other_polynomial (h : ∀ P Q : ℤ, P - Q = c^2 * d^2 - a^2 * b^2) 
  (P : ℤ) (hP : P = a^2 * b^2 + c^2 * d^2 - 2 * a * b * c * d) : 
  (∃ Q : ℤ, Q = 2 * c^2 * d^2 - 2 * a * b * c * d) ∨ 
  (∃ Q : ℤ, Q = 2 * a^2 * b^2 - 2 * a * b * c * d) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_other_polynomial_l339_33971


namespace NUMINAMATH_GPT_parabola_vertex_coordinate_l339_33921

theorem parabola_vertex_coordinate :
  ∀ x_P : ℝ, 
  (P : ℝ × ℝ) → 
  (P = (x_P, 1/2 * x_P^2)) → 
  (dist P (0, 1/2) = 3) →
  P.2 = 5 / 2 :=
by sorry

end NUMINAMATH_GPT_parabola_vertex_coordinate_l339_33921


namespace NUMINAMATH_GPT_find_natural_numbers_l339_33938

theorem find_natural_numbers (n : ℕ) (x : ℕ) (y : ℕ) (hx : n = 10 * x + y) (hy : 10 * x + y = 14 * x) : n = 14 ∨ n = 28 :=
by
  sorry

end NUMINAMATH_GPT_find_natural_numbers_l339_33938
