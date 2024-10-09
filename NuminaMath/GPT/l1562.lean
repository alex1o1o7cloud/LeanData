import Mathlib

namespace solve_a_l1562_156272

variable (a : ℝ)

theorem solve_a (h : ∃ b : ℝ, (9 * x^2 + 12 * x + a) = (3 * x + b) ^ 2) : a = 4 :=
by
   sorry

end solve_a_l1562_156272


namespace mailman_distribution_l1562_156271

theorem mailman_distribution 
    (total_mail_per_block : ℕ)
    (blocks : ℕ)
    (houses_per_block : ℕ)
    (h1 : total_mail_per_block = 32)
    (h2 : blocks = 55)
    (h3 : houses_per_block = 4) :
  total_mail_per_block / houses_per_block = 8 :=
by
  sorry

end mailman_distribution_l1562_156271


namespace smallest_rectangles_cover_square_l1562_156273

theorem smallest_rectangles_cover_square :
  ∃ (n : ℕ), n = 8 ∧ ∀ (a : ℕ), ∀ (b : ℕ), (a = 2) ∧ (b = 4) → 
  ∃ (s : ℕ), s = 8 ∧ (s * s) / (a * b) = n :=
by
  sorry

end smallest_rectangles_cover_square_l1562_156273


namespace find_fx_for_l1562_156260

theorem find_fx_for {f : ℕ → ℤ} (h1 : f 0 = 1) (h2 : ∀ x, f (x + 1) = f x + 2 * x + 3) : f 2012 = 4052169 :=
by
  sorry

end find_fx_for_l1562_156260


namespace depth_multiple_of_rons_height_l1562_156225

theorem depth_multiple_of_rons_height (h d : ℕ) (Ron_height : h = 13) (water_depth : d = 208) : d = 16 * h := by
  sorry

end depth_multiple_of_rons_height_l1562_156225


namespace george_second_half_questions_l1562_156293

noncomputable def george_first_half_questions : ℕ := 6
noncomputable def points_per_question : ℕ := 3
noncomputable def george_final_score : ℕ := 30

theorem george_second_half_questions :
  (george_final_score - (george_first_half_questions * points_per_question)) / points_per_question = 4 :=
by
  sorry

end george_second_half_questions_l1562_156293


namespace inequality_lemma_l1562_156299

theorem inequality_lemma (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  (1 / (b * c + c * d + d * a - 1)) +
  (1 / (a * b + c * d + d * a - 1)) +
  (1 / (a * b + b * c + d * a - 1)) +
  (1 / (a * b + b * c + c * d - 1)) ≤ 2 :=
sorry

end inequality_lemma_l1562_156299


namespace copper_zinc_mixture_mass_bounds_l1562_156275

theorem copper_zinc_mixture_mass_bounds :
  ∀ (x y : ℝ) (D1 D2 : ℝ),
    (400 = x + y) →
    (50 = x / D1 + y / D2) →
    (8.8 ≤ D1 ∧ D1 ≤ 9) →
    (7.1 ≤ D2 ∧ D2 ≤ 7.2) →
    (200 ≤ x ∧ x ≤ 233) ∧ (167 ≤ y ∧ y ≤ 200) :=
sorry

end copper_zinc_mixture_mass_bounds_l1562_156275


namespace AM_GM_proof_equality_condition_l1562_156278

variable (a b : ℝ)
variable (ha : 0 < a) (hb : 0 < b)

theorem AM_GM_proof : (a + b)^3 / (a^2 * b) ≥ 27 / 4 :=
sorry

theorem equality_condition : (a + b)^3 / (a^2 * b) = 27 / 4 ↔ a = 2 * b :=
sorry

end AM_GM_proof_equality_condition_l1562_156278


namespace find_c_l1562_156203

theorem find_c (c : ℝ) 
  (h : (⟨9, c⟩ : ℝ × ℝ) = (11/13 : ℝ) • ⟨-3, 2⟩) : 
  c = 19 :=
sorry

end find_c_l1562_156203


namespace product_of_numbers_l1562_156223

theorem product_of_numbers (a b : ℕ) (hcf : ℕ := 12) (lcm : ℕ := 205) (ha : Nat.gcd a b = hcf) (hb : Nat.lcm a b = lcm) : a * b = 2460 := by
  sorry

end product_of_numbers_l1562_156223


namespace number_of_grade11_students_l1562_156210

-- Define the total number of students in the high school.
def total_students : ℕ := 900

-- Define the total number of students selected in the sample.
def sample_students : ℕ := 45

-- Define the number of Grade 10 students in the sample.
def grade10_students_sample : ℕ := 20

-- Define the number of Grade 12 students in the sample.
def grade12_students_sample : ℕ := 10

-- Prove the number of Grade 11 students in the school is 300.
theorem number_of_grade11_students :
  (sample_students - grade10_students_sample - grade12_students_sample) * (total_students / sample_students) = 300 :=
by
  sorry

end number_of_grade11_students_l1562_156210


namespace sqrt2_times_sqrt5_eq_sqrt10_l1562_156253

theorem sqrt2_times_sqrt5_eq_sqrt10 : (Real.sqrt 2) * (Real.sqrt 5) = Real.sqrt 10 := 
by
  sorry

end sqrt2_times_sqrt5_eq_sqrt10_l1562_156253


namespace bike_covered_distance_l1562_156261

theorem bike_covered_distance
  (time : ℕ) 
  (truck_distance : ℕ) 
  (speed_difference : ℕ) 
  (bike_speed truck_speed : ℕ)
  (h_time : time = 8)
  (h_truck_distance : truck_distance = 112)
  (h_speed_difference : speed_difference = 3)
  (h_truck_speed : truck_speed = truck_distance / time)
  (h_speed_relation : truck_speed = bike_speed + speed_difference) :
  bike_speed * time = 88 :=
by
  -- The proof is omitted
  sorry

end bike_covered_distance_l1562_156261


namespace overall_average_of_marks_l1562_156291

theorem overall_average_of_marks (n total_boys passed_boys failed_boys avg_passed avg_failed : ℕ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 105)
  (h3 : failed_boys = 15)
  (h4 : total_boys = passed_boys + failed_boys)
  (h5 : avg_passed = 39)
  (h6 : avg_failed = 15) :
  ((passed_boys * avg_passed + failed_boys * avg_failed) / total_boys = 36) :=
by
  sorry

end overall_average_of_marks_l1562_156291


namespace book_total_pages_l1562_156252

-- Define the conditions given in the problem
def pages_per_night : ℕ := 12
def nights_to_finish : ℕ := 10

-- State that the total number of pages in the book is 120 given the conditions
theorem book_total_pages : (pages_per_night * nights_to_finish) = 120 :=
by sorry

end book_total_pages_l1562_156252


namespace angle_same_terminal_side_315_l1562_156267

theorem angle_same_terminal_side_315 (k : ℤ) : ∃ α, α = k * 360 + 315 ∧ α = -45 :=
by
  use -45
  sorry

end angle_same_terminal_side_315_l1562_156267


namespace pieces_of_meat_per_slice_eq_22_l1562_156236

def number_of_pepperoni : Nat := 30
def number_of_ham : Nat := 2 * number_of_pepperoni
def number_of_sausage : Nat := number_of_pepperoni + 12
def total_meat : Nat := number_of_pepperoni + number_of_ham + number_of_sausage
def number_of_slices : Nat := 6

theorem pieces_of_meat_per_slice_eq_22 : total_meat / number_of_slices = 22 :=
by
  sorry

end pieces_of_meat_per_slice_eq_22_l1562_156236


namespace divide_45_to_get_900_l1562_156285

theorem divide_45_to_get_900 (x : ℝ) (h : 45 / x = 900) : x = 0.05 :=
by
  sorry

end divide_45_to_get_900_l1562_156285


namespace max_area_14_5_l1562_156200

noncomputable def rectangle_max_area (P D : ℕ) (x y : ℝ) : ℝ :=
  if (2 * x + 2 * y = P) ∧ (x^2 + y^2 = D^2) then x * y else 0

theorem max_area_14_5 :
  ∃ (x y : ℝ), (2 * x + 2 * y = 14) ∧ (x^2 + y^2 = 5^2) ∧ rectangle_max_area 14 5 x y = 12.25 :=
by
  sorry

end max_area_14_5_l1562_156200


namespace kevin_ends_with_604_cards_l1562_156254

theorem kevin_ends_with_604_cards : 
  ∀ (initial_cards found_cards : ℕ), initial_cards = 65 → found_cards = 539 → initial_cards + found_cards = 604 :=
by
  intros initial_cards found_cards h_initial h_found
  sorry

end kevin_ends_with_604_cards_l1562_156254


namespace increase_in_area_400ft2_l1562_156240

theorem increase_in_area_400ft2 (l w : ℝ) (h₁ : l = 60) (h₂ : w = 20)
  (h₃ : 4 * (l + w) = 4 * (4 * (l + w) / 4 / 4 )):
  (4 * (l + w) / 4) ^ 2 - l * w = 400 := by
  sorry

end increase_in_area_400ft2_l1562_156240


namespace red_pigment_weight_in_brown_paint_l1562_156280

theorem red_pigment_weight_in_brown_paint :
  ∀ (M G : ℝ), 
    (M + G = 10) → 
    (0.5 * M + 0.3 * G = 4) →
    0.5 * M = 2.5 :=
by sorry

end red_pigment_weight_in_brown_paint_l1562_156280


namespace birds_on_fence_l1562_156226

theorem birds_on_fence :
  let i := 12           -- initial birds
  let added1 := 8       -- birds that land first
  let T := i + added1   -- total first stage birds
  
  let fly_away1 := 5
  let join1 := 3
  let W := T - fly_away1 + join1   -- birds after some fly away, others join
  
  let D := W * 2       -- birds doubles
  
  let fly_away2 := D * 0.25  -- 25% fly away
  let D_after_fly_away := D - fly_away2
  
  let return_birds := 2        -- 2.5 birds return, rounded down to 2
  let final_birds := D_after_fly_away + return_birds
  
  final_birds = 29 := 
by {
  sorry
}

end birds_on_fence_l1562_156226


namespace kim_knit_sweaters_total_l1562_156298

theorem kim_knit_sweaters_total :
  ∀ (M T W R F : ℕ), 
    M = 8 →
    T = M + 2 →
    W = T - 4 →
    R = T - 4 →
    F = M / 2 →
    M + T + W + R + F = 34 :=
by
  intros M T W R F hM hT hW hR hF
  rw [hM, hT, hW, hR, hF]
  norm_num
  sorry

end kim_knit_sweaters_total_l1562_156298


namespace toy_production_difference_l1562_156294

variables (w t : ℕ)
variable  (t_nonneg : 0 < t) -- assuming t is always non-negative for a valid working hour.
variable  (h : w = 3 * t)

theorem toy_production_difference : 
  (w * t) - ((w + 5) * (t - 3)) = 4 * t + 15 :=
by
  sorry

end toy_production_difference_l1562_156294


namespace weight_of_empty_jar_l1562_156243

variable (W : ℝ) -- Weight of the empty jar
variable (w : ℝ) -- Weight of water for one-fifth of the jar

-- Conditions
variable (h1 : W + w = 560)
variable (h2 : W + 4 * w = 740)

-- Theorem statement
theorem weight_of_empty_jar (W w : ℝ) (h1 : W + w = 560) (h2 : W + 4 * w = 740) : W = 500 := 
by
  sorry

end weight_of_empty_jar_l1562_156243


namespace tan_alpha_eq_three_sin_cos_l1562_156256

theorem tan_alpha_eq_three_sin_cos (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin α * Real.cos α = 3 / 10 :=
by 
  sorry

end tan_alpha_eq_three_sin_cos_l1562_156256


namespace triangle_height_l1562_156274

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) 
  (h_area : area = 615) (h_base : base = 123) 
  (area_formula : area = (base * height) / 2) : height = 10 := 
by 
  sorry

end triangle_height_l1562_156274


namespace total_caffeine_is_correct_l1562_156204

def first_drink_caffeine := 250 -- milligrams
def first_drink_size := 12 -- ounces

def second_drink_caffeine_per_ounce := (first_drink_caffeine / first_drink_size) * 3
def second_drink_size := 8 -- ounces
def second_drink_caffeine := second_drink_caffeine_per_ounce * second_drink_size

def third_drink_concentration := 18 -- milligrams per milliliter
def third_drink_size := 150 -- milliliters
def third_drink_caffeine := third_drink_concentration * third_drink_size

def caffeine_pill_caffeine := first_drink_caffeine + second_drink_caffeine + third_drink_caffeine

def total_caffeine_consumed := first_drink_caffeine + second_drink_caffeine + third_drink_caffeine + caffeine_pill_caffeine

theorem total_caffeine_is_correct : total_caffeine_consumed = 6900 :=
by
  sorry

end total_caffeine_is_correct_l1562_156204


namespace sum_x_y_is_9_l1562_156250

-- Definitions of the conditions
variables (x y S : ℝ)
axiom h1 : x + y = S
axiom h2 : x - y = 3
axiom h3 : x^2 - y^2 = 27

-- The theorem to prove
theorem sum_x_y_is_9 : S = 9 :=
by
  -- Placeholder for the proof
  sorry

end sum_x_y_is_9_l1562_156250


namespace find_period_for_interest_l1562_156248

noncomputable def period_for_compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) : ℝ :=
  (Real.log A - Real.log P) / (n * Real.log (1 + r / n))

theorem find_period_for_interest :
  period_for_compound_interest 8000 0.15 1 11109 = 2 := 
sorry

end find_period_for_interest_l1562_156248


namespace dot_product_equals_6_l1562_156249

-- Define the vectors
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)

-- Define the scalar multiplication and addition
def scaled_added_vector : ℝ × ℝ := (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2)

-- Define the dot product
def dot_product : ℝ := scaled_added_vector.1 * vec_a.1 + scaled_added_vector.2 * vec_a.2

-- Assertion that the dot product is equal to 6
theorem dot_product_equals_6 : dot_product = 6 :=
by
  sorry

end dot_product_equals_6_l1562_156249


namespace difference_between_blue_and_red_balls_l1562_156242

-- Definitions and conditions
def number_of_blue_balls := ℕ
def number_of_red_balls := ℕ
def difference_between_balls (m n : ℕ) := m - n

-- Problem statement: Prove that the difference between number_of_blue_balls and number_of_red_balls
-- can be any natural number greater than 1.
theorem difference_between_blue_and_red_balls (m n : ℕ) (h1 : m > n) (h2 : 
  let P_same := (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1))
  let P_diff := 2 * (n * m) / ((n + m) * (n + m - 1))
  P_same = P_diff
  ) : ∃ a : ℕ, a > 1 ∧ a = m - n :=
by
  sorry

end difference_between_blue_and_red_balls_l1562_156242


namespace depth_of_second_hole_l1562_156213

theorem depth_of_second_hole :
  let workers1 := 45
  let hours1 := 8
  let depth1 := 30
  let man_hours1 := workers1 * hours1 -- 360 man-hours
  let workers2 := 45 + 35 -- 80 workers
  let hours2 := 6
  let man_hours2 := workers2 * hours2 -- 480 man-hours
  let depth2 := (man_hours2 * depth1) / man_hours1 -- value to solve for
  depth2 = 40 :=
by
  sorry

end depth_of_second_hole_l1562_156213


namespace sherman_total_weekly_driving_time_l1562_156259

def daily_commute_time : Nat := 1  -- 1 hour for daily round trip commute time
def work_days : Nat := 5  -- Sherman works 5 days a week
def weekend_day_driving_time : Nat := 2  -- 2 hours of driving each weekend day
def weekend_days : Nat := 2  -- There are 2 weekend days

theorem sherman_total_weekly_driving_time :
  daily_commute_time * work_days + weekend_day_driving_time * weekend_days = 9 := 
by
  sorry

end sherman_total_weekly_driving_time_l1562_156259


namespace initial_manufacturing_cost_l1562_156245

theorem initial_manufacturing_cost
  (P : ℝ) -- selling price
  (initial_cost new_cost : ℝ)
  (initial_profit new_profit : ℝ)
  (h1 : initial_profit = 0.25 * P)
  (h2 : new_profit = 0.50 * P)
  (h3 : new_cost = 50)
  (h4 : new_profit = P - new_cost)
  (h5 : initial_profit = P - initial_cost) :
  initial_cost = 75 := 
by
  sorry

end initial_manufacturing_cost_l1562_156245


namespace initial_percentage_of_water_is_20_l1562_156247

theorem initial_percentage_of_water_is_20 : 
  ∀ (P : ℝ) (total_initial_volume added_water total_final_volume final_percentage initial_water_percentage : ℝ), 
    total_initial_volume = 125 ∧ 
    added_water = 8.333333333333334 ∧ 
    total_final_volume = total_initial_volume + added_water ∧ 
    final_percentage = 25 ∧ 
    initial_water_percentage = (initial_water_percentage / total_initial_volume) * 100 ∧ 
    (final_percentage / 100) * total_final_volume = added_water + (initial_water_percentage / 100) * total_initial_volume → 
    initial_water_percentage = 20 := 
by 
  sorry

end initial_percentage_of_water_is_20_l1562_156247


namespace area_of_polygon_ABLFKJ_l1562_156222

theorem area_of_polygon_ABLFKJ 
  (side_length : ℝ) (area_square : ℝ) (midpoint_l : ℝ) (area_triangle : ℝ)
  (remaining_area_each_square : ℝ) (total_area : ℝ)
  (h1 : side_length = 6)
  (h2 : area_square = side_length * side_length)
  (h3 : midpoint_l = side_length / 2)
  (h4 : area_triangle = 0.5 * side_length * midpoint_l)
  (h5 : remaining_area_each_square = area_square - 2 * area_triangle)
  (h6 : total_area = 3 * remaining_area_each_square)
  : total_area = 54 :=
by
  sorry

end area_of_polygon_ABLFKJ_l1562_156222


namespace cameron_gold_tokens_l1562_156257

/-- Cameron starts with 90 red tokens and 60 blue tokens. 
  Booth 1 exchange: 3 red tokens for 1 gold token and 2 blue tokens.
  Booth 2 exchange: 2 blue tokens for 1 gold token and 1 red token.
  Cameron stops when fewer than 3 red tokens or 2 blue tokens remain.
  Prove that the number of gold tokens Cameron ends up with is 148.
-/
theorem cameron_gold_tokens :
  ∃ (x y : ℕ), 
    90 - 3 * x + y < 3 ∧
    60 + 2 * x - 2 * y < 2 ∧
    (x + y = 148) :=
  sorry

end cameron_gold_tokens_l1562_156257


namespace find_cos_7theta_l1562_156287

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end find_cos_7theta_l1562_156287


namespace exists_k_l1562_156251

def satisfies_condition (a b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → (a n - a (n-1)) * (a n - a (n-2)) + (b n - b (n-1)) * (b n - b (n-2)) = 0

theorem exists_k (a b : ℕ → ℤ) 
  (h : satisfies_condition a b) : 
  ∃ k : ℕ, k > 0 ∧ a k = a (k + 2008) :=
sorry

end exists_k_l1562_156251


namespace value_of_ac_over_bd_l1562_156231

theorem value_of_ac_over_bd (a b c d : ℝ) 
  (h1 : a = 4 * b)
  (h2 : b = 3 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := 
by
  sorry

end value_of_ac_over_bd_l1562_156231


namespace binary_sum_is_11_l1562_156228

-- Define the binary numbers
def b1 : ℕ := 5  -- equivalent to 101 in binary
def b2 : ℕ := 6  -- equivalent to 110 in binary

-- Define the expected sum in decimal
def expected_sum : ℕ := 11

-- The theorem statement
theorem binary_sum_is_11 : b1 + b2 = expected_sum := by
  sorry

end binary_sum_is_11_l1562_156228


namespace monotonic_decreasing_intervals_l1562_156220

theorem monotonic_decreasing_intervals (α : ℝ) (hα : α < 0) :
  (∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → x ^ α > y ^ α) ∧ 
  (∀ x y : ℝ, x < y ∧ 0 < x ∧ 0 < y → x ^ α > y ^ α) :=
by
  sorry

end monotonic_decreasing_intervals_l1562_156220


namespace sin_cos_product_l1562_156237

theorem sin_cos_product (x : ℝ) (h₁ : 0 < x) (h₂ : x < π / 2) (h₃ : Real.sin x = 3 * Real.cos x) : 
  Real.sin x * Real.cos x = 3 / 10 :=
by
  sorry

end sin_cos_product_l1562_156237


namespace polynomial_inequality_l1562_156201

theorem polynomial_inequality (a b c : ℝ)
  (h1 : ∃ r1 r2 r3 : ℝ, (r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3) ∧ 
    (∀ t : ℝ, (t - r1) * (t - r2) * (t - r3) = t^3 + a*t^2 + b*t + c))
  (h2 : ¬ ∃ x : ℝ, (x^2 + x + 2013)^3 + a*(x^2 + x + 2013)^2 + b*(x^2 + x + 2013) + c = 0) :
  t^3 + a*2013^2 + b*2013 + c > 1 / 64 :=
sorry

end polynomial_inequality_l1562_156201


namespace average_monthly_balance_is_150_l1562_156290

-- Define the balances for each month
def balance_jan : ℕ := 100
def balance_feb : ℕ := 200
def balance_mar : ℕ := 150
def balance_apr : ℕ := 150

-- Define the number of months
def num_months : ℕ := 4

-- Define the total sum of balances
def total_balance : ℕ := balance_jan + balance_feb + balance_mar + balance_apr

-- Define the average balance
def average_balance : ℕ := total_balance / num_months

-- Goal is to prove that the average monthly balance is 150 dollars
theorem average_monthly_balance_is_150 : average_balance = 150 :=
by
  sorry

end average_monthly_balance_is_150_l1562_156290


namespace sahil_selling_price_l1562_156262

def initial_cost : ℝ := 14000
def repair_cost : ℝ := 5000
def transportation_charges : ℝ := 1000
def profit_percent : ℝ := 50

noncomputable def total_cost : ℝ := initial_cost + repair_cost + transportation_charges
noncomputable def profit : ℝ := profit_percent / 100 * total_cost
noncomputable def selling_price : ℝ := total_cost + profit

theorem sahil_selling_price :
  selling_price = 30000 := by
  sorry

end sahil_selling_price_l1562_156262


namespace lemonade_stand_total_profit_l1562_156288

theorem lemonade_stand_total_profit :
  let day1_revenue := 21 * 4
  let day1_expenses := 10 + 5 + 3
  let day1_profit := day1_revenue - day1_expenses

  let day2_revenue := 18 * 5
  let day2_expenses := 12 + 6 + 4
  let day2_profit := day2_revenue - day2_expenses

  let day3_revenue := 25 * 4
  let day3_expenses := 8 + 4 + 3 + 2
  let day3_profit := day3_revenue - day3_expenses

  let total_profit := day1_profit + day2_profit + day3_profit

  total_profit = 217 := by
    sorry

end lemonade_stand_total_profit_l1562_156288


namespace problem_solution_l1562_156244

variables (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x_0 : ℝ, x_0^2 + (a-1)*x_0 + 1 < 0

theorem problem_solution (h₁ : p a ∨ q a) (h₂ : ¬(p a ∧ q a)) :
  -1 ≤ a ∧ a ≤ 1 ∨ a > 3 :=
sorry

end problem_solution_l1562_156244


namespace cylinder_surface_area_l1562_156241

theorem cylinder_surface_area (side : ℝ) (h : ℝ) (r : ℝ) : 
  side = 2 ∧ h = side ∧ r = side → 
  (2 * Real.pi * r^2 + 2 * Real.pi * r * h) = 16 * Real.pi := 
by
  intro h
  sorry

end cylinder_surface_area_l1562_156241


namespace incorrect_tripling_radius_l1562_156263

-- Let r be the radius of a circle, and A be its area.
-- The claim is that tripling the radius quadruples the area.
-- We need to prove this claim is incorrect.

theorem incorrect_tripling_radius (r : ℝ) (A : ℝ) (π : ℝ) (hA : A = π * r^2) : 
    (π * (3 * r)^2) ≠ 4 * A :=
by
  sorry

end incorrect_tripling_radius_l1562_156263


namespace find_m_l1562_156284

theorem find_m (m : ℝ) : 
  (∀ (x y : ℝ), (y = x + m ∧ x = 0) → y = m) ∧
  (∀ (x y : ℝ), (y = 2 * x - 2 ∧ x = 0) → y = -2) ∧
  (∀ (x : ℝ), (∃ y : ℝ, (y = x + m ∧ x = 0) ∧ (y = 2 * x - 2 ∧ x = 0))) → 
  m = -2 :=
by 
  sorry

end find_m_l1562_156284


namespace ratio_alison_brittany_l1562_156232

def kent_money : ℕ := 1000
def brooke_money : ℕ := 2 * kent_money
def brittany_money : ℕ := 4 * brooke_money
def alison_money : ℕ := 4000

theorem ratio_alison_brittany : alison_money * 2 = brittany_money :=
by
  sorry

end ratio_alison_brittany_l1562_156232


namespace rationalize_denominator_l1562_156211

theorem rationalize_denominator (cbrt : ℝ → ℝ) (h₁ : cbrt 81 = 3 * cbrt 3) :
  1 / (cbrt 3 + cbrt 81) = cbrt 9 / 12 :=
sorry

end rationalize_denominator_l1562_156211


namespace smallest_k_l1562_156230

theorem smallest_k (k : ℕ) (h1 : k > 1) (h2 : k % 19 = 1) (h3 : k % 7 = 1) (h4 : k % 3 = 1) : k = 400 :=
by
  sorry

end smallest_k_l1562_156230


namespace part1_part2_l1562_156270

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := |2 * x| + |2 * x - 3|

-- Part 1: Proving the inequality solution
theorem part1 (x : ℝ) (h : f x ≤ 5) :
  -1/2 ≤ x ∧ x ≤ 2 :=
sorry

-- Part 2: Proving the range of m
theorem part2 (x₀ m : ℝ) (h1 : x₀ ∈ Set.Ici 1)
  (h2 : f x₀ + m ≤ x₀ + 3/x₀) :
  m ≤ 1 :=
sorry

end part1_part2_l1562_156270


namespace arlo_books_l1562_156216

theorem arlo_books (total_items : ℕ) (books_ratio : ℕ) (pens_ratio : ℕ) (notebooks_ratio : ℕ) 
  (ratio_sum : ℕ) (items_per_part : ℕ) (parts_for_books : ℕ) (total_parts : ℕ) :
  total_items = 600 →
  books_ratio = 7 →
  pens_ratio = 3 →
  notebooks_ratio = 2 →
  total_parts = books_ratio + pens_ratio + notebooks_ratio →
  items_per_part = total_items / total_parts →
  parts_for_books = books_ratio →
  parts_for_books * items_per_part = 350 := by
  intros
  sorry

end arlo_books_l1562_156216


namespace incorrect_conclusion_D_l1562_156234

-- Define lines and planes
variables (l m n : Type) -- lines
variables (α β γ : Type) -- planes

-- Define the conditions
def intersection_planes (p1 p2 : Type) : Type := sorry
def perpendicular (a b : Type) : Prop := sorry

-- Given conditions for option D
axiom h1 : intersection_planes α β = m
axiom h2 : intersection_planes β γ = l
axiom h3 : intersection_planes γ α = n
axiom h4 : perpendicular l m
axiom h5 : perpendicular l n

-- Theorem stating that the conclusion of option D is incorrect
theorem incorrect_conclusion_D : ¬ perpendicular m n :=
by sorry

end incorrect_conclusion_D_l1562_156234


namespace max_arithmetic_sum_l1562_156238

def a1 : ℤ := 113
def d : ℤ := -4

def S (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem max_arithmetic_sum : S 29 = 1653 :=
by
  sorry

end max_arithmetic_sum_l1562_156238


namespace unique_solution_of_equation_l1562_156292

theorem unique_solution_of_equation (x y : ℝ) (h : |x + 2| + (y - 1)^2 = 0) : x = -2 ∧ y = 1 :=
by
  sorry

end unique_solution_of_equation_l1562_156292


namespace initial_deck_card_count_l1562_156268

theorem initial_deck_card_count (r n : ℕ) (h1 : n = 2 * r) (h2 : n + 4 = 3 * r) : r + n = 12 := by
  sorry

end initial_deck_card_count_l1562_156268


namespace jovana_total_shells_l1562_156265

def initial_amount : ℕ := 5
def added_amount : ℕ := 23
def total_amount : ℕ := 28

theorem jovana_total_shells : initial_amount + added_amount = total_amount := by
  sorry

end jovana_total_shells_l1562_156265


namespace x5_y5_z5_value_is_83_l1562_156296

noncomputable def find_x5_y5_z5_value (x y z : ℝ) : Prop :=
  (x + y + z = 3) ∧ 
  (x^3 + y^3 + z^3 = 15) ∧
  (x^4 + y^4 + z^4 = 35) ∧
  (x^2 + y^2 + z^2 < 10) →
  x^5 + y^5 + z^5 = 83

theorem x5_y5_z5_value_is_83 (x y z : ℝ) :
  find_x5_y5_z5_value x y z :=
  sorry

end x5_y5_z5_value_is_83_l1562_156296


namespace unbroken_seashells_l1562_156289

theorem unbroken_seashells (total_seashells broken_seashells unbroken_seashells : ℕ) 
  (h_total : total_seashells = 7) (h_broken : broken_seashells = 4) 
  (h_unbroken : unbroken_seashells = total_seashells - broken_seashells) : 
  unbroken_seashells = 3 :=
by 
  rw [h_total, h_broken] at h_unbroken
  exact h_unbroken

end unbroken_seashells_l1562_156289


namespace total_apples_l1562_156221

theorem total_apples (x : ℕ) : 
    (x - x / 5 - x / 12 - x / 8 - x / 20 - x / 4 - x / 7 - x / 30 - 4 * (x / 30) - 300 ≤ 50) -> 
    x = 3360 :=
by
    sorry

end total_apples_l1562_156221


namespace system_solution_a_l1562_156209

theorem system_solution_a (x y a : ℝ) (h1 : 3 * x + y = a) (h2 : 2 * x + 5 * y = 2 * a) (hx : x = 3) : a = 13 :=
by
  sorry

end system_solution_a_l1562_156209


namespace problem1_problem2_l1562_156208

noncomputable def tan_inv_3_value : ℝ := -4 / 5

theorem problem1 (α : ℝ) (h : Real.tan α = 3) :
  Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = tan_inv_3_value := 
sorry

noncomputable def f (θ : ℝ) : ℝ := 
  (2 * Real.cos θ ^ 3 + Real.sin (2 * Real.pi - θ) ^ 2 + 
   Real.sin (Real.pi / 2 + θ) - 3) / 
  (2 + 2 * Real.cos (Real.pi + θ) ^ 2 + Real.cos (-θ))

theorem problem2 :
  f (Real.pi / 3) = -1 / 2 :=
sorry

end problem1_problem2_l1562_156208


namespace parallel_lines_implies_value_of_a_l1562_156258

theorem parallel_lines_implies_value_of_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y = 0 ∧ x + (a-1)*y + (a^2-1) = 0 → 
  (- a / 2) = - (1 / (a-1))) → a = 2 :=
sorry

end parallel_lines_implies_value_of_a_l1562_156258


namespace tom_apple_fraction_l1562_156279

theorem tom_apple_fraction (initial_oranges initial_apples oranges_sold_fraction oranges_remaining total_fruits_remaining apples_initial apples_sold_fraction : ℕ→ℚ) :
  initial_oranges = 40 →
  initial_apples = 70 →
  oranges_sold_fraction = 1 / 4 →
  oranges_remaining = initial_oranges - initial_oranges * oranges_sold_fraction →
  total_fruits_remaining = 65 →
  total_fruits_remaining = oranges_remaining + (initial_apples - initial_apples * apples_sold_fraction) →
  apples_sold_fraction = 1 / 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end tom_apple_fraction_l1562_156279


namespace count_integer_solutions_l1562_156297

theorem count_integer_solutions :
  (2 * 9^2 + 5 * 9 * -4 + 3 * (-4)^2 = 30) →
  ∃ S : Finset (ℤ × ℤ), (∀ x y : ℤ, ((2 * x ^ 2 + 5 * x * y + 3 * y ^ 2 = 30) ↔ (x, y) ∈ S)) ∧ 
  S.card = 16 :=
by sorry

end count_integer_solutions_l1562_156297


namespace pens_bought_l1562_156205

theorem pens_bought
  (P : ℝ)
  (cost := 36 * P)
  (discount := 0.99 * P)
  (profit_percent := 0.1)
  (profit := (40 * discount) - cost)
  (profit_eq : profit = profit_percent * cost) :
  40 = 40 := 
by
  sorry

end pens_bought_l1562_156205


namespace Alice_spent_19_percent_l1562_156207

variable (A B A': ℝ)
def Bob_less_money_than_Alice (A B : ℝ) : Prop :=
  B = 0.9 * A

def Alice_less_money_than_Bob (B A' : ℝ) : Prop :=
  A' = 0.9 * B

theorem Alice_spent_19_percent (A B A' : ℝ) 
  (h1 : Bob_less_money_than_Alice A B)
  (h2 : Alice_less_money_than_Bob B A') :
  ((A - A') / A) * 100 = 19 :=
by
  sorry

end Alice_spent_19_percent_l1562_156207


namespace chair_cost_l1562_156283

/--
Nadine went to a garage sale and spent $56. She bought a table for $34 and 2 chairs.
Each chair cost the same amount.
Prove that one chair cost $11.
-/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) (total_cost : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  total_cost = 56 - 34 →
  total_cost / num_chairs = 11 :=
by
  sorry

end chair_cost_l1562_156283


namespace injective_functions_count_l1562_156215

theorem injective_functions_count (m n : ℕ) (h_mn : m ≥ n) (h_n2 : n ≥ 2) :
  ∃ k, k = Nat.choose m n * (2^n - n - 1) :=
sorry

end injective_functions_count_l1562_156215


namespace angle_same_after_minutes_l1562_156277

def angle_between_hands (H M : ℝ) : ℝ :=
  abs (30 * H - 5.5 * M)

theorem angle_same_after_minutes (x : ℝ) :
  x = 54 + 6 / 11 → 
  angle_between_hands (5 + (x / 60)) x = 150 :=
by
  sorry

end angle_same_after_minutes_l1562_156277


namespace cone_from_sector_radius_l1562_156202

theorem cone_from_sector_radius (r : ℝ) (slant_height : ℝ) : 
  (r = 9) ∧ (slant_height = 12) ↔ 
  (∃ (sector_angle : ℝ) (sector_radius : ℝ), 
    sector_angle = 270 ∧ sector_radius = 12 ∧ 
    slant_height = sector_radius ∧ 
    (2 * π * r = sector_angle / 360 * 2 * π * sector_radius)) :=
by
  sorry

end cone_from_sector_radius_l1562_156202


namespace factor_polynomial_l1562_156286

variable (x : ℝ)

theorem factor_polynomial : (270 * x^3 - 90 * x^2 + 18 * x) = 18 * x * (15 * x^2 - 5 * x + 1) :=
by 
  sorry

end factor_polynomial_l1562_156286


namespace mrs_hilt_has_more_money_l1562_156214

/-- Mrs. Hilt has two pennies, two dimes, and two nickels. 
    Jacob has four pennies, one nickel, and one dime. 
    Prove that Mrs. Hilt has $0.13 more than Jacob. -/
theorem mrs_hilt_has_more_money 
  (hilt_pennies hilt_dimes hilt_nickels : ℕ)
  (jacob_pennies jacob_dimes jacob_nickels : ℕ)
  (value_penny value_nickel value_dime : ℝ)
  (H1 : hilt_pennies = 2) (H2 : hilt_dimes = 2) (H3 : hilt_nickels = 2)
  (H4 : jacob_pennies = 4) (H5 : jacob_dimes = 1) (H6 : jacob_nickels = 1)
  (H7 : value_penny = 0.01) (H8 : value_nickel = 0.05) (H9 : value_dime = 0.10) :
  ((hilt_pennies * value_penny + hilt_dimes * value_dime + hilt_nickels * value_nickel) 
   - (jacob_pennies * value_penny + jacob_dimes * value_dime + jacob_nickels * value_nickel) 
   = 0.13) :=
by sorry

end mrs_hilt_has_more_money_l1562_156214


namespace range_of_quadratic_function_is_geq_11_over_4_l1562_156295

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - x + 3

-- Define the range of the quadratic function
def range_of_quadratic_function := {y : ℝ | ∃ x : ℝ, quadratic_function x = y}

-- Prove the statement
theorem range_of_quadratic_function_is_geq_11_over_4 : range_of_quadratic_function = {y : ℝ | y ≥ 11 / 4} :=
by
  sorry

end range_of_quadratic_function_is_geq_11_over_4_l1562_156295


namespace triangle_inequality_values_l1562_156233

theorem triangle_inequality_values (x : ℕ) :
  x ≥ 2 ∧ x < 10 ↔ (x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9) :=
by sorry

end triangle_inequality_values_l1562_156233


namespace kim_fraction_of_shirts_given_l1562_156224

open Nat

theorem kim_fraction_of_shirts_given (d : ℕ) (s_left : ℕ) (one_dozen := 12) 
  (original_shirts := 4 * one_dozen) 
  (given_shirts := original_shirts - s_left) 
  (fraction_given := given_shirts / original_shirts) 
  (hc1 : d = one_dozen) 
  (hc2 : s_left = 32) 
  : fraction_given = 1 / 3 := 
by 
  sorry

end kim_fraction_of_shirts_given_l1562_156224


namespace taoqi_has_higher_utilization_rate_l1562_156281

noncomputable def area_square (side_length : ℝ) : ℝ :=
  side_length * side_length

noncomputable def area_circle (radius : ℝ) : ℝ :=
  Real.pi * radius * radius

noncomputable def utilization_rate (cut_area : ℝ) (original_area : ℝ) : ℝ :=
  cut_area / original_area

noncomputable def tao_qi_utilization_rate : ℝ :=
  let side_length := 9
  let square_area := area_square side_length
  let radius := side_length / 2
  let circle_area := area_circle radius
  utilization_rate circle_area square_area

noncomputable def xiao_xiao_utilization_rate : ℝ :=
  let diameter := 9
  let radius := diameter / 2
  let large_circle_area := area_circle radius
  let small_circle_radius := diameter / 6
  let small_circle_area := area_circle small_circle_radius
  let total_small_circles_area := 7 * small_circle_area
  utilization_rate total_small_circles_area large_circle_area

-- Theorem statement reflecting the proof problem:
theorem taoqi_has_higher_utilization_rate :
  tao_qi_utilization_rate > xiao_xiao_utilization_rate := by sorry

end taoqi_has_higher_utilization_rate_l1562_156281


namespace Julie_hours_per_week_school_l1562_156255

noncomputable def summer_rate : ℚ := 4500 / (36 * 10)

noncomputable def school_rate : ℚ := summer_rate * 1.10

noncomputable def total_school_hours_needed : ℚ := 9000 / school_rate

noncomputable def hours_per_week_school : ℚ := total_school_hours_needed / 40

theorem Julie_hours_per_week_school : hours_per_week_school = 16.36 := by
  sorry

end Julie_hours_per_week_school_l1562_156255


namespace original_price_of_computer_l1562_156229

theorem original_price_of_computer (P : ℝ) (h1 : 1.30 * P = 364) (h2 : 2 * P = 560) : P = 280 :=
by 
  -- The proof is skipped as per instruction
  sorry

end original_price_of_computer_l1562_156229


namespace stamp_total_cost_l1562_156269

theorem stamp_total_cost :
  let price_A := 2
  let price_B := 3
  let price_C := 5
  let num_A := 150
  let num_B := 90
  let num_C := 60
  let discount_A := if num_A > 100 then 0.20 else 0
  let discount_B := if num_B > 50 then 0.15 else 0
  let discount_C := if num_C > 30 then 0.10 else 0
  let cost_A := num_A * price_A * (1 - discount_A)
  let cost_B := num_B * price_B * (1 - discount_B)
  let cost_C := num_C * price_C * (1 - discount_C)
  cost_A + cost_B + cost_C = 739.50 := sorry

end stamp_total_cost_l1562_156269


namespace average_cost_of_fruit_l1562_156246

variable (apples bananas oranges total_cost total_pieces avg_cost : ℕ)

theorem average_cost_of_fruit (h1 : apples = 12)
                              (h2 : bananas = 4)
                              (h3 : oranges = 4)
                              (h4 : total_cost = apples * 2 + bananas * 1 + oranges * 3)
                              (h5 : total_pieces = apples + bananas + oranges)
                              (h6 : avg_cost = total_cost / total_pieces) :
                              avg_cost = 2 :=
by sorry

end average_cost_of_fruit_l1562_156246


namespace henry_correct_answers_l1562_156212

theorem henry_correct_answers (c w : ℕ) (h1 : c + w = 15) (h2 : 6 * c - 3 * w = 45) : c = 10 :=
by
  sorry

end henry_correct_answers_l1562_156212


namespace walking_speed_l1562_156219

theorem walking_speed 
  (v : ℕ) -- v represents the man's walking speed in kmph
  (distance_formula : distance = speed * time)
  (distance_walking : distance = v * 9)
  (distance_running : distance = 24 * 3) : 
  v = 8 :=
by
  sorry

end walking_speed_l1562_156219


namespace new_man_weight_l1562_156264

theorem new_man_weight (avg_increase : ℝ) (crew_weight : ℝ) (new_man_weight : ℝ) 
(h_avg_increase : avg_increase = 1.8) (h_crew_weight : crew_weight = 53) :
  new_man_weight = crew_weight + 10 * avg_increase :=
by
  -- Here we will use the conditions to prove the theorem
  sorry

end new_man_weight_l1562_156264


namespace num_from_1_to_200_not_squares_or_cubes_l1562_156276

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l1562_156276


namespace inverse_five_eq_two_l1562_156239

-- Define the function f(x) = x^2 + 1 for x >= 0
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the condition x >= 0
def nonneg (x : ℝ) : Prop := x ≥ 0

-- State the problem: proving that the inverse function f⁻¹(5) = 2
theorem inverse_five_eq_two : ∃ x : ℝ, nonneg x ∧ f x = 5 ∧ x = 2 :=
by
  sorry

end inverse_five_eq_two_l1562_156239


namespace print_papers_in_time_l1562_156282

theorem print_papers_in_time :
  ∃ (n : ℕ), 35 * 15 * n = 500000 * 21 * n := by
  sorry

end print_papers_in_time_l1562_156282


namespace no_solutions_for_inequalities_l1562_156206

theorem no_solutions_for_inequalities (x y z t : ℝ) :
  |x| < |y - z + t| →
  |y| < |x - z + t| →
  |z| < |x - y + t| →
  |t| < |x - y + z| →
  False :=
by
  sorry

end no_solutions_for_inequalities_l1562_156206


namespace largest_square_side_length_largest_rectangle_dimensions_l1562_156218

variable (a b : ℝ) (h : a > 0) (k : b > 0)

-- Part (a): Side length of the largest possible square
theorem largest_square_side_length (h : a > 0) (k : b > 0) :
  ∃ (s : ℝ), s = (a * b) / (a + b) := sorry

-- Part (b): Dimensions of the largest possible rectangle
theorem largest_rectangle_dimensions (h : a > 0) (k : b > 0) :
  ∃ (x y : ℝ), x = a / 2 ∧ y = b / 2 := sorry

end largest_square_side_length_largest_rectangle_dimensions_l1562_156218


namespace at_least_one_real_root_l1562_156217

theorem at_least_one_real_root (a : ℝ) :
  (4*a)^2 - 4*(-4*a + 3) ≥ 0 ∨
  ((a - 1)^2 - 4*a^2) ≥ 0 ∨
  (2*a)^2 - 4*(-2*a) ≥ 0 := sorry

end at_least_one_real_root_l1562_156217


namespace ambulance_reachable_area_l1562_156227

theorem ambulance_reachable_area :
  let travel_time_minutes := 8
  let travel_time_hours := (travel_time_minutes : ℝ) / 60
  let speed_on_road := 60 -- speed in miles per hour
  let speed_off_road := 10 -- speed in miles per hour
  let distance_on_road := speed_on_road * travel_time_hours
  distance_on_road = 8 → -- this verifies the distance covered on road
  let area := (2 * distance_on_road) ^ 2
  area = 256 := sorry

end ambulance_reachable_area_l1562_156227


namespace reciprocal_opposite_abs_val_l1562_156266

theorem reciprocal_opposite_abs_val (a : ℚ) (h : a = -1 - 2/7) :
    (1 / a = -7/9) ∧ (-a = 1 + 2/7) ∧ (|a| = 1 + 2/7) := 
sorry

end reciprocal_opposite_abs_val_l1562_156266


namespace c_alone_finishes_job_in_7_5_days_l1562_156235

theorem c_alone_finishes_job_in_7_5_days (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : A + B + C = 1 / 5) :
  1 / C = 7.5 :=
by
  -- The proof is omitted
  sorry

end c_alone_finishes_job_in_7_5_days_l1562_156235
