import Mathlib

namespace waiter_customers_l277_277541

-- Define initial conditions
def initial_customers : ℕ := 47
def customers_left : ℕ := 41
def new_customers : ℕ := 20

-- Calculate remaining customers after some left
def remaining_customers : ℕ := initial_customers - customers_left

-- Calculate the total customers after getting new ones
def total_customers : ℕ := remaining_customers + new_customers

-- State the theorem to prove the final total customers
theorem waiter_customers : total_customers = 26 := by
  -- We include sorry for the proof placeholder
  sorry

end waiter_customers_l277_277541


namespace order_of_numbers_l277_277362

theorem order_of_numbers (a b c : ℝ) (h1 : a = 6^0.5) (h2 : b = 0.5^6) (h3 : c = Real.log 6 / Real.log 0.5) : 
  c < b ∧ b < a :=
by {
  have h4 : a > 1, from sorry,
  have h5 : 0 < b ∧ b < 1, from sorry,
  have h6 : c < 0, from sorry,
  exact ⟨h6, h5.2.trans h4⟩,
}

end order_of_numbers_l277_277362


namespace speed_of_stream_l277_277402

-- Conditions
variables (v : ℝ) -- speed of the stream in kmph
variables (boat_speed_still_water : ℝ := 10) -- man's speed in still water in kmph
variables (distance : ℝ := 90) -- distance traveled down the stream in km
variables (time : ℝ := 5) -- time taken to travel the distance down the stream in hours

-- Proof statement
theorem speed_of_stream : v = 8 :=
  by
    -- effective speed down the stream = boat_speed_still_water + v
    -- given that distance = speed * time
    -- 90 = (10 + v) * 5
    -- solving for v
    sorry

end speed_of_stream_l277_277402


namespace determine_m_l277_277700

theorem determine_m (x m : ℝ) (h₁ : 2 * x + m = 6) (h₂ : x = 2) : m = 2 := by
  sorry

end determine_m_l277_277700


namespace red_balloon_probability_l277_277197

-- Define the conditions
def initial_red_balloons := 2
def initial_blue_balloons := 4
def inflated_red_balloons := 2
def inflated_blue_balloons := 2

-- Define the total number of balloons after inflation
def total_red_balloons := initial_red_balloons + inflated_red_balloons
def total_blue_balloons := initial_blue_balloons + inflated_blue_balloons
def total_balloons := total_red_balloons + total_blue_balloons

-- Define the probability calculation
def red_probability := (total_red_balloons : ℚ) / total_balloons * 100

-- The theorem to prove
theorem red_balloon_probability : red_probability = 40 := by
  sorry -- Skipping the proof itself

end red_balloon_probability_l277_277197


namespace karthik_weight_average_l277_277180

noncomputable def average_probable_weight_of_karthik (weight : ℝ) : Prop :=
  (55 < weight ∧ weight < 62) ∧
  (50 < weight ∧ weight < 60) ∧
  (weight ≤ 58) →
  weight = 56.5

theorem karthik_weight_average :
  ∀ weight : ℝ, average_probable_weight_of_karthik weight :=
by
  sorry

end karthik_weight_average_l277_277180


namespace sum_of_all_four_digit_numbers_l277_277289

def digits : List ℕ := [1, 2, 3, 4, 5]

noncomputable def four_digit_numbers := 
  (Digits.permutations digits).filter (λ l => l.length = 4)

noncomputable def sum_of_numbers (nums : List (List ℕ)) : ℕ :=
  nums.foldl (λ acc num => acc + (num.foldl (λ acc' digit => acc' * 10 + digit) 0)) 0

theorem sum_of_all_four_digit_numbers :
  sum_of_numbers four_digit_numbers = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_l277_277289


namespace collinear_points_XYZ_l277_277526

open EuclideanGeometry

theorem collinear_points_XYZ
  (O A B C Z X Y : Point)
  (hOA : Collinear O A)
  (hOB : Collinear O B)
  (hOC : Collinear O C)
  (hZ : ∃ (circle1 circle2 : Circle), circle1.Diameter = lineSegment O A ∧ circle2.Diameter = lineSegment O B ∧ circle1 ∩ circle2 = {Z})
  (hX : ∃ (circle3 circle4 : Circle), circle3.Diameter = lineSegment O B ∧ circle4.Diameter = lineSegment O C ∧ circle3 ∩ circle4 = {X})
  (hY : ∃ (circle5 circle6 : Circle), circle5.Diameter = lineSegment O C ∧ circle6.Diameter = lineSegment O A ∧ circle5 ∩ circle6 = {Y}) :
  Collinear X Y Z :=
sorry

end collinear_points_XYZ_l277_277526


namespace boxes_containing_neither_l277_277036

theorem boxes_containing_neither
  (total_boxes : ℕ)
  (boxes_with_stickers : ℕ)
  (boxes_with_cards : ℕ)
  (boxes_with_both : ℕ)
  (h1 : total_boxes = 15)
  (h2 : boxes_with_stickers = 8)
  (h3 : boxes_with_cards = 5)
  (h4 : boxes_with_both = 3) :
  (total_boxes - (boxes_with_stickers + boxes_with_cards - boxes_with_both)) = 5 :=
by
  sorry

end boxes_containing_neither_l277_277036


namespace inequality_example_l277_277052

open Real

theorem inequality_example 
    (x y z : ℝ) 
    (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1):
    (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := 
by 
  sorry

end inequality_example_l277_277052


namespace arithmetic_geometric_progression_l277_277934

theorem arithmetic_geometric_progression (a d : ℝ)
    (h1 : 2 * (a - d) * a * (a + d + 7) = 1000)
    (h2 : a^2 = 2 * (a - d) * (a + d + 7)) :
    d = 8 ∨ d = -8 := 
    sorry

end arithmetic_geometric_progression_l277_277934


namespace solve_inequality_l277_277565

theorem solve_inequality (x : ℝ) (h1: 3 * x - 8 ≠ 0) :
  5 ≤ x / (3 * x - 8) ∧ x / (3 * x - 8) < 10 ↔ (8 / 3) < x ∧ x ≤ (20 / 7) := 
sorry

end solve_inequality_l277_277565


namespace beads_per_package_eq_40_l277_277304

theorem beads_per_package_eq_40 (b r : ℕ) (x : ℕ) (total_beads : ℕ) 
(h1 : b = 3) (h2 : r = 5) (h3 : total_beads = 320) (h4 : total_beads = (b + r) * x) :
  x = 40 := by
  sorry

end beads_per_package_eq_40_l277_277304


namespace greatest_integer_x_l277_277954

theorem greatest_integer_x :
  ∃ (x : ℤ), (∀ (y : ℤ), (8 : ℝ) / 11 > (x : ℝ) / 15) ∧
    ¬ (8 / 11 > (x + 1 : ℝ) / 15) ∧
    x = 10 :=
by
  sorry

end greatest_integer_x_l277_277954


namespace trip_cost_is_correct_l277_277326

-- Given conditions
def bills_cost : ℕ := 3500
def save_per_month : ℕ := 500
def savings_duration_months : ℕ := 2 * 12
def savings : ℕ := save_per_month * savings_duration_months
def remaining_after_bills : ℕ := 8500

-- Prove that the cost of the trip to Paris is 3500 dollars
theorem trip_cost_is_correct : (savings - remaining_after_bills) = bills_cost :=
sorry

end trip_cost_is_correct_l277_277326


namespace problem_statement_l277_277251

open Real

theorem problem_statement (t : ℝ) :
  cos (2 * t) ≠ 0 ∧ sin (2 * t) ≠ 0 →
  cos⁻¹ (2 * t) + sin⁻¹ (2 * t) + cos⁻¹ (2 * t) * sin⁻¹ (2 * t) = 5 →
  (∃ k : ℤ, t = arctan (1/2) + π * k) ∨ (∃ n : ℤ, t = arctan (1/3) + π * n) :=
by
  sorry

end problem_statement_l277_277251


namespace hyperbola_asymptote_a_value_l277_277968

theorem hyperbola_asymptote_a_value (a : ℝ) (h : 0 < a) 
  (asymptote_eq : y = (3 / 5) * x) :
  (x^2 / a^2 - y^2 / 9 = 1) → a = 5 :=
by
  sorry

end hyperbola_asymptote_a_value_l277_277968


namespace find_k_and_direction_l277_277165

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ := (0, 1)
def c (k : ℝ) : ℝ × ℝ := (k * a.1 + b.1, k * a.2 + b.2)
def d : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def parallel (u v : ℝ × ℝ) : Prop := ∃ λ : ℝ, u = (λ * v.1, λ * v.2)

theorem find_k_and_direction (k : ℝ) (h : parallel (c k) d) : k = -1 ∧ ∃ λ : ℝ, λ < 0 ∧ c k = (λ * d.1, λ * d.2) :=
by 
    sorry

end find_k_and_direction_l277_277165


namespace author_hardcover_percentage_l277_277983

variable {TotalPaperCopies : Nat}
variable {PricePerPaperCopy : ℝ}
variable {TotalHardcoverCopies : Nat}
variable {PricePerHardcoverCopy : ℝ}
variable {PaperPercentage : ℝ}
variable {TotalEarnings : ℝ}

theorem author_hardcover_percentage (TotalPaperCopies : Nat)
  (PricePerPaperCopy : ℝ) (TotalHardcoverCopies : Nat)
  (PricePerHardcoverCopy : ℝ) (PaperPercentage TotalEarnings : ℝ)
  (h1 : TotalPaperCopies = 32000) (h2 : PricePerPaperCopy = 0.20)
  (h3 : TotalHardcoverCopies = 15000) (h4 : PricePerHardcoverCopy = 0.40)
  (h5 : PaperPercentage = 0.06) (h6 : TotalEarnings = 1104) :
  (720 / (15000 * 0.40) * 100) = 12 := by
  sorry

end author_hardcover_percentage_l277_277983


namespace find_pairs_l277_277840

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end find_pairs_l277_277840


namespace negation_of_exists_l277_277229

theorem negation_of_exists (x : ℕ) : (¬ ∃ x : ℕ, x^2 ≤ x) := 
by 
  sorry

end negation_of_exists_l277_277229


namespace condition_on_y_existence_of_r_l277_277297

-- Define the necessary conditions
variables {a b p : ℕ} (prime_p : Nat.Prime p) (coprime_abp : Nat.gcd a b = 1 ∧ Nat.gcd a p = 1 ∧ Nat.gcd b p = 1)
variables (n : ℕ)

-- T is defined as { x | x = a + n * b, n ∈ {0, 1, ..., p-1} }
def T := {x | ∃ n : ℕ, n < p ∧ x = a + n * b}

-- Proposition 1
theorem condition_on_y (y : ℕ) (ht : ∀ (i j ∈ T), i ≠ j → y ^ i + i % p ≠ y ^ j + j % p) :
  p ∣ y ∨ y ^ b ≡ 1 [MOD p] :=
sorry

-- Proposition 2
theorem existence_of_r (y t : ℕ) :
  ∃ r ∈ {r | ∃ n : ℕ, r = a + n * b}, (y ^ r + r) % p = t % p :=
sorry

end condition_on_y_existence_of_r_l277_277297


namespace smallest_prime_perf_sqr_minus_eight_l277_277772

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_perf_sqr_minus_eight (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 8

theorem smallest_prime_perf_sqr_minus_eight :
  ∃ (n : ℕ), is_prime n ∧ is_perf_sqr_minus_eight n ∧ (∀ m : ℕ, is_prime m ∧ is_perf_sqr_minus_eight m → n ≤ m) :=
sorry

end smallest_prime_perf_sqr_minus_eight_l277_277772


namespace at_least_three_equal_l277_277701

theorem at_least_three_equal (a b c d : ℕ) (h1 : (a + b) ^ 2 ∣ c * d)
                                (h2 : (a + c) ^ 2 ∣ b * d)
                                (h3 : (a + d) ^ 2 ∣ b * c)
                                (h4 : (b + c) ^ 2 ∣ a * d)
                                (h5 : (b + d) ^ 2 ∣ a * c)
                                (h6 : (c + d) ^ 2 ∣ a * b) :
  ∃ x : ℕ, (x = a ∧ x = b ∧ x = c) ∨ (x = a ∧ x = b ∧ x = d) ∨ (x = a ∧ x = c ∧ x = d) ∨ (x = b ∧ x = c ∧ x = d) :=
sorry

end at_least_three_equal_l277_277701


namespace stock_value_sale_l277_277252

theorem stock_value_sale
  (X : ℝ)
  (h1 : 0.20 * X * 0.10 - 0.80 * X * 0.05 = -350) :
  X = 17500 := by
  -- Proof goes here
  sorry

end stock_value_sale_l277_277252


namespace tiling_remainder_is_888_l277_277109

noncomputable def boardTilingWithThreeColors (n : ℕ) : ℕ :=
  if n = 8 then
    4 * (21 * (3^3 - 3*2^3 + 3) +
         35 * (3^4 - 4*2^4 + 6) +
         35 * (3^5 - 5*2^5 + 10) +
         21 * (3^6 - 6*2^6 + 15) +
         7 * (3^7 - 7*2^7 + 21) +
         1 * (3^8 - 8*2^8 + 28))
  else
    0

theorem tiling_remainder_is_888 :
  boardTilingWithThreeColors 8 % 1000 = 888 :=
by
  sorry

end tiling_remainder_is_888_l277_277109


namespace positive_integer_in_base_proof_l277_277959

noncomputable def base_conversion_problem (A B : ℕ) (n : ℕ) : Prop :=
  n = 9 * A + B ∧ n = 8 * B + A ∧ A < 9 ∧ B < 8 ∧ A ≠ 0 ∧ B ≠ 0

theorem positive_integer_in_base_proof (A B n : ℕ) (h : base_conversion_problem A B n) : n = 0 :=
sorry

end positive_integer_in_base_proof_l277_277959


namespace remaining_dimes_l277_277889

-- Define the initial quantity of dimes Joan had
def initial_dimes : Nat := 5

-- Define the quantity of dimes Joan spent
def dimes_spent : Nat := 2

-- State the theorem we need to prove
theorem remaining_dimes : initial_dimes - dimes_spent = 3 := by
  sorry

end remaining_dimes_l277_277889


namespace percent_twelve_equals_eighty_four_l277_277242

theorem percent_twelve_equals_eighty_four (x : ℝ) (h : (12 / 100) * x = 84) : x = 700 :=
by
  sorry

end percent_twelve_equals_eighty_four_l277_277242


namespace area_of_triangle_PQR_l277_277707

noncomputable def point := ℝ × ℝ

def P : point := (1, 1)
def Q : point := (4, 1)
def R : point := (3, 4)

def triangle_area (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)

theorem area_of_triangle_PQR :
  triangle_area P Q R = 9 / 2 :=
by
  sorry

end area_of_triangle_PQR_l277_277707


namespace area_of_rectangle_ABCD_l277_277004

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l277_277004


namespace hotdogs_total_l277_277742

theorem hotdogs_total:
  let e := 2.5
  let l := 2 * (e * 2)
  let m := 7
  let h := 1.5 * (e * 2)
  let z := 0.5
  (e * 2 + l + m + h + z) = 30 := 
by
  sorry

end hotdogs_total_l277_277742


namespace find_prime_p_l277_277144

theorem find_prime_p :
  ∃ p : ℕ, Prime p ∧ (∃ a b : ℤ, p = 5 ∧ 1 < p ∧ p ≤ 11 ∧ (a^2 + p * a - 720 * p = 0) ∧ (b^2 - p * b + 720 * p = 0)) :=
sorry

end find_prime_p_l277_277144


namespace rectangle_area_l277_277011

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l277_277011


namespace relationship_xyz_l277_277704

theorem relationship_xyz (x y z : ℝ) (h1 : x = Real.log x) (h2 : y = Real.logb 5 2) (h3 : z = Real.exp (-0.5)) : x > z ∧ z > y :=
by
  sorry

end relationship_xyz_l277_277704


namespace total_red_and_green_peaches_l277_277763

-- Define the number of red peaches and green peaches.
def red_peaches : ℕ := 6
def green_peaches : ℕ := 16

-- Theorem stating the sum of red and green peaches is 22.
theorem total_red_and_green_peaches : red_peaches + green_peaches = 22 := 
by
  -- Proof would go here but is not required
  sorry

end total_red_and_green_peaches_l277_277763


namespace train_length_l277_277978

theorem train_length (x : ℕ)
  (h1 : ∀ (x : ℕ), (790 + x) / 33 = (860 - x) / 22) : x = 200 := by
  sorry

end train_length_l277_277978


namespace find_sum_of_a_and_b_l277_277172

theorem find_sum_of_a_and_b (a b : ℝ) (h1 : 0.005 * a = 0.65) (h2 : 0.0125 * b = 1.04) : a + b = 213.2 :=
  sorry

end find_sum_of_a_and_b_l277_277172


namespace total_spent_on_toys_l277_277469

-- Definitions for costs
def cost_car : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def cost_truck : ℝ := 5.86

-- The statement to prove
theorem total_spent_on_toys : cost_car + cost_skateboard + cost_truck = 25.62 := by
  sorry

end total_spent_on_toys_l277_277469


namespace complex_number_in_first_quadrant_l277_277919

theorem complex_number_in_first_quadrant :
  let z := (Complex.I / (1 + Complex.I)) in
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_number_in_first_quadrant_l277_277919


namespace Stan_pays_magician_l277_277750

theorem Stan_pays_magician :
  let hours_per_day := 3
  let days_per_week := 7
  let weeks := 2
  let hourly_rate := 60
  let total_hours := hours_per_day * days_per_week * weeks
  let total_payment := hourly_rate * total_hours
  total_payment = 2520 := 
by 
  sorry

end Stan_pays_magician_l277_277750


namespace legacy_total_earnings_l277_277392

def floors := 4
def rooms_per_floor := 10
def hours_per_room := 6
def hourly_rate := 15
def total_rooms := floors * rooms_per_floor
def total_hours := total_rooms * hours_per_room
def total_earnings := total_hours * hourly_rate

theorem legacy_total_earnings :
  total_earnings = 3600 :=
by
  -- Proof to be filled in
  sorry

end legacy_total_earnings_l277_277392


namespace sum_four_digit_numbers_l277_277287

def digits : List ℕ := [1, 2, 3, 4, 5]

/-- 
  Prove that the sum of all four-digit numbers that can be formed 
  using the digits 1, 2, 3, 4, 5 exactly once is 399960.
-/
theorem sum_four_digit_numbers : 
  (Finset.sum 
    (Finset.map 
      (λ l, 
        l.nth_le 0 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1000 + 
        l.nth_le 1 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 100 + 
        l.nth_le 2 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 10 + 
        l.nth_le 3 (by simp [l.length_eq_of_perm length, digits.length, dec_trivial]) * 1) 
      (digits.permutations.filter (λ l, l.nodup ∧ l.length = 4))) id) 
  = 399960 :=
sorry

end sum_four_digit_numbers_l277_277287


namespace find_p_l277_277594

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def hyperbola_focus : ℝ × ℝ :=
  (2, 0)

theorem find_p (p : ℝ) (h : p > 0) (hp : parabola_focus p = hyperbola_focus) : p = 4 :=
by
  sorry

end find_p_l277_277594


namespace whole_numbers_between_sqrts_l277_277721

theorem whole_numbers_between_sqrts :
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let start := Nat.ceil lower_bound
  let end_ := Nat.floor upper_bound
  ∃ n, n = end_ - start + 1 ∧ n = 7 := by
  sorry

end whole_numbers_between_sqrts_l277_277721


namespace smallest_prime_8_less_than_square_l277_277780

theorem smallest_prime_8_less_than_square :
  ∃ p : ℕ, (∃ n : ℤ, p = n^2 - 8) ∧ Nat.Prime p ∧ p > 0 ∧ (∀ q : ℕ, (∃ m : ℤ, q = m^2 - 8) ∧ Nat.Prime q → q ≥ p) :=
sorry

end smallest_prime_8_less_than_square_l277_277780


namespace Ali_possible_scores_l277_277409

-- Defining the conditions
def categories := 5
def questions_per_category := 3
def correct_answers_points := 12
def total_questions := categories * questions_per_category
def incorrect_answers := total_questions - correct_answers_points

-- Defining the bonuses based on cases

-- All 3 incorrect answers in 1 category
def case_1_bonus := 4
def case_1_total := correct_answers_points + case_1_bonus

-- 3 incorrect answers split into 2 categories
def case_2_bonus := 3
def case_2_total := correct_answers_points + case_2_bonus

-- 3 incorrect answers split into 3 categories
def case_3_bonus := 2
def case_3_total := correct_answers_points + case_3_bonus

theorem Ali_possible_scores : 
  case_1_total = 16 ∧ case_2_total = 15 ∧ case_3_total = 14 :=
by
  -- Skipping the proof here
  sorry

end Ali_possible_scores_l277_277409


namespace competition_results_l277_277332

-- Participants and positions
inductive Participant : Type
| Oleg
| Olya
| Polya
| Pasha

-- Places in the competition (1st, 2nd, 3rd, 4th)
def Place := Fin 4

-- Statements made by the children
def Olya_statement1 : Prop := ∀ p, p % 2 = 1 -> p = Participant.Oleg ∨ p = Participant.Pasha
def Oleg_statement1 : Prop := ∃ p1 p2: Place, p1 < p2 ∧ (p1 = p2 + 1)
def Pasha_statement1 : Prop := ∀ p, p % 2 = 1 -> (p = Place 1 ∨ p = Place 3)

-- Truthfulness of the statements
def only_one_truthful (Olya_true : Prop) (Oleg_true : Prop) (Pasha_true : Prop) :=
  (Olya_true ∧ ¬ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ Oleg_true ∧ ¬ Pasha_true) ∨
  (¬ Olya_true ∧ ¬ Oleg_true ∧ Pasha_true)

-- The actual positions
def positions : Participant → Place
| Participant.Oleg  := 0
| Participant.Olya  := 1
| Participant.Polya := 2
| Participant.Pasha := 3

-- The Lean statement to prove
theorem competition_results :
  ((Oleg_statement1 ↔ positions Participant.Oleg = 0) ∧ 
  (Olya_statement1 ↔ positions Participant.Olya = 1) ∧ 
  (Pasha_statement1 ↔ positions Participant.Pasha = 3)) ∧ 
  only_one_truthful (positions Participant.Oleg = 0) 
                    (positions Participant.Olya = 0) 
                    (positions Participant.Pasha = 0) ∧
  positions Participant.Oleg = 0 ∧ 
  positions Participant.Olya = 1 ∧
  positions Participant.Polya = 2 ∧
  positions Participant.Pasha = 3 := 
sorry

end competition_results_l277_277332


namespace vanessa_deleted_30_files_l277_277649

-- Define the initial conditions
def original_files : Nat := 16 + 48
def files_left : Nat := 34

-- Define the number of files deleted
def files_deleted : Nat := original_files - files_left

-- The theorem to prove the number of files deleted
theorem vanessa_deleted_30_files : files_deleted = 30 := by
  sorry

end vanessa_deleted_30_files_l277_277649


namespace rectangular_field_area_eq_l277_277887

-- Definitions based on the problem's conditions
def length (x : ℝ) := x
def width (x : ℝ) := 60 - x
def area (x : ℝ) := x * (60 - x)

-- The proof statement
theorem rectangular_field_area_eq (x : ℝ) (h₀ : x + (60 - x) = 60) (h₁ : area x = 864) :
  x * (60 - x) = 864 :=
by
  -- Using the provided conditions and definitions, we aim to prove the equation.
  sorry

end rectangular_field_area_eq_l277_277887


namespace trapezoid_area_l277_277996

theorem trapezoid_area (a b d1 d2 : ℝ) (ha : 0 < a) (hb : 0 < b) (hd1 : 0 < d1) (hd2 : 0 < d2)
  (hbase : a = 11) (hbase2 : b = 4) (hdiagonal1 : d1 = 9) (hdiagonal2 : d2 = 12) :
  (∃ area : ℝ, area = 54) :=
by
  sorry

end trapezoid_area_l277_277996


namespace min_red_hair_students_l277_277666

theorem min_red_hair_students (B N R : ℕ) 
  (h1 : B + N + R = 50)
  (h2 : N ≥ B - 1)
  (h3 : R ≥ N - 1) :
  R = 17 := sorry

end min_red_hair_students_l277_277666


namespace rectangle_area_l277_277022

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l277_277022


namespace scenario_1_scenario_2_scenario_3_scenario_4_l277_277648

-- Definitions based on conditions
def prob_A_hit : ℚ := 2 / 3
def prob_B_hit : ℚ := 3 / 4

-- Scenario 1: Prove that the probability of A shooting 3 times and missing at least once is 19/27
theorem scenario_1 : 
  (1 - (prob_A_hit ^ 3)) = 19 / 27 :=
by sorry

-- Scenario 2: Prove that the probability of A hitting the target exactly 2 times and B hitting the target exactly 1 time after each shooting twice is 1/6
theorem scenario_2 : 
  (2 * ((prob_A_hit ^ 2) * (1 - prob_A_hit)) * (2 * (prob_B_hit * (1 - prob_B_hit)))) = 1 / 6 :=
by sorry

-- Scenario 3: Prove that the probability of A missing the target and B hitting the target 2 times after each shooting twice is 1/16
theorem scenario_3 :
  ((1 - prob_A_hit) ^ 2) * (prob_B_hit ^ 2) = 1 / 16 :=
by sorry

-- Scenario 4: Prove that the probability that both A and B hit the target once after each shooting twice is 1/6
theorem scenario_4 : 
  (2 * (prob_A_hit * (1 - prob_A_hit)) * 2 * (prob_B_hit * (1 - prob_B_hit))) = 1 / 6 :=
by sorry

end scenario_1_scenario_2_scenario_3_scenario_4_l277_277648


namespace option_c_correct_l277_277523

theorem option_c_correct (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by
  sorry

end option_c_correct_l277_277523


namespace positive_difference_is_30_l277_277414

-- Define the absolute value equation condition
def abs_condition (x : ℝ) : Prop := abs (x - 3) = 15

-- Define the solutions to the absolute value equation
def solution1 : ℝ := 18
def solution2 : ℝ := -12

-- Define the positive difference of the solutions
def positive_difference : ℝ := abs (solution1 - solution2)

-- Theorem statement: the positive difference is 30
theorem positive_difference_is_30 : positive_difference = 30 :=
by
  sorry

end positive_difference_is_30_l277_277414


namespace area_of_rectangle_l277_277025

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l277_277025


namespace adam_final_score_l277_277423

theorem adam_final_score : 
  let science_correct := 5
  let science_points := 10
  let history_correct := 3
  let history_points := 5
  let history_multiplier := 2
  let sports_correct := 1
  let sports_points := 15
  let literature_correct := 1
  let literature_points := 7
  let literature_penalty := 3
  
  let science_total := science_correct * science_points
  let history_total := (history_correct * history_points) * history_multiplier
  let sports_total := sports_correct * sports_points
  let literature_total := (literature_correct * literature_points) - literature_penalty
  
  let final_score := science_total + history_total + sports_total + literature_total
  final_score = 99 := by 
    sorry

end adam_final_score_l277_277423


namespace rectangle_area_l277_277015

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l277_277015


namespace total_toys_l277_277683

theorem total_toys (toys_kamari : ℕ) (toys_anais : ℕ) (h1 : toys_kamari = 65) (h2 : toys_anais = toys_kamari + 30) :
  toys_kamari + toys_anais = 160 :=
by 
  sorry

end total_toys_l277_277683


namespace correct_conclusions_l277_277446

open Set Int Rat Nat

theorem correct_conclusions :
  (¬ (∅ = {0})) ∧ 
  (∀ a : ℤ, -a ∈ ℤ) ∧ 
  (Infinite (SetOf (λ y, ∃ x : ℚ, y = 4 * x))) ∧ 
  (\#(Subsets {x | -1 < x ∧ x < 3 ∧ x ∈ ℕ}) = 8) →
  True := 
by {
  intro h,
  sorry,
}

end correct_conclusions_l277_277446


namespace perfect_square_solutions_l277_277208

theorem perfect_square_solutions (a b : ℕ) (ha : a > b) (ha_pos : 0 < a) (hb_pos : 0 < b) (hA : ∃ k : ℕ, a^2 + 4 * b + 1 = k^2) (hB : ∃ l : ℕ, b^2 + 4 * a + 1 = l^2) :
  a = 8 ∧ b = 4 ∧ (a^2 + 4 * b + 1 = (a+1)^2) ∧ (b^2 + 4 * a + 1 = (b + 3)^2) :=
by
  sorry

end perfect_square_solutions_l277_277208


namespace age_contradiction_l277_277625

-- Given the age ratios and future age of Sandy
def current_ages (x : ℕ) : ℕ × ℕ × ℕ := (4 * x, 3 * x, 5 * x)
def sandy_age_after_6_years (age_sandy_current : ℕ) : ℕ := age_sandy_current + 6

-- Given conditions
def ratio_condition (x : ℕ) (age_sandy age_molly age_danny : ℕ) : Prop :=
  current_ages x = (age_sandy, age_molly, age_danny)

def sandy_age_condition (age_sandy_current : ℕ) : Prop :=
  sandy_age_after_6_years age_sandy_current = 30

def age_sum_condition (age_molly age_danny : ℕ) : Prop :=
  age_molly + age_danny = (age_molly + 4) + (age_danny + 4)

-- Main theorem
theorem age_contradiction : ∃ x age_sandy age_molly age_danny, 
  ratio_condition x age_sandy age_molly age_danny ∧
  sandy_age_condition age_sandy ∧
  (¬ age_sum_condition age_molly age_danny) := 
by
  -- Omitting the proof; the focus is on setting up the statement only
  sorry

end age_contradiction_l277_277625


namespace fraction_of_raisins_l277_277253

-- Define the cost of a single pound of raisins
variables (R : ℝ) -- R represents the cost of one pound of raisins

-- Conditions
def mixed_raisins := 5 -- Chris mixed 5 pounds of raisins
def mixed_nuts := 4 -- with 4 pounds of nuts
def nuts_cost_ratio := 3 -- A pound of nuts costs 3 times as much as a pound of raisins

-- Statement to prove
theorem fraction_of_raisins
  (R_pos : R > 0) : (5 * R) / ((5 * R) + (4 * (3 * R))) = 5 / 17 :=
by
  -- The proof is omitted here.
  sorry

end fraction_of_raisins_l277_277253


namespace marbles_solution_l277_277670

def marbles_problem : Prop :=
  let total_marbles := 20
  let blue_marbles := 6
  let red_marbles := 9
  let total_prob_red_white := 0.7
  let white_marbles := 5
  total_marbles = blue_marbles + red_marbles + white_marbles ∧
  (white_marbles / total_marbles + red_marbles / total_marbles = total_prob_red_white)

theorem marbles_solution : marbles_problem :=
by {
  sorry
}

end marbles_solution_l277_277670


namespace evaluate_expression_l277_277284

theorem evaluate_expression (A B : ℝ) (hA : A = 2^7) (hB : B = 3^6) : (A ^ (1 / 3)) * (B ^ (1 / 2)) = 108 * 2 ^ (1 / 3) :=
by
  sorry

end evaluate_expression_l277_277284


namespace legacy_total_earnings_l277_277391

def floors := 4
def rooms_per_floor := 10
def hours_per_room := 6
def hourly_rate := 15
def total_rooms := floors * rooms_per_floor
def total_hours := total_rooms * hours_per_room
def total_earnings := total_hours * hourly_rate

theorem legacy_total_earnings :
  total_earnings = 3600 :=
by
  -- Proof to be filled in
  sorry

end legacy_total_earnings_l277_277391


namespace subset_proper_l277_277715

def M : Set ℝ := {x | x^2 - x ≤ 0}

def N : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem subset_proper : N ⊂ M := by
  sorry

end subset_proper_l277_277715


namespace kaleb_books_l277_277386

-- Define the initial number of books
def initial_books : ℕ := 34

-- Define the number of books sold
def books_sold : ℕ := 17

-- Define the number of books bought
def books_bought : ℕ := 7

-- Prove that the final number of books is 24
theorem kaleb_books (h : initial_books - books_sold + books_bought = 24) : initial_books - books_sold + books_bought = 24 :=
by
  exact h

end kaleb_books_l277_277386


namespace part1_l277_277969

noncomputable def a : ℕ → ℝ
| 0       := some initial value -- initial value must be specified for a_0
| (n + 1) := 1 / (2 - a n)

theorem part1 (h : ∀ n : ℕ, n > 0 → (2 - a n) * a (n + 1) = 1) : 
  ∃ l, (filter.at_top.map a).tendsto l ∧ l = 1 :=
sorry

end part1_l277_277969


namespace exists_root_in_interval_l277_277276

open Real

theorem exists_root_in_interval : ∃ x, 1.1 < x ∧ x < 1.2 ∧ (x^2 + 12*x - 15 = 0) :=
by {
  let f := λ x : ℝ, x^2 + 12*x - 15,
  have h1 : f 1.1 = -0.59 :=  sorry,
  have h2 : f 1.2 = 0.84 := sorry,
  have sign_change : (f 1.1) * (f 1.2) < 0,
  { rw [h1, h2], linarith, },
  exact exists_has_deriv_at_eq_zero (by norm_num1) (by norm_num1) (by linarith)
}

end exists_root_in_interval_l277_277276


namespace total_hours_correct_l277_277048

def hours_watching_tv_per_day : ℕ := 4
def days_per_week : ℕ := 7
def days_playing_video_games_per_week : ℕ := 3

def tv_hours_per_week : ℕ := hours_watching_tv_per_day * days_per_week
def video_game_hours_per_day : ℕ := hours_watching_tv_per_day / 2
def video_game_hours_per_week : ℕ := video_game_hours_per_day * days_playing_video_games_per_week

def total_hours_per_week : ℕ := tv_hours_per_week + video_game_hours_per_week

theorem total_hours_correct :
  total_hours_per_week = 34 := by
  sorry

end total_hours_correct_l277_277048


namespace factorize_problem1_factorize_problem2_factorize_problem3_factorize_problem4_l277_277693

-- Problem 1: Prove equivalence for factorizing -2a^2 + 4a.
theorem factorize_problem1 (a : ℝ) : -2 * a^2 + 4 * a = -2 * a * (a - 2) := 
by sorry

-- Problem 2: Prove equivalence for factorizing 4x^3 y - 9xy^3.
theorem factorize_problem2 (x y : ℝ) : 4 * x^3 * y - 9 * x * y^3 = x * y * (2 * x + 3 * y) * (2 * x - 3 * y) := 
by sorry

-- Problem 3: Prove equivalence for factorizing 4x^2 - 12x + 9.
theorem factorize_problem3 (x : ℝ) : 4 * x^2 - 12 * x + 9 = (2 * x - 3)^2 := 
by sorry

-- Problem 4: Prove equivalence for factorizing (a+b)^2 - 6(a+b) + 9.
theorem factorize_problem4 (a b : ℝ) : (a + b)^2 - 6 * (a + b) + 9 = (a + b - 3)^2 := 
by sorry

end factorize_problem1_factorize_problem2_factorize_problem3_factorize_problem4_l277_277693


namespace first_marvelous_monday_after_school_starts_l277_277885

def is_marvelous_monday (year : ℕ) (month : ℕ) (day : ℕ) (start_day : ℕ) : Prop :=
  let days_in_month := if month = 9 then 30 else if month = 10 then 31 else 0
  let fifth_monday := start_day + 28
  let is_monday := (fifth_monday - 1) % 7 = 0
  month = 10 ∧ day = 30 ∧ is_monday

theorem first_marvelous_monday_after_school_starts :
  ∃ (year month day : ℕ),
    year = 2023 ∧ month = 10 ∧ day = 30 ∧ is_marvelous_monday year month day 4 := sorry

end first_marvelous_monday_after_school_starts_l277_277885


namespace triangular_weight_60_l277_277925

def round_weight := ℝ
def triangular_weight := ℝ
def rectangular_weight := 90

variables (c t : ℝ)

-- Conditions
axiom condition1 : c + t = 3 * c
axiom condition2 : 4 * c + t = t + c + rectangular_weight

theorem triangular_weight_60 : t = 60 :=
  sorry

end triangular_weight_60_l277_277925


namespace remaining_nap_time_is_three_hours_l277_277271

-- Define the flight time and the times spent on various activities
def flight_time_minutes := 11 * 60 + 20
def reading_time_minutes := 2 * 60
def movie_time_minutes := 4 * 60
def dinner_time_minutes := 30
def radio_time_minutes := 40
def game_time_minutes := 60 + 10

-- Calculate the total time spent on activities
def total_activity_time_minutes :=
  reading_time_minutes + movie_time_minutes + dinner_time_minutes + radio_time_minutes + game_time_minutes

-- Calculate the remaining time for a nap
def remaining_nap_time_minutes :=
  flight_time_minutes - total_activity_time_minutes

-- Convert the remaining nap time to hours
def remaining_nap_time_hours :=
  remaining_nap_time_minutes / 60

-- The statement to be proved
theorem remaining_nap_time_is_three_hours :
  remaining_nap_time_hours = 3 := by
  sorry

#check remaining_nap_time_is_three_hours -- This will check if the theorem statement is correct

end remaining_nap_time_is_three_hours_l277_277271


namespace focus_parabola_l277_277226

theorem focus_parabola (x : ℝ) (y : ℝ): (y = 8 * x^2) → (0, 1 / 32) = (0, 1 / 32) :=
by
  intro h
  sorry

end focus_parabola_l277_277226


namespace similar_sizes_combination_possible_l277_277323

theorem similar_sizes_combination_possible 
    (similar : Nat → Nat → Prop := λ x y, x ≤ y ∧ y ≤ 2 * x)
    (combine_piles : List Nat → Nat ∃ combined : Nat, (∀ x y ∈ combined, similar x y) → True
    (piles : List Nat) : True :=
sorry

end similar_sizes_combination_possible_l277_277323


namespace extra_men_needed_approx_is_60_l277_277964

noncomputable def extra_men_needed : ℝ :=
  let total_distance := 15.0   -- km
  let total_days := 300.0      -- days
  let initial_workforce := 40.0 -- men
  let completed_distance := 2.5 -- km
  let elapsed_days := 100.0    -- days

  let remaining_distance := total_distance - completed_distance -- km
  let remaining_days := total_days - elapsed_days               -- days

  let current_rate := completed_distance / elapsed_days -- km/day
  let required_rate := remaining_distance / remaining_days -- km/day

  let required_factor := required_rate / current_rate
  let new_workforce := initial_workforce * required_factor
  let extra_men := new_workforce - initial_workforce

  extra_men

theorem extra_men_needed_approx_is_60 :
  abs (extra_men_needed - 60) < 1 :=
sorry

end extra_men_needed_approx_is_60_l277_277964


namespace hypotenuse_length_l277_277435

theorem hypotenuse_length {a b c : ℝ} (h1 : a = 3) (h2 : b = 4) (h3 : c ^ 2 = a ^ 2 + b ^ 2) : c = 5 :=
by
  sorry

end hypotenuse_length_l277_277435


namespace three_buses_interval_l277_277941

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l277_277941


namespace inequality_condition_l277_277317

theorem inequality_condition
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 2015) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / Real.sqrt 2015 :=
by
  sorry

end inequality_condition_l277_277317


namespace smallest_whole_number_for_inequality_l277_277652

theorem smallest_whole_number_for_inequality:
  ∃ (x : ℕ), (2 : ℝ) / 5 + (x : ℝ) / 9 > 1 ∧ ∀ (y : ℕ), (2 : ℝ) / 5 + (y : ℝ) / 9 > 1 → x ≤ y :=
by
  sorry

end smallest_whole_number_for_inequality_l277_277652


namespace pirate_coins_total_l277_277901

theorem pirate_coins_total (x : ℕ) (hx : x ≠ 0) (h_paul : ∃ k : ℕ, k = x / 2) (h_pete : ∃ m : ℕ, m = 5 * (x / 2)) 
  (h_ratio : (m : ℝ) = (k : ℝ) * 5) : (x = 4) → 
  ∃ total : ℕ, total = k + m ∧ total = 12 :=
by {
  sorry
}

end pirate_coins_total_l277_277901


namespace equilateral_triangle_perimeter_l277_277029

-- Definitions based on conditions
def equilateral_triangle_side : ℕ := 8

-- The statement we need to prove
theorem equilateral_triangle_perimeter : 3 * equilateral_triangle_side = 24 := by
  sorry

end equilateral_triangle_perimeter_l277_277029


namespace verify_placements_l277_277331

-- Definitions for participants and their possible places
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

-- Each participant should be mapped to a place (1, 2, 3, 4)
def Place : Participant → ℕ := λ p,
  match p with
  | Participant.Olya => 2
  | Participant.Oleg => 1
  | Participant.Polya => 3
  | Participant.Pasha => 4

-- Conditions based on the problem statement
def statement_Olya : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Polya % 2 = 1 ∧ Place Participant.Pasha % 2 = 1)

def statement_Oleg : Prop :=
  (abs (Place Participant.Oleg - Place Participant.Olya) = 1)

def statement_Pasha : Prop :=
  (Place Participant.Oleg % 2 = 1 ∧ Place Participant.Olya % 2 = 1 ∧ Place Participant.Polya % 2 = 1)

-- Only one child tells the truth and the others lie
def exactly_one_true (a b c : Prop) : Prop := (a ∨ b ∨ c) ∧ (a → ¬b ∧ ¬c) ∧ (b → ¬a ∧ ¬c) ∧ (c → ¬a ∧ ¬b)

-- The main theorem to be proven
theorem verify_placements :
  exactly_one_true (statement_Olya) (statement_Oleg) (statement_Pasha) ∧ 
  Place Participant.Olya = 2 ∧
  Place Participant.Oleg = 1 ∧
  Place Participant.Polya = 3 ∧
  Place Participant.Pasha = 4 :=
by
  sorry

end verify_placements_l277_277331


namespace solve_for_x_l277_277457

theorem solve_for_x (x : ℝ) (h : 1 - 2 * (1 / (1 + x)) = 1 / (1 + x)) : x = 2 := 
  sorry

end solve_for_x_l277_277457


namespace no_real_solution_to_system_l277_277221

theorem no_real_solution_to_system :
  ∀ (x y z : ℝ), (x + y - 2 - 4 * x * y = 0) ∧
                 (y + z - 2 - 4 * y * z = 0) ∧
                 (z + x - 2 - 4 * z * x = 0) → false := 
by 
    intros x y z h
    rcases h with ⟨h1, h2, h3⟩
    -- Here would be the proof steps, which are omitted.
    sorry

end no_real_solution_to_system_l277_277221


namespace equation_has_three_distinct_solutions_iff_l277_277853

theorem equation_has_three_distinct_solutions_iff (a : ℝ) : 
  (∃ x_1 x_2 x_3 : ℝ, x_1 ≠ x_2 ∧ x_2 ≠ x_3 ∧ x_1 ≠ x_3 ∧ 
    (x_1 * |x_1 - a| = 1) ∧ (x_2 * |x_2 - a| = 1) ∧ (x_3 * |x_3 - a| = 1)) ↔ a > 2 :=
by
  sorry


end equation_has_three_distinct_solutions_iff_l277_277853


namespace find_principal_amount_l277_277999

theorem find_principal_amount (A R T : ℝ) (P : ℝ) : 
  A = 1680 → R = 0.05 → T = 2.4 → 1.12 * P = 1680 → P = 1500 :=
by
  intros hA hR hT hEq
  sorry

end find_principal_amount_l277_277999


namespace uphill_distance_is_100_l277_277673

def speed_uphill := 30  -- km/hr
def speed_downhill := 60  -- km/hr
def distance_downhill := 50  -- km
def avg_speed := 36  -- km/hr

-- Let d be the distance traveled uphill
variable (d : ℕ)

-- total distance is d + 50 km
def total_distance := d + distance_downhill

-- total time is (time uphill) + (time downhill)
def total_time := (d / speed_uphill) + (distance_downhill / speed_downhill)

theorem uphill_distance_is_100 (d : ℕ) (h : avg_speed = total_distance / total_time) : d = 100 :=
by
  sorry  -- proof is omitted

end uphill_distance_is_100_l277_277673


namespace abc_inequality_l277_277318

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) :=
sorry

end abc_inequality_l277_277318


namespace probability_is_stable_frequency_l277_277461

/-- Definition of probability: the stable theoretical value reflecting the likelihood of event occurrence. -/
def probability (event : Type) : ℝ := sorry 

/-- Definition of frequency: the empirical count of how often an event occurs in repeated experiments. -/
def frequency (event : Type) (trials : ℕ) : ℝ := sorry 

/-- The statement that "probability is the stable value of frequency" is correct. -/
theorem probability_is_stable_frequency (event : Type) (trials : ℕ) :
  probability event = sorry ↔ true := 
by 
  -- This is where the proof would go, but is replaced with sorry for now. 
  sorry

end probability_is_stable_frequency_l277_277461


namespace depth_of_box_l277_277795

theorem depth_of_box (length width depth : ℕ) (side_length : ℕ)
  (h_length : length = 30)
  (h_width : width = 48)
  (h_side_length : Nat.gcd length width = side_length)
  (h_cubes : side_length ^ 3 = 216)
  (h_volume : 80 * (side_length ^ 3) = length * width * depth) :
  depth = 12 :=
by
  sorry

end depth_of_box_l277_277795


namespace giraffes_count_l277_277416

def numZebras : ℕ := 12

def numCamels : ℕ := numZebras / 2

def numMonkeys : ℕ := numCamels * 4

def numGiraffes : ℕ := numMonkeys - 22

theorem giraffes_count :
  numGiraffes = 2 :=
by 
  sorry

end giraffes_count_l277_277416


namespace solve_rational_equation_l277_277426

theorem solve_rational_equation (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 4/5) :
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 20*x - 40)/(5*x - 4) = -5 ↔ x = -3 :=
by 
  sorry

end solve_rational_equation_l277_277426


namespace area_of_ABCD_l277_277000

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l277_277000


namespace Andrena_more_than_Debelyn_l277_277556

-- Define initial dolls count for each person
def Debelyn_initial_dolls : ℕ := 20
def Christel_initial_dolls : ℕ := 24

-- Define dolls given by Debelyn and Christel
def Debelyn_gift_dolls : ℕ := 2
def Christel_gift_dolls : ℕ := 5

-- Define remaining dolls for Debelyn and Christel after giving dolls away
def Debelyn_final_dolls : ℕ := Debelyn_initial_dolls - Debelyn_gift_dolls
def Christel_final_dolls : ℕ := Christel_initial_dolls - Christel_gift_dolls

-- Define Andrena's dolls after transactions
def Andrena_dolls : ℕ := Christel_final_dolls + 2

-- Define the Lean statement for proving Andrena has 3 more dolls than Debelyn
theorem Andrena_more_than_Debelyn : Andrena_dolls = Debelyn_final_dolls + 3 := by
  -- Here you would prove the statement
  sorry

end Andrena_more_than_Debelyn_l277_277556


namespace base8_to_decimal_l277_277245

theorem base8_to_decimal (n : ℕ) (h : n = 54321) : 
  (5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0) = 22737 := 
by
  sorry

end base8_to_decimal_l277_277245


namespace Beto_can_determine_xy_l277_277200

theorem Beto_can_determine_xy (m n : ℤ) :
  (∃ k t : ℤ, 0 < t ∧ m = 2 * k + 1 ∧ n = 2 * t * (2 * k + 1)) ↔ 
  (∀ x y : ℝ, (∃ a b : ℝ, a ≠ b ∧ x = a ∧ y = b) →
    ∃ xy_val : ℝ, (x^m + y^m = xy_val) ∧ (x^n + y^n = xy_val)) := 
sorry

end Beto_can_determine_xy_l277_277200


namespace arithmetic_sequence_a2_l277_277860

theorem arithmetic_sequence_a2 (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a1_a3 : a 1 + a 3 = 2) : a 2 = 1 :=
sorry

end arithmetic_sequence_a2_l277_277860


namespace minimum_stool_height_l277_277681

def ceiling_height : ℤ := 280
def alice_height : ℤ := 150
def reach : ℤ := alice_height + 30
def light_bulb_height : ℤ := ceiling_height - 15

theorem minimum_stool_height : 
  ∃ h : ℤ, reach + h = light_bulb_height ∧ h = 85 :=
by
  sorry

end minimum_stool_height_l277_277681


namespace positive_area_triangles_5x5_grid_l277_277589

-- Define the range of the grid
def grid_points (n : Nat) : List (Nat × Nat) :=
  (List.range n).bind (λ x, (List.range n).map (λ y, (x + 1, y + 1)))

-- Define the condition for positive area of a triangle
def is_positive_area (p1 p2 p3 : Nat × Nat) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x2 - x1) * (y3 - y1) ≠ (x3 - x1) * (y2 - y1)

-- Count the number of triangles with positive area
def count_positive_area_triangles (n : Nat) : Nat :=
  let points := grid_points n
  points.choose 3 |>.count (λ t, match t with (p1, p2, p3) => is_positive_area p1 p2 p3)

-- Statement of the theorem
theorem positive_area_triangles_5x5_grid : count_positive_area_triangles 5 = 2170 :=
by
  sorry

end positive_area_triangles_5x5_grid_l277_277589


namespace coordinate_identification_l277_277864

noncomputable def x1 := (4 * Real.pi) / 5
noncomputable def y1 := -(Real.pi) / 5

noncomputable def x2 := (12 * Real.pi) / 5
noncomputable def y2 := -(3 * Real.pi) / 5

noncomputable def x3 := (4 * Real.pi) / 3
noncomputable def y3 := -(Real.pi) / 3

theorem coordinate_identification :
  (x1, y1) = (4 * Real.pi / 5, -(Real.pi) / 5) ∧
  (x2, y2) = (12 * Real.pi / 5, -(3 * Real.pi) / 5) ∧
  (x3, y3) = (4 * Real.pi / 3, -(Real.pi) / 3) :=
by
  -- proof goes here
  sorry

end coordinate_identification_l277_277864


namespace rectangle_perimeter_l277_277404

theorem rectangle_perimeter 
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ)
  (relatively_prime : Nat.gcd (a_4 + a_7 + a_9) (a_2 + a_8 + a_6) = 1)
  (h1 : a_1 + a_2 = a_4)
  (h2 : a_1 + a_4 = a_5)
  (h3 : a_4 + a_5 = a_7)
  (h4 : a_5 + a_7 = a_9)
  (h5 : a_2 + a_4 + a_7 = a_8)
  (h6 : a_2 + a_8 = a_6)
  (h7 : a_1 + a_5 + a_9 = a_3)
  (h8 : a_3 + a_6 = a_8 + a_7) :
  2 * ((a_4 + a_7 + a_9) + (a_2 + a_8 + a_6)) = 164 := 
sorry -- proof omitted

end rectangle_perimeter_l277_277404


namespace polygon_sides_exterior_angle_l277_277456

theorem polygon_sides_exterior_angle (n : ℕ) (h : 360 / 24 = n) : n = 15 := by
  sorry

end polygon_sides_exterior_angle_l277_277456


namespace math_problem_l277_277480

variable {x p q r : ℝ}

-- Conditions and Theorem
theorem math_problem (h1 : ∀ x, (x ≤ -5 ∨ 20 ≤ x ∧ x ≤ 30) ↔ (0 ≤ (x - p) * (x - q) / (x - r)))
  (h2 : p < q) : p + 2 * q + 3 * r = 65 := 
sorry

end math_problem_l277_277480


namespace find_f_neg3_l277_277793

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom sum_equation : f 1 + f 2 + f 3 + f 4 + f 5 = 6

theorem find_f_neg3 : f (-3) = 6 := by
  sorry

end find_f_neg3_l277_277793


namespace somu_present_age_l277_277663

theorem somu_present_age (S F : ℕ) (h1 : S = (1 / 3) * F)
    (h2 : S - 5 = (1 / 5) * (F - 5)) : S = 10 := by
  sorry

end somu_present_age_l277_277663


namespace probability_one_solve_l277_277623

variables {p1 p2 : ℝ}

theorem probability_one_solve (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 * (1 - p2) + p2 * (1 - p1)) := 
sorry

end probability_one_solve_l277_277623


namespace quadratic_roots_range_l277_277913

theorem quadratic_roots_range (m : ℝ) :
  (∃ p n : ℝ, p > 0 ∧ n < 0 ∧ 2 * p^2 + (m + 1) * p + m = 0 ∧ 2 * n^2 + (m + 1) * n + m = 0) →
  m < 0 :=
by
  sorry

end quadratic_roots_range_l277_277913


namespace interval_with_three_buses_l277_277939

theorem interval_with_three_buses (interval_two_buses : ℕ) (total_route_time : ℕ) (bus_count : ℕ) : 
  interval_two_buses = 21 → total_route_time = 2 * interval_two_buses → bus_count = 3 → 
  (total_route_time / bus_count) = 14 :=
by
  intros h1 h2 h3
  rw [h1, h3, ← h2]
  simp
  sorry

end interval_with_three_buses_l277_277939


namespace Christina_driving_time_l277_277418

theorem Christina_driving_time 
  (speed_Christina : ℕ) 
  (speed_friend : ℕ) 
  (total_distance : ℕ)
  (friend_driving_time : ℕ) 
  (distance_by_Christina : ℕ) 
  (time_driven_by_Christina : ℕ) 
  (total_driving_time : ℕ)
  (h1 : speed_Christina = 30)
  (h2 : speed_friend = 40) 
  (h3 : total_distance = 210)
  (h4 : friend_driving_time = 3)
  (h5 : speed_friend * friend_driving_time = 120)
  (h6 : total_distance - 120 = distance_by_Christina)
  (h7 : distance_by_Christina = 90)
  (h8 : distance_by_Christina / speed_Christina = 3)
  (h9 : time_driven_by_Christina = 3)
  (h10 : time_driven_by_Christina * 60 = 180) :
    total_driving_time = 180 := 
by
  sorry

end Christina_driving_time_l277_277418


namespace largest_possible_A_l277_277097

theorem largest_possible_A (A B : ℕ) (h1 : A = 5 * 2 + B) (h2 : B < 5) : A ≤ 14 :=
by
  have h3 : A ≤ 10 + 4 := sorry
  exact h3

end largest_possible_A_l277_277097


namespace test_scores_order_l277_277193

def kaleana_score : ℕ := 75

variable (M Q S : ℕ)

-- Assuming conditions from the problem
axiom h1 : Q = kaleana_score
axiom h2 : M < max Q S
axiom h3 : S > min Q M
axiom h4 : M ≠ Q ∧ Q ≠ S ∧ M ≠ S

-- Theorem statement
theorem test_scores_order (M Q S : ℕ) (h1 : Q = kaleana_score) (h2 : M < max Q S) (h3 : S > min Q M) (h4 : M ≠ Q ∧ Q ≠ S ∧ M ≠ S) :
  M < Q ∧ Q < S :=
sorry

end test_scores_order_l277_277193


namespace find_a_from_binomial_l277_277829

variable (x : ℝ) (a : ℝ)

def binomial_term (r : ℕ) : ℝ :=
  (Nat.choose 5 r) * ((-a)^r) * x^(5 - 2 * r)

theorem find_a_from_binomial :
  (∃ x : ℝ, ∃ a : ℝ, (binomial_term x a 1 = 10)) → a = -2 :=
by 
  sorry

end find_a_from_binomial_l277_277829


namespace cannot_form_complex_pattern_l277_277809

structure GeometricPieces where
  triangles : Nat
  squares : Nat

def possibleToForm (pieces : GeometricPieces) : Bool :=
  sorry -- Since the formation logic is unknown, it is incomplete.

theorem cannot_form_complex_pattern : 
  let pieces := GeometricPieces.mk 8 7
  ¬ possibleToForm pieces = true := 
sorry

end cannot_form_complex_pattern_l277_277809


namespace expression_equivalence_l277_277957

-- Define the initial expression
def expr (w : ℝ) : ℝ := 3 * w + 4 - 2 * w^2 - 5 * w - 6 + w^2 + 7 * w + 8 - 3 * w^2

-- Define the simplified expression
def simplified_expr (w : ℝ) : ℝ := 5 * w - 4 * w^2 + 6

-- Theorem stating the equivalence
theorem expression_equivalence (w : ℝ) : expr w = simplified_expr w :=
by
  -- we would normally simplify and prove here, but we state the theorem and skip the proof for now.
  sorry

end expression_equivalence_l277_277957


namespace c_amount_correct_b_share_correct_l277_277395

-- Conditions
def total_sum : ℝ := 246    -- Total sum of money
def c_share : ℝ := 48      -- C's share in Rs
def c_per_rs : ℝ := 0.40   -- C's amount per Rs

-- Expressing the given condition c_share = total sum * c_per_rs
theorem c_amount_correct : c_share = total_sum * c_per_rs := 
  by
  -- Substitute that can be more elaboration of the calculations done
  sorry

-- Additional condition for the total per Rs distribution
axiom a_b_c_total : ∀ (a b : ℝ), a + b + c_per_rs = 1

-- Proving B's share per Rs is approximately 0.4049
theorem b_share_correct : ∃ a b : ℝ, c_share = 246 * 0.40 ∧ a + b + 0.40 = 1 ∧ b = 1 - (48 / 246) - 0.40 := 
  by
  -- Substitute that can be elaboration of the proof arguments done in the translated form
  sorry

end c_amount_correct_b_share_correct_l277_277395


namespace A_days_l277_277970

theorem A_days (B_days : ℕ) (total_wage A_wage : ℕ) (h_B_days : B_days = 15) (h_total_wage : total_wage = 3000) (h_A_wage : A_wage = 1800) :
  ∃ A_days : ℕ, A_days = 10 := by
  sorry

end A_days_l277_277970


namespace bus_interval_three_buses_l277_277946

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l277_277946


namespace interest_rate_calc_l277_277661

theorem interest_rate_calc
  (P : ℝ) (A : ℝ) (T : ℝ) (SI : ℝ := A - P)
  (R : ℝ := (SI * 100) / (P * T))
  (hP : P = 750)
  (hA : A = 950)
  (hT : T = 5) :
  R = 5.33 :=
by
  sorry

end interest_rate_calc_l277_277661


namespace inequality_proof_l277_277489

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (habc : a * b * (1 / (a * b)) = 1) :
  a^2 + b^2 + (1 / (a * b))^2 + 3 ≥ 2 * (1 / a + 1 / b + a * b) := 
by sorry

end inequality_proof_l277_277489


namespace expression_value_l277_277239

theorem expression_value : 2016 - 2017 + 2018 - 2019 + 2020 = 2018 := 
by 
  sorry

end expression_value_l277_277239


namespace rectangle_area_l277_277016

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l277_277016


namespace phi_range_l277_277447

noncomputable def f (ω φ x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + φ) + 1

theorem phi_range (ω φ : ℝ) 
  (h₀ : ω > 0)
  (h₁ : |φ| ≤ Real.pi / 2)
  (h₂ : ∃ x₁ x₂, x₁ ≠ x₂ ∧ f ω φ x₁ = 2 ∧ f ω φ x₂ = 2 ∧ |x₂ - x₁| = Real.pi / 3)
  (h₃ : ∀ x, x ∈ Set.Ioo (-Real.pi / 8) (Real.pi / 3) → f ω φ x > 1) :
  φ ∈ Set.Icc (Real.pi / 4) (Real.pi / 3) :=
sorry

end phi_range_l277_277447


namespace domain_of_function_l277_277888

def function_undefined_at (x : ℝ) : Prop :=
  ∃ y : ℝ, y = (x - 3) / (x - 2)

theorem domain_of_function (x : ℝ) : ¬(x = 2) ↔ function_undefined_at x :=
sorry

end domain_of_function_l277_277888


namespace tenth_term_arithmetic_sequence_l277_277785

theorem tenth_term_arithmetic_sequence :
  let a_1 := (1 : ℝ) / 2
  let a_2 := (5 : ℝ) / 6
  let d := a_2 - a_1
  (a_1 + 9 * d) = 7 / 2 := 
by
  sorry

end tenth_term_arithmetic_sequence_l277_277785


namespace max_value_of_g_l277_277431

noncomputable def f1 (x : ℝ) : ℝ := 3 * x + 3
noncomputable def f2 (x : ℝ) : ℝ := (1/3) * x + 2
noncomputable def f3 (x : ℝ) : ℝ := -x + 8

noncomputable def g (x : ℝ) : ℝ := min (min (f1 x) (f2 x)) (f3 x)

theorem max_value_of_g : ∃ x : ℝ, g x = 3.5 :=
by
  sorry

end max_value_of_g_l277_277431


namespace area_of_rectangle_ABCD_l277_277007

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l277_277007


namespace remainder_3001_3002_3003_3004_3005_mod_17_l277_277094

theorem remainder_3001_3002_3003_3004_3005_mod_17 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 7 := 
begin
  sorry
end

end remainder_3001_3002_3003_3004_3005_mod_17_l277_277094


namespace least_xy_l277_277629

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 9) : xy = 108 := by
  sorry

end least_xy_l277_277629


namespace donna_paid_correct_amount_l277_277533

-- Define the original price, discount rate, and sales tax rate
def original_price : ℝ := 200
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.10

-- Define the total amount Donna paid
def total_amount_donna_paid : ℝ := 165

-- Define a theorem to express the proof problem
theorem donna_paid_correct_amount :
  let discount_amount := original_price * discount_rate in
  let sale_price := original_price - discount_amount in
  let sales_tax_amount := sale_price * sales_tax_rate in
  let total_amount := sale_price + sales_tax_amount in
  total_amount = total_amount_donna_paid :=
by
  sorry

end donna_paid_correct_amount_l277_277533


namespace remainder_196c_2008_mod_97_l277_277612

theorem remainder_196c_2008_mod_97 (c : ℤ) : ((196 * c) ^ 2008) % 97 = 44 := by
  sorry

end remainder_196c_2008_mod_97_l277_277612


namespace total_hours_correct_l277_277047

def hours_watching_tv_per_day : ℕ := 4
def days_per_week : ℕ := 7
def days_playing_video_games_per_week : ℕ := 3

def tv_hours_per_week : ℕ := hours_watching_tv_per_day * days_per_week
def video_game_hours_per_day : ℕ := hours_watching_tv_per_day / 2
def video_game_hours_per_week : ℕ := video_game_hours_per_day * days_playing_video_games_per_week

def total_hours_per_week : ℕ := tv_hours_per_week + video_game_hours_per_week

theorem total_hours_correct :
  total_hours_per_week = 34 := by
  sorry

end total_hours_correct_l277_277047


namespace value_is_twenty_l277_277389

theorem value_is_twenty (n : ℕ) (h : n = 16) : 32 - 12 = 20 :=
by {
  -- Simplification of the proof process
  sorry
}

end value_is_twenty_l277_277389


namespace original_ratio_of_boarders_to_day_students_l277_277508

theorem original_ratio_of_boarders_to_day_students
    (original_boarders : ℕ)
    (new_boarders : ℕ)
    (new_ratio_b_d : ℕ → ℕ)
    (no_switch : Prop)
    (no_leave : Prop)
  : (original_boarders = 220) ∧ (new_boarders = 44) ∧ (new_ratio_b_d 1 = 2) ∧ no_switch ∧ no_leave →
  ∃ (original_day_students : ℕ), original_day_students = 528 ∧ (220 / 44 = 5) ∧ (528 / 44 = 12)
  := by
    sorry

end original_ratio_of_boarders_to_day_students_l277_277508


namespace mn_equals_neg16_l277_277298

theorem mn_equals_neg16 (m n : ℤ) (h1 : m = -2) (h2 : |n| = 8) (h3 : m + n > 0) : m * n = -16 := by
  sorry

end mn_equals_neg16_l277_277298


namespace rectangle_area_l277_277019

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l277_277019


namespace division_remainder_l277_277096

def p (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30
def d (x : ℝ) : ℝ := 4 * x - 8

theorem division_remainder : (∃ q r, p(2) = d(2) * q + r ∧ d(2) ≠ 0 ∧ r = 10) :=
by
  sorry

end division_remainder_l277_277096


namespace find_a11_l277_277312

variable (a : ℕ → ℝ)

axiom geometric_seq (a : ℕ → ℝ) (r : ℝ) : ∀ n, a (n + 1) = a n * r

variable (r : ℝ)
variable (h3 : a 3 = 4)
variable (h7 : a 7 = 12)

theorem find_a11 : a 11 = 36 := by
  sorry

end find_a11_l277_277312


namespace integers_exist_for_eqns_l277_277608

theorem integers_exist_for_eqns (a b c : ℤ) :
  ∃ (p1 q1 r1 p2 q2 r2 : ℤ), 
    a = q1 * r2 - q2 * r1 ∧ 
    b = r1 * p2 - r2 * p1 ∧ 
    c = p1 * q2 - p2 * q1 :=
  sorry

end integers_exist_for_eqns_l277_277608


namespace simplify_and_evaluate_l277_277219

theorem simplify_and_evaluate (m : ℝ) (h : m = 5) :
  (m + 2 - (5 / (m - 2))) / ((3 * m - m^2) / (m - 2)) = - (8 / 5) :=
by
  sorry

end simplify_and_evaluate_l277_277219


namespace reject_null_hypothesis_proof_l277_277379

noncomputable def sample_size_1 : ℕ := 14
noncomputable def sample_size_2 : ℕ := 10
noncomputable def sample_variance_x : ℝ := 0.84
noncomputable def sample_variance_y : ℝ := 2.52
noncomputable def significance_level : ℝ := 0.1

-- Normal populations X and Y
def NormalPopulation_X : Type := sorry
def NormalPopulation_Y : Type := sorry

-- Hypotheses
def null_hypothesis : Prop := σ NormalPopulation_X ^ 2 = σ NormalPopulation_Y ^ 2
def alternative_hypothesis : Prop := σ NormalPopulation_X ^ 2 ≠ σ NormalPopulation_Y ^ 2

-- F-statistic for comparing variances
noncomputable def F_statistic : ℝ := sample_variance_y / sample_variance_x

-- Degrees of freedom
noncomputable def df_numerator : ℕ := sample_size_2 - 1
noncomputable def df_denominator : ℕ := sample_size_1 - 1

-- Critical value at alpha / 2 = 0.05 for two-tailed test
noncomputable def F_critical_value : ℝ := 2.72 -- approximate value

-- Test criterion
def reject_null_hypothesis : Prop := F_statistic > F_critical_value

-- Goal: Prove or disprove the null hypothesis given the conditions
theorem reject_null_hypothesis_proof : reject_null_hypothesis → alternative_hypothesis :=
by
  -- Proof omitted.
  sorry

end reject_null_hypothesis_proof_l277_277379


namespace fish_bird_apple_fraction_l277_277682

theorem fish_bird_apple_fraction (M : ℝ) (hM : 0 < M) :
  let R_fish := 120
  let R_bird := 60
  let R_total := 180
  let T := M / R_total
  let fish_fraction := (R_fish * T) / M
  let bird_fraction := (R_bird * T) / M
  fish_fraction = 2/3 ∧ bird_fraction = 1/3 := by
  sorry

end fish_bird_apple_fraction_l277_277682


namespace max_correct_answers_l277_277621

theorem max_correct_answers (a b c : ℕ) :
  a + b + c = 50 ∧ 4 * a - c = 99 ∧ b = 50 - a - c ∧ 50 - a - c ≥ 0 →
  a ≤ 29 := by
  sorry

end max_correct_answers_l277_277621


namespace competition_end_time_l277_277259

def time := ℕ × ℕ -- Representing time as a pair of hours and minutes

def start_time : time := (15, 15) -- 3:15 PM is represented as 15:15 in 24-hour format
def duration := 1825 -- Duration in minutes
def end_time : time := (21, 40) -- 9:40 PM is represented as 21:40 in 24-hour format

def add_minutes (t : time) (m : ℕ) : time :=
  let (h, min) := t
  let total_minutes := h * 60 + min + m
  (total_minutes / 60 % 24, total_minutes % 60)

theorem competition_end_time :
  add_minutes start_time duration = end_time :=
by
  -- The proof would go here
  sorry

end competition_end_time_l277_277259


namespace count_valid_M_l277_277868

open Real

def count_valid_integers_less_than (n: ℕ) : ℕ := sorry

theorem count_valid_M :
  count_valid_integers_less_than 2000 = 412 :=
sorry

end count_valid_M_l277_277868


namespace exists_non_decreasing_subsequences_l277_277624

theorem exists_non_decreasing_subsequences {a b c : ℕ → ℕ} : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
sorry

end exists_non_decreasing_subsequences_l277_277624


namespace Ivan_pays_1_point_5_times_more_l277_277130

theorem Ivan_pays_1_point_5_times_more (x y : ℝ) (h : x = 2 * y) : 1.5 * (0.6 * x + 0.8 * y) = x + y :=
by
  sorry

end Ivan_pays_1_point_5_times_more_l277_277130


namespace range_of_a_l277_277902

open Real

/-- Proposition p: x^2 + 2*a*x + 4 > 0 for all x in ℝ -/
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: the exponential function (3 - 2*a)^x is increasing -/
def q (a : ℝ) : Prop :=
  3 - 2*a > 1

/-- Given p ∧ q, prove that -2 < a < 1 -/
theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : -2 < a ∧ a < 1 :=
sorry

end range_of_a_l277_277902


namespace thre_digit_num_condition_l277_277794

theorem thre_digit_num_condition (n : ℕ) (h : n = 735) :
  (n % 35 = 0) ∧ (Nat.digits 10 n).sum = 15 := by
  sorry

end thre_digit_num_condition_l277_277794


namespace exists_sequence_satisfying_conditions_l277_277129

def F : ℕ → ℕ := sorry

theorem exists_sequence_satisfying_conditions :
  (∀ n, ∃ k, F k = n) ∧ 
  (∀ n, ∃ m > n, F m = n) ∧ 
  (∀ n ≥ 2, F (F (n ^ 163)) = F (F n) + F (F 361)) :=
sorry

end exists_sequence_satisfying_conditions_l277_277129


namespace base_addition_is_10_l277_277127

-- The problem states that adding two numbers in a particular base results in a third number in the same base.
def valid_base_10_addition (n m k b : ℕ) : Prop :=
  let n_b := n / b^2 * b^2 + (n / b % b) * b + n % b
  let m_b := m / b^2 * b^2 + (m / b % b) * b + m % b
  let k_b := k / b^2 * b^2 + (k / b % b) * b + k % b
  n_b + m_b = k_b

theorem base_addition_is_10 : valid_base_10_addition 172 156 340 10 :=
  sorry

end base_addition_is_10_l277_277127


namespace p_neither_necessary_nor_sufficient_l277_277577

def p (x y : ℝ) : Prop := x + y ≠ -2
def q (x : ℝ) : Prop := x ≠ 0
def r (y : ℝ) : Prop := y ≠ -1

theorem p_neither_necessary_nor_sufficient (x y : ℝ) (h1: p x y) (h2: q x) (h3: r y) :
  ¬(p x y → q x) ∧ ¬(q x → p x y) := 
by 
  sorry

end p_neither_necessary_nor_sufficient_l277_277577


namespace minimum_value_of_expression_l277_277958

theorem minimum_value_of_expression (x : ℝ) (hx : x ≠ 0) : 
  (x^2 + 1 / x^2) ≥ 2 ∧ (x^2 + 1 / x^2 = 2 ↔ x = 1 ∨ x = -1) := 
by
  sorry

end minimum_value_of_expression_l277_277958


namespace event_day_is_Sunday_l277_277485

def days_in_week := 7

def event_day := 1500

def start_day := "Friday"

def day_of_week_according_to_mod : Nat → String 
| 0 => "Friday"
| 1 => "Saturday"
| 2 => "Sunday"
| 3 => "Monday"
| 4 => "Tuesday"
| 5 => "Wednesday"
| 6 => "Thursday"
| _ => "Invalid"

theorem event_day_is_Sunday : day_of_week_according_to_mod (event_day % days_in_week) = "Sunday" :=
sorry

end event_day_is_Sunday_l277_277485


namespace range_of_a_l277_277732

theorem range_of_a (a : ℝ) (h : (2 - a)^3 > (a - 1)^3) : a < 3/2 :=
sorry

end range_of_a_l277_277732


namespace train_crosses_second_platform_l277_277265

theorem train_crosses_second_platform (
  length_train length_platform1 length_platform2 : ℝ) 
  (time_platform1 : ℝ) 
  (H1 : length_train = 100)
  (H2 : length_platform1 = 200)
  (H3 : length_platform2 = 300)
  (H4 : time_platform1 = 15) :
  ∃ t : ℝ, t = 20 := by
  sorry

end train_crosses_second_platform_l277_277265


namespace c_negative_l277_277032

theorem c_negative (a b c : ℝ) (h₁ : a + b + c < 0) (h₂ : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) : 
  c < 0 :=
sorry

end c_negative_l277_277032


namespace quadratic_trinomial_properties_l277_277427

noncomputable def quadratic_trinomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_trinomial_properties
  (p : ℝ → ℝ)
  (h1 : p (1/2) = -49/4)
  (h2 : ∀ (x : ℝ), p x = quadratic_trinomial 1 (-1) (-12) x)
  (roots : ∃ x1 x2 : ℝ, p x = quadratic_trinomial 1 (-1) (-12) x1 ∧ p x = quadratic_trinomial 1 (-1) (-12) x2)
  (roots_sum : (roots x1)^4 + (roots x2)^4 = 337) :
  p x = quadratic_trinomial 1 (-1) (-12) x :=
by
  sorry

end quadratic_trinomial_properties_l277_277427


namespace two_digit_numbers_l277_277138

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem two_digit_numbers (a b : ℕ) (h1 : is_digit a) (h2 : is_digit b) 
  (h3 : a ≠ b) (h4 : (a + b) = 11) : 
  (∃ n m : ℕ, (n = 10 * a + b) ∧ (m = 10 * b + a) ∧ (∃ k : ℕ, (10 * a + b)^2 - (10 * b + a)^2 = k^2)) := 
sorry

end two_digit_numbers_l277_277138


namespace constant_sequence_if_and_only_if_arith_geo_progression_l277_277285

/-- A sequence a_n is both an arithmetic and geometric progression if and only if it is constant --/
theorem constant_sequence_if_and_only_if_arith_geo_progression (a : ℕ → ℝ) :
  (∃ q d : ℝ, (∀ n : ℕ, a (n+1) - a n = d) ∧ (∀ n : ℕ, a n = a 0 * q ^ n)) ↔ (∃ c : ℝ, ∀ n : ℕ, a n = c) := 
sorry

end constant_sequence_if_and_only_if_arith_geo_progression_l277_277285


namespace rectangle_area_l277_277018

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l277_277018


namespace max_value_of_expression_l277_277223

open Real

theorem max_value_of_expression
  (x y : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : x^2 - 2 * x * y + 3 * y^2 = 10) 
  : x^2 + 2 * x * y + 3 * y^2 ≤ 10 * (45 + 42 * sqrt 3) := 
sorry

end max_value_of_expression_l277_277223


namespace find_solution_set_l277_277739

-- Define the problem
def absolute_value_equation_solution_set (x : ℝ) : Prop :=
  |x - 2| + |2 * x - 3| = |3 * x - 5|

-- Define the expected solution set
def solution_set (x : ℝ) : Prop :=
  x ≤ 3 / 2 ∨ 2 ≤ x

-- The proof problem statement
theorem find_solution_set :
  ∀ x : ℝ, absolute_value_equation_solution_set x ↔ solution_set x :=
sorry -- No proof required, so we use 'sorry' to skip the proof

end find_solution_set_l277_277739


namespace nat_pairs_satisfy_conditions_l277_277841

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end nat_pairs_satisfy_conditions_l277_277841


namespace fraction_neither_cell_phones_nor_pagers_l277_277545

theorem fraction_neither_cell_phones_nor_pagers
  (E : ℝ) -- total number of employees (E must be positive)
  (h1 : 0 < E)
  (frac_cell_phones : ℝ)
  (H1 : frac_cell_phones = (2 / 3))
  (frac_pagers : ℝ)
  (H2 : frac_pagers = (2 / 5))
  (frac_both : ℝ)
  (H3 : frac_both = 0.4) :
  (1 / 3) = (1 - frac_cell_phones - frac_pagers + frac_both) :=
by
  -- setup definitions, conditions and final proof
  sorry

end fraction_neither_cell_phones_nor_pagers_l277_277545


namespace closest_fraction_to_medals_won_l277_277120

theorem closest_fraction_to_medals_won :
  let gamma_fraction := (13:ℚ) / 80
  let fraction_1_4 := (1:ℚ) / 4
  let fraction_1_5 := (1:ℚ) / 5
  let fraction_1_6 := (1:ℚ) / 6
  let fraction_1_7 := (1:ℚ) / 7
  let fraction_1_8 := (1:ℚ) / 8
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_4) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_5) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_7) ∧
  abs (gamma_fraction - fraction_1_6) <
    abs (gamma_fraction - fraction_1_8) := by
  sorry

end closest_fraction_to_medals_won_l277_277120


namespace digit_sum_divisible_by_9_l277_277824

theorem digit_sum_divisible_by_9 (n : ℕ) (h : n < 10) : 
  (8 + 6 + 5 + n + 7 + 4 + 3 + 2) % 9 = 0 ↔ n = 1 := 
by sorry 

end digit_sum_divisible_by_9_l277_277824


namespace factor_polynomial_equiv_l277_277830

theorem factor_polynomial_equiv :
  (x^2 + 2 * x + 1) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 7 * x + 1) * (x^2 + 3 * x + 7) :=
by sorry

end factor_polynomial_equiv_l277_277830


namespace connie_num_markers_l277_277282

def num_red_markers (T : ℝ) := 0.41 * T
def num_total_markers (num_blue_markers : ℝ) (T : ℝ) := num_red_markers T + num_blue_markers

theorem connie_num_markers (T : ℝ) (h1 : num_total_markers 23 T = T) : T = 39 :=
by
sorry

end connie_num_markers_l277_277282


namespace initial_contribution_amount_l277_277250

variable (x : ℕ)
variable (workers : ℕ := 1200)
variable (total_with_extra_contribution: ℕ := 360000)
variable (extra_contribution_each: ℕ := 50)

theorem initial_contribution_amount :
  (workers * x = total_with_extra_contribution - workers * extra_contribution_each) →
  workers * x = 300000 :=
by
  intro h
  sorry

end initial_contribution_amount_l277_277250


namespace interval_with_three_buses_l277_277938

theorem interval_with_three_buses (interval_two_buses : ℕ) (total_route_time : ℕ) (bus_count : ℕ) : 
  interval_two_buses = 21 → total_route_time = 2 * interval_two_buses → bus_count = 3 → 
  (total_route_time / bus_count) = 14 :=
by
  intros h1 h2 h3
  rw [h1, h3, ← h2]
  simp
  sorry

end interval_with_three_buses_l277_277938


namespace seventh_grade_caps_collection_l277_277510

theorem seventh_grade_caps_collection (A B C : ℕ) 
  (h1 : A = (B + C) / 2) 
  (h2 : B = (A + C) / 3)
  (h3 : C = 150) : A + B + C = 360 := 
by 
  sorry

end seventh_grade_caps_collection_l277_277510


namespace right_triangle_max_area_l277_277114

theorem right_triangle_max_area
  (a b : ℝ) (h_a_nonneg : 0 ≤ a) (h_b_nonneg : 0 ≤ b)
  (h_right_triangle : a^2 + b^2 = 20^2)
  (h_perimeter : a + b + 20 = 48) :
  (1 / 2) * a * b = 96 :=
by
  sorry

end right_triangle_max_area_l277_277114


namespace books_added_is_10_l277_277930

-- Define initial number of books on the shelf
def initial_books : ℕ := 38

-- Define the final number of books on the shelf
def final_books : ℕ := 48

-- Define the number of books that Marta added
def books_added : ℕ := final_books - initial_books

-- Theorem stating that Marta added 10 books
theorem books_added_is_10 : books_added = 10 :=
by
  sorry

end books_added_is_10_l277_277930


namespace lollipops_left_l277_277345

def problem_conditions : Prop :=
  ∃ (lollipops_bought lollipops_eaten lollipops_left : ℕ),
    lollipops_bought = 12 ∧
    lollipops_eaten = 5 ∧
    lollipops_left = lollipops_bought - lollipops_eaten

theorem lollipops_left (lollipops_bought lollipops_eaten lollipops_left : ℕ) 
  (hb : lollipops_bought = 12) (he : lollipops_eaten = 5) (hl : lollipops_left = lollipops_bought - lollipops_eaten) : 
  lollipops_left = 7 := 
by 
  sorry

end lollipops_left_l277_277345


namespace find_m_l277_277311

theorem find_m (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (4 / x) + (9 / y) = m) (h4 : ∃ x y , x + y = 5/6) : m = 30 :=
sorry

end find_m_l277_277311


namespace sqrt_simplification_l277_277495

-- Define a constant for the given number
def n : ℕ := 360000

-- State the theorem we want to prove
theorem sqrt_simplification (h : sqrt n = 600) : sqrt 360000 = 600 := 
by assumption

end sqrt_simplification_l277_277495


namespace base8_to_decimal_l277_277244

theorem base8_to_decimal (n : ℕ) (h : n = 54321) : 
  (5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0) = 22737 := 
by
  sorry

end base8_to_decimal_l277_277244


namespace sum_first_n_terms_geometric_sequence_l277_277761

def geometric_sequence_sum (n : ℕ) (k : ℝ) : ℝ :=
  if n = 0 then 0 else (3 * 2^n + k)

theorem sum_first_n_terms_geometric_sequence (k : ℝ) :
  (geometric_sequence_sum 1 k = 6 + k) ∧ 
  (∀ n > 1, geometric_sequence_sum n k - geometric_sequence_sum (n - 1) k = 3 * 2^(n-1))
  → k = -3 :=
by
  sorry

end sum_first_n_terms_geometric_sequence_l277_277761


namespace area_of_rectangle_l277_277028

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l277_277028


namespace min_red_hair_students_l277_277667

theorem min_red_hair_students (B N R : ℕ) 
  (h1 : B + N + R = 50)
  (h2 : N ≥ B - 1)
  (h3 : R ≥ N - 1) :
  R = 17 := sorry

end min_red_hair_students_l277_277667


namespace piles_can_be_reduced_l277_277324

/-! 
  We define similar sizes as the difference between sizes being at most a factor of two.
  Given any number of piles of stones, we aim to prove that these piles can be combined 
  iteratively into one single pile.
-/

def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

theorem piles_can_be_reduced (n : ℕ) :
  ∃ pile : ℕ, (pile = n) ∧ (∀ piles : list ℕ, list.sum piles = n → 
    (∃ piles' : list ℕ, list.sum piles' = n ∧ list.length piles' = 1)) :=
by
  -- Placeholder for the proof.
  sorry

end piles_can_be_reduced_l277_277324


namespace complex_number_quadrant_l277_277918

def z : ℂ := (↑complex.I) / (1 + ↑complex.I)

theorem complex_number_quadrant : z.re > 0 ∧ z.im > 0 := sorry

end complex_number_quadrant_l277_277918


namespace sum_of_terms_arithmetic_sequence_l277_277041

variable {S : ℕ → ℕ}
variable {k : ℕ}

-- Given conditions
axiom S_k : S k = 2
axiom S_3k : S (3 * k) = 18

-- The statement to prove
theorem sum_of_terms_arithmetic_sequence : S (4 * k) = 32 := by
  sorry

end sum_of_terms_arithmetic_sequence_l277_277041


namespace base_n_system_digits_l277_277748

theorem base_n_system_digits (N : ℕ) (h : N ≥ 6) :
  ((N - 1) ^ 4).digits N = [N-4, 5, N-4, 1] :=
by
  sorry

end base_n_system_digits_l277_277748


namespace unique_diff_of_cubes_l277_277903

theorem unique_diff_of_cubes (n k : ℕ) (h : 61 = n^3 - k^3) : n = 5 ∧ k = 4 :=
sorry

end unique_diff_of_cubes_l277_277903


namespace simplify_expression_l277_277107

variable {a b : ℝ}

theorem simplify_expression : (4 * a^3 * b - 2 * a * b) / (2 * a * b) = 2 * a^2 - 1 := by
  sorry

end simplify_expression_l277_277107


namespace area_of_rectangle_l277_277105

theorem area_of_rectangle (w d : ℝ) (h_w : w = 4) (h_d : d = 5) : ∃ l : ℝ, (w^2 + l^2 = d^2) ∧ (w * l = 12) :=
by
  sorry

end area_of_rectangle_l277_277105


namespace max_subjects_per_teacher_l277_277406

theorem max_subjects_per_teacher (maths physics chemistry : ℕ) (min_teachers : ℕ)
  (h_math : maths = 6) (h_physics : physics = 5) (h_chemistry : chemistry = 5) (h_min_teachers : min_teachers = 4) :
  (maths + physics + chemistry) / min_teachers = 4 :=
by
  -- the proof will be here
  sorry

end max_subjects_per_teacher_l277_277406


namespace frank_total_cans_l277_277432

def cansCollectedSaturday : List Nat := [4, 6, 5, 7, 8]
def cansCollectedSunday : List Nat := [6, 5, 9]
def cansCollectedMonday : List Nat := [8, 8]

def totalCansCollected (lst1 lst2 lst3 : List Nat) : Nat :=
  lst1.sum + lst2.sum + lst3.sum

theorem frank_total_cans :
  totalCansCollected cansCollectedSaturday cansCollectedSunday cansCollectedMonday = 66 :=
by
  sorry

end frank_total_cans_l277_277432


namespace rectangle_area_l277_277014

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l277_277014


namespace triangular_weight_is_60_l277_277922

/-- Suppose there are weights: 5 identical round, 2 identical triangular, and 1 rectangular weight of 90 grams.
    The conditions are: 
    1. One round weight and one triangular weight balance three round weights.
    2. Four round weights and one triangular weight balance one triangular weight, one round weight, and one rectangular weight.
    Prove that the weight of the triangular weight is 60 grams. -/
theorem triangular_weight_is_60 
  (R T : ℕ)  -- We declare weights of round and triangular weights as natural numbers
  (h1 : R + T = 3 * R)  -- The first balance condition
  (h2 : 4 * R + T = T + R + 90)  -- The second balance condition
  : T = 60 := 
by
  sorry  -- Proof omitted

end triangular_weight_is_60_l277_277922


namespace min_width_of_garden_l277_277474

theorem min_width_of_garden (w : ℝ) (h : w*(w + 10) ≥ 150) : w ≥ 10 :=
by
  sorry

end min_width_of_garden_l277_277474


namespace three_person_subcommittees_count_l277_277719

theorem three_person_subcommittees_count : ∃ n k, n = 8 ∧ k = 3 ∧ nat.choose n k = 56 :=
begin
  use [8, 3],
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end three_person_subcommittees_count_l277_277719


namespace total_cost_of_apples_and_bananas_l277_277646

variable (a b : ℝ)

theorem total_cost_of_apples_and_bananas (a b : ℝ) : 2 * a + 3 * b = 2 * a + 3 * b :=
by
  sorry

end total_cost_of_apples_and_bananas_l277_277646


namespace the_inequality_l277_277339

theorem the_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (a / (1 + b)) + (b / (1 + c)) + (c / (1 + a)) ≥ 3 / 2 :=
by sorry

end the_inequality_l277_277339


namespace ratio_of_x_y_l277_277371

theorem ratio_of_x_y (x y : ℝ) (h : x + y = 3 * (x - y)) : x / y = 2 :=
by
  sorry

end ratio_of_x_y_l277_277371


namespace min_value_inequality_l277_277203

open Real

theorem min_value_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (bc / a) + (ac / b) + (ab / c) ≥ 1 := 
by 
  -- Proof goes here
  sorry

end min_value_inequality_l277_277203


namespace pat_more_hours_than_jane_l277_277338

theorem pat_more_hours_than_jane (H P K M J : ℝ) 
  (h_total : H = P + K + M + J)
  (h_pat : P = 2 * K)
  (h_mark : M = (1/3) * P)
  (h_jane : J = (1/2) * M)
  (H290 : H = 290) :
  P - J = 120.83 := 
by
  sorry

end pat_more_hours_than_jane_l277_277338


namespace possible_values_of_a_plus_b_l277_277455

variable (a b : ℤ)

theorem possible_values_of_a_plus_b (h1 : |a| = 2) (h2 : |b| = a) :
  (a + b = 0 ∨ a + b = 4 ∨ a + b = -4) :=
sorry

end possible_values_of_a_plus_b_l277_277455


namespace no_solution_when_k_equals_7_l277_277149

noncomputable def no_solution_eq (k x : ℝ) : Prop :=
  (x - 3) / (x - 4) = (x - k) / (x - 8)
  
theorem no_solution_when_k_equals_7 :
  ∀ x : ℝ, x ≠ 4 → x ≠ 8 → ¬ no_solution_eq 7 x :=
by
  sorry

end no_solution_when_k_equals_7_l277_277149


namespace sum_of_four_digit_numbers_formed_by_digits_1_to_5_l277_277290

theorem sum_of_four_digit_numbers_formed_by_digits_1_to_5 :
  let S := {1, 2, 3, 4, 5}
  let four_digits_sum (n1 n2 n3 n4 : ℕ) :=
    1000 * n1 + 100 * n2 + 10 * n3 + n4
  (∀ a b c d ∈ S, a ≠ b → b ≠ c → c ≠ d → d ≠ a → a ≠ c → b ≠ d 
  → sum (four_digits_sum a b c d) = 399960) := sorry

end sum_of_four_digit_numbers_formed_by_digits_1_to_5_l277_277290


namespace probability_of_purple_marble_l277_277671

theorem probability_of_purple_marble (p_blue p_green p_purple : ℝ) 
  (h_blue : p_blue = 0.3) 
  (h_green : p_green = 0.4) 
  (h_sum : p_blue + p_green + p_purple = 1) : 
  p_purple = 0.3 := 
by 
  -- proof goes here
  sorry

end probability_of_purple_marble_l277_277671


namespace correct_student_mark_l277_277752

theorem correct_student_mark (x : ℕ) : 
  (∀ (n : ℕ), n = 30) →
  (∀ (avg correct_avg wrong_mark correct_mark : ℕ), 
    avg = 100 ∧ 
    correct_avg = 98 ∧ 
    wrong_mark = 70 ∧ 
    (n * avg) - wrong_mark + correct_mark = n * correct_avg) →
  x = 10 := by
  intros
  sorry

end correct_student_mark_l277_277752


namespace four_student_round_table_l277_277537

theorem four_student_round_table
  (G : SimpleGraph (Fin 2021))
  (h_deg : ∀ v, G.degree v ≥ 45) :
  ∃ (v1 v2 v3 v4 : Fin 2021), 
    G.Adj v1 v2 ∧ G.Adj v2 v3 ∧ G.Adj v3 v4 ∧ G.Adj v4 v1 ∧ 
    ¬ v1 = v3 ∧ ¬ v2 = v4 :=
by
  sorry

end four_student_round_table_l277_277537


namespace proportion_calculation_l277_277173

theorem proportion_calculation (x y : ℝ) (h1 : 0.75 / x = 5 / y) (h2 : x = 1.2) : y = 8 :=
by
  sorry

end proportion_calculation_l277_277173


namespace calculate_weight_l277_277049

theorem calculate_weight (W : ℝ) (h : 0.75 * W + 2 = 62) : W = 80 :=
by
  sorry

end calculate_weight_l277_277049


namespace total_quantities_l277_277072

theorem total_quantities (n S S₃ S₂ : ℕ) (h₁ : S = 6 * n) (h₂ : S₃ = 4 * 3) (h₃ : S₂ = 33 * 2) (h₄ : S = S₃ + S₂) : n = 13 :=
by
  sorry

end total_quantities_l277_277072


namespace find_pairs_l277_277839

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end find_pairs_l277_277839


namespace factorize_ax_squared_minus_9a_l277_277563

theorem factorize_ax_squared_minus_9a (a x : ℝ) : 
  a * x^2 - 9 * a = a * (x - 3) * (x + 3) :=
sorry

end factorize_ax_squared_minus_9a_l277_277563


namespace mike_total_time_spent_l277_277045

theorem mike_total_time_spent : 
  let hours_watching_tv_per_day := 4
  let days_per_week := 7
  let days_playing_video_games := 3
  let hours_playing_video_games_per_day := hours_watching_tv_per_day / 2
  let total_hours_watching_tv := hours_watching_tv_per_day * days_per_week
  let total_hours_playing_video_games := hours_playing_video_games_per_day * days_playing_video_games
  let total_time_spent := total_hours_watching_tv + total_hours_playing_video_games
  total_time_spent = 34 :=
by
  sorry

end mike_total_time_spent_l277_277045


namespace mike_passing_percentage_l277_277487

theorem mike_passing_percentage :
  ∀ (score shortfall max_marks : ℕ), 
    score = 212 ∧ shortfall = 25 ∧ max_marks = 790 →
    (score + shortfall) / max_marks * 100 = 30 :=
by
  intros score shortfall max_marks h
  have h1 : score = 212 := h.1
  have h2 : shortfall = 25 := h.2.1
  have h3 : max_marks = 790 := h.2.2
  rw [h1, h2, h3]
  sorry

end mike_passing_percentage_l277_277487


namespace max_k_l277_277449

def A : Finset ℕ := {0,1,2,3,4,5,6,7,8,9}

def valid_collection (B : ℕ → Finset ℕ) (k : ℕ) : Prop :=
  ∀ i j : ℕ, i < k → j < k → i ≠ j → (B i ∩ B j).card ≤ 2

theorem max_k (B : ℕ → Finset ℕ) : ∃ k, valid_collection B k → k ≤ 175 := sorry

end max_k_l277_277449


namespace inscribed_square_ab_l277_277812

theorem inscribed_square_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^2 + b^2 = 32) : 2 * a * b = -7 :=
by
  sorry

end inscribed_square_ab_l277_277812


namespace Lagrange_interpol_equiv_x_squared_l277_277053

theorem Lagrange_interpol_equiv_x_squared (a b c x : ℝ)
    (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
    c^2 * ((x - a) * (x - b)) / ((c - a) * (c - b)) +
    b^2 * ((x - a) * (x - c)) / ((b - a) * (b - c)) +
    a^2 * ((x - b) * (x - c)) / ((a - b) * (a - c)) = x^2 := 
    sorry

end Lagrange_interpol_equiv_x_squared_l277_277053


namespace swimming_speed_in_still_water_l277_277804

theorem swimming_speed_in_still_water (v : ℝ) 
  (h_current_speed : 2 = 2) 
  (h_time_distance : 7 = 7) 
  (h_effective_speed : v - 2 = 14 / 7) : 
  v = 4 :=
sorry

end swimming_speed_in_still_water_l277_277804


namespace intersection_points_in_plane_l277_277636

-- Define the cones with parallel axes and equal angles
def cone1 (a1 b1 c1 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a1)^2 + (y - b1)^2 = k^2 * (z - c1)^2

def cone2 (a2 b2 c2 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a2)^2 + (y - b2)^2 = k^2 * (z - c2)^2

-- Given conditions
variable (a1 b1 c1 a2 b2 c2 k : ℝ)

-- The theorem to be proven
theorem intersection_points_in_plane (x y z : ℝ) 
  (h1 : cone1 a1 b1 c1 k x y z) (h2 : cone2 a2 b2 c2 k x y z) : 
  ∃ (A B C D : ℝ), A * x + B * y + C * z + D = 0 :=
by
  sorry

end intersection_points_in_plane_l277_277636


namespace sum_of_all_four_digit_numbers_formed_l277_277288

open List

noncomputable def sum_of_four_digit_numbers (digits : List ℕ) : ℕ :=
  let perms := digits.permutations.filter (λ l, l.length = 4)
  let nums := perms.map (λ l, 1000 * l.head + 100 * l.nthLe 1 sorry + 10 * l.nthLe 2 sorry + l.nthLe 3 sorry)
  nums.sum

theorem sum_of_all_four_digit_numbers_formed : sum_of_four_digit_numbers [1, 2, 3, 4, 5] = 399960 :=
by
  sorry

end sum_of_all_four_digit_numbers_formed_l277_277288


namespace equal_intercepts_condition_l277_277445

theorem equal_intercepts_condition (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  (a = b ∨ c = 0) ↔ (c = 0 ∨ (c ≠ 0 ∧ a = b)) :=
by sorry

end equal_intercepts_condition_l277_277445


namespace f_monotonic_non_overlapping_domains_domain_of_sum_l277_277481

axiom f : ℝ → ℝ
axiom f_decreasing : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ → x₁ ≤ 1 → -1 ≤ x₂ → x₂ ≤ 1 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

theorem f_monotonic : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ → x₁ ≤ 1 → -1 ≤ x₂ → x₂ ≤ 1 → x₁ ≤ x₂ → f x₁ ≥ f x₂ := sorry

theorem non_overlapping_domains : ∀ c : ℝ, (c - 1 > c^2 + 1 → c > 2) ∧ (c^2 - 1 > c + 1 → c < -1) := sorry

theorem domain_of_sum : 
  ∀ c : ℝ,
  -1 ≤ c ∧ c ≤ 2 →
  (∃ a b : ℝ, 
    ((-1 ≤ c ∧ c ≤ 0) ∨ (1 ≤ c ∧ c ≤ 2) → a = c^2 - 1 ∧ b = c + 1) ∧ 
    (0 < c ∧ c < 1 → a = c - 1 ∧ b = c^2 + 1)
  ) := sorry

end f_monotonic_non_overlapping_domains_domain_of_sum_l277_277481


namespace exists_root_interval_l277_277279

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_interval :
  (f 1.1 < 0) ∧ (f 1.2 > 0) → ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 := 
by
  intro h
  sorry

end exists_root_interval_l277_277279


namespace zoo_tickets_total_cost_l277_277113

-- Define the given conditions
def num_children := 6
def num_adults := 10
def cost_child_ticket := 10
def cost_adult_ticket := 16

-- Calculate the expected total cost
def total_cost := 220

-- State the theorem
theorem zoo_tickets_total_cost :
  num_children * cost_child_ticket + num_adults * cost_adult_ticket = total_cost :=
by
  sorry

end zoo_tickets_total_cost_l277_277113


namespace rectangle_area_ratio_is_three_l277_277735

variables {a b : ℝ}

-- Rectangle ABCD with midpoint F on CD, BC = 3 * BE
def rectangle_midpoint_condition (CD_length : ℝ) (BC_length : ℝ) (BE_length : ℝ) (F_midpoint : Prop) :=
  F_midpoint ∧ BC_length = 3 * BE_length

-- Areas and the ratio
def area_rectangle (CD_length BC_length : ℝ) : ℝ :=
  CD_length * BC_length

def area_shaded (a b : ℝ) : ℝ :=
  2 * a * b

theorem rectangle_area_ratio_is_three (h : rectangle_midpoint_condition (2 * a) (3 * b) b (F_midpoint := True)) :
  area_rectangle (2 * a) (3 * b) = 3 * area_shaded a b :=
by
  unfold rectangle_midpoint_condition at h
  unfold area_rectangle area_shaded
  rw [←mul_assoc, ←mul_assoc]
  sorry

end rectangle_area_ratio_is_three_l277_277735


namespace kevin_eggs_l277_277131

theorem kevin_eggs : 
  ∀ (bonnie george cheryl kevin : ℕ),
  bonnie = 13 → 
  george = 9 → 
  cheryl = 56 → 
  cheryl = bonnie + george + kevin + 29 →
  kevin = 5 :=
by
  intros bonnie george cheryl kevin h_bonnie h_george h_cheryl h_eqn
  subst h_bonnie
  subst h_george
  subst h_cheryl
  simp at h_eqn
  sorry

end kevin_eggs_l277_277131


namespace basket_ratio_l277_277460

variable (S A H : ℕ)

theorem basket_ratio 
  (alex_baskets : A = 8) 
  (hector_baskets : H = 2 * S) 
  (total_baskets : A + S + H = 80) : 
  (S : ℚ) / (A : ℚ) = 3 := 
by 
  sorry

end basket_ratio_l277_277460


namespace point_not_in_region_l277_277805

theorem point_not_in_region : ¬ (3 * 2 + 2 * 0 < 6) :=
by simp [lt_irrefl]

end point_not_in_region_l277_277805


namespace train_has_96_cars_l277_277984

def train_cars_count (cars_in_15_seconds : Nat) (time_for_15_seconds : Nat) (total_time_seconds : Nat) : Nat :=
  total_time_seconds * cars_in_15_seconds / time_for_15_seconds

theorem train_has_96_cars :
  train_cars_count 8 15 180 = 96 :=
by
  sorry

end train_has_96_cars_l277_277984


namespace half_day_division_l277_277674

theorem half_day_division : 
  ∃ (n m : ℕ), n * m = 43200 ∧ (∃! (k : ℕ), k = 60) := sorry

end half_day_division_l277_277674


namespace greatest_divisor_of_arithmetic_sequence_l277_277380

theorem greatest_divisor_of_arithmetic_sequence (x c : ℤ) (h_odd : x % 2 = 1) (h_even : c % 2 = 0) :
  15 ∣ (15 * (x + 7 * c)) :=
sorry

end greatest_divisor_of_arithmetic_sequence_l277_277380


namespace negation_of_no_vegetarian_students_eat_at_cafeteria_l277_277759

variable (Student : Type) 
variable (isVegetarian : Student → Prop)
variable (eatsAtCafeteria : Student → Prop)

theorem negation_of_no_vegetarian_students_eat_at_cafeteria :
  (∀ x, isVegetarian x → ¬ eatsAtCafeteria x) →
  (∃ x, isVegetarian x ∧ eatsAtCafeteria x) :=
by
  sorry

end negation_of_no_vegetarian_students_eat_at_cafeteria_l277_277759


namespace eccentricity_of_ellipse_l277_277713

theorem eccentricity_of_ellipse {b : ℝ} (hb : 0 < b ∧ b < 2) :
  let e := Real.sqrt (4 - b^2) / 2 in
  (∃ (x₀ y₀ : ℝ), (x₀^2 / 4 + y₀^2 / b^2 = 1 ∧ x₀^2 / 2 - y₀^2 = 1 ∧
    (- (x₀ * b) / (4 * y₀)) * (x₀ / (2 * y₀)) = -1)) →
  e = Real.sqrt 3 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l277_277713


namespace exists_n_geq_k_l277_277758

theorem exists_n_geq_k (a : ℕ → ℕ) (h_distinct : ∀ i j : ℕ, i ≠ j → a i ≠ a j) 
    (h_positive : ∀ i : ℕ, a i > 0) :
    ∀ k : ℕ, ∃ n : ℕ, n > k ∧ a n ≥ n :=
by
  intros k
  sorry

end exists_n_geq_k_l277_277758


namespace exists_polyhedron_with_given_vertices_and_edges_l277_277826

theorem exists_polyhedron_with_given_vertices_and_edges :
  ∃ (V : Finset (String)) (E : Finset (Finset (String))),
    V = { "A", "B", "C", "D", "E", "F", "G", "H" } ∧
    E = { { "A", "B" }, { "A", "C" }, { "A", "H" }, { "B", "C" },
          { "B", "D" }, { "C", "D" }, { "D", "E" }, { "E", "F" },
          { "E", "G" }, { "F", "G" }, { "F", "H" }, { "G", "H" } } ∧
    (V.card : ℤ) - (E.card : ℤ) + 6 = 2 :=
by
  sorry

end exists_polyhedron_with_given_vertices_and_edges_l277_277826


namespace gravel_per_truckload_l277_277405

def truckloads_per_mile : ℕ := 3
def miles_day1 : ℕ := 4
def miles_day2 : ℕ := 2 * miles_day1 - 1
def total_paved_miles : ℕ := miles_day1 + miles_day2
def total_road_length : ℕ := 16
def miles_remaining : ℕ := total_road_length - total_paved_miles
def remaining_truckloads : ℕ := miles_remaining * truckloads_per_mile
def barrels_needed : ℕ := 6
def gravel_per_pitch : ℕ := 5
def P : ℚ := barrels_needed / remaining_truckloads
def G : ℚ := gravel_per_pitch * P

theorem gravel_per_truckload :
  G = 2 :=
by
  sorry

end gravel_per_truckload_l277_277405


namespace part1_part2_l277_277206

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem part1 (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) : |f a x| ≤ 5/4 :=
by
  sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∃ x ∈ Set.Icc (-1:ℝ) (1:ℝ), f a x = 17/8) : a = -2 :=
by
  sorry

end part1_part2_l277_277206


namespace roots_of_quadratic_l277_277926

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic (m : ℝ) :
  let a := 1
  let b := (3 * m - 1)
  let c := (2 * m^2 - m)
  discriminant a b c ≥ 0 :=
by
  sorry

end roots_of_quadratic_l277_277926


namespace remaining_stock_is_120_l277_277329

-- Definitions derived from conditions
def green_beans_weight : ℕ := 60
def rice_weight : ℕ := green_beans_weight - 30
def sugar_weight : ℕ := green_beans_weight - 10
def rice_lost_weight : ℕ := rice_weight / 3
def sugar_lost_weight : ℕ := sugar_weight / 5
def remaining_rice : ℕ := rice_weight - rice_lost_weight
def remaining_sugar : ℕ := sugar_weight - sugar_lost_weight
def remaining_stock_weight : ℕ := remaining_rice + remaining_sugar + green_beans_weight

-- Theorem
theorem remaining_stock_is_120 : remaining_stock_weight = 120 := by
  sorry

end remaining_stock_is_120_l277_277329


namespace evaluate_expression_l277_277692

theorem evaluate_expression : 68 + (126 / 18) + (35 * 13) - 300 - (420 / 7) = 170 := by
  sorry

end evaluate_expression_l277_277692


namespace rectangle_area_l277_277916

theorem rectangle_area (W : ℕ) (hW : W = 5) (L : ℕ) (hL : L = 4 * W) : ∃ (A : ℕ), A = L * W ∧ A = 100 := 
by
  use 100
  sorry

end rectangle_area_l277_277916


namespace total_food_correct_l277_277505

def max_food_per_guest : ℕ := 2
def min_guests : ℕ := 162
def total_food_cons : ℕ := min_guests * max_food_per_guest

theorem total_food_correct : total_food_cons = 324 := by
  sorry

end total_food_correct_l277_277505


namespace sector_area_is_nine_l277_277158

-- Defining the given conditions
def arc_length (r θ : ℝ) : ℝ := r * θ
def sector_area (r θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Given conditions
variables (r : ℝ) (θ : ℝ)
variable (h1 : arc_length r θ = 6)
variable (h2 : θ = 2)

-- Goal: Prove that the area of the sector is 9
theorem sector_area_is_nine : sector_area r θ = 9 := by
  sorry

end sector_area_is_nine_l277_277158


namespace factor_polynomial_int_l277_277831

theorem factor_polynomial_int : 
  ∀ x : ℤ, 5 * (x + 3) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 = 
           (5 * x^2 + 81 * x + 315) * (x^2 + 16 * x + 213) := 
by
  intros
  norm_num
  sorry

end factor_polynomial_int_l277_277831


namespace sum_of_squares_eight_l277_277928

theorem sum_of_squares_eight (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 28) : x^2 + y^2 = 8 := 
  sorry

end sum_of_squares_eight_l277_277928


namespace percentage_to_decimal_l277_277965

theorem percentage_to_decimal : (5 / 100 : ℚ) = 0.05 := by
  sorry

end percentage_to_decimal_l277_277965


namespace shopkeeper_profit_percentage_goal_l277_277977

-- Definitions for CP, MP and discount percentage
variable (CP : ℝ)
noncomputable def MP : ℝ := CP * 1.32
noncomputable def discount_percentage : ℝ := 0.18939393939393938
noncomputable def SP : ℝ := MP CP - (discount_percentage * MP CP)
noncomputable def profit : ℝ := SP CP - CP
noncomputable def profit_percentage : ℝ := (profit CP / CP) * 100

-- Theorem stating that the profit percentage is approximately 7%
theorem shopkeeper_profit_percentage_goal :
  abs (profit_percentage CP - 7) < 0.01 := sorry

end shopkeeper_profit_percentage_goal_l277_277977


namespace number_of_whole_numbers_between_sqrts_l277_277723

noncomputable def count_whole_numbers_between_sqrts : ℕ :=
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let min_int := Int.ceil lower_bound
  let max_int := Int.floor upper_bound
  Int.natAbs (max_int - min_int + 1)

theorem number_of_whole_numbers_between_sqrts :
  count_whole_numbers_between_sqrts = 7 :=
by
  sorry

end number_of_whole_numbers_between_sqrts_l277_277723


namespace intersection_A_B_eq_l277_277585

def A : Set ℝ := { x | (x / (x - 1)) ≥ 0 }

def B : Set ℝ := { y | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B_eq :
  (A ∩ B) = { y : ℝ | 1 < y } :=
sorry

end intersection_A_B_eq_l277_277585


namespace proposition_false_l277_277257

theorem proposition_false : ¬ ∀ x ∈ ({1, -1, 0} : Set ℤ), 2 * x + 1 > 0 := by
  sorry

end proposition_false_l277_277257


namespace find_k_when_root_is_zero_l277_277865

-- Define the quadratic equation and what it implies
theorem find_k_when_root_is_zero (k : ℝ) (h : (k-1) * 0^2 + 6 * 0 + k^2 - k = 0) :
  k = 0 :=
by
  -- The proof steps would go here, but we're skipping it as instructed
  sorry

end find_k_when_root_is_zero_l277_277865


namespace additional_machines_needed_l277_277662

theorem additional_machines_needed
  (machines : ℕ)
  (days : ℕ)
  (one_fourth_less_days : ℕ)
  (machine_days_total : ℕ)
  (machines_needed : ℕ)
  (additional_machines : ℕ) 
  (h1 : machines = 15) 
  (h2 : days = 36)
  (h3 : one_fourth_less_days = 27)
  (h4 : machine_days_total = machines * days)
  (h5 : machines_needed = machine_days_total / one_fourth_less_days) :
  additional_machines = machines_needed - machines → additional_machines = 5 :=
by
  admit -- sorry

end additional_machines_needed_l277_277662


namespace jensen_meetings_percentage_l277_277191

theorem jensen_meetings_percentage :
  ∃ (first second third total_work_day total_meeting_time : ℕ),
    total_work_day = 600 ∧
    first = 35 ∧
    second = 2 * first ∧
    third = first + second ∧
    total_meeting_time = first + second + third ∧
    (total_meeting_time * 100) / total_work_day = 35 := sorry

end jensen_meetings_percentage_l277_277191


namespace combine_piles_l277_277321

theorem combine_piles (n : ℕ) (piles : list ℕ) (h_piles : list.sum piles = n) (h_similar : ∀ x y ∈ piles, x ≤ y → y ≤ 2 * x) :
  ∃ pile, pile ∈ piles ∧ pile = n := sorry

end combine_piles_l277_277321


namespace all_three_pass_prob_at_least_one_pass_prob_l277_277376

section
variables {Ω : Type*} {P : ProbabilityMeasure Ω}
variables (A B C : Event Ω)

-- Hypotheses
-- Individual A passes with probability 0.8
-- Individual B passes with probability 0.6
-- Individual C passes with probability 0.5
-- The events are independent
hypothesis p_A : P A = 0.8
hypothesis p_B : P B = 0.6
hypothesis p_C : P C = 0.5
hypothesis ind_AB : Independent P A B
hypothesis ind_AC : Independent P A C
hypothesis ind_BC : Independent P B C

-- Probability that all three individuals pass the test
def prob_all_three_pass : ℝ := P (A ∩ B ∩ C)

-- Probability that at least one of the three individuals passes the test
def prob_at_least_one_pass : ℝ := P (A ∪ B ∪ C)

-- Statements
theorem all_three_pass_prob : prob_all_three_pass = 0.24 :=
begin
  sorry
end

theorem at_least_one_pass_prob : prob_at_least_one_pass = 0.96 :=
begin
  sorry
end

end

end all_three_pass_prob_at_least_one_pass_prob_l277_277376


namespace farmer_planning_problem_l277_277800

theorem farmer_planning_problem
  (A : ℕ) (D : ℕ)
  (h1 : A = 120 * D)
  (h2 : ∀ t : ℕ, t = 85 * (D + 5) + 40)
  (h3 : 85 * (D + 5) + 40 = 120 * D) : 
  A = 1560 ∧ D = 13 := 
by
  sorry

end farmer_planning_problem_l277_277800


namespace trader_profit_percentage_l277_277659

theorem trader_profit_percentage
  (P : ℝ)
  (h1 : P > 0)
  (buy_price : ℝ := 0.80 * P)
  (sell_price : ℝ := 1.60 * P) :
  (sell_price - P) / P * 100 = 60 := 
by sorry

end trader_profit_percentage_l277_277659


namespace clara_weight_l277_277927

theorem clara_weight (a c : ℝ) (h1 : a + c = 220) (h2 : c - a = c / 3) : c = 88 :=
by
  sorry

end clara_weight_l277_277927


namespace solve_for_x_l277_277786

theorem solve_for_x (x y : ℤ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 :=
by
  sorry

end solve_for_x_l277_277786


namespace percent_first_question_l277_277796

variable (A B : ℝ) (A_inter_B : ℝ) (A_union_B : ℝ)

-- Given conditions
def condition1 : B = 0.49 := sorry
def condition2 : A_inter_B = 0.32 := sorry
def condition3 : A_union_B = 0.80 := sorry
def union_formula : A_union_B = A + B - A_inter_B := 
by sorry

-- Prove that A = 0.63
theorem percent_first_question (h1 : B = 0.49) 
                               (h2 : A_inter_B = 0.32) 
                               (h3 : A_union_B = 0.80) 
                               (h4 : A_union_B = A + B - A_inter_B) : 
                               A = 0.63 :=
by sorry

end percent_first_question_l277_277796


namespace no_such_abc_exists_l277_277188

-- Define the conditions for the leading coefficients and constant terms
def leading_coeff_conditions (a b c : ℝ) : Prop :=
  ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0))

def constant_term_conditions (a b c : ℝ) : Prop :=
  ((c > 0 ∧ a < 0 ∧ b < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0) ∨ (b > 0 ∧ c < 0 ∧ a < 0))

-- The final statement that encapsulates the contradiction
theorem no_such_abc_exists : ¬ ∃ a b c : ℝ, leading_coeff_conditions a b c ∧ constant_term_conditions a b c :=
by
  sorry

end no_such_abc_exists_l277_277188


namespace determine_f_zero_l277_277609

variable (f : ℝ → ℝ)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 + y) + 4 * (f x) * y

theorem determine_f_zero (h1: functional_equation f)
    (h2 : f 2 = 4) : f 0 = 0 := 
sorry

end determine_f_zero_l277_277609


namespace triangle_angle_A_l277_277459

variable {a b c : ℝ} {A : ℝ}

theorem triangle_angle_A (h : a^2 = b^2 + c^2 - b * c) : A = 2 * Real.pi / 3 :=
by
  sorry

end triangle_angle_A_l277_277459


namespace distinct_zeros_arithmetic_geometric_sequence_l277_277909

theorem distinct_zeros_arithmetic_geometric_sequence 
  (a b p q : ℝ)
  (h1 : a ≠ b)
  (h2 : a + b = p)
  (h3 : ab = q)
  (h4 : p > 0)
  (h5 : q > 0)
  (h6 : (a = 4 ∧ b = 1) ∨ (a = 1 ∧ b = 4))
  : p + q = 9 := 
sorry

end distinct_zeros_arithmetic_geometric_sequence_l277_277909


namespace future_age_relation_l277_277236

-- Conditions
def son_present_age : ℕ := 8
def father_present_age : ℕ := 4 * son_present_age

-- Theorem statement
theorem future_age_relation : ∃ x : ℕ, 32 + x = 3 * (8 + x) ↔ x = 4 :=
by {
  sorry
}

end future_age_relation_l277_277236


namespace binomial_probability_eq_l277_277706

noncomputable def binomial_pmf (n : ℕ) (p : ℚ) : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.mixture (n+1) (λ k, Mathbin.binom n k • (p^k • (1 - p)^(n - k)))

theorem binomial_probability_eq
  (n : ℕ) (p : ℚ) (X : ℕ → MeasureTheory.Meas ℕ)
  (h1 : ProbabilityMassFunction.likelihood X = binomial_pmf n p)
  (h2 : (X.map ennreal.to_real).expected_value = 2)
  (h3 : (X.map ennreal.to_real).variance = 4 / 3) :
  ProbabilityMassFunction.probability X 2 = 80 / 243 :=
sorry

end binomial_probability_eq_l277_277706


namespace complex_quadrant_l277_277920

theorem complex_quadrant (z : ℂ) (h : z = (↑0 + 1*I) / (1 + 1*I)) : z.re > 0 ∧ z.im > 0 := 
by
  sorry

end complex_quadrant_l277_277920


namespace sqrt_expression_nonneg_l277_277233

theorem sqrt_expression_nonneg {b : ℝ} : b - 3 ≥ 0 ↔ b ≥ 3 := by
  sorry

end sqrt_expression_nonneg_l277_277233


namespace tenth_term_is_correct_l277_277782

-- Define the first term and common difference for the sequence
def a1 : ℚ := 1 / 2
def d : ℚ := 1 / 3

-- The property that defines the n-th term of the arithmetic sequence
def a (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Statement to prove that the tenth term in the arithmetic sequence is 7 / 2
theorem tenth_term_is_correct : a 10 = 7 / 2 := 
by 
  -- To be filled in with the proof later
  sorry

end tenth_term_is_correct_l277_277782


namespace commercials_per_hour_l277_277751

theorem commercials_per_hour (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : ∃ x : ℝ, x = (1 - p) * 60 := 
sorry

end commercials_per_hour_l277_277751


namespace VishalInvestedMoreThanTrishulBy10Percent_l277_277243

variables (R T V : ℝ)

-- Given conditions
def RaghuInvests (R : ℝ) : Prop := R = 2500
def TrishulInvests (R T : ℝ) : Prop := T = 0.9 * R
def TotalInvestment (R T V : ℝ) : Prop := V + T + R = 7225
def PercentageInvestedMore (T V : ℝ) (P : ℝ) : Prop := P * T = V - T

-- Main theorem to prove
theorem VishalInvestedMoreThanTrishulBy10Percent (R T V : ℝ) (P : ℝ) :
  RaghuInvests R ∧ TrishulInvests R T ∧ TotalInvestment R T V → PercentageInvestedMore T V P → P = 0.1 :=
by
  intros
  sorry

end VishalInvestedMoreThanTrishulBy10Percent_l277_277243


namespace circle_properties_l277_277566

noncomputable def circle_center (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x = 0

theorem circle_properties (x y : ℝ) :
  circle_center x y ↔ ((x - 2)^2 + y^2 = 2^2) ∧ ((2, 0) = (2, 0)) :=
by
  sorry

end circle_properties_l277_277566


namespace find_p_l277_277525

theorem find_p (m n p : ℝ)
  (h1 : m = 5 * n + 5)
  (h2 : m + 2 = 5 * (n + p) + 5) :
  p = 2 / 5 :=
by
  sorry

end find_p_l277_277525


namespace inverse_f_486_l277_277727

-- Define the function f with given properties.
def f : ℝ → ℝ := sorry

-- Condition 1: f(5) = 2
axiom f_at_5 : f 5 = 2

-- Condition 2: f(3x) = 3f(x) for all x
axiom f_scale : ∀ x, f (3 * x) = 3 * f x

-- Proposition: f⁻¹(486) = 1215
theorem inverse_f_486 : (∃ x, f x = 486) → ∀ x, f x = 486 → x = 1215 :=
by sorry

end inverse_f_486_l277_277727


namespace smallest_whole_number_l277_277248

theorem smallest_whole_number (m : ℕ) :
  m % 2 = 1 ∧
  m % 3 = 1 ∧
  m % 4 = 1 ∧
  m % 5 = 1 ∧
  m % 6 = 1 ∧
  m % 8 = 1 ∧
  m % 11 = 0 → 
  m = 1801 :=
by
  intros h
  sorry

end smallest_whole_number_l277_277248


namespace tile_rectangle_condition_l277_277038

theorem tile_rectangle_condition (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) : 
  (∃ q, m = k * q) ∨ (∃ r, n = k * r) :=
sorry

end tile_rectangle_condition_l277_277038


namespace jim_miles_remaining_l277_277104

theorem jim_miles_remaining (total_miles : ℕ) (miles_driven : ℕ) (total_miles_eq : total_miles = 1200) (miles_driven_eq : miles_driven = 384) :
  total_miles - miles_driven = 816 :=
by
  sorry

end jim_miles_remaining_l277_277104


namespace original_visual_range_l277_277797

theorem original_visual_range
  (V : ℝ)
  (h1 : 2.5 * V = 150) :
  V = 60 :=
by
  sorry

end original_visual_range_l277_277797


namespace original_number_is_fraction_l277_277746

theorem original_number_is_fraction (x : ℚ) (h : 1 + 1/x = 7/3) : x = 3/4 :=
sorry

end original_number_is_fraction_l277_277746


namespace regular_decagon_interior_angle_l277_277452

theorem regular_decagon_interior_angle {n : ℕ} (h1 : n = 10) (h2 : ∀ (k : ℕ), k = 10 → (180 * (k - 2)) / 10 = 144) : 
  (∃ θ : ℕ, θ = 180 * (n - 2) / n ∧ n = 10 ∧ θ = 144) :=
by
  sorry

end regular_decagon_interior_angle_l277_277452


namespace minimum_lines_for_regions_l277_277422

theorem minimum_lines_for_regions (n : ℕ) : 1 + n * (n + 1) / 2 ≥ 1000 ↔ n ≥ 45 :=
sorry

end minimum_lines_for_regions_l277_277422


namespace neg_exists_eq_forall_ne_l277_277230

variable (x : ℝ)

theorem neg_exists_eq_forall_ne : (¬ ∃ x : ℝ, x^2 - 2 * x = 0) ↔ ∀ x : ℝ, x^2 - 2 * x ≠ 0 := by
  sorry

end neg_exists_eq_forall_ne_l277_277230


namespace three_buses_interval_l277_277940

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l277_277940


namespace cristina_catches_nicky_l277_277330

-- Definitions from the conditions
def cristina_speed : ℝ := 4 -- meters per second
def nicky_speed : ℝ := 3 -- meters per second
def nicky_head_start : ℝ := 36 -- meters

-- The proof to find the time 't'
theorem cristina_catches_nicky (t : ℝ) : cristina_speed * t = nicky_head_start + nicky_speed * t -> t = 36 := by
  intros h
  sorry

end cristina_catches_nicky_l277_277330


namespace percent_correct_l277_277336

theorem percent_correct (x : ℕ) : 
  (5 * 100.0 / 7) = 71.43 :=
by
  sorry

end percent_correct_l277_277336


namespace lilibeth_and_friends_strawberries_l277_277327

-- Define the conditions
def baskets_filled_by_lilibeth : ℕ := 6
def strawberries_per_basket : ℕ := 50
def friends_count : ℕ := 3

-- Define the total number of strawberries picked by Lilibeth and her friends 
def total_strawberries_picked : ℕ :=
  (baskets_filled_by_lilibeth * strawberries_per_basket) * (1 + friends_count)

-- The theorem to prove
theorem lilibeth_and_friends_strawberries : total_strawberries_picked = 1200 := 
by
  sorry

end lilibeth_and_friends_strawberries_l277_277327


namespace pages_revised_twice_theorem_l277_277234

noncomputable def pages_revised_twice (total_pages : ℕ) (cost_per_page : ℕ) (revision_cost_per_page : ℕ) 
                                      (pages_revised_once : ℕ) (total_cost : ℕ) : ℕ :=
  let pages_revised_twice := (total_cost - (total_pages * cost_per_page + pages_revised_once * revision_cost_per_page)) 
                             / (revision_cost_per_page * 2)
  pages_revised_twice

theorem pages_revised_twice_theorem : 
  pages_revised_twice 100 10 5 30 1350 = 20 :=
by
  unfold pages_revised_twice
  norm_num

end pages_revised_twice_theorem_l277_277234


namespace quilt_cost_proof_l277_277471

-- Definitions for conditions
def length := 7
def width := 8
def cost_per_sq_foot := 40

-- Definition for the calculation of the area
def area := length * width

-- Definition for the calculation of the cost
def total_cost := area * cost_per_sq_foot

-- Theorem stating the final proof
theorem quilt_cost_proof : total_cost = 2240 := by
  sorry

end quilt_cost_proof_l277_277471


namespace value_of_a_purely_imaginary_l277_277174

-- Define the conditions under which a given complex number is purely imaginary
def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = Complex.im z * Complex.I ∧ b ≠ 0

-- Define the complex number based on the variable a
def given_complex_number (a : ℝ) : ℂ :=
  ⟨a^2 - 3*a + 2, a - 1⟩

-- The proof statement
theorem value_of_a_purely_imaginary :
  is_purely_imaginary (given_complex_number 2) := sorry

end value_of_a_purely_imaginary_l277_277174


namespace complement_A_eq_l277_277586

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1}

theorem complement_A_eq :
  U \ A = {0, 2} :=
by
  sorry

end complement_A_eq_l277_277586


namespace problem_1_problem_2_l277_277714

noncomputable def f (a x : ℝ) : ℝ := |x + a| + |x + 1/a|

theorem problem_1 (x : ℝ) : f 2 x > 3 ↔ x < -(11 / 4) ∨ x > 1 / 4 := sorry

theorem problem_2 (a m : ℝ) (ha : a > 0) : f a m + f a (-1 / m) ≥ 4 := sorry

end problem_1_problem_2_l277_277714


namespace adults_collectively_ate_l277_277879

theorem adults_collectively_ate (A : ℕ) (C : ℕ) (total_cookies : ℕ) (share : ℝ) (each_child_gets : ℕ)
  (hC : C = 4) (hTotal : total_cookies = 120) (hShare : share = 1/3) (hEachChild : each_child_gets = 20)
  (children_gets : ℕ) (hChildrenGets : children_gets = C * each_child_gets) :
  children_gets = (2/3 : ℝ) * total_cookies → (share : ℝ) * total_cookies = 40 :=
by
  -- Placeholder for simplified proof
  sorry

end adults_collectively_ate_l277_277879


namespace combined_perimeter_l277_277115

theorem combined_perimeter (side_square : ℝ) (a b c : ℝ) (diameter : ℝ) 
  (h_square : side_square = 7) 
  (h_triangle : a = 5 ∧ b = 6 ∧ c = 7) 
  (h_diameter : diameter = 4) : 
  4 * side_square + (a + b + c) + (2 * Real.pi * (diameter / 2) + diameter) = 50 + 2 * Real.pi := 
by 
  sorry

end combined_perimeter_l277_277115


namespace geometric_mean_of_means_l277_277063

open Finset

-- Define the geometric mean of a set
def geometric_mean (S : Finset ℝ) (h : ∀ x ∈ S, x > 0) : ℝ :=
  (S.prod id) ^ (1 / S.card)

-- Main theorem statement
theorem geometric_mean_of_means (S : Finset ℝ) (h : ∀ x ∈ S, x > 0) :
  geometric_mean S h = geometric_mean (S.powerset.filter (fun T => T ≠ ∅)).product (fun T => geometric_mean T sorry) sorry :=
sorry

end geometric_mean_of_means_l277_277063


namespace steve_marbles_l277_277341

-- Define the initial condition variables
variables (S Steve_initial Sam_initial Sally_initial Sarah_initial Steve_now : ℕ)

-- Conditions
def cond1 : Sam_initial = 2 * Steve_initial := by sorry
def cond2 : Sally_initial = Sam_initial - 5 := by sorry
def cond3 : Sarah_initial = Steve_initial + 3 := by sorry
def cond4 : Steve_now = Steve_initial + 3 := by sorry
def cond5 : Sam_initial - (3 + 3 + 4) = 6 := by sorry

-- Goal
theorem steve_marbles : Steve_now = 11 := by sorry

end steve_marbles_l277_277341


namespace passing_marks_l277_277971

variable (T P : ℝ)

-- condition 1: 0.30T = P - 30
def condition1 : Prop := 0.30 * T = P - 30

-- condition 2: 0.45T = P + 15
def condition2 : Prop := 0.45 * T = P + 15

-- Proof Statement: P = 120 (passing marks)
theorem passing_marks (T P : ℝ) (h1 : condition1 T P) (h2 : condition2 T P) : P = 120 := 
  sorry

end passing_marks_l277_277971


namespace equilateral_triangle_bound_l277_277576

theorem equilateral_triangle_bound (n k : ℕ) (h_n_gt_3 : n > 3) 
  (h_k_triangles : ∃ T : Finset (Finset (ℝ × ℝ)), T.card = k ∧ ∀ t ∈ T, 
  ∃ a b c : (ℝ × ℝ), t = {a, b, c} ∧ dist a b = 1 ∧ dist b c = 1 ∧ dist c a = 1) :
  k < (2 * n) / 3 :=
by
  sorry

end equilateral_triangle_bound_l277_277576


namespace value_of_M_l277_277081

theorem value_of_M (G A M E: ℕ) (hG : G = 15)
(hGAME : G + A + M + E = 50)
(hMEGA : M + E + G + A = 55)
(hAGE : A + G + E = 40) : 
M = 15 := sorry

end value_of_M_l277_277081


namespace compare_neg_fractions_and_neg_values_l277_277122

theorem compare_neg_fractions_and_neg_values :
  (- (3 : ℚ) / 4 > - (4 : ℚ) / 5) ∧ (-(-3 : ℤ) > -|(3 : ℤ)|) :=
by
  apply And.intro
  sorry
  sorry

end compare_neg_fractions_and_neg_values_l277_277122


namespace filling_time_calculation_l277_277400

namespace TankerFilling

-- Define the filling rates
def fill_rate_A : ℚ := 1 / 60
def fill_rate_B : ℚ := 1 / 40
def combined_fill_rate : ℚ := fill_rate_A + fill_rate_B

-- Define the time variable
variable (T : ℚ)

-- State the theorem to be proved
theorem filling_time_calculation
  (h_fill_rate_A : fill_rate_A = 1 / 60)
  (h_fill_rate_B : fill_rate_B = 1 / 40)
  (h_combined_fill_rate : combined_fill_rate = 1 / 24) :
  (fill_rate_B * (T / 2) + combined_fill_rate * (T / 2)) = 1 → T = 30 :=
by
  intros h
  -- Proof will go here
  sorry

end TankerFilling

end filling_time_calculation_l277_277400


namespace value_of_f_m_plus_one_depends_on_m_l277_277591

def f (x a : ℝ) : ℝ := x^2 - x + a

theorem value_of_f_m_plus_one_depends_on_m (m a : ℝ) (h : f (-m) a < 0) :
  (∃ m, f (m + 1) a < 0) ∧ (∃ m, f (m + 1) a > 0) :=
by
  sorry

end value_of_f_m_plus_one_depends_on_m_l277_277591


namespace linda_age_difference_l277_277328

/-- 
Linda is some more than 2 times the age of Jane.
In five years, the sum of their ages will be 28.
Linda's age at present is 13.
Prove that Linda's age is 3 years more than 2 times Jane's age.
-/
theorem linda_age_difference {L J : ℕ} (h1 : L = 13)
  (h2 : (L + 5) + (J + 5) = 28) : L - 2 * J = 3 :=
by sorry

end linda_age_difference_l277_277328


namespace Gwendolyn_will_take_50_hours_to_read_l277_277451

def GwendolynReadingTime (sentences_per_hour : ℕ) (sentences_per_paragraph : ℕ) (paragraphs_per_page : ℕ) (pages : ℕ) : ℕ :=
  (sentences_per_paragraph * paragraphs_per_page * pages) / sentences_per_hour

theorem Gwendolyn_will_take_50_hours_to_read 
  (h1 : 200 = 200)
  (h2 : 10 = 10)
  (h3 : 20 = 20)
  (h4 : 50 = 50) :
  GwendolynReadingTime 200 10 20 50 = 50 := by
  sorry

end Gwendolyn_will_take_50_hours_to_read_l277_277451


namespace sum_modulo_nine_l277_277145

theorem sum_modulo_nine :
  (88135 + 88136 + 88137 + 88138 + 88139 + 88140) % 9 = 3 := 
by
  sorry

end sum_modulo_nine_l277_277145


namespace snowballs_made_by_brother_l277_277605

/-- Janet makes 50 snowballs and her brother makes the remaining snowballs. Janet made 25% of the total snowballs. 
    Prove that her brother made 150 snowballs. -/
theorem snowballs_made_by_brother (total_snowballs : ℕ) (janet_snowballs : ℕ) (fraction_janet : ℚ)
  (h1 : janet_snowballs = 50) (h2 : fraction_janet = 25 / 100) (h3 : janet_snowballs = fraction_janet * total_snowballs) :
  total_snowballs - janet_snowballs = 150 :=
by
  sorry

end snowballs_made_by_brother_l277_277605


namespace tangent_circles_l277_277377

theorem tangent_circles (a b c : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 = a^2 → (x-b)^2 + (y-c)^2 = a^2) →
    ( (b^2 + c^2) / (a^2) = 4 ) :=
by
  intro h
  have h_dist : (b^2 + c^2) = (2 * a) ^ 2 := sorry
  have h_div : (b^2 + c^2) / (a^2) = 4 := sorry
  exact h_div

end tangent_circles_l277_277377


namespace rectangle_area_l277_277020

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l277_277020


namespace rabbit_toy_cost_l277_277891

theorem rabbit_toy_cost 
  (cost_pet_food : ℝ) 
  (cost_cage : ℝ) 
  (found_dollar : ℝ)
  (total_cost : ℝ) 
  (h1 : cost_pet_food = 5.79) 
  (h2 : cost_cage = 12.51)
  (h3 : found_dollar = 1.00)
  (h4 : total_cost = 24.81):
  ∃ (cost_rabbit_toy : ℝ), cost_rabbit_toy = 7.51 := by
  let cost_rabbit_toy := total_cost - (cost_pet_food + cost_cage) + found_dollar
  use cost_rabbit_toy
  sorry

end rabbit_toy_cost_l277_277891


namespace problem_1_problem_2_l277_277042

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | -4 ≤ x ∧ x < 2 }
def B : Set ℝ := { x | -1 < x ∧ x ≤ 3 }
def P : Set ℝ := { x | x ≤ 0 ∨ x ≥ 5 / 2 }

theorem problem_1 : A ∩ B = { x | -1 < x ∧ x < 2 } :=
sorry

theorem problem_2 : (U \ B) ∪ P = { x | x ≤ 0 ∨ x ≥ 5 / 2 } :=
sorry

end problem_1_problem_2_l277_277042


namespace complex_transformation_l277_277119

open Complex

theorem complex_transformation :
  let z := -1 + (7 : ℂ) * I
  let rotation := (1 / 2 + (Real.sqrt 3) / 2 * I)
  let dilation := 2
  (z * rotation * dilation = -22 - ((Real.sqrt 3) - 7) * I) :=
by
  sorry

end complex_transformation_l277_277119


namespace sara_lunch_total_cost_l277_277906

noncomputable def cost_hotdog : ℝ := 5.36
noncomputable def cost_salad : ℝ := 5.10
noncomputable def cost_soda : ℝ := 2.75
noncomputable def cost_fries : ℝ := 3.20
noncomputable def discount_rate : ℝ := 0.15
noncomputable def tax_rate : ℝ := 0.08

noncomputable def total_cost_before_discount_tax : ℝ :=
  cost_hotdog + cost_salad + cost_soda + cost_fries

noncomputable def discount : ℝ :=
  discount_rate * total_cost_before_discount_tax

noncomputable def discounted_total : ℝ :=
  total_cost_before_discount_tax - discount

noncomputable def tax : ℝ := 
  tax_rate * discounted_total

noncomputable def final_total : ℝ :=
  discounted_total + tax

theorem sara_lunch_total_cost : final_total = 15.07 :=
by
  sorry

end sara_lunch_total_cost_l277_277906


namespace inequality_proof_l277_277575

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : -b > 0) (h3 : a > -b) (h4 : c < 0) : 
  a * (1 - c) > b * (c - 1) :=
sorry

end inequality_proof_l277_277575


namespace cake_eaten_after_four_trips_l277_277656

-- Define the fraction of the cake eaten on each trip
def fraction_eaten (n : Nat) : ℚ :=
  (1 / 3) ^ n

-- Define the total cake eaten after four trips
def total_eaten_after_four_trips : ℚ :=
  fraction_eaten 1 + fraction_eaten 2 + fraction_eaten 3 + fraction_eaten 4

-- The mathematical statement we want to prove
theorem cake_eaten_after_four_trips : total_eaten_after_four_trips = 40 / 81 := 
by
  sorry

end cake_eaten_after_four_trips_l277_277656


namespace current_selling_price_is_correct_profit_per_unit_is_correct_l277_277535

variable (a : ℝ)

def original_selling_price (a : ℝ) : ℝ :=
  a * 1.22

def current_selling_price (a : ℝ) : ℝ :=
  original_selling_price a * 0.85

def profit_per_unit (a : ℝ) : ℝ :=
  current_selling_price a - a

theorem current_selling_price_is_correct : current_selling_price a = 1.037 * a :=
by
  unfold current_selling_price original_selling_price
  sorry

theorem profit_per_unit_is_correct : profit_per_unit a = 0.037 * a :=
by
  unfold profit_per_unit current_selling_price original_selling_price
  sorry

end current_selling_price_is_correct_profit_per_unit_is_correct_l277_277535


namespace radius_of_first_cylinder_l277_277264

theorem radius_of_first_cylinder :
  ∀ (rounds1 rounds2 : ℕ) (r2 r1 : ℝ), rounds1 = 70 → rounds2 = 49 → r2 = 20 → 
  (2 * Real.pi * r1 * rounds1 = 2 * Real.pi * r2 * rounds2) → r1 = 14 :=
by
  sorry

end radius_of_first_cylinder_l277_277264


namespace find_pairs_l277_277834

theorem find_pairs (a b : ℕ) :
  (∃ (a b : ℕ), (b^2 - a ≠ 0) ∧ (a^2 - b ≠ 0) ∧ (a^2 + b) / (b^2 - a) ∈ ℤ ∧ (b^2 + a) / (a^2 - b) ∈ ℤ) → 
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end find_pairs_l277_277834


namespace area_of_rectangle_ABCD_l277_277008

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l277_277008


namespace smallest_possible_positive_value_l277_277870

theorem smallest_possible_positive_value (a b : ℤ) (h : a > b) :
  ∃ (x : ℚ), x = (a + b) / (a - b) + (a - b) / (a + b) ∧ x = 2 :=
sorry

end smallest_possible_positive_value_l277_277870


namespace find_percentage_l277_277258

variable (P : ℝ)

/-- A number P% that satisfies the condition is 65. -/
theorem find_percentage (h : ((P / 100) * 40 = ((5 / 100) * 60) + 23)) : P = 65 :=
sorry

end find_percentage_l277_277258


namespace distinct_values_f_in_interval_l277_277637

noncomputable def f (x : ℝ) : ℤ :=
  ⌊x⌋ + ⌊2 * x⌋ + ⌊(5 * x) / 3⌋ + ⌊3 * x⌋ + ⌊4 * x⌋

theorem distinct_values_f_in_interval : 
  ∃ n : ℕ, n = 734 ∧ 
    ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 100 → 
      f x = f y → x = y :=
sorry

end distinct_values_f_in_interval_l277_277637


namespace speed_ratio_A_to_B_l277_277950

variables {u v : ℝ}

axiom perp_lines_intersect_at_o : true
axiom points_move_along_lines_at_constant_speed : true
axiom point_A_at_O_B_500_yards_away_at_t_0 : true
axiom after_2_minutes_A_and_B_equidistant : 2 * u = 500 - 2 * v
axiom after_10_minutes_A_and_B_equidistant : 10 * u = 10 * v - 500

theorem speed_ratio_A_to_B : u / v = 2 / 3 :=
by 
  sorry

end speed_ratio_A_to_B_l277_277950


namespace greatest_int_multiple_of_9_remainder_l277_277477

theorem greatest_int_multiple_of_9_remainder():
  exists (M : ℕ), (∀ d₁ d₂ : ℤ, d₁ ≠ d₂ → d₁ ∈ digits 10 M → d₂ ∈ digits 10 M) ∧
                (9 ∣ M) ∧
                (∀ N : ℕ, (∀ d₁ d₂ : ℤ, d₁ ≠ d₂ → d₁ ∈ digits 10 N → d₂ ∈ digits 10 N) →
                          (9 ∣ N) → N ≤ M) ∧
                (M % 1000 = 963) := 
by {
  sorry
}

end greatest_int_multiple_of_9_remainder_l277_277477


namespace greatest_integer_b_l277_277567

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem greatest_integer_b (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b:ℝ) * x + 12 ≠ 0) ↔ b = 6 := 
by
  sorry

end greatest_integer_b_l277_277567


namespace sin2θ_value_l277_277578

theorem sin2θ_value (θ : Real) (h1 : Real.sin θ = 4/5) (h2 : Real.sin θ - Real.cos θ > 1) : Real.sin (2*θ) = -24/25 := 
by 
  sorry

end sin2θ_value_l277_277578


namespace pizza_payment_l277_277899

theorem pizza_payment (n : ℕ) (cost : ℕ) (total : ℕ) 
  (h1 : n = 3) 
  (h2 : cost = 8) 
  (h3 : total = n * cost) : 
  total = 24 :=
by 
  rw [h1, h2] at h3 
  exact h3

end pizza_payment_l277_277899


namespace triangles_with_vertex_A_l277_277050

theorem triangles_with_vertex_A : 
  ∃ (A : Point) (remaining_points : Finset Point), 
    (remaining_points.card = 8) → 
    (∃ (n : ℕ), n = (Nat.choose 8 2) ∧ n = 28) :=
by
  sorry

end triangles_with_vertex_A_l277_277050


namespace calculate_expression_l277_277415

variable (x y : ℝ)

theorem calculate_expression : (-2 * x^2 * y) ^ 2 = 4 * x^4 * y^2 := by
  sorry

end calculate_expression_l277_277415


namespace evaluate_expression_l277_277134

theorem evaluate_expression :
  let x := (1/4 : ℚ)
  let y := (4/5 : ℚ)
  let z := (-2 : ℚ)
  x^3 * y^2 * z^2 = 1/25 :=
by
  sorry

end evaluate_expression_l277_277134


namespace quilt_cost_l277_277473

theorem quilt_cost :
  let length := 7
  let width := 8
  let cost_per_sq_ft := 40
  let area := length * width
  let total_cost := area * cost_per_sq_ft
  total_cost = 2240 :=
by
  sorry

end quilt_cost_l277_277473


namespace carriage_problem_l277_277601

theorem carriage_problem (x : ℕ) : 
  3 * (x - 2) = 2 * x + 9 := 
sorry

end carriage_problem_l277_277601


namespace smallest_prime_less_than_perf_square_l277_277777

-- Define a predicate for being a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

-- The main goal
theorem smallest_prime_less_than_perf_square : ∃ n : ℕ, is_prime n ∧ ∃ m : ℕ, n = m^2 - 8 ∧ (∀ k : ℕ, is_prime k ∧ ∃ l : ℕ, k = l^2 - 8 → k ≥ n) :=
begin
  use 17,
  split,
  -- Proof that 17 is a prime number
  {
    unfold is_prime,
    split,
    { exact dec_trivial },
    { intros d hd,
      have h_d : d = 1 ∨ d = 17,
      { cases d,
        { exfalso, linarith, },
        { cases d,
          { left, refl, },
          { right, linarith [Nat.Prime.not_dvd_one 17 hd], }, }, },
      exact h_d, },
  },
  -- Proof that 17 is 8 less than a perfect square and the smallest such prime
  {
    use 5,
    split,
    { refl, },
    { intros k hk,
      cases hk with hk_prime hk_cond,
      cases hk_cond with l hl,
      rw hl,
      have : l ≥ 5,
      { intros,
        linarith, },
      exact this, },
  }
end

end smallest_prime_less_than_perf_square_l277_277777


namespace complete_the_square_l277_277098

theorem complete_the_square (m n : ℕ) :
  (∀ x : ℝ, x^2 - 6 * x = 1 → (x - m)^2 = n) → m + n = 13 :=
by
  sorry

end complete_the_square_l277_277098


namespace solve_equation_l277_277146

theorem solve_equation (x : ℝ) :
  (16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 60 → x = 4 := by
  sorry

end solve_equation_l277_277146


namespace length_of_AB_l277_277603

noncomputable def AB_CD_sum_240 (AB CD : ℝ) (h : ℝ) : Prop :=
  AB + CD = 240

noncomputable def ratio_of_areas (AB CD : ℝ) : Prop :=
  AB / CD = 5 / 3

theorem length_of_AB (AB CD : ℝ) (h : ℝ) (h_ratio : ratio_of_areas AB CD) (h_sum : AB_CD_sum_240 AB CD h) : AB = 150 :=
by
  unfold ratio_of_areas at h_ratio
  unfold AB_CD_sum_240 at h_sum
  sorry

end length_of_AB_l277_277603


namespace tony_lego_sets_l277_277738

theorem tony_lego_sets
  (price_lego price_sword price_dough : ℕ)
  (num_sword num_dough total_cost : ℕ)
  (L : ℕ)
  (h1 : price_lego = 250)
  (h2 : price_sword = 120)
  (h3 : price_dough = 35)
  (h4 : num_sword = 7)
  (h5 : num_dough = 10)
  (h6 : total_cost = 1940)
  (h7 : total_cost = price_lego * L + price_sword * num_sword + price_dough * num_dough) :
  L = 3 := 
by
  sorry

end tony_lego_sets_l277_277738


namespace fixed_monthly_fee_l277_277820

/-
  We want to prove that given two conditions:
  1. x + y = 12.48
  2. x + 2y = 17.54
  The fixed monthly fee (x) is 7.42.
-/

theorem fixed_monthly_fee (x y : ℝ) 
  (h1 : x + y = 12.48) 
  (h2 : x + 2 * y = 17.54) : 
  x = 7.42 := 
sorry

end fixed_monthly_fee_l277_277820


namespace problem_D_l277_277205

variable (f : ℕ → ℝ)

-- Function condition: If f(k) ≥ k^2, then f(k+1) ≥ (k+1)^2
axiom f_property (k : ℕ) (hk : f k ≥ k^2) : f (k + 1) ≥ (k + 1)^2

theorem problem_D (hf4 : f 4 ≥ 25) : ∀ k ≥ 4, f k ≥ k^2 :=
by
  sorry

end problem_D_l277_277205


namespace mean_temperature_l277_277504

theorem mean_temperature
  (temps : List ℤ) 
  (h_temps : temps = [-8, -3, -7, -6, 0, 4, 6, 5, -1, 2]) :
  (temps.sum: ℚ) / temps.length = -0.8 := 
by
  sorry

end mean_temperature_l277_277504


namespace binom_15_4_l277_277549

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l277_277549


namespace distinct_real_roots_find_p_l277_277582

theorem distinct_real_roots (p : ℝ) : 
  let f := (fun x => (x - 3) * (x - 2) - p^2)
  let Δ := 1 + 4 * p ^ 2 
  0 < Δ :=
by sorry

theorem find_p (x1 x2 p : ℝ) : 
  (x1 + x2 = 5) → 
  (x1 * x2 = 6 - p^2) → 
  (x1^2 + x2^2 = 3 * x1 * x2) → 
  (p = 1 ∨ p = -1) :=
by sorry

end distinct_real_roots_find_p_l277_277582


namespace find_number_added_l277_277522

theorem find_number_added (x : ℕ) : (1250 / 50) + x = 7525 ↔ x = 7500 := by
  sorry

end find_number_added_l277_277522


namespace halfway_fraction_l277_277850

-- Assume a definition for the two fractions
def fracA : ℚ := 1 / 4
def fracB : ℚ := 1 / 7

-- Define the target property we want to prove
theorem halfway_fraction : (fracA + fracB) / 2 = 11 / 56 := 
by 
  -- Proof will happen here, adding sorry to indicate it's skipped for now
  sorry

end halfway_fraction_l277_277850


namespace abc_gt_16_abc_geq_3125_div_108_l277_277340

variables {a b c α β : ℝ}

-- Define the conditions
def conditions (a b c α β : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b > 0 ∧
  (a * α^2 + b * α - c = 0) ∧
  (a * β^2 + b * β - c = 0) ∧
  (α ≠ β) ∧
  (α^3 + b * α^2 + a * α - c = 0) ∧
  (β^3 + b * β^2 + a * β - c = 0)

-- State the first proof problem
theorem abc_gt_16 (h : conditions a b c α β) : a * b * c > 16 :=
sorry

-- State the second proof problem
theorem abc_geq_3125_div_108 (h : conditions a b c α β) : a * b * c ≥ 3125 / 108 :=
sorry

end abc_gt_16_abc_geq_3125_div_108_l277_277340


namespace verify_integer_pairs_l277_277835

open Nat

theorem verify_integer_pairs (a b : ℕ) :
  (∃ k1 : ℤ, ↑(a^2) + ↑b = k1 * (↑(b^2) - ↑a)) ∧
  (∃ k2 : ℤ, ↑(b^2) + ↑a = k2 * (↑(a^2) - ↑b)) →
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ 
  (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end verify_integer_pairs_l277_277835


namespace quadratic_no_real_roots_l277_277875

theorem quadratic_no_real_roots (c : ℝ) : (∀ x : ℝ, x^2 + 2 * x + c ≠ 0) → c > 1 :=
by
  sorry

end quadratic_no_real_roots_l277_277875


namespace A_and_C_amount_l277_277117

variables (A B C : ℝ)

def amounts_satisfy_conditions : Prop :=
  (A + B + C = 500) ∧ (B + C = 320) ∧ (C = 20)

theorem A_and_C_amount (h : amounts_satisfy_conditions A B C) : A + C = 200 :=
by {
  sorry
}

end A_and_C_amount_l277_277117


namespace competition_results_correct_l277_277335

theorem competition_results_correct :
  ∃ (first second third fourth : String), 
    (first = "Oleg" ∧ second = "Olya" ∧ third = "Polya" ∧ fourth = "Pasha") ∧
    ∀ (claims : String → String → Prop),
      (claims "Olya" "all_odd_places_boys") ∧ 
      (claims "Oleg" "consecutive_places_with_olya") ∧
      (claims "Pasha" "all_odd_places_names_start_O") ∧
      ∃ (truth_teller : String), 
        truth_teller = "Oleg" ∧ 
        (claims "Oleg" "first_place") ∧ 
        ¬ (claims "Olya" "first_place") ∧ 
        ¬ (claims "Pasha" "first_place") ∧ 
        ¬ (claims "Polya" "first_place") :=
sorry

end competition_results_correct_l277_277335


namespace units_digit_of_product_of_first_three_positive_composite_numbers_l277_277956

theorem units_digit_of_product_of_first_three_positive_composite_numbers :
  (4 * 6 * 8) % 10 = 2 :=
by sorry

end units_digit_of_product_of_first_three_positive_composite_numbers_l277_277956


namespace adam_apples_count_l277_277542

variable (Jackie_apples : ℕ)
variable (extra_apples : ℕ)
variable (Adam_apples : ℕ)

theorem adam_apples_count (h1 : Jackie_apples = 9) (h2 : extra_apples = 5) (h3 : Adam_apples = Jackie_apples + extra_apples) :
  Adam_apples = 14 := 
by 
  sorry

end adam_apples_count_l277_277542


namespace impossible_to_repaint_white_l277_277349

-- Define the board as a 7x7 grid 
def boardSize : ℕ := 7

-- Define the initial coloring function (checkerboard with corners black)
def initialColor (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

-- Define the repainting operation allowed
def repaint (cell1 cell2 : (ℕ × ℕ)) (color1 color2 : Prop) : Prop :=
  ¬color1 = color2 

-- Define the main theorem to prove
theorem impossible_to_repaint_white :
  ¬(∃ f : ℕ × ℕ -> Prop, 
    (∀ i j, (i < boardSize) → (j < boardSize) → (f (i, j) = true)) ∧ 
    (∀ i j, (i < boardSize - 1) → (repaint (i, j) (i, j+1) (f (i, j)) (f (i, j+1))) ∧
             (i < boardSize - 1) → (repaint (i, j) (i+1, j) (f (i, j)) (f (i+1, j)))))
  :=
  sorry

end impossible_to_repaint_white_l277_277349


namespace remainder_3001_3005_mod_17_l277_277093

theorem remainder_3001_3005_mod_17 :
  ((3001 % 17) * (3002 % 17) * (3003 % 17) * (3004 % 17) * (3005 % 17)) % 17 = 2 := by
  have h1 : 3001 % 17 = 10 := by norm_num
  have h2 : 3002 % 17 = 11 := by norm_num
  have h3 : 3003 % 17 = 12 := by norm_num
  have h4 : 3004 % 17 = 13 := by norm_num
  have h5 : 3005 % 17 = 14 := by norm_num
  calc
    ((3001 % 17) * (3002 % 17) * (3003 % 17) * (3004 % 17) * (3005 % 17)) % 17
      = (10 * 11 * 12 * 13 * 14) % 17 : by rw [h1, h2, h3, h4, h5]
    ... = 2 : by norm_num

end remainder_3001_3005_mod_17_l277_277093


namespace find_d_minus_c_l277_277917

theorem find_d_minus_c (c d x : ℝ) (h : c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d) : (d - c = 45) :=
  sorry

end find_d_minus_c_l277_277917


namespace inequality_proof_l277_277765

variables {a b : ℝ}

theorem inequality_proof :
  a^2 + b^2 - 1 - a^2 * b^2 <= 0 ↔ (a^2 - 1) * (b^2 - 1) >= 0 :=
by sorry

end inequality_proof_l277_277765


namespace pyramid_side_length_l277_277632

noncomputable def side_length_of_square_base (area_of_lateral_face : ℝ) (slant_height : ℝ) : ℝ :=
  2 * area_of_lateral_face / slant_height

theorem pyramid_side_length 
  (area_of_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_of_lateral_face = 120)
  (h2 : slant_height = 24) :
  side_length_of_square_base area_of_lateral_face slant_height = 10 :=
by
  -- Skipping the proof details.
  sorry

end pyramid_side_length_l277_277632


namespace length_of_PS_l277_277031

-- Define the problem conditions directly
variables (P Q R S : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]
           (PQ : Real) (PR : Real) (cosP : Real)
           (angleP : ℝ)

-- Assume the given conditions
variables (h1 : PQ = 4) (h2 : PR = 8) (h3 : cosP = 1/4)
           (PQR_is_triangle : Triangle P Q R) (PS_bisects_angle_P : Bisects PS angleP)

-- The goal is to prove the specified length of PS
theorem length_of_PS : PQ = 4 → PR = 8 → cosP = 1 / 4 → ∃ PS : ℝ, PS = 4 := by
  sorry

end length_of_PS_l277_277031


namespace piles_to_single_pile_l277_277319

-- Define the condition similar_sizes
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

-- Define the inductive step of combining stones
def combine_stones (piles : List ℕ) : List ℕ :=
  if ∃ x y, x ∈ piles ∧ y ∈ piles ∧ similar_sizes x y then
    let ⟨x, hx, y, hy, hsim⟩ := Classical.some_spec (Classical.some_spec_exists _)
    List.cons (x + y) (List.erase (List.erase piles x) y)
  else
    piles

-- Prove that a collection of piles can be reduced to a single pile of size n
theorem piles_to_single_pile (piles : List ℕ) (h : ∀ x ∈ piles, x = 1) : 
  ∃ p, list.length (Iterator.iterate combine_stones piles.count) 1 = 1 := by
  sorry

end piles_to_single_pile_l277_277319


namespace y_coordinate_equidistant_l277_277090

theorem y_coordinate_equidistant : ∃ y : ℝ, (∀ A B : ℝ × ℝ, A = (-3, 0) → B = (-2, 5) → dist (0, y) A = dist (0, y) B) ∧ y = 2 :=
by
  sorry

end y_coordinate_equidistant_l277_277090


namespace residue_neg_998_mod_28_l277_277689

theorem residue_neg_998_mod_28 : ∃ r : ℤ, r = -998 % 28 ∧ 0 ≤ r ∧ r < 28 ∧ r = 10 := 
by sorry

end residue_neg_998_mod_28_l277_277689


namespace find_pairs_l277_277838

theorem find_pairs (a b : ℕ) : 
  (∃ (a b : ℕ), 
    (∃ (k₁ k₂ : ℤ), 
      a^2 + b = k₁ * (b^2 - a) ∧ b^2 + a = k₂ * (a^2 - b))) 
      ↔ (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) := sorry

end find_pairs_l277_277838


namespace inequality_solution_solution_set_l277_277152

noncomputable def f (x a : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

theorem inequality_solution (a : ℝ) : 
  f 1 a > 0 ↔ 3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3 :=
by sorry

theorem solution_set (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 3 → f x a > b) ∧ (∃ x, -1 ≤ x ∧ x ≤ 3 ∧ f x a = b) ↔ 
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ b = -3 :=
by sorry

end inequality_solution_solution_set_l277_277152


namespace problem_1_problem_2_l277_277425

-- Definition of the operation ⊕
def my_oplus (a b : ℚ) : ℚ := (a + 3 * b) / 2

-- Prove that 4(2 ⊕ 5) = 34
theorem problem_1 : 4 * my_oplus 2 5 = 34 := 
by sorry

-- Definitions of A and B
def A (x y : ℚ) : ℚ := x^2 + 2 * x * y + y^2
def B (x y : ℚ) : ℚ := -2 * x * y + y^2

-- Prove that (A ⊕ B) + (B ⊕ A) = 2x^2 + 4y^2
theorem problem_2 (x y : ℚ) : 
  my_oplus (A x y) (B x y) + my_oplus (B x y) (A x y) = 2 * x^2 + 4 * y^2 := 
by sorry

end problem_1_problem_2_l277_277425


namespace calculate_expression_l277_277819

variables (x y : ℝ)

theorem calculate_expression (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x - y) / (Real.sqrt x + Real.sqrt y) - (x - 2 * Real.sqrt (x * y) + y) / (Real.sqrt x - Real.sqrt y) = 0 :=
by
  sorry

end calculate_expression_l277_277819


namespace derivative_f_at_1_l277_277871

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * Real.sin x

theorem derivative_f_at_1 : (deriv f 1) = 2 + 2 * Real.cos 1 := 
sorry

end derivative_f_at_1_l277_277871


namespace pyramid_base_sidelength_l277_277634

theorem pyramid_base_sidelength (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120) (hh : h = 24) (area_eq : A = 1/2 * s * h) : s = 10 := by
  sorry

end pyramid_base_sidelength_l277_277634


namespace neg_sin_prop_iff_l277_277448

theorem neg_sin_prop_iff :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by sorry

end neg_sin_prop_iff_l277_277448


namespace favouring_more_than_one_is_39_l277_277598

def percentage_favouring_more_than_one (x : ℝ) : Prop :=
  let sum_two : ℝ := 8 + 6 + 4 + 2 + 7 + 5 + 3 + 5 + 3 + 2
  let sum_three : ℝ := 1 + 0.5 + 0.3 + 0.8 + 0.2 + 0.1 + 1.5 + 0.7 + 0.3 + 0.4
  let all_five : ℝ := 0.2
  x = sum_two - sum_three - all_five

theorem favouring_more_than_one_is_39 : percentage_favouring_more_than_one 39 := 
by
  sorry

end favouring_more_than_one_is_39_l277_277598


namespace system_solution_l277_277066

noncomputable def solve_system (x y : ℝ) : Prop :=
  (x + y = 20) ∧ (Real.logBase 4 x + Real.logBase 4 y = 1 + Real.logBase 4 9) ∧
  ((x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18))

theorem system_solution : ∃ x y : ℝ, solve_system x y :=
  sorry

end system_solution_l277_277066


namespace opposite_of_neg_11_l277_277506

-- Define the opposite (negative) of a number
def opposite (a : ℤ) : ℤ := -a

-- Prove that the opposite of -11 is 11
theorem opposite_of_neg_11 : opposite (-11) = 11 := 
by
  -- Proof not required, so using sorry as placeholder
  sorry

end opposite_of_neg_11_l277_277506


namespace piles_can_be_combined_l277_277320

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end piles_can_be_combined_l277_277320


namespace unique_decomposition_of_two_reciprocals_l277_277441

theorem unique_decomposition_of_two_reciprocals (p : ℕ) (hp : Nat.Prime p) (hp_ne_two : p ≠ 2) :
  ∃ (x y : ℕ), x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 2 / (p : ℝ)) := sorry

end unique_decomposition_of_two_reciprocals_l277_277441


namespace joe_paint_usage_l277_277384

theorem joe_paint_usage :
  let initial_paint := 360
  let first_week_usage := (1 / 3: ℝ) * initial_paint
  let remaining_after_first_week := initial_paint - first_week_usage
  let second_week_usage := (1 / 5: ℝ) * remaining_after_first_week
  let total_usage := first_week_usage + second_week_usage
  total_usage = 168 :=
by
  sorry

end joe_paint_usage_l277_277384


namespace sophie_one_dollar_bills_l277_277502

theorem sophie_one_dollar_bills (x y z : ℕ) 
  (h1 : x + y + z = 55) 
  (h2 : x + 2 * y + 5 * z = 126) 
  : x = 18 := by
  sorry

end sophie_one_dollar_bills_l277_277502


namespace nat_pairs_satisfy_conditions_l277_277842

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end nat_pairs_satisfy_conditions_l277_277842


namespace polygon_interior_angle_eq_l277_277815

theorem polygon_interior_angle_eq (n : ℕ) (h : ∀ i, 1 ≤ i → i ≤ n → (interior_angle : ℝ) = 108) : n = 5 := 
sorry

end polygon_interior_angle_eq_l277_277815


namespace sin_is_odd_and_has_zero_point_l277_277982

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def has_zero_point (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

theorem sin_is_odd_and_has_zero_point :
  is_odd_function sin ∧ has_zero_point sin := 
  by sorry

end sin_is_odd_and_has_zero_point_l277_277982


namespace tangent_line_sin_at_pi_l277_277849

theorem tangent_line_sin_at_pi :
  ∀ (f : ℝ → ℝ), 
    (∀ x, f x = Real.sin x) → ∀ x y, (x, y) = (Real.pi, 0) → 
    ∃ (m : ℝ) (b : ℝ), (∀ x, y = m * x + b) ∧ (m = -1) ∧ (b = Real.pi) :=
by
  sorry

end tangent_line_sin_at_pi_l277_277849


namespace amount_per_person_l277_277792

theorem amount_per_person (total_amount : ℕ) (num_persons : ℕ) (amount_each : ℕ)
  (h1 : total_amount = 42900) (h2 : num_persons = 22) (h3 : amount_each = 1950) :
  total_amount / num_persons = amount_each :=
by
  -- Proof to be filled
  sorry

end amount_per_person_l277_277792


namespace cos_2alpha_plus_5pi_by_12_l277_277156

open Real

noncomputable def alpha : ℝ := sorry

axiom alpha_obtuse : π / 2 < alpha ∧ alpha < π

axiom sin_alpha_plus_pi_by_3 : sin (alpha + π / 3) = -4 / 5

theorem cos_2alpha_plus_5pi_by_12 : 
  cos (2 * alpha + 5 * π / 12) = 17 * sqrt 2 / 50 :=
by sorry

end cos_2alpha_plus_5pi_by_12_l277_277156


namespace sum_of_numbers_l277_277235

theorem sum_of_numbers (x : ℕ) (first_num second_num third_num sum : ℕ) 
  (h1 : 5 * x = first_num) 
  (h2 : 3 * x = second_num)
  (h3 : 4 * x = third_num) 
  (h4 : second_num = 27)
  : first_num + second_num + third_num = 108 :=
by {
  sorry
}

end sum_of_numbers_l277_277235


namespace relationship_abc_l277_277157

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.exp (-Real.pi)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_abc : b < a ∧ a < c :=
by
  -- proofs would be added here
  sorry

end relationship_abc_l277_277157


namespace quadratic_eq_l277_277581

noncomputable def roots (r s : ℝ): Prop := r + s = 12 ∧ r * s = 27 ∧ (r = 2 * s ∨ s = 2 * r)

theorem quadratic_eq (r s : ℝ) (h : roots r s) : 
   Polynomial.C 1 * (X^2 - Polynomial.C (r + s) * X + Polynomial.C (r * s)) = X ^ 2 - 12 * X + 27 := 
sorry

end quadratic_eq_l277_277581


namespace max_value_of_y_over_x_l277_277728

theorem max_value_of_y_over_x {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 2 * y = 3) :
  y / x ≤ 9 / 8 :=
sorry

end max_value_of_y_over_x_l277_277728


namespace Andrena_more_than_Debelyn_l277_277554

-- Definitions based on the problem conditions
def Debelyn_initial := 20
def Debelyn_gift_to_Andrena := 2
def Christel_initial := 24
def Christel_gift_to_Andrena := 5
def Andrena_more_than_Christel := 2

-- Calculating the number of dolls each person has after the gifts
def Debelyn_final := Debelyn_initial - Debelyn_gift_to_Andrena
def Christel_final := Christel_initial - Christel_gift_to_Andrena
def Andrena_final := Christel_final + Andrena_more_than_Christel

-- The proof problem statement
theorem Andrena_more_than_Debelyn : Andrena_final - Debelyn_final = 3 := by
  sorry

end Andrena_more_than_Debelyn_l277_277554


namespace proof_M_inter_N_eq_01_l277_277300
open Set

theorem proof_M_inter_N_eq_01 :
  let M := {x : ℤ | x^2 = x}
  let N := {-1, 0, 1}
  M ∩ N = {0, 1} := by
  sorry

end proof_M_inter_N_eq_01_l277_277300


namespace race_times_l277_277387

theorem race_times (x y : ℕ) (h1 : 5 * x + 1 = 4 * y) (h2 : 5 * y - 8 = 4 * x) :
  5 * x = 15 ∧ 5 * y = 20 :=
by
  sorry

end race_times_l277_277387


namespace triangle_pentagon_side_ratio_l277_277118

theorem triangle_pentagon_side_ratio (triangle_perimeter : ℕ) (pentagon_perimeter : ℕ) 
  (h1 : triangle_perimeter = 60) (h2 : pentagon_perimeter = 60) :
  (triangle_perimeter / 3 : ℚ) / (pentagon_perimeter / 5 : ℚ) = 5 / 3 :=
by {
  sorry
}

end triangle_pentagon_side_ratio_l277_277118


namespace negation_of_proposition_l277_277643

theorem negation_of_proposition :
  ¬ (∃ x_0 : ℤ, 2 * x_0 + x_0 + 1 ≤ 0) ↔ ∀ x : ℤ, 2 * x + x + 1 > 0 :=
by sorry

end negation_of_proposition_l277_277643


namespace smallest_prime_8_less_than_square_l277_277779

theorem smallest_prime_8_less_than_square :
  ∃ p : ℕ, (∃ n : ℤ, p = n^2 - 8) ∧ Nat.Prime p ∧ p > 0 ∧ (∀ q : ℕ, (∃ m : ℤ, q = m^2 - 8) ∧ Nat.Prime q → q ≥ p) :=
sorry

end smallest_prime_8_less_than_square_l277_277779


namespace hyperbola_eccentricity_l277_277711

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (e : ℝ) (h3 : e = (Real.sqrt 3) / 2) 
  (h4 : a ^ 2 = b ^ 2 + (Real.sqrt 3) ^ 2) : (Real.sqrt 5) / 2 = 
    (Real.sqrt (a ^ 2 + b ^ 2)) / a :=
by
  sorry

end hyperbola_eccentricity_l277_277711


namespace jerky_remaining_after_giving_half_l277_277466

-- Define the main conditions as variables
def days := 5
def initial_jerky := 40
def jerky_per_day := 1 + 1 + 2

-- Calculate total consumption
def total_consumption := jerky_per_day * days

-- Calculate remaining jerky
def remaining_jerky := initial_jerky - total_consumption

-- Calculate final jerky after giving half to her brother
def jerky_left := remaining_jerky / 2

-- Statement to be proved
theorem jerky_remaining_after_giving_half :
  jerky_left = 10 :=
by
  -- Proof will go here
  sorry

end jerky_remaining_after_giving_half_l277_277466


namespace area_of_rectangle_l277_277027

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l277_277027


namespace counterfeit_probability_l277_277825

open Finset

theorem counterfeit_probability :
  let A := 5.choose 2 / 20.choose 2
  let B := (5.choose 2 + (5.choose 1 * 15.choose 1)) / 20.choose 2
  P(A | B) = (5.choose 2) / (5.choose 2 + 5.choose 1 * 15.choose 1) := 
by
  sorry

end counterfeit_probability_l277_277825


namespace sin_identity_l277_277859

variable (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = Real.sqrt 3 / 2)

theorem sin_identity : Real.sin (3 * Real.pi / 4 - α) = Real.sqrt 3 / 2 := by
  sorry

end sin_identity_l277_277859


namespace book_pages_l277_277033

noncomputable def totalPages := 240

theorem book_pages : 
  ∀ P : ℕ, 
    (1 / 2) * P + (1 / 4) * P + (1 / 6) * P + 20 = P → 
    P = totalPages :=
by
  intro P
  intros h
  sorry

end book_pages_l277_277033


namespace seven_fifths_of_fraction_l277_277428

theorem seven_fifths_of_fraction :
  (7 / 5) * (-18 / 4) = -63 / 10 :=
by
  sorry

end seven_fifths_of_fraction_l277_277428


namespace ryan_flyers_l277_277280

theorem ryan_flyers (total_flyers : ℕ) (alyssa_flyers : ℕ) (scott_flyers : ℕ) (belinda_percentage : ℚ) (belinda_flyers : ℕ) (ryan_flyers : ℕ)
  (htotal : total_flyers = 200)
  (halyssa : alyssa_flyers = 67)
  (hscott : scott_flyers = 51)
  (hbelinda_percentage : belinda_percentage = 0.20)
  (hbelinda : belinda_flyers = belinda_percentage * total_flyers)
  (hryan : ryan_flyers = total_flyers - (alyssa_flyers + scott_flyers + belinda_flyers)) :
  ryan_flyers = 42 := by
    sorry

end ryan_flyers_l277_277280


namespace number_of_digits_of_n_l277_277610

theorem number_of_digits_of_n :
  ∃ n : ℕ,
    (n > 0) ∧ 
    (15 ∣ n) ∧ 
    (∃ m : ℕ, n^2 = m^4) ∧ 
    (∃ k : ℕ, n^4 = k^2) ∧ 
    Nat.digits 10 n = 5 :=
by
  sorry

end number_of_digits_of_n_l277_277610


namespace painting_perimeter_l277_277051

-- Definitions for the problem conditions
def frame_thickness : ℕ := 3
def frame_area : ℕ := 108

-- Declaration that expresses the given conditions and the problem's conclusion
theorem painting_perimeter {w h : ℕ} (h_frame : (w + 2 * frame_thickness) * (h + 2 * frame_thickness) - w * h = frame_area) :
  2 * (w + h) = 24 :=
by
  sorry

end painting_perimeter_l277_277051


namespace pipe_fills_tank_without_leak_l277_277677

theorem pipe_fills_tank_without_leak (T : ℝ) (h1 : 1 / 6 = 1 / T - 1 / 12) : T = 4 :=
by
  sorry

end pipe_fills_tank_without_leak_l277_277677


namespace least_possible_number_l277_277403

theorem least_possible_number (k : ℕ) (n : ℕ) (r : ℕ) (h1 : k = 34 * n + r) 
  (h2 : k / 5 = r + 8) (h3 : r < 34) : k = 68 :=
by
  -- Proof to be filled
  sorry

end least_possible_number_l277_277403


namespace smallest_prime_8_less_than_square_l277_277778

theorem smallest_prime_8_less_than_square :
  ∃ p : ℕ, (∃ n : ℤ, p = n^2 - 8) ∧ Nat.Prime p ∧ p > 0 ∧ (∀ q : ℕ, (∃ m : ℤ, q = m^2 - 8) ∧ Nat.Prime q → q ≥ p) :=
sorry

end smallest_prime_8_less_than_square_l277_277778


namespace only_option_d_determines_location_l277_277383

-- Define the problem conditions in Lean
inductive LocationOption where
  | OptionA : LocationOption
  | OptionB : LocationOption
  | OptionC : LocationOption
  | OptionD : LocationOption

-- Define a function that takes a LocationOption and returns whether it can determine a specific location
def determine_location (option : LocationOption) : Prop :=
  match option with
  | LocationOption.OptionD => True
  | LocationOption.OptionA => False
  | LocationOption.OptionB => False
  | LocationOption.OptionC => False

-- Prove that only option D can determine a specific location
theorem only_option_d_determines_location : ∀ (opt : LocationOption), determine_location opt ↔ opt = LocationOption.OptionD := by
  intro opt
  cases opt
  · simp [determine_location, LocationOption.OptionA]
  · simp [determine_location, LocationOption.OptionB]
  · simp [determine_location, LocationOption.OptionC]
  · simp [determine_location, LocationOption.OptionD]

end only_option_d_determines_location_l277_277383


namespace size_of_each_bottle_l277_277314

-- Defining given conditions
def petals_per_ounce : ℕ := 320
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes : ℕ := 800
def bottles : ℕ := 20

-- Proving the size of each bottle in ounces
theorem size_of_each_bottle : (petals_per_rose * roses_per_bush * bushes / petals_per_ounce) / bottles = 12 := by
  sorry

end size_of_each_bottle_l277_277314


namespace max_gas_tank_capacity_l277_277655

-- Definitions based on conditions
def start_gas : ℕ := 10
def gas_used_store : ℕ := 6
def gas_used_doctor : ℕ := 2
def refill_needed : ℕ := 10

-- Theorem statement based on the equivalence proof problem
theorem max_gas_tank_capacity : 
  start_gas - (gas_used_store + gas_used_doctor) + refill_needed = 12 :=
by
  -- Proof steps go here
  sorry

end max_gas_tank_capacity_l277_277655


namespace weighted_average_of_angles_l277_277737

def triangle_inequality (a b c α β γ : ℝ) : Prop :=
  (a - b) * (α - β) ≥ 0 ∧ (b - c) * (β - γ) ≥ 0 ∧ (a - c) * (α - γ) ≥ 0

noncomputable def angle_sum (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi

theorem weighted_average_of_angles (a b c α β γ : ℝ)
  (h1 : triangle_inequality a b c α β γ)
  (h2 : angle_sum α β γ) :
  Real.pi / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < Real.pi / 2 :=
by
  sorry

end weighted_average_of_angles_l277_277737


namespace total_students_in_class_l277_277210

def current_students : ℕ := 6 * 3
def students_bathroom : ℕ := 5
def students_canteen : ℕ := 5 * 5
def group1_students : ℕ := 4
def group2_students : ℕ := 5
def group3_students : ℕ := 6
def group4_students : ℕ := 3
def new_group_students : ℕ := group1_students + group2_students + group3_students + group4_students
def germany_students : ℕ := 3
def france_students : ℕ := 4
def norway_students : ℕ := 3
def italy_students : ℕ := 2
def spain_students : ℕ := 2
def australia_students : ℕ := 1
def foreign_exchange_students : ℕ :=
  germany_students + france_students + norway_students + italy_students + spain_students + australia_students

def total_students : ℕ :=
  current_students + students_bathroom + students_canteen + new_group_students + foreign_exchange_students

theorem total_students_in_class : total_students = 81 := by
  rfl  -- Reflective equality since total_students already sums to 81 based on the definitions

end total_students_in_class_l277_277210


namespace Christina_driving_time_l277_277417

theorem Christina_driving_time 
  (speed_Christina : ℕ) 
  (speed_friend : ℕ) 
  (total_distance : ℕ)
  (friend_driving_time : ℕ) 
  (distance_by_Christina : ℕ) 
  (time_driven_by_Christina : ℕ) 
  (total_driving_time : ℕ)
  (h1 : speed_Christina = 30)
  (h2 : speed_friend = 40) 
  (h3 : total_distance = 210)
  (h4 : friend_driving_time = 3)
  (h5 : speed_friend * friend_driving_time = 120)
  (h6 : total_distance - 120 = distance_by_Christina)
  (h7 : distance_by_Christina = 90)
  (h8 : distance_by_Christina / speed_Christina = 3)
  (h9 : time_driven_by_Christina = 3)
  (h10 : time_driven_by_Christina * 60 = 180) :
    total_driving_time = 180 := 
by
  sorry

end Christina_driving_time_l277_277417


namespace units_cost_l277_277933

theorem units_cost (x y z : ℝ) 
  (h1 : 3 * x + 7 * y + z = 3.15)
  (h2 : 4 * x + 10 * y + z = 4.20) : 
  x + y + z = 1.05 :=
by 
  sorry

end units_cost_l277_277933


namespace speed_of_woman_in_still_water_l277_277980

noncomputable def V_w : ℝ := 5
variable (V_s : ℝ)

-- Conditions:
def downstream_condition : Prop := (V_w + V_s) * 6 = 54
def upstream_condition : Prop := (V_w - V_s) * 6 = 6

theorem speed_of_woman_in_still_water 
    (h1 : downstream_condition V_s) 
    (h2 : upstream_condition V_s) : 
    V_w = 5 :=
by
    -- Proof omitted
    sorry

end speed_of_woman_in_still_water_l277_277980


namespace simplify_sqrt_360000_l277_277497

-- Define the given conditions
def factorization : 360000 = 3600 * 100 := rfl
def sqrt_3600 : Real.sqrt 3600 = 60 := by norm_num
def sqrt_100 : Real.sqrt 100 = 10 := by norm_num

-- Define the main statement to be proved
theorem simplify_sqrt_360000 : Real.sqrt 360000 = 600 :=
by
  rw [factorization, Real.sqrt_mul', sqrt_3600, sqrt_100]
  norm_num

end simplify_sqrt_360000_l277_277497


namespace least_number_to_multiply_l277_277641

theorem least_number_to_multiply (x : ℕ) :
  (72 * x) % 112 = 0 → x = 14 :=
by 
  sorry

end least_number_to_multiply_l277_277641


namespace sally_spent_total_l277_277491

section SallySpending

def peaches : ℝ := 12.32
def cherries : ℝ := 11.54
def total_spent : ℝ := peaches + cherries

theorem sally_spent_total :
  total_spent = 23.86 := by
  sorry

end SallySpending

end sally_spent_total_l277_277491


namespace open_box_volume_l277_277973

-- Define the initial conditions
def length_of_sheet := 100
def width_of_sheet := 50
def height_of_parallelogram := 10
def base_of_parallelogram := 10

-- Define the expected dimensions of the box after cutting
def length_of_box := length_of_sheet - 2 * base_of_parallelogram
def width_of_box := width_of_sheet - 2 * base_of_parallelogram
def height_of_box := height_of_parallelogram

-- Define the expected volume of the box
def volume_of_box := length_of_box * width_of_box * height_of_box

-- Theorem to prove the correct volume of the box based on the given dimensions
theorem open_box_volume : volume_of_box = 24000 := by
  -- The proof will be included here
  sorry

end open_box_volume_l277_277973


namespace percentage_altered_votes_got_is_50_l277_277572

def original_votes_got := 10
def original_votes_twilight := 12
def original_votes_art_of_deal := 20

def votes_after_tampering :=
  original_votes_got +
  (original_votes_twilight / 2) +
  (original_votes_art_of_deal * 0.2)

def percentage_votes_got :=
  (original_votes_got / votes_after_tampering) * 100

theorem percentage_altered_votes_got_is_50 :
  percentage_votes_got = 50 := by
  sorry

end percentage_altered_votes_got_is_50_l277_277572


namespace polynomial_constant_term_q_l277_277039

theorem polynomial_constant_term_q (p q r : Polynomial ℚ)
  (h1 : r = p * q)
  (hp_const : p.eval 0 = 5)
  (hr_const : r.eval 0 = -15) :
  q.eval 0 = -3 :=
sorry

end polynomial_constant_term_q_l277_277039


namespace arccos_gt_arctan_on_interval_l277_277139

noncomputable def c : ℝ := sorry -- placeholder for the numerical solution of arccos x = arctan x

theorem arccos_gt_arctan_on_interval (x : ℝ) (hx : -1 ≤ x ∧ x < c) :
  Real.arccos x > Real.arctan x := 
sorry

end arccos_gt_arctan_on_interval_l277_277139


namespace range_of_f_l277_277788

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) + 2

theorem range_of_f (h : ∀ x : ℝ, x ≤ 1) : (f '' {x : ℝ | x ≤ 1}) = {y : ℝ | 1 ≤ y ∧ y ≤ 2} :=
by
  sorry

end range_of_f_l277_277788


namespace percent_full_time_more_than_three_years_l277_277184

variable (total_associates : ℕ)
variable (second_year_percentage : ℕ)
variable (third_year_percentage : ℕ)
variable (non_first_year_percentage : ℕ)
variable (part_time_percentage : ℕ)
variable (part_time_more_than_two_years_percentage : ℕ)
variable (full_time_more_than_three_years_percentage : ℕ)

axiom condition_1 : second_year_percentage = 30
axiom condition_2 : third_year_percentage = 20
axiom condition_3 : non_first_year_percentage = 60
axiom condition_4 : part_time_percentage = 10
axiom condition_5 : part_time_more_than_two_years_percentage = 5

theorem percent_full_time_more_than_three_years : 
  full_time_more_than_three_years_percentage = 10 := 
sorry

end percent_full_time_more_than_three_years_l277_277184


namespace points_subtracted_per_wrong_answer_l277_277346

theorem points_subtracted_per_wrong_answer 
  (total_problems : ℕ) 
  (wrong_answers : ℕ) 
  (score : ℕ) 
  (points_per_right_answer : ℕ) 
  (correct_answers : ℕ)
  (subtracted_points : ℕ) 
  (expected_points : ℕ) 
  (points_subtracted : ℕ) :
  total_problems = 25 → 
  wrong_answers = 3 → 
  score = 85 → 
  points_per_right_answer = 4 → 
  correct_answers = total_problems - wrong_answers → 
  expected_points = correct_answers * points_per_right_answer → 
  subtracted_points = expected_points - score → 
  points_subtracted = subtracted_points / wrong_answers → 
  points_subtracted = 1 := 
by
  intros;
  sorry

end points_subtracted_per_wrong_answer_l277_277346


namespace polynomial_factors_integers_l277_277143

theorem polynomial_factors_integers (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 500)
  (h₃ : ∃ a : ℤ, n = a * (a + 1)) :
  n ≤ 21 :=
sorry

end polynomial_factors_integers_l277_277143


namespace min_red_hair_students_l277_277669

variable (B N R : ℕ)
variable (total_students : ℕ := 50)

theorem min_red_hair_students :
  B + N + R = total_students →
  (∀ i, B > i → N > 0) →
  (∀ i, N > i → R > 0) →
  R ≥ 17 :=
by {
  -- The specifics of the proof are omitted as per the instruction
  sorry
}

end min_red_hair_students_l277_277669


namespace minimum_value_MP_MF_l277_277709

noncomputable def min_value (M P : ℝ × ℝ) (F : ℝ × ℝ) : ℝ := |dist M P + dist M F|

theorem minimum_value_MP_MF :
  ∀ (M : ℝ × ℝ), (M.2 ^ 2 = 4 * M.1) →
  ∀ (F : ℝ × ℝ), (F = (1, 0)) →
  ∀ (P : ℝ × ℝ), (P = (3, 1)) →
  min_value M P F = 4 :=
by
  intros M h_para F h_focus P h_fixed
  rw [min_value]
  sorry

end minimum_value_MP_MF_l277_277709


namespace cube_volume_l277_277915

theorem cube_volume (d : ℝ) (s : ℝ) (h : d = 3 * Real.sqrt 3) (h_s : s * Real.sqrt 3 = d) : s ^ 3 = 27 := by
  -- Assuming h: the formula for the given space diagonal
  -- Assuming h_s: the formula connecting side length and the space diagonal
  sorry

end cube_volume_l277_277915


namespace area_of_ABCD_l277_277002

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l277_277002


namespace inequality_solution_l277_277065

theorem inequality_solution (x : ℝ) : (x + 3) / 2 - (5 * x - 1) / 5 ≥ 0 ↔ x ≤ 17 / 5 :=
by
  sorry

end inequality_solution_l277_277065


namespace greatest_integer_solution_l277_277952

theorem greatest_integer_solution :
  ∃ x : ℤ, (∃ (k : ℤ), (8 : ℚ) / 11 > k / 15 ∧ k = 10) ∧ x = 10 :=
by {
  sorry
}

end greatest_integer_solution_l277_277952


namespace two_discounts_l277_277364

theorem two_discounts (p : ℝ) : (0.9 * 0.9 * p) = 0.81 * p :=
by
  sorry

end two_discounts_l277_277364


namespace exponential_inequality_l277_277089

theorem exponential_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 :=
sorry

end exponential_inequality_l277_277089


namespace initial_eggs_count_l277_277931

theorem initial_eggs_count (harry_adds : ℕ) (total_eggs : ℕ) (initial_eggs : ℕ) :
  harry_adds = 5 → total_eggs = 52 → initial_eggs = total_eggs - harry_adds → initial_eggs = 47 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end initial_eggs_count_l277_277931


namespace probability_scoring_less_than_8_l277_277976

theorem probability_scoring_less_than_8 
  (P10 P9 P8 : ℝ) 
  (hP10 : P10 = 0.3) 
  (hP9 : P9 = 0.3) 
  (hP8 : P8 = 0.2) : 
  1 - (P10 + P9 + P8) = 0.2 := 
by 
  sorry

end probability_scoring_less_than_8_l277_277976


namespace sphere_radius_eq_three_of_volume_eq_surface_area_l277_277299

theorem sphere_radius_eq_three_of_volume_eq_surface_area
  (r : ℝ) 
  (h1 : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : 
  r = 3 :=
sorry

end sphere_radius_eq_three_of_volume_eq_surface_area_l277_277299


namespace speed_of_A_l277_277390

theorem speed_of_A (B_speed : ℕ) (crossings : ℕ) (H : B_speed = 3 ∧ crossings = 5 ∧ 5 * (1 / (x + B_speed)) = 1) : x = 2 :=
by
  sorry

end speed_of_A_l277_277390


namespace optimal_strategies_and_value_l277_277088

-- Define the payoff matrix for the two-player zero-sum game
def payoff_matrix : Matrix (Fin 2) (Fin 2) ℕ := ![![12, 22], ![32, 2]]

-- Define the optimal mixed strategies for both players
def optimal_strategy_row_player : Fin 2 → ℚ
| 0 => 3 / 4
| 1 => 1 / 4

def optimal_strategy_column_player : Fin 2 → ℚ
| 0 => 1 / 2
| 1 => 1 / 2

-- Define the value of the game
def value_of_game := (17 : ℚ)

theorem optimal_strategies_and_value :
  (∀ i j, (optimal_strategy_row_player 0 * payoff_matrix 0 j + optimal_strategy_row_player 1 * payoff_matrix 1 j = value_of_game) ∧
           (optimal_strategy_column_player 0 * payoff_matrix i 0 + optimal_strategy_column_player 1 * payoff_matrix i 1 = value_of_game)) :=
by 
  -- sorry is used as a placeholder for the proof
  sorry

end optimal_strategies_and_value_l277_277088


namespace total_questions_in_test_l277_277882

theorem total_questions_in_test :
  ∃ x, (5 * x = total_questions) ∧ 
       (20 : ℚ) / total_questions > (60 / 100 : ℚ) ∧ 
       (20 : ℚ) / total_questions < (70 / 100 : ℚ) ∧ 
       total_questions = 30 :=
by
  sorry

end total_questions_in_test_l277_277882


namespace neg_univ_prop_l277_277642

-- Translate the original mathematical statement to a Lean 4 statement.
theorem neg_univ_prop :
  (¬(∀ x : ℝ, x^2 ≠ x)) ↔ (∃ x : ℝ, x^2 = x) :=
by
  sorry

end neg_univ_prop_l277_277642


namespace length_in_scientific_notation_l277_277587

theorem length_in_scientific_notation : (161000 : ℝ) = 1.61 * 10^5 := 
by 
  -- Placeholder proof
  sorry

end length_in_scientific_notation_l277_277587


namespace prob_both_calligraphy_is_correct_prob_one_each_is_correct_l277_277975

section ProbabilityOfVolunteerSelection

variable (C P : ℕ) -- C = number of calligraphy competition winners, P = number of painting competition winners
variable (total_pairs : ℕ := 6 * (6 - 1) / 2) -- Number of ways to choose 2 out of 6 participants, binomial coefficient (6 choose 2)

-- Condition variables
def num_calligraphy_winners : ℕ := 4
def num_painting_winners : ℕ := 2
def num_total_winners : ℕ := num_calligraphy_winners + num_painting_winners

-- Number of pairs of both calligraphy winners
def pairs_both_calligraphy : ℕ := 4 * (4 - 1) / 2
-- Number of pairs of one calligraphy and one painting winner
def pairs_one_each : ℕ := 4 * 2

-- Probability calculations
def prob_both_calligraphy : ℚ := pairs_both_calligraphy / total_pairs
def prob_one_each : ℚ := pairs_one_each / total_pairs

-- Theorem statements to prove the probabilities of selected types of volunteers
theorem prob_both_calligraphy_is_correct : 
  prob_both_calligraphy = 2/5 := sorry

theorem prob_one_each_is_correct : 
  prob_one_each = 8/15 := sorry

end ProbabilityOfVolunteerSelection

end prob_both_calligraphy_is_correct_prob_one_each_is_correct_l277_277975


namespace complete_the_square_l277_277099

theorem complete_the_square (m n : ℕ) :
  (∀ x : ℝ, x^2 - 6 * x = 1 → (x - m)^2 = n) → m + n = 13 :=
by
  sorry

end complete_the_square_l277_277099


namespace find_c_l277_277225

/-- Define the conditions given in the problem --/
def parabola_equation (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def vertex_condition (a b c : ℝ) : Prop := 
  ∀ x, parabola_equation a b c x = a * (x - 3)^2 - 1

def passes_through_point (a b c : ℝ) : Prop := 
  parabola_equation a b c 1 = 5

/-- The main statement -/
theorem find_c (a b c : ℝ) 
  (h_vertex : vertex_condition a b c) 
  (h_point : passes_through_point a b c) :
  c = 12.5 :=
sorry

end find_c_l277_277225


namespace find_point_B_l277_277450

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def point_A : ℝ × ℝ := (-3, -1)
def line_y_eq_2x (x : ℝ) : ℝ × ℝ := (x, 2 * x)
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1 

theorem find_point_B (B : ℝ × ℝ) (hB : B = line_y_eq_2x B.1) (h_parallel : is_parallel (B.1 + 3, B.2 + 1) vector_a) :
  B = (2, 4) := 
  sorry

end find_point_B_l277_277450


namespace factorization_correct_l277_277993

theorem factorization_correct : 
  ∀ x : ℝ, (x^2 + 1) * (x^3 - x^2 + x - 1) = (x^2 + 1)^2 * (x - 1) :=
by
  intros
  sorry

end factorization_correct_l277_277993


namespace sqrt_simplification_l277_277498

theorem sqrt_simplification : Real.sqrt 360000 = 600 :=
by 
  sorry

end sqrt_simplification_l277_277498


namespace car_value_proof_l277_277034

-- Let's define the variables and the conditions.
def car_sold_value : ℝ := 20000
def sticker_price_new_car : ℝ := 30000
def percent_sold : ℝ := 0.80
def percent_paid : ℝ := 0.90
def out_of_pocket : ℝ := 11000

theorem car_value_proof :
  (percent_paid * sticker_price_new_car - percent_sold * car_sold_value = out_of_pocket) →
  car_sold_value = 20000 := 
by
  intros h
  -- Introduction of any intermediate steps if necessary should just invoke the sorry to indicate the need for proof later
  exact sorry

end car_value_proof_l277_277034


namespace garden_roller_length_l277_277912

theorem garden_roller_length
  (diameter : ℝ)
  (total_area : ℝ)
  (revolutions : ℕ)
  (pi : ℝ)
  (circumference : ℝ)
  (area_per_revolution : ℝ)
  (length : ℝ)
  (h1 : diameter = 1.4)
  (h2 : total_area = 44)
  (h3 : revolutions = 5)
  (h4 : pi = (22 / 7))
  (h5 : circumference = pi * diameter)
  (h6 : area_per_revolution = total_area / (revolutions : ℝ))
  (h7 : area_per_revolution = circumference * length) :
  length = 7 := by
  sorry

end garden_roller_length_l277_277912


namespace probability_greater_than_n_l277_277883

theorem probability_greater_than_n (n : ℕ) : 
  (1 ≤ n ∧ n ≤ 5) → (∃ k, k = 6 - n - 1 ∧ k / 6 = 1 / 2) → n = 3 := 
by sorry

end probability_greater_than_n_l277_277883


namespace avg_tickets_per_member_is_66_l277_277397

-- Definitions based on the problem's conditions
def avg_female_tickets : ℕ := 70
def male_to_female_ratio : ℕ := 2
def avg_male_tickets : ℕ := 58

-- Let the number of male members be M and number of female members be F
variables (M : ℕ) (F : ℕ)
def num_female_members : ℕ := male_to_female_ratio * M

-- Total tickets sold by males
def total_male_tickets : ℕ := avg_male_tickets * M

-- Total tickets sold by females
def total_female_tickets : ℕ := avg_female_tickets * num_female_members M

-- Total tickets sold by all members
def total_tickets_sold : ℕ := total_male_tickets M + total_female_tickets M

-- Total number of members
def total_members : ℕ := M + num_female_members M

-- Statement to prove: the average number of tickets sold per member is 66
theorem avg_tickets_per_member_is_66 : total_tickets_sold M / total_members M = 66 :=
by 
  sorry

end avg_tickets_per_member_is_66_l277_277397


namespace exists_root_in_interval_l277_277277

open Real

theorem exists_root_in_interval : ∃ x, 1.1 < x ∧ x < 1.2 ∧ (x^2 + 12*x - 15 = 0) :=
by {
  let f := λ x : ℝ, x^2 + 12*x - 15,
  have h1 : f 1.1 = -0.59 :=  sorry,
  have h2 : f 1.2 = 0.84 := sorry,
  have sign_change : (f 1.1) * (f 1.2) < 0,
  { rw [h1, h2], linarith, },
  exact exists_has_deriv_at_eq_zero (by norm_num1) (by norm_num1) (by linarith)
}

end exists_root_in_interval_l277_277277


namespace solve_for_t_l277_277316

theorem solve_for_t (t : ℝ) (h1 : 60 * t + 80 * ((10 : ℝ)/3 - t) = 220) 
  (h2 : 0 ≤ t) : 60 * t + 80 * ((10 : ℝ)/3 - t) = 220 :=
by
  sorry

end solve_for_t_l277_277316


namespace three_person_subcommittees_l277_277720

theorem three_person_subcommittees (n k : ℕ) (h1 : n = 8) (h2 : k = 3) : nat.choose n k = 56 := 
by
  rw [h1, h2]
  norm_num
  sorry

end three_person_subcommittees_l277_277720


namespace minimum_value_of_expression_l277_277154

noncomputable def min_expression_value (a b : ℝ) : ℝ :=
  1 / (1 + a) + 4 / (2 + b)

theorem minimum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + 3 * b = 7) : 
  min_expression_value a b ≥ (13 + 4 * Real.sqrt 3) / 14 :=
by
  sorry

end minimum_value_of_expression_l277_277154


namespace min_red_hair_students_l277_277668

variable (B N R : ℕ)
variable (total_students : ℕ := 50)

theorem min_red_hair_students :
  B + N + R = total_students →
  (∀ i, B > i → N > 0) →
  (∀ i, N > i → R > 0) →
  R ≥ 17 :=
by {
  -- The specifics of the proof are omitted as per the instruction
  sorry
}

end min_red_hair_students_l277_277668


namespace find_fourth_term_l277_277862

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)
variable (a_1 a_4 d : ℕ)

-- Conditions
axiom sum_first_5 : S_n 5 = 35
axiom sum_first_9 : S_n 9 = 117
axiom sum_closed_form_first_5 : 5 * a_1 + (5 * (5 - 1)) / 2 * d = 35
axiom sum_closed_form_first_9 : 9 * a_1 + (9 * (9 - 1)) / 2 * d = 117
axiom nth_term_closed_form : ∀ n, a_n n = a_1 + (n-1)*d

-- Target
theorem find_fourth_term : a_4 = 10 := by
  sorry

end find_fourth_term_l277_277862


namespace arccos_gt_arctan_l277_277140

open Real

theorem arccos_gt_arctan (x : ℝ) (hx : x ∈ set.Ico (-1 : ℝ) 1) : arccos x > arctan x :=
sorry

end arccos_gt_arctan_l277_277140


namespace biology_marks_l277_277283

theorem biology_marks 
  (e m p c : ℤ) 
  (avg : ℚ) 
  (marks_biology : ℤ)
  (h1 : e = 70) 
  (h2 : m = 63) 
  (h3 : p = 80)
  (h4 : c = 63)
  (h5 : avg = 68.2) 
  (h6 : avg * 5 = (e + m + p + c + marks_biology)) : 
  marks_biology = 65 :=
sorry

end biology_marks_l277_277283


namespace interval_with_three_buses_l277_277936

theorem interval_with_three_buses (interval_two_buses : ℕ) (total_route_time : ℕ) (bus_count : ℕ) : 
  interval_two_buses = 21 → total_route_time = 2 * interval_two_buses → bus_count = 3 → 
  (total_route_time / bus_count) = 14 :=
by
  intros h1 h2 h3
  rw [h1, h3, ← h2]
  simp
  sorry

end interval_with_three_buses_l277_277936


namespace sqrt_of_360000_l277_277500

theorem sqrt_of_360000 : sqrt 360000 = 600 := by
  sorry

end sqrt_of_360000_l277_277500


namespace smallest_x_for_non_prime_expression_l277_277520

/-- The smallest positive integer x for which x^2 + x + 41 is not a prime number is 40. -/
theorem smallest_x_for_non_prime_expression : ∃ x : ℕ, x > 0 ∧ x^2 + x + 41 = 41 * 41 ∧ (∀ y : ℕ, 0 < y ∧ y < x → Prime (y^2 + y + 41)) := 
sorry

end smallest_x_for_non_prime_expression_l277_277520


namespace true_inverse_negation_l277_277421

theorem true_inverse_negation : ∀ (α β : ℝ),
  (α = β) ↔ (α = β) := 
sorry

end true_inverse_negation_l277_277421


namespace probability_at_least_half_girls_l277_277617

noncomputable def binomial (n k : ℕ) : ℚ := (nat.choose n k : ℚ)

theorem probability_at_least_half_girls :
  let p_girl := 0.52
  let p_boy := 0.48
  let n := 7
  let p_4 := binomial n 4 * (p_girl)^4 * (p_boy)^3
  let p_5 := binomial n 5 * (p_girl)^5 * (p_boy)^2
  let p_6 := binomial n 6 * (p_girl)^6 * (p_boy)^1
  let p_7 := binomial n 7 * (p_girl)^7 * (p_boy)^0
  p_4 + p_5 + p_6 + p_7 ≈ 0.98872 := sorry

end probability_at_least_half_girls_l277_277617


namespace evening_customers_l277_277532

-- Define the conditions
def matinee_price : ℕ := 5
def evening_price : ℕ := 7
def opening_night_price : ℕ := 10
def popcorn_price : ℕ := 10
def num_matinee_customers : ℕ := 32
def num_opening_night_customers : ℕ := 58
def total_revenue : ℕ := 1670

-- Define the number of evening customers as a variable
variable (E : ℕ)

-- Prove that the number of evening customers E equals 40 given the conditions
theorem evening_customers :
  5 * num_matinee_customers +
  7 * E +
  10 * num_opening_night_customers +
  10 * (num_matinee_customers + E + num_opening_night_customers) / 2 = total_revenue
  → E = 40 :=
by
  intro h
  sorry

end evening_customers_l277_277532


namespace total_weight_of_nuts_l277_277798

theorem total_weight_of_nuts:
  let almonds := 0.14
  let pecans := 0.38
  let walnuts := 0.22
  let cashews := 0.47
  let pistachios := 0.29
  almonds + pecans + walnuts + cashews + pistachios = 1.50 :=
by
  sorry

end total_weight_of_nuts_l277_277798


namespace paul_erasers_l277_277900

theorem paul_erasers (E : ℕ) (E_crayons : E + 353 = 391) : E = 38 := 
by
  sorry

end paul_erasers_l277_277900


namespace rectangle_area_l277_277017

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l277_277017


namespace angle_sum_in_triangle_l277_277179

theorem angle_sum_in_triangle (A B C : ℝ) (h₁ : A + B = 90) (h₂ : A + B + C = 180) : C = 90 := by
  sorry

end angle_sum_in_triangle_l277_277179


namespace pennies_on_friday_l277_277218

-- Define the initial number of pennies and the function for doubling
def initial_pennies : Nat := 3
def double (n : Nat) : Nat := 2 * n

-- Prove the number of pennies on Friday
theorem pennies_on_friday : double (double (double (double initial_pennies))) = 48 := by
  sorry

end pennies_on_friday_l277_277218


namespace construct_circle_feasible_l277_277911

theorem construct_circle_feasible (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : b^2 > (a^2 + c^2) / 2) :
  ∃ x y d : ℝ, 
  d > 0 ∧ 
  (d / 2)^2 = y^2 + (a / 2)^2 ∧ 
  (d / 2)^2 = (y - x)^2 + (b / 2)^2 ∧ 
  (d / 2)^2 = (y - 2 * x)^2 + (c / 2)^2 :=
sorry

end construct_circle_feasible_l277_277911


namespace number_of_good_weeks_l277_277166

-- Definitions from conditions
def tough_week_sales : ℕ := 800
def good_week_sales : ℕ := 2 * tough_week_sales
def tough_weeks : ℕ := 3
def total_money_made : ℕ := 10400
def total_tough_week_sales : ℕ := tough_weeks * tough_week_sales
def total_good_week_sales : ℕ := total_money_made - total_tough_week_sales

-- Question to be proven
theorem number_of_good_weeks (G : ℕ) : 
  (total_good_week_sales = G * good_week_sales) → G = 5 := by
  sorry

end number_of_good_weeks_l277_277166


namespace opposite_of_neg_abs_is_positive_two_l277_277231

theorem opposite_of_neg_abs_is_positive_two : -(abs (-2)) = -2 :=
by sorry

end opposite_of_neg_abs_is_positive_two_l277_277231


namespace all_points_below_line_l277_277292

theorem all_points_below_line (a b : ℝ) (n : ℕ) (x y : ℕ → ℝ)
  (h1 : b > a)
  (h2 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → x k = a + ((k : ℝ) * (b - a) / (n + 1)))
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k = a * (b / a) ^ (k / (n + 1))) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k < x k := 
sorry

end all_points_below_line_l277_277292


namespace expected_male_teachers_in_sample_l277_277150

theorem expected_male_teachers_in_sample 
  (total_male total_female sample_size : ℕ) 
  (h1 : total_male = 56) 
  (h2 : total_female = 42) 
  (h3 : sample_size = 14) :
  (total_male * sample_size) / (total_male + total_female) = 8 :=
by
  sorry

end expected_male_teachers_in_sample_l277_277150


namespace binom_15_4_l277_277550

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l277_277550


namespace smallest_prime_perf_sqr_minus_eight_l277_277773

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_perf_sqr_minus_eight (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 8

theorem smallest_prime_perf_sqr_minus_eight :
  ∃ (n : ℕ), is_prime n ∧ is_perf_sqr_minus_eight n ∧ (∀ m : ℕ, is_prime m ∧ is_perf_sqr_minus_eight m → n ≤ m) :=
sorry

end smallest_prime_perf_sqr_minus_eight_l277_277773


namespace explicit_form_correct_l277_277697

-- Define the original function form
def f (a b x : ℝ) := 4*x^3 + a*x^2 + b*x + 5

-- Given tangent line slope condition at x = 1
axiom tangent_slope : ∀ (a b : ℝ), (12 * 1^2 + 2 * a * 1 + b = -12)

-- Given the point (1, f(1)) lies on the tangent line y = -12x
axiom tangent_point : ∀ (a b : ℝ), (4 * 1^3 + a * 1^2 + b * 1 + 5 = -12)

-- Definition for the specific f(x) found in solution
def f_explicit (x : ℝ) := 4*x^3 - 3*x^2 - 18*x + 5

-- Finding maximum and minimum values on interval [-3, 1]
def max_value : ℝ := -76
def min_value : ℝ := 16

theorem explicit_form_correct : 
  ∃ a b : ℝ, 
  (∀ x, f a b x = f_explicit x) ∧ 
  (max_value = 16) ∧ 
  (min_value = -76) := 
by
  sorry

end explicit_form_correct_l277_277697


namespace sqrt_360000_eq_600_l277_277493

theorem sqrt_360000_eq_600 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_eq_600_l277_277493


namespace smallest_prime_perf_sqr_minus_eight_l277_277774

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_perf_sqr_minus_eight (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 8

theorem smallest_prime_perf_sqr_minus_eight :
  ∃ (n : ℕ), is_prime n ∧ is_perf_sqr_minus_eight n ∧ (∀ m : ℕ, is_prime m ∧ is_perf_sqr_minus_eight m → n ≤ m) :=
sorry

end smallest_prime_perf_sqr_minus_eight_l277_277774


namespace teachers_like_at_least_one_l277_277509

theorem teachers_like_at_least_one (T C B N: ℕ) 
    (total_teachers : T + C + N = 90)  -- Total number of teachers plus neither equals 90
    (tea_teachers : T = 66)           -- Teachers who like tea is 66
    (coffee_teachers : C = 42)        -- Teachers who like coffee is 42
    (both_beverages : B = 3 * N)      -- Teachers who like both is three times neither
    : T + C - B = 81 :=               -- Teachers who like at least one beverage
by 
  sorry

end teachers_like_at_least_one_l277_277509


namespace solve_for_x_l277_277787

theorem solve_for_x : ∃ x : ℚ, x + 5/6 = 7/18 + 1/2 ∧ x = -7/18 := by
  sorry

end solve_for_x_l277_277787


namespace range_of_a_l277_277161

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - 2 * a

theorem range_of_a (a : ℝ) :
  (∃ (x₀ : ℝ), x₀ ≤ a ∧ f x₀ a ≥ 0) ↔ (a ∈ Set.Icc (-1 : ℝ) 0 ∪ Set.Ici 2) := by
  sorry

end range_of_a_l277_277161


namespace value_of_expression_l277_277590

theorem value_of_expression
  (a b x y : ℝ)
  (h1 : a + b = 0)
  (h2 : x * y = 1) : 
  2 * (a + b) + (7 / 4) * (x * y) = 7 / 4 := 
sorry

end value_of_expression_l277_277590


namespace elementary_school_classes_count_l277_277536

theorem elementary_school_classes_count (E : ℕ) (donate_per_class : ℕ) (middle_school_classes : ℕ) (total_balls : ℕ) :
  donate_per_class = 5 →
  middle_school_classes = 5 →
  total_balls = 90 →
  5 * 2 * E + 5 * 2 * middle_school_classes = total_balls →
  E = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end elementary_school_classes_count_l277_277536


namespace average_speed_l277_277814

theorem average_speed (x : ℝ) (h1 : x > 0) :
  let dist1 := x
  let speed1 := 40
  let dist2 := 4 * x
  let speed2 := 20
  let total_dist := dist1 + dist2
  let time1 := dist1 / speed1
  let time2 := dist2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_dist / total_time
  avg_speed = 200 / 9 :=
by
  -- Definitions
  let dist1 := x
  let speed1 := 40
  let dist2 := 4 * x
  let speed2 := 20
  let total_dist := dist1 + dist2
  let time1 := dist1 / speed1
  let time2 := dist2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_dist / total_time
  -- Proof structure, concluding with the correct answer.
  sorry

end average_speed_l277_277814


namespace p_minus_q_l277_277611

-- Define the given equation as a predicate.
def eqn (x : ℝ) : Prop := (3*x - 9) / (x*x + 3*x - 18) = x + 3

-- Define the values p and q as distinct solutions.
def p_and_q (p q : ℝ) : Prop := eqn p ∧ eqn q ∧ p ≠ q ∧ p > q

theorem p_minus_q {p q : ℝ} (h : p_and_q p q) : p - q = 2 := sorry

end p_minus_q_l277_277611


namespace find_five_digit_number_l277_277647

theorem find_five_digit_number
  (x y : ℕ)
  (h1 : 10 * y + x - (10000 * x + y) = 34767)
  (h2 : 10 * y + x + (10000 * x + y) = 86937) :
  10000 * x + y = 26035 := by
  sorry

end find_five_digit_number_l277_277647


namespace plane_boat_ratio_l277_277516

theorem plane_boat_ratio (P B : ℕ) (h1 : P > B) (h2 : B ≤ 2) (h3 : P + B = 10) : P = 8 ∧ B = 2 ∧ P / B = 4 := by
  sorry

end plane_boat_ratio_l277_277516


namespace weight_difference_l277_277073

open Real

theorem weight_difference (W_A W_B W_C W_D W_E : ℝ)
  (h1 : (W_A + W_B + W_C) / 3 = 50)
  (h2 : W_A = 73)
  (h3 : (W_A + W_B + W_C + W_D) / 4 = 53)
  (h4 : (W_B + W_C + W_D + W_E) / 4 = 51) :
  W_E - W_D = 3 := 
sorry

end weight_difference_l277_277073


namespace gasoline_reduction_l277_277254

theorem gasoline_reduction
  (P Q : ℝ)
  (h1 : 0 < P)
  (h2 : 0 < Q)
  (price_increase_percent : ℝ := 0.25)
  (spending_increase_percent : ℝ := 0.05)
  (new_price : ℝ := P * (1 + price_increase_percent))
  (new_total_cost : ℝ := (P * Q) * (1 + spending_increase_percent)) :
  100 - (100 * (new_total_cost / new_price) / Q) = 16 :=
by
  sorry

end gasoline_reduction_l277_277254


namespace natural_number_pairs_int_l277_277845

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end natural_number_pairs_int_l277_277845


namespace basketball_first_half_score_l277_277878

/-- 
In a college basketball match between Team Alpha and Team Beta, the game was tied at the end 
of the second quarter. The number of points scored by Team Alpha in each of the four quarters
formed an increasing geometric sequence, and the number of points scored by Team Beta in each
of the four quarters formed an increasing arithmetic sequence. At the end of the fourth quarter, 
Team Alpha had won by two points, with neither team scoring more than 100 points. 
Prove that the total number of points scored by the two teams in the first half is 24.
-/
theorem basketball_first_half_score 
  (a r : ℕ) (b d : ℕ)
  (h1 : a + a * r = b + (b + d))
  (h2 : a + a * r + a * r^2 + a * r^3 = b + (b + d) + (b + 2 * d) + (b + 3 * d) + 2)
  (h3 : a + a * r + a * r^2 + a * r^3 ≤ 100)
  (h4 : b + (b + d) + (b + 2 * d) + (b + 3 * d) ≤ 100) : 
  a + a * r + b + (b + d) = 24 :=
  sorry

end basketball_first_half_score_l277_277878


namespace consecutive_even_integer_bases_l277_277736

/-- Given \(X\) and \(Y\) are consecutive even positive integers and the equation
\[ 241_X + 36_Y = 94_{X+Y} \]
this theorem proves that \(X + Y = 22\). -/
theorem consecutive_even_integer_bases (X Y : ℕ) (h1 : X > 0) (h2 : Y = X + 2)
    (h3 : 2 * X^2 + 4 * X + 1 + 3 * Y + 6 = 9 * (X + Y) + 4) : 
    X + Y = 22 :=
by sorry

end consecutive_even_integer_bases_l277_277736


namespace negation_of_proposition_true_l277_277232

theorem negation_of_proposition_true :
  (¬ (∀ x: ℝ, x^2 < 1 → -1 < x ∧ x < 1)) ↔ (∃ x: ℝ, x^2 ≥ 1 ∧ (x ≤ -1 ∨ x ≥ 1)) :=
by
  sorry

end negation_of_proposition_true_l277_277232


namespace inclination_angle_of_line_l277_277080

-- Lean definition for the line equation and inclination angle problem
theorem inclination_angle_of_line : 
  ∃ θ : ℝ, (θ ∈ Set.Ico 0 Real.pi) ∧ (∀ x y: ℝ, x + y - 1 = 0 → Real.tan θ = -1) ∧ θ = 3 * Real.pi / 4 :=
sorry

end inclination_angle_of_line_l277_277080


namespace sqrt_simplification_l277_277494

-- Define a constant for the given number
def n : ℕ := 360000

-- State the theorem we want to prove
theorem sqrt_simplification (h : sqrt n = 600) : sqrt 360000 = 600 := 
by assumption

end sqrt_simplification_l277_277494


namespace rhombus_area_l277_277055

-- Define the parameters given in the problem
namespace MathProof

def perimeter (EFGH : ℝ) : ℝ := 80
def diagonal_EG (EFGH : ℝ) : ℝ := 30

-- Considering the rhombus EFGH with the given perimeter and diagonal
theorem rhombus_area : 
  ∃ (area : ℝ), area = 150 * Real.sqrt 7 ∧ 
  (perimeter EFGH = 80) ∧ 
  (diagonal_EG EFGH = 30) :=
  sorry
end MathProof

end rhombus_area_l277_277055


namespace inequality_abc_equality_condition_l277_277740

theorem inequality_abc (a b c : ℝ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (ab : ℝ) / (c - 1) + (bc : ℝ) / (a - 1) + (ca : ℝ) / (b - 1) ≥ 12 :=
sorry

theorem equality_condition (a b c : ℝ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (ab : ℝ) / (c - 1) + (bc : ℝ) / (a - 1) + (ca : ℝ) / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2 :=
sorry

end inequality_abc_equality_condition_l277_277740


namespace range_of_a_l277_277583

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * x + a * Real.log x

theorem range_of_a (a : ℝ) : 
  (∀ t : ℝ, t ≥ 1 → f (2 * t - 1) a ≥ 2 * f t a - 3) ↔ a < 2 := 
by 
  sorry

end range_of_a_l277_277583


namespace inverse_proportion_k_value_l277_277755

theorem inverse_proportion_k_value (k m : ℝ) 
  (h1 : m = k / 3) 
  (h2 : 6 = k / (m - 1)) 
  : k = 6 :=
by
  sorry

end inverse_proportion_k_value_l277_277755


namespace symmetry_in_mathematics_l277_277935

-- Define the options
def optionA := "summation of harmonic series from 1 to 100"
def optionB := "general quadratic equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0"
def optionC := "Law of Sines: a / sin A = b / sin B = c / sin C"
def optionD := "arithmetic operation: 123456789 * 9 + 10 = 1111111111"

-- Define the symmetry property
def exhibits_symmetry (option: String) : Prop :=
  option = optionC

-- The theorem to prove
theorem symmetry_in_mathematics : ∃ option, exhibits_symmetry option := by
  use optionC
  sorry

end symmetry_in_mathematics_l277_277935


namespace car_speed_is_100_l277_277370

def avg_speed (d1 d2 t: ℕ) := (d1 + d2) / t = 80

theorem car_speed_is_100 
  (x : ℕ)
  (speed_second_hour : ℕ := 60)
  (total_time : ℕ := 2)
  (h : avg_speed x speed_second_hour total_time):
  x = 100 :=
by
  unfold avg_speed at h
  sorry

end car_speed_is_100_l277_277370


namespace total_games_in_season_l277_277071

theorem total_games_in_season (n_teams : ℕ) (games_between_each_team : ℕ) (non_conf_games_per_team : ℕ) 
  (h_teams : n_teams = 8) (h_games_between : games_between_each_team = 3) (h_non_conf : non_conf_games_per_team = 3) :
  let games_within_league := (n_teams * (n_teams - 1) / 2) * games_between_each_team
  let games_outside_league := n_teams * non_conf_games_per_team
  games_within_league + games_outside_league = 108 := by
  sorry

end total_games_in_season_l277_277071


namespace simultaneous_in_Quadrant_I_l277_277895

def in_Quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem simultaneous_in_Quadrant_I (c x y : ℝ) : 
  (2 * x - y = 5) ∧ (c * x + y = 4) ↔ in_Quadrant_I x y ∧ (-2 < c ∧ c < 8 / 5) :=
sorry

end simultaneous_in_Quadrant_I_l277_277895


namespace red_balloon_probability_l277_277196

-- Define the conditions
def initial_red_balloons := 2
def initial_blue_balloons := 4
def inflated_red_balloons := 2
def inflated_blue_balloons := 2

-- Define the total number of balloons after inflation
def total_red_balloons := initial_red_balloons + inflated_red_balloons
def total_blue_balloons := initial_blue_balloons + inflated_blue_balloons
def total_balloons := total_red_balloons + total_blue_balloons

-- Define the probability calculation
def red_probability := (total_red_balloons : ℚ) / total_balloons * 100

-- The theorem to prove
theorem red_balloon_probability : red_probability = 40 := by
  sorry -- Skipping the proof itself

end red_balloon_probability_l277_277196


namespace binom_15_4_l277_277551

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l277_277551


namespace proof_statements_correct_l277_277181

variable (candidates : Nat) (sample_size : Nat)

def is_sampling_survey (survey_type : String) : Prop :=
  survey_type = "sampling"

def is_population (pop_size sample_size : Nat) : Prop :=
  (pop_size = 60000) ∧ (sample_size = 1000)

def is_sample (sample_size pop_size : Nat) : Prop :=
  sample_size < pop_size

def sample_size_correct (sample_size : Nat) : Prop :=
  sample_size = 1000

theorem proof_statements_correct :
  ∀ (survey_type : String) (pop_size sample_size : Nat),
  is_sampling_survey survey_type →
  is_population pop_size sample_size →
  is_sample sample_size pop_size →
  sample_size_correct sample_size →
  survey_type = "sampling" ∧
  pop_size = 60000 ∧
  sample_size = 1000 :=
by
  intros survey_type pop_size sample_size hs hp hsamp hsiz
  sorry

end proof_statements_correct_l277_277181


namespace hotel_elevator_cubic_value_l277_277557

noncomputable def hotel_elevator_cubic : ℚ → ℚ := sorry

theorem hotel_elevator_cubic_value :
  hotel_elevator_cubic 11 = 11 ∧
  hotel_elevator_cubic 12 = 12 ∧
  hotel_elevator_cubic 13 = 14 ∧
  hotel_elevator_cubic 14 = 15 →
  hotel_elevator_cubic 15 = 13 :=
sorry

end hotel_elevator_cubic_value_l277_277557


namespace area_S4_is_3_125_l277_277810

theorem area_S4_is_3_125 (S_1 : Type) (area_S1 : ℝ) 
  (hS1 : area_S1 = 25)
  (bisect_and_construct : ∀ (S : Type) (area : ℝ),
    ∃ S' : Type, ∃ area' : ℝ, area' = area / 2) :
  ∃ S_4 : Type, ∃ area_S4 : ℝ, area_S4 = 3.125 :=
by
  sorry

end area_S4_is_3_125_l277_277810


namespace range_of_k_intersection_l277_277307

theorem range_of_k_intersection (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (k^2 - 1) * x1^2 + 4 * k * x1 + 10 = 0 ∧ (k^2 - 1) * x2^2 + 4 * k * x2 + 10 = 0) ↔ (-1 < k ∧ k < 1) :=
by
  sorry

end range_of_k_intersection_l277_277307


namespace reciprocal_of_sum_l277_277091

theorem reciprocal_of_sum : (1 / (1 / 3 + 1 / 4)) = 12 / 7 := 
by sorry

end reciprocal_of_sum_l277_277091


namespace functional_equation_solution_l277_277694

theorem functional_equation_solution (f : ℚ → ℚ) (H : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ k : ℚ, ∀ x : ℚ, f x = k * x := 
sorry

end functional_equation_solution_l277_277694


namespace ratio_of_age_difference_l277_277401

-- Define the ages of the scrolls and the ratio R
variables (S1 S2 S3 S4 S5 : ℕ)
variables (R : ℚ)

-- Conditions
axiom h1 : S1 = 4080
axiom h5 : S5 = 20655
axiom h2 : S2 - S1 = R * S5
axiom h3 : S3 - S2 = R * S5
axiom h4 : S4 - S3 = R * S5
axiom h6 : S5 - S4 = R * S5

-- The theorem to prove
theorem ratio_of_age_difference : R = 16575 / 82620 :=
by 
  sorry

end ratio_of_age_difference_l277_277401


namespace polygon_sides_l277_277806

theorem polygon_sides (n : ℕ) (D : ℕ) (hD : D = 77) (hFormula : D = n * (n - 3) / 2) (hVertex : n = n) : n + 1 = 15 :=
by
  sorry

end polygon_sides_l277_277806


namespace seven_distinct_integers_exist_pair_l277_277215

theorem seven_distinct_integers_exist_pair (a : Fin 7 → ℕ) (h_distinct : Function.Injective a)
  (h_bound : ∀ i, 1 ≤ a i ∧ a i ≤ 126) :
  ∃ i j : Fin 7, i ≠ j ∧ (1 / 2 : ℚ) ≤ (a i : ℚ) / a j ∧ (a i : ℚ) / a j ≤ 2 := sorry

end seven_distinct_integers_exist_pair_l277_277215


namespace ant_positions_l277_277087

noncomputable def side_lengths (a b c : ℝ) : Prop :=
  (a + 2 = b) ∧ (b + 2 = c) ∧ (4 * c / c - 2 + 1) = 3 ∧ (4 * c / (c - 4) - 1) = 3

theorem ant_positions (a b c : ℝ) (v : ℝ) (ha : side_lengths a b c) : 
  a = 4 ∧ b = 6 ∧ c = 8 :=
by
  sorry

end ant_positions_l277_277087


namespace winner_percentage_l277_277884

variable (votes_winner : ℕ) (win_by : ℕ)
variable (total_votes : ℕ)
variable (percentage_winner : ℕ)

-- Conditions
def conditions : Prop :=
  votes_winner = 930 ∧
  win_by = 360 ∧
  total_votes = votes_winner + (votes_winner - win_by) ∧
  percentage_winner = (votes_winner * 100) / total_votes

-- Theorem to prove
theorem winner_percentage (h : conditions votes_winner win_by total_votes percentage_winner) : percentage_winner = 62 :=
sorry

end winner_percentage_l277_277884


namespace verify_integer_pairs_l277_277837

open Nat

theorem verify_integer_pairs (a b : ℕ) :
  (∃ k1 : ℤ, ↑(a^2) + ↑b = k1 * (↑(b^2) - ↑a)) ∧
  (∃ k2 : ℤ, ↑(b^2) + ↑a = k2 * (↑(a^2) - ↑b)) →
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ 
  (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end verify_integer_pairs_l277_277837


namespace probability_more_than_five_draws_is_20_over_63_l277_277111

open Probability

-- Define the conditions as types
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 5
def total_pennies : ℕ := shiny_pennies + dull_pennies

def total_ways : ℕ := nat.choose total_pennies shiny_pennies -- Calculated as 126

-- Define the events and their probabilities
def event_more_than_five_draws : ℕ := (nat.choose 5 3) * (nat.choose 4 1) -- Calculated as 40

def probability_event : ℚ := event_more_than_five_draws / total_ways -- Simplified to 20/63

-- Prove the probability condition
theorem probability_more_than_five_draws_is_20_over_63 :
  probability_event = 20 / 63 ∧ 20 + 63 = 83 :=
by
  have h1 : total_ways = 126 := by sorry
  have h2 : event_more_than_five_draws = 40 := by sorry
  have h3 : probability_event = 20 / 63 := by
    rw [event_more_than_five_draws, total_ways]
    sorry
  exact ⟨h3, rfl⟩

end probability_more_than_five_draws_is_20_over_63_l277_277111


namespace evaluate_expression_l277_277828

theorem evaluate_expression : 2 + (2 / (2 + (2 / (2 + 3)))) = 17 / 6 := 
by
  sorry

end evaluate_expression_l277_277828


namespace problem_solution_l277_277708

theorem problem_solution :
  ∃ x y z : ℕ,
    0 < x ∧ 0 < y ∧ 0 < z ∧
    x^2 + y^2 + z^2 = 2 * (y * z + 1) ∧
    x + y + z = 4032 ∧
    x^2 * y + z = 4031 :=
by
  sorry

end problem_solution_l277_277708


namespace non_binary_listeners_l277_277372

theorem non_binary_listeners (listen_total males_listen females_dont_listen non_binary_dont_listen dont_listen_total : ℕ) 
  (h_listen_total : listen_total = 250) 
  (h_males_listen : males_listen = 85) 
  (h_females_dont_listen : females_dont_listen = 95) 
  (h_non_binary_dont_listen : non_binary_dont_listen = 45) 
  (h_dont_listen_total : dont_listen_total = 230) : 
  (listen_total - males_listen - (dont_listen_total - females_dont_listen - non_binary_dont_listen)) = 70 :=
by 
  -- Let nbl be the number of non-binary listeners
  let nbl := listen_total - males_listen - (dont_listen_total - females_dont_listen - non_binary_dont_listen)
  -- We need to show nbl = 70
  show nbl = 70
  sorry

end non_binary_listeners_l277_277372


namespace crows_and_trees_l277_277747

theorem crows_and_trees : ∃ (x y : ℕ), 3 * y + 5 = x ∧ 5 * (y - 1) = x ∧ x = 20 ∧ y = 5 :=
by
  sorry

end crows_and_trees_l277_277747


namespace least_number_to_multiply_for_multiple_of_112_l277_277639

theorem least_number_to_multiply_for_multiple_of_112 (n : ℕ) : 
  (Nat.lcm 72 112) / 72 = 14 := 
sorry

end least_number_to_multiply_for_multiple_of_112_l277_277639


namespace conic_section_type_l277_277687

theorem conic_section_type (x y : ℝ) : 
  9 * x^2 - 36 * y^2 = 36 → 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1) :=
by
  sorry

end conic_section_type_l277_277687


namespace barbara_removed_total_sheets_l277_277818

theorem barbara_removed_total_sheets :
  let bundles_colored := 3
  let bunches_white := 2
  let heaps_scrap := 5
  let sheets_per_bunch := 4
  let sheets_per_bundle := 2
  let sheets_per_heap := 20
  bundles_colored * sheets_per_bundle + bunches_white * sheets_per_bunch + heaps_scrap * sheets_per_heap = 114 :=
by
  sorry

end barbara_removed_total_sheets_l277_277818


namespace total_triangles_in_figure_l277_277124

theorem total_triangles_in_figure :
  let row1 := 3
  let row2 := 2
  let row3 := 1
  let small_triangles := row1 + row2 + row3
  let two_small_comb := 3
  let three_small_comb := 1
  let all_small_comb := 1
  small_triangles + two_small_comb + three_small_comb + all_small_comb = 11 :=
by
  let row1 := 3
  let row2 := 2
  let row3 := 1
  let small_triangles := row1 + row2 + row3
  let two_small_comb := 3
  let three_small_comb := 1
  let all_small_comb := 1
  show small_triangles + two_small_comb + three_small_comb + all_small_comb = 11
  sorry

end total_triangles_in_figure_l277_277124


namespace geometric_sequence_fourth_term_l277_277077

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) 
  (h1 : 3 * x + 3 = r * x)
  (h2 : 6 * x + 6 = r * (3 * x + 3)) :
  x = -3 ∧ r = 2 → (x * r^3 = -24) :=
by
  sorry

end geometric_sequence_fourth_term_l277_277077


namespace absent_minded_scientist_two_packages_probability_l277_277733

noncomputable def probability_two_packages : ℝ :=
  let n := 10 in
  (2^n - 1).toReal / ((2^(n - 1)) * n).toReal

theorem absent_minded_scientist_two_packages_probability :
  probability_two_packages = 0.1998 := by
  sorry

end absent_minded_scientist_two_packages_probability_l277_277733


namespace vector_equality_l277_277164

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_equality {a x : V} (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a :=
by
  sorry

end vector_equality_l277_277164


namespace correct_pronoun_possessive_l277_277749

theorem correct_pronoun_possessive : 
  (∃ (pronoun : String), 
    pronoun = "whose" ∧ 
    pronoun = "whose" ∨ pronoun = "who" ∨ pronoun = "that" ∨ pronoun = "which") := 
by
  -- the proof would go here
  sorry

end correct_pronoun_possessive_l277_277749


namespace rectangle_area_l277_277012

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l277_277012


namespace above_line_sign_l277_277857

theorem above_line_sign (A B C x y : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
(h_above : ∃ y₁, Ax + By₁ + C = 0 ∧ y > y₁) : 
  (Ax + By + C > 0 ∧ B > 0) ∨ (Ax + By + C < 0 ∧ B < 0) := 
by
  sorry

end above_line_sign_l277_277857


namespace terminal_side_in_quadrant_l277_277454

theorem terminal_side_in_quadrant (α : ℝ) (h : α = -5) : 
  ∃ (q : ℕ), q = 4 ∧ 270 ≤ (α + 360) % 360 ∧ (α + 360) % 360 < 360 := by 
  sorry

end terminal_side_in_quadrant_l277_277454


namespace cookie_division_l277_277110

theorem cookie_division (C : ℝ) (blue_fraction : ℝ := 1/4) (green_fraction_of_remaining : ℝ := 5/9)
  (remaining_fraction : ℝ := 3/4) (green_fraction : ℝ := 5/12) :
  blue_fraction + green_fraction = 2/3 := by
  sorry

end cookie_division_l277_277110


namespace find_pairs_l277_277832

theorem find_pairs (a b : ℕ) :
  (∃ (a b : ℕ), (b^2 - a ≠ 0) ∧ (a^2 - b ≠ 0) ∧ (a^2 + b) / (b^2 - a) ∈ ℤ ∧ (b^2 + a) / (a^2 - b) ∈ ℤ) → 
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end find_pairs_l277_277832


namespace min_value_l277_277443

-- Given conditions
variable {x y : ℝ}
variable (h1 : 0 < x) (h2 : 0 < y)
variable (h_parallel : (1:ℝ) / 2 = (x - 2) / (-6 * y))

-- Question: Prove the minimum value of 3/x + 1/y is 6
theorem min_value (h1 : 0 < x) (h2 : 0 < y) (h_parallel : (1:ℝ) / 2 = (x - 2) / (-6 * y)) :
  ∃ x y, (1 * x = 3 * y) ∧ ((x + 3 * y = 2) ∧ (x = 1) ∧ (y = 1/3)) ∧ (3 / x + 1 / y) = 6 :=
by
  sorry

end min_value_l277_277443


namespace shaded_area_correct_l277_277880

noncomputable def grid_width : ℕ := 15
noncomputable def grid_height : ℕ := 5
noncomputable def triangle_base : ℕ := 15
noncomputable def triangle_height : ℕ := 3
noncomputable def total_area : ℝ := (grid_width * grid_height : ℝ)
noncomputable def triangle_area : ℝ := (1 / 2) * triangle_base * triangle_height
noncomputable def shaded_area : ℝ := total_area - triangle_area

theorem shaded_area_correct : shaded_area = 52.5 := 
by sorry

end shaded_area_correct_l277_277880


namespace factorize_expression_l277_277137

variable {a b : ℕ}

theorem factorize_expression (a b : ℕ) : 9 * a - 6 * b = 3 * (3 * a - 2 * b) :=
by
  sorry

end factorize_expression_l277_277137


namespace find_x_l277_277222

theorem find_x (a b x : ℝ) (h1 : ∀ a b, a * b = 2 * a - b) (h2 : 2 * (6 * x) = 2) : x = 10 := 
sorry

end find_x_l277_277222


namespace train_platform_length_equal_l277_277357

theorem train_platform_length_equal 
  (v : ℝ) (t : ℝ) (L_train : ℝ)
  (h1 : v = 144 * (1000 / 3600))
  (h2 : t = 60)
  (h3 : L_train = 1200) :
  L_train = 2400 - L_train := 
sorry

end train_platform_length_equal_l277_277357


namespace pile_division_possible_l277_277322

theorem pile_division_possible (n : ℕ) :
  ∃ (division : list ℕ), (∀ x ∈ division, x = 1) ∧ division.sum = n :=
by
  sorry

end pile_division_possible_l277_277322


namespace general_formula_sum_b_l277_277599

-- Define the arithmetic sequence
def arithmetic_sequence (a d: ℕ) (n: ℕ) := a + (n - 1) * d

-- Given conditions
def a1 : ℕ := 1
def d : ℕ := 2
def a (n : ℕ) : ℕ := arithmetic_sequence a1 d n
def b (n : ℕ) : ℕ := 2 ^ a n

-- Formula for the arithmetic sequence
theorem general_formula (n : ℕ) : a n = 2 * n - 1 := 
by sorry

-- Sum of the first n terms of b_n
theorem sum_b (n : ℕ) : (Finset.range n).sum b = (2 / 3) * (4 ^ n - 1) :=
by sorry

end general_formula_sum_b_l277_277599


namespace perpendicular_lines_k_value_l277_277128

theorem perpendicular_lines_k_value (k : ℝ) : 
  (∃ (m₁ m₂ : ℝ), (m₁ = k/3) ∧ (m₂ = 3) ∧ (m₁ * m₂ = -1)) → k = -1 :=
by
  sorry

end perpendicular_lines_k_value_l277_277128


namespace exchange_rmb_ways_l277_277135

theorem exchange_rmb_ways : 
  {n : ℕ // ∃ (x y z : ℕ), x + 2 * y + 5 * z = 10 ∧ n = 10} :=
sorry

end exchange_rmb_ways_l277_277135


namespace find_equation_of_line_l277_277241

theorem find_equation_of_line
  (m b : ℝ) 
  (h1 : ∃ k : ℝ, (k^2 - 2*k + 3 = k*m + b ∧ ∃ d : ℝ, d = 4) 
        ∧ (4*m - k^2 + 2*m*k - 3 + b = 0)) 
  (h2 : 8 = 2*m + b)
  (h3 : b ≠ 0) 
  : y = 8 :=
by 
  sorry

end find_equation_of_line_l277_277241


namespace arithmetic_sequence_properties_l277_277360

theorem arithmetic_sequence_properties
    (n s1 s2 s3 : ℝ)
    (h1 : s1 = 8)
    (h2 : s2 = 50)
    (h3 : s3 = 134)
    (h4 : n = 8) :
    n^2 * s3 - 3 * n * s1 * s2 + 2 * s1^2 = 0 := 
by {
  sorry
}

end arithmetic_sequence_properties_l277_277360


namespace find_sale_month_4_l277_277802

-- Define the given sales data
def sale_month_1: ℕ := 5124
def sale_month_2: ℕ := 5366
def sale_month_3: ℕ := 5808
def sale_month_5: ℕ := 6124
def sale_month_6: ℕ := 4579
def average_sale_per_month: ℕ := 5400

-- Define the goal: Sale in the fourth month
def sale_month_4: ℕ := 5399

-- Prove that the total sales conforms to the given average sale
theorem find_sale_month_4 :
  sale_month_1 + sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5 + sale_month_6 = 6 * average_sale_per_month :=
by
  sorry

end find_sale_month_4_l277_277802


namespace max_value_7x_10y_z_l277_277142

theorem max_value_7x_10y_z (x y z : ℝ) 
  (h : x^2 + 2 * x + (1 / 5) * y^2 + 7 * z^2 = 6) : 
  7 * x + 10 * y + z ≤ 55 := 
sorry

end max_value_7x_10y_z_l277_277142


namespace mike_total_time_spent_l277_277046

theorem mike_total_time_spent : 
  let hours_watching_tv_per_day := 4
  let days_per_week := 7
  let days_playing_video_games := 3
  let hours_playing_video_games_per_day := hours_watching_tv_per_day / 2
  let total_hours_watching_tv := hours_watching_tv_per_day * days_per_week
  let total_hours_playing_video_games := hours_playing_video_games_per_day * days_playing_video_games
  let total_time_spent := total_hours_watching_tv + total_hours_playing_video_games
  total_time_spent = 34 :=
by
  sorry

end mike_total_time_spent_l277_277046


namespace find_pairs_l277_277833

theorem find_pairs (a b : ℕ) :
  (∃ (a b : ℕ), (b^2 - a ≠ 0) ∧ (a^2 - b ≠ 0) ∧ (a^2 + b) / (b^2 - a) ∈ ℤ ∧ (b^2 + a) / (a^2 - b) ∈ ℤ) → 
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end find_pairs_l277_277833


namespace talias_fathers_age_l277_277186

-- Definitions based on the conditions
variable (T M F : ℕ)

-- The conditions
axiom h1 : T + 7 = 20
axiom h2 : M = 3 * T
axiom h3 : F + 3 = M

-- Goal: Prove that Talia's father (F) is currently 36 years old
theorem talias_fathers_age : F = 36 :=
by
  sorry

end talias_fathers_age_l277_277186


namespace algebraic_expression_value_l277_277705

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  (x - 1)^2 - x * (x - 3) + (x + 1) * (x - 1) = 3 :=
by
  sorry

end algebraic_expression_value_l277_277705


namespace line_through_points_l277_277302

variable (A1 B1 A2 B2 : ℝ)

def line1 : Prop := -7 * A1 + 9 * B1 = 1
def line2 : Prop := -7 * A2 + 9 * B2 = 1

theorem line_through_points (h1 : line1 A1 B1) (h2 : line1 A2 B2) :
  ∃ (k : ℝ), (∀ (x y : ℝ), y - B1 = k * (x - A1)) ∧ (-7 * (x : ℝ) + 9 * y = 1) := 
by sorry

end line_through_points_l277_277302


namespace probability_sum_at_least_15_l277_277876

-- Define the total number of balls
def num_balls : ℕ := 8

-- Define the valid outcomes summing to at least 15
def valid_outcomes : List (ℕ × ℕ) := [(7, 8), (8, 7), (8, 8)]

-- Calculate the probability
def probability := (valid_outcomes.length : ℚ) / (num_balls * num_balls)

-- Define the theorem to be proved
theorem probability_sum_at_least_15 : probability = 3 / 64 := by
  sorry

end probability_sum_at_least_15_l277_277876


namespace sqrt_360000_eq_600_l277_277492

theorem sqrt_360000_eq_600 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_eq_600_l277_277492


namespace price_of_basic_computer_l277_277762

-- Conditions
variables (C P : ℝ)
axiom cond1 : C + P = 2500
axiom cond2 : 3 * P = C + 500

-- Prove that the price of the basic computer is $1750
theorem price_of_basic_computer : C = 1750 :=
by 
  sorry

end price_of_basic_computer_l277_277762


namespace problem_equivalence_l277_277437

open Set

variable {R : Set ℝ} 

def setA : Set ℝ := {x | x^2 - 2 * x ≥ 0}
def setB : Set ℝ := {x | x > 1}
def complementA : Set ℝ := {x | 0 < x ∧ x < 2}
def intersection : Set ℝ := complementA ∩ setB

theorem problem_equivalence : intersection = {x | 1 < x ∧ x < 2} :=
by
  -- This is where the proof would go.
  sorry

end problem_equivalence_l277_277437


namespace xiao_li_first_three_l277_277189

def q1_proba_correct (p1 p2 p3 : ℚ) : ℚ :=
  p1 * p2 * p3 + 
  (1 - p1) * p2 * p3 + 
  p1 * (1 - p2) * p3 + 
  p1 * p2 * (1 - p3)

theorem xiao_li_first_three (p1 p2 p3 : ℚ) (h1 : p1 = 3/4) (h2 : p2 = 1/2) (h3 : p3 = 5/6) :
  q1_proba_correct p1 p2 p3 = 11 / 24 := by
  rw [h1, h2, h3]
  sorry

end xiao_li_first_three_l277_277189


namespace problem_statement_l277_277872

theorem problem_statement (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 ∧ x^2 + y^2 = 104 :=
by
  sorry

end problem_statement_l277_277872


namespace sum_series_eq_260_l277_277281

theorem sum_series_eq_260 : (2 + 12 + 22 + 32 + 42) + (10 + 20 + 30 + 40 + 50) = 260 := by
  sorry

end sum_series_eq_260_l277_277281


namespace ratio_value_l277_277171

theorem ratio_value (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := 
by
  sorry

end ratio_value_l277_277171


namespace jason_pokemon_cards_l277_277035

theorem jason_pokemon_cards :
  ∀ (initial_cards trade_benny_lost trade_benny_gain trade_sean_lost trade_sean_gain give_to_brother : ℕ),
  initial_cards = 5 →
  trade_benny_lost = 2 →
  trade_benny_gain = 3 →
  trade_sean_lost = 3 →
  trade_sean_gain = 4 →
  give_to_brother = 2 →
  initial_cards - trade_benny_lost + trade_benny_gain - trade_sean_lost + trade_sean_gain - give_to_brother = 5 :=
by
  intros
  sorry

end jason_pokemon_cards_l277_277035


namespace three_person_subcommittees_from_eight_l277_277718

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem three_person_subcommittees_from_eight (n k : ℕ) (h_n : n = 8) (h_k : k = 3) :
  combination n k = 56 := by
  sorry

end three_person_subcommittees_from_eight_l277_277718


namespace total_shaded_area_of_rectangles_l277_277518

theorem total_shaded_area_of_rectangles (w1 l1 w2 l2 ow ol : ℕ) 
  (h1 : w1 = 4) (h2 : l1 = 12) (h3 : w2 = 5) (h4 : l2 = 10) (h5 : ow = 4) (h6 : ol = 5) :
  (w1 * l1 + w2 * l2 - ow * ol = 78) :=
by
  sorry

end total_shaded_area_of_rectangles_l277_277518


namespace garbage_accumulation_correct_l277_277354

-- Given conditions
def garbage_days_per_week : ℕ := 3
def garbage_per_collection : ℕ := 200
def duration_weeks : ℕ := 2

-- Week 1: Full garbage accumulation
def week1_garbage_accumulation : ℕ := garbage_days_per_week * garbage_per_collection

-- Week 2: Half garbage accumulation due to the policy
def week2_garbage_accumulation : ℕ := week1_garbage_accumulation / 2

-- Total garbage accumulation over the 2 weeks
def total_garbage_accumulation (week1 week2 : ℕ) : ℕ := week1 + week2

-- Proof statement
theorem garbage_accumulation_correct :
  total_garbage_accumulation week1_garbage_accumulation week2_garbage_accumulation = 900 := by
  sorry

end garbage_accumulation_correct_l277_277354


namespace rectangle_area_l277_277013

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l277_277013


namespace original_amount_water_l277_277530

theorem original_amount_water (O : ℝ) (h1 : (0.75 = 0.05 * O)) : O = 15 :=
by sorry

end original_amount_water_l277_277530


namespace C_younger_than_A_l277_277665

variables (A B C : ℕ)

-- Original Condition
axiom age_condition : A + B = B + C + 17

-- Lean Statement to Prove
theorem C_younger_than_A (A B C : ℕ) (h : A + B = B + C + 17) : C + 17 = A :=
by {
  -- Proof would go here but is omitted.
  sorry
}

end C_younger_than_A_l277_277665


namespace ratio_of_boat_to_stream_l277_277260

theorem ratio_of_boat_to_stream (B S : ℝ) (h : ∀ D : ℝ, D / (B - S) = 2 * (D / (B + S))) :
  B / S = 3 :=
by 
  sorry

end ratio_of_boat_to_stream_l277_277260


namespace exists_consecutive_with_square_factors_l277_277295

theorem exists_consecutive_with_square_factors (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ m : ℕ, m^2 ∣ (k + i) ∧ m > 1 :=
by {
  sorry
}

end exists_consecutive_with_square_factors_l277_277295


namespace triangle_area_l277_277178

noncomputable def area_triangle (b c angle_C : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin angle_C

theorem triangle_area :
  let b := 1
  let c := Real.sqrt 3
  let angle_C := 2 * Real.pi / 3
  area_triangle b c (Real.sin angle_C) = Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_l277_277178


namespace number_is_nine_l277_277963

theorem number_is_nine (x : ℤ) (h : 3 * (2 * x + 9) = 81) : x = 9 :=
by
  sorry

end number_is_nine_l277_277963


namespace female_guests_from_jays_family_l277_277044

theorem female_guests_from_jays_family (total_guests : ℕ) (percent_females : ℝ) (percent_from_jays_family : ℝ)
    (h1 : total_guests = 240) (h2 : percent_females = 0.60) (h3 : percent_from_jays_family = 0.50) :
    total_guests * percent_females * percent_from_jays_family = 72 := by
  sorry

end female_guests_from_jays_family_l277_277044


namespace average_weight_increase_l277_277347

theorem average_weight_increase (A : ℝ) (hA : 8 * A + 20 = (80 : ℝ) + (8 * (A - (60 - 80) / 8))) :
  ((8 * A + 20) / 8) - A = (2.5 : ℝ) :=
by
  sorry

end average_weight_increase_l277_277347


namespace natural_number_pairs_int_l277_277844

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end natural_number_pairs_int_l277_277844


namespace circle_origin_range_l277_277874

theorem circle_origin_range (m : ℝ) : 
  (0 - m)^2 + (0 + m)^2 < 4 → -Real.sqrt 2 < m ∧ m < Real.sqrt 2 :=
sorry

end circle_origin_range_l277_277874


namespace percentage_of_Hindu_boys_l277_277881

theorem percentage_of_Hindu_boys (total_boys : ℕ) (muslim_percentage : ℕ) (sikh_percentage : ℕ)
  (other_community_boys : ℕ) (H : total_boys = 850) (H1 : muslim_percentage = 44) 
  (H2 : sikh_percentage = 10) (H3 : other_community_boys = 153) :
  let muslim_boys := muslim_percentage * total_boys / 100
  let sikh_boys := sikh_percentage * total_boys / 100
  let non_hindu_boys := muslim_boys + sikh_boys + other_community_boys
  let hindu_boys := total_boys - non_hindu_boys
  (hindu_boys * 100 / total_boys : ℚ) = 28 := 
by
  sorry

end percentage_of_Hindu_boys_l277_277881


namespace inversion_count_seq1_inversion_count_seq2_odd_inversion_count_seq2_even_inversion_count_reversed_seq_l277_277808

-- 1. Inversion count for a_n = -2n + 19
theorem inversion_count_seq1 : 
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 100 → (∑ i in Finset.range (n - 1), i) = 4950 := 
by
  sorry

-- 2. Inversion count for piecewise sequence
def piecewise_seq (n : ℕ) : ℚ :=
  if n % 2 = 1 then (1/3)^n else -n/(n+1)

theorem inversion_count_seq2_odd : 
  ∀ (k : ℕ), 1 ≤ k ∧ k % 2 = 1 → (∑ i in Finset.range k, i / 2) + (∑ j in Finset.range ((k-1) / 2), j) = (3 * k^2 - 4 * k + 1) / 8 :=
by
  sorry

theorem inversion_count_seq2_even : 
  ∀ (k : ℕ), 1 ≤ k ∧ k % 2 = 0 → (∑ i in Finset.range k, i / 2) + (∑ j in Finset.range (k / 2), j) = (3 * k^2 - 2 * k) / 8 :=
by
  sorry

-- 3. Inversion count for reversed sequence
theorem inversion_count_reversed_seq : 
  ∀ (n a : ℕ), 0 ≤ a ∧ a ≤ n * (n - 1) / 2 → (∑ i in Finset.range n, (n - 1 - i) - a) = n * (n - 1) / 2 - a :=
by
  sorry

end inversion_count_seq1_inversion_count_seq2_odd_inversion_count_seq2_even_inversion_count_reversed_seq_l277_277808


namespace quadratic_inequality_l277_277731

-- Defining the quadratic expression
def quadratic_expr (a x : ℝ) : ℝ :=
  (a + 2) * x^2 + 2 * (a + 2) * x + 4

-- Statement to be proven
theorem quadratic_inequality {a : ℝ} :
  (∀ x : ℝ, quadratic_expr a x > 0) ↔ -2 ≤ a ∧ a < 2 :=
by
  sorry -- Proof omitted

end quadratic_inequality_l277_277731


namespace runners_meet_again_l277_277148

theorem runners_meet_again 
  (v1 v2 v3 v4 v5 : ℕ)
  (h1 : v1 = 32) 
  (h2 : v2 = 40) 
  (h3 : v3 = 48) 
  (h4 : v4 = 56) 
  (h5 : v5 = 64) 
  (h6 : 400 % (v2 - v1) = 0)
  (h7 : 400 % (v3 - v2) = 0)
  (h8 : 400 % (v4 - v3) = 0)
  (h9 : 400 % (v5 - v4) = 0) :
  ∃ t : ℕ, t = 500 :=
by sorry

end runners_meet_again_l277_277148


namespace ratio_of_place_values_l277_277602

def thousands_place_value : ℝ := 1000
def tenths_place_value : ℝ := 0.1

theorem ratio_of_place_values : thousands_place_value / tenths_place_value = 10000 := by
  sorry

end ratio_of_place_values_l277_277602


namespace john_running_time_l277_277817

theorem john_running_time
  (x : ℚ)
  (h1 : 15 * x + 10 * (9 - x) = 100)
  (h2 : 0 ≤ x)
  (h3 : x ≤ 9) :
  x = 2 := by
  sorry

end john_running_time_l277_277817


namespace triangle_y_values_l277_277679

theorem triangle_y_values (y : ℕ) :
  (8 + 11 > y^2) ∧ (y^2 + 8 > 11) ∧ (y^2 + 11 > 8) ↔ y = 2 ∨ y = 3 ∨ y = 4 :=
by
  sorry

end triangle_y_values_l277_277679


namespace abs_quadratic_inequality_solution_l277_277558

theorem abs_quadratic_inequality_solution (x : ℝ) :
  |x^2 - 4 * x + 3| ≤ 3 ↔ 0 ≤ x ∧ x ≤ 4 :=
by sorry

end abs_quadratic_inequality_solution_l277_277558


namespace area_of_rectangle_ABCD_l277_277005

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l277_277005


namespace cone_angle_l277_277356

theorem cone_angle (r l : ℝ) (α : ℝ)
  (h1 : 2 * Real.pi * r = Real.pi * l) 
  (h2 : Real.cos α = r / l) : α = Real.pi / 3 :=
by
  sorry

end cone_angle_l277_277356


namespace set_of_x_satisfying_2f_less_than_x_plus_1_l277_277204

theorem set_of_x_satisfying_2f_less_than_x_plus_1 (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : ∀ x : ℝ, deriv f x > 1 / 2) :
  { x : ℝ | 2 * f x < x + 1 } = { x : ℝ | x < 1 } :=
by
  sorry

end set_of_x_satisfying_2f_less_than_x_plus_1_l277_277204


namespace rectangle_side_ratio_square_l277_277729

noncomputable def ratio_square (a b : ℝ) : ℝ :=
(a / b) ^ 2

theorem rectangle_side_ratio_square (a b : ℝ) (h : (a - b) / (a + b) = 1 / 3) : 
  ratio_square a b = 4 := by
  sorry

end rectangle_side_ratio_square_l277_277729


namespace evaluate_expression_l277_277133

theorem evaluate_expression : (164^2 - 148^2) / 16 = 312 := 
by 
  sorry

end evaluate_expression_l277_277133


namespace min_a_b_l277_277207

theorem min_a_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 45 * a + b = 2021) : a + b = 85 :=
sorry

end min_a_b_l277_277207


namespace exist_positive_integers_x_y_z_l277_277482

theorem exist_positive_integers_x_y_z (n : ℕ) (hn : n > 0) : 
  ∃ (x y z : ℕ), 
    x = 2^(n^2) * 3^(n+1) ∧
    y = 2^(n^2 - n) * 3^n ∧
    z = 2^(n^2 - 2*n + 2) * 3^(n-1) ∧
    x^(n-1) + y^n = z^(n+1) :=
by {
  -- placeholder for the proof
  sorry
}

end exist_positive_integers_x_y_z_l277_277482


namespace range_of_a_l277_277162

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ 1 → (x^2 + a * x + 9) ≥ 0) : a ≥ -6 := 
sorry

end range_of_a_l277_277162


namespace squares_total_l277_277559

def number_of_squares (figure : Type) : ℕ := sorry

theorem squares_total (figure : Type) : number_of_squares figure = 38 := sorry

end squares_total_l277_277559


namespace soda_costs_94_cents_l277_277985

theorem soda_costs_94_cents (b s: ℤ) (h1 : 4 * b + 3 * s = 500) (h2 : 3 * b + 4 * s = 540) : s = 94 := 
by
  sorry

end soda_costs_94_cents_l277_277985


namespace condition_on_a_l277_277512

theorem condition_on_a (a : ℝ) : 
  (∀ x : ℝ, (5 * x - 3 < 3 * x + 5) → (x < a)) ↔ (a ≥ 4) :=
by
  sorry

end condition_on_a_l277_277512


namespace calculate_expression_l277_277986

theorem calculate_expression : 5 * 7 + 9 * 4 - 36 / 3 + 48 / 4 = 71 := by
  sorry

end calculate_expression_l277_277986


namespace ral_current_age_l277_277058

theorem ral_current_age (Ral_age Suri_age : ℕ) (h1 : Ral_age = 2 * Suri_age) (h2 : Suri_age + 3 = 16) : Ral_age = 26 :=
by {
  -- Proof goes here
  sorry
}

end ral_current_age_l277_277058


namespace percentage_increase_from_boys_to_total_l277_277823

def DamesSchoolBoys : ℕ := 2000
def DamesSchoolGirls : ℕ := 5000
def TotalAttendance : ℕ := DamesSchoolBoys + DamesSchoolGirls
def PercentageIncrease (initial final : ℕ) : ℚ := ((final - initial) / initial) * 100

theorem percentage_increase_from_boys_to_total :
  PercentageIncrease DamesSchoolBoys TotalAttendance = 250 :=
by
  sorry

end percentage_increase_from_boys_to_total_l277_277823


namespace no_cubic_solution_l277_277359

theorem no_cubic_solution (t : ℤ) : ¬ ∃ k : ℤ, (7 * t + 3 = k ^ 3) := by
  sorry

end no_cubic_solution_l277_277359


namespace integer_roots_of_polynomial_l277_277848

theorem integer_roots_of_polynomial :
  {x : ℤ | x^3 - 4*x^2 - 14*x + 24 = 0} = {-4, -3, 3} := by
  sorry

end integer_roots_of_polynomial_l277_277848


namespace incorrect_sum_Sn_l277_277084

-- Define the geometric sequence sum formula
def Sn (a r : ℕ) (n : ℕ) : ℕ := a * (1 - r^n) / (1 - r)

-- Define the given values
def S1 : ℕ := 8
def S2 : ℕ := 20
def S3 : ℕ := 36
def S4 : ℕ := 65

-- The main proof statement
theorem incorrect_sum_Sn : 
  ∃ (a r : ℕ), 
  a = 8 ∧ 
  Sn a r 1 = S1 ∧ 
  Sn a r 2 = S2 ∧ 
  Sn a r 3 ≠ S3 ∧ 
  Sn a r 4 = S4 :=
by sorry

end incorrect_sum_Sn_l277_277084


namespace equation1_equation2_equation3_equation4_l277_277220

theorem equation1 (x : ℝ) : (x - 1) ^ 2 - 5 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

theorem equation2 (x : ℝ) : x * (x + 4) = -3 * (x + 4) ↔ x = -4 ∨ x = -3 := by
  sorry

theorem equation3 (y : ℝ) : 2 * y ^ 2 - 5 * y + 2 = 0 ↔ y = 1 / 2 ∨ y = 2 := by
  sorry

theorem equation4 (m : ℝ) : 2 * m ^ 2 - 7 * m - 3 = 0 ↔ m = (7 + Real.sqrt 73) / 4 ∨ m = (7 - Real.sqrt 73) / 4 := by
  sorry

end equation1_equation2_equation3_equation4_l277_277220


namespace bus_interval_three_buses_l277_277944

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l277_277944


namespace minimum_trucks_on_lot_l277_277212

variable (total_trucks : ℕ) (rented_percentage : ℝ) (max_rented_trucks : ℕ)
variable (initial_trucks : total_trucks = 24) (returned_percentage : rented_percentage = 0.5) (max_trucks_rented : max_rented_trucks = 24)

theorem minimum_trucks_on_lot (total_trucks : ℕ) (rented_percentage : ℝ) (max_rented_trucks : ℕ)
  (initial_trucks : total_trucks = 24) (returned_percentage : rented_percentage = 0.5) (max_trucks_rented : max_rented_trucks = 24) :
  max_rented_trucks / 2 = 12 :=
by sorry

end minimum_trucks_on_lot_l277_277212


namespace sum_of_all_four_digit_numbers_l277_277291

-- Let us define the set of digits
def digits : set ℕ := {1, 2, 3, 4, 5}

-- We will define a function that generates the four-digit numbers
def four_digit_numbers := {n : ℕ // ∃ a b c d : ℕ, 
                                      a ∈ digits ∧ 
                                      b ∈ digits ∧ 
                                      c ∈ digits ∧ 
                                      d ∈ digits ∧ 
                                      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ 
                                      b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                                      n = 1000 * a + 100 * b + 10 * c + d}

-- Define a function to calculate the sum of all elements in a set of numbers
def sum_set (s : set ℕ) : ℕ := s.fold (λa b, a + b) 0

theorem sum_of_all_four_digit_numbers :
  sum_set {n | ∃ x : four_digit_numbers, x.val = n} = 399960 :=
sorry

end sum_of_all_four_digit_numbers_l277_277291


namespace larger_number_of_two_with_conditions_l277_277725

theorem larger_number_of_two_with_conditions (x y : ℕ) (h1 : x * y = 30) (h2 : x + y = 13) : max x y = 10 :=
by
  sorry

end larger_number_of_two_with_conditions_l277_277725


namespace base_6_conversion_l277_277628

-- Define the conditions given in the problem
def base_6_to_10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

-- given that 524_6 = 2cd_10 and c, d are base-10 digits, prove that (c * d) / 12 = 3/4
theorem base_6_conversion (c d : ℕ) (h1 : base_6_to_10 5 2 4 = 196) (h2 : 2 * 10 * c + d = 196) :
  (c * d) / 12 = 3 / 4 :=
sorry

end base_6_conversion_l277_277628


namespace fall_increase_l277_277408

noncomputable def percentage_increase_in_fall (x : ℝ) : ℝ :=
  x

theorem fall_increase :
  ∃ (x : ℝ), (1 + percentage_increase_in_fall x / 100) * (1 - 19 / 100) = 1 + 11.71 / 100 :=
by
  sorry

end fall_increase_l277_277408


namespace ral_age_is_26_l277_277060

def ral_current_age (suri_age : ℕ) (ral_age : ℕ) : Prop :=
  ral_age = 2 * suri_age

theorem ral_age_is_26 (suri_current_age : ℕ) (ral_current_age : ℕ) (h1 : suri_current_age + 3 = 16) (h2 : ral_age = 2 * suri_age) : ral_current_age = 26 := 
by
  sorry

end ral_age_is_26_l277_277060


namespace remainder_when_divided_by_l277_277771

def P (x : ℤ) : ℤ := 5 * x^8 - 2 * x^7 - 8 * x^6 + 3 * x^4 + 5 * x^3 - 13
def D (x : ℤ) : ℤ := 3 * (x - 3)

theorem remainder_when_divided_by (x : ℤ) : P 3 = 23364 :=
by {
  -- This is where the calculation steps would go, but we're omitting them.
  sorry
}

end remainder_when_divided_by_l277_277771


namespace max_elements_A_union_B_l277_277296

noncomputable def sets_with_conditions (A B : Finset ℝ ) (n : ℕ) : Prop :=
  (∀ (s : Finset ℝ), s.card = n ∧ s ⊆ A → s.sum id ∈ B) ∧
  (∀ (s : Finset ℝ), s.card = n ∧ s ⊆ B → s.prod id ∈ A)

theorem max_elements_A_union_B {A B : Finset ℝ} (n : ℕ) (hn : 1 < n)
    (hA : A.card ≥ n) (hB : B.card ≥ n)
    (h_condition : sets_with_conditions A B n) :
    A.card + B.card ≤ 2 * n :=
  sorry

end max_elements_A_union_B_l277_277296


namespace common_ratio_of_geometric_sequence_l277_277756

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem common_ratio_of_geometric_sequence
  (a1 d : ℝ) (h1 : d ≠ 0)
  (h2 : (a_n a1 d 5) * (a_n a1 d 20) = (a_n a1 d 10) ^ 2) :
  (a_n a1 d 10) / (a_n a1 d 5) = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l277_277756


namespace number_of_elephants_l277_277734

theorem number_of_elephants (giraffes penguins total_animals elephants : ℕ)
  (h1 : giraffes = 5)
  (h2 : penguins = 2 * giraffes)
  (h3 : penguins = total_animals / 5)
  (h4 : elephants = total_animals * 4 / 100) :
  elephants = 2 := by
  -- The proof is omitted
  sorry

end number_of_elephants_l277_277734


namespace function_passes_through_fixed_point_l277_277754

noncomputable def passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : Prop :=
  ∃ y : ℝ, y = a^(1-1) + 1 ∧ y = 2

theorem function_passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : passes_through_fixed_point a h :=
by
  sorry

end function_passes_through_fixed_point_l277_277754


namespace line_through_intersection_and_origin_l277_277914

-- Definitions of the lines
def l1 (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l2 (x y : ℝ) : Prop := y = 1 - x

-- Prove that the line passing through the intersection of l1 and l2 and the origin has the equation 3x + 2y = 0
theorem line_through_intersection_and_origin (x y : ℝ) 
  (h1 : 2 * x - y + 7 = 0) (h2 : y = 1 - x) : 3 * x + 2 * y = 0 := 
sorry

end line_through_intersection_and_origin_l277_277914


namespace value_of_f_g_l277_277155

def f (x : ℝ) : ℝ := x^2 - 3*x + 7
def g (x : ℝ) : ℝ := x + 4

theorem value_of_f_g (h₁ : f (g 3) = 35) (h₂ : g (f 3) = 11) : f (g 3) - g (f 3) = 24 :=
by
  calc
    f (g 3) - g (f 3) = 35 - 11 := by rw [h₁, h₂]
                      _         = 24 := by norm_num

end value_of_f_g_l277_277155


namespace interval_with_three_buses_l277_277937

theorem interval_with_three_buses (interval_two_buses : ℕ) (total_route_time : ℕ) (bus_count : ℕ) : 
  interval_two_buses = 21 → total_route_time = 2 * interval_two_buses → bus_count = 3 → 
  (total_route_time / bus_count) = 14 :=
by
  intros h1 h2 h3
  rw [h1, h3, ← h2]
  simp
  sorry

end interval_with_three_buses_l277_277937


namespace distinct_units_digits_of_squares_mod_6_l277_277588

theorem distinct_units_digits_of_squares_mod_6 : 
  ∃ (s : Finset ℕ), s = {0, 1, 4, 3} ∧ s.card = 4 :=
by
  sorry

end distinct_units_digits_of_squares_mod_6_l277_277588


namespace ratio_of_larger_to_smaller_l277_277645

theorem ratio_of_larger_to_smaller 
    (x y : ℝ) 
    (hx : x > 0) 
    (hy : y > 0) 
    (h : x + y = 7 * (x - y)) : 
    x / y = 4 / 3 := 
by 
    sorry

end ratio_of_larger_to_smaller_l277_277645


namespace expand_expression_l277_277561

theorem expand_expression (x : ℝ) : (x - 1) * (4 * x + 5) = 4 * x^2 + x - 5 := 
by
  -- Proof omitted
  sorry

end expand_expression_l277_277561


namespace unique_divisors_form_l277_277847

theorem unique_divisors_form (n : ℕ) (h₁ : n > 1)
    (h₂ : ∀ d : ℕ, d ∣ n ∧ d > 1 → ∃ a r : ℕ, a > 1 ∧ r > 1 ∧ d = a^r + 1) :
    n = 10 := by
  sorry

end unique_divisors_form_l277_277847


namespace area_of_triangle_formed_by_lines_l277_277651

def line1 (x : ℝ) : ℝ := 5
def line2 (x : ℝ) : ℝ := 1 + x
def line3 (x : ℝ) : ℝ := 1 - x

theorem area_of_triangle_formed_by_lines :
  let A := (4, 5)
  let B := (-4, 5)
  let C := (0, 1)
  (1 / 2) * abs (4 * 5 + (-4) * 1 + 0 * 5 - (5 * (-4) + 1 * 4 + 5 * 0)) = 16 := by
  sorry

end area_of_triangle_formed_by_lines_l277_277651


namespace plane_can_be_colored_l277_277365

-- Define a structure for a triangle and the plane divided into triangles
structure Triangle :=
(vertices : Fin 3 → ℕ) -- vertices labeled with ℕ, interpreted as 0, 1, 2

structure Plane :=
(triangles : Set Triangle)
(adjacent : Triangle → Triangle → Prop)
(labels_correct : ∀ {t1 t2 : Triangle}, adjacent t1 t2 → 
  ∀ i j: Fin 3, t1.vertices i ≠ t1.vertices j)
(adjacent_conditions: ∀ t1 t2: Triangle, adjacent t1 t2 → 
  ∃ v, (∃ i: Fin 3, t1.vertices i = v) ∧ (∃ j: Fin 3, t2.vertices j = v))

theorem plane_can_be_colored (p : Plane) : 
  ∃ (c : Triangle → ℕ), (∀ t1 t2, p.adjacent t1 t2 → c t1 ≠ c t2) :=
sorry

end plane_can_be_colored_l277_277365


namespace cos_squared_identity_l277_277438

variable (θ : ℝ)

-- Given condition
def tan_theta : Prop := Real.tan θ = 2

-- Question: Find the value of cos²(θ + π/4)
theorem cos_squared_identity (h : tan_theta θ) : Real.cos (θ + Real.pi / 4) ^ 2 = 1 / 10 := 
  sorry

end cos_squared_identity_l277_277438


namespace area_of_ABCD_l277_277001

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l277_277001


namespace candies_in_box_more_than_pockets_l277_277213

theorem candies_in_box_more_than_pockets (x : ℕ) : 
  let initial_pockets := 2 * x
  let pockets_after_return := 2 * (x - 6)
  let candies_returned_to_box := 12
  let total_candies_after_return := initial_pockets + candies_returned_to_box
  (total_candies_after_return - pockets_after_return) = 24 :=
by
  sorry

end candies_in_box_more_than_pockets_l277_277213


namespace unoccupied_volume_proof_l277_277192

-- Definitions based on conditions
def tank_length : ℕ := 12
def tank_width : ℕ := 10
def tank_height : ℕ := 8
def tank_volume : ℕ := tank_length * tank_width * tank_height

def oil_fill_ratio : ℚ := 2 / 3
def ice_cube_volume : ℕ := 1
def number_of_ice_cubes : ℕ := 15

-- Volume calculations
def oil_volume : ℚ := oil_fill_ratio * tank_volume
def total_ice_volume : ℚ := number_of_ice_cubes * ice_cube_volume
def occupied_volume : ℚ := oil_volume + total_ice_volume

-- The final question to be proved
theorem unoccupied_volume_proof : tank_volume - occupied_volume = 305 := by
  sorry

end unoccupied_volume_proof_l277_277192


namespace smallest_multiple_not_20_25_l277_277521

theorem smallest_multiple_not_20_25 :
  ∃ n : ℕ, n > 0 ∧ n % Nat.lcm 50 75 = 0 ∧ n % 20 ≠ 0 ∧ n % 25 ≠ 0 ∧ n = 750 :=
by
  sorry

end smallest_multiple_not_20_25_l277_277521


namespace div_identity_l277_277247

theorem div_identity :
  let a := 6 / 2
  let b := a * 3
  120 / b = 120 / 9 :=
by
  sorry

end div_identity_l277_277247


namespace table_capacity_l277_277799

theorem table_capacity :
  ∀ (n_invited no_show tables : ℕ), n_invited = 47 → no_show = 7 → tables = 8 → 
  (n_invited - no_show) / tables = 5 := by
  intros n_invited no_show tables h_invited h_no_show h_tables
  sorry

end table_capacity_l277_277799


namespace inequality_holds_l277_277216

theorem inequality_holds (a b : ℝ) (ha : 0 ≤ a) (ha' : a ≤ 1) (hb : 0 ≤ b) (hb' : b ≤ 1) : 
  a^5 + b^3 + (a - b)^2 ≤ 2 :=
sorry

end inequality_holds_l277_277216


namespace problem_solution_l277_277907

noncomputable def solveSystem : Prop :=
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 : ℝ),
    (x1 + x2 + x3 = 6) ∧
    (x2 + x3 + x4 = 9) ∧
    (x3 + x4 + x5 = 3) ∧
    (x4 + x5 + x6 = -3) ∧
    (x5 + x6 + x7 = -9) ∧
    (x6 + x7 + x8 = -6) ∧
    (x7 + x8 + x1 = -2) ∧
    (x8 + x1 + x2 = 2) ∧
    (x1 = 1) ∧
    (x2 = 2) ∧
    (x3 = 3) ∧
    (x4 = 4) ∧
    (x5 = -4) ∧
    (x6 = -3) ∧
    (x7 = -2) ∧
    (x8 = -1)

theorem problem_solution : solveSystem :=
by
  -- Skip the proof for now
  sorry

end problem_solution_l277_277907


namespace most_probable_germinated_seeds_l277_277228

noncomputable theory
open_locale big_operators

def binom_prob {n : ℕ} (p : ℝ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem most_probable_germinated_seeds (n : ℕ) (p : ℝ) (h_n : n = 9) (h_p : p = 0.8) :
  let k₀ := (range (n + 1)).max_on (binom_prob p) in k₀ = 7 ∨ k₀ = 8 :=
sorry

end most_probable_germinated_seeds_l277_277228


namespace remainder_when_divided_by_4x_minus_8_l277_277095

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30

-- Define the divisor d(x)
def d (x : ℝ) : ℝ := 4 * x - 8

-- The specific value where the remainder theorem applies (root of d(x) = 0 is x = 2)
def x₀ : ℝ := 2

-- Prove the remainder when p(x) is divided by d(x) is 10
theorem remainder_when_divided_by_4x_minus_8 :
  (p x₀ = 10) :=
by
  -- The proof will be filled in here.
  sorry

end remainder_when_divided_by_4x_minus_8_l277_277095


namespace car_speed_in_kmh_l277_277116

theorem car_speed_in_kmh (rev_per_min : ℕ) (circumference : ℕ) (speed : ℕ) 
  (h1 : rev_per_min = 400) (h2 : circumference = 4) : speed = 96 :=
  sorry

end car_speed_in_kmh_l277_277116


namespace jane_rejects_percent_l277_277893

theorem jane_rejects_percent :
  -- Declare the conditions as hypotheses
  ∀ (P : ℝ) (J : ℝ) (john_frac_reject : ℝ) (total_reject_percent : ℝ) (jane_inspect_frac : ℝ),
  john_frac_reject = 0.005 →
  total_reject_percent = 0.0075 →
  jane_inspect_frac = 5 / 6 →
  -- Given the rejection equation
  (john_frac_reject * (1 / 6) * P + (J / 100) * jane_inspect_frac * P = total_reject_percent * P) →
  -- Prove that Jane rejected 0.8% of the products she inspected
  J = 0.8 :=
by {
  sorry
}

end jane_rejects_percent_l277_277893


namespace total_supermarkets_FGH_chain_l277_277932

def supermarkets_us : ℕ := 47
def supermarkets_difference : ℕ := 10
def supermarkets_canada : ℕ := supermarkets_us - supermarkets_difference
def total_supermarkets : ℕ := supermarkets_us + supermarkets_canada

theorem total_supermarkets_FGH_chain : total_supermarkets = 84 :=
by 
  sorry

end total_supermarkets_FGH_chain_l277_277932


namespace original_wattage_l277_277675

theorem original_wattage (W : ℝ) (h1 : 143 = 1.30 * W) : W = 110 := 
by
  sorry

end original_wattage_l277_277675


namespace find_f_neg_2_l277_277712

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 3^x - 1 else sorry -- we'll define this not for non-negative x properly later

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_f_neg_2 (hodd : is_odd_function f) (hpos : ∀ x : ℝ, 0 ≤ x → f x = 3^x - 1) :
  f (-2) = -8 :=
by
  -- Proof omitted
  sorry

end find_f_neg_2_l277_277712


namespace set_non_neg_even_set_primes_up_to_10_eq_sol_set_l277_277562

noncomputable def non_neg_even (x : ℕ) : Prop := x % 2 = 0 ∧ x ≤ 10
def primes_up_to_10 (x : ℕ) : Prop := Nat.Prime x ∧ x ≤ 10
def eq_sol (x : ℤ) : Prop := x^2 + 2*x - 15 = 0

theorem set_non_neg_even :
  {x : ℕ | non_neg_even x} = {0, 2, 4, 6, 8, 10} := by
  sorry

theorem set_primes_up_to_10 :
  {x : ℕ | primes_up_to_10 x} = {2, 3, 5, 7} := by
  sorry

theorem eq_sol_set :
  {x : ℤ | eq_sol x} = {-5, 3} := by
  sorry

end set_non_neg_even_set_primes_up_to_10_eq_sol_set_l277_277562


namespace smallest_base_to_represent_124_with_three_digits_l277_277781

theorem smallest_base_to_represent_124_with_three_digits : 
  ∃ (b : ℕ), b^2 ≤ 124 ∧ 124 < b^3 ∧ ∀ c, (c^2 ≤ 124 ∧ 124 < c^3) → (5 ≤ c) :=
by
  sorry

end smallest_base_to_represent_124_with_three_digits_l277_277781


namespace luke_good_games_l277_277484

-- Definitions
def bought_from_friend : ℕ := 2
def bought_from_garage_sale : ℕ := 2
def defective_games : ℕ := 2

-- The theorem we want to prove
theorem luke_good_games :
  bought_from_friend + bought_from_garage_sale - defective_games = 2 := 
by 
  sorry

end luke_good_games_l277_277484


namespace ral_current_age_l277_277057

theorem ral_current_age (Ral_age Suri_age : ℕ) (h1 : Ral_age = 2 * Suri_age) (h2 : Suri_age + 3 = 16) : Ral_age = 26 :=
by {
  -- Proof goes here
  sorry
}

end ral_current_age_l277_277057


namespace percent_red_prob_l277_277195

-- Define the conditions
def initial_red := 2
def initial_blue := 4
def additional_red := 2
def additional_blue := 2
def total_balloons := initial_red + initial_blue + additional_red + additional_blue
def total_red := initial_red + additional_red

-- State the theorem
theorem percent_red_prob : (total_red.toFloat / total_balloons.toFloat) * 100 = 40 :=
by
  sorry

end percent_red_prob_l277_277195


namespace distinct_points_4_l277_277998

theorem distinct_points_4 (x y : ℝ) :
  (x + y = 7 ∨ 3 * x - 2 * y = -6) ∧ (x - y = -2 ∨ 4 * x + y = 10) →
  (x, y) =
    (5 / 2, 9 / 2) ∨ 
    (x, y) = (1, 6) ∨
    (x, y) = (-2, 0) ∨ 
    (x, y) = (14 / 11, 74 / 11) :=
sorry

end distinct_points_4_l277_277998


namespace probability_correct_answers_at_least_half_l277_277069

theorem probability_correct_answers_at_least_half :
  let n := 16
  let k := 8
  let p := 3 / 4
  let threshold := 0.999
  binomial_cdf_complement n p k ≤ threshold :=
by
  sorry

end probability_correct_answers_at_least_half_l277_277069


namespace find_interest_rate_l277_277429

noncomputable def compoundInterestRate (P A : ℝ) (t : ℕ) : ℝ := 
  ((A / P) ^ (1 / t)) - 1

theorem find_interest_rate :
  ∀ (P A : ℝ) (t : ℕ),
    P = 1200 → 
    A = 1200 + 873.60 →
    t = 3 →
    compoundInterestRate P A t = 0.2 :=
by
  intros P A t hP hA ht
  sorry

end find_interest_rate_l277_277429


namespace neighbors_receive_28_mangoes_l277_277488

/-- 
  Mr. Wong harvested 560 mangoes. He sold half, gave 50 to his family,
  and divided the remaining mangoes equally among 8 neighbors.
  Each neighbor should receive 28 mangoes.
-/
theorem neighbors_receive_28_mangoes : 
  ∀ (initial : ℕ) (sold : ℕ) (given : ℕ) (neighbors : ℕ), 
  initial = 560 → 
  sold = initial / 2 → 
  given = 50 → 
  neighbors = 8 → 
  (initial - sold - given) / neighbors = 28 := 
by 
  intros initial sold given neighbors
  sorry

end neighbors_receive_28_mangoes_l277_277488


namespace initial_salty_cookies_count_l277_277622

-- Define initial conditions
def initial_sweet_cookies : ℕ := 9
def sweet_cookies_ate : ℕ := 36
def salty_cookies_left : ℕ := 3
def salty_cookies_ate : ℕ := 3

-- Theorem to prove the initial salty cookies count
theorem initial_salty_cookies_count (initial_salty_cookies : ℕ) 
    (initial_sweet_cookies : initial_sweet_cookies = 9) 
    (sweet_cookies_ate : sweet_cookies_ate = 36)
    (salty_cookies_ate : salty_cookies_ate = 3) 
    (salty_cookies_left : salty_cookies_left = 3) : 
    initial_salty_cookies = 6 := 
sorry

end initial_salty_cookies_count_l277_277622


namespace binom_15_4_l277_277552

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binom_15_4_l277_277552


namespace intersection_points_A_B_segment_length_MN_l277_277444

section PolarCurves

-- Given conditions
def curve1 (ρ θ : ℝ) : Prop := ρ^2 * Real.cos (2 * θ) = 8
def curve2 (θ : ℝ) : Prop := θ = Real.pi / 6
def is_on_line (x y t : ℝ) : Prop := x = 2 + Real.sqrt 3 / 2 * t ∧ y = 1 / 2 * t

-- Polar coordinates of points A and B
theorem intersection_points_A_B :
  ∃ (ρ₁ ρ₂ θ₁ θ₂ : ℝ), curve1 ρ₁ θ₁ ∧ curve2 θ₁ ∧ curve1 ρ₂ θ₂ ∧ curve2 θ₂ ∧
    (ρ₁, θ₁) = (4, Real.pi / 6) ∧ (ρ₂, θ₂) = (4, -Real.pi / 6) :=
sorry

-- Length of the segment MN
theorem segment_length_MN :
  ∀ t : ℝ, curve1 (2 + Real.sqrt 3 / 2 * t) (1 / 2 * t) →
    ∃ t₁ t₂ : ℝ, (is_on_line (2 + Real.sqrt 3 / 2 * t₁) (1 / 2 * t₁) t₁) ∧
                (is_on_line (2 + Real.sqrt 3 / 2 * t₂) (1 / 2 * t₂) t₂) ∧
                Real.sqrt ((2 * -Real.sqrt 3 * 4)^2 - 4 * (-8)) = 4 * Real.sqrt 5 :=
sorry

end PolarCurves

end intersection_points_A_B_segment_length_MN_l277_277444


namespace percent_red_prob_l277_277194

-- Define the conditions
def initial_red := 2
def initial_blue := 4
def additional_red := 2
def additional_blue := 2
def total_balloons := initial_red + initial_blue + additional_red + additional_blue
def total_red := initial_red + additional_red

-- State the theorem
theorem percent_red_prob : (total_red.toFloat / total_balloons.toFloat) * 100 = 40 :=
by
  sorry

end percent_red_prob_l277_277194


namespace limit_S_over_V_l277_277607

noncomputable theory

open Real

-- Define the curve C
def curve (x : ℝ) : ℝ := 1 / x

-- Define tangent line at A(1, 1)
def tangent_linear_equation (x y : ℝ) : Prop := x + y = 2

-- Define point P on the curve C
def point_P (t : ℝ) (ht : t > 0) : ℝ × ℝ := (t, 1 / t)

-- Define parallel line m passing through point P
def parallel_line_m (t x y : ℝ) (ht : t > 0) : Prop := y = -x + t + 1 / t

-- Define intersection point Q of line m and curve C
def point_Q (t : ℝ) (ht : t > 0) : ℝ × ℝ := (1 / t, t)

-- Define the area S
def area_S (t : ℝ) (ht : t > 0) : ℝ := 2 * abs (log t)

-- Define the volume V
def volume_V (t : ℝ) (ht : t > 0) : ℝ := π * abs (t - 1 / t)

-- Final limit statement
theorem limit_S_over_V : tendsto (fun (t : ℝ) => 2 * abs (log t) / (π * abs (t - 1 / t))) (𝓝[<] 1) (𝓝 (2 / π)) := 
sorry

end limit_S_over_V_l277_277607


namespace rectangle_perimeter_eq_l277_277217

noncomputable def rectangle_perimeter (x y : ℝ) := 2 * (x + y)

theorem rectangle_perimeter_eq (x y a b : ℝ)
  (h_area_rect : x * y = 2450)
  (h_area_ellipse : a * b = 2450)
  (h_foci_distance : x + y = 2 * a)
  (h_diag : x^2 + y^2 = 4 * (a^2 - b^2))
  (h_b : b = Real.sqrt (a^2 - 1225))
  : rectangle_perimeter x y = 120 * Real.sqrt 17 := by
  sorry

end rectangle_perimeter_eq_l277_277217


namespace sum_of_three_numbers_l277_277514

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + c * a = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_l277_277514


namespace batsman_new_average_l277_277102

def batsman_average_after_16_innings (A : ℕ) (new_avg : ℕ) (runs_16th : ℕ) : Prop :=
  15 * A + runs_16th = 16 * new_avg

theorem batsman_new_average (A : ℕ) (runs_16th : ℕ) (h1 : batsman_average_after_16_innings A (A + 3) runs_16th) : A + 3 = 19 :=
by
  sorry

end batsman_new_average_l277_277102


namespace apple_price_33_kgs_l277_277269

theorem apple_price_33_kgs (l q : ℕ) (h1 : 30 * l + 6 * q = 366) (h2 : 15 * l = 150) : 
  30 * l + 3 * q = 333 :=
by
  sorry

end apple_price_33_kgs_l277_277269


namespace ratio_RN_NS_l277_277313

noncomputable def ratio_RN_NS_is_one_to_one : Prop :=
  let A : (ℝ × ℝ) := (0, 10)
  let B : (ℝ × ℝ) := (10, 10)
  let C : (ℝ × ℝ) := (10, 0)
  let D : (ℝ × ℝ) := (0, 0)
  let F : (ℝ × ℝ) := (3, 0)
  let N : (ℝ × ℝ) := ((0 + 3) / 2, (10 + 0) / 2)
  let slope : ℝ := 3 / 10
  let R : (ℝ × ℝ) := (18, 10)
  let S : (ℝ × ℝ) := (-15, 0)
  let dist := λ P Q : (ℝ × ℝ), real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)
  dist N R = dist N S

theorem ratio_RN_NS : ratio_RN_NS_is_one_to_one := by
  sorry

end ratio_RN_NS_l277_277313


namespace ral_age_is_26_l277_277059

def ral_current_age (suri_age : ℕ) (ral_age : ℕ) : Prop :=
  ral_age = 2 * suri_age

theorem ral_age_is_26 (suri_current_age : ℕ) (ral_current_age : ℕ) (h1 : suri_current_age + 3 = 16) (h2 : ral_age = 2 * suri_age) : ral_current_age = 26 := 
by
  sorry

end ral_age_is_26_l277_277059


namespace students_band_and_chorus_l277_277486

theorem students_band_and_chorus (Total Band Chorus Union Intersection : ℕ) 
  (h₁ : Total = 300) 
  (h₂ : Band = 110) 
  (h₃ : Chorus = 140) 
  (h₄ : Union = 220) :
  Intersection = Band + Chorus - Union :=
by
  -- Given the conditions, the proof would follow here.
  sorry

end students_band_and_chorus_l277_277486


namespace perfect_square_solution_l277_277569

theorem perfect_square_solution (x : ℤ) : 
  ∃ k : ℤ, x^2 - 14 * x - 256 = k^2 ↔ x = 15 ∨ x = -1 :=
by
  sorry

end perfect_square_solution_l277_277569


namespace teams_have_equal_people_l277_277412

-- Definitions capturing the conditions
def managers : Nat := 3
def employees : Nat := 3
def teams : Nat := 3

-- The total number of people
def total_people : Nat := managers + employees

-- The proof statement
theorem teams_have_equal_people : total_people / teams = 2 := by
  sorry

end teams_have_equal_people_l277_277412


namespace inequality_add_l277_277869

theorem inequality_add {a b c : ℝ} (h : a > b) : a + c > b + c :=
sorry

end inequality_add_l277_277869


namespace usable_area_l277_277529

def garden_length : ℕ := 20
def garden_width : ℕ := 18
def pond_side : ℕ := 4

theorem usable_area :
  garden_length * garden_width - pond_side * pond_side = 344 :=
by
  sorry

end usable_area_l277_277529


namespace places_proven_l277_277333

-- Definitions based on the problem conditions
inductive Place
| first
| second
| third
| fourth

def is_boy : String -> Prop
| "Oleg" => True
| "Olya" => False
| "Polya" => False
| "Pasha" => False
| _ => False

def name_starts_with_O : String -> Prop
| n => (n.head! = 'O')

noncomputable def determine_places : Prop :=
  ∃ (olegs_place olyas_place polyas_place pashas_place : Place),
  -- Statements and truth conditions
  ∃ (truthful : String), truthful ∈ ["Oleg", "Olya", "Polya", "Pasha"] ∧ 
  ∀ (person : String), 
    (person ≠ truthful → ∀ (statement : Place -> Prop), ¬ statement (person_to_place person)) ∧
    (person = truthful → person_to_place person = Place.first) ∧
    (person = truthful → 
      match person with
        | "Olya" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → is_boy (place_to_person p)
        | "Oleg" => ∃ (p : Place), (person_to_place "Oleg" = p ∧ person_to_place "Olya" = succ_place p ∨ 
                                    person_to_place "Olya" = p ∧ person_to_place "Oleg" = succ_place p)
        | "Pasha" => ∀ (p : Place), (p = Place.first ∨ p = Place.third) → name_starts_with_O (place_to_person p)
        | _ => True
      end)

-- Helper functions to relate places to persons
def person_to_place : String -> Place
| "Oleg" => Place.first
| "Olya" => Place.second
| "Polya" => Place.third
| "Pasha" => Place.fourth
| _ => Place.first -- Default, shouldn't happen

def place_to_person : Place -> String
| Place.first => "Oleg"
| Place.second => "Olya"
| Place.third => "Polya"
| Place.fourth => "Pasha"

def succ_place : Place → Place
| Place.first => Place.second
| Place.second => Place.third
| Place.third => Place.fourth
| Place.fourth => Place.first -- No logical next in this context.

theorem places_proven : determine_places :=
by
  sorry

end places_proven_l277_277333


namespace expr_eval_l277_277419

def expr : ℕ := 3 * 3^4 - 9^27 / 9^25

theorem expr_eval : expr = 162 := by
  -- Proof will be written here if needed
  sorry

end expr_eval_l277_277419


namespace female_guests_from_jays_family_l277_277043

theorem female_guests_from_jays_family (total_guests : ℕ) (percent_females : ℝ) (percent_from_jays_family : ℝ)
    (h1 : total_guests = 240) (h2 : percent_females = 0.60) (h3 : percent_from_jays_family = 0.50) :
    total_guests * percent_females * percent_from_jays_family = 72 := by
  sorry

end female_guests_from_jays_family_l277_277043


namespace area_of_ABCD_l277_277003

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l277_277003


namespace least_number_to_multiply_for_multiple_of_112_l277_277638

theorem least_number_to_multiply_for_multiple_of_112 (n : ℕ) : 
  (Nat.lcm 72 112) / 72 = 14 := 
sorry

end least_number_to_multiply_for_multiple_of_112_l277_277638


namespace range_of_m_for_second_quadrant_l277_277580

theorem range_of_m_for_second_quadrant (m : ℝ) :
  (P : ℝ × ℝ) → P = (1 + m, 3) → P.fst < 0 → m < -1 :=
by
  intro P hP hQ
  sorry

end range_of_m_for_second_quadrant_l277_277580


namespace correct_operation_l277_277654

theorem correct_operation :
  (∀ (a : ℤ), 2 * a - a ≠ 1) ∧
  (∀ (a : ℤ), (a^2)^4 ≠ a^6) ∧
  (∀ (a b : ℤ), (a * b)^2 ≠ a * b^2) ∧
  (∀ (a : ℤ), a^3 * a^2 = a^5) :=
by
  sorry

end correct_operation_l277_277654


namespace find_balls_l277_277085

-- Define the variables for the number of red, yellow, and white balls
variables (x y z : ℚ)

-- State the conditions as hypotheses
def conditions (x y z : ℚ) :=
  x + y + z = 160 ∧ 
  x - (x / 3) + y - (y / 4) + z - (z / 5) = 120 ∧ 
  x - (x / 5) + y - (y / 4) + z - (z / 3) = 116

-- The theorem should state that the number of each colored ball can be found
theorem find_balls (x y z : ℚ) (h : conditions x y z) : x = 45 ∧ y = 40 ∧ z = 75 :=
by {
  have h1 : x + y + z = 160 := h.1,
  have h2 : x - (x / 3) + y - (y / 4) + z - (z / 5) = 120 := h.2,
  have h3 : x - (x / 5) + y - (y / 4) + z - (z / 3) = 116 := h.3,
  -- Normally the proof would continue from here
  sorry
}

end find_balls_l277_277085


namespace nat_pairs_satisfy_conditions_l277_277843

theorem nat_pairs_satisfy_conditions :
  ∃ (a b : ℕ), 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∧
  (∃ k : ℤ, (a^2 + b) = k * (b^2 - a)) ∧
  (∃ l : ℤ, (b^2 + a) = l * (a^2 - b)) := 
sorry

end nat_pairs_satisfy_conditions_l277_277843


namespace problem1_problem2_l277_277987

-- Problem 1
theorem problem1 :
  2 * Real.cos (Real.pi / 4) + (Real.pi - Real.sqrt 3)^0 - Real.sqrt 8 = 1 - Real.sqrt 2 := 
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) (h : m ≠ 1) :
  (2 / (m - 1) + 1) / ((2 * m + 2) / (m^2 - 2 * m + 1)) = (m - 1) / 2 :=
by
  sorry

end problem1_problem2_l277_277987


namespace cost_of_each_ruler_l277_277597
-- Import the necessary library

-- Define the conditions and statement
theorem cost_of_each_ruler (students : ℕ) (rulers_each : ℕ) (cost_per_ruler : ℕ) (total_cost : ℕ) 
  (cond1 : students = 42)
  (cond2 : students / 2 < 42 / 2)
  (cond3 : cost_per_ruler > rulers_each)
  (cond4 : students * rulers_each * cost_per_ruler = 2310) : 
  cost_per_ruler = 11 :=
sorry

end cost_of_each_ruler_l277_277597


namespace base7_difference_l277_277695

theorem base7_difference (a b : ℕ) (h₁ : a = 12100) (h₂ : b = 3666) :
  ∃ c, c = 1111 ∧ (a - b = c) := by
sorry

end base7_difference_l277_277695


namespace quarters_in_school_year_l277_277929

variable (students : ℕ) (artworks_per_student_per_quarter : ℕ) (total_artworks : ℕ) (school_years : ℕ)

def number_of_quarters (students : ℕ) (artworks_per_student_per_quarter : ℕ) (total_artworks : ℕ) (school_years : ℕ) : ℕ :=
  (total_artworks / (students * artworks_per_student_per_quarter * school_years))

theorem quarters_in_school_year :
  number_of_quarters 15 2 240 2 = 4 :=
by sorry

end quarters_in_school_year_l277_277929


namespace kelly_initial_sony_games_l277_277198

def nintendo_games : ℕ := 46
def sony_games_given_away : ℕ := 101
def sony_games_left : ℕ := 31

theorem kelly_initial_sony_games :
  sony_games_given_away + sony_games_left = 132 :=
by
  sorry

end kelly_initial_sony_games_l277_277198


namespace perimeter_of_regular_polygon_l277_277442

/-- 
Given a regular polygon with a central angle of 45 degrees and a side length of 5,
the perimeter of the polygon is 40.
-/
theorem perimeter_of_regular_polygon 
  (central_angle : ℝ) (side_length : ℝ) (h1 : central_angle = 45)
  (h2 : side_length = 5) :
  ∃ P, P = 40 :=
by
  sorry

end perimeter_of_regular_polygon_l277_277442


namespace area_of_rectangle_l277_277024

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l277_277024


namespace order_of_numbers_l277_277363

theorem order_of_numbers :
  let a := 6 ^ 0.5
  let b := 0.5 ^ 6
  let c := Real.log 6 / Real.log 0.5
  c < b ∧ b < a :=
by
  sorry

end order_of_numbers_l277_277363


namespace five_n_minus_twelve_mod_nine_l277_277873

theorem five_n_minus_twelve_mod_nine (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end five_n_minus_twelve_mod_nine_l277_277873


namespace min_additional_games_l277_277224

-- Definitions of parameters
def initial_total_games : ℕ := 5
def initial_falcon_wins : ℕ := 2
def win_percentage_threshold : ℚ := 91 / 100

-- Theorem stating the minimum value for N
theorem min_additional_games (N : ℕ) : (initial_falcon_wins + N : ℚ) / (initial_total_games + N : ℚ) ≥ win_percentage_threshold → N ≥ 29 :=
by
  sorry

end min_additional_games_l277_277224


namespace intersection_point_on_square_diagonal_l277_277539

theorem intersection_point_on_square_diagonal (a b c : ℝ) (h : c = (a + b) / 2) :
  (b / 2) = (-a / 2) + c :=
by
  sorry

end intersection_point_on_square_diagonal_l277_277539


namespace sum_of_digits_nine_ab_l277_277604

noncomputable def sum_digits_base_10 (n : ℕ) : ℕ :=
-- Function to compute the sum of digits of a number in base 10
sorry

def a : ℕ := 6 * ((10^1500 - 1) / 9)

def b : ℕ := 3 * ((10^1500 - 1) / 9)

def nine_ab : ℕ := 9 * a * b

theorem sum_of_digits_nine_ab :
  sum_digits_base_10 nine_ab = 13501 :=
sorry

end sum_of_digits_nine_ab_l277_277604


namespace arithmetic_sequence_150th_term_l277_277458

theorem arithmetic_sequence_150th_term :
  let a₁ := 3
  let d := 5
  let n := 150
  (a₁ + (n - 1) * d) = 748 :=
by
  let a₁ := 3
  let d := 5
  let n := 150
  show a₁ + (n - 1) * d = 748
  sorry

end arithmetic_sequence_150th_term_l277_277458


namespace ratio_of_tangent_to_circumference_l277_277753

theorem ratio_of_tangent_to_circumference
  {r x : ℝ}  -- radius of the circle and length of the tangent
  (hT : x = 2 * π * r)  -- given the length of tangent PQ
  (hA : (1 / 2) * x * r = π * r^2)  -- given the area equivalence

  : (x / (2 * π * r)) = 1 :=  -- desired ratio
by
  -- proof omitted, just using sorry to indicate proof
  sorry

end ratio_of_tangent_to_circumference_l277_277753


namespace arithmetic_seq_a2_l277_277439

theorem arithmetic_seq_a2 (a : ℕ → ℤ) (d : ℤ) (h1 : d = -2) 
  (h2 : (a 1 + a 5) / 2 = -1) : 
  a 2 = 1 :=
by
  sorry

end arithmetic_seq_a2_l277_277439


namespace determine_a_l277_277861

noncomputable def imaginary_unit : ℂ := Complex.I

def is_on_y_axis (z : ℂ) : Prop :=
  z.re = 0

theorem determine_a (a : ℝ) : 
  is_on_y_axis (⟨(a - 3 * imaginary_unit.re), -(a - 3 * imaginary_unit.im)⟩ / ⟨(1 - imaginary_unit.re), -(1 - imaginary_unit.im)⟩) → 
  a = -3 :=
sorry

end determine_a_l277_277861


namespace smallest_prime_less_than_perf_square_l277_277776

-- Define a predicate for being a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

-- The main goal
theorem smallest_prime_less_than_perf_square : ∃ n : ℕ, is_prime n ∧ ∃ m : ℕ, n = m^2 - 8 ∧ (∀ k : ℕ, is_prime k ∧ ∃ l : ℕ, k = l^2 - 8 → k ≥ n) :=
begin
  use 17,
  split,
  -- Proof that 17 is a prime number
  {
    unfold is_prime,
    split,
    { exact dec_trivial },
    { intros d hd,
      have h_d : d = 1 ∨ d = 17,
      { cases d,
        { exfalso, linarith, },
        { cases d,
          { left, refl, },
          { right, linarith [Nat.Prime.not_dvd_one 17 hd], }, }, },
      exact h_d, },
  },
  -- Proof that 17 is 8 less than a perfect square and the smallest such prime
  {
    use 5,
    split,
    { refl, },
    { intros k hk,
      cases hk with hk_prime hk_cond,
      cases hk_cond with l hl,
      rw hl,
      have : l ≥ 5,
      { intros,
        linarith, },
      exact this, },
  }
end

end smallest_prime_less_than_perf_square_l277_277776


namespace size_G_eq_size_L_l277_277483

open Finset

def S (n : ℕ) : Finset (ℕ × ℕ) :=
  {x | 1 ≤ x.1 ∧ x.1 < x.2 ∧ x.2 ≤ n}.toFinset

def G (n : ℕ) : Finset (ℕ × ℕ) :=
  {x | 1 ≤ x.1 ∧ x.1 < x.2 ∧ x.2 ≤ n ∧ x.2 > 2 * x.1}.toFinset

def L (n : ℕ) : Finset (ℕ × ℕ) :=
  {x | 1 ≤ x.1 ∧ x.1 < x.2 ∧ x.2 ≤ n ∧ x.2 < 2 * x.1}.toFinset

theorem size_G_eq_size_L (n : ℕ) (hn : n ≥ 3) : 
  (G n).card = (L n).card :=
by
  sorry

end size_G_eq_size_L_l277_277483


namespace hyperbola_symmetric_slopes_l277_277863

/-- 
Let \(M(x_0, y_0)\) and \(N(-x_0, -y_0)\) be points symmetric about the origin on the hyperbola 
\(\frac{x^2}{16} - \frac{y^2}{4} = 1\). Let \(P(x, y)\) be any point on the hyperbola. 
When the slopes \(k_{PM}\) and \(k_{PN}\) both exist, then \(k_{PM} \cdot k_{PN} = \frac{1}{4}\),
independent of the position of \(P\).
-/
theorem hyperbola_symmetric_slopes (x x0 y y0: ℝ) 
  (hP: x^2 / 16 - y^2 / 4 = 1)
  (hM: x0^2 / 16 - y0^2 / 4 = 1)
  (h_slop_M : x ≠ x0)
  (h_slop_N : x ≠ x0):
  ((y - y0) / (x - x0)) * ((y + y0) / (x + x0)) = 1 / 4 := 
sorry

end hyperbola_symmetric_slopes_l277_277863


namespace sum_four_digit_numbers_l277_277286

theorem sum_four_digit_numbers : 
  let digits := [1, 2, 3, 4, 5]
  let perms := digits.permutations
  ∑ p in perms.filter (λ x, x.length = 4), (1000 * x.head + 100 * x[1] + 10 * x[2] + x[3]) = 399960 := 
by sorry

end sum_four_digit_numbers_l277_277286


namespace competition_places_l277_277334

def participants := ["Olya", "Oleg", "Polya", "Pasha"]
def placements := Array.range 1 5

-- Define statements made by each child
def Olya_claims_odd_boys (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → (name = "Oleg" ∨ name = "Pasha")

def Oleg_claims_consecutive_with_Olya (placement : String → Nat) : Prop :=
  abs (placement "Oleg" - placement "Olya") = 1

def Pasha_claims_odd_O_names (placement : String → Nat) : Prop :=
  ∀ name, (placement name % 2 = 1) → name.startsWith "O"

-- Define the main problem statement
theorem competition_places :
  ∃ (placement : String → Nat),
    placement "Oleg" = 1 ∧
    placement "Olya" = 2 ∧
    placement "Polya" = 3 ∧
    placement "Pasha" = 4 ∧
    (∃ name, (name = "Oleg" ∨ name = "Olya" ∨ name = "Polya" ∨ name = "Pasha") ∧
      ((name = "Oleg" → (placement "Oleg" = 1 ∧ Oleg_claims_consecutive_with_Olya placement)) ∧
       (name = "Olya" → (placement "Olya" = 1 ∧ Olya_claims_odd_boys placement)) ∧
       (name = "Pasha" → (placement "Pasha" = 1 ∧ Pasha_claims_odd_O_names placement)))) :=
by
  have placement : String → Nat := λ name => match name with
    | "Olya" => 2
    | "Oleg" => 1
    | "Polya" => 3
    | "Pasha" => 4
    | _      => 0
  use placement
  simp [placement, Oleg_claims_consecutive_with_Olya, Olya_claims_odd_boys, Pasha_claims_odd_O_names]
  sorry

end competition_places_l277_277334


namespace mary_total_nickels_l277_277616

-- Define the initial number of nickels Mary had
def mary_initial_nickels : ℕ := 7

-- Define the number of nickels her dad gave her
def mary_received_nickels : ℕ := 5

-- The goal is to prove the total number of nickels Mary has now is 12
theorem mary_total_nickels : mary_initial_nickels + mary_received_nickels = 12 :=
by
  sorry

end mary_total_nickels_l277_277616


namespace gcd_of_54000_and_36000_l277_277992

theorem gcd_of_54000_and_36000 : Nat.gcd 54000 36000 = 18000 := 
by sorry

end gcd_of_54000_and_36000_l277_277992


namespace find_number_satisfying_equation_l277_277147

theorem find_number_satisfying_equation :
  ∃ x : ℝ, (196 * x^3) / 568 = 43.13380281690141 ∧ x = 5 :=
by
  sorry

end find_number_satisfying_equation_l277_277147


namespace range_of_x_l277_277303

-- Define the function h(a).
def h (a : ℝ) : ℝ := a^2 + 2 * a + 3

-- Define the main theorem
theorem range_of_x (a : ℝ) (x : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) : 
  x^2 + 4 * x - 2 ≤ h a → -5 ≤ x ∧ x ≤ 1 :=
sorry

end range_of_x_l277_277303


namespace max_candies_theorem_l277_277398

-- Defining constants: the number of students and the total number of candies.
def n : ℕ := 40
def T : ℕ := 200

-- Defining the condition that each student takes at least 2 candies.
def min_candies_per_student : ℕ := 2

-- Calculating the minimum total number of candies taken by 39 students.
def min_total_for_39_students := min_candies_per_student * (n - 1)

-- The maximum number of candies one student can take.
def max_candies_one_student_can_take := T - min_total_for_39_students

-- The statement to prove.
theorem max_candies_theorem : max_candies_one_student_can_take = 122 :=
by
  sorry

end max_candies_theorem_l277_277398


namespace simple_interest_rate_l277_277660

def principal : ℕ := 600
def amount : ℕ := 950
def time : ℕ := 5
def expected_rate : ℚ := 11.67

theorem simple_interest_rate (P A T : ℕ) (R : ℚ) :
  P = principal → A = amount → T = time → R = expected_rate →
  (A = P + P * R * T / 100) :=
by
  intros hP hA hT hR
  sorry

end simple_interest_rate_l277_277660


namespace matchstick_game_winner_a_matchstick_game_winner_b_l277_277764

def is_winning_position (pile1 pile2 : Nat) : Bool :=
  (pile1 % 2 = 1) && (pile2 % 2 = 1)

theorem matchstick_game_winner_a : is_winning_position 101 201 = true := 
by
  -- Theorem statement for (101 matches, 201 matches)
  -- The second player wins
  sorry

theorem matchstick_game_winner_b : is_winning_position 100 201 = false := 
by
  -- Theorem statement for (100 matches, 201 matches)
  -- The first player wins
  sorry

end matchstick_game_winner_a_matchstick_game_winner_b_l277_277764


namespace sqrt_E_nature_l277_277894

def E (x : ℤ) : ℤ :=
  let a := x
  let b := x + 1
  let c := a * b
  let d := b * c
  a^2 + b^2 + c^2 + d^2

theorem sqrt_E_nature : ∀ x : ℤ, (∃ n : ℤ, n^2 = E x) ∧ (∃ m : ℤ, m^2 ≠ E x) :=
  by
  sorry

end sqrt_E_nature_l277_277894


namespace remainder_3001_3002_3003_3004_3005_mod_17_l277_277092

theorem remainder_3001_3002_3003_3004_3005_mod_17 :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 12 := by
  sorry

end remainder_3001_3002_3003_3004_3005_mod_17_l277_277092


namespace perfect_square_fraction_l277_277613

open Nat

theorem perfect_square_fraction (a b : ℕ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : (ab + 1) ∣ (a^2 + b^2)) : ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
by 
  sorry

end perfect_square_fraction_l277_277613


namespace square_area_l277_277267

theorem square_area (x : ℚ) (h : 3 * x - 12 = 15 - 2 * x) : (3 * (27 / 5) - 12)^2 = 441 / 25 :=
by
  sorry

end square_area_l277_277267


namespace verify_equation_holds_l277_277658

noncomputable def verify_equation (m n : ℝ) : Prop :=
  1.55 * Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2)) 
  - Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2)) 
  = 2 * Real.sqrt (3 * m - n)

theorem verify_equation_holds (m n : ℝ) (h : 9 * m^2 - n^2 ≥ 0) : verify_equation m n :=
by
  -- Proof goes here. 
  -- Implement the proof as per the solution steps sketched in the problem statement.
  sorry

end verify_equation_holds_l277_277658


namespace prime_factors_of_product_l277_277866

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : List ℕ :=
  -- Assume we have a function that returns a list of prime factors of n
  sorry

def num_distinct_primes (n : ℕ) : ℕ :=
  (prime_factors n).toFinset.card

theorem prime_factors_of_product :
  num_distinct_primes (85 * 87 * 91 * 94) = 8 :=
by
  have prod_factorizations : 85 = 5 * 17 ∧ 87 = 3 * 29 ∧ 91 = 7 * 13 ∧ 94 = 2 * 47 := 
    by sorry -- each factorization step
  sorry

end prime_factors_of_product_l277_277866


namespace pyramid_side_length_l277_277633

noncomputable def side_length_of_square_base (area_of_lateral_face : ℝ) (slant_height : ℝ) : ℝ :=
  2 * area_of_lateral_face / slant_height

theorem pyramid_side_length 
  (area_of_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_of_lateral_face = 120)
  (h2 : slant_height = 24) :
  side_length_of_square_base area_of_lateral_face slant_height = 10 :=
by
  -- Skipping the proof details.
  sorry

end pyramid_side_length_l277_277633


namespace min_value_of_f_l277_277699

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f : ∃ x : ℝ, (f x = -(1 / Real.exp 1)) ∧ (∀ y : ℝ, f y ≥ f x) := by
  sorry

end min_value_of_f_l277_277699


namespace rain_probability_l277_277083

theorem rain_probability :
  let p := (3:ℚ) / 4 in
  let q :=  1 - p in
  let prob_no_rain_four_days := q ^ 4 in
  let prob_rain_at_least_once := 1 - prob_no_rain_four_days in
  prob_rain_at_least_once = 255 / 256 :=
by
  sorry

end rain_probability_l277_277083


namespace area_rectangle_around_right_triangle_l277_277411

theorem area_rectangle_around_right_triangle (AB BC : ℕ) (hAB : AB = 5) (hBC : BC = 6) :
    let ADE_area := AB * BC
    ADE_area = 30 := by
  sorry

end area_rectangle_around_right_triangle_l277_277411


namespace min_value_sin_cos_expr_l277_277570

open Real

theorem min_value_sin_cos_expr :
  (∀ x : ℝ, sin x ^ 4 + (3 / 2) * cos x ^ 4 ≥ 3 / 5) ∧ 
  (∃ x : ℝ, sin x ^ 4 + (3 / 2) * cos x ^ 4 = 3 / 5) :=
by
  sorry

end min_value_sin_cos_expr_l277_277570


namespace line_contains_diameter_of_circle_l277_277352

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 8 = 0

noncomputable def equation_of_line (x y : ℝ) : Prop :=
  2*x - y - 1 = 0

theorem line_contains_diameter_of_circle :
  (∃ x y : ℝ, equation_of_circle x y ∧ equation_of_line x y) :=
sorry

end line_contains_diameter_of_circle_l277_277352


namespace base_8_addition_l277_277543

-- Definitions
def five_base_8 : ℕ := 5
def thirteen_base_8 : ℕ := 1 * 8 + 3 -- equivalent of (13)_8 in base 10

-- Theorem statement
theorem base_8_addition :
  (five_base_8 + thirteen_base_8) = 2 * 8 + 0 :=
sorry

end base_8_addition_l277_277543


namespace trigonometric_identity_l277_277547

theorem trigonometric_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 := 
by
  sorry

end trigonometric_identity_l277_277547


namespace percentage_mike_has_l277_277618
-- Definitions and conditions
variables (phone_cost : ℝ) (additional_needed : ℝ)
def amount_mike_has := phone_cost - additional_needed

-- Main statement
theorem percentage_mike_has (phone_cost : ℝ) (additional_needed : ℝ) (h1 : phone_cost = 1300) (h2 : additional_needed = 780) : 
  (amount_mike_has phone_cost additional_needed) * 100 / phone_cost = 40 :=
by
  sorry

end percentage_mike_has_l277_277618


namespace B_can_finish_alone_in_27_5_days_l277_277399

-- Definitions of work rates
variable (B A C : Type)

-- Conditions translations
def efficiency_of_A (x : ℝ) : Prop := ∀ (work_rate_A work_rate_B : ℝ), work_rate_A = 1 / (2 * x) ∧ work_rate_B = 1 / x
def efficiency_of_C (x : ℝ) : Prop := ∀ (work_rate_C work_rate_B : ℝ), work_rate_C = 1 / (3 * x) ∧ work_rate_B = 1 / x
def combined_work_rate (x : ℝ) : Prop := (1 / (2 * x) + 1 / x + 1 / (3 * x)) = 1 / 15

-- Proof statement
theorem B_can_finish_alone_in_27_5_days :
  ∃ (x : ℝ), efficiency_of_A x ∧ efficiency_of_C x ∧ combined_work_rate x ∧ x = 27.5 :=
sorry

end B_can_finish_alone_in_27_5_days_l277_277399


namespace legacy_earnings_l277_277394

theorem legacy_earnings 
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (hours_per_room : ℕ)
  (earnings_per_hour : ℕ)
  (total_floors : floors = 4)
  (total_rooms_per_floor : rooms_per_floor = 10)
  (time_per_room : hours_per_room = 6)
  (rate_per_hour : earnings_per_hour = 15) :
  floors * rooms_per_floor * hours_per_room * earnings_per_hour = 3600 := 
by
  sorry

end legacy_earnings_l277_277394


namespace sandwiches_cost_l277_277381

theorem sandwiches_cost (sandwiches sodas : ℝ) 
  (cost_sandwich : ℝ := 2.44)
  (cost_soda : ℝ := 0.87)
  (num_sodas : ℕ := 4)
  (total_cost : ℝ := 8.36)
  (total_soda_cost : ℝ := cost_soda * num_sodas)
  (total_sandwich_cost : ℝ := total_cost - total_soda_cost):
  sandwiches = (total_sandwich_cost / cost_sandwich) → sandwiches = 2 := by 
  sorry

end sandwiches_cost_l277_277381


namespace find_y_l277_277463

def angle_at_W (RWQ RWT QWR TWQ : ℝ) :=  RWQ + RWT + QWR + TWQ

theorem find_y 
  (RWQ RWT QWR TWQ : ℝ)
  (h1 : RWQ = 90) 
  (h2 : RWT = 3 * y)
  (h3 : QWR = y)
  (h4 : TWQ = 90) 
  (h_sum : angle_at_W RWQ RWT QWR TWQ = 360)  
  : y = 67.5 :=
by
  sorry

end find_y_l277_277463


namespace reflect_P_y_axis_l277_277187

def P : ℝ × ℝ := (2, 1)

def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

theorem reflect_P_y_axis :
  reflect_y_axis P = (-2, 1) :=
by
  sorry

end reflect_P_y_axis_l277_277187


namespace merchant_markup_percentage_l277_277531

theorem merchant_markup_percentage (CP MP SP : ℝ) (x : ℝ) (H_CP : CP = 100)
  (H_MP : MP = CP + (x / 100 * CP)) 
  (H_SP_discount : SP = MP * 0.80) 
  (H_SP_profit : SP = CP * 1.12) : 
  x = 40 := 
by
  sorry

end merchant_markup_percentage_l277_277531


namespace gcf_90_108_l277_277246

-- Given two integers 90 and 108
def a : ℕ := 90
def b : ℕ := 108

-- Question: What is the greatest common factor (GCF) of 90 and 108?
theorem gcf_90_108 : Nat.gcd a b = 18 :=
by {
  sorry
}

end gcf_90_108_l277_277246


namespace find_2a_plus_b_l277_277630

open Function

noncomputable def f (a b : ℝ) (x : ℝ) := 2 * a * x - 3 * b
noncomputable def g (x : ℝ) := 5 * x + 4
noncomputable def h (a b : ℝ) (x : ℝ) := g (f a b x)
noncomputable def h_inv (x : ℝ) := 2 * x - 9

theorem find_2a_plus_b (a b : ℝ) (h_comp_inv_eq_id : ∀ x, h a b (h_inv x) = x) :
  2 * a + b = 1 / 15 := 
sorry

end find_2a_plus_b_l277_277630


namespace necessarily_positive_l277_277490

-- Definitions based on given conditions
variables {x y z : ℝ}

-- Stating the problem
theorem necessarily_positive : (0 < x ∧ x < 1) → (-2 < y ∧ y < 0) → (0 < z ∧ z < 1) → (x + y^2 > 0) :=
by
  intros hx hy hz
  sorry

end necessarily_positive_l277_277490


namespace students_suggested_bacon_l277_277546

-- Defining the conditions
def total_students := 310
def mashed_potatoes_students := 185

-- Lean statement for proving the equivalent problem
theorem students_suggested_bacon : total_students - mashed_potatoes_students = 125 := by
  sorry -- Proof is omitted

end students_suggested_bacon_l277_277546


namespace probability_of_4_rainy_days_out_of_6_l277_277966

noncomputable def probability_of_rain_on_given_day : ℝ := 0.5

noncomputable def probability_of_rain_on_exactly_k_days (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem probability_of_4_rainy_days_out_of_6 :
  probability_of_rain_on_exactly_k_days 6 4 probability_of_rain_on_given_day = 0.234375 :=
by
  sorry

end probability_of_4_rainy_days_out_of_6_l277_277966


namespace average_mileage_highway_l277_277410

theorem average_mileage_highway (H : Real) : 
  (∀ d : Real, (d / 7.6) > 23 → false) → 
  (280.6 / 23 = H) → 
  H = 12.2 := by
  sorry

end average_mileage_highway_l277_277410


namespace position_of_term_in_sequence_l277_277369

theorem position_of_term_in_sequence 
    (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : ∀ n, a (n + 1) - a n = 7 * n) :
    ∃ n, a n = 35351 ∧ n = 101 :=
by
  sorry

end position_of_term_in_sequence_l277_277369


namespace area_relation_l277_277544

open Real

noncomputable def S_OMN (a b c d θ : ℝ) : ℝ := 1 / 2 * abs (b * c - a * d) * sin θ
noncomputable def S_ABCD (a b c d θ : ℝ) : ℝ := 2 * abs (b * c - a * d) * sin θ

theorem area_relation (a b c d θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
    4 * (S_OMN a b c d θ) = S_ABCD a b c d θ :=
by
  sorry

end area_relation_l277_277544


namespace remaining_nap_time_is_three_hours_l277_277270

-- Define the flight time and the times spent on various activities
def flight_time_minutes := 11 * 60 + 20
def reading_time_minutes := 2 * 60
def movie_time_minutes := 4 * 60
def dinner_time_minutes := 30
def radio_time_minutes := 40
def game_time_minutes := 60 + 10

-- Calculate the total time spent on activities
def total_activity_time_minutes :=
  reading_time_minutes + movie_time_minutes + dinner_time_minutes + radio_time_minutes + game_time_minutes

-- Calculate the remaining time for a nap
def remaining_nap_time_minutes :=
  flight_time_minutes - total_activity_time_minutes

-- Convert the remaining nap time to hours
def remaining_nap_time_hours :=
  remaining_nap_time_minutes / 60

-- The statement to be proved
theorem remaining_nap_time_is_three_hours :
  remaining_nap_time_hours = 3 := by
  sorry

#check remaining_nap_time_is_three_hours -- This will check if the theorem statement is correct

end remaining_nap_time_is_three_hours_l277_277270


namespace rectangle_area_l277_277009

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l277_277009


namespace remainder_of_greatest_integer_multiple_of_9_no_repeats_l277_277478

noncomputable def greatest_integer_multiple_of_9_no_repeats : ℕ :=
  9876543210 -- this should correspond to the greatest number meeting the criteria, but it's identified via more specific logic in practice

theorem remainder_of_greatest_integer_multiple_of_9_no_repeats : 
  (greatest_integer_multiple_of_9_no_repeats % 1000) = 621 := 
  by sorry

end remainder_of_greatest_integer_multiple_of_9_no_repeats_l277_277478


namespace distance_between_stripes_l277_277813

/-
Problem statement:
Given:
1. The street has parallel curbs 30 feet apart.
2. The length of the curb between the stripes is 10 feet.
3. Each stripe is 60 feet long.

Prove:
The distance between the stripes is 5 feet.
-/

-- Definitions:
def distance_between_curbs : ℝ := 30
def length_between_stripes_on_curb : ℝ := 10
def length_of_each_stripe : ℝ := 60

-- Theorem statement:
theorem distance_between_stripes :
  ∃ d : ℝ, (length_between_stripes_on_curb * distance_between_curbs = length_of_each_stripe * d) ∧ d = 5 :=
by
  sorry

end distance_between_stripes_l277_277813


namespace additional_people_needed_l277_277690

-- Define the initial number of people and time they take to mow the lawn 
def initial_people : ℕ := 8
def initial_time : ℕ := 3

-- Define total person-hours required to mow the lawn
def total_person_hours : ℕ := initial_people * initial_time

-- Define the time in which we want to find out how many people can mow the lawn
def desired_time : ℕ := 2

-- Define the number of people needed in desired_time to mow the lawn
def required_people : ℕ := total_person_hours / desired_time

-- Define the additional people required to mow the lawn in desired_time
def additional_people : ℕ := required_people - initial_people

-- Statement to be proved
theorem additional_people_needed : additional_people = 4 := by
  -- Proof to be filled in
  sorry

end additional_people_needed_l277_277690


namespace harriet_trip_time_l277_277249

theorem harriet_trip_time :
  ∀ (t1 : ℝ) (s1 s2 t2 d : ℝ), 
  t1 = 2.8 ∧ 
  s1 = 110 ∧ 
  s2 = 140 ∧ 
  d = s1 * t1 ∧ 
  t2 = d / s2 → 
  t1 + t2 = 5 :=
by intros t1 s1 s2 t2 d
   sorry

end harriet_trip_time_l277_277249


namespace eval_arith_expression_l277_277123

theorem eval_arith_expression : 2 + 3^2 * 4 - 5 + 6 / 2 = 36 := 
by sorry

end eval_arith_expression_l277_277123


namespace parabola_tangent_xsum_l277_277614

theorem parabola_tangent_xsum
  (p : ℝ) (hp : p > 0) 
  (X_A X_B X_M : ℝ) 
  (hxM_line : ∃ y, y = -2 * p ∧ y = -2 * p)
  (hxA_tangent : ∃ y, y = (X_A / p) * (X_A - X_M) - 2 * p)
  (hxB_tangent : ∃ y, y = (X_B / p) * (X_B - X_M) - 2 * p) :
  2 * X_M = X_A + X_B :=
by
  sorry

end parabola_tangent_xsum_l277_277614


namespace max_price_reduction_l277_277074

theorem max_price_reduction (CP SP : ℝ) (profit_margin : ℝ) (max_reduction : ℝ) :
  CP = 1000 ∧ SP = 1500 ∧ profit_margin = 0.05 → SP - max_reduction = CP * (1 + profit_margin) → max_reduction = 450 :=
by {
  sorry
}

end max_price_reduction_l277_277074


namespace determine_m_l277_277306

-- Define the fractional equation condition
def fractional_eq (m x : ℝ) : Prop := (m/(x - 2) + 2*x/(x - 2) = 1)

-- Define the main theorem statement
theorem determine_m (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ x ≠ 2 ∧ fractional_eq m x) : m = -4 :=
sorry

end determine_m_l277_277306


namespace value_of_expr_l277_277382

theorem value_of_expr : (365^2 - 349^2) / 16 = 714 := by
  sorry

end value_of_expr_l277_277382


namespace cost_of_45_daffodils_equals_75_l277_277274

-- Conditions
def cost_of_15_daffodils : ℝ := 25
def number_of_daffodils_in_bouquet_15 : ℕ := 15
def number_of_daffodils_in_bouquet_45 : ℕ := 45
def directly_proportional (n m : ℕ) (c_n c_m : ℝ) : Prop := c_n / n = c_m / m

-- Statement to prove
theorem cost_of_45_daffodils_equals_75 :
  ∀ (c : ℝ), directly_proportional number_of_daffodils_in_bouquet_45 number_of_daffodils_in_bouquet_15 c cost_of_15_daffodils → c = 75 :=
by
  intro c hypothesis
  -- Proof would go here.
  sorry

end cost_of_45_daffodils_equals_75_l277_277274


namespace legacy_earnings_l277_277393

theorem legacy_earnings 
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (hours_per_room : ℕ)
  (earnings_per_hour : ℕ)
  (total_floors : floors = 4)
  (total_rooms_per_floor : rooms_per_floor = 10)
  (time_per_room : hours_per_room = 6)
  (rate_per_hour : earnings_per_hour = 15) :
  floors * rooms_per_floor * hours_per_room * earnings_per_hour = 3600 := 
by
  sorry

end legacy_earnings_l277_277393


namespace rectangle_area_l277_277021

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l277_277021


namespace find_investment_duration_l277_277696

theorem find_investment_duration :
  ∀ (A P R I : ℝ) (T : ℝ),
    A = 1344 →
    P = 1200 →
    R = 5 →
    I = A - P →
    I = (P * R * T) / 100 →
    T = 2.4 :=
by
  intros A P R I T hA hP hR hI1 hI2
  sorry

end find_investment_duration_l277_277696


namespace bad_arrangement_count_l277_277361

open List

-- Define the concept of an arrangement being "bad"
def bad_arrangement (l : List ℕ) : Prop :=
  (∀ n, n ∈ range 1 22 → 
    ¬∃ (k : ℕ) (s : List ℕ), s.sum = n ∧ s.length = k ∧ l.rotate k.is_cycle) 

-- Define the count of distinct bad arrangements
def num_bad_arrangements := 
  {l : List ℕ // l.perm [1, 2, 3, 4, 5, 6] 
    ∧ bad_arrangement l}

theorem bad_arrangement_count : 
  Fintype.card num_bad_arrangements = 3 :=
sorry

end bad_arrangement_count_l277_277361


namespace find_two_digit_number_l277_277201

def product_of_digits (n : ℕ) : ℕ := 
-- Implementation that calculates the product of the digits of n
sorry

def sum_of_digits (n : ℕ) : ℕ := 
-- Implementation that calculates the sum of the digits of n
sorry

theorem find_two_digit_number (M : ℕ) (h1 : 10 ≤ M ∧ M < 100) (h2 : M = product_of_digits M + sum_of_digits M + 1) : M = 18 :=
by
  sorry

end find_two_digit_number_l277_277201


namespace max_marks_l277_277650

theorem max_marks (M : ℝ) (h : 0.92 * M = 460) : M = 500 :=
by
  sorry

end max_marks_l277_277650


namespace y_at_x_eq_120_l277_277769

@[simp] def custom_op (a b : ℕ) : ℕ := List.prod (List.map (λ i => a + i) (List.range b))

theorem y_at_x_eq_120 {x y : ℕ}
  (h1 : custom_op x (custom_op y 2) = 420)
  (h2 : x = 4)
  (h3 : y = 2) :
  custom_op y x = 120 := by
  sorry

end y_at_x_eq_120_l277_277769


namespace previous_spider_weight_l277_277801

noncomputable def giant_spider_weight (prev_spider_weight : ℝ) : ℝ :=
  2.5 * prev_spider_weight

noncomputable def leg_cross_sectional_area : ℝ := 0.5
noncomputable def leg_pressure : ℝ := 4
noncomputable def legs : ℕ := 8

noncomputable def force_per_leg : ℝ := leg_pressure * leg_cross_sectional_area
noncomputable def total_weight : ℝ := force_per_leg * (legs : ℝ)

theorem previous_spider_weight (prev_spider_weight : ℝ) (h_giant : giant_spider_weight prev_spider_weight = total_weight) : prev_spider_weight = 6.4 :=
by
  sorry

end previous_spider_weight_l277_277801


namespace bigger_number_in_ratio_l277_277517

theorem bigger_number_in_ratio (x : ℕ) (h : 11 * x = 143) : 8 * x = 104 :=
by
  sorry

end bigger_number_in_ratio_l277_277517


namespace old_manufacturing_cost_l277_277126

theorem old_manufacturing_cost (P : ℝ) (h1 : 50 = 0.50 * P) : 0.60 * P = 60 :=
by
  sorry

end old_manufacturing_cost_l277_277126


namespace circle_area_percentage_increase_l277_277308

theorem circle_area_percentage_increase (r : ℝ) (h : r > 0) :
  let original_area := (Real.pi * r^2)
  let new_radius := (2.5 * r)
  let new_area := (Real.pi * new_radius^2)
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  percentage_increase = 525 := by
  let original_area := Real.pi * r^2
  let new_radius := 2.5 * r
  let new_area := Real.pi * new_radius^2
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  sorry

end circle_area_percentage_increase_l277_277308


namespace verify_integer_pairs_l277_277836

open Nat

theorem verify_integer_pairs (a b : ℕ) :
  (∃ k1 : ℤ, ↑(a^2) + ↑b = k1 * (↑(b^2) - ↑a)) ∧
  (∃ k2 : ℤ, ↑(b^2) + ↑a = k2 * (↑(a^2) - ↑b)) →
  (a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ 
  (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3) :=
sorry

end verify_integer_pairs_l277_277836


namespace equations_have_different_graphs_l277_277789

theorem equations_have_different_graphs :
  ¬(∀ x : ℝ, (2 * (x - 3)) / (x + 3) = 2 * (x - 3) ∧ 
              (x + 3) * ((2 * x^2 - 18) / (x + 3)) = 2 * x^2 - 18 ∧
              (2 * x - 3) = (2 * (x - 3)) ∧ 
              (2 * x - 3) = (2 * x - 3)) :=
by
  sorry

end equations_have_different_graphs_l277_277789


namespace rectangle_perimeter_l277_277211

theorem rectangle_perimeter (w : ℝ) (P : ℝ) (l : ℝ) (A : ℝ) 
  (h1 : l = 18)
  (h2 : A = l * w)
  (h3 : P = 2 * l + 2 * w) 
  (h4 : A + P = 2016) : 
  P = 234 :=
by
  sorry

end rectangle_perimeter_l277_277211


namespace probability_x_plus_y_lt_4_in_square_l277_277262

theorem probability_x_plus_y_lt_4_in_square :
  let square := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  let region := {p : ℝ × ℝ | p ∈ square ∧ p.1 + p.2 < 4}
  (measure_of region / measure_of square) = 7 / 9 := sorry

end probability_x_plus_y_lt_4_in_square_l277_277262


namespace solve_system_l277_277896

theorem solve_system (x y : ℝ) (h1 : x + 3 * y = 20) (h2 : x + y = 10) : x = 5 ∧ y = 5 := 
by 
  sorry

end solve_system_l277_277896


namespace sqrt_of_360000_l277_277501

theorem sqrt_of_360000 : sqrt 360000 = 600 := by
  sorry

end sqrt_of_360000_l277_277501


namespace find_u_value_l277_277082

theorem find_u_value (u : ℤ) : ∀ (y : ℤ → ℤ), 
  (y 2 = 8) → (y 4 = 14) → (y 6 = 20) → 
  (∀ x, (x % 2 = 0) → (y (x + 2) = y x + 6)) → 
  y 18 = u → u = 56 :=
by
  intros y h2 h4 h6 pattern h18
  sorry

end find_u_value_l277_277082


namespace quadratic_sum_solutions_l277_277921

noncomputable def sum_of_solutions (a b c : ℝ) : ℝ := 
  (-b/a)

theorem quadratic_sum_solutions : 
  ∀ x : ℝ, sum_of_solutions 1 (-9) (-45) = 9 := 
by
  intro x
  sorry

end quadratic_sum_solutions_l277_277921


namespace smallest_prime_less_than_perf_square_l277_277775

-- Define a predicate for being a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

-- The main goal
theorem smallest_prime_less_than_perf_square : ∃ n : ℕ, is_prime n ∧ ∃ m : ℕ, n = m^2 - 8 ∧ (∀ k : ℕ, is_prime k ∧ ∃ l : ℕ, k = l^2 - 8 → k ≥ n) :=
begin
  use 17,
  split,
  -- Proof that 17 is a prime number
  {
    unfold is_prime,
    split,
    { exact dec_trivial },
    { intros d hd,
      have h_d : d = 1 ∨ d = 17,
      { cases d,
        { exfalso, linarith, },
        { cases d,
          { left, refl, },
          { right, linarith [Nat.Prime.not_dvd_one 17 hd], }, }, },
      exact h_d, },
  },
  -- Proof that 17 is 8 less than a perfect square and the smallest such prime
  {
    use 5,
    split,
    { refl, },
    { intros k hk,
      cases hk with hk_prime hk_cond,
      cases hk_cond with l hl,
      rw hl,
      have : l ≥ 5,
      { intros,
        linarith, },
      exact this, },
  }
end

end smallest_prime_less_than_perf_square_l277_277775


namespace number_of_two_legged_birds_l277_277462

theorem number_of_two_legged_birds
  (b m i : ℕ)  -- Number of birds (b), mammals (m), and insects (i)
  (h_heads : b + m + i = 300)  -- Condition on total number of heads
  (h_legs : 2 * b + 4 * m + 6 * i = 980)  -- Condition on total number of legs
  : b = 110 :=
by
  sorry

end number_of_two_legged_birds_l277_277462


namespace triangle_area_l277_277664

theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (h_perimeter : perimeter = 40) (h_inradius : inradius = 2.5) : 
  (inradius * (perimeter / 2)) = 50 :=
by
  -- Lean 4 statement code
  sorry

end triangle_area_l277_277664


namespace cadence_worked_old_company_l277_277413

theorem cadence_worked_old_company (y : ℕ) (h1 : (426000 : ℝ) = 
    5000 * 12 * y + 6000 * 12 * (y + 5 / 12)) :
    y = 3 := by
    sorry

end cadence_worked_old_company_l277_277413


namespace ways_to_fifth_floor_l277_277974

theorem ways_to_fifth_floor (floors : ℕ) (staircases : ℕ) (h_floors : floors = 5) (h_staircases : staircases = 2) :
  (staircases ^ (floors - 1)) = 16 :=
by
  rw [h_floors, h_staircases]
  sorry

end ways_to_fifth_floor_l277_277974


namespace average_tickets_per_member_l277_277396

theorem average_tickets_per_member (M F : ℕ) (A_f A_m : ℕ)
  (h1 : A_f = 70) (h2 : r = 1/2) (h3 : A_m = 58) (hF : F = 2 * M) : 
  (198 * M) / (3 * M) = 66 := 
begin
  sorry

end average_tickets_per_member_l277_277396


namespace one_element_in_A_inter_B_range_m_l277_277584

theorem one_element_in_A_inter_B_range_m (m : ℝ) :
  let A := {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = -x^2 + m * x - 1}
  let B := {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = 3 - x ∧ 0 ≤ x ∧ x ≤ 3}
  (∃! p, p ∈ A ∧ p ∈ B) → (m = 3 ∨ m > 10 / 3) :=
by
  sorry

end one_element_in_A_inter_B_range_m_l277_277584


namespace find_b_l277_277177

open Real

noncomputable def triangle_b (a b c : ℝ) (A B C : ℝ) (sin_A sin_B : ℝ) (area : ℝ) : Prop :=
  B < π / 2 ∧
  sin_B = sqrt 7 / 4 ∧
  area = 5 * sqrt 7 / 4 ∧
  sin_A / sin_B = 5 * c / (2 * b) ∧
  a = 5 / 2 * c ∧
  area = 1 / 2 * a * c * sin_B

theorem find_b (a b c : ℝ) (A B C : ℝ) (sin_A sin_B : ℝ) (area : ℝ) :
  triangle_b a b c A B C sin_A sin_B area → b = sqrt 14 := by
  sorry

end find_b_l277_277177


namespace power_func_passes_point_l277_277595

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem power_func_passes_point (f : ℝ → ℝ) (h : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) 
  (h_point : f 9 = 1 / 3) : f 25 = 1 / 5 :=
sorry

end power_func_passes_point_l277_277595


namespace sin_cos_value_sin_minus_cos_value_tan_value_l277_277153

variable (x : ℝ)

theorem sin_cos_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.sin x * Real.cos x = - 12 / 25 := 
sorry

theorem sin_minus_cos_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.sin x - Real.cos x = - 7 / 5 := 
sorry

theorem tan_value 
  (h1 : - (Real.pi / 2) < x) 
  (h2 : x < 0) 
  (h3 : Real.sin x + Real.cos x = 1 / 5) : 
  Real.tan x = - 3 / 4 := 
sorry

end sin_cos_value_sin_minus_cos_value_tan_value_l277_277153


namespace distinguishable_arrangements_l277_277168

theorem distinguishable_arrangements :
  let n := 9
  let n1 := 3
  let n2 := 2
  let n3 := 4
  (Nat.factorial n) / ((Nat.factorial n1) * (Nat.factorial n2) * (Nat.factorial n3)) = 1260 :=
by sorry

end distinguishable_arrangements_l277_277168


namespace relationship_between_c_squared_and_ab_l277_277301

theorem relationship_between_c_squared_and_ab (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_c : c = (a + b) / 2) : 
  c^2 ≥ a * b := 
sorry

end relationship_between_c_squared_and_ab_l277_277301


namespace primeFactors_of_3_pow_6_minus_1_l277_277121

def calcPrimeFactorsSumAndSumOfSquares (n : ℕ) : ℕ × ℕ :=
  let factors := [2, 7, 13]  -- Given directly
  let sum_factors := 2 + 7 + 13
  let sum_squares := 2^2 + 7^2 + 13^2
  (sum_factors, sum_squares)

theorem primeFactors_of_3_pow_6_minus_1 :
  calcPrimeFactorsSumAndSumOfSquares (3^6 - 1) = (22, 222) :=
by
  sorry

end primeFactors_of_3_pow_6_minus_1_l277_277121


namespace rangeOfA_l277_277730

theorem rangeOfA (a : ℝ) : 
  (∃ x : ℝ, 9^x + a * 3^x + 4 = 0) → a ≤ -4 :=
by
  sorry

end rangeOfA_l277_277730


namespace largest_whole_number_l277_277519

theorem largest_whole_number (n : ℤ) (h : (1 : ℝ) / 4 + n / 8 < 2) : n ≤ 13 :=
by {
  sorry
}

end largest_whole_number_l277_277519


namespace wednesday_more_than_half_millet_l277_277744

namespace BirdFeeder

-- Define the initial conditions
def initial_amount_millet (total_seeds : ℚ) : ℚ := 0.4 * total_seeds
def initial_amount_other (total_seeds : ℚ) : ℚ := 0.6 * total_seeds

-- Define the daily consumption
def eaten_millet (millet : ℚ) : ℚ := 0.2 * millet
def eaten_other (other : ℚ) : ℚ := other

-- Define the seed addition every other day
def add_seeds (day : ℕ) (seeds : ℚ) : Prop :=
  day % 2 = 1 → seeds = 1

-- Define the daily update of the millet and other seeds in the feeder
def daily_update (day : ℕ) (millet : ℚ) (other : ℚ) : ℚ × ℚ :=
  let remaining_millet := (1 - 0.2) * millet
  let remaining_other := 0
  if day % 2 = 1 then
    (remaining_millet + initial_amount_millet 1, initial_amount_other 1)
  else
    (remaining_millet, remaining_other)

-- Define the main property to prove
def more_than_half_millet (day : ℕ) (millet : ℚ) (other : ℚ) : Prop :=
  millet > 0.5 * (millet + other)

-- Define the theorem statement
theorem wednesday_more_than_half_millet
  (millet : ℚ := initial_amount_millet 1)
  (other : ℚ := initial_amount_other 1) :
  ∃ day, day = 3 ∧ more_than_half_millet day millet other :=
  by
  sorry

end BirdFeeder

end wednesday_more_than_half_millet_l277_277744


namespace greatest_integer_x_l277_277953

theorem greatest_integer_x :
  ∃ (x : ℤ), (∀ (y : ℤ), (8 : ℝ) / 11 > (x : ℝ) / 15) ∧
    ¬ (8 / 11 > (x + 1 : ℝ) / 15) ∧
    x = 10 :=
by
  sorry

end greatest_integer_x_l277_277953


namespace fractional_part_of_cake_eaten_l277_277657

theorem fractional_part_of_cake_eaten :
  let total_eaten := 1 / 3 + 1 / 3^2 + 1 / 3^3 + 1 / 3^4
  in total_eaten = 40 / 81 :=
by
  sorry

end fractional_part_of_cake_eaten_l277_277657


namespace stephen_total_distance_l277_277627

theorem stephen_total_distance :
  let mountain_height := 40000
  let ascent_fraction := 3 / 4
  let descent_fraction := 2 / 3
  let extra_distance_fraction := 0.10
  let normal_trips := 8
  let harsh_trips := 2
  let ascent_distance := ascent_fraction * mountain_height
  let descent_distance := descent_fraction * ascent_distance
  let normal_trip_distance := ascent_distance + descent_distance
  let harsh_trip_extra_distance := extra_distance_fraction * ascent_distance
  let harsh_trip_distance := ascent_distance + harsh_trip_extra_distance + descent_distance
  let total_normal_distance := normal_trip_distance * normal_trips
  let total_harsh_distance := harsh_trip_distance * harsh_trips
  let total_distance := total_normal_distance + total_harsh_distance
  total_distance = 506000 :=
by
  sorry

end stephen_total_distance_l277_277627


namespace tan_double_angle_l277_277574

theorem tan_double_angle (α β : ℝ) (h1 : Real.tan (α + β) = 7) (h2 : Real.tan (α - β) = 1) : 
  Real.tan (2 * α) = -4/3 :=
by
  sorry

end tan_double_angle_l277_277574


namespace find_first_hour_speed_l277_277513

variable (x : ℝ)

-- Conditions
def speed_second_hour : ℝ := 60
def average_speed_two_hours : ℝ := 102.5

theorem find_first_hour_speed (h1 : average_speed_two_hours = (x + speed_second_hour) / 2) : 
  x = 145 := 
by
  sorry

end find_first_hour_speed_l277_277513


namespace ral_current_age_l277_277056

theorem ral_current_age (Ral_age Suri_age : ℕ) (h1 : Ral_age = 2 * Suri_age) (h2 : Suri_age + 3 = 16) : Ral_age = 26 :=
by {
  -- Proof goes here
  sorry
}

end ral_current_age_l277_277056


namespace remainder_of_sum_of_integers_l277_277910

theorem remainder_of_sum_of_integers (a b c : ℕ)
  (h₁ : a % 30 = 15) (h₂ : b % 30 = 5) (h₃ : c % 30 = 10) :
  (a + b + c) % 30 = 0 := by
  sorry

end remainder_of_sum_of_integers_l277_277910


namespace area_of_rectangle_l277_277026

theorem area_of_rectangle (y : ℕ) (h1 : 4 * (y^2) = 4 * 20^2) (h2 : 8 * y = 160) : 
    4 * (20^2) = 1600 := by 
  sorry -- Skip proof, only statement required

end area_of_rectangle_l277_277026


namespace initial_mixture_volume_is_165_l277_277803

noncomputable def initial_volume_of_mixture (initial_milk_volume initial_water_volume water_added final_milk_water_ratio : ℕ) : ℕ :=
  if (initial_milk_volume + initial_water_volume) = 5 * (initial_milk_volume / 3) &&
     initial_water_volume = 2 * (initial_milk_volume / 3) &&
     water_added = 66 &&
     final_milk_water_ratio = 3 / 4 then
    5 * (initial_milk_volume / 3)
  else
    0

theorem initial_mixture_volume_is_165 :
  ∀ initial_milk_volume initial_water_volume water_added final_milk_water_ratio,
    initial_volume_of_mixture initial_milk_volume initial_water_volume water_added final_milk_water_ratio = 165 :=
by
  intros
  sorry

end initial_mixture_volume_is_165_l277_277803


namespace jimmy_earnings_l277_277606

theorem jimmy_earnings : 
  let price15 := 15
  let price20 := 20
  let discount := 5
  let sale_price15 := price15 - discount
  let sale_price20 := price20 - discount
  let num_low_worth := 4
  let num_high_worth := 1
  num_low_worth * sale_price15 + num_high_worth * sale_price20 = 55 :=
by
  sorry

end jimmy_earnings_l277_277606


namespace range_of_half_alpha_minus_beta_l277_277436

theorem range_of_half_alpha_minus_beta (α β : ℝ) (hα : 1 < α ∧ α < 3) (hβ : -4 < β ∧ β < 2) :
  -3 / 2 < (1 / 2) * α - β ∧ (1 / 2) * α - β < 11 / 2 :=
sorry

end range_of_half_alpha_minus_beta_l277_277436


namespace intersection_M_N_l277_277596

-- Define sets M and N
def M := { x : ℝ | ∃ t : ℝ, x = 2^(-t) }
def N := { y : ℝ | ∃ x : ℝ, y = Real.sin x }

-- Prove the intersection of M and N
theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y ≤ 1 } :=
by sorry

end intersection_M_N_l277_277596


namespace sum_tenth_powers_l277_277743

theorem sum_tenth_powers (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : a^10 + b^10 = 123 :=
  sorry

end sum_tenth_powers_l277_277743


namespace operation_result_l277_277293

theorem operation_result (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : (-3 : ℚ) * (5 : ℚ) = (13 : ℚ) / (11 : ℚ) :=
by
  let op := λ (a b : ℚ), (a - 2 * b) / (2 * a - b)
  have h := op (-3) 5
  sorry

end operation_result_l277_277293


namespace sqrt_simplification_l277_277499

theorem sqrt_simplification : Real.sqrt 360000 = 600 :=
by 
  sorry

end sqrt_simplification_l277_277499


namespace three_digit_number_is_657_l277_277238

theorem three_digit_number_is_657 :
  ∃ (a b c : ℕ), (100 * a + 10 * b + c = 657) ∧ (a + b + c = 18) ∧ (a = b + 1) ∧ (c = b + 2) :=
by
  sorry

end three_digit_number_is_657_l277_277238


namespace solution_set_f_l277_277353

def f (x a b : ℝ) : ℝ := (x - 2) * (a * x + b)

theorem solution_set_f (a b : ℝ) (h1 : b = 2 * a) (h2 : 0 < a) :
  {x | f (2 - x) a b > 0} = {x | x < 0 ∨ 4 < x} :=
by
  sorry

end solution_set_f_l277_277353


namespace find_c_value_l277_277757

def projection_condition (v u : ℝ × ℝ) (c : ℝ) : Prop :=
  let v := (5, c)
  let u := (3, 2)
  let dot_product := (v.fst * u.fst + v.snd * u.snd)
  let norm_u_sq := (u.fst^2 + u.snd^2)
  (dot_product / norm_u_sq) * u.fst = -28 / 13 * u.fst

theorem find_c_value : ∃ c : ℝ, projection_condition (5, c) (3, 2) c :=
by
  use -43 / 2
  unfold projection_condition
  sorry

end find_c_value_l277_277757


namespace probability_all_same_color_l277_277524

theorem probability_all_same_color :
  let red_plates := 7
  let blue_plates := 5
  let total_plates := red_plates + blue_plates
  let total_combinations := Nat.choose total_plates 3
  let red_combinations := Nat.choose red_plates 3
  let blue_combinations := Nat.choose blue_plates 3
  let favorable_combinations := red_combinations + blue_combinations
  let probability := (favorable_combinations : ℚ) / total_combinations
  probability = 9 / 44 :=
by 
  sorry

end probability_all_same_color_l277_277524


namespace part1_part2_l277_277342

-- Statements derived from Step c)
theorem part1 {m : ℝ} (h : ∃ x : ℝ, m - |5 - 2 * x| - |2 * x - 1| = 0) : 4 ≤ m := by
  sorry

theorem part2 {x : ℝ} (hx : |x - 3| + |x + 4| ≤ 8) : -9 / 2 ≤ x ∧ x ≤ 7 / 2 := by
  sorry

end part1_part2_l277_277342


namespace factorize_expression_l277_277424

theorem factorize_expression (a x y : ℝ) : 2 * x * (a - 2) - y * (2 - a) = (a - 2) * (2 * x + y) := 
by 
  sorry

end factorize_expression_l277_277424


namespace three_buses_interval_l277_277943

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l277_277943


namespace triangle_equilateral_of_angle_and_side_sequences_l277_277176

theorem triangle_equilateral_of_angle_and_side_sequences 
  (A B C : ℝ) (a b c : ℝ) 
  (h_angles_arith_seq: B = (A + C) / 2)
  (h_sides_geom_seq : b^2 = a * c) 
  (h_sum_angles : A + B + C = 180) 
  (h_pos_angles : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h_pos_sides : 0 < a ∧ 0 < b ∧ 0 < c) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c :=
by
  sorry

end triangle_equilateral_of_angle_and_side_sequences_l277_277176


namespace faces_painted_morning_l277_277905

def faces_of_cuboid : ℕ := 6
def faces_painted_evening : ℕ := 3

theorem faces_painted_morning : faces_of_cuboid - faces_painted_evening = 3 := 
by 
  sorry

end faces_painted_morning_l277_277905


namespace fourth_person_height_l277_277240

variables (H1 H2 H3 H4 : ℝ)

theorem fourth_person_height :
  H2 = H1 + 2 →
  H3 = H2 + 3 →
  H4 = H3 + 6 →
  H1 + H2 + H3 + H4 = 288 →
  H4 = 78.5 :=
by
  intros h2_def h3_def h4_def total_height
  -- Proof steps would follow here
  sorry

end fourth_person_height_l277_277240


namespace garden_stone_calculation_l277_277337

/-- A rectangular garden with dimensions 15m by 2m and patio stones of dimensions 0.5m by 0.5m requires 120 stones to be fully covered -/
theorem garden_stone_calculation :
  let garden_length := 15
  let garden_width := 2
  let stone_length := 0.5
  let stone_width := 0.5
  let area_garden := garden_length * garden_width
  let area_stone := stone_length * stone_width
  let num_stones := area_garden / area_stone
  num_stones = 120 :=
by
  sorry

end garden_stone_calculation_l277_277337


namespace find_n_l277_277568

theorem find_n : ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 11] ∧ n = 3 :=
by
  sorry

end find_n_l277_277568


namespace find_g_of_2_l277_277227

open Real

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_2
  (H: ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) + x = 1) : g 2 = -1 :=
by
  sorry

end find_g_of_2_l277_277227


namespace total_amount_pqr_l277_277385

theorem total_amount_pqr (p q r : ℕ) (T : ℕ) 
  (hr : r = 2 / 3 * (T - r))
  (hr_value : r = 1600) : 
  T = 4000 :=
by
  sorry

end total_amount_pqr_l277_277385


namespace Amy_homework_time_l277_277256

def mathProblems : Nat := 18
def spellingProblems : Nat := 6
def problemsPerHour : Nat := 4
def totalProblems : Nat := mathProblems + spellingProblems
def totalHours : Nat := totalProblems / problemsPerHour

theorem Amy_homework_time :
  totalHours = 6 := by
  sorry

end Amy_homework_time_l277_277256


namespace avg_growth_rate_l277_277112

theorem avg_growth_rate {a p q x : ℝ} (h_eq : (1 + p) * (1 + q) = (1 + x) ^ 2) : 
  x ≤ (p + q) / 2 := 
by
  sorry

end avg_growth_rate_l277_277112


namespace whole_numbers_between_sqrts_l277_277722

theorem whole_numbers_between_sqrts :
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let start := Nat.ceil lower_bound
  let end_ := Nat.floor upper_bound
  ∃ n, n = end_ - start + 1 ∧ n = 7 := by
  sorry

end whole_numbers_between_sqrts_l277_277722


namespace find_a3_plus_a9_l277_277159

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

-- Conditions stating sequence is arithmetic and a₁ + a₆ + a₁₁ = 3
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def a_1_6_11_sum (a : ℕ → ℝ) : Prop :=
  a 1 + a 6 + a 11 = 3

theorem find_a3_plus_a9 
  (h_arith : is_arithmetic_sequence a d)
  (h_sum : a_1_6_11_sum a) : 
  a 3 + a 9 = 2 := 
sorry

end find_a3_plus_a9_l277_277159


namespace eccentricity_of_hyperbola_l277_277151

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (4:ℝ) * a^2 = c^2) : ℝ :=
  c / a

theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (4:ℝ) * a^2 = c^2) :
  hyperbola_eccentricity a b c ha hb h = 2 :=
by
  sorry


end eccentricity_of_hyperbola_l277_277151


namespace carla_paints_120_square_feet_l277_277680

def totalWork : ℕ := 360
def ratioAlex : ℕ := 3
def ratioBen : ℕ := 5
def ratioCarla : ℕ := 4
def ratioTotal : ℕ := ratioAlex + ratioBen + ratioCarla
def workPerPart : ℕ := totalWork / ratioTotal
def carlasWork : ℕ := ratioCarla * workPerPart

theorem carla_paints_120_square_feet : carlasWork = 120 := by
  sorry

end carla_paints_120_square_feet_l277_277680


namespace ratio_of_shaded_area_l277_277571

-- Definitions
variable (S : Type) [Field S]
variable (square_area shaded_area : S) -- Areas of the square and the shaded regions.
variable (PX XQ : S) -- Lengths such that PX = 3 * XQ.

-- Conditions
axiom condition1 : PX = 3 * XQ
axiom condition2 : shaded_area / square_area = 0.375

-- Goal
theorem ratio_of_shaded_area (PX XQ square_area shaded_area : S) [Field S] 
  (condition1 : PX = 3 * XQ)
  (condition2 : shaded_area / square_area = 0.375) : shaded_area / square_area = 0.375 := 
  by
  sorry

end ratio_of_shaded_area_l277_277571


namespace problem_1_problem_2_l277_277169

theorem problem_1 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ)
  (h : (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) :
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 1 :=
sorry

theorem problem_2 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ)
  (h : (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) :
  a_0 + a_2 + a_4 + a_6 = 365 :=
sorry

end problem_1_problem_2_l277_277169


namespace area_of_triangle_ABC_l277_277355

def Point : Type := (ℝ × ℝ)

def A : Point := (0, 0)
def B : Point := (2, 2)
def C : Point := (2, 0)

def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem area_of_triangle_ABC :
  triangle_area A B C = 2 :=
by
  sorry

end area_of_triangle_ABC_l277_277355


namespace zero_sequence_if_p_balanced_l277_277741

def p_balanced (a : ℕ → ℤ) (p : ℕ) : Prop :=
∀ k : ℕ, (∑ i in Finset.range 50 \ Finset.Ico 0 k, a (i*p + k)) = (∑ i in Finset.Ico 0 k, a (i*p + k))

theorem zero_sequence_if_p_balanced :
  (∀ p ∈ ({3, 5, 7, 11, 13, 17} : Finset ℕ), p_balanced (fun k => if k < 50 then a k else 0) p) → 
  (∀ k, k < 50 → a k = 0) :=
begin
  sorry
end

end zero_sequence_if_p_balanced_l277_277741


namespace entrance_exit_plans_l277_277368

-- Definitions as per the conditions in the problem
def south_gates : Nat := 4
def north_gates : Nat := 3
def west_gates : Nat := 2

-- Conditions translated into Lean definitions
def ways_to_enter := south_gates + north_gates
def ways_to_exit := west_gates + north_gates

-- The theorem to be proved: the number of entrance and exit plans
theorem entrance_exit_plans : ways_to_enter * ways_to_exit = 35 := by
  sorry

end entrance_exit_plans_l277_277368


namespace squares_total_l277_277560

def number_of_squares (figure : Type) : ℕ := sorry

theorem squares_total (figure : Type) : number_of_squares figure = 38 := sorry

end squares_total_l277_277560


namespace donna_paid_165_l277_277534

def original_price : ℝ := 200
def discount_rate : ℝ := 0.25
def tax_rate : ℝ := 0.1

def sale_price := original_price * (1 - discount_rate)
def tax := sale_price * tax_rate
def total_amount_paid := sale_price + tax

theorem donna_paid_165 : total_amount_paid = 165 := by
  sorry

end donna_paid_165_l277_277534


namespace original_average_weight_l277_277375

theorem original_average_weight 
  (W : ℝ)
  (h1 : 7 * W + 110 + 60 = 9 * 78) : 
  W = 76 := 
by
  sorry

end original_average_weight_l277_277375


namespace problem_conditions_l277_277079

theorem problem_conditions (m : ℝ) (hf_pow : m^2 - m - 1 = 1) (hf_inc : m > 0) : m = 2 :=
sorry

end problem_conditions_l277_277079


namespace total_distance_proof_l277_277070

-- Define the conditions
def first_half_time := 20
def second_half_time := 30
def average_time_per_kilometer := 5

-- Calculate the total time
def total_time := first_half_time + second_half_time

-- State the proof problem: prove that the total distance is 10 kilometers
theorem total_distance_proof : 
  (total_time / average_time_per_kilometer) = 10 :=
  by sorry

end total_distance_proof_l277_277070


namespace michael_combinations_l277_277886

-- Conditions defined as variables
variables (n k : ℕ)

-- The combination formula
def combination (n k : ℕ) : ℕ := n.choose k

-- The specific problem instance
theorem michael_combinations : combination 8 3 = 56 := by
  sorry

end michael_combinations_l277_277886


namespace gingerbread_to_bagels_l277_277600

theorem gingerbread_to_bagels (gingerbread drying_rings bagels : ℕ) 
  (h1 : gingerbread = 1 → drying_rings = 6) 
  (h2 : drying_rings = 9 → bagels = 4) 
  (h3 : gingerbread = 3) : bagels = 8 :=
by
  sorry

end gingerbread_to_bagels_l277_277600


namespace probability_x_add_y_lt_4_in_square_l277_277261

noncomputable def square_area : ℝ := 3 * 3

noncomputable def triangle_area : ℝ := (1 / 2) * 2 * 2

noncomputable def region_area : ℝ := square_area - triangle_area

noncomputable def probability (A B : ℝ) : ℝ := A / B

theorem probability_x_add_y_lt_4_in_square :
  probability region_area square_area = 7 / 9 :=
by 
  sorry

end probability_x_add_y_lt_4_in_square_l277_277261


namespace amusement_park_people_l277_277527

theorem amusement_park_people (students adults free : ℕ) (total_people paid : ℕ) :
  students = 194 →
  adults = 235 →
  free = 68 →
  total_people = students + adults →
  paid = total_people - free →
  paid - free = 293 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end amusement_park_people_l277_277527


namespace meeting_point_l277_277315

def same_start (x : ℝ) (y : ℝ) : Prop := x = y

def walk_time (x : ℝ) (y : ℝ) (t : ℝ) : Prop := 
  x * t + y * t = 24

def hector_speed (s : ℝ) : ℝ := s

def jane_speed (s : ℝ) : ℝ := 3 * s

theorem meeting_point (s t : ℝ) :
  same_start 0 0 ∧ walk_time (hector_speed s) (jane_speed s) t → t = 6 / s ∧ (6 : ℝ) = 6 :=
by
  intros h
  sorry

end meeting_point_l277_277315


namespace relayRaceOrders_l277_277199

def countRelayOrders (s1 s2 s3 s4 : String) : Nat :=
  if s1 = "Laura" then
    (if s2 ≠ "Laura" ∧ s3 ≠ "Laura" ∧ s4 ≠ "Laura" then
      if (s2 = "Alice" ∨ s2 = "Bob" ∨ s2 = "Cindy") ∧ 
         (s3 = "Alice" ∨ s3 = "Bob" ∨ s3 = "Cindy") ∧ 
         (s4 = "Alice" ∨ s4 = "Bob" ∨ s4 = "Cindy") then
        if s2 ≠ s3 ∧ s3 ≠ s4 ∧ s2 ≠ s4 then 6 else 0
      else 0
    else 0)
  else 0

theorem relayRaceOrders : countRelayOrders "Laura" "Alice" "Bob" "Cindy" = 6 := 
by sorry

end relayRaceOrders_l277_277199


namespace survey_min_people_l277_277185

theorem survey_min_people (p : ℕ) : 
  (∃ p, ∀ k ∈ [18, 10, 5, 9], k ∣ p) → p = 90 :=
by sorry

end survey_min_people_l277_277185


namespace inverse_proportion_quadrants_l277_277175

theorem inverse_proportion_quadrants (k : ℝ) : (∀ x, x ≠ 0 → ((x < 0 → (2 - k) / x > 0) ∧ (x > 0 → (2 - k) / x < 0))) → k > 2 :=
by sorry

end inverse_proportion_quadrants_l277_277175


namespace least_number_subtracted_l277_277653

theorem least_number_subtracted (n k : ℕ) (h₁ : n = 123457) (h₂ : k = 79) : ∃ r, n % k = r ∧ r = 33 :=
by
  sorry

end least_number_subtracted_l277_277653


namespace jens_son_age_l277_277190

theorem jens_son_age
  (J : ℕ)
  (S : ℕ)
  (h1 : J = 41)
  (h2 : J = 3 * S - 7) :
  S = 16 :=
by
  sorry

end jens_son_age_l277_277190


namespace find_m_eccentricity_l277_277350

theorem find_m_eccentricity :
  (∃ m : ℝ, (m > 0) ∧ (∃ c : ℝ, (c = 4 - m ∧ c = (1 / 2) * 2) ∨ (c = m - 4 ∧ c = (1 / 2) * 2)) ∧
  (m = 3 ∨ m = 16 / 3)) :=
sorry

end find_m_eccentricity_l277_277350


namespace repeating_decimal_as_fraction_l277_277136

-- Define the repeating decimal x as .overline{37}
def x : ℚ := 37 / 99

-- The theorem we need to prove
theorem repeating_decimal_as_fraction : x = 37 / 99 := by
  sorry

end repeating_decimal_as_fraction_l277_277136


namespace distance_AC_100_l277_277949

theorem distance_AC_100 (d_AB : ℝ) (t1 : ℝ) (t2 : ℝ) (AC : ℝ) (CB : ℝ) :
  d_AB = 150 ∧ t1 = 3 ∧ t2 = 12 ∧ d_AB = AC + CB ∧ AC / 3 = CB / 12 → AC = 100 := 
by
  sorry

end distance_AC_100_l277_277949


namespace pile_of_stones_l277_277325

def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

theorem pile_of_stones (n : ℕ) (f : ℕ → ℕ): (∀ i, 1 ≤ f i ∧ f i ≤ n) → 
  (∀ j k, similar_sizes (f j) (f k)) → True :=
by
  simp
  exact true.intro


end pile_of_stones_l277_277325


namespace find_s_2_l277_277040

def t (x : ℝ) : ℝ := 4 * x - 6
def s (y : ℝ) : ℝ := y^2 + 5 * y - 7

theorem find_s_2 : s 2 = 7 := by
  sorry

end find_s_2_l277_277040


namespace unique_intersection_point_l277_277851

def line1 (x y : ℝ) : Prop := 3 * x + 2 * y = 9
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 10
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1
def line5 (x y : ℝ) : Prop := x + y = 4

theorem unique_intersection_point :
  ∃! (p : ℝ × ℝ), 
     line1 p.1 p.2 ∧ 
     line2 p.1 p.2 ∧ 
     line3 p.1 ∧ 
     line4 p.2 ∧ 
     line5 p.1 p.2 :=
sorry

end unique_intersection_point_l277_277851


namespace total_servings_l277_277768

/-- The first jar contains 24 2/3 tablespoons of peanut butter. -/
def first_jar_pb : ℚ := 74 / 3

/-- The second jar contains 19 1/2 tablespoons of peanut butter. -/
def second_jar_pb : ℚ := 39 / 2

/-- One serving size is 3 tablespoons. -/
def serving_size : ℚ := 3

/-- The total servings of peanut butter in both jars is 14 13/18 servings. -/
theorem total_servings : (first_jar_pb + second_jar_pb) / serving_size = 14 + 13 / 18 :=
by
  sorry

end total_servings_l277_277768


namespace max_sum_pyramid_on_hexagonal_face_l277_277745

structure hexagonal_prism :=
(faces_initial : ℕ)
(vertices_initial : ℕ)
(edges_initial : ℕ)

structure pyramid_added :=
(faces_total : ℕ)
(vertices_total : ℕ)
(edges_total : ℕ)
(total_sum : ℕ)

theorem max_sum_pyramid_on_hexagonal_face (h : hexagonal_prism) :
  (h = ⟨8, 12, 18⟩) →
  ∃ p : pyramid_added, 
    p = ⟨13, 13, 24, 50⟩ :=
by
  sorry

end max_sum_pyramid_on_hexagonal_face_l277_277745


namespace new_ratio_of_milk_to_water_l277_277183

theorem new_ratio_of_milk_to_water
  (total_volume : ℕ) (initial_ratio_milk : ℕ) (initial_ratio_water : ℕ) (added_water : ℕ)
  (h_total_volume : total_volume = 45)
  (h_initial_ratio : initial_ratio_milk = 4 ∧ initial_ratio_water = 1)
  (h_added_water : added_water = 11) :
  let initial_milk := (initial_ratio_milk * total_volume) / (initial_ratio_milk + initial_ratio_water)
  let initial_water := (initial_ratio_water * total_volume) / (initial_ratio_milk + initial_ratio_water)
  let new_water := initial_water + added_water
  let gcd := Nat.gcd initial_milk new_water
  (initial_milk / gcd : ℕ) = 9 ∧ (new_water / gcd : ℕ) = 5 :=
by
  sorry

end new_ratio_of_milk_to_water_l277_277183


namespace motorcyclist_speed_before_delay_l277_277676

/-- Given conditions and question:
1. The motorcyclist was delayed by 0.4 hours.
2. After the delay, the motorcyclist increased his speed by 10 km/h.
3. The motorcyclist made up for the lost time over a stretch of 80 km.
-/
theorem motorcyclist_speed_before_delay :
  ∃ x : ℝ, (80 / x - 0.4 = 80 / (x + 10)) ∧ x = 40 :=
sorry

end motorcyclist_speed_before_delay_l277_277676


namespace arithmetic_mean_l277_277620

theorem arithmetic_mean (a b c : ℚ) (h₁ : a = 8 / 12) (h₂ : b = 10 / 12) (h₃ : c = 9 / 12) :
  c = (a + b) / 2 :=
by
  sorry

end arithmetic_mean_l277_277620


namespace range_of_a_l277_277856

theorem range_of_a (a : ℝ) : (-1/3 ≤ a) ∧ (a ≤ 2/3) ↔ (∀ x y : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → y = a * x + 1/3) :=
by
  sorry

end range_of_a_l277_277856


namespace probability_corner_within_5_hops_l277_277433

-- Define the problem parameters: grid size and the initial state on an edge
def grid_size : Nat := 4

inductive State
| corner : State
| edge : State (edge_pos : Nat) (edge_pos < 4)
| center : State (x : Nat) (y : Nat) (1 ≤ x ∧ x ≤ 2) (1 ≤ y ∧ y ≤ 2)

-- Define transition probabilities
noncomputable def transition_probability (s1 s2 : State) : ℚ :=
  match s1, s2 with
  | State.edge _, State.corner _ => 1 / 4
  | _, _ => sorry  -- Detail possible transitions (simplified for brevity).

-- Recursive probability definition
noncomputable def p_n (n : Nat) (s : State) : ℚ :=
  match n, s with
  | 0, State.corner _ => 1
  | 0, State.edge _ => 0
  | 0, State.center _ => 0
  | n+1, s => Σ s' , transition_probability s s' * p_n n s'  -- simulate transitions

-- Main problem statement
theorem probability_corner_within_5_hops : p_n 5 (State.edge 0) = 299 / 1024 :=
by
  sorry

end probability_corner_within_5_hops_l277_277433


namespace golden_section_BC_length_l277_277214

-- Definition of a golden section point
def is_golden_section_point (A B C : ℝ) : Prop :=
  ∃ (φ : ℝ), φ = (1 + Real.sqrt 5) / 2 ∧ B = φ * C

-- The given problem translated to Lean
theorem golden_section_BC_length (A B C : ℝ) (h1 : is_golden_section_point A B C) (h2 : B - A = 6) : 
  C - B = 3 * Real.sqrt 5 - 3 ∨ C - B = 9 - 3 * Real.sqrt 5 :=
by
  sorry

end golden_section_BC_length_l277_277214


namespace combination_8_3_l277_277294

theorem combination_8_3 : nat.choose 8 3 = 56 :=
by sorry

end combination_8_3_l277_277294


namespace total_books_together_l277_277904

-- Given conditions
def SamBooks : Nat := 110
def JoanBooks : Nat := 102

-- Theorem to prove the total number of books they have together
theorem total_books_together : SamBooks + JoanBooks = 212 := 
by
  sorry

end total_books_together_l277_277904


namespace find_m_of_inverse_proportion_l277_277305

theorem find_m_of_inverse_proportion (k : ℝ) (m : ℝ) 
(A_cond : (-1) * 3 = k) 
(B_cond : 2 * m = k) : 
m = -3 / 2 := 
by 
  sorry

end find_m_of_inverse_proportion_l277_277305


namespace circle_tangent_independence_l277_277877

noncomputable def e1 (r : ℝ) (β : ℝ) := r * Real.tan β
noncomputable def e2 (r : ℝ) (α : ℝ) := r * Real.tan α
noncomputable def e3 (r : ℝ) (β α : ℝ) := r * Real.tan (β - α)

theorem circle_tangent_independence 
  (O : ℝ) (r β α : ℝ) (hβ : β < π / 2) (hα : 0 < α) (hαβ : α < β) :
  (e1 r β) * (e2 r α) * (e3 r β α) / ((e1 r β) - (e2 r α) - (e3 r β α)) = r^2 :=
by
  sorry

end circle_tangent_independence_l277_277877


namespace series_sum_equality_l277_277821

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, 12^k / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem series_sum_equality : sum_series = 1 := 
by sorry

end series_sum_equality_l277_277821


namespace construct_3x3x3_cube_l277_277991

theorem construct_3x3x3_cube :
  ∃ (cubes_1x2x2 : Finset (Set (Fin 3 × Fin 3 × Fin 3))),
  ∃ (cubes_1x1x1 : Finset (Fin 3 × Fin 3 × Fin 3)),
  cubes_1x2x2.card = 6 ∧ 
  cubes_1x1x1.card = 3 ∧ 
  (∀ c ∈ cubes_1x2x2, ∃ a b : Fin 3, ∀ x, x = (a, b, 0) ∨ x = (a, b, 1) ∨ x = (a, b, 2)) ∧
  (∀ c ∈ cubes_1x1x1, ∃ a b c : Fin 3, ∀ x, x = (a, b, c)) :=
sorry

end construct_3x3x3_cube_l277_277991


namespace true_statement_given_conditions_l277_277855

theorem true_statement_given_conditions (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a < b) :
  |1| / |a| > |1| / |b| := 
by
  sorry

end true_statement_given_conditions_l277_277855


namespace greatest_integer_multiple_of_9_l277_277476

noncomputable def M := 
  max (n : ℕ) (h1 : n % 9 = 0) (h2 : ∀ i j : ℤ, 1 ≤ i < j ≤ nat_digits n → i ≠ j)

theorem greatest_integer_multiple_of_9:
  (∀ i j : ℤ, 1 ≤ i < j ≤ nat_digits M → i ≠ j) 
  → (M % 9 = 0) 
  → (∃ k : ℕ, k = max (n : ℕ), n % 1000 = 981) :=
by
  sorry

#check greatest_integer_multiple_of_9

end greatest_integer_multiple_of_9_l277_277476


namespace men_handshakes_l277_277086

theorem men_handshakes (n : ℕ) (h : n * (n - 1) / 2 = 435) : n = 30 :=
sorry

end men_handshakes_l277_277086


namespace rectangle_area_l277_277010

theorem rectangle_area (y : ℝ) (h1 : 2 * (2 * y) + 2 * (2 * y) = 160) : 
  (2 * y) * (2 * y) = 1600 :=
by
  sorry

end rectangle_area_l277_277010


namespace plane_equivalent_l277_277125

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2*s - 3*t, 1 + s, 4 - 3*s + t)

def plane_equation (x y z : ℝ) : Prop :=
  x - 7*y + 3*z - 8 = 0

theorem plane_equivalent :
  ∃ (s t : ℝ), parametric_plane s t = (x, y, z) ↔ plane_equation x y z :=
by
  sorry

end plane_equivalent_l277_277125


namespace rohan_monthly_salary_l277_277255

theorem rohan_monthly_salary (s : ℝ) 
  (h_food : s * 0.40 = f)
  (h_rent : s * 0.20 = hr) 
  (h_entertainment : s * 0.10 = e)
  (h_conveyance : s * 0.10 = c)
  (h_savings : s * 0.20 = 1000) : 
  s = 5000 := 
sorry

end rohan_monthly_salary_l277_277255


namespace taxi_ride_distance_l277_277593

theorem taxi_ride_distance (initial_fare additional_fare total_fare : ℝ) 
  (initial_distance : ℝ) (additional_distance increment_distance : ℝ) :
  initial_fare = 1.0 →
  additional_fare = 0.45 →
  initial_distance = 1/5 →
  increment_distance = 1/5 →
  total_fare = 7.3 →
  additional_distance = (total_fare - initial_fare) / additional_fare →
  (initial_distance + additional_distance * increment_distance) = 3 := 
by sorry

end taxi_ride_distance_l277_277593


namespace bus_interval_three_buses_l277_277947

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l277_277947


namespace cryptarithm_solution_exists_l277_277064

theorem cryptarithm_solution_exists :
  ∃ (L E S O : ℕ), L ≠ E ∧ L ≠ S ∧ L ≠ O ∧ E ≠ S ∧ E ≠ O ∧ S ≠ O ∧
  (L < 10) ∧ (E < 10) ∧ (S < 10) ∧ (O < 10) ∧
  (1000 * O + 100 * S + 10 * E + L) +
  (100 * S + 10 * E + L) +
  (10 * E + L) +
  L = 10034 ∧
  ((L = 6 ∧ E = 7 ∧ S = 4 ∧ O = 9) ∨
   (L = 6 ∧ E = 7 ∧ S = 9 ∧ O = 8)) :=
by
  -- The proof is omitted here.
  sorry

end cryptarithm_solution_exists_l277_277064


namespace natural_number_pairs_int_l277_277846

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end natural_number_pairs_int_l277_277846


namespace vectors_are_coplanar_l277_277716

-- Definitions of the vectors a, b, and c.
def a (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -2)
def b : ℝ × ℝ × ℝ := (0, 1, 2)
def c : ℝ × ℝ × ℝ := (1, 0, 0)

-- The proof statement 
theorem vectors_are_coplanar (x : ℝ) 
  (h : ∃ m n : ℝ, a x = (n, m, 2 * m)) : 
  x = -1 :=
sorry

end vectors_are_coplanar_l277_277716


namespace cindy_correct_operation_l277_277988

-- Let's define the conditions and proof statement in Lean 4.

variable (x : ℝ)
axiom incorrect_operation : (x - 7) / 5 = 25

theorem cindy_correct_operation :
  (x - 5) / 7 = 18 + 1 / 7 :=
sorry

end cindy_correct_operation_l277_277988


namespace not_both_hit_prob_l277_277816

-- Defining the probabilities
def prob_archer_A_hits : ℚ := 1 / 3
def prob_archer_B_hits : ℚ := 1 / 2

-- Defining event B as both hit the bullseye
def prob_both_hit : ℚ := prob_archer_A_hits * prob_archer_B_hits

-- Defining the complementary event of not both hitting the bullseye
def prob_not_both_hit : ℚ := 1 - prob_both_hit

theorem not_both_hit_prob : prob_not_both_hit = 5 / 6 := by
  -- This is the statement we are trying to prove.
  sorry

end not_both_hit_prob_l277_277816


namespace jesse_blocks_total_l277_277037

-- Define the number of building blocks used for each structure and the remaining blocks
def blocks_building : ℕ := 80
def blocks_farmhouse : ℕ := 123
def blocks_fenced_in_area : ℕ := 57
def blocks_left : ℕ := 84

-- Prove that the total number of building blocks Jesse started with is 344
theorem jesse_blocks_total : blocks_building + blocks_farmhouse + blocks_fenced_in_area + blocks_left = 344 :=
by
  calc
    blocks_building + blocks_farmhouse + blocks_fenced_in_area + blocks_left
      = 80 + 123 + 57 + 84 : by refl
  ... = 260 + 84 : by simp
  ... = 344 : by norm_num

end jesse_blocks_total_l277_277037


namespace least_number_to_multiply_l277_277640

theorem least_number_to_multiply (x : ℕ) :
  (72 * x) % 112 = 0 → x = 14 :=
by 
  sorry

end least_number_to_multiply_l277_277640


namespace table_price_l277_277103

theorem table_price :
  ∃ C T : ℝ, (2 * C + T = 0.6 * (C + 2 * T)) ∧ (C + T = 72) ∧ (T = 63) :=
by
  sorry

end table_price_l277_277103


namespace arithmetic_sequence_prop_l277_277858

theorem arithmetic_sequence_prop (a1 d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 6 > S 7) (h2 : S 7 > S 5)
  (hSn : ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d) :
  (d < 0) ∧ (S 11 > 0) ∧ (|a1 + 5 * d| > |a1 + 6 * d|) := 
by
  sorry

end arithmetic_sequence_prop_l277_277858


namespace find_angle_and_area_l277_277309

theorem find_angle_and_area (a b c : ℝ) (C : ℝ)
  (h₁: (a^2 + b^2 - c^2) * Real.tan C = Real.sqrt 2 * a * b)
  (h₂: c = 2)
  (h₃: b = 2 * Real.sqrt 2) : 
  C = Real.pi / 4 ∧ a = 2 ∧ (∃ S : ℝ, S = 1 / 2 * a * c ∧ S = 2) :=
by
  -- We assume sorry here since the focus is on setting up the problem statement correctly
  sorry

end find_angle_and_area_l277_277309


namespace sin_neg_three_pi_over_four_l277_277994

theorem sin_neg_three_pi_over_four : Real.sin (-3 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_three_pi_over_four_l277_277994


namespace car_distance_problem_l277_277948

theorem car_distance_problem
  (d y z r : ℝ)
  (initial_distance : d = 113)
  (right_turn_distance : y = 15)
  (second_car_distance : z = 35)
  (remaining_distance : r = 28)
  (x : ℝ) :
  2 * x + z + y + r = d → 
  x = 17.5 :=
by
  intros h
  sorry  

end car_distance_problem_l277_277948


namespace exterior_angle_BAC_l277_277540

-- Definitions for the problem conditions
def regular_nonagon_interior_angle :=
  140

def square_interior_angle :=
  90

-- The proof statement
theorem exterior_angle_BAC (regular_nonagon_interior_angle square_interior_angle : ℝ) : 
  regular_nonagon_interior_angle = 140 ∧ square_interior_angle = 90 -> 
  ∃ (BAC : ℝ), BAC = 130 :=
by
  sorry

end exterior_angle_BAC_l277_277540


namespace triangle_side_length_l277_277430

theorem triangle_side_length 
  (side1 : ℕ) (side2 : ℕ) (side3 : ℕ) (P : ℕ)
  (h_side1 : side1 = 5)
  (h_side3 : side3 = 30)
  (h_P : P = 55) :
  side1 + side2 + side3 = P → side2 = 20 :=
by
  intros h
  sorry 

end triangle_side_length_l277_277430


namespace bus_interval_three_buses_l277_277945

theorem bus_interval_three_buses (T : ℕ) (h : T = 21) : (T * 2) / 3 = 14 :=
by
  sorry

end bus_interval_three_buses_l277_277945


namespace solve_linear_system_l277_277626

theorem solve_linear_system :
  ∃ (x y : ℚ), (4 * x - 3 * y = 2) ∧ (6 * x + 5 * y = 1) ∧ (x = 13 / 38) ∧ (y = -4 / 19) :=
by
  sorry

end solve_linear_system_l277_277626


namespace no_such_xy_between_988_and_1991_l277_277388

theorem no_such_xy_between_988_and_1991 :
  ¬ ∃ (x y : ℕ), 988 ≤ x ∧ x < y ∧ y ≤ 1991 ∧ 
  (∃ a b : ℕ, xy = x * y ∧ (xy + x = a^2 ∧ xy + y = b^2)) :=
by
  sorry

end no_such_xy_between_988_and_1991_l277_277388


namespace fraction_of_coins_in_decade_1800_through_1809_l277_277468

theorem fraction_of_coins_in_decade_1800_through_1809 (total_coins : ℕ) (coins_in_decade : ℕ) (c : total_coins = 30) (d : coins_in_decade = 5) : coins_in_decade / (total_coins : ℚ) = 1 / 6 :=
by
  sorry

end fraction_of_coins_in_decade_1800_through_1809_l277_277468


namespace quilt_cost_proof_l277_277470

-- Definitions for conditions
def length := 7
def width := 8
def cost_per_sq_foot := 40

-- Definition for the calculation of the area
def area := length * width

-- Definition for the calculation of the cost
def total_cost := area * cost_per_sq_foot

-- Theorem stating the final proof
theorem quilt_cost_proof : total_cost = 2240 := by
  sorry

end quilt_cost_proof_l277_277470


namespace base_angles_isosceles_triangle_l277_277268

-- Define the conditions
def isIsoscelesTriangle (A B C : ℝ) : Prop :=
  (A = B ∨ B = C ∨ C = A)

def exteriorAngle (A B C : ℝ) (ext_angle : ℝ) : Prop :=
  ext_angle = (180 - (A + B)) ∨ ext_angle = (180 - (B + C)) ∨ ext_angle = (180 - (C + A))

-- Define the theorem
theorem base_angles_isosceles_triangle (A B C : ℝ) (ext_angle : ℝ) :
  isIsoscelesTriangle A B C ∧ exteriorAngle A B C ext_angle ∧ ext_angle = 110 →
  A = 55 ∨ A = 70 ∨ B = 55 ∨ B = 70 ∨ C = 55 ∨ C = 70 :=
by sorry

end base_angles_isosceles_triangle_l277_277268


namespace triangular_weight_60_l277_277924

def round_weight := ℝ
def triangular_weight := ℝ
def rectangular_weight := 90

variables (c t : ℝ)

-- Conditions
axiom condition1 : c + t = 3 * c
axiom condition2 : 4 * c + t = t + c + rectangular_weight

theorem triangular_weight_60 : t = 60 :=
  sorry

end triangular_weight_60_l277_277924


namespace find_k_l277_277170

theorem find_k (k : ℝ) (h : (3:ℝ)^4 + k * (3:ℝ)^2 - 26 = 0) : k = -55 / 9 := 
by sorry

end find_k_l277_277170


namespace pyramid_base_sidelength_l277_277635

theorem pyramid_base_sidelength (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120) (hh : h = 24) (area_eq : A = 1/2 * s * h) : s = 10 := by
  sorry

end pyramid_base_sidelength_l277_277635


namespace bob_repayment_days_l277_277981

theorem bob_repayment_days :
  ∃ (x : ℕ), (15 + 3 * x ≥ 45) ∧ (∀ y : ℕ, (15 + 3 * y ≥ 45) → x ≤ y) ∧ x = 10 := 
by
  sorry

end bob_repayment_days_l277_277981


namespace ral_age_is_26_l277_277061

def ral_current_age (suri_age : ℕ) (ral_age : ℕ) : Prop :=
  ral_age = 2 * suri_age

theorem ral_age_is_26 (suri_current_age : ℕ) (ral_current_age : ℕ) (h1 : suri_current_age + 3 = 16) (h2 : ral_age = 2 * suri_age) : ral_current_age = 26 := 
by
  sorry

end ral_age_is_26_l277_277061


namespace number_of_whole_numbers_between_sqrts_l277_277724

noncomputable def count_whole_numbers_between_sqrts : ℕ :=
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let min_int := Int.ceil lower_bound
  let max_int := Int.floor upper_bound
  Int.natAbs (max_int - min_int + 1)

theorem number_of_whole_numbers_between_sqrts :
  count_whole_numbers_between_sqrts = 7 :=
by
  sorry

end number_of_whole_numbers_between_sqrts_l277_277724


namespace swimming_pool_width_l277_277373

theorem swimming_pool_width (length width vol depth : ℝ) 
  (H_length : length = 60) 
  (H_depth : depth = 0.5) 
  (H_vol_removal : vol = 2250 / 7.48052) 
  (H_vol_eq : vol = (length * width) * depth) : 
  width = 10.019 :=
by
  -- Assuming the correctness of floating-point arithmetic for the purpose of this example
  sorry

end swimming_pool_width_l277_277373


namespace inequality_interval_l277_277688

theorem inequality_interval : ∀ x : ℝ, (x^2 - 3 * x - 4 < 0) ↔ (-1 < x ∧ x < 4) :=
by
  intro x
  sorry

end inequality_interval_l277_277688


namespace length_increase_percentage_l277_277351

theorem length_increase_percentage (L B : ℝ) (x : ℝ) (h1 : (L + (x / 100) * L) * (B - (5 / 100) * B) = 1.14 * L * B) : x = 20 := by 
  sorry

end length_increase_percentage_l277_277351


namespace brad_running_speed_l277_277897

-- Definitions based on the given conditions
def distance_between_homes : ℝ := 24
def maxwell_walking_speed : ℝ := 4
def maxwell_time_to_meet : ℝ := 3

/-- Brad's running speed is 6 km/h given the conditions of the problem. -/
theorem brad_running_speed : (distance_between_homes - (maxwell_walking_speed * maxwell_time_to_meet)) / (maxwell_time_to_meet - 1) = 6 := by
  sorry

end brad_running_speed_l277_277897


namespace tenth_term_arithmetic_sequence_l277_277784

theorem tenth_term_arithmetic_sequence :
  let a_1 := (1 : ℝ) / 2
  let a_2 := (5 : ℝ) / 6
  let d := a_2 - a_1
  (a_1 + 9 * d) = 7 / 2 := 
by
  sorry

end tenth_term_arithmetic_sequence_l277_277784


namespace number_of_multiples_of_6_between_5_and_125_l277_277867

theorem number_of_multiples_of_6_between_5_and_125 : 
  ∃ k : ℕ, (5 < 6 * k ∧ 6 * k < 125) → k = 20 :=
sorry

end number_of_multiples_of_6_between_5_and_125_l277_277867


namespace area_of_region_l277_277132

-- Definitions drawn from conditions
def circle_radius := 36
def num_small_circles := 8

-- Main statement to be proven
theorem area_of_region :
  ∃ K : ℝ, 
    K = π * (circle_radius ^ 2) - num_small_circles * π * ((circle_radius * (Real.sqrt 2 - 1)) ^ 2) ∧
    ⌊ K ⌋ = ⌊ π * (circle_radius ^ 2) - num_small_circles * π * ((circle_radius * (Real.sqrt 2 - 1)) ^ 2) ⌋ :=
  sorry

end area_of_region_l277_277132


namespace five_digit_palindromes_count_l277_277167

theorem five_digit_palindromes_count : 
  ∃ (a b c : Fin 10), (a ≠ 0) ∧ (∃ (count : Nat), count = 9 * 10 * 10 ∧ count = 900) :=
by
  sorry

end five_digit_palindromes_count_l277_277167


namespace rational_numbers_product_power_l277_277440

theorem rational_numbers_product_power (a b : ℚ) (h : |a - 2| + (2 * b + 1)^2 = 0) :
  (a * b)^2013 = -1 :=
sorry

end rational_numbers_product_power_l277_277440


namespace simplify_expression_l277_277960

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (b * c + a * c + a * b) :=
by
  sorry

end simplify_expression_l277_277960


namespace calculate_expression_l277_277685

theorem calculate_expression :
  ((12 ^ 12 / 12 ^ 11) ^ 2 * 4 ^ 2) / 2 ^ 4 = 144 :=
by
  sorry

end calculate_expression_l277_277685


namespace circles_intersect_l277_277507

variable (r1 r2 d : ℝ)
variable (h1 : r1 = 4)
variable (h2 : r2 = 5)
variable (h3 : d = 7)

theorem circles_intersect : 1 < d ∧ d < r1 + r2 :=
by sorry

end circles_intersect_l277_277507


namespace Andrena_more_than_Debelyn_l277_277555

-- Define initial dolls count for each person
def Debelyn_initial_dolls : ℕ := 20
def Christel_initial_dolls : ℕ := 24

-- Define dolls given by Debelyn and Christel
def Debelyn_gift_dolls : ℕ := 2
def Christel_gift_dolls : ℕ := 5

-- Define remaining dolls for Debelyn and Christel after giving dolls away
def Debelyn_final_dolls : ℕ := Debelyn_initial_dolls - Debelyn_gift_dolls
def Christel_final_dolls : ℕ := Christel_initial_dolls - Christel_gift_dolls

-- Define Andrena's dolls after transactions
def Andrena_dolls : ℕ := Christel_final_dolls + 2

-- Define the Lean statement for proving Andrena has 3 more dolls than Debelyn
theorem Andrena_more_than_Debelyn : Andrena_dolls = Debelyn_final_dolls + 3 := by
  -- Here you would prove the statement
  sorry

end Andrena_more_than_Debelyn_l277_277555


namespace area_of_rectangle_ABCD_l277_277006

-- Conditions
variables {ABCD : Type} [nonempty ABCD]
variable (P : ℕ)
axiom four_identical_squares : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x
axiom perimeter_eq : P = 160

-- Proof problem
theorem area_of_rectangle_ABCD (h1 : ∀ (A B C D : ABCD), ∃ (x : ℕ), 4 * x)
                               (h2 : P = 160) : ∃ (area : ℕ), area = 1024 :=
by sorry

end area_of_rectangle_ABCD_l277_277006


namespace hyperbola_asymptote_slope_l277_277822

theorem hyperbola_asymptote_slope
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c ≠ -a ∧ c ≠ a)
  (H1 : (c ≠ -a ∧ c ≠ a) ∧ (a ≠ 0) ∧ (b ≠ 0))
  (H_perp : (c + a) * (c - a) * (a * a * a * a) + (b * b * b * b) = 0) :
  abs (b / a) = 1 :=
by
  sorry  -- Proof here is not required as per the given instructions

end hyperbola_asymptote_slope_l277_277822


namespace solution_set_of_f_lt_exp_l277_277710

noncomputable def f : ℝ → ℝ := sorry -- assume f is a differentiable function

-- Define the conditions
axiom h_deriv : ∀ x : ℝ, deriv f x < f x
axiom h_periodic : ∀ x : ℝ, f (x + 2) = f (x - 2)
axiom h_value_at_4 : f 4 = 1

-- The main statement to be proved
theorem solution_set_of_f_lt_exp :
  ∀ x : ℝ, (f x < Real.exp x ↔ x > 0) :=
by
  intro x
  sorry

end solution_set_of_f_lt_exp_l277_277710


namespace increased_volume_l277_277263

theorem increased_volume (l w h : ℕ) 
  (volume_eq : l * w * h = 4500) 
  (surface_area_eq : l * w + l * h + w * h = 900) 
  (edges_sum_eq : l + w + h = 54) :
  (l + 1) * (w + 1) * (h + 1) = 5455 := 
by 
  sorry

end increased_volume_l277_277263


namespace prime_triplets_satisfy_condition_l277_277995

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_triplets_satisfy_condition :
  ∀ p q r : ℕ,
    is_prime p → is_prime q → is_prime r →
    (p * (r - 1) = q * (r + 7)) →
    (p = 3 ∧ q = 2 ∧ r = 17) ∨ 
    (p = 7 ∧ q = 3 ∧ r = 7) ∨
    (p = 5 ∧ q = 3 ∧ r = 13) :=
by
  sorry

end prime_triplets_satisfy_condition_l277_277995


namespace ratio_of_novels_read_l277_277475

theorem ratio_of_novels_read (jordan_read : ℕ) (alexandre_read : ℕ)
  (h_jordan_read : jordan_read = 120) 
  (h_diff : jordan_read = alexandre_read + 108) :
  alexandre_read / jordan_read = 1 / 10 :=
by
  -- Proof skipped
  sorry

end ratio_of_novels_read_l277_277475


namespace smallest_a_l277_277955

theorem smallest_a (a : ℕ) (h₁ : Nat.gcd a 70 > 1) (h₂ : Nat.gcd a 84 > 1) : a = 14 :=
sorry

end smallest_a_l277_277955


namespace min_sqrt_eq_sum_sqrt_implies_param_l277_277564

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem min_sqrt_eq_sum_sqrt_implies_param (a b c : ℝ) (r s t : ℝ)
    (h1 : 0 < a ∧ a ≤ 1)
    (h2 : 0 < b ∧ b ≤ 1)
    (h3 : 0 < c ∧ c ≤ 1)
    (h4 : min (sqrt ((a * b + 1) / (a * b * c))) (min (sqrt ((b * c + 1) / (a * b * c))) (sqrt ((a * c + 1) / (a * b * c)))) 
          = (sqrt ((1 - a) / a) + sqrt ((1 - b) / b) + sqrt ((1 - c) / c))) :
    ∃ r, a = 1 / (1 + r^2) ∧ b = 1 / (1 + (1 / r^2)) ∧ c = (r + 1 / r)^2 / (1 + (r + 1 / r)^2) :=
sorry

end min_sqrt_eq_sum_sqrt_implies_param_l277_277564


namespace probability_good_or_excellent_l277_277182

noncomputable def P_H1 : ℚ := 5 / 21
noncomputable def P_H2 : ℚ := 10 / 21
noncomputable def P_H3 : ℚ := 6 / 21

noncomputable def P_A_given_H1 : ℚ := 1
noncomputable def P_A_given_H2 : ℚ := 1
noncomputable def P_A_given_H3 : ℚ := 1 / 3

noncomputable def P_A : ℚ := 
  P_H1 * P_A_given_H1 + 
  P_H2 * P_A_given_H2 + 
  P_H3 * P_A_given_H3

theorem probability_good_or_excellent : P_A = 17 / 21 :=
by
  sorry

end probability_good_or_excellent_l277_277182


namespace capacity_of_new_bucket_l277_277108

def number_of_old_buckets : ℕ := 26
def capacity_of_old_bucket : ℝ := 13.5
def total_volume : ℝ := number_of_old_buckets * capacity_of_old_bucket
def number_of_new_buckets : ℕ := 39

theorem capacity_of_new_bucket :
  total_volume / number_of_new_buckets = 9 :=
sorry

end capacity_of_new_bucket_l277_277108


namespace necessary_but_not_sufficient_l277_277967

noncomputable def represents_ellipse (m : ℝ) : Prop :=
  2 < m ∧ m < 6 ∧ m ≠ 4

theorem necessary_but_not_sufficient (m : ℝ) :
  represents_ellipse (m) ↔ (2 < m ∧ m < 6) :=
by
  sorry

end necessary_but_not_sufficient_l277_277967


namespace find_increase_in_radius_l277_277378

def volume_cylinder (r h : ℝ) := π * r^2 * h

theorem find_increase_in_radius (x : ℝ) :
  let r := 5
  let h1 := 4
  let h2 := h1 + 2
  volume_cylinder (r + x) h1 = volume_cylinder r h2 →
  x = (5*(Real.sqrt 6 - 2))/2 :=
by
  let r := 5
  let h1 := 4
  let h2 := h1 + 2
  let volume_cylinder := fun (r h : ℝ) => π * r^2 * h
  sorry

end find_increase_in_radius_l277_277378


namespace solve_problem1_solve_problem2_l277_277062

noncomputable def problem1 (m n : ℝ) : Prop :=
  (m + n) ^ 2 - 10 * (m + n) + 25 = (m + n - 5) ^ 2

noncomputable def problem2 (x : ℝ) : Prop :=
  ((x ^ 2 - 6 * x + 8) * (x ^ 2 - 6 * x + 10) + 1) = (x - 3) ^ 4

-- Placeholder for proofs
theorem solve_problem1 (m n : ℝ) : problem1 m n :=
by
  sorry

theorem solve_problem2 (x : ℝ) : problem2 x :=
by
  sorry

end solve_problem1_solve_problem2_l277_277062


namespace minimum_red_chips_l277_277672

theorem minimum_red_chips (w b r : ℕ) (h1 : b ≥ w / 4) (h2 : b ≤ r / 6) (h3 : w + b ≥ 75) : r ≥ 90 :=
sorry

end minimum_red_chips_l277_277672


namespace base_7_units_digit_l277_277348

theorem base_7_units_digit : ((156 + 97) % 7) = 1 := 
by
  sorry

end base_7_units_digit_l277_277348


namespace multiply_add_distribute_l277_277548

theorem multiply_add_distribute :
  42 * 25 + 58 * 42 = 3486 := by
  sorry

end multiply_add_distribute_l277_277548


namespace completing_the_square_l277_277101

theorem completing_the_square (x m n : ℝ) 
  (h : x^2 - 6 * x = 1) 
  (hm : (x - m)^2 = n) : 
  m + n = 13 :=
sorry

end completing_the_square_l277_277101


namespace solution_of_system_l277_277068

def log4 (n : ℝ) : ℝ := log n / log 4

theorem solution_of_system (x y : ℝ) (hx : x + y = 20)
  (hy : log4 x + log4 y = 1 + log4 9) :
  (x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18) :=
by sorry

end solution_of_system_l277_277068


namespace right_angled_triangle_max_area_l277_277852

theorem right_angled_triangle_max_area (a b : ℝ) (h : a + b = 4) : (1 / 2) * a * b ≤ 2 :=
by 
  sorry

end right_angled_triangle_max_area_l277_277852


namespace team_points_behind_l277_277030

-- Define the points for Max, Dulce and the condition for Val
def max_points : ℕ := 5
def dulce_points : ℕ := 3
def combined_points_max_dulce : ℕ := max_points + dulce_points
def val_points : ℕ := 2 * combined_points_max_dulce

-- Define the total points for their team and the opponents' team
def their_team_points : ℕ := max_points + dulce_points + val_points
def opponents_team_points : ℕ := 40

-- Proof statement
theorem team_points_behind : opponents_team_points - their_team_points = 16 :=
by
  sorry

end team_points_behind_l277_277030


namespace sequence_correctness_l277_277202

def sequence_a (n : ℕ) : ℤ :=
  if n = 1 then -2
  else -(2^(n - 1))

def partial_sum_S (n : ℕ) : ℤ := -2^n

theorem sequence_correctness (n : ℕ) (h : n ≥ 1) :
  (sequence_a 1 = -2) ∧ (∀ n ≥ 2, sequence_a (n + 1) = partial_sum_S n) ∧
  (sequence_a n = -(2^(n - 1))) ∧ (partial_sum_S n = -2^n) :=
by
  sorry

end sequence_correctness_l277_277202


namespace alcohol_percentage_second_vessel_l277_277979

theorem alcohol_percentage_second_vessel:
  ∃ x : ℝ, 
  let alcohol_in_first := 0.25 * 2
  let alcohol_in_second := 0.01 * x * 6
  let total_alcohol := 0.29 * 8
  alcohol_in_first + alcohol_in_second = total_alcohol → 
  x = 30.333333333333332 :=
by
  sorry

end alcohol_percentage_second_vessel_l277_277979


namespace area_EYH_trapezoid_l277_277766

theorem area_EYH_trapezoid (EF GH : ℕ) (EF_len : EF = 15) (GH_len : GH = 35) 
(Area_trapezoid : (EF + GH) * 16 / 2 = 400) : 
∃ (EYH_area : ℕ), EYH_area = 84 := by
  sorry

end area_EYH_trapezoid_l277_277766


namespace balance_test_l277_277854

variable (a b h c : ℕ)

theorem balance_test
  (h1 : 4 * a + 2 * b + h = 21 * c)
  (h2 : 2 * a = b + h + 5 * c) :
  b + 2 * h = 11 * c :=
sorry

end balance_test_l277_277854


namespace elizabeth_haircut_l277_277691

theorem elizabeth_haircut (t s f : ℝ) (ht : t = 0.88) (hs : s = 0.5) : f = t - s := by
  sorry

end elizabeth_haircut_l277_277691


namespace skilled_picker_capacity_minimize_costs_l277_277962

theorem skilled_picker_capacity (x : ℕ) (h1 : ∀ x : ℕ, ∀ s : ℕ, s = 3 * x) (h2 : 450 * 25 = 3 * x * 25 + 600) :
  s = 30 :=
by
  sorry

theorem minimize_costs (s n m : ℕ)
(h1 : s ≤ 20)
(h2 : n ≤ 15)
(h3 : 600 = s * 30 + n * 10)
(h4 : ∀ y, y = s * 300 + n * 80) :
  m = 15 ∧ s = 15 :=
by
  sorry

end skilled_picker_capacity_minimize_costs_l277_277962


namespace y_intercept_with_z_3_l277_277420

theorem y_intercept_with_z_3 : 
  ∀ x y : ℝ, (4 * x + 6 * y - 2 * 3 = 24) → (x = 0) → y = 5 :=
by
  intros x y h1 h2
  sorry

end y_intercept_with_z_3_l277_277420


namespace find_a_l277_277160

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (√3 * sin x * cos x + cos x ^ 2 + a)

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc (-π / 6) (π / 3) → f a x ≤ 1 + a + 1/2) ∧
  (∀ x : ℝ, x ∈ set.Icc (-π / 6) (π / 3) → f a x ≥ -1/2 + a + 1/2) →
  a = 0 :=
begin
  assume h,
  sorry -- Proof omitted
end

end find_a_l277_277160


namespace grassy_plot_width_l277_277678

/-- A rectangular grassy plot has a length of 100 m and a certain width. 
It has a gravel path 2.5 m wide all round it on the inside. The cost of gravelling 
the path at 0.90 rupees per square meter is 742.5 rupees. 
Prove that the width of the grassy plot is 60 meters. -/
theorem grassy_plot_width 
  (length : ℝ)
  (path_width : ℝ)
  (cost_per_sq_meter : ℝ)
  (total_cost : ℝ)
  (width : ℝ) : 
  length = 100 ∧ 
  path_width = 2.5 ∧ 
  cost_per_sq_meter = 0.9 ∧ 
  total_cost = 742.5 → 
  width = 60 := 
by sorry

end grassy_plot_width_l277_277678


namespace choir_members_l277_277358

theorem choir_members (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 11 = 6) (h3 : 200 ≤ n ∧ n ≤ 300) :
  n = 220 :=
sorry

end choir_members_l277_277358


namespace dan_total_purchase_cost_l277_277990

noncomputable def snake_toy_cost : ℝ := 11.76
noncomputable def cage_cost : ℝ := 14.54
noncomputable def heat_lamp_cost : ℝ := 6.25
noncomputable def cage_discount_rate : ℝ := 0.10
noncomputable def sales_tax_rate : ℝ := 0.08
noncomputable def found_dollar : ℝ := 1.00

noncomputable def total_cost : ℝ :=
  let cage_discount := cage_discount_rate * cage_cost
  let discounted_cage := cage_cost - cage_discount
  let subtotal_before_tax := snake_toy_cost + discounted_cage + heat_lamp_cost
  let sales_tax := sales_tax_rate * subtotal_before_tax
  let total_after_tax := subtotal_before_tax + sales_tax
  total_after_tax - found_dollar

theorem dan_total_purchase_cost : total_cost = 32.58 :=
  by 
    -- Placeholder for the proof
    sorry

end dan_total_purchase_cost_l277_277990


namespace exists_root_interval_l277_277278

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_interval :
  (f 1.1 < 0) ∧ (f 1.2 > 0) → ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 := 
by
  intro h
  sorry

end exists_root_interval_l277_277278


namespace domain_linear_domain_rational_domain_sqrt_domain_sqrt_denominator_domain_rational_complex_domain_arcsin_l277_277141

-- 1. Domain of z = 4 - x - 2y
theorem domain_linear (x y : ℝ) : true := 
by sorry

-- 2. Domain of p = 3 / (x^2 + y^2)
theorem domain_rational (x y : ℝ) : x^2 + y^2 ≠ 0 → true := 
by sorry

-- 3. Domain of z = sqrt(1 - x^2 - y^2)
theorem domain_sqrt (x y : ℝ) : 1 - x^2 - y^2 ≥ 0 → true := 
by sorry

-- 4. Domain of q = 1 / sqrt(xy)
theorem domain_sqrt_denominator (x y : ℝ) : xy > 0 → true := 
by sorry

-- 5. Domain of u = x^2 y / (2x + 1 - y)
theorem domain_rational_complex (x y : ℝ) : 2x + 1 - y ≠ 0 → true := 
by sorry

-- 6. Domain of v = arcsin(x + y)
theorem domain_arcsin (x y : ℝ) : -1 ≤ x + y ∧ x + y ≤ 1 → true := 
by sorry

end domain_linear_domain_rational_domain_sqrt_domain_sqrt_denominator_domain_rational_complex_domain_arcsin_l277_277141


namespace colbert_materials_needed_l277_277686

def wooden_planks_needed (total_needed quarter_in_stock : ℕ) : ℕ :=
  let total_purchased := total_needed - quarter_in_stock / 4
  (total_purchased + 7) / 8 -- ceil division by 8

def iron_nails_needed (total_needed thirty_percent_provided : ℕ) : ℕ :=
  let total_purchased := total_needed - total_needed * thirty_percent_provided / 100
  (total_purchased + 24) / 25 -- ceil division by 25

def fabric_needed (total_needed third_provided : ℚ) : ℚ :=
  total_needed - total_needed / third_provided

def metal_brackets_needed (total_needed in_stock multiple : ℕ) : ℕ :=
  let total_purchased := total_needed - in_stock
  (total_purchased + multiple - 1) / multiple * multiple -- ceil to next multiple of 5

theorem colbert_materials_needed :
  wooden_planks_needed 250 62 = 24 ∧
  iron_nails_needed 500 30 = 14 ∧
  fabric_needed 10 3 = 6.67 ∧
  metal_brackets_needed 40 10 5 = 30 :=
by sorry

end colbert_materials_needed_l277_277686


namespace three_person_subcommittees_l277_277717

theorem three_person_subcommittees (n k : ℕ) (h_n : n = 8) (h_k : k = 3) : nat.choose n k = 56 := by
  rw [h_n, h_k]
  norm_num
  sorry

end three_person_subcommittees_l277_277717


namespace alicia_art_left_l277_277266

-- Definition of the problem conditions.
def initial_pieces : ℕ := 70
def donated_pieces : ℕ := 46

-- The theorem to prove the number of art pieces left is 24.
theorem alicia_art_left : initial_pieces - donated_pieces = 24 := 
by
  sorry

end alicia_art_left_l277_277266


namespace rainfall_in_may_l277_277237

-- Define the rainfalls for the months
def march_rain : ℝ := 3.79
def april_rain : ℝ := 4.5
def june_rain : ℝ := 3.09
def july_rain : ℝ := 4.67

-- Define the average rainfall over five months
def avg_rain : ℝ := 4

-- Define total rainfall calculation
def calc_total_rain (may_rain : ℝ) : ℝ :=
  march_rain + april_rain + may_rain + june_rain + july_rain

-- Problem statement: proving the rainfall in May
theorem rainfall_in_may : ∃ (may_rain : ℝ), calc_total_rain may_rain = avg_rain * 5 ∧ may_rain = 3.95 :=
sorry

end rainfall_in_may_l277_277237


namespace intersection_A_B_l277_277209

def A := {x : ℝ | x > 3}
def B := {x : ℝ | (x - 1) * (x - 4) < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 3 < x ∧ x < 4} :=
by
  sorry

end intersection_A_B_l277_277209


namespace greatest_integer_multiple_9_remainder_1000_l277_277479

noncomputable def M : ℕ := 
  max {n | (n % 9 = 0) ∧ (∀ (i j : ℕ), (i ≠ j → (n / 10^i % 10) ≠ (n / 10^j % 10)))}

theorem greatest_integer_multiple_9_remainder_1000 :
  (M % 1000) = 810 := 
by
  sorry

end greatest_integer_multiple_9_remainder_1000_l277_277479


namespace sum_possible_values_l277_277367

theorem sum_possible_values (M : ℝ) (h : M * (M - 6) = -5) : ∀ x ∈ {M | M * (M - 6) = -5}, x + (-x) = 6 :=
by sorry

end sum_possible_values_l277_277367


namespace nap_time_l277_277273

-- Definitions of given conditions
def flight_duration : ℕ := 680
def reading_time : ℕ := 120
def movie_time : ℕ := 240
def dinner_time : ℕ := 30
def radio_time : ℕ := 40
def game_time : ℕ := 70

def total_activity_time : ℕ := reading_time + movie_time + dinner_time + radio_time + game_time

-- Theorem statement
theorem nap_time : (flight_duration - total_activity_time) / 60 = 3 := by
  -- Here would go the proof steps verifying the equality
  sorry

end nap_time_l277_277273


namespace three_buses_interval_l277_277942

theorem three_buses_interval (interval_two_buses : ℕ) (loop_time : ℕ) :
  interval_two_buses = 21 →
  loop_time = interval_two_buses * 2 →
  (loop_time / 3) = 14 :=
by
  intros h1 h2
  rw [h1] at h2
  simp at h2
  sorry

end three_buses_interval_l277_277942


namespace solve_system_l277_277067

noncomputable def system_solution (x y : ℝ) :=
  x + y = 20 ∧ x * y = 36

theorem solve_system :
  (system_solution 18 2) ∧ (system_solution 2 18) :=
  sorry

end solve_system_l277_277067


namespace conference_handshakes_l277_277684

theorem conference_handshakes (total_people : ℕ) (group1_people : ℕ) (group2_people : ℕ)
  (group1_knows_each_other : group1_people = 25)
  (group2_knows_no_one_in_group1 : group2_people = 15)
  (total_group : total_people = group1_people + group2_people)
  (total_handshakes : ℕ := group2_people * (group1_people + group2_people - 1) - group2_people * (group2_people - 1) / 2) :
  total_handshakes = 480 := by
  -- Placeholder for proof
  sorry

end conference_handshakes_l277_277684


namespace armistice_day_is_wednesday_l277_277631

-- Define the starting date
def start_day : Nat := 5 -- 5 represents Friday if we consider 0 = Sunday

-- Define the number of days after which armistice was signed
def days_after : Nat := 2253

-- Define the target day (Wednesday = 3)
def expected_day : Nat := 3

-- Define the function to calculate the day of the week after a number of days
def day_after_n_days (start_day : Nat) (n : Nat) : Nat :=
  (start_day + n) % 7

-- Define the theorem to prove the equivalent mathematical problem
theorem armistice_day_is_wednesday : day_after_n_days start_day days_after = expected_day := by
  sorry

end armistice_day_is_wednesday_l277_277631


namespace maria_sister_drank_l277_277615

-- Define the conditions
def initial_bottles : ℝ := 45.0
def maria_drank : ℝ := 14.0
def remaining_bottles : ℝ := 23.0

-- Define the problem statement to prove the number of bottles Maria's sister drank
theorem maria_sister_drank (initial_bottles maria_drank remaining_bottles : ℝ) : 
    (initial_bottles - maria_drank) - remaining_bottles = 8.0 :=
by
  sorry

end maria_sister_drank_l277_277615


namespace probability_of_green_ball_is_157_over_495_l277_277344

-- Definitions of the number of balls in each container
def balls_in_container_I := (10, 2, 3) -- (red, green, blue)
def balls_in_container_II := (5, 4, 2) -- (red, green, blue)
def balls_in_container_III := (3, 5, 3) -- (red, green, blue)

-- Definition of random selection probability
def probability_of_selecting_container := (1 : ℚ) / 3

-- Calculating probability of selecting a green ball from a given container
def probability_of_green_ball_from_I : ℚ := 2 / (10 + 2 + 3)
def probability_of_green_ball_from_II : ℚ := 4 / (5 + 4 + 2)
def probability_of_green_ball_from_III : ℚ := 5 / (3 + 5 + 3)

-- Combined probability for each container
def combined_probability_I : ℚ := probability_of_selecting_container * probability_of_green_ball_from_I
def combined_probability_II : ℚ := probability_of_selecting_container * probability_of_green_ball_from_II
def combined_probability_III : ℚ := probability_of_selecting_container * probability_of_green_ball_from_III

-- Total probability of selecting a green ball
def total_probability_of_green_ball : ℚ := combined_probability_I + combined_probability_II + combined_probability_III

-- The statement to be proved in Lean 4
theorem probability_of_green_ball_is_157_over_495 : total_probability_of_green_ball = 157 / 495 :=
by
  sorry

end probability_of_green_ball_is_157_over_495_l277_277344


namespace angle_at_intersection_l277_277807

theorem angle_at_intersection (n : ℕ) (h₁ : n = 8)
  (h₂ : ∀ i j : ℕ, (i + 1) % n ≠ j ∧ i < j)
  (h₃ : ∀ i : ℕ, i < n)
  (h₄ : ∀ i j : ℕ, (i + 1) % n = j ∨ (i + n - 1) % n = j)
  : (2 * (180 / n - (180 * (n - 2) / n) / 2)) = 90 :=
by
  sorry

end angle_at_intersection_l277_277807


namespace elena_savings_l277_277827

theorem elena_savings :
  let original_cost := 7 * 3
  let discount_rate := 0.25
  let rebate := 5
  let disc_amount := original_cost * discount_rate
  let price_after_discount := original_cost - disc_amount
  let final_price := price_after_discount - rebate
  original_cost - final_price = 10.25 :=
by
  sorry

end elena_savings_l277_277827


namespace minimum_value_of_h_l277_277997

noncomputable def h (x : ℝ) : ℝ := x + (1 / x) + (1 / (x + (1 / x))^2)

theorem minimum_value_of_h : (∀ x : ℝ, x > 0 → h x ≥ 2.25) ∧ (h 1 = 2.25) :=
by
  sorry

end minimum_value_of_h_l277_277997


namespace janette_beef_jerky_left_l277_277467

def total_beef_jerky : ℕ := 40
def days_camping : ℕ := 5
def daily_consumption_per_meal : list ℕ := [1, 1, 2]
def brother_share_fraction : ℚ := 1/2

theorem janette_beef_jerky_left : 
  let daily_total_consumption := daily_consumption_per_meal.sum,
      total_consumption := days_camping * daily_total_consumption,
      remaining_jerky := total_beef_jerky - total_consumption,
      brother_share := remaining_jerky * brother_share_fraction
  in (remaining_jerky - brother_share) = 10 := 
by sorry

end janette_beef_jerky_left_l277_277467


namespace cost_of_5_dozen_l277_277890

noncomputable def price_per_dozen : ℝ :=
  24 / 3

noncomputable def cost_before_tax (num_dozen : ℝ) : ℝ :=
  num_dozen * price_per_dozen

noncomputable def cost_after_tax (num_dozen : ℝ) : ℝ :=
  (1 + 0.10) * cost_before_tax num_dozen

theorem cost_of_5_dozen :
  cost_after_tax 5 = 44 := 
sorry

end cost_of_5_dozen_l277_277890


namespace tenth_term_is_correct_l277_277783

-- Define the first term and common difference for the sequence
def a1 : ℚ := 1 / 2
def d : ℚ := 1 / 3

-- The property that defines the n-th term of the arithmetic sequence
def a (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Statement to prove that the tenth term in the arithmetic sequence is 7 / 2
theorem tenth_term_is_correct : a 10 = 7 / 2 := 
by 
  -- To be filled in with the proof later
  sorry

end tenth_term_is_correct_l277_277783


namespace cricketer_initial_average_l277_277528

def initial_bowling_average
  (runs_for_last_5_wickets : ℝ)
  (decreased_average : ℝ)
  (final_wickets : ℝ)
  (initial_wickets : ℝ)
  (initial_average : ℝ) : Prop :=
  (initial_average * initial_wickets + runs_for_last_5_wickets) / final_wickets =
    initial_average - decreased_average

theorem cricketer_initial_average :
  initial_bowling_average 26 0.4 85 80 12 :=
by
  unfold initial_bowling_average
  sorry

end cricketer_initial_average_l277_277528


namespace simplify_sqrt_360000_l277_277496

-- Define the given conditions
def factorization : 360000 = 3600 * 100 := rfl
def sqrt_3600 : Real.sqrt 3600 = 60 := by norm_num
def sqrt_100 : Real.sqrt 100 = 10 := by norm_num

-- Define the main statement to be proved
theorem simplify_sqrt_360000 : Real.sqrt 360000 = 600 :=
by
  rw [factorization, Real.sqrt_mul', sqrt_3600, sqrt_100]
  norm_num

end simplify_sqrt_360000_l277_277496


namespace triangular_array_sum_of_digits_l277_277407

def triangular_sum (N : ℕ) : ℕ := N * (N + 1) / 2

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem triangular_array_sum_of_digits :
  ∃ N : ℕ, triangular_sum N = 2080 ∧ sum_of_digits N = 10 :=
by
  sorry

end triangular_array_sum_of_digits_l277_277407


namespace sum_of_squares_of_four_consecutive_even_numbers_l277_277760

open Int

theorem sum_of_squares_of_four_consecutive_even_numbers (x y z w : ℤ) 
    (hx : x % 2 = 0) (hy : y = x + 2) (hz : z = x + 4) (hw : w = x + 6)
    : x + y + z + w = 36 → x^2 + y^2 + z^2 + w^2 = 344 := by
  sorry

end sum_of_squares_of_four_consecutive_even_numbers_l277_277760


namespace smallest_sum_of_sequence_l277_277366

theorem smallest_sum_of_sequence {
  A B C D k : ℕ
} (h1 : 2 * B = A + C)
  (h2 : D - C = (C - B) ^ 2)
  (h3 : 4 * B = 3 * C)
  (h4 : B = 3 * k)
  (h5 : C = 4 * k)
  (h6 : A = 2 * k)
  (h7 : D = 4 * k + k ^ 2) :
  A + B + C + D = 14 :=
by
  sorry

end smallest_sum_of_sequence_l277_277366


namespace solve_eqn_l277_277343

theorem solve_eqn (x : ℚ) (h1 : x ≠ 4) (h2 : x ≠ 6) :
  (x + 11) / (x - 4) = (x - 1) / (x + 6) → x = -31 / 11 :=
by
sorry

end solve_eqn_l277_277343


namespace crates_sold_on_monday_l277_277434

variable (M : ℕ)
variable (h : M + 2 * M + (2 * M - 2) + M = 28)

theorem crates_sold_on_monday : M = 5 :=
by
  sorry

end crates_sold_on_monday_l277_277434


namespace lcm_72_108_2100_l277_277698

theorem lcm_72_108_2100 : Nat.lcm (Nat.lcm 72 108) 2100 = 37800 := by
  sorry

end lcm_72_108_2100_l277_277698


namespace remainder_3n_mod_7_l277_277592

theorem remainder_3n_mod_7 (n : ℤ) (k : ℤ) (h : n = 7*k + 1) :
  (3 * n) % 7 = 3 := by
  sorry

end remainder_3n_mod_7_l277_277592


namespace percent_game_of_thrones_altered_l277_277573

def votes_game_of_thrones : ℕ := 10
def votes_twilight : ℕ := 12
def votes_art_of_deal : ℕ := 20

def altered_votes_art_of_deal : ℕ := votes_art_of_deal - (votes_art_of_deal * 80 / 100)
def altered_votes_twilight : ℕ := votes_twilight / 2
def total_altered_votes : ℕ := altered_votes_art_of_deal + altered_votes_twilight + votes_game_of_thrones

theorem percent_game_of_thrones_altered :
  ((votes_game_of_thrones * 100) / total_altered_votes) = 50 := by
  sorry

end percent_game_of_thrones_altered_l277_277573


namespace determine_f_2014_l277_277078

open Function

noncomputable def f : ℕ → ℕ :=
  sorry

theorem determine_f_2014
  (h1 : f 2 = 0)
  (h2 : f 3 > 0)
  (h3 : f 6042 = 2014)
  (h4 : ∀ m n : ℕ, f (m + n) - f m - f n ∈ ({0, 1} : Set ℕ)) :
  f 2014 = 671 :=
sorry

end determine_f_2014_l277_277078


namespace quadratic_function_properties_l277_277054

noncomputable def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 :=
by
  sorry

end quadratic_function_properties_l277_277054


namespace quilt_cost_l277_277472

theorem quilt_cost :
  let length := 7
  let width := 8
  let cost_per_sq_ft := 40
  let area := length * width
  let total_cost := area * cost_per_sq_ft
  total_cost = 2240 :=
by
  sorry

end quilt_cost_l277_277472


namespace spherical_to_rectangular_coords_l277_277989

theorem spherical_to_rectangular_coords :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = 5 * Real.sin (Real.pi / 3) * Real.cos (Real.pi / 4) ∧
  y = 5 * Real.sin (Real.pi / 3) * Real.sin (Real.pi / 4) ∧
  z = 5 * Real.cos (Real.pi / 3) ∧
  x = 5 * (Real.sqrt 3) / 2 * (Real.sqrt 2) / 2 ∧
  y = 5 * (Real.sqrt 3) / 2 * (Real.sqrt 2) / 2 ∧
  z = 2.5 ∧
  (x = (5 * Real.sqrt 6) / 4 ∧ y = (5 * Real.sqrt 6) / 4 ∧ z = 2.5) :=
by {
  sorry
}

end spherical_to_rectangular_coords_l277_277989


namespace distance_to_big_rock_l277_277791

variables (D : ℝ) (stillWaterSpeed : ℝ) (currentSpeed : ℝ) (totalTime : ℝ)

-- Define the conditions as constraints
def conditions := 
  stillWaterSpeed = 6 ∧
  currentSpeed = 1 ∧
  totalTime = 1 ∧
  (D / (stillWaterSpeed - currentSpeed) + D / (stillWaterSpeed + currentSpeed) = totalTime)

-- The theorem to prove the distance to Big Rock
theorem distance_to_big_rock (h : conditions D 6 1 1) : D = 35 / 12 :=
sorry

end distance_to_big_rock_l277_277791


namespace completing_the_square_l277_277100

theorem completing_the_square (x m n : ℝ) 
  (h : x^2 - 6 * x = 1) 
  (hm : (x - m)^2 = n) : 
  m + n = 13 :=
sorry

end completing_the_square_l277_277100


namespace coefficient_of_friction_l277_277767

/-- Assume m, Pi and ΔL are positive real numbers, and g is the acceleration due to gravity. 
We need to prove that the coefficient of friction μ is given by Pi / (m * g * ΔL). --/
theorem coefficient_of_friction (m Pi ΔL g : ℝ) (h_m : 0 < m) (h_Pi : 0 < Pi) (h_ΔL : 0 < ΔL) (h_g : 0 < g) :
  ∃ μ : ℝ, μ = Pi / (m * g * ΔL) :=
sorry

end coefficient_of_friction_l277_277767


namespace real_root_range_of_a_l277_277702

theorem real_root_range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + x + |a - 1/4| + |a| = 0) ↔ (0 ≤ a ∧ a ≤ 1/4) :=
by
  sorry

end real_root_range_of_a_l277_277702


namespace percentage_difference_l277_277453

theorem percentage_difference : (45 / 100 * 60) - (35 / 100 * 40) = 13 := by
  sorry

end percentage_difference_l277_277453


namespace sum_A_J_l277_277464

variable (A B C D E F G H I J : ℕ)

-- Conditions
axiom h1 : C = 7
axiom h2 : A + B + C = 40
axiom h3 : B + C + D = 40
axiom h4 : C + D + E = 40
axiom h5 : D + E + F = 40
axiom h6 : E + F + G = 40
axiom h7 : F + G + H = 40
axiom h8 : G + H + I = 40
axiom h9 : H + I + J = 40

-- Proof statement
theorem sum_A_J : A + J = 33 :=
by
  sorry

end sum_A_J_l277_277464


namespace rectangle_area_l277_277023

theorem rectangle_area (y : ℕ) (h : 10 * y = 160) : 4 * (y * y) = 1024 :=
by
  have y_value: y = 16 := by linarith
  rw y_value
  calc
    4 * (16 * 16) = 4 * 256 : by rfl
               ... = 1024 : by rfl

end rectangle_area_l277_277023


namespace eq_abc_gcd_l277_277503

theorem eq_abc_gcd
  (a b c d : ℕ)
  (h1 : a^a * b^(a + b) = c^c * d^(c + d))
  (h2 : Nat.gcd a b = 1)
  (h3 : Nat.gcd c d = 1) : 
  a = c ∧ b = d := 
sorry

end eq_abc_gcd_l277_277503


namespace projectile_reaches_35m_first_at_10_over_7_l277_277075

theorem projectile_reaches_35m_first_at_10_over_7 :
  ∃ (t : ℝ), (y : ℝ) = -4.9 * t^2 + 30 * t ∧ y = 35 ∧ t = 10 / 7 :=
by
  sorry

end projectile_reaches_35m_first_at_10_over_7_l277_277075


namespace triangle_area_l277_277511

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 180 := 
by 
  -- proof is skipped with sorry
  sorry

end triangle_area_l277_277511


namespace hazel_additional_days_l277_277374

theorem hazel_additional_days (school_year_days : ℕ) (miss_percent : ℝ) (already_missed : ℕ)
  (h1 : school_year_days = 180)
  (h2 : miss_percent = 0.05)
  (h3 : already_missed = 6) :
  (⌊miss_percent * school_year_days⌋ - already_missed) = 3 :=
by
  sorry

end hazel_additional_days_l277_277374


namespace determine_m_value_l277_277163

theorem determine_m_value
  (m : ℝ)
  (h : ∀ x : ℝ, -7 < x ∧ x < -1 ↔ mx^2 + 8 * m * x + 28 < 0) :
  m = 4 := by
  sorry

end determine_m_value_l277_277163


namespace find_a_minus_b_l277_277726

variable {a b : ℤ}

theorem find_a_minus_b (h1 : a^2 = 9) (h2 : |b| = 4) (h3 : a > b) : a - b = 7 :=
  sorry

end find_a_minus_b_l277_277726


namespace geometric_sequence_fourth_term_l277_277076

theorem geometric_sequence_fourth_term (x : ℝ) (h : (3 * x + 3) ^ 2 = x * (6 * x + 6)) :
  (∀ n : ℕ, 0 < n → (x, 3 * x + 3, 6 * x + 6)) = -24 := by
  sorry

end geometric_sequence_fourth_term_l277_277076


namespace mike_total_spending_is_497_50_l277_277898

def rose_bush_price : ℝ := 75
def rose_bush_count : ℕ := 6
def rose_bush_discount : ℝ := 0.10
def friend_rose_bushes : ℕ := 2
def tax_rose_bushes : ℝ := 0.05

def aloe_price : ℝ := 100
def aloe_count : ℕ := 2
def tax_aloe : ℝ := 0.07

def calculate_total_cost_for_mike : ℝ :=
  let total_rose_bush_cost := rose_bush_price * rose_bush_count
  let discount := total_rose_bush_cost * rose_bush_discount
  let cost_after_discount := total_rose_bush_cost - discount
  let sales_tax_rose_bushes := tax_rose_bushes * cost_after_discount
  let cost_rose_bushes_after_tax := cost_after_discount + sales_tax_rose_bushes

  let total_aloe_cost := aloe_price * aloe_count
  let sales_tax_aloe := tax_aloe * total_aloe_cost

  let total_cost_friend_rose_bushes := friend_rose_bushes * (rose_bush_price - (rose_bush_price * rose_bush_discount))
  let sales_tax_friend_rose_bushes := tax_rose_bushes * total_cost_friend_rose_bushes
  let total_cost_friend := total_cost_friend_rose_bushes + sales_tax_friend_rose_bushes

  let total_mike_rose_bushes := cost_rose_bushes_after_tax - total_cost_friend

  let total_cost_mike_aloe := total_aloe_cost + sales_tax_aloe

  total_mike_rose_bushes + total_cost_mike_aloe

theorem mike_total_spending_is_497_50 : calculate_total_cost_for_mike = 497.50 := by
  sorry

end mike_total_spending_is_497_50_l277_277898


namespace initial_roses_l277_277515

theorem initial_roses (R : ℕ) (initial_orchids : ℕ) (current_orchids : ℕ) (current_roses : ℕ) (added_orchids : ℕ) (added_roses : ℕ) :
  initial_orchids = 84 →
  current_orchids = 91 →
  current_roses = 14 →
  added_orchids = current_orchids - initial_orchids →
  added_roses = added_orchids →
  (R + added_roses = current_roses) →
  R = 7 :=
by
  sorry

end initial_roses_l277_277515


namespace complete_the_square_l277_277961

theorem complete_the_square (x : ℝ) : x^2 + 6 * x + 3 = 0 ↔ (x + 3)^2 = 6 := 
by
  sorry

end complete_the_square_l277_277961


namespace rectangle_section_properties_l277_277790

structure Tetrahedron where
  edge_length : ℝ

structure RectangleSection where
  perimeter : ℝ
  area : ℝ

def regular_tetrahedron : Tetrahedron :=
  { edge_length := 1 }

theorem rectangle_section_properties :
  ∀ (rect : RectangleSection), 
  (∃ tetra : Tetrahedron, tetra = regular_tetrahedron) →
  (rect.perimeter = 2) ∧ (0 ≤ rect.area) ∧ (rect.area ≤ 1/4) :=
by
  -- Provide the hypothesis of the existence of such a tetrahedron and rectangular section
  sorry

end rectangle_section_properties_l277_277790


namespace tan_beta_minus_2alpha_l277_277579

theorem tan_beta_minus_2alpha
  (α β : ℝ)
  (h1 : Real.tan α = 1/2)
  (h2 : Real.tan (α - β) = -1/3) :
  Real.tan (β - 2 * α) = -1/7 := 
sorry

end tan_beta_minus_2alpha_l277_277579


namespace find_Japanese_students_l277_277275

theorem find_Japanese_students (C K J : ℕ) (hK: K = (6 * C) / 11) (hJ: J = C / 8) (hK_value: K = 48) : J = 11 :=
by
  sorry

end find_Japanese_students_l277_277275


namespace greatest_integer_solution_l277_277951

theorem greatest_integer_solution :
  ∃ x : ℤ, (∃ (k : ℤ), (8 : ℚ) / 11 > k / 15 ∧ k = 10) ∧ x = 10 :=
by {
  sorry
}

end greatest_integer_solution_l277_277951


namespace average_annual_growth_rate_eq_l277_277310

-- Definition of variables based on given conditions
def sales_2021 := 298 -- in 10,000 units
def sales_2023 := 850 -- in 10,000 units
def years := 2

-- Problem statement in Lean 4
theorem average_annual_growth_rate_eq :
  sales_2021 * (1 + x) ^ years = sales_2023 :=
sorry

end average_annual_growth_rate_eq_l277_277310


namespace nap_time_l277_277272

-- Definitions of given conditions
def flight_duration : ℕ := 680
def reading_time : ℕ := 120
def movie_time : ℕ := 240
def dinner_time : ℕ := 30
def radio_time : ℕ := 40
def game_time : ℕ := 70

def total_activity_time : ℕ := reading_time + movie_time + dinner_time + radio_time + game_time

-- Theorem statement
theorem nap_time : (flight_duration - total_activity_time) / 60 = 3 := by
  -- Here would go the proof steps verifying the equality
  sorry

end nap_time_l277_277272


namespace volume_of_63_ounces_l277_277106

variable {V W : ℝ}
variable (k : ℝ)

def directly_proportional (V W : ℝ) (k : ℝ) : Prop :=
  V = k * W

theorem volume_of_63_ounces (h1 : directly_proportional 48 112 k)
                            (h2 : directly_proportional V 63 k) :
  V = 27 := by
  sorry

end volume_of_63_ounces_l277_277106


namespace square_area_l277_277770

theorem square_area (s : ℕ) (h : s = 13) : s * s = 169 := by
  sorry

end square_area_l277_277770


namespace tan_alpha_implies_fraction_l277_277703

theorem tan_alpha_implies_fraction (α : ℝ) (h : Real.tan α = -3/2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.cos α - Real.sin α) = 1 / 5 := 
sorry

end tan_alpha_implies_fraction_l277_277703


namespace percentage_of_class_taking_lunch_l277_277644

theorem percentage_of_class_taking_lunch 
  (total_students : ℕ)
  (boys_ratio : ℕ := 6)
  (girls_ratio : ℕ := 4)
  (boys_percentage_lunch : ℝ := 0.60)
  (girls_percentage_lunch : ℝ := 0.40) :
  total_students = 100 →
  (6 / (6 + 4) * 100) = 60 →
  (4 / (6 + 4) * 100) = 40 →
  (boys_percentage_lunch * 60 + girls_percentage_lunch * 40) = 52 →
  ℝ :=
    by
      intros
      sorry

end percentage_of_class_taking_lunch_l277_277644


namespace find_vanessa_age_l277_277892

/-- Define the initial conditions and goal -/
theorem find_vanessa_age (V : ℕ) (Kevin_age current_time future_time : ℕ) :
  Kevin_age = 16 ∧ future_time = current_time + 5 ∧
  (Kevin_age + future_time - current_time) = 3 * (V + future_time - current_time) →
  V = 2 := 
by
  sorry

end find_vanessa_age_l277_277892


namespace smaller_square_area_percentage_l277_277811

noncomputable def area_percentage_of_smaller_square :=
  let side_length_large_square : ℝ := 4
  let area_large_square := side_length_large_square ^ 2
  let side_length_smaller_square := side_length_large_square / 5
  let area_smaller_square := side_length_smaller_square ^ 2
  (area_smaller_square / area_large_square) * 100
theorem smaller_square_area_percentage :
  area_percentage_of_smaller_square = 4 := 
sorry

end smaller_square_area_percentage_l277_277811


namespace triangular_weight_is_60_l277_277923

/-- Suppose there are weights: 5 identical round, 2 identical triangular, and 1 rectangular weight of 90 grams.
    The conditions are: 
    1. One round weight and one triangular weight balance three round weights.
    2. Four round weights and one triangular weight balance one triangular weight, one round weight, and one rectangular weight.
    Prove that the weight of the triangular weight is 60 grams. -/
theorem triangular_weight_is_60 
  (R T : ℕ)  -- We declare weights of round and triangular weights as natural numbers
  (h1 : R + T = 3 * R)  -- The first balance condition
  (h2 : 4 * R + T = T + R + 90)  -- The second balance condition
  : T = 60 := 
by
  sorry  -- Proof omitted

end triangular_weight_is_60_l277_277923


namespace Andrena_more_than_Debelyn_l277_277553

-- Definitions based on the problem conditions
def Debelyn_initial := 20
def Debelyn_gift_to_Andrena := 2
def Christel_initial := 24
def Christel_gift_to_Andrena := 5
def Andrena_more_than_Christel := 2

-- Calculating the number of dolls each person has after the gifts
def Debelyn_final := Debelyn_initial - Debelyn_gift_to_Andrena
def Christel_final := Christel_initial - Christel_gift_to_Andrena
def Andrena_final := Christel_final + Andrena_more_than_Christel

-- The proof problem statement
theorem Andrena_more_than_Debelyn : Andrena_final - Debelyn_final = 3 := by
  sorry

end Andrena_more_than_Debelyn_l277_277553


namespace stan_average_speed_l277_277908

/-- Given two trips with specified distances and times, prove that the overall average speed is 55 mph. -/
theorem stan_average_speed :
  let distance1 := 300
  let hours1 := 5
  let minutes1 := 20
  let distance2 := 360
  let hours2 := 6
  let minutes2 := 40
  let total_distance := distance1 + distance2
  let total_time := (hours1 + minutes1 / 60) + (hours2 + minutes2 / 60)
  total_distance / total_time = 55 := 
sorry

end stan_average_speed_l277_277908


namespace mrs_randall_total_teaching_years_l277_277619

def years_teaching_third_grade : ℕ := 18
def years_teaching_second_grade : ℕ := 8

theorem mrs_randall_total_teaching_years : years_teaching_third_grade + years_teaching_second_grade = 26 :=
by
  sorry

end mrs_randall_total_teaching_years_l277_277619


namespace coupon_savings_difference_l277_277538

-- Definitions based on conditions
def P (p : ℝ) := 120 + p
def savings_coupon_A (p : ℝ) := 24 + 0.20 * p
def savings_coupon_B := 35
def savings_coupon_C (p : ℝ) := 0.30 * p

-- Conditions
def condition_A_saves_at_least_B (p : ℝ) := savings_coupon_A p ≥ savings_coupon_B
def condition_A_saves_at_least_C (p : ℝ) := savings_coupon_A p ≥ savings_coupon_C p

-- Proof problem
theorem coupon_savings_difference :
  ∀ (p : ℝ), 55 ≤ p ∧ p ≤ 240 → (P 240 - P 55) = 185 :=
by
  sorry

end coupon_savings_difference_l277_277538


namespace compound_cost_correct_l277_277972

noncomputable def compound_cost_per_pound (limestone_cost shale_mix_cost : ℝ) (total_weight limestone_weight : ℝ) : ℝ :=
  let shale_mix_weight := total_weight - limestone_weight
  let total_cost := (limestone_weight * limestone_cost) + (shale_mix_weight * shale_mix_cost)
  total_cost / total_weight

theorem compound_cost_correct :
  compound_cost_per_pound 3 5 100 37.5 = 4.25 := by
  sorry

end compound_cost_correct_l277_277972


namespace area_of_ABC_l277_277465

noncomputable def area_of_triangle (AB AC angleB : ℝ) : ℝ :=
  0.5 * AB * AC * Real.sin angleB

theorem area_of_ABC :
  area_of_triangle 5 3 (120 * Real.pi / 180) = (15 * Real.sqrt 3) / 4 :=
by
  sorry

end area_of_ABC_l277_277465
