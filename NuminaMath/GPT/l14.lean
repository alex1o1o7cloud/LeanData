import Mathlib

namespace optimal_station_placement_l14_14411

def distance_between_buildings : ℕ := 50
def workers_in_building (n : ℕ) : ℕ := n

def total_walking_distance (x : ℝ) : ℝ :=
  |x| + 2 * |x - 50| + 3 * |x - 100| + 4 * |x - 150| + 5 * |x - 200|

theorem optimal_station_placement : ∃ x : ℝ, x = 150 ∧ (∀ y : ℝ, total_walking_distance x ≤ total_walking_distance y) :=
  sorry

end optimal_station_placement_l14_14411


namespace problem_l14_14217

noncomputable def a_seq (n : ℕ) : ℚ := sorry

def is_geometric_sequence (seq : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n : ℕ, seq (n + 1) = q * seq n

theorem problem (h_positive : ∀ n : ℕ, 0 < a_seq n)
                (h_ratio : ∀ n : ℕ, 2 * a_seq n = 3 * a_seq (n + 1))
                (h_product : a_seq 1 * a_seq 4 = 8 / 27) :
  is_geometric_sequence a_seq (2 / 3) ∧ 
  (∃ n : ℕ, a_seq n = 16 / 81 ∧ n = 6) :=
by
  sorry

end problem_l14_14217


namespace expression_positive_intervals_l14_14279

theorem expression_positive_intervals :
  {x : ℝ | (x + 2) * (x - 3) > 0} = {x | x < -2} ∪ {x | x > 3} :=
by
  sorry

end expression_positive_intervals_l14_14279


namespace six_digit_palindromes_count_l14_14530

theorem six_digit_palindromes_count : 
  ∃ n : ℕ, n = 27 ∧ 
  (∀ (A B C : ℕ), 
       (A = 6 ∨ A = 7 ∨ A = 8) ∧ 
       (B = 6 ∨ B = 7 ∨ B = 8) ∧ 
       (C = 6 ∨ C = 7 ∨ C = 8) → 
       ∃ p : ℕ, 
         p = (A * 10^5 + B * 10^4 + C * 10^3 + C * 10^2 + B * 10 + A) ∧ 
         (6 ≤ p / 10^5 ∧ p / 10^5 ≤ 8) ∧ 
         (6 ≤ (p / 10^4) % 10 ∧ (p / 10^4) % 10 ≤ 8) ∧ 
         (6 ≤ (p / 10^3) % 10 ∧ (p / 10^3) % 10 ≤ 8)) :=
  by sorry

end six_digit_palindromes_count_l14_14530


namespace positive_abc_l14_14481

theorem positive_abc (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : a > 0 ∧ b > 0 ∧ c > 0 := 
by
  sorry

end positive_abc_l14_14481


namespace packs_of_beef_l14_14863

noncomputable def pounds_per_pack : ℝ := 4
noncomputable def price_per_pound : ℝ := 5.50
noncomputable def total_paid : ℝ := 110
noncomputable def price_per_pack : ℝ := price_per_pound * pounds_per_pack

theorem packs_of_beef (n : ℝ) (h : n = total_paid / price_per_pack) : n = 5 := 
by
  sorry

end packs_of_beef_l14_14863


namespace seeds_per_flowerbed_l14_14282

theorem seeds_per_flowerbed :
  ∀ (total_seeds flowerbeds seeds_per_bed : ℕ), 
  total_seeds = 32 → 
  flowerbeds = 8 → 
  seeds_per_bed = total_seeds / flowerbeds → 
  seeds_per_bed = 4 :=
  by 
    intros total_seeds flowerbeds seeds_per_bed h_total h_flowerbeds h_calc
    rw [h_total, h_flowerbeds] at h_calc
    exact h_calc

end seeds_per_flowerbed_l14_14282


namespace nonneg_for_all_x_iff_a_in_range_l14_14823

def f (x a : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

theorem nonneg_for_all_x_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end nonneg_for_all_x_iff_a_in_range_l14_14823


namespace no_arithmetic_sqrt_of_neg_real_l14_14389

theorem no_arithmetic_sqrt_of_neg_real (x : ℝ) (h : x < 0) : ¬ ∃ y : ℝ, y * y = x :=
by
  sorry

end no_arithmetic_sqrt_of_neg_real_l14_14389


namespace opposite_face_of_lime_is_black_l14_14848

-- Define the colors
inductive Color
| P | C | M | S | K | L

-- Define the problem conditions
def face_opposite (c : Color) : Color := sorry

-- Theorem statement
theorem opposite_face_of_lime_is_black : face_opposite Color.L = Color.K := sorry

end opposite_face_of_lime_is_black_l14_14848


namespace cooper_pies_days_l14_14009

theorem cooper_pies_days :
  ∃ d : ℕ, 7 * d - 50 = 34 ∧ d = 12 :=
by
  sorry

end cooper_pies_days_l14_14009


namespace mike_spent_on_car_parts_l14_14557

-- Define the costs as constants
def cost_speakers : ℝ := 118.54
def cost_tires : ℝ := 106.33
def cost_cds : ℝ := 4.58

-- Define the total cost of car parts excluding the CDs
def total_cost_car_parts : ℝ := cost_speakers + cost_tires

-- The theorem we want to prove
theorem mike_spent_on_car_parts :
  total_cost_car_parts = 224.87 := 
by 
  -- Proof omitted
  sorry

end mike_spent_on_car_parts_l14_14557


namespace youseff_blocks_l14_14884

theorem youseff_blocks (x : ℕ) 
  (H1 : (1 : ℚ) * x = (1/3 : ℚ) * x + 8) : 
  x = 12 := 
sorry

end youseff_blocks_l14_14884


namespace totalCarsProduced_is_29621_l14_14581

def numSedansNA    := 3884
def numSUVsNA      := 2943
def numPickupsNA   := 1568

def numSedansEU    := 2871
def numSUVsEU      := 2145
def numPickupsEU   := 643

def numSedansASIA  := 5273
def numSUVsASIA    := 3881
def numPickupsASIA := 2338

def numSedansSA    := 1945
def numSUVsSA      := 1365
def numPickupsSA   := 765

def totalCarsProduced : Nat :=
  numSedansNA + numSUVsNA + numPickupsNA +
  numSedansEU + numSUVsEU + numPickupsEU +
  numSedansASIA + numSUVsASIA + numPickupsASIA +
  numSedansSA + numSUVsSA + numPickupsSA

theorem totalCarsProduced_is_29621 : totalCarsProduced = 29621 :=
by
  sorry

end totalCarsProduced_is_29621_l14_14581


namespace female_democrats_count_l14_14302

theorem female_democrats_count 
  (F M : ℕ) 
  (total_participants : F + M = 750)
  (female_democrats : ℕ := F / 2) 
  (male_democrats : ℕ := M / 4)
  (total_democrats : female_democrats + male_democrats = 250) :
  female_democrats = 125 := 
sorry

end female_democrats_count_l14_14302


namespace winning_strategy_for_pawns_l14_14246

def wiit_or_siti_wins (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 3 * k + 2) ∨ (∃ k : ℕ, n ≠ 3 * k + 2)

theorem winning_strategy_for_pawns (n : ℕ) : wiit_or_siti_wins n :=
sorry

end winning_strategy_for_pawns_l14_14246


namespace chord_length_range_l14_14115

variable {x y : ℝ}

def center : ℝ × ℝ := (4, 5)
def radius : ℝ := 13
def point : ℝ × ℝ := (1, 1)
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 169

-- statement: prove the range of |AB| for specific conditions
theorem chord_length_range :
  ∀ line : (ℝ × ℝ) → (ℝ × ℝ) → Prop,
  (line center point → line (x, y) (x, y) ∧ circle_eq x y)
  → 24 ≤ abs (dist (x, y) (x, y)) ∧ abs (dist (x, y) (x, y)) ≤ 26 :=
by
  sorry

end chord_length_range_l14_14115


namespace am_gm_inequality_l14_14456

theorem am_gm_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c) ^ 2 :=
by
  sorry

end am_gm_inequality_l14_14456


namespace determine_polynomial_l14_14205

theorem determine_polynomial (p : ℝ → ℝ) (h : ∀ x : ℝ, 1 + p x = (p (x - 1) + p (x + 1)) / 2) :
  ∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b * x + c := by
  sorry

end determine_polynomial_l14_14205


namespace complete_the_square_l14_14476

theorem complete_the_square (x : ℝ) : 
  x^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 = 6 := 
by {
  -- This is where you would provide the proof
  sorry
}

end complete_the_square_l14_14476


namespace parallel_lines_k_l14_14589

theorem parallel_lines_k (k : ℝ) :
  (∃ (x y : ℝ), (k-3) * x + (4-k) * y + 1 = 0 ∧ 2 * (k-3) * x - 2 * y + 3 = 0) →
  (k = 3 ∨ k = 5) :=
by
  sorry

end parallel_lines_k_l14_14589


namespace solve_system_l14_14929

theorem solve_system (x y : ℝ) (h1 : x^2 + y^2 + x + y = 50) (h2 : x * y = 20) :
  (x = 5 ∧ y = 4) ∨ (x = 4 ∧ y = 5) ∨ (x = -5 + Real.sqrt 5 ∧ y = -5 - Real.sqrt 5) ∨ (x = -5 - Real.sqrt 5 ∧ y = -5 + Real.sqrt 5) :=
by
  sorry

end solve_system_l14_14929


namespace count_negative_numbers_l14_14525

theorem count_negative_numbers : 
  let n1 := abs (-2)
  let n2 := - abs (3^2)
  let n3 := - (3^2)
  let n4 := (-2)^(2023)
  (if n1 < 0 then 1 else 0) + (if n2 < 0 then 1 else 0) + (if n3 < 0 then 1 else 0) + (if n4 < 0 then 1 else 0) = 3 := 
by
  sorry

end count_negative_numbers_l14_14525


namespace taxi_speed_l14_14495

theorem taxi_speed (v : ℕ) (h₁ : v > 30) (h₂ : ∃ t₁ t₂ : ℕ, t₁ = 3 ∧ t₂ = 3 ∧ 
                    v * t₁ = (v - 30) * (t₁ + t₂)) : 
                    v = 60 :=
by
  sorry

end taxi_speed_l14_14495


namespace base_angles_isosceles_triangle_l14_14141

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

end base_angles_isosceles_triangle_l14_14141


namespace factorial_div_result_l14_14335

theorem factorial_div_result : Nat.factorial 13 / Nat.factorial 11 = 156 :=
sorry

end factorial_div_result_l14_14335


namespace power_minus_self_even_l14_14091

theorem power_minus_self_even (a n : ℕ) (ha : 0 < a) (hn : 0 < n) : Even (a^n - a) := by
  sorry

end power_minus_self_even_l14_14091


namespace num_rectangles_in_grid_l14_14114

theorem num_rectangles_in_grid : 
  let width := 35
  let height := 44
  ∃ n, n = 87 ∧ 
  ∀ x y, (1 ≤ x ∧ x ≤ width) ∧ (1 ≤ y ∧ y ≤ height) → 
    n = (x * (x + 1) / 2) * (y * (y + 1) / 2) := 
by
  sorry

end num_rectangles_in_grid_l14_14114


namespace calculate_area_ADC_l14_14713

def area_AD (BD DC : ℕ) (area_ABD : ℕ) := 
  area_ABD * DC / BD

theorem calculate_area_ADC
  (BD DC : ℕ) 
  (h_ratio : BD = 5 * DC / 2)
  (area_ABD : ℕ)
  (h_area_ABD : area_ABD = 35) :
  area_AD BD DC area_ABD = 14 := 
by 
  sorry

end calculate_area_ADC_l14_14713


namespace Shekar_weighted_average_l14_14573

def score_weighted_sum (scores_weights : List (ℕ × ℚ)) : ℚ :=
  scores_weights.foldl (fun acc sw => acc + (sw.1 * sw.2 : ℚ)) 0

def Shekar_scores_weights : List (ℕ × ℚ) :=
  [(76, 0.20), (65, 0.15), (82, 0.10), (67, 0.15), (55, 0.10), (89, 0.05), (74, 0.05),
   (63, 0.10), (78, 0.05), (71, 0.05)]

theorem Shekar_weighted_average : score_weighted_sum Shekar_scores_weights = 70.55 := by
  sorry

end Shekar_weighted_average_l14_14573


namespace problem_I_problem_II_l14_14958

noncomputable def f (x m : ℝ) : ℝ := |x + m^2| + |x - 2*m - 3|

theorem problem_I (x m : ℝ) : f x m ≥ 2 :=
by 
  sorry

theorem problem_II (m : ℝ) : f 2 m ≤ 16 ↔ -3 ≤ m ∧ m ≤ Real.sqrt 14 - 1 :=
by 
  sorry

end problem_I_problem_II_l14_14958


namespace train_pass_bridge_l14_14494

-- Define variables and conditions
variables (train_length bridge_length : ℕ)
          (train_speed_kmph : ℕ)

-- Convert speed from km/h to m/s
def train_speed_mps(train_speed_kmph : ℕ) : ℚ :=
  (train_speed_kmph * 1000) / 3600

-- Total distance to cover
def total_distance(train_length bridge_length : ℕ) : ℕ :=
  train_length + bridge_length

-- Time to pass the bridge
def time_to_pass_bridge(train_length bridge_length : ℕ) (train_speed_kmph : ℕ) : ℚ :=
  (total_distance train_length bridge_length) / (train_speed_mps train_speed_kmph)

-- The proof statement
theorem train_pass_bridge :
  time_to_pass_bridge 360 140 50 = 36 := 
by
  -- actual proof would go here
  sorry

end train_pass_bridge_l14_14494


namespace determine_m_minus_n_l14_14769

-- Definitions of the conditions
variables {m n : ℝ}

-- The proof statement
theorem determine_m_minus_n (h_eq : ∀ x y : ℝ, x^(4 - 3 * |m|) + y^(3 * |n|) = 2009 → x + y = 2009)
  (h_prod_lt_zero : m * n < 0)
  (h_sum : 0 < m + n ∧ m + n ≤ 3) : m - n = 4/3 := 
sorry

end determine_m_minus_n_l14_14769


namespace find_k_l14_14504

theorem find_k (x k : ℝ) (h : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 5)) (hk : k ≠ 0) : k = 5 :=
sorry

end find_k_l14_14504


namespace trigonometric_identity_proof_l14_14154

noncomputable def four_sin_40_minus_tan_40 : ℝ :=
  4 * Real.sin (40 * Real.pi / 180) - Real.tan (40 * Real.pi / 180)

theorem trigonometric_identity_proof : four_sin_40_minus_tan_40 = Real.sqrt 3 := by
  sorry

end trigonometric_identity_proof_l14_14154


namespace average_and_fourth_number_l14_14758

theorem average_and_fourth_number {x : ℝ} (h_avg : ((1 + 2 + 4 + 6 + 9 + 9 + 10 + 12 + x) / 9) = 7) :
  x = 10 ∧ 6 = 6 :=
by
  sorry

end average_and_fourth_number_l14_14758


namespace spiral_wire_length_l14_14842

noncomputable def wire_length (turns : ℕ) (height : ℝ) (circumference : ℝ) : ℝ :=
  Real.sqrt (height^2 + (turns * circumference)^2)

theorem spiral_wire_length
  (turns : ℕ) (height : ℝ) (circumference : ℝ)
  (turns_eq : turns = 10)
  (height_eq : height = 9)
  (circumference_eq : circumference = 4) :
  wire_length turns height circumference = 41 := 
by
  rw [turns_eq, height_eq, circumference_eq]
  simp [wire_length]
  norm_num
  rw [Real.sqrt_eq_rpow]
  norm_num
  sorry

end spiral_wire_length_l14_14842


namespace hannah_dog_food_l14_14077

def dog_food_consumption : Prop :=
  let dog1 : ℝ := 1.5 * 2
  let dog2 : ℝ := (1.5 * 2) * 1
  let dog3 : ℝ := (dog2 + 2.5) * 3
  let dog4 : ℝ := 1.2 * (dog2 + 2.5) * 2
  let dog5 : ℝ := 0.8 * 1.5 * 4
  let total_food := dog1 + dog2 + dog3 + dog4 + dog5
  total_food = 40.5

theorem hannah_dog_food : dog_food_consumption :=
  sorry

end hannah_dog_food_l14_14077


namespace sara_cakes_sales_l14_14227

theorem sara_cakes_sales :
  let cakes_per_day := 4
  let days_per_week := 5
  let weeks := 4
  let price_per_cake := 8
  let cakes_per_week := cakes_per_day * days_per_week
  let total_cakes := cakes_per_week * weeks
  let total_money := total_cakes * price_per_cake
  total_money = 640 := 
by
  sorry

end sara_cakes_sales_l14_14227


namespace prime_has_two_square_numbers_l14_14042

noncomputable def isSquareNumber (p q : ℕ) : Prop :=
  p > q ∧ Nat.Prime p ∧ Nat.Prime q ∧ ¬ p^2 ∣ (q^(p-1) - 1)

theorem prime_has_two_square_numbers (p : ℕ) (hp : Nat.Prime p) (h5 : p ≥ 5) :
  ∃ q1 q2 : ℕ, isSquareNumber p q1 ∧ isSquareNumber p q2 ∧ q1 ≠ q2 :=
by 
  sorry

end prime_has_two_square_numbers_l14_14042


namespace painting_cost_l14_14346

theorem painting_cost (total_cost : ℕ) (num_paintings : ℕ) (price : ℕ)
  (h1 : total_cost = 104)
  (h2 : 10 < num_paintings)
  (h3 : num_paintings < 60)
  (h4 : total_cost = num_paintings * price)
  (h5 : price ∈ {d ∈ {d : ℕ | d > 0} | total_cost % d = 0}) :
  price = 2 ∨ price = 4 ∨ price = 8 :=
by
  sorry

end painting_cost_l14_14346


namespace trapezoid_QR_length_l14_14582

noncomputable def length_QR (PQ RS area altitude : ℕ) : ℝ :=
  24 - Real.sqrt 11 - 2 * Real.sqrt 24

theorem trapezoid_QR_length :
  ∀ (PQ RS area altitude : ℕ), 
  area = 240 → altitude = 10 → PQ = 12 → RS = 22 →
  length_QR PQ RS area altitude = 24 - Real.sqrt 11 - 2 * Real.sqrt 24 :=
by
  intros PQ RS area altitude h_area h_altitude h_PQ h_RS
  unfold length_QR
  sorry

end trapezoid_QR_length_l14_14582


namespace triangle_perimeter_l14_14028

-- Define the given sides of the triangle
def side_a := 15
def side_b := 6
def side_c := 12

-- Define the function to calculate the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

-- The theorem stating that the perimeter of the given triangle is 33
theorem triangle_perimeter : perimeter side_a side_b side_c = 33 := by
  -- We can include the proof later
  sorry

end triangle_perimeter_l14_14028


namespace science_book_pages_l14_14487

def history_pages := 300
def novel_pages := history_pages / 2
def science_pages := novel_pages * 4

theorem science_book_pages : science_pages = 600 := by
  -- Given conditions:
  -- The novel has half as many pages as the history book, the history book has 300 pages,
  -- and the science book has 4 times as many pages as the novel.
  sorry

end science_book_pages_l14_14487


namespace bridge_extension_length_l14_14991

theorem bridge_extension_length (width_of_river length_of_existing_bridge additional_length_needed : ℕ)
  (h1 : width_of_river = 487)
  (h2 : length_of_existing_bridge = 295)
  (h3 : additional_length_needed = width_of_river - length_of_existing_bridge) :
  additional_length_needed = 192 :=
by {
  -- The steps of the proof would go here, but we use sorry for now.
  sorry
}

end bridge_extension_length_l14_14991


namespace intersection_count_is_one_l14_14268

theorem intersection_count_is_one :
  (∀ x y : ℝ, y = 2 * x^3 + 6 * x + 1 → y = -3 / x^2) → ∃! p : ℝ × ℝ, p.2 = 2 * p.1^3 + 6 * p.1 + 1 ∧ p.2 = -3 / p.1 :=
sorry

end intersection_count_is_one_l14_14268


namespace quotient_of_division_l14_14074

theorem quotient_of_division (a b : ℕ) (r q : ℕ) (h1 : a = 1637) (h2 : b + 1365 = a) (h3 : a = b * q + r) (h4 : r = 5) : q = 6 :=
by
  -- Placeholder for proof
  sorry

end quotient_of_division_l14_14074


namespace plant_arrangement_count_l14_14434

-- Define the count of identical plants
def basil_count := 3
def aloe_count := 2

-- Define the count of identical lamps in each color
def white_lamp_count := 3
def red_lamp_count := 3

-- Define the total ways to arrange the plants under the lamps.
def arrangement_ways := 128

-- Formalize the problem statement proving the arrangements count
theorem plant_arrangement_count :
  (∃ f : Fin (basil_count + aloe_count) → Fin (white_lamp_count + red_lamp_count), True) ↔
  arrangement_ways = 128 :=
sorry

end plant_arrangement_count_l14_14434


namespace dinner_handshakes_l14_14171

def num_couples := 8
def num_people_per_couple := 2
def num_attendees := num_couples * num_people_per_couple

def shakes_per_person (n : Nat) := n - 2
def total_possible_shakes (n : Nat) := (n * shakes_per_person n) / 2

theorem dinner_handshakes : total_possible_shakes num_attendees = 112 :=
by
  sorry

end dinner_handshakes_l14_14171


namespace sequence_a2017_l14_14906

theorem sequence_a2017 (a : ℕ → ℚ) (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n / (3 * a n + 2)) :
  a 2017 = 1 / 3026 :=
sorry

end sequence_a2017_l14_14906


namespace tennis_tournament_l14_14688

noncomputable def tennis_tournament_n (k : ℕ) : ℕ := 8 * k + 1

theorem tennis_tournament (n : ℕ) :
  (∃ k : ℕ, n = tennis_tournament_n k) ↔
  (∃ k : ℕ, n = 8 * k + 1) :=
by sorry

end tennis_tournament_l14_14688


namespace one_fourth_to_fourth_power_is_decimal_l14_14132

def one_fourth : ℚ := 1 / 4

theorem one_fourth_to_fourth_power_is_decimal :
  (one_fourth ^ 4 : ℚ) = 0.00390625 := 
by sorry

end one_fourth_to_fourth_power_is_decimal_l14_14132


namespace intersection_of_A_and_B_l14_14896

-- Define the set A as the solutions to the equation x^2 - 4 = 0
def A : Set ℝ := { x | x^2 - 4 = 0 }

-- Define the set B as the explicit set {1, 2}
def B : Set ℝ := {1, 2}

-- Prove that the intersection of sets A and B is {2}
theorem intersection_of_A_and_B : A ∩ B = {2} :=
by
  unfold A B
  sorry

end intersection_of_A_and_B_l14_14896


namespace paths_E_to_G_through_F_and_H_l14_14137

-- Define positions of E, F, H, and G on the grid.
structure Point where
  x : ℕ
  y : ℕ

def E : Point := { x := 0, y := 0 }
def F : Point := { x := 3, y := 2 }
def H : Point := { x := 5, y := 4 }
def G : Point := { x := 8, y := 4 }

-- Function to calculate number of paths from one point to another given the number of right and down steps
def paths (start goal : Point) : ℕ :=
  let right_steps := goal.x - start.x
  let down_steps := goal.y - start.y
  Nat.choose (right_steps + down_steps) right_steps

theorem paths_E_to_G_through_F_and_H : paths E F * paths F H * paths H G = 60 := by
  sorry

end paths_E_to_G_through_F_and_H_l14_14137


namespace cheese_fries_cost_l14_14367

def jim_money : ℝ := 20
def cousin_money : ℝ := 10
def combined_money : ℝ := jim_money + cousin_money
def expenditure : ℝ := 0.80 * combined_money
def cheeseburger_cost : ℝ := 3
def milkshake_cost : ℝ := 5
def cheeseburgers_cost : ℝ := 2 * cheeseburger_cost
def milkshakes_cost : ℝ := 2 * milkshake_cost
def meal_cost : ℝ := cheeseburgers_cost + milkshakes_cost

theorem cheese_fries_cost :
  let cheese_fries_cost := expenditure - meal_cost 
  cheese_fries_cost = 8 := 
by
  sorry

end cheese_fries_cost_l14_14367


namespace problem_part_I_problem_part_II_l14_14271

-- Define the function f(x) given by the problem
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 5

-- Define the conditions for part (Ⅰ)
def conditions_part_I (a x : ℝ) : Prop :=
  (1 ≤ x ∧ x ≤ a) ∧ (1 ≤ f x a ∧ f x a ≤ a)

-- Lean statement for part (Ⅰ)
theorem problem_part_I (a : ℝ) (h : a > 1) :
  (∀ x, conditions_part_I a x) → a = 2 := by sorry

-- Define the conditions for part (Ⅱ)
def decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 2 → f x a ≥ f y a

def abs_difference_condition (a : ℝ) : Prop :=
  ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ a + 1 ∧ 1 ≤ x2 ∧ x2 ≤ a + 1 → |f x1 a - f x2 a| ≤ 4

-- Lean statement for part (Ⅱ)
theorem problem_part_II (a : ℝ) (h : a > 1) :
  (decreasing_on_interval a) ∧ (abs_difference_condition a) → (2 ≤ a ∧ a ≤ 3) := by sorry

end problem_part_I_problem_part_II_l14_14271


namespace percentage_of_students_owning_cats_l14_14816

theorem percentage_of_students_owning_cats (dogs cats total : ℕ) (h_dogs : dogs = 45) (h_cats : cats = 75) (h_total : total = 500) : 
  (cats / total) * 100 = 15 :=
by
  sorry

end percentage_of_students_owning_cats_l14_14816


namespace original_number_l14_14644

theorem original_number (x : ℤ) (h : x / 2 = 9) : x = 18 := by
  sorry

end original_number_l14_14644


namespace ratio_of_x_intercepts_l14_14332

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v : ℝ)
  (hu : u = -b / 5) (hv : v = -b / 3) : u / v = 3 / 5 := by
  sorry

end ratio_of_x_intercepts_l14_14332


namespace geometric_sequence_form_l14_14053

-- Definitions for sequences and common difference/ratio
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ (m n : ℕ), a n = a m + (n - m) * d

def isGeometricSeq (b : ℕ → ℝ) (q : ℝ) :=
  ∀ (m n : ℕ), b n = b m * q ^ (n - m)

-- Problem statement: given an arithmetic sequence, find the form of the corresponding geometric sequence
theorem geometric_sequence_form
  (b : ℕ → ℝ) (q : ℝ) (m n : ℕ) (b_m : ℝ) (q_pos : q > 0) :
  (∀ (m n : ℕ), b n = b m * q ^ (n - m)) :=
sorry

end geometric_sequence_form_l14_14053


namespace rectangle_area_l14_14614

-- Definitions from conditions:
def side_length : ℕ := 16 / 4
def area_B : ℕ := side_length * side_length
def probability_not_within_B : ℝ := 0.4666666666666667

-- Main statement to prove
theorem rectangle_area (A : ℝ) (h1 : side_length = 4)
 (h2 : area_B = 16)
 (h3 : probability_not_within_B = 0.4666666666666667) :
   A * 0.5333333333333333 = 16 → A = 30 :=
by
  intros h
  sorry


end rectangle_area_l14_14614


namespace jackson_email_problem_l14_14455

variables (E_0 E_1 E_2 E_3 X : ℕ)

/-- Jackson's email deletion and receipt problem -/
theorem jackson_email_problem
  (h1 : E_1 = E_0 - 50 + 15)
  (h2 : E_2 = E_1 - X + 5)
  (h3 : E_3 = E_2 + 10)
  (h4 : E_3 = 30) :
  X = 50 :=
sorry

end jackson_email_problem_l14_14455


namespace sumOddDivisorsOf90_l14_14149

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l14_14149


namespace reduced_price_l14_14192

theorem reduced_price (P R : ℝ) (h1 : R = 0.8 * P) (h2 : 600 = (600 / P + 4) * R) : R = 30 := 
by
  sorry

end reduced_price_l14_14192


namespace min_value_l14_14073

theorem min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : (a - 1) * 1 + 1 * (2 * b) = 0) :
    (2 / a) + (1 / b) = 8 :=
  sorry

end min_value_l14_14073


namespace sum_of_values_l14_14317

def f (x : Int) : Int := Int.natAbs x - 3
def g (x : Int) : Int := -x

def fogof (x : Int) : Int := f (g (f x))

theorem sum_of_values :
  (fogof (-5)) + (fogof (-4)) + (fogof (-3)) + (fogof (-2)) + (fogof (-1)) + (fogof 0) + (fogof 1) + (fogof 2) + (fogof 3) + (fogof 4) + (fogof 5) = -17 :=
by
  sorry

end sum_of_values_l14_14317


namespace consecutive_negative_integers_sum_l14_14597

theorem consecutive_negative_integers_sum (n : ℤ) (hn : n < 0) (hn1 : n + 1 < 0) (hprod : n * (n + 1) = 2550) : n + (n + 1) = -101 :=
by
  sorry

end consecutive_negative_integers_sum_l14_14597


namespace average_age_of_women_is_37_33_l14_14580

noncomputable def women_average_age (A : ℝ) : ℝ :=
  let total_age_men := 12 * A
  let removed_men_age := (25 : ℝ) + 15 + 30
  let new_average := A + 3.5
  let total_age_with_women := 12 * new_average
  let total_age_women := total_age_with_women -  (total_age_men - removed_men_age)
  total_age_women / 3

theorem average_age_of_women_is_37_33 (A : ℝ) (h_avg : women_average_age A = 37.33) :
  true :=
by
  sorry

end average_age_of_women_is_37_33_l14_14580


namespace son_age_18_l14_14663

theorem son_age_18 (F S : ℤ) (h1 : F = S + 20) (h2 : F + 2 = 2 * (S + 2)) : S = 18 :=
by
  sorry

end son_age_18_l14_14663


namespace eval_exp_l14_14783

theorem eval_exp : (3^3)^2 = 729 := sorry

end eval_exp_l14_14783


namespace range_of_a_l14_14849

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * Real.cos (Real.pi / 2 - x)

theorem range_of_a (a : ℝ) (h_condition : f (2 * a ^ 2) + f (a - 3) + f 0 < 0) : -3/2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l14_14849


namespace most_likely_units_digit_is_5_l14_14683

-- Define the problem conditions
def in_range (n : ℕ) := 1 ≤ n ∧ n ≤ 8
def Jack_pick (J : ℕ) := in_range J
def Jill_pick (J K : ℕ) := in_range K ∧ J ≠ K

-- Define the function to get the units digit of the sum
def units_digit (J K : ℕ) := (J + K) % 10

-- Define the proposition stating the most likely units digit is 5
theorem most_likely_units_digit_is_5 :
  ∃ (d : ℕ), d = 5 ∧
    (∃ (J K : ℕ), Jack_pick J → Jill_pick J K → units_digit J K = d) :=
sorry

end most_likely_units_digit_is_5_l14_14683


namespace cindy_correct_answer_l14_14316

/-- 
Cindy accidentally first subtracted 9 from a number, then multiplied the result 
by 2 before dividing by 6, resulting in an answer of 36. 
Following these steps, she was actually supposed to subtract 12 from the 
number and then divide by 8. What would her answer have been had she worked the 
problem correctly?
-/
theorem cindy_correct_answer :
  ∀ (x : ℝ), (2 * (x - 9) / 6 = 36) → ((x - 12) / 8 = 13.125) :=
by
  intro x
  sorry

end cindy_correct_answer_l14_14316


namespace negation_proposition_l14_14664

open Real

theorem negation_proposition (h : ∀ x : ℝ, x^2 - 2*x - 1 > 0) :
  ¬ (∀ x : ℝ, x^2 - 2*x - 1 > 0) = ∃ x_0 : ℝ, x_0^2 - 2*x_0 - 1 ≤ 0 :=
by 
  sorry

end negation_proposition_l14_14664


namespace shrink_ray_coffee_l14_14506

theorem shrink_ray_coffee (num_cups : ℕ) (ounces_per_cup : ℕ) (shrink_factor : ℝ) 
  (h1 : num_cups = 5) 
  (h2 : ounces_per_cup = 8) 
  (h3 : shrink_factor = 0.5) 
  : num_cups * ounces_per_cup * shrink_factor = 20 :=
by
  rw [h1, h2, h3]
  simp
  norm_num

end shrink_ray_coffee_l14_14506


namespace total_packages_sold_l14_14158

variable (P : ℕ)

/-- An automobile parts supplier charges 25 per package of gaskets. 
    When a customer orders more than 10 packages of gaskets, the supplier charges 4/5 the price for each package in excess of 10.
    During a certain week, the supplier received 1150 in payment for the gaskets. --/
def cost (P : ℕ) : ℕ :=
  if P > 10 then 250 + (P - 10) * 20 else P * 25

theorem total_packages_sold :
  cost P = 1150 → P = 55 := by
  sorry

end total_packages_sold_l14_14158


namespace sqrt_14_plus_2_range_l14_14747

theorem sqrt_14_plus_2_range :
  5 < Real.sqrt 14 + 2 ∧ Real.sqrt 14 + 2 < 6 :=
by
  sorry

end sqrt_14_plus_2_range_l14_14747


namespace binom_factorial_eq_120_factorial_l14_14964

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binom_factorial_eq_120_factorial : (factorial (binomial 10 3)) = factorial 120 := by
  sorry

end binom_factorial_eq_120_factorial_l14_14964


namespace locus_of_circle_centers_l14_14566

theorem locus_of_circle_centers (a : ℝ) (x0 y0 : ℝ) :
  { (α, β) | (x0 - α)^2 + (y0 - β)^2 = a^2 } = 
  { (x, y) | (x - x0)^2 + (y - y0)^2 = a^2 } :=
by
  sorry

end locus_of_circle_centers_l14_14566


namespace price_reduction_achieves_profit_l14_14218

theorem price_reduction_achieves_profit :
  ∃ x : ℝ, (40 - x) * (20 + 2 * (x / 4) * 8) = 1200 ∧ x = 20 :=
by
  sorry

end price_reduction_achieves_profit_l14_14218


namespace transform_expression_l14_14386

variable {a : ℝ}

theorem transform_expression (h : a - 1 < 0) : 
  (a - 1) * Real.sqrt (-1 / (a - 1)) = -Real.sqrt (1 - a) :=
by
  sorry

end transform_expression_l14_14386


namespace number_of_words_with_at_least_one_consonant_l14_14199

def total_5_letter_words : ℕ := 6 ^ 5

def total_5_letter_vowel_words : ℕ := 2 ^ 5

def total_5_letter_words_with_consonant : ℕ := total_5_letter_words - total_5_letter_vowel_words

theorem number_of_words_with_at_least_one_consonant :
  total_5_letter_words_with_consonant = 7744 :=
  by
    -- We assert the calculation follows correctly:
    -- total_5_letter_words == 6^5 = 7776
    -- total_5_letter_vowel_words == 2^5 = 32
    -- 7776 - 32 == 7744
    sorry

end number_of_words_with_at_least_one_consonant_l14_14199


namespace find_positive_root_l14_14467

open Real

theorem find_positive_root 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (x : ℝ) :
  sqrt (a * b * x * (a + b + x)) + sqrt (b * c * x * (b + c + x)) + sqrt (c * a * x * (c + a + x)) = sqrt (a * b * c * (a + b + c)) →
  x = (a * b * c) / (a * b + b * c + c * a + 2 * sqrt (a * b * c * (a + b + c))) := 
sorry

end find_positive_root_l14_14467


namespace difference_sum_first_100_odds_evens_l14_14550

def sum_first_n_odds (n : ℕ) : ℕ :=
  n^2

def sum_first_n_evens (n : ℕ) : ℕ :=
  n * (n-1)

theorem difference_sum_first_100_odds_evens :
  sum_first_n_odds 100 - sum_first_n_evens 100 = 100 := by
  sorry

end difference_sum_first_100_odds_evens_l14_14550


namespace sin_of_right_triangle_l14_14274

open Real

theorem sin_of_right_triangle (Q : ℝ) (h : 3 * sin Q = 4 * cos Q) : sin Q = 4 / 5 :=
by
  sorry

end sin_of_right_triangle_l14_14274


namespace range_of_m_l14_14809

theorem range_of_m (x m : ℝ) (h1 : x + 3 = 3 * x - m) (h2 : x ≥ 0) : m ≥ -3 := by
  sorry

end range_of_m_l14_14809


namespace least_integer_x_l14_14523

theorem least_integer_x (x : ℤ) (h : 3 * |x| - 2 * x + 8 < 23) : x = -3 :=
sorry

end least_integer_x_l14_14523


namespace total_flowers_in_3_hours_l14_14300

-- Constants representing the number of each type of flower
def roses : ℕ := 12
def sunflowers : ℕ := 15
def tulips : ℕ := 9
def daisies : ℕ := 18
def orchids : ℕ := 6
def total_flowers : ℕ := 60

-- Number of flowers each bee can pollinate in an hour
def bee_A_rate (roses sunflowers tulips: ℕ) : ℕ := 2 + 3 + 1
def bee_B_rate (daisies orchids: ℕ) : ℕ := 4 + 1
def bee_C_rate (roses sunflowers tulips daisies orchids: ℕ) : ℕ := 1 + 2 + 2 + 3 + 1

-- Total number of flowers pollinated by all bees in an hour
def total_bees_rate (bee_A_rate bee_B_rate bee_C_rate: ℕ) : ℕ := bee_A_rate + bee_B_rate + bee_C_rate

-- Proving the total flowers pollinated in 3 hours
theorem total_flowers_in_3_hours : total_bees_rate 6 5 9 * 3 = total_flowers := 
by {
  sorry
}

end total_flowers_in_3_hours_l14_14300


namespace simplify_expression_l14_14315

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  ((2 * x^2)^3 - 6 * x^3 * (x^3 - 2 * x^2)) / (2 * x^4) = x^2 + 6 * x :=
by 
  -- We provide 'sorry' hack to skip the proof
  -- Replace this with the actual proof to ensure correctness.
  sorry

end simplify_expression_l14_14315


namespace range_of_sum_l14_14224

theorem range_of_sum (x y : ℝ) (h : x^2 + x + y^2 + y = 0) : 
  -2 ≤ x + y ∧ x + y ≤ 0 :=
sorry

end range_of_sum_l14_14224


namespace find_k_l14_14736

def f (n : ℤ) : ℤ :=
if n % 2 = 0 then n / 2 else n + 3

theorem find_k (k : ℤ) (h_odd : k % 2 = 1) (h_f_f_f_k : f (f (f k)) = 27) : k = 105 := by
  sorry

end find_k_l14_14736


namespace candle_duration_1_hour_per_night_l14_14318

-- Definitions based on the conditions
def burn_rate_2_hours (candles: ℕ) (nights: ℕ) : ℕ := nights / candles -- How long each candle lasts when burned for 2 hours per night

-- Given conditions provided
def nights_24 : ℕ := 24
def candles_6 : ℕ := 6

-- The duration a candle lasts when burned for 2 hours every night
def candle_duration_2_hours_per_night : ℕ := burn_rate_2_hours candles_6 nights_24 -- => 4 (not evaluated here)

-- Theorem to prove the duration a candle lasts when burned for 1 hour every night
theorem candle_duration_1_hour_per_night : candle_duration_2_hours_per_night * 2 = 8 :=
by
  sorry -- The proof is omitted, only the statement is required

-- Note: candle_duration_2_hours_per_night = 4 by the given conditions 
-- This leads to 4 * 2 = 8, which matches the required number of nights the candle lasts when burned for 1 hour per night.

end candle_duration_1_hour_per_night_l14_14318


namespace average_speed_l14_14417

theorem average_speed (v : ℝ) (h : 500 / v - 500 / (v + 10) = 2) : v = 45.25 :=
by
  sorry

end average_speed_l14_14417


namespace movies_watched_total_l14_14593

theorem movies_watched_total :
  ∀ (Timothy2009 Theresa2009 Timothy2010 Theresa2010 total : ℕ),
    Timothy2009 = 24 →
    Timothy2010 = Timothy2009 + 7 →
    Theresa2010 = 2 * Timothy2010 →
    Theresa2009 = Timothy2009 / 2 →
    total = Timothy2009 + Timothy2010 + Theresa2009 + Theresa2010 →
    total = 129 :=
by
  intros Timothy2009 Theresa2009 Timothy2010 Theresa2010 total
  sorry

end movies_watched_total_l14_14593


namespace triangle_area_inscribed_rectangle_area_l14_14011

theorem triangle_area (m n : ℝ) : ∃ (S : ℝ), S = m * n := 
sorry

theorem inscribed_rectangle_area (m n : ℝ) : ∃ (A : ℝ), A = (2 * m^2 * n^2) / (m + n)^2 :=
sorry

end triangle_area_inscribed_rectangle_area_l14_14011


namespace seventh_grade_problem_l14_14150

theorem seventh_grade_problem (x y : ℕ) (h1 : x + y = 12) (h2 : 6 * x = 3 * 4 * y) :
  (x + y = 12 ∧ 6 * x = 3 * 4 * y) :=
by
  apply And.intro
  . exact h1
  . exact h2

end seventh_grade_problem_l14_14150


namespace nine_chapters_problem_l14_14633

def cond1 (x y : ℕ) : Prop := y = 6 * x - 6
def cond2 (x y : ℕ) : Prop := y = 5 * x + 5

theorem nine_chapters_problem (x y : ℕ) :
  (cond1 x y ∧ cond2 x y) ↔ (y = 6 * x - 6 ∧ y = 5 * x + 5) :=
by
  sorry

end nine_chapters_problem_l14_14633


namespace solve_inequality_l14_14682

theorem solve_inequality : { x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4 } = { x | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3) } :=
by
  sorry

end solve_inequality_l14_14682


namespace solutions_to_equation_l14_14968

theorem solutions_to_equation :
  ∀ x : ℝ, (x + 1) * (x - 2) = x + 1 ↔ x = -1 ∨ x = 3 :=
by
  sorry

end solutions_to_equation_l14_14968


namespace actual_price_of_food_l14_14128

noncomputable def food_price (total_spent: ℝ) (tip_percent: ℝ) (tax_percent: ℝ) (discount_percent: ℝ) : ℝ :=
  let P := total_spent / ((1 + tip_percent) * (1 + tax_percent) * (1 - discount_percent))
  P

theorem actual_price_of_food :
  food_price 198 0.20 0.10 0.15 = 176.47 :=
by
  sorry

end actual_price_of_food_l14_14128


namespace find_a_b_l14_14528

theorem find_a_b (a b x y : ℝ) (h₀ : a + b = 10) (h₁ : a / x + b / y = 1) (h₂ : x + y = 16) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
    (a = 1 ∧ b = 9) ∨ (a = 9 ∧ b = 1) :=
by
  sorry

end find_a_b_l14_14528


namespace max_min_values_l14_14472

theorem max_min_values (x y : ℝ) 
  (h : (x - 3)^2 + 4 * (y - 1)^2 = 4) :
  ∃ (t u : ℝ), (∀ (z : ℝ), (x-3)^2 + 4*(y-1)^2 = 4 → t ≤ (x+y-3)/(x-y+1) ∧ (x+y-3)/(x-y+1) ≤ u) ∧ t = -1 ∧ u = 1 := 
by
  sorry

end max_min_values_l14_14472


namespace complete_the_square_l14_14480

theorem complete_the_square (x : ℝ) : x^2 + 6 * x + 3 = 0 ↔ (x + 3)^2 = 6 := 
by
  sorry

end complete_the_square_l14_14480


namespace total_house_rent_l14_14477

theorem total_house_rent (P S R : ℕ)
  (h1 : S = 5 * P)
  (h2 : R = 3 * P)
  (h3 : R = 1800) : 
  S + P + R = 5400 :=
by
  sorry

end total_house_rent_l14_14477


namespace triangle_inequality_proof_l14_14306

noncomputable def triangle_inequality (A B C a b c : ℝ) (hABC : A + B + C = Real.pi) : Prop :=
  Real.pi / 3 ≤ (a * A + b * B + c * C) / (a + b + c) ∧ (a * A + b * B + c * C) / (a + b + c) < Real.pi / 2

theorem triangle_inequality_proof (A B C a b c : ℝ) (hABC : A + B + C = Real.pi) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h₁: A + B + C = Real.pi) (h₂: ∀ {x y : ℝ}, A ≥ B  → a ≥ b → A * b + B * a ≤ A * a + B * b) 
  (h₃: ∀ {x y : ℝ}, x + y > 0 → A * x + B * y + C * (a + b - x - y) > 0) : 
  triangle_inequality A B C a b c hABC :=
by
  sorry

end triangle_inequality_proof_l14_14306


namespace remainder_calculation_l14_14920

theorem remainder_calculation : 
  ∀ (dividend divisor quotient remainder : ℕ), 
  dividend = 158 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 :=
by
  intros dividend divisor quotient remainder hdividend hdivisor hquotient heq
  sorry

end remainder_calculation_l14_14920


namespace box_cost_coffee_pods_l14_14739

theorem box_cost_coffee_pods :
  ∀ (days : ℕ) (cups_per_day : ℕ) (pods_per_box : ℕ) (total_cost : ℕ), 
  days = 40 → cups_per_day = 3 → pods_per_box = 30 → total_cost = 32 → 
  total_cost / ((days * cups_per_day) / pods_per_box) = 8 := 
by
  intros days cups_per_day pods_per_box total_cost hday hcup hpod hcost
  sorry

end box_cost_coffee_pods_l14_14739


namespace average_of_xyz_l14_14329

variable (x y z : ℝ)

theorem average_of_xyz (h : (5 / 4) * (x + y + z) = 20) : (x + y + z) / 3 = 16 / 3 := by
  sorry

end average_of_xyz_l14_14329


namespace find_x_parallel_l14_14144

theorem find_x_parallel (x : ℝ) 
  (a : ℝ × ℝ := (x, 2)) 
  (b : ℝ × ℝ := (2, 4)) 
  (h : a.1 * b.2 = a.2 * b.1) :
  x = 1 := 
by
  sorry

end find_x_parallel_l14_14144


namespace solve_z_plus_inv_y_l14_14241

theorem solve_z_plus_inv_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 4) (h3 : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 :=
sorry

end solve_z_plus_inv_y_l14_14241


namespace option_d_is_correct_l14_14364

theorem option_d_is_correct (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b :=
by
  sorry

end option_d_is_correct_l14_14364


namespace playground_girls_l14_14120

theorem playground_girls (total_children boys girls : ℕ) (h1 : boys = 40) (h2 : total_children = 117) (h3 : total_children = boys + girls) : girls = 77 := 
by 
  sorry

end playground_girls_l14_14120


namespace diagonal_length_l14_14089

theorem diagonal_length (d : ℝ) 
  (offset1 offset2 : ℝ) 
  (area : ℝ) 
  (h_offsets : offset1 = 11) 
  (h_offsets2 : offset2 = 9) 
  (h_area : area = 400) : d = 40 :=
by 
  sorry

end diagonal_length_l14_14089


namespace total_salaries_l14_14311

theorem total_salaries (A_salary B_salary : ℝ)
  (hA : A_salary = 1500)
  (hsavings : 0.05 * A_salary = 0.15 * B_salary) :
  A_salary + B_salary = 2000 :=
by {
  sorry
}

end total_salaries_l14_14311


namespace points_difference_l14_14372

theorem points_difference :
  let points_td := 7
  let points_epc := 1
  let points_fg := 3
  
  let touchdowns_BG := 6
  let epc_BG := 4
  let fg_BG := 2
  
  let touchdowns_CF := 8
  let epc_CF := 6
  let fg_CF := 3
  
  let total_BG := touchdowns_BG * points_td + epc_BG * points_epc + fg_BG * points_fg
  let total_CF := touchdowns_CF * points_td + epc_CF * points_epc + fg_CF * points_fg
  
  total_CF - total_BG = 19 := by
  sorry

end points_difference_l14_14372


namespace number_of_oxygen_atoms_l14_14936

theorem number_of_oxygen_atoms 
  (M_weight : ℝ)
  (H_weight : ℝ)
  (Cl_weight : ℝ)
  (O_weight : ℝ)
  (MW_formula : M_weight = H_weight + Cl_weight + n * O_weight)
  (M_weight_eq : M_weight = 68)
  (H_weight_eq : H_weight = 1)
  (Cl_weight_eq : Cl_weight = 35.5)
  (O_weight_eq : O_weight = 16)
  : n = 2 := 
  by sorry

end number_of_oxygen_atoms_l14_14936


namespace isosceles_triangle_EF_length_l14_14229

theorem isosceles_triangle_EF_length (DE DF EF DK EK KF : ℝ)
  (h1 : DE = 5) (h2 : DF = 5) (h3 : DK^2 + EK^2 = DE^2) (h4 : DK^2 + KF^2 = EF^2)
  (h5 : EK + KF = EF) (h6 : EK = 4 * KF) :
  EF = Real.sqrt 10 :=
by sorry

end isosceles_triangle_EF_length_l14_14229


namespace polynomial_solution_l14_14925

theorem polynomial_solution (p : ℝ → ℝ) (h : ∀ x, p (p x) = x * (p x) ^ 2 + x ^ 3) : 
  p = id :=
by {
    sorry
}

end polynomial_solution_l14_14925


namespace shortest_routes_l14_14342

def side_length : ℕ := 10
def refuel_distance : ℕ := 30
def num_squares_per_refuel := refuel_distance / side_length

theorem shortest_routes (A B : Type) (distance_AB : ℕ) (shortest_paths : Π (A B : Type), ℕ) : 
  shortest_paths A B = 54 := by
  sorry

end shortest_routes_l14_14342


namespace packets_of_gum_is_eight_l14_14674

-- Given conditions
def pieces_left : ℕ := 2
def pieces_chewed : ℕ := 54
def pieces_per_packet : ℕ := 7

-- Given he chews all the gum except for pieces_left pieces, and chews pieces_chewed pieces at once
def total_pieces_of_gum (pieces_chewed pieces_left : ℕ) : ℕ :=
  pieces_chewed + pieces_left

-- Calculate the number of packets
def number_of_packets (total_pieces pieces_per_packet : ℕ) : ℕ :=
  total_pieces / pieces_per_packet

-- The final theorem asserting the number of packets is 8
theorem packets_of_gum_is_eight : number_of_packets (total_pieces_of_gum pieces_chewed pieces_left) pieces_per_packet = 8 :=
  sorry

end packets_of_gum_is_eight_l14_14674


namespace smaller_molds_radius_l14_14361

theorem smaller_molds_radius (r : ℝ) : 
  (∀ V_large V_small : ℝ, 
     V_large = (2/3) * π * (2:ℝ)^3 ∧
     V_small = (2/3) * π * r^3 ∧
     8 * V_small = V_large) → r = 1 := by
  sorry

end smaller_molds_radius_l14_14361


namespace pulley_weight_l14_14187

theorem pulley_weight (M g : ℝ) (hM_pos : 0 < M) (F : ℝ := 50) :
  (g ≠ 0) → (M * g = 100) :=
by
  sorry

end pulley_weight_l14_14187


namespace gcd_equation_solutions_l14_14070

theorem gcd_equation_solutions:
  ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y^2 + Nat.gcd x y ^ 3 = x * y * Nat.gcd x y 
  → (x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = 6) ∨ (x = 5 ∧ y = 2) ∨ (x = 5 ∧ y = 3) := 
by
  intros x y h
  sorry

end gcd_equation_solutions_l14_14070


namespace age_difference_l14_14811

theorem age_difference (S M : ℕ) 
  (h1 : S = 35)
  (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 37 :=
by
  sorry

end age_difference_l14_14811


namespace speed_of_sound_l14_14995

theorem speed_of_sound (time_blasts : ℝ) (distance_traveled : ℝ) (time_heard : ℝ) (speed : ℝ) 
  (h_blasts : time_blasts = 30 * 60) -- time between the two blasts in seconds 
  (h_distance : distance_traveled = 8250) -- distance in meters
  (h_heard : time_heard = 30 * 60 + 25) -- time when man heard the second blast
  (h_relationship : speed = distance_traveled / (time_heard - time_blasts)) : 
  speed = 330 :=
sorry

end speed_of_sound_l14_14995


namespace problem_statement_l14_14860

def oper (x : ℕ) (w : ℕ) := (2^x) / (2^w)

theorem problem_statement : ∃ n : ℕ, oper (oper 4 2) n = 2 ↔ n = 3 :=
by sorry

end problem_statement_l14_14860


namespace total_students_l14_14064

theorem total_students (rank_right rank_left : ℕ) (h1 : rank_right = 16) (h2 : rank_left = 6) : rank_right + rank_left - 1 = 21 := by
  sorry

end total_students_l14_14064


namespace arnold_protein_intake_l14_14256

def protein_in_collagen_powder (scoops : ℕ) : ℕ := if scoops = 1 then 9 else 18

def protein_in_protein_powder (scoops : ℕ) : ℕ := 21 * scoops

def protein_in_steak : ℕ := 56

def protein_in_greek_yogurt : ℕ := 15

def protein_in_almonds (cups : ℕ) : ℕ := 6 * cups

theorem arnold_protein_intake :
  protein_in_collagen_powder 1 + 
  protein_in_protein_powder 2 + 
  protein_in_steak + 
  protein_in_greek_yogurt + 
  protein_in_almonds 2 = 134 :=
by
  -- Sorry, the proof is omitted intentionally
  sorry

end arnold_protein_intake_l14_14256


namespace Paige_recycled_pounds_l14_14489

-- Definitions based on conditions from step a)
def points_per_pound := 1 / 4
def friends_pounds_recycled := 2
def total_points := 4

-- The proof statement (no proof required)
theorem Paige_recycled_pounds :
  let total_pounds_recycled := total_points * 4
  let paige_pounds_recycled := total_pounds_recycled - friends_pounds_recycled
  paige_pounds_recycled = 14 :=
by
  sorry

end Paige_recycled_pounds_l14_14489


namespace max_value_sum_faces_edges_vertices_l14_14213

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

def pyramid_faces_added : ℕ := 4
def pyramid_base_faces_covered : ℕ := 1
def pyramid_edges_added : ℕ := 4
def pyramid_vertices_added : ℕ := 1

def resulting_faces : ℕ := rectangular_prism_faces - pyramid_base_faces_covered + pyramid_faces_added
def resulting_edges : ℕ := rectangular_prism_edges + pyramid_edges_added
def resulting_vertices : ℕ := rectangular_prism_vertices + pyramid_vertices_added

def sum_resulting_faces_edges_vertices : ℕ := resulting_faces + resulting_edges + resulting_vertices

theorem max_value_sum_faces_edges_vertices : sum_resulting_faces_edges_vertices = 34 :=
by
  sorry

end max_value_sum_faces_edges_vertices_l14_14213


namespace collinear_points_m_equals_4_l14_14352

theorem collinear_points_m_equals_4 (m : ℝ)
  (h1 : (3 - 12) / (1 - -2) = (-6 - 12) / (m - -2)) : m = 4 :=
by
  sorry

end collinear_points_m_equals_4_l14_14352


namespace linda_age_l14_14006

variable (s j l : ℕ)

theorem linda_age (h1 : (s + j + l) / 3 = 11) 
                  (h2 : l - 5 = s) 
                  (h3 : j + 4 = 3 * (s + 4) / 4) :
                  l = 14 := by
  sorry

end linda_age_l14_14006


namespace count_three_digit_numbers_using_1_and_2_l14_14222

theorem count_three_digit_numbers_using_1_and_2 : 
  let n := 3
  let d := [1, 2]
  let total_combinations := (List.length d)^n
  let invalid_combinations := 2
  total_combinations - invalid_combinations = 6 :=
by
  let n := 3
  let d := [1, 2]
  let total_combinations := (List.length d)^n
  let invalid_combinations := 2
  show total_combinations - invalid_combinations = 6
  sorry

end count_three_digit_numbers_using_1_and_2_l14_14222


namespace fraction_product_l14_14289

theorem fraction_product :
  (3 / 7) * (5 / 8) * (9 / 13) * (11 / 17) = 1485 / 12376 := 
by
  sorry

end fraction_product_l14_14289


namespace lines_intersect_first_quadrant_l14_14832

theorem lines_intersect_first_quadrant (k : ℝ) :
  (∃ (x y : ℝ), 2 * x + 7 * y = 14 ∧ k * x - y = k + 1 ∧ x > 0 ∧ y > 0) ↔ k > 0 :=
by
  sorry

end lines_intersect_first_quadrant_l14_14832


namespace count_4_digit_divisible_by_45_l14_14460

theorem count_4_digit_divisible_by_45 : 
  ∃ n, n = 11 ∧ (∀ a b : ℕ, a + b = 2 ∨ a + b = 11 → (20 + b * 10 + 5) % 45 = 0) :=
sorry

end count_4_digit_divisible_by_45_l14_14460


namespace miriam_pushups_l14_14630

theorem miriam_pushups :
  let p_M := 5
  let p_T := 7
  let p_W := 2 * p_T
  let p_Th := (p_M + p_T + p_W) / 2
  let p_F := p_M + p_T + p_W + p_Th
  p_F = 39 := by
  sorry

end miriam_pushups_l14_14630


namespace Sam_has_4_French_Bulldogs_l14_14761

variable (G F : ℕ)

theorem Sam_has_4_French_Bulldogs
  (h1 : G = 3)
  (h2 : 3 * G + 2 * F = 17) :
  F = 4 :=
sorry

end Sam_has_4_French_Bulldogs_l14_14761


namespace inequality_proof_l14_14608

variable {a b c d : ℝ}

theorem inequality_proof
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_pos_d : 0 < d)
  (h_inequality : a / b < c / d) :
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d := 
by
  sorry

end inequality_proof_l14_14608


namespace mountain_climbing_time_proof_l14_14745

noncomputable def mountain_climbing_time (x : ℝ) : ℝ := (x + 2) / 4

theorem mountain_climbing_time_proof (x : ℝ) (h1 : (x / 3 + (x + 2) / 4 = 4)) : mountain_climbing_time x = 2 := by
  -- assume the given conditions and proof steps explicitly
  sorry

end mountain_climbing_time_proof_l14_14745


namespace max_inscribed_circle_area_of_triangle_l14_14835

theorem max_inscribed_circle_area_of_triangle
  (a b : ℝ)
  (ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (f1 f2 : ℝ × ℝ)
  (F1_coords : f1 = (-1, 0))
  (F2_coords : f2 = (1, 0))
  (P Q : ℝ × ℝ)
  (line_through_F2 : ∀ y : ℝ, x = 1 → y^2 = 9 / 4)
  (P_coords : P = (1, 3/2))
  (Q_coords : Q = (1, -3/2))
  : (π * (3 / 4)^2 = 9 * π / 16) :=
  sorry

end max_inscribed_circle_area_of_triangle_l14_14835


namespace december_sales_fraction_l14_14604

noncomputable def average_sales (A : ℝ) := 11 * A
noncomputable def december_sales (A : ℝ) := 3 * A
noncomputable def total_sales (A : ℝ) := average_sales A + december_sales A

theorem december_sales_fraction (A : ℝ) (h1 : december_sales A = 3 * A)
  (h2 : average_sales A = 11 * A) :
  december_sales A / total_sales A = 3 / 14 :=
by
  sorry

end december_sales_fraction_l14_14604


namespace gcd_difference_5610_210_10_l14_14965

theorem gcd_difference_5610_210_10 : Int.gcd 5610 210 - 10 = 20 := by
  sorry

end gcd_difference_5610_210_10_l14_14965


namespace property_tax_difference_correct_l14_14621

-- Define the tax rates for different ranges
def tax_rate (value : ℕ) : ℝ :=
  if value ≤ 10000 then 0.05
  else if value ≤ 20000 then 0.075
  else if value ≤ 30000 then 0.10
  else 0.125

-- Define the progressive tax calculation for a given assessed value
def calculate_tax (value : ℕ) : ℝ :=
  if value ≤ 10000 then value * 0.05
  else if value ≤ 20000 then 10000 * 0.05 + (value - 10000) * 0.075
  else if value <= 30000 then 10000 * 0.05 + 10000 * 0.075 + (value - 20000) * 0.10
  else 10000 * 0.05 + 10000 * 0.075 + 10000 * 0.10 + (value - 30000) * 0.125

-- Define the initial and new assessed values
def initial_value : ℕ := 20000
def new_value : ℕ := 28000

-- Define the difference in tax calculation
def tax_difference : ℝ := calculate_tax new_value - calculate_tax initial_value

theorem property_tax_difference_correct : tax_difference = 550 := by
  sorry

end property_tax_difference_correct_l14_14621


namespace equation_of_line_perpendicular_l14_14096

theorem equation_of_line_perpendicular 
  (P : ℝ × ℝ) (hx : P.1 = -1) (hy : P.2 = 2)
  (a b c : ℝ) (h_line : 2 * a - 3 * b + 4 = 0)
  (l : ℝ → ℝ) (h_perpendicular : ∀ x, l x = -(3/2) * x)
  (h_passing : l (-1) = 2)
  : a * 3 + b * 2 - 1 = 0 :=
sorry

end equation_of_line_perpendicular_l14_14096


namespace find_y_coordinate_l14_14437

theorem find_y_coordinate (x2 : ℝ) (y1 : ℝ) :
  (∃ m : ℝ, m = (y1 - 0) / (10 - 4) ∧ (-8 - y1) = m * (x2 - 10)) →
  y1 = -8 :=
by
  sorry

end find_y_coordinate_l14_14437


namespace find_a_10_l14_14724

-- We define the arithmetic sequence and sum properties
def arithmetic_seq (a_1 d : ℚ) (a_n : ℕ → ℚ) :=
  ∀ n, a_n n = a_1 + d * n

def sum_arithmetic_seq (a : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  ∀ n, S_n n = n * (a 1 + a n) / 2

-- Conditions given in the problem
def given_conditions (a_1 : ℚ) (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  arithmetic_seq a_1 1 a_n ∧ sum_arithmetic_seq a_n S_n ∧ S_n 6 = 4 * S_n 3

-- The theorem to prove
theorem find_a_10 (a_1 : ℚ) (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) 
  (h : given_conditions a_1 a_n S_n) : a_n 10 = 19 / 2 :=
by sorry

end find_a_10_l14_14724


namespace recurring_fraction_division_l14_14527

noncomputable def recurring_833 := 5 / 6
noncomputable def recurring_1666 := 5 / 3

theorem recurring_fraction_division : 
  (recurring_833 / recurring_1666) = 1 / 2 := 
by 
  sorry

end recurring_fraction_division_l14_14527


namespace quad_eq_pos_neg_root_l14_14941

theorem quad_eq_pos_neg_root (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ + x₂ = 2 ∧ x₁ * x₂ = a + 1) ↔ a < -1 :=
by sorry

end quad_eq_pos_neg_root_l14_14941


namespace sin_30_eq_half_l14_14681

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l14_14681


namespace find_multiple_l14_14310

theorem find_multiple (a b m : ℤ) (h1 : a * b = m * (a + b) + 12) 
(h2 : b = 10) (h3 : b - a = 6) : m = 2 :=
by {
  sorry
}

end find_multiple_l14_14310


namespace max_product_of_sum_300_l14_14394

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l14_14394


namespace angle_sum_straight_line_l14_14722

  theorem angle_sum_straight_line (x : ℝ) (h : 90 + x + 20 = 180) : x = 70 :=
  by
    sorry
  
end angle_sum_straight_line_l14_14722


namespace cos_value_l14_14470

theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by sorry

end cos_value_l14_14470


namespace range_of_m_l14_14440

theorem range_of_m (m : ℝ) (x0 : ℝ)
  (h : (4^(-x0) - m * 2^(-x0 + 1)) = -(4^x0 - m * 2^(x0 + 1))) :
  m ≥ 1/2 :=
sorry

end range_of_m_l14_14440


namespace multiplication_correct_l14_14380

theorem multiplication_correct : 3795421 * 8634.25 = 32774670542.25 := by
  sorry

end multiplication_correct_l14_14380


namespace point_distance_to_focus_of_parabola_with_focus_distance_l14_14570

def parabola_with_focus_distance (focus_distance : ℝ) (p : ℝ × ℝ) : Prop :=
  let f := (0, focus_distance)
  let directrix := -focus_distance
  let (x, y) := p
  let distance_to_focus := Real.sqrt ((x - 0)^2 + (y - focus_distance)^2)
  let distance_to_directrix := abs (y - directrix)
  distance_to_focus = distance_to_directrix

theorem point_distance_to_focus_of_parabola_with_focus_distance 
  (focus_distance : ℝ) (y_axis_distance : ℝ) (p : ℝ × ℝ)
  (h_focus_distance : focus_distance = 4)
  (h_y_axis_distance : abs (p.1) = 1) :
  parabola_with_focus_distance focus_distance p →
  Real.sqrt ((p.1 - 0)^2 + (p.2 - focus_distance)^2) = 5 :=
by
  sorry

end point_distance_to_focus_of_parabola_with_focus_distance_l14_14570


namespace original_number_divisible_l14_14662

theorem original_number_divisible (n : ℕ) (h : (n - 8) % 20 = 0) : n = 28 := 
by
  sorry

end original_number_divisible_l14_14662


namespace impossible_configuration_l14_14293

theorem impossible_configuration : 
  ¬∃ (f : ℕ → ℕ) (h : ∀n, 1 ≤ f n ∧ f n ≤ 5) (perm : ∀i j, if i < j then f i ≠ f j else true), 
  (f 0 = 3) ∧ (f 1 = 4) ∧ (f 2 = 2) ∧ (f 3 = 1) ∧ (f 4 = 5) :=
sorry

end impossible_configuration_l14_14293


namespace smallest_N_divisible_l14_14989

theorem smallest_N_divisible (N x : ℕ) (H: N - 24 = 84 * Nat.lcm x 60) : N = 5064 :=
by
  sorry

end smallest_N_divisible_l14_14989


namespace sum_largest_smallest_gx_l14_14003

noncomputable def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + 3

theorem sum_largest_smallest_gx : (∀ x, 1 ≤ x ∧ x ≤ 10 → True) → ∀ (a b : ℝ), (∃ x, 1 ≤ x ∧ x ≤ 10 ∧ g x = a) → (∃ y, 1 ≤ y ∧ y ≤ 10 ∧ g y = b) → a + b = -1 :=
by
  intro h x y hx hy
  sorry

end sum_largest_smallest_gx_l14_14003


namespace cloth_woven_on_30th_day_l14_14568

theorem cloth_woven_on_30th_day :
  (∃ d : ℚ, (30 * 5 + ((30 * 29) / 2) * d = 390) ∧ (5 + 29 * d = 21)) :=
by sorry

end cloth_woven_on_30th_day_l14_14568


namespace time_in_still_water_l14_14978

-- Define the conditions
variable (S x y : ℝ)
axiom condition1 : S / (x + y) = 6
axiom condition2 : S / (x - y) = 8

-- Define the proof statement
theorem time_in_still_water : S / x = 48 / 7 :=
by
  -- The proof is omitted
  sorry

end time_in_still_water_l14_14978


namespace ratio_of_numbers_l14_14349

theorem ratio_of_numbers (x y : ℕ) (h1 : x + y = 124) (h2 : y = 3 * x) : x / Nat.gcd x y = 1 ∧ y / Nat.gcd x y = 3 := by
  sorry

end ratio_of_numbers_l14_14349


namespace avg_visitors_on_sundays_l14_14125

theorem avg_visitors_on_sundays (avg_other_days : ℕ) (avg_month : ℕ) (days_in_month sundays other_days : ℕ) (total_month_visitors : ℕ) (total_other_days_visitors : ℕ) (S : ℕ):
  avg_other_days = 240 →
  avg_month = 285 →
  days_in_month = 30 →
  sundays = 5 →
  other_days = 25 →
  total_month_visitors = avg_month * days_in_month →
  total_other_days_visitors = avg_other_days * other_days →
  5 * S + total_other_days_visitors = total_month_visitors →
  S = 510 :=
by
  intros _
          _
          _
          _
          _
          _
          _
          h
  -- Proof goes here
  sorry

end avg_visitors_on_sundays_l14_14125


namespace percentage_number_l14_14977

theorem percentage_number (b : ℕ) (h : b = 100) : (320 * b / 100) = 320 :=
by
  sorry

end percentage_number_l14_14977


namespace evaluate_expression_l14_14038

theorem evaluate_expression :
  let sum1 := 3 + 6 + 9
  let sum2 := 2 + 5 + 8
  (sum1 / sum2 - sum2 / sum1) = 11 / 30 :=
by
  let sum1 := 3 + 6 + 9
  let sum2 := 2 + 5 + 8
  sorry

end evaluate_expression_l14_14038


namespace cube_inequality_l14_14450

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
by
  sorry

end cube_inequality_l14_14450


namespace cost_increase_per_scrap_rate_l14_14647

theorem cost_increase_per_scrap_rate (x : ℝ) :
  ∀ x Δx, y = 56 + 8 * x → Δx = 1 → y + Δy = 56 + 8 * (x + Δx) → Δy = 8 :=
by
  sorry

end cost_increase_per_scrap_rate_l14_14647


namespace no_positive_integers_satisfy_equation_l14_14423

theorem no_positive_integers_satisfy_equation :
  ¬ ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a^2 = b^11 + 23 :=
by
  sorry

end no_positive_integers_satisfy_equation_l14_14423


namespace combined_length_of_legs_is_ten_l14_14927

-- Define the conditions given in the problem.
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ c = a * Real.sqrt 2

def hypotenuse_length (c : ℝ) : Prop :=
  c = 7.0710678118654755

def perimeter_condition (a b c perimeter : ℝ) : Prop :=
  perimeter = a + b + c ∧ perimeter = 10 + c

-- Prove the combined length of the two legs is 10.
theorem combined_length_of_legs_is_ten :
  ∃ (a b c : ℝ), is_isosceles_right_triangle a b c →
  hypotenuse_length c →
  ∀ perimeter : ℝ, perimeter_condition a b c perimeter →
  2 * a = 10 :=
by
  sorry

end combined_length_of_legs_is_ten_l14_14927


namespace minimum_b_value_l14_14263

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.log (x^2 - 2 * a))^2

theorem minimum_b_value (a : ℝ) : ∃ x_0 > 0, f x_0 a ≤ (4 / 5) :=
sorry

end minimum_b_value_l14_14263


namespace number_of_mismatching_socks_l14_14446

-- Define the conditions
def total_socks : Nat := 25
def pairs_of_matching_socks : Nat := 4
def socks_per_pair : Nat := 2
def matching_socks : Nat := pairs_of_matching_socks * socks_per_pair

-- State the theorem
theorem number_of_mismatching_socks : total_socks - matching_socks = 17 :=
by
  -- Skip the proof
  sorry

end number_of_mismatching_socks_l14_14446


namespace average_score_first_10_matches_l14_14578

theorem average_score_first_10_matches (A : ℕ) 
  (h1 : 0 < A) 
  (h2 : 10 * A + 15 * 70 = 25 * 66) : A = 60 :=
by
  sorry

end average_score_first_10_matches_l14_14578


namespace paper_clips_in_morning_l14_14807

variable (p : ℕ) (used left : ℕ)

theorem paper_clips_in_morning (h1 : left = 26) (h2 : used = 59) (h3 : left = p - used) : p = 85 :=
by
  sorry

end paper_clips_in_morning_l14_14807


namespace equation1_solutions_equation2_solutions_l14_14942

theorem equation1_solutions (x : ℝ) :
  x ^ 2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 := by
  sorry

theorem equation2_solutions (x : ℝ) :
  2 * x ^ 2 - 2 * x = 1 ↔ x = (1 + Real.sqrt 3) / 2 ∨ x = (1 - Real.sqrt 3) / 2 := by
  sorry

end equation1_solutions_equation2_solutions_l14_14942


namespace Cagney_and_Lacey_Cupcakes_l14_14921

-- Conditions
def CagneyRate := 1 / 25 -- cupcakes per second
def LaceyRate := 1 / 35 -- cupcakes per second
def TotalTimeInSeconds := 10 * 60 -- total time in seconds
def LaceyPrepTimeInSeconds := 1 * 60 -- Lacey's preparation time in seconds
def EffectiveWorkTimeInSeconds := TotalTimeInSeconds - LaceyPrepTimeInSeconds -- effective working time

-- Calculate combined rate
def CombinedRate := 1 / (1 / CagneyRate + 1 / LaceyRate) -- combined rate in cupcakes per second

-- Calculate the total number of cupcakes frosted
def TotalCupcakesFrosted := EffectiveWorkTimeInSeconds * CombinedRate -- total cupcakes frosted

-- We state the theorem that corresponds to our proof problem
theorem Cagney_and_Lacey_Cupcakes : TotalCupcakesFrosted = 37 := by
  sorry

end Cagney_and_Lacey_Cupcakes_l14_14921


namespace proposition_neg_p_and_q_false_l14_14412

theorem proposition_neg_p_and_q_false (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end proposition_neg_p_and_q_false_l14_14412


namespace train_speed_is_5400432_kmh_l14_14197

noncomputable def train_speed_kmh (time_to_pass_platform : ℝ) (time_to_pass_man : ℝ) (length_platform : ℝ) : ℝ :=
  let speed_m_per_s := length_platform / (time_to_pass_platform - time_to_pass_man)
  speed_m_per_s * 3.6

theorem train_speed_is_5400432_kmh :
  train_speed_kmh 35 20 225.018 = 54.00432 :=
by
  sorry

end train_speed_is_5400432_kmh_l14_14197


namespace factorization_identity_l14_14254

theorem factorization_identity (a : ℝ) : (a + 3) * (a - 7) + 25 = (a - 2) ^ 2 :=
by
  sorry

end factorization_identity_l14_14254


namespace problem_1_problem_2_l14_14537

open BigOperators

-- Question 1
theorem problem_1 (a : Fin 2021 → ℝ) :
  (1 + 2 * x) ^ 2020 = ∑ i in Finset.range 2021, a i * x ^ i →
  (∑ i in Finset.range 2021, (i * a i)) = 4040 * 3 ^ 2019 :=
sorry

-- Question 2
theorem problem_2 (a : Fin 2021 → ℝ) :
  (1 - x) ^ 2020 = ∑ i in Finset.range 2021, a i * x ^ i →
  ((∑ i in Finset.range 2021, 1 / a i)) = 2021 / 1011 :=
sorry

end problem_1_problem_2_l14_14537


namespace find_x_l14_14680

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

def a : ℝ × ℝ := (1, 2)

def b (x : ℝ) : ℝ × ℝ := (x, -2)

def c (x : ℝ) : ℝ × ℝ := (1 - x, 4)

theorem find_x (x : ℝ) (h : vector_dot_product a (c x) = 0) : x = 9 :=
by
  sorry

end find_x_l14_14680


namespace domain_of_f_l14_14684

noncomputable def f (x : ℝ) : ℝ := (x^3 + 8) / (x - 8)

theorem domain_of_f : ∀ x : ℝ, x ≠ 8 ↔ ∃ y : ℝ, f x = y :=
  by admit

end domain_of_f_l14_14684


namespace solutions_of_quadratic_eq_l14_14818

theorem solutions_of_quadratic_eq : 
    {x : ℝ | x^2 - 3 * x = 0} = {0, 3} :=
sorry

end solutions_of_quadratic_eq_l14_14818


namespace zoe_candy_bars_needed_l14_14729

def total_cost : ℝ := 485
def grandma_contribution : ℝ := 250
def per_candy_earning : ℝ := 1.25
def required_candy_bars : ℕ := 188

theorem zoe_candy_bars_needed :
  (total_cost - grandma_contribution) / per_candy_earning = required_candy_bars :=
by
  sorry

end zoe_candy_bars_needed_l14_14729


namespace physics_marks_l14_14520

variables (P C M : ℕ)

theorem physics_marks (h1 : P + C + M = 195)
                      (h2 : P + M = 180)
                      (h3 : P + C = 140) : P = 125 :=
by
  sorry

end physics_marks_l14_14520


namespace meeting_time_l14_14720

-- Define the conditions
def distance : ℕ := 600  -- distance between A and B
def speed_A_to_B : ℕ := 70  -- speed of the first person
def speed_B_to_A : ℕ := 80  -- speed of the second person
def start_time : ℕ := 10  -- start time in hours

-- State the problem formally in Lean 4
theorem meeting_time : (distance / (speed_A_to_B + speed_B_to_A)) + start_time = 14 := 
by
  sorry

end meeting_time_l14_14720


namespace quadratic_function_properties_l14_14336

noncomputable def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 :=
by
  sorry

end quadratic_function_properties_l14_14336


namespace g_18_66_l14_14135

def g (x y : ℕ) : ℕ := sorry

axiom g_prop1 : ∀ x, g x x = x
axiom g_prop2 : ∀ x y, g x y = g y x
axiom g_prop3 : ∀ x y, (x + 2 * y) * g x y = y * g x (x + 2 * y)

theorem g_18_66 : g 18 66 = 198 :=
by
  sorry

end g_18_66_l14_14135


namespace geometric_arithmetic_sequence_sum_l14_14298

theorem geometric_arithmetic_sequence_sum {a b : ℕ → ℝ} (q : ℝ) (n : ℕ) 
(h1 : a 2 = 2)
(h2 : a 2 = 2)
(h3 : 2 * (a 3 + 1) = a 2 + a 4)
(h4 : ∀ (n : ℕ), (a (n + 1)) = a 0 * q ^ (n + 1))
(h5 : b n = n * (n + 1)) :
a 8 + (b 8 - b 7) = 144 :=
by { sorry }

end geometric_arithmetic_sequence_sum_l14_14298


namespace leak_empty_tank_time_l14_14753

theorem leak_empty_tank_time (A L : ℝ) (hA : A = 1 / 10) (hAL : A - L = 1 / 15) : (1 / L = 30) :=
sorry

end leak_empty_tank_time_l14_14753


namespace two_pow_n_minus_one_divisible_by_seven_iff_l14_14666

theorem two_pow_n_minus_one_divisible_by_seven_iff (n : ℕ) (h : n > 0) :
  (2^n - 1) % 7 = 0 ↔ n % 3 = 0 :=
sorry

end two_pow_n_minus_one_divisible_by_seven_iff_l14_14666


namespace stu_books_count_l14_14828

theorem stu_books_count (S : ℕ) (h1 : S + 4 * S = 45) : S = 9 := 
by
  sorry

end stu_books_count_l14_14828


namespace lines_parallel_distinct_l14_14497

theorem lines_parallel_distinct (a : ℝ) : 
  (∀ x y : ℝ, (2 * x - a * y + 1 = 0) → ((a - 1) * x - y + a = 0)) ↔ 
  a = 2 := 
sorry

end lines_parallel_distinct_l14_14497


namespace work_completion_time_l14_14391

theorem work_completion_time (A_work_rate B_work_rate C_work_rate : ℝ) 
  (hA : A_work_rate = 1 / 8) 
  (hB : B_work_rate = 1 / 16) 
  (hC : C_work_rate = 1 / 16) : 
  1 / (A_work_rate + B_work_rate + C_work_rate) = 4 :=
by
  -- Proof goes here
  sorry

end work_completion_time_l14_14391


namespace length_of_GH_l14_14441

def EF := 180
def IJ := 120

theorem length_of_GH (EF_parallel_GH : true) (GH_parallel_IJ : true) : GH = 72 := 
sorry

end length_of_GH_l14_14441


namespace gcd_36_48_72_l14_14331

theorem gcd_36_48_72 : Int.gcd (Int.gcd 36 48) 72 = 12 := by
  have h1 : 36 = 2^2 * 3^2 := by norm_num
  have h2 : 48 = 2^4 * 3 := by norm_num
  have h3 : 72 = 2^3 * 3^2 := by norm_num
  sorry

end gcd_36_48_72_l14_14331


namespace power_function_properties_l14_14131

theorem power_function_properties (α : ℝ) (h : (3 : ℝ) ^ α = 27) :
  (α = 3) →
  (∀ x : ℝ, (x ^ α) = x ^ 3) ∧
  (∀ x : ℝ, x ^ α = -(((-x) ^ α))) ∧
  (∀ x y : ℝ, x < y → x ^ α < y ^ α) ∧
  (∀ y : ℝ, ∃ x : ℝ, x ^ α = y) :=
by
  sorry

end power_function_properties_l14_14131


namespace total_wheels_at_park_l14_14022

-- Define the problem based on the given conditions
def num_bicycles : ℕ := 6
def num_tricycles : ℕ := 15
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3

-- Statement to prove the total number of wheels is 57
theorem total_wheels_at_park : (num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle) = 57 := by
  -- This will be filled in with the proof.
  sorry

end total_wheels_at_park_l14_14022


namespace angle_in_third_quadrant_l14_14166

theorem angle_in_third_quadrant (θ : ℝ) (h : θ = 2010) : ((θ % 360) > 180 ∧ (θ % 360) < 270) :=
by
  sorry

end angle_in_third_quadrant_l14_14166


namespace area_of_region_l14_14534

theorem area_of_region : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 6*x - 8*y = 16) → 
  (π * 41) = (π * 41) :=
by
  sorry

end area_of_region_l14_14534


namespace cubic_equation_unique_real_solution_l14_14140

theorem cubic_equation_unique_real_solution :
  (∃ (m : ℝ), ∀ x : ℝ, x^3 - 4*x - m = 0 → x = 2) ↔ m = -8 :=
by sorry

end cubic_equation_unique_real_solution_l14_14140


namespace ted_worked_hours_l14_14695

variable (t : ℝ)
variable (julie_rate ted_rate combined_rate : ℝ)
variable (julie_alone_time : ℝ)
variable (job_done : ℝ)

theorem ted_worked_hours :
  julie_rate = 1 / 10 →
  ted_rate = 1 / 8 →
  combined_rate = julie_rate + ted_rate →
  julie_alone_time = 0.9999999999999998 →
  job_done = combined_rate * t + julie_rate * julie_alone_time →
  t = 4 :=
by
  sorry

end ted_worked_hours_l14_14695


namespace min_inverse_ab_l14_14163

theorem min_inverse_ab (a b : ℝ) (h1 : a + a * b + 2 * b = 30) (h2 : a > 0) (h3 : b > 0) :
  ∃ m : ℝ, m = 1 / 18 ∧ (∀ x y : ℝ, (x + x * y + 2 * y = 30) → (x > 0) → (y > 0) → 1 / (x * y) ≥ m) :=
sorry

end min_inverse_ab_l14_14163


namespace problem_inequality_l14_14841

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

axiom f_pos : ∀ x : ℝ, x > 0 → f x > 0

axiom f_increasing : ∀ x y : ℝ, x > 0 → y > 0 → x ≤ y → (f x / x) ≤ (f y / y)

theorem problem_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * ((f a + f b) / (a + b) + (f b + f c) / (b + c) + (f c + f a) / (c + a)) ≥ 
    3 * (f a + f b + f c) / (a + b + c) + (f a / a + f b / b + f c / c) :=
sorry

end problem_inequality_l14_14841


namespace product_of_repeating_decimal_and_22_l14_14479

noncomputable def repeating_decimal_to_fraction : ℚ :=
  0.45 + 0.0045 * (10 ^ (-2 : ℤ))

theorem product_of_repeating_decimal_and_22 : (repeating_decimal_to_fraction * 22 = 10) :=
by
  sorry

end product_of_repeating_decimal_and_22_l14_14479


namespace complement_intersection_l14_14966

open Set

def U : Set ℕ := {2, 3, 4, 5, 6}
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem complement_intersection :
  (U \ A) ∩ B = {3} :=
sorry

end complement_intersection_l14_14966


namespace articles_produced_l14_14376

theorem articles_produced (x y : ℕ) :
  (x * x * x * (1 / (x^2 : ℝ))) = x → (y * y * y * (1 / (x^2 : ℝ))) = (y^3 / x^2 : ℝ) :=
by
  sorry

end articles_produced_l14_14376


namespace cost_per_bag_of_potatoes_l14_14297

variable (x : ℕ)

def chickens_cost : ℕ := 5 * 3
def celery_cost : ℕ := 4 * 2
def total_paid : ℕ := 35
def potatoes_cost (x : ℕ) : ℕ := 2 * x

theorem cost_per_bag_of_potatoes : 
  chickens_cost + celery_cost + potatoes_cost x = total_paid → x = 6 :=
by
  sorry

end cost_per_bag_of_potatoes_l14_14297


namespace steven_more_peaches_than_apples_l14_14654

-- Definitions
def apples_steven := 11
def peaches_steven := 18

-- Theorem statement
theorem steven_more_peaches_than_apples : (peaches_steven - apples_steven) = 7 := by 
  sorry

end steven_more_peaches_than_apples_l14_14654


namespace trains_cross_in_9_seconds_l14_14000

noncomputable def time_to_cross (length1 length2 : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / (speed1 + speed2)

theorem trains_cross_in_9_seconds :
  time_to_cross 240 260.04 (120 * (5 / 18)) (80 * (5 / 18)) = 9 := 
by
  sorry

end trains_cross_in_9_seconds_l14_14000


namespace user_level_1000_l14_14321

noncomputable def user_level (points : ℕ) : ℕ :=
if points >= 1210 then 18
else if points >= 1000 then 17
else if points >= 810 then 16
else if points >= 640 then 15
else if points >= 490 then 14
else if points >= 360 then 13
else if points >= 250 then 12
else if points >= 160 then 11
else if points >= 90 then 10
else 0

theorem user_level_1000 : user_level 1000 = 17 :=
by {
  -- proof will be written here
  sorry
}

end user_level_1000_l14_14321


namespace charge_per_person_on_second_day_l14_14748

noncomputable def charge_second_day (k : ℕ) (x : ℝ) :=
  let total_revenue := 30 * k + 5 * k * x + 32.5 * k
  let total_visitors := 20 * k
  (total_revenue / total_visitors = 5)

theorem charge_per_person_on_second_day
  (k : ℕ) (hx : charge_second_day k 7.5) :
  7.5 = 7.5 :=
sorry

end charge_per_person_on_second_day_l14_14748


namespace large_cartridge_pages_correct_l14_14359

-- Define the conditions
def small_cartridge_pages : ℕ := 600
def medium_cartridge_pages : ℕ := 2 * 3 * small_cartridge_pages / 6
def large_cartridge_pages : ℕ := 2 * 3 * medium_cartridge_pages / 6

-- The theorem to prove
theorem large_cartridge_pages_correct :
  large_cartridge_pages = 1350 :=
by
  sorry

end large_cartridge_pages_correct_l14_14359


namespace taxes_taken_out_l14_14052

theorem taxes_taken_out
  (gross_pay : ℕ)
  (retirement_percentage : ℝ)
  (net_pay_after_taxes : ℕ)
  (tax_amount : ℕ) :
  gross_pay = 1120 →
  retirement_percentage = 0.25 →
  net_pay_after_taxes = 740 →
  tax_amount = gross_pay - (gross_pay * retirement_percentage) - net_pay_after_taxes :=
by
  sorry

end taxes_taken_out_l14_14052


namespace max_f_l14_14296

noncomputable def f (θ : ℝ) : ℝ :=
  Real.cos (θ / 2) * (1 + Real.sin θ)

theorem max_f : ∀ (θ : ℝ), 0 < θ ∧ θ < π → f θ ≤ (4 * Real.sqrt 3) / 9 :=
by
  sorry

end max_f_l14_14296


namespace watch_cost_l14_14524

variables (w s : ℝ)

theorem watch_cost (h1 : w + s = 120) (h2 : w = 100 + s) : w = 110 :=
by
  sorry

end watch_cost_l14_14524


namespace Tia_drove_192_more_miles_l14_14008

noncomputable def calculate_additional_miles (s_C t_C : ℝ) : ℝ :=
  let d_C := s_C * t_C
  let d_M := (s_C + 8) * (t_C + 3)
  let d_T := (s_C + 12) * (t_C + 4)
  d_T - d_C

theorem Tia_drove_192_more_miles (s_C t_C : ℝ) (h1 : d_M = d_C + 120) (h2 : d_M = (s_C + 8) * (t_C + 3)) : calculate_additional_miles s_C t_C = 192 :=
by {
  sorry
}

end Tia_drove_192_more_miles_l14_14008


namespace vector_parallel_has_value_x_l14_14624

-- Define the vectors a and b
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

-- The theorem statement
theorem vector_parallel_has_value_x :
  ∀ x : ℝ, parallel a (b x) → x = 6 :=
by
  intros x h
  sorry

end vector_parallel_has_value_x_l14_14624


namespace pascal_sixth_element_row_20_l14_14938

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
by 
  sorry

end pascal_sixth_element_row_20_l14_14938


namespace Phillip_correct_total_l14_14651

def number_questions_math : ℕ := 40
def number_questions_english : ℕ := 50
def percentage_correct_math : ℚ := 0.75
def percentage_correct_english : ℚ := 0.98

noncomputable def total_correct_answers : ℚ :=
  (number_questions_math * percentage_correct_math) + (number_questions_english * percentage_correct_english)

theorem Phillip_correct_total : total_correct_answers = 79 := by
  sorry

end Phillip_correct_total_l14_14651


namespace triangle_arithmetic_progression_l14_14228

theorem triangle_arithmetic_progression (a d : ℝ) 
(h1 : (a-2*d)^2 + a^2 = (a+2*d)^2) 
(h2 : ∃ x : ℝ, (a = x * d) ∨ (d = x * a))
: (6 ∣ 6*d) ∧ (12 ∣ 6*d) ∧ (18 ∣ 6*d) ∧ (24 ∣ 6*d) ∧ (30 ∣ 6*d)
:= by
  sorry

end triangle_arithmetic_progression_l14_14228


namespace remainder_consec_even_div12_l14_14188

theorem remainder_consec_even_div12 (n : ℕ) (h: n % 2 = 0)
  (h1: 11234 ≤ n ∧ n + 12 ≥ 11246) : 
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) + (n + 12)) % 12 = 6 :=
by 
  sorry

end remainder_consec_even_div12_l14_14188


namespace factorize_P_l14_14638

noncomputable def P (y : ℝ) : ℝ :=
  (16 * y ^ 7 - 36 * y ^ 5 + 8 * y) - (4 * y ^ 7 - 12 * y ^ 5 - 8 * y)

theorem factorize_P (y : ℝ) : P y = 8 * y * (3 * y ^ 6 - 6 * y ^ 4 + 4) :=
  sorry

end factorize_P_l14_14638


namespace arith_expression_evaluation_l14_14295

theorem arith_expression_evaluation :
  2 + (1/6:ℚ) + (((4.32:ℚ) - 1.68 - (1 + 8/25:ℚ)) * (5/11:ℚ) - (2/7:ℚ)) / (1 + 9/35:ℚ) = 2 + 101/210 := by
  sorry

end arith_expression_evaluation_l14_14295


namespace third_year_students_sampled_correct_l14_14960

-- The given conditions
def first_year_students := 700
def second_year_students := 670
def third_year_students := 630
def total_samples := 200
def total_students := first_year_students + second_year_students + third_year_students

-- The proportion of third-year students
def third_year_proportion := third_year_students / total_students

-- The number of third-year students to be selected
def samples_third_year := total_samples * third_year_proportion

theorem third_year_students_sampled_correct :
  samples_third_year = 63 :=
by
  -- We skip the actual proof for this statement with sorry
  sorry

end third_year_students_sampled_correct_l14_14960


namespace pure_alcohol_added_l14_14872

theorem pure_alcohol_added (x : ℝ) (h1 : 6 * 0.40 = 2.4)
    (h2 : (2.4 + x) / (6 + x) = 0.50) : x = 1.2 :=
by
  sorry

end pure_alcohol_added_l14_14872


namespace point_P_below_line_l14_14794

def line_equation (x y : ℝ) : ℝ := 2 * x - y + 3

def point_below_line (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  2 * x - y + 3 > 0

theorem point_P_below_line :
  point_below_line (1, -1) :=
by
  sorry

end point_P_below_line_l14_14794


namespace portia_high_school_students_l14_14083

theorem portia_high_school_students (P L : ℕ) (h1 : P = 4 * L) (h2 : P + L = 2500) : P = 2000 := by
  sorry

end portia_high_school_students_l14_14083


namespace coeff_x2_expansion_l14_14660

theorem coeff_x2_expansion (n r : ℕ) (a b : ℤ) :
  n = 5 → a = 1 → b = 2 → r = 2 →
  (Nat.choose n r) * (a^(n - r)) * (b^r) = 40 :=
by
  intros Hn Ha Hb Hr
  rw [Hn, Ha, Hb, Hr]
  simp
  sorry

end coeff_x2_expansion_l14_14660


namespace sum_of_integers_l14_14956

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
sorry

end sum_of_integers_l14_14956


namespace prob_below_8_correct_l14_14601

-- Defining the probabilities of hitting the 10, 9, and 8 rings
def prob_10 : ℝ := 0.20
def prob_9 : ℝ := 0.30
def prob_8 : ℝ := 0.10

-- Defining the event of scoring below 8
def prob_below_8 : ℝ := 1 - (prob_10 + prob_9 + prob_8)

-- The main theorem to prove: the probability of scoring below 8 is 0.40
theorem prob_below_8_correct : prob_below_8 = 0.40 :=
by 
  -- We need to show this proof in a separate proof phase
  sorry

end prob_below_8_correct_l14_14601


namespace maria_paid_9_l14_14230

-- Define the conditions as variables/constants
def regular_price : ℝ := 15
def discount_rate : ℝ := 0.40

-- Calculate the discount amount
def discount_amount := discount_rate * regular_price

-- Calculate the final price after discount
def final_price := regular_price - discount_amount

-- The goal is to show that the final price is equal to 9
theorem maria_paid_9 : final_price = 9 := 
by
  -- put your proof here
  sorry

end maria_paid_9_l14_14230


namespace landscape_length_l14_14039

theorem landscape_length (b length : ℕ) (A_playground : ℕ) (h1 : length = 4 * b) (h2 : A_playground = 1200) (h3 : A_playground = (1 / 3 : ℚ) * (length * b)) :
  length = 120 :=
by
  sorry

end landscape_length_l14_14039


namespace arithmetic_sequence_common_difference_l14_14979

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ) 
    (h1 : S 2 = 2 * a_1 + d)
    (h2 : S 3 = 3 * a_1 + 3 * d)
    (h : 2 * S 3 = 3 * S 2 + 6) : d = 2 := 
by
  sorry

end arithmetic_sequence_common_difference_l14_14979


namespace P_sufficient_but_not_necessary_for_Q_l14_14655

def P (x : ℝ) : Prop := abs (2 * x - 3) < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

theorem P_sufficient_but_not_necessary_for_Q : 
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬ P x) := 
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l14_14655


namespace complement_A_complement_A_intersection_B_intersection_A_B_complement_intersection_A_B_l14_14430

def U : Set ℝ := {x | x ≥ -2}
def A : Set ℝ := {x | 2 < x ∧ x < 10}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}

theorem complement_A :
  (U \ A) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x ≥ 10} :=
by sorry

theorem complement_A_intersection_B :
  (U \ A) ∩ B = {2} :=
by sorry

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 8} :=
by sorry

theorem complement_intersection_A_B :
  U \ (A ∩ B) = {x | -2 ≤ x ∧ x ≤ 2 ∨ x > 8} :=
by sorry

end complement_A_complement_A_intersection_B_intersection_A_B_complement_intersection_A_B_l14_14430


namespace smallest_number_is_20_l14_14339

theorem smallest_number_is_20 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a ≤ b) (h5 : b ≤ c)
  (mean_condition : (a + b + c) / 3 = 30)
  (median_condition : b = 31)
  (largest_condition : b = c - 8) :
  a = 20 :=
sorry

end smallest_number_is_20_l14_14339


namespace actual_time_l14_14514

def digit_in_range (a b : ℕ) : Prop := 
  (a = b + 1 ∨ a = b - 1)

def time_malfunctioned (h m : ℕ) : Prop :=
  digit_in_range 0 (h / 10) ∧ -- tens of hour digit (0 -> 1 or 9)
  digit_in_range 0 (h % 10) ∧ -- units of hour digit (0 -> 1 or 9)
  digit_in_range 5 (m / 10) ∧ -- tens of minute digit (5 -> 4 or 6)
  digit_in_range 9 (m % 10)   -- units of minute digit (9 -> 8 or 0)

theorem actual_time : ∃ h m : ℕ, time_malfunctioned h m ∧ h = 11 ∧ m = 48 :=
by
  sorry

end actual_time_l14_14514


namespace tan_angle_sum_l14_14010

theorem tan_angle_sum
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 :=
by
  sorry

end tan_angle_sum_l14_14010


namespace engineering_department_men_l14_14109

theorem engineering_department_men (total_students men_percentage women_count : ℕ) (h_percentage : men_percentage = 70) (h_women : women_count = 180) (h_total : total_students = (women_count * 100) / (100 - men_percentage)) : 
  (total_students * men_percentage / 100) = 420 :=
by
  sorry

end engineering_department_men_l14_14109


namespace solve_for_y_l14_14665

theorem solve_for_y (x y : ℝ) (h : x - 2 = 4 * y + 3) : y = (x - 5) / 4 :=
by
  sorry

end solve_for_y_l14_14665


namespace root_increases_implies_m_neg7_l14_14930

theorem root_increases_implies_m_neg7 
  (m : ℝ) 
  (h : ∃ x : ℝ, x ≠ 3 ∧ x = -m - 4 → x = 3) 
  : m = -7 := by
  sorry

end root_increases_implies_m_neg7_l14_14930


namespace reece_climbs_15_times_l14_14108

/-
Given:
1. Keaton's ladder height: 30 feet.
2. Keaton climbs: 20 times.
3. Reece's ladder is 4 feet shorter than Keaton's ladder.
4. Total length of ladders climbed by both is 11880 inches.

Prove:
Reece climbed his ladder 15 times.
-/

theorem reece_climbs_15_times :
  let keaton_ladder_feet := 30
  let keaton_climbs := 20
  let reece_ladder_feet := keaton_ladder_feet - 4
  let total_length_inches := 11880
  let feet_to_inches (feet : ℕ) := 12 * feet
  let keaton_ladder_inches := feet_to_inches keaton_ladder_feet
  let reece_ladder_inches := feet_to_inches reece_ladder_feet
  let keaton_total_climbed := keaton_ladder_inches * keaton_climbs
  let reece_total_climbed := total_length_inches - keaton_total_climbed
  let reece_climbs := reece_total_climbed / reece_ladder_inches
  reece_climbs = 15 :=
by
  sorry

end reece_climbs_15_times_l14_14108


namespace better_sequence_is_BAB_l14_14130

def loss_prob_andrei : ℝ := 0.4
def loss_prob_boris : ℝ := 0.3

def win_prob_andrei : ℝ := 1 - loss_prob_andrei
def win_prob_boris : ℝ := 1 - loss_prob_boris

def prob_qualify_ABA : ℝ :=
  win_prob_andrei * loss_prob_boris * win_prob_andrei +
  win_prob_andrei * win_prob_boris +
  loss_prob_andrei * win_prob_boris * win_prob_andrei

def prob_qualify_BAB : ℝ :=
  win_prob_boris * loss_prob_andrei * win_prob_boris +
  win_prob_boris * win_prob_andrei +
  loss_prob_boris * win_prob_andrei * win_prob_boris

theorem better_sequence_is_BAB : prob_qualify_BAB = 0.742 ∧ prob_qualify_BAB > prob_qualify_ABA :=
by 
  sorry

end better_sequence_is_BAB_l14_14130


namespace cos_C_in_triangle_l14_14912

theorem cos_C_in_triangle 
  (A B C : ℝ) 
  (h_triangle : A + B + C = 180)
  (sin_A : Real.sin A = 4 / 5) 
  (cos_B : Real.cos B = 12 / 13) : 
  Real.cos C = -16 / 65 :=
by
  sorry

end cos_C_in_triangle_l14_14912


namespace winning_candidate_percentage_votes_l14_14439

theorem winning_candidate_percentage_votes
  (total_votes : ℕ) (majority_votes : ℕ) (P : ℕ) 
  (h1 : total_votes = 6500) 
  (h2 : majority_votes = 1300) 
  (h3 : (P * total_votes) / 100 - ((100 - P) * total_votes) / 100 = majority_votes) : 
  P = 60 :=
sorry

end winning_candidate_percentage_votes_l14_14439


namespace certain_number_value_l14_14738

variable {t b c x : ℕ}

theorem certain_number_value 
  (h1 : (t + b + c + 14 + x) / 5 = 12) 
  (h2 : (t + b + c + 29) / 4 = 15) : 
  x = 15 := 
by
  sorry

end certain_number_value_l14_14738


namespace pump_capacity_l14_14079

-- Define parameters and assumptions
def tank_volume : ℝ := 1000
def fill_percentage : ℝ := 0.85
def fill_time : ℝ := 1
def num_pumps : ℝ := 8
def pump_efficiency : ℝ := 0.75
def required_fill_volume : ℝ := fill_percentage * tank_volume

-- Assumed total effective capacity must meet the required fill volume
theorem pump_capacity (C : ℝ) : 
  (num_pumps * pump_efficiency * C = required_fill_volume) → 
  C = 850.0 / 6.0 :=
by
  sorry

end pump_capacity_l14_14079


namespace variable_is_eleven_l14_14670

theorem variable_is_eleven (x : ℕ) (h : (1/2)^22 * (1/81)^x = 1/(18^22)) : x = 11 :=
by
  sorry

end variable_is_eleven_l14_14670


namespace determine_good_numbers_l14_14395

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), (∀ k : Fin n, ∃ m : ℕ, k.1 + (a k).1 + 1 = m * m)

theorem determine_good_numbers :
  is_good_number 13 ∧ is_good_number 15 ∧ is_good_number 17 ∧ is_good_number 19 ∧ ¬is_good_number 11 :=
by
  sorry

end determine_good_numbers_l14_14395


namespace rearrange_marked_cells_below_diagonal_l14_14812

theorem rearrange_marked_cells_below_diagonal (n : ℕ) (marked_cells : Finset (Fin n × Fin n)) :
  marked_cells.card = n - 1 →
  ∃ row_permutation col_permutation : Equiv (Fin n) (Fin n), ∀ (i j : Fin n),
    (row_permutation i, col_permutation j) ∈ marked_cells → j < i :=
by
  sorry

end rearrange_marked_cells_below_diagonal_l14_14812


namespace ratio_of_w_to_y_l14_14907

variables (w x y z : ℚ)

theorem ratio_of_w_to_y:
  (w / x = 5 / 4) →
  (y / z = 5 / 3) →
  (z / x = 1 / 5) →
  (w / y = 15 / 4) :=
by
  intros hwx hyz hzx
  sorry

end ratio_of_w_to_y_l14_14907


namespace original_number_of_members_l14_14891

-- Define the initial conditions
variables (x y : ℕ)

-- First condition: if five 9-year-old members leave
def condition1 : Prop := x * y - 45 = (y + 1) * (x - 5)

-- Second condition: if five 17-year-old members join
def condition2 : Prop := x * y + 85 = (y + 1) * (x + 5)

-- The theorem to be proven
theorem original_number_of_members (h1 : condition1 x y) (h2 : condition2 x y) : x = 20 :=
by sorry

end original_number_of_members_l14_14891


namespace length_PC_l14_14939

-- Define lengths of the sides of triangle ABC.
def AB := 10
def BC := 8
def CA := 7

-- Define the similarity condition
def similar_triangles (PA PC : ℝ) : Prop :=
  PA / PC = AB / CA

-- Define the extension of side BC to point P
def extension_condition (PA PC : ℝ) : Prop :=
  PA = PC + BC

theorem length_PC (PC : ℝ) (PA : ℝ) :
  similar_triangles PA PC → extension_condition PA PC → PC = 56 / 3 :=
by
  intro h_sim h_ext
  sorry

end length_PC_l14_14939


namespace eighth_graders_taller_rows_remain_ordered_l14_14365

-- Part (a)

theorem eighth_graders_taller {n : ℕ} (h8 : Fin n → ℚ) (h7 : Fin n → ℚ)
  (ordered8 : ∀ i j : Fin n, i ≤ j → h8 i ≤ h8 j)
  (ordered7 : ∀ i j : Fin n, i ≤ j → h7 i ≤ h7 j)
  (initial_condition : ∀ i : Fin n, h8 i > h7 i) :
  ∀ i : Fin n, h8 i > h7 i :=
sorry

-- Part (b)

theorem rows_remain_ordered {m n : ℕ} (h : Fin m → Fin n → ℚ)
  (row_ordered : ∀ i : Fin m, ∀ j k : Fin n, j ≤ k → h i j ≤ h i k)
  (column_ordered_after : ∀ j : Fin n, ∀ i k : Fin m, i ≤ k → h i j ≤ h k j) :
  ∀ i : Fin m, ∀ j k : Fin n, j ≤ k → h i j ≤ h i k :=
sorry

end eighth_graders_taller_rows_remain_ordered_l14_14365


namespace largest_multiple_of_15_less_than_500_l14_14798

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 < 500 ∧ ∀ m : ℕ, m * 15 < 500 → m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l14_14798


namespace fraction_of_left_handed_non_throwers_l14_14058

theorem fraction_of_left_handed_non_throwers 
  (total_players : ℕ) (throwers : ℕ) (right_handed_players : ℕ) (all_throwers_right_handed : throwers ≤ right_handed_players) 
  (total_players_eq : total_players = 70) 
  (throwers_eq : throwers = 46) 
  (right_handed_players_eq : right_handed_players = 62) 
  : (total_players - throwers) = 24 → ((right_handed_players - throwers) = 16 → (24 - 16) = 8 → ((8 : ℚ) / 24 = 1/3)) := 
by 
  intros;
  sorry

end fraction_of_left_handed_non_throwers_l14_14058


namespace min_dot_product_on_hyperbola_l14_14822

theorem min_dot_product_on_hyperbola (x1 y1 x2 y2 : ℝ) 
  (hA : x1^2 - y1^2 = 2) 
  (hB : x2^2 - y2^2 = 2)
  (h_x1 : x1 > 0) 
  (h_x2 : x2 > 0) : 
  x1 * x2 + y1 * y2 ≥ 2 :=
sorry

end min_dot_product_on_hyperbola_l14_14822


namespace dart_hit_number_list_count_l14_14643

def number_of_dart_hit_lists (darts dartboards : ℕ) : ℕ :=
  11  -- Based on the solution, the hard-coded answer is 11.

theorem dart_hit_number_list_count : number_of_dart_hit_lists 6 4 = 11 := 
by 
  sorry

end dart_hit_number_list_count_l14_14643


namespace correct_answers_count_l14_14916

-- Define the conditions from the problem
def total_questions : ℕ := 25
def correct_points : ℕ := 4
def incorrect_points : ℤ := -1
def total_score : ℤ := 85

-- State the theorem
theorem correct_answers_count :
  ∃ x : ℕ, (x ≤ total_questions) ∧ 
           (total_questions - x : ℕ) ≥ 0 ∧ 
           (correct_points * x + incorrect_points * (total_questions - x) = total_score) :=
sorry

end correct_answers_count_l14_14916


namespace greatest_of_three_consecutive_integers_with_sum_21_l14_14723

theorem greatest_of_three_consecutive_integers_with_sum_21 :
  ∃ (x : ℤ), (x + (x + 1) + (x + 2) = 21) ∧ ((x + 2) = 8) :=
by
  sorry

end greatest_of_three_consecutive_integers_with_sum_21_l14_14723


namespace maximum_ab_l14_14652

theorem maximum_ab (a b c : ℝ) (h1 : a + b + c = 4) (h2 : 3 * a + 2 * b - c = 0) : 
  ab <= 1/3 := 
by 
  sorry

end maximum_ab_l14_14652


namespace winnie_retains_lollipops_l14_14350

theorem winnie_retains_lollipops :
  let lollipops_total := 60 + 105 + 5 + 230
  let friends := 13
  lollipops_total % friends = 10 :=
by
  let lollipops_total := 60 + 105 + 5 + 230
  let friends := 13
  show lollipops_total % friends = 10
  sorry

end winnie_retains_lollipops_l14_14350


namespace g_3_2_plus_g_3_5_l14_14093

def g (x y : ℚ) : ℚ :=
  if x + y ≤ 5 then (x * y - x + 3) / (3 * x) else (x * y - y - 3) / (-3 * y)

theorem g_3_2_plus_g_3_5 : g 3 2 + g 3 5 = 1/5 := by
  sorry

end g_3_2_plus_g_3_5_l14_14093


namespace area_of_square_plot_l14_14833

theorem area_of_square_plot (price_per_foot : ℕ) (total_cost : ℕ) (h_price : price_per_foot = 58) (h_cost : total_cost = 2088) :
  ∃ s : ℕ, s^2 = 81 := by
  sorry

end area_of_square_plot_l14_14833


namespace megan_pictures_l14_14796

theorem megan_pictures (pictures_zoo pictures_museum pictures_deleted : ℕ)
  (hzoo : pictures_zoo = 15)
  (hmuseum : pictures_museum = 18)
  (hdeleted : pictures_deleted = 31) :
  (pictures_zoo + pictures_museum) - pictures_deleted = 2 :=
by
  sorry

end megan_pictures_l14_14796


namespace angle_between_bisectors_l14_14483

theorem angle_between_bisectors (β γ : ℝ) (h_sum : β + γ = 130) : (β / 2) + (γ / 2) = 65 :=
by
  have h : β + γ = 130 := h_sum
  sorry

end angle_between_bisectors_l14_14483


namespace mul_582964_99999_l14_14562

theorem mul_582964_99999 : 582964 * 99999 = 58295817036 := by
  sorry

end mul_582964_99999_l14_14562


namespace min_cos_C_l14_14294

theorem min_cos_C (A B C : ℝ) (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
  (h1 : (1 / Real.sin A) + (2 / Real.sin B) = 3 * ((1 / Real.tan A) + (1 / Real.tan B))) :
  Real.cos C ≥ (2 * Real.sqrt 10 - 2) / 9 := 
sorry

end min_cos_C_l14_14294


namespace incorrect_calculation_d_l14_14299

theorem incorrect_calculation_d : (1 / 3) / (-1) ≠ 3 * (-1) := 
by {
  -- we'll leave the body of the proof as sorry.
  sorry
}

end incorrect_calculation_d_l14_14299


namespace f2011_eq_two_l14_14947

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodicity_eqn : ∀ x : ℝ, f (x + 6) = f (x) + f 3
axiom f1_eq_two : f 1 = 2

theorem f2011_eq_two : f 2011 = 2 := 
by 
  sorry

end f2011_eq_two_l14_14947


namespace positive_divisors_840_multiple_of_4_l14_14890

theorem positive_divisors_840_multiple_of_4 :
  let n := 840
  let prime_factors := (2^3 * 3^1 * 5^1 * 7^1)
  (∀ k : ℕ, k ∣ n → k % 4 = 0 → ∀ a b c d : ℕ, 2 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1 →
  k = 2^a * 3^b * 5^c * 7^d) → 
  (∃ count, count = 16) :=
by {
  sorry
}

end positive_divisors_840_multiple_of_4_l14_14890


namespace trick_proof_l14_14013

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l14_14013


namespace max_diff_consecutive_slightly_unlucky_l14_14962

def is_slightly_unlucky (n : ℕ) : Prop := (n.digits 10).sum % 13 = 0

theorem max_diff_consecutive_slightly_unlucky :
  ∃ n m : ℕ, is_slightly_unlucky n ∧ is_slightly_unlucky m ∧ (m > n) ∧ ∀ k, (is_slightly_unlucky k ∧ k > n ∧ k < m) → false → (m - n) = 79 :=
sorry

end max_diff_consecutive_slightly_unlucky_l14_14962


namespace alicia_total_deductions_in_cents_l14_14398

def Alicia_hourly_wage : ℝ := 25
def local_tax_rate : ℝ := 0.015
def retirement_contribution_rate : ℝ := 0.03

theorem alicia_total_deductions_in_cents :
  let wage_cents := Alicia_hourly_wage * 100
  let tax_deduction := wage_cents * local_tax_rate
  let after_tax_earnings := wage_cents - tax_deduction
  let retirement_contribution := after_tax_earnings * retirement_contribution_rate
  let total_deductions := tax_deduction + retirement_contribution
  total_deductions = 111 :=
by
  sorry

end alicia_total_deductions_in_cents_l14_14398


namespace max_tickets_l14_14208

theorem max_tickets (cost : ℝ) (budget : ℝ) (max_tickets : ℕ) (h1 : cost = 15.25) (h2 : budget = 200) :
  max_tickets = 13 :=
by
  sorry

end max_tickets_l14_14208


namespace problem_statement_l14_14735

-- Define the problem
theorem problem_statement (a b : ℝ) (h : a - b = 1 / 2) : -3 * (b - a) = 3 / 2 := 
  sorry

end problem_statement_l14_14735


namespace third_pasture_cows_l14_14813

theorem third_pasture_cows (x y : ℝ) (H1 : x + 27 * y = 18) (H2 : 2 * x + 84 * y = 51) : 
  10 * x + 10 * 3 * y = 60 -> 60 / 3 = 20 :=
by
  sorry

end third_pasture_cows_l14_14813


namespace jacket_total_price_correct_l14_14216

/-- The original price of the jacket -/
def original_price : ℝ := 120

/-- The initial discount rate -/
def initial_discount_rate : ℝ := 0.15

/-- The additional discount in dollars -/
def additional_discount : ℝ := 10

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.10

/-- The calculated total amount the shopper pays for the jacket including all discounts and tax -/
def total_amount_paid : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_rate)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  price_after_additional_discount * (1 + sales_tax_rate)

theorem jacket_total_price_correct : total_amount_paid = 101.20 :=
  sorry

end jacket_total_price_correct_l14_14216


namespace total_cost_l14_14584

-- Definition of the conditions
def cost_sharing (x : ℝ) : Prop :=
  let initial_cost := x / 5
  let new_cost := x / 7
  initial_cost - 15 = new_cost

-- The statement we need to prove
theorem total_cost (x : ℝ) (h : cost_sharing x) : x = 262.50 := by
  sorry

end total_cost_l14_14584


namespace minimize_wood_frame_l14_14473

noncomputable def min_wood_frame (x y : ℝ) : Prop :=
  let area_eq : Prop := x * y + x^2 / 4 = 8
  let length := 2 * (x + y) + Real.sqrt 2 * x
  let y_expr := 8 / x - x / 4
  let length_expr := (3 / 2 + Real.sqrt 2) * x + 16 / x
  let min_x := Real.sqrt (16 / (3 / 2 + Real.sqrt 2))
  area_eq ∧ y = y_expr ∧ length = length_expr ∧ x = 2.343 ∧ y = 2.828

theorem minimize_wood_frame : ∃ x y : ℝ, min_wood_frame x y :=
by
  use 2.343
  use 2.828
  unfold min_wood_frame
  -- we leave the proof of the properties as sorry
  sorry

end minimize_wood_frame_l14_14473


namespace incorrect_statements_l14_14442

-- Definitions based on conditions from the problem.

def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c < 0
def solution_set (a b c : ℝ) : Set ℝ := {x | quadratic_inequality a b c x}
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Lean statements of the conditions and the final proof problem.
theorem incorrect_statements (a b c : ℝ) (M : Set ℝ) :
  (M = ∅ → (a < 0 ∧ discriminant a b c < 0) → false) ∧
  (M = {x | x ≠ x0} → a < b → (a + 4 * c) / (b - a) = 2 + 2 * Real.sqrt 2 → false) := sorry

end incorrect_statements_l14_14442


namespace milan_minutes_billed_l14_14139

noncomputable def total_bill : ℝ := 23.36
noncomputable def monthly_fee : ℝ := 2.00
noncomputable def cost_per_minute : ℝ := 0.12

theorem milan_minutes_billed :
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
sorry

end milan_minutes_billed_l14_14139


namespace solve_for_x_l14_14756

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 9 / (x / 3)) : x = 15 * Real.sqrt 1.8 ∨ x = -15 * Real.sqrt 1.8 := 
by
  sorry

end solve_for_x_l14_14756


namespace jenny_distance_from_school_l14_14616

-- Definitions based on the given conditions.
def kernels_per_feet : ℕ := 1
def feet_per_kernel : ℕ := 25
def squirrel_fraction_eaten : ℚ := 1/4
def remaining_kernels : ℕ := 150

-- Problem statement in Lean 4.
theorem jenny_distance_from_school : 
  ∀ (P : ℕ), (3/4:ℚ) * P = 150 → P * feet_per_kernel = 5000 :=
by
  intros P h
  sorry

end jenny_distance_from_school_l14_14616


namespace minimize_total_resistance_l14_14563

variable (a1 a2 a3 a4 a5 a6 : ℝ)
variable (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧ a5 > a6)

/-- Theorem: Given resistances a1, a2, a3, a4, a5, a6 such that a1 > a2 > a3 > a4 > a5 > a6, 
arranging them in the sequence a1 > a2 > a3 > a4 > a5 > a6 minimizes the total resistance
for the assembled component. -/
theorem minimize_total_resistance : 
  True := 
sorry

end minimize_total_resistance_l14_14563


namespace xyz_value_l14_14185

theorem xyz_value (x y z : ℝ)
    (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14) :
    x * y * z = 16 / 3 := by
    sorry

end xyz_value_l14_14185


namespace probability_of_grid_being_black_l14_14117

noncomputable def probability_grid_black_after_rotation : ℚ := sorry

theorem probability_of_grid_being_black:
  probability_grid_black_after_rotation = 429 / 21845 :=
sorry

end probability_of_grid_being_black_l14_14117


namespace dice_probability_l14_14157

theorem dice_probability :
  let num_dice := 6
  let prob_one_digit := 9 / 20
  let prob_two_digit := 11 / 20
  let num_combinations := Nat.choose num_dice (num_dice / 2)
  let prob_each_combination := (prob_one_digit ^ 3) * (prob_two_digit ^ 3)
  let total_probability := num_combinations * prob_each_combination
  total_probability = 4851495 / 16000000 := by
    let num_dice := 6
    let prob_one_digit := 9 / 20
    let prob_two_digit := 11 / 20
    let num_combinations := Nat.choose num_dice (num_dice / 2)
    let prob_each_combination := (prob_one_digit ^ 3) * (prob_two_digit ^ 3)
    let total_probability := num_combinations * prob_each_combination
    sorry

end dice_probability_l14_14157


namespace final_price_after_discounts_l14_14026

theorem final_price_after_discounts (m : ℝ) : (0.8 * m - 10) = selling_price :=
by
  sorry

end final_price_after_discounts_l14_14026


namespace shuxue_count_l14_14855

theorem shuxue_count : 
  (∃ (count : ℕ), count = (List.length (List.filter (λ n => (30 * n.1 + 3 * n.2 < 100) 
    ∧ (30 * n.1 + 3 * n.2 > 9)) 
      (List.product 
        (List.range' 1 3) -- Possible values for "a" are 1 to 3
        (List.range' 1 9)) -- Possible values for "b" are 1 to 9
    ))) ∧ count = 9 :=
  sorry

end shuxue_count_l14_14855


namespace work_time_relation_l14_14698

theorem work_time_relation (m n k x y z : ℝ) 
    (h1 : 1 / x = m / (y + z)) 
    (h2 : 1 / y = n / (x + z)) 
    (h3 : 1 / z = k / (x + y)) : 
    k = (m + n + 2) / (m * n - 1) :=
by
  sorry

end work_time_relation_l14_14698


namespace sum_three_times_m_and_half_n_square_diff_minus_square_sum_l14_14239

-- Problem (1) Statement
theorem sum_three_times_m_and_half_n (m n : ℝ) : 3 * m + 1 / 2 * n = 3 * m + 1 / 2 * n :=
by
  sorry

-- Problem (2) Statement
theorem square_diff_minus_square_sum (a b : ℝ) : (a - b) ^ 2 - (a + b) ^ 2 = (a - b) ^ 2 - (a + b) ^ 2 :=
by
  sorry

end sum_three_times_m_and_half_n_square_diff_minus_square_sum_l14_14239


namespace model_A_selected_count_l14_14982

def production_A := 1200
def production_B := 6000
def production_C := 2000
def total_selected := 46

def total_production := production_A + production_B + production_C

theorem model_A_selected_count :
  (production_A / total_production) * total_selected = 6 := by
  sorry

end model_A_selected_count_l14_14982


namespace cookies_with_five_cups_l14_14292

-- Define the initial condition: Lee can make 24 cookies with 3 cups of flour
def cookies_per_cup := 24 / 3

-- Theorem stating Lee can make 40 cookies with 5 cups of flour
theorem cookies_with_five_cups : 5 * cookies_per_cup = 40 :=
by
  sorry

end cookies_with_five_cups_l14_14292


namespace parabola_focus_distance_l14_14283

-- defining the problem in Lean
theorem parabola_focus_distance
  (A : ℝ × ℝ)
  (hA : A.2^2 = 4 * A.1)
  (h_distance : |A.1| = 3)
  (F : ℝ × ℝ)
  (hF : F = (1, 0)) :
  |(A.1 - F.1)^2 + (A.2 - F.2)^2| = 4 := 
sorry

end parabola_focus_distance_l14_14283


namespace unique_primes_solution_l14_14475

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_primes_solution (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  p^3 - q^5 = (p + q)^2 ↔ (p = 7 ∧ q = 3) :=
by
  sorry

end unique_primes_solution_l14_14475


namespace minimum_value_of_sum_of_squares_l14_14799

noncomputable def minimum_of_sum_of_squares (a b : ℝ) : ℝ :=
  a^2 + b^2

theorem minimum_value_of_sum_of_squares (a b : ℝ) (h : |a * b| = 6) :
  a^2 + b^2 ≥ 12 :=
by {
  sorry
}

end minimum_value_of_sum_of_squares_l14_14799


namespace profit_percentage_is_25_l14_14234

variable (CP MP : ℝ) (d : ℝ)

/-- Given an article with a cost price of Rs. 85.5, a marked price of Rs. 112.5, 
    and a 5% discount on the marked price, the profit percentage on the cost 
    price is 25%. -/
theorem profit_percentage_is_25
  (hCP : CP = 85.5)
  (hMP : MP = 112.5)
  (hd : d = 0.05) :
  ((MP - (MP * d) - CP) / CP * 100) = 25 := 
sorry

end profit_percentage_is_25_l14_14234


namespace complement_of_A_in_U_l14_14545

open Set

variable (U : Set ℤ := { -2, -1, 0, 1, 2 })
variable (A : Set ℤ := { x | 0 < Int.natAbs x ∧ Int.natAbs x < 2 })

theorem complement_of_A_in_U :
  U \ A = { -2, 0, 2 } :=
by
  sorry

end complement_of_A_in_U_l14_14545


namespace not_divides_l14_14635

theorem not_divides (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) : ¬ d ∣ a^(2^n) + 1 := 
sorry

end not_divides_l14_14635


namespace john_piano_lessons_l14_14555

theorem john_piano_lessons (total_cost piano_cost original_price_per_lesson discount : ℕ) 
    (total_spent : ℕ) : 
    total_spent = piano_cost + ((total_cost - piano_cost) / (original_price_per_lesson - discount)) → 
    total_cost = 1100 ∧ piano_cost = 500 ∧ original_price_per_lesson = 40 ∧ discount = 10 → 
    (total_cost - piano_cost) / (original_price_per_lesson - discount) = 20 :=
by
  intros h1 h2
  sorry

end john_piano_lessons_l14_14555


namespace total_cost_proof_l14_14690

def tuition_fee : ℕ := 1644
def room_and_board_cost : ℕ := tuition_fee - 704
def total_cost : ℕ := tuition_fee + room_and_board_cost

theorem total_cost_proof : total_cost = 2584 := 
by
  sorry

end total_cost_proof_l14_14690


namespace find_value_of_D_l14_14569

theorem find_value_of_D (C : ℕ) (D : ℕ) (k : ℕ) (h : C = (10^D) * k) (hD : k % 10 ≠ 0) : D = 69 := by
  sorry

end find_value_of_D_l14_14569


namespace ratio_of_Patrick_to_Joseph_l14_14970

def countries_traveled_by_George : Nat := 6
def countries_traveled_by_Joseph : Nat := countries_traveled_by_George / 2
def countries_traveled_by_Zack : Nat := 18
def countries_traveled_by_Patrick : Nat := countries_traveled_by_Zack / 2

theorem ratio_of_Patrick_to_Joseph : countries_traveled_by_Patrick / countries_traveled_by_Joseph = 3 :=
by
  -- The definition conditions have already been integrated above
  sorry

end ratio_of_Patrick_to_Joseph_l14_14970


namespace at_least_half_team_B_can_serve_on_submarine_l14_14452

theorem at_least_half_team_B_can_serve_on_submarine
    (max_height : ℕ)
    (team_A_avg_height : ℕ)
    (team_B_median_height : ℕ)
    (team_C_tallest_height : ℕ)
    (team_D_mode_height : ℕ)
    (h1 : max_height = 168)
    (h2 : team_A_avg_height = 166)
    (h3 : team_B_median_height = 167)
    (h4 : team_C_tallest_height = 169)
    (h5 : team_D_mode_height = 167) :
  ∀ (height : ℕ), height ≤ max_height → ∃ (b_sailors : ℕ → Prop) (H : ∃ n, b_sailors n),
  (∃ (n_half : ℕ), (∀ h ≤ team_B_median_height, b_sailors h) ∧ (2 * n_half ≤ n)) :=
sorry

end at_least_half_team_B_can_serve_on_submarine_l14_14452


namespace speed_equation_l14_14969

theorem speed_equation
  (dA dB : ℝ)
  (sB : ℝ)
  (sA : ℝ)
  (time_difference : ℝ)
  (h1 : dA = 800)
  (h2 : dB = 400)
  (h3 : sA = 1.2 * sB)
  (h4 : time_difference = 4) :
  (dA / sA - dB / sB = time_difference) :=
by
  sorry

end speed_equation_l14_14969


namespace equilateral_triangle_sum_l14_14392

theorem equilateral_triangle_sum (x y : ℕ) (h1 : x + 5 = 14) (h2 : y + 11 = 14) : x + y = 12 :=
by
  sorry

end equilateral_triangle_sum_l14_14392


namespace max_value_of_f_l14_14899

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem max_value_of_f (a : ℝ) (h : -2 < a ∧ a ≤ 0) : 
  ∀ x ∈ (Set.Icc 0 (a + 2)), f x ≤ 3 :=
sorry

end max_value_of_f_l14_14899


namespace alexis_total_sewing_time_l14_14414

-- Define the time to sew a skirt and a coat
def t_skirt : ℕ := 2
def t_coat : ℕ := 7

-- Define the numbers of skirts and coats
def n_skirts : ℕ := 6
def n_coats : ℕ := 4

-- Define the total time
def total_time : ℕ := t_skirt * n_skirts + t_coat * n_coats

-- State the theorem
theorem alexis_total_sewing_time : total_time = 40 :=
by
  -- the proof would go here; we're skipping the proof as per instructions
  sorry

end alexis_total_sewing_time_l14_14414


namespace problem_l14_14627

theorem problem {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h : 3 * a * b = a + 3 * b) :
  (3 * a + b >= 16/3) ∧
  (a * b >= 4/3) ∧
  (a^2 + 9 * b^2 >= 8) ∧
  (¬ (b > 1/2)) :=
by
  sorry

end problem_l14_14627


namespace range_of_a_given_quadratic_condition_l14_14909

theorem range_of_a_given_quadratic_condition:
  (∀ (a : ℝ), (∀ (x : ℝ), x^2 - 3 * a * x + 9 ≥ 0) → (-2 ≤ a ∧ a ≤ 2)) :=
by
  sorry

end range_of_a_given_quadratic_condition_l14_14909


namespace compute_cos_2_sum_zero_l14_14986

theorem compute_cos_2_sum_zero (x y z : ℝ)
  (h1 : Real.cos (x + Real.pi / 4) + Real.cos (y + Real.pi / 4) + Real.cos (z + Real.pi / 4) = 0)
  (h2 : Real.sin (x + Real.pi / 4) + Real.sin (y + Real.pi / 4) + Real.sin (z + Real.pi / 4) = 0) :
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 0 :=
by
  sorry

end compute_cos_2_sum_zero_l14_14986


namespace find_k_l14_14795

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def construct_number (k : ℕ) : ℕ :=
  let n := 1000
  let a := (10^(2000 - k) - 1) / 9
  let b := (10^(1001) - 1) / 9
  a * 10^(1001) + k * 10^(1001 - k) - b

theorem find_k : ∀ k : ℕ, (construct_number k > 0) ∧ (isPerfectSquare (construct_number k) ↔ k = 2) := 
by 
  intro k
  sorry

end find_k_l14_14795


namespace find_value_l14_14087

variable (a : ℝ) (h : a + 1/a = 7)

theorem find_value :
  a^2 + 1/a^2 = 47 :=
sorry

end find_value_l14_14087


namespace system1_solution_exists_system2_solution_exists_l14_14875

-- System (1)
theorem system1_solution_exists (x y : ℝ) (h1 : y = 2 * x - 5) (h2 : 3 * x + 4 * y = 2) : 
  x = 2 ∧ y = -1 :=
by
  sorry

-- System (2)
theorem system2_solution_exists (x y : ℝ) (h1 : 3 * x - y = 8) (h2 : (y - 1) / 3 = (x + 5) / 5) : 
  x = 5 ∧ y = 7 :=
by
  sorry

end system1_solution_exists_system2_solution_exists_l14_14875


namespace range_of_a_l14_14701

noncomputable def f (x : ℝ) (a : ℝ) := x * Real.log x + a / x + 3
noncomputable def g (x : ℝ) := x^3 - x^2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Icc (1/2) 2 → x2 ∈ Set.Icc (1/2) 2 → f x1 a - g x2 ≥ 0) →
  1 ≤ a :=
by
  sorry

end range_of_a_l14_14701


namespace wall_width_l14_14803

theorem wall_width (w h l V : ℝ) (h_eq : h = 4 * w) (l_eq : l = 3 * h) (V_eq : V = w * h * l) (v_val : V = 10368) : w = 6 :=
  sorry

end wall_width_l14_14803


namespace range_of_t_l14_14454

theorem range_of_t (a b c t: ℝ) 
  (h1 : 6 * a = 2 * b - 6)
  (h2 : 6 * a = 3 * c)
  (h3 : b ≥ 0)
  (h4 : c ≤ 2)
  (h5 : t = 2 * a + b - c) : 
  0 ≤ t ∧ t ≤ 6 :=
sorry

end range_of_t_l14_14454


namespace best_model_is_model4_l14_14686

-- Define the R^2 values for each model
def R_squared_model1 : ℝ := 0.25
def R_squared_model2 : ℝ := 0.80
def R_squared_model3 : ℝ := 0.50
def R_squared_model4 : ℝ := 0.98

-- Define the highest R^2 value and which model it belongs to
theorem best_model_is_model4 (R1 R2 R3 R4 : ℝ) (h1 : R1 = R_squared_model1) (h2 : R2 = R_squared_model2) (h3 : R3 = R_squared_model3) (h4 : R4 = R_squared_model4) : 
  (R4 = 0.98) ∧ (R4 > R1) ∧ (R4 > R2) ∧ (R4 > R3) :=
by
  sorry

end best_model_is_model4_l14_14686


namespace quadratic_eq_has_two_distinct_real_roots_l14_14319

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Problem statement: Prove that the quadratic equation x^2 + 3x - 2 = 0 has two distinct real roots
theorem quadratic_eq_has_two_distinct_real_roots :
  discriminant 1 3 (-2) > 0 :=
by
  -- Proof goes here
  sorry

end quadratic_eq_has_two_distinct_real_roots_l14_14319


namespace find_desired_expression_l14_14100

variable (y : ℝ)

theorem find_desired_expression
  (h : y + Real.sqrt (y^2 - 4) + (1 / (y - Real.sqrt (y^2 - 4))) = 12) :
  y^2 + Real.sqrt (y^4 - 4) + (1 / (y^2 - Real.sqrt (y^4 - 4))) = 200 / 9 :=
sorry

end find_desired_expression_l14_14100


namespace satisfies_negative_inverse_l14_14066

noncomputable def f1 (x : ℝ) : ℝ := x - 1/x
noncomputable def f2 (x : ℝ) : ℝ := x + 1/x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x
noncomputable def f4 (x : ℝ) : ℝ :=
  if x < 1 then x
  else if x = 1 then 0
  else -1/x

theorem satisfies_negative_inverse :
  { f | (∀ x : ℝ, f (1 / x) = -f x) } = {f1, f3, f4} :=
sorry

end satisfies_negative_inverse_l14_14066


namespace band_members_count_l14_14201

theorem band_members_count :
  ∃ n k m : ℤ, n = 10 * k + 4 ∧ n = 12 * m + 6 ∧ 200 ≤ n ∧ n ≤ 300 ∧ n = 254 :=
by
  -- Declaration of the theorem properties
  sorry

end band_members_count_l14_14201


namespace complementary_angle_ratio_l14_14742

noncomputable def smaller_angle_measure (x : ℝ) : ℝ := 
  3 * (90 / 7)

theorem complementary_angle_ratio :
  ∀ (A B : ℝ), (B = 4 * (90 / 7)) → (A = 3 * (90 / 7)) → 
  (A + B = 90) → A = 38.57142857142857 :=
by
  intros A B hB hA hSum
  sorry

end complementary_angle_ratio_l14_14742


namespace sum_first_11_terms_eq_99_l14_14249

variable {a_n : ℕ → ℝ} -- assuming the sequence values are real numbers
variable (S : ℕ → ℝ) -- sum of the first n terms
variable (a₃ a₆ a₉ : ℝ)
variable (h_sequence : ∀ n, a_n n = aₙ 1 + (n - 1) * (a_n 2 - aₙ 1)) -- sequence is arithmetic
variable (h_condition : a₃ + a₉ = 27 - a₆) -- given condition

theorem sum_first_11_terms_eq_99 
  (h_a₃ : a₃ = a_n 3) 
  (h_a₆ : a₆ = a_n 6) 
  (h_a₉ : a₉ = a_n 9) 
  (h_S : S 11 = 11 * a₆) : 
  S 11 = 99 := 
by 
  sorry


end sum_first_11_terms_eq_99_l14_14249


namespace custom_operation_example_l14_14401

def custom_operation (a b : ℚ) : ℚ :=
  a^3 - 2 * a * b + 4

theorem custom_operation_example : custom_operation 4 (-9) = 140 :=
by
  sorry

end custom_operation_example_l14_14401


namespace contest_B_third_place_4_competitions_l14_14556

/-- Given conditions:
1. There are three contestants: A, B, and C.
2. Scores for the first three places in each knowledge competition are \(a\), \(b\), and \(c\) where \(a > b > c\) and \(a, b, c ∈ ℕ^*\).
3. The final score of A is 26 points.
4. The final scores of both B and C are 11 points.
5. Contestant B won first place in one of the competitions.
Prove that Contestant B won third place in four competitions.
-/
theorem contest_B_third_place_4_competitions
  (a b c : ℕ)
  (ha : a > b)
  (hb : b > c)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hc_pos : 0 < c)
  (hA_score : a + a + a + a + b + c = 26)
  (hB_score : a + c + c + c + c + b = 11)
  (hC_score : b + b + b + b + c + c = 11) :
  ∃ n1 n3 : ℕ,
    n1 = 1 ∧ n3 = 4 ∧
    ∃ k m l p1 p2 : ℕ,
      n1 * a + k * a + l * a + m * a + p1 * a + p2 * a + p1 * b + k * b + p2 * b + n3 * c = 11 := sorry

end contest_B_third_place_4_competitions_l14_14556


namespace problem1_problem2_problem3_l14_14381

variable (a b : ℝ)
variable (h_pos_a : a > 0)
variable (h_pos_b : b > 0)
variable (h_cond1 : a ≥ (1 / a) + (2 / b))
variable (h_cond2 : b ≥ (3 / a) + (2 / b))

/-- Statement 1: Prove that a + b ≥ 4 under the given conditions. -/
theorem problem1 : (a + b) ≥ 4 := 
by 
  sorry

/-- Statement 2: Prove that a^2 + b^2 ≥ 3 + 2√6 under the given conditions. -/
theorem problem2 : (a^2 + b^2) ≥ (3 + 2 * Real.sqrt 6) := 
by 
  sorry

/-- Statement 3: Prove that (1/a) + (1/b) < 1 + (√2/2) under the given conditions. -/
theorem problem3 : (1 / a) + (1 / b) < 1 + (Real.sqrt 2 / 2) := 
by 
  sorry

end problem1_problem2_problem3_l14_14381


namespace largest_possible_value_of_s_l14_14706

theorem largest_possible_value_of_s (r s : Nat) (h1 : r ≥ s) (h2 : s ≥ 3)
  (h3 : (r - 2) * s * 61 = (s - 2) * r * 60) : s = 121 :=
sorry

end largest_possible_value_of_s_l14_14706


namespace boulder_splash_width_l14_14668

theorem boulder_splash_width :
  (6 * (1/4) + 3 * (1 / 2) + 2 * b = 7) -> b = 2 := by
  sorry

end boulder_splash_width_l14_14668


namespace evaluate_fraction_l14_14385

theorem evaluate_fraction : (3 / (1 - 3 / 4) = 12) := by
  have h : (1 - 3 / 4) = 1 / 4 := by
    sorry
  rw [h]
  sorry

end evaluate_fraction_l14_14385


namespace combined_weight_correct_l14_14490

-- Define Jake's present weight
def Jake_weight : ℕ := 196

-- Define the weight loss
def weight_loss : ℕ := 8

-- Define Jake's weight after losing weight
def Jake_weight_after_loss : ℕ := Jake_weight - weight_loss

-- Define the relationship between Jake's weight after loss and his sister's weight
def sister_weight : ℕ := Jake_weight_after_loss / 2

-- Define the combined weight
def combined_weight : ℕ := Jake_weight + sister_weight

-- Prove that the combined weight is 290 pounds
theorem combined_weight_correct : combined_weight = 290 :=
by
  sorry

end combined_weight_correct_l14_14490


namespace egg_processing_l14_14606

theorem egg_processing (E : ℕ) 
  (h1 : (24 / 25) * E + 12 = (99 / 100) * E) : 
  E = 400 :=
sorry

end egg_processing_l14_14606


namespace isosceles_if_interior_angles_equal_l14_14261

-- Definition for a triangle
structure Triangle :=
  (A B C : Type)

-- Defining isosceles triangle condition
def is_isosceles (T : Triangle) :=
  ∃ a b c : ℝ, (a = b) ∨ (b = c) ∨ (a = c)

-- Defining the angle equality condition
def interior_angles_equal (T : Triangle) :=
  ∃ a b c : ℝ, (a = b) ∨ (b = c) ∨ (a = c)

-- Main theorem stating the contrapositive
theorem isosceles_if_interior_angles_equal (T : Triangle) : 
  interior_angles_equal T → is_isosceles T :=
by sorry

end isosceles_if_interior_angles_equal_l14_14261


namespace four_g_users_scientific_notation_l14_14023

-- Condition for scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

-- Given problem in scientific notation form
theorem four_g_users_scientific_notation :
  ∃ a n, is_scientific_notation a n 1030000000 ∧ a = 1.03 ∧ n = 9 :=
sorry

end four_g_users_scientific_notation_l14_14023


namespace pencils_to_sell_for_desired_profit_l14_14830

/-- Definitions based on the conditions provided in the problem. -/
def total_pencils : ℕ := 2000
def cost_per_pencil : ℝ := 0.20
def sell_price_per_pencil : ℝ := 0.40
def desired_profit : ℝ := 160
def total_cost : ℝ := total_pencils * cost_per_pencil

/-- The theorem considers all the conditions and asks to prove the number of pencils to sell -/
theorem pencils_to_sell_for_desired_profit : 
  (desired_profit + total_cost) / sell_price_per_pencil = 1400 :=
by 
  sorry

end pencils_to_sell_for_desired_profit_l14_14830


namespace negation_of_prop_p_l14_14928

open Classical

theorem negation_of_prop_p:
  (¬ ∀ x : ℕ, x > 0 → (1 / 2) ^ x ≤ 1 / 2) ↔ ∃ x : ℕ, x > 0 ∧ (1 / 2) ^ x > 1 / 2 := 
by
  sorry

end negation_of_prop_p_l14_14928


namespace total_number_of_participants_l14_14379

theorem total_number_of_participants (boys_achieving_distance : ℤ) (frequency : ℝ) (h1 : boys_achieving_distance = 8) (h2 : frequency = 0.4) : 
  (boys_achieving_distance : ℝ) / frequency = 20 := 
by 
  sorry

end total_number_of_participants_l14_14379


namespace isosceles_triangle_x_sum_l14_14631

theorem isosceles_triangle_x_sum :
  ∀ (x : ℝ), (∃ (a b : ℝ), a + b + 60 = 180 ∧ (a = x ∨ b = x) ∧ (a = b ∨ a = 60 ∨ b = 60))
  → (60 + 60 + 60 = 180) :=
by
  intro x h
  sorry

end isosceles_triangle_x_sum_l14_14631


namespace green_balloons_count_l14_14980

-- Define the conditions
def total_balloons : Nat := 50
def red_balloons : Nat := 12
def blue_balloons : Nat := 7

-- Define the proof problem
theorem green_balloons_count : 
  let green_balloons := total_balloons - (red_balloons + blue_balloons)
  green_balloons = 31 :=
by
  sorry

end green_balloons_count_l14_14980


namespace four_digit_numbers_neither_5_nor_7_l14_14763

-- Define the range of four-digit numbers
def four_digit_numbers : Set ℕ := {x | 1000 ≤ x ∧ x ≤ 9999}

-- Define the predicates for multiples of 5, 7, and 35
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0
def is_multiple_of_7 (n : ℕ) : Prop := n % 7 = 0
def is_multiple_of_35 (n : ℕ) : Prop := n % 35 = 0

-- Using set notation to define the sets of multiples
def multiples_of_5 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_5 n}
def multiples_of_7 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_7 n}
def multiples_of_35 : Set ℕ := {n | n ∈ four_digit_numbers ∧ is_multiple_of_35 n}

-- Total count of 4-digit numbers
def total_four_digit_numbers : ℕ := 9000

-- Count of multiples of 5, 7, and 35 within 4-digit numbers
def count_multiples_of_5 : ℕ := 1800
def count_multiples_of_7 : ℕ := 1286
def count_multiples_of_35 : ℕ := 257

-- Count of multiples of 5 or 7 using the principle of inclusion-exclusion
def count_multiples_of_5_or_7 : ℕ := count_multiples_of_5 + count_multiples_of_7 - count_multiples_of_35

-- Prove that the number of 4-digit numbers which are multiples of neither 5 nor 7 is 6171
theorem four_digit_numbers_neither_5_nor_7 : 
  (total_four_digit_numbers - count_multiples_of_5_or_7) = 6171 := 
by 
  sorry

end four_digit_numbers_neither_5_nor_7_l14_14763


namespace largest_c_for_range_of_f_l14_14097

def has_real_roots (a b c : ℝ) : Prop :=
  b * b - 4 * a * c ≥ 0

theorem largest_c_for_range_of_f (c : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x + c = 7) ↔ c ≤ 37 / 4 := by
  sorry

end largest_c_for_range_of_f_l14_14097


namespace probability_of_drawing_two_white_balls_l14_14242

-- Define the total number of balls and their colors
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def total_balls : ℕ := red_balls + white_balls

-- Define the total number of ways to draw 2 balls from 4
def total_draw_ways : ℕ := (total_balls.choose 2)

-- Define the number of ways to draw 2 white balls
def white_draw_ways : ℕ := (white_balls.choose 2)

-- Define the probability of drawing 2 white balls
def probability_white_draw : ℚ := white_draw_ways / total_draw_ways

-- The main theorem statement to prove
theorem probability_of_drawing_two_white_balls :
  probability_white_draw = 1 / 6 := by
  sorry

end probability_of_drawing_two_white_balls_l14_14242


namespace complex_number_solution_l14_14588

open Complex

theorem complex_number_solution (z : ℂ) (h : (z - 2 * I) * (2 - I) = 5) : z = 2 + 3 * I :=
  sorry

end complex_number_solution_l14_14588


namespace negation_of_p_is_universal_l14_14971

-- Define the proposition p
def p : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

-- The proof statement for the negation of p
theorem negation_of_p_is_universal : ¬p ↔ ∀ x : ℝ, Real.exp x - x - 1 > 0 :=
by sorry

end negation_of_p_is_universal_l14_14971


namespace cyclists_meet_after_24_minutes_l14_14521

noncomputable def meet_time (D : ℝ) (vm vb : ℝ) : ℝ :=
  D / (2.5 * D - 12)

theorem cyclists_meet_after_24_minutes
  (D vm vb : ℝ)
  (h_vm : 1/3 * vm + 2 = D/2)
  (h_vb : 1/2 * vb = D/2 - 3) :
  meet_time D vm vb = 24 :=
by
  sorry

end cyclists_meet_after_24_minutes_l14_14521


namespace find_side_b_l14_14948

theorem find_side_b
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b * Real.sin A = 3 * c * Real.sin B)
  (h2 : a = 3)
  (h3 : Real.cos B = 2 / 3) :
  b = Real.sqrt 6 :=
by
  sorry

end find_side_b_l14_14948


namespace apples_initial_count_l14_14996

theorem apples_initial_count 
  (trees : ℕ)
  (apples_per_tree_picked : ℕ)
  (apples_picked_in_total : ℕ)
  (apples_remaining : ℕ)
  (initial_apples : ℕ) 
  (h1 : trees = 3) 
  (h2 : apples_per_tree_picked = 8) 
  (h3 : apples_picked_in_total = trees * apples_per_tree_picked)
  (h4 : apples_remaining = 9) 
  (h5 : initial_apples = apples_picked_in_total + apples_remaining) : 
  initial_apples = 33 :=
by sorry

end apples_initial_count_l14_14996


namespace no_integer_roots_l14_14755

theorem no_integer_roots (x : ℤ) : ¬ (x^2 + 2^2018 * x + 2^2019 = 0) :=
sorry

end no_integer_roots_l14_14755


namespace simplify_expression_l14_14047

theorem simplify_expression (a b : ℝ) : 
  (2 * a^2 * b - 5 * a * b) - 2 * (-a * b + a^2 * b) = -3 * a * b :=
by
  sorry

end simplify_expression_l14_14047


namespace bisection_next_interval_l14_14203

def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_next_interval (h₀ : f 2.5 > 0) (h₁ : f 2 < 0) :
  ∃ a b, (2 < 2.5) ∧ f 2 < 0 ∧ f 2.5 > 0 ∧ a = 2 ∧ b = 2.5 :=
by
  sorry

end bisection_next_interval_l14_14203


namespace result_number_of_edges_l14_14880

-- Define the conditions
def hexagon (side_length : ℕ) : Prop := side_length = 1 ∧ (∃ edges, edges = 6 ∧ edges = 6 * side_length)
def triangle (side_length : ℕ) : Prop := side_length = 1 ∧ (∃ edges, edges = 3 ∧ edges = 3 * side_length)

-- State the theorem
theorem result_number_of_edges (side_length_hex : ℕ) (side_length_tri : ℕ)
  (h_h : hexagon side_length_hex) (h_t : triangle side_length_tri)
  (aligned_edge_to_edge : side_length_hex = side_length_tri ∧ side_length_hex = 1 ∧ side_length_tri = 1) :
  ∃ edges, edges = 5 :=
by
  -- Proof is not provided, it is marked with sorry
  sorry

end result_number_of_edges_l14_14880


namespace sum_of_four_triangles_l14_14164

theorem sum_of_four_triangles :
  ∀ (x y : ℝ), 3 * x + 2 * y = 27 → 2 * x + 3 * y = 23 → 4 * y = 12 :=
by
  intros x y h1 h2
  sorry

end sum_of_four_triangles_l14_14164


namespace sum_of_interior_angles_l14_14258

def f (n : ℕ) : ℚ := (n - 2) * 180

theorem sum_of_interior_angles (n : ℕ) : f (n + 1) = f n + 180 :=
by
  unfold f
  sorry

end sum_of_interior_angles_l14_14258


namespace mower_value_drop_l14_14309

theorem mower_value_drop :
  ∀ (initial_value value_six_months value_after_year : ℝ) (percentage_drop_six_months percentage_drop_next_year : ℝ),
  initial_value = 100 →
  percentage_drop_six_months = 0.25 →
  value_six_months = initial_value * (1 - percentage_drop_six_months) →
  value_after_year = 60 →
  percentage_drop_next_year = 1 - (value_after_year / value_six_months) →
  percentage_drop_next_year * 100 = 20 :=
by
  intros initial_value value_six_months value_after_year percentage_drop_six_months percentage_drop_next_year
  intros h1 h2 h3 h4 h5
  sorry

end mower_value_drop_l14_14309


namespace watch_correction_needed_l14_14457

def watch_loses_rate : ℚ := 15 / 4  -- rate of loss per day in minutes
def initial_set_time : ℕ := 15  -- March 15th at 10 A.M.
def report_time : ℕ := 24  -- March 24th at 4 P.M.
def correction (loss_rate per_day min_hrs : ℚ) (days_hrs : ℚ) : ℚ :=
  (days_hrs * (loss_rate / (per_day * min_hrs)))

theorem watch_correction_needed :
  correction watch_loses_rate 24 60 (222) = 34.6875 := 
sorry

end watch_correction_needed_l14_14457


namespace a_is_constant_l14_14036

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_ineq : ∀ n, a n ≥ (a (n+2) + a (n+1) + a (n-1) + a (n-2)) / 4)

theorem a_is_constant : ∀ n m, a n = a m :=
by
  sorry

end a_is_constant_l14_14036


namespace circle_ellipse_intersect_four_points_l14_14793

theorem circle_ellipse_intersect_four_points (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = a^2 → y = x^2 / 2 - a) →
  a > 1 :=
by
  sorry

end circle_ellipse_intersect_four_points_l14_14793


namespace inequality_solution_l14_14544

theorem inequality_solution (x : ℝ) :
    (∀ t : ℝ, abs (t - 3) + abs (2 * t + 1) ≥ abs (2 * x - 1) + abs (x + 2)) ↔ 
    (-1 / 2 ≤ x ∧ x ≤ 5 / 6) :=
by
  sorry

end inequality_solution_l14_14544


namespace remainder_45_to_15_l14_14952

theorem remainder_45_to_15 : ∀ (N : ℤ) (k : ℤ), N = 45 * k + 31 → N % 15 = 1 :=
by
  intros N k h
  sorry

end remainder_45_to_15_l14_14952


namespace income_fraction_from_tips_l14_14400

variable (S T : ℝ)

theorem income_fraction_from_tips :
  (T = (9 / 4) * S) → (T / (S + T) = 9 / 13) :=
by
  sorry

end income_fraction_from_tips_l14_14400


namespace vectors_coplanar_l14_14940

def vector3 := ℝ × ℝ × ℝ

def scalar_triple_product (a b c : vector3) : ℝ :=
  match a, b, c with
  | (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) =>
    a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

theorem vectors_coplanar : scalar_triple_product (-3, 3, 3) (-4, 7, 6) (3, 0, -1) = 0 :=
by
  sorry

end vectors_coplanar_l14_14940


namespace negation_of_existential_l14_14085

theorem negation_of_existential (h : ∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≤ 0) : 
  ∀ x : ℝ, x^3 - x^2 + 1 > 0 :=
sorry

end negation_of_existential_l14_14085


namespace oil_bill_january_l14_14571

theorem oil_bill_january (F J : ℝ)
  (h1 : F / J = 5 / 4)
  (h2 : (F + 30) / J = 3 / 2) :
  J = 120 :=
sorry

end oil_bill_january_l14_14571


namespace exist_xyz_modular_l14_14375

theorem exist_xyz_modular {n a b c : ℕ} (hn : 0 < n) (ha : a ≤ 3 * n ^ 2 + 4 * n) (hb : b ≤ 3 * n ^ 2 + 4 * n) (hc : c ≤ 3 * n ^ 2 + 4 * n) :
  ∃ (x y z : ℤ), abs x ≤ 2 * n ∧ abs y ≤ 2 * n ∧ abs z ≤ 2 * n ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 :=
sorry

end exist_xyz_modular_l14_14375


namespace obtuse_triangle_area_side_l14_14672

theorem obtuse_triangle_area_side (a b : ℝ) (C : ℝ) 
  (h1 : a = 8) 
  (h2 : C = 150 * (π / 180)) -- converting degrees to radians
  (h3 : 1 / 2 * a * b * Real.sin C = 24) : 
  b = 12 :=
by sorry

end obtuse_triangle_area_side_l14_14672


namespace number_of_poles_l14_14418

theorem number_of_poles (side_length : ℝ) (distance_between_poles : ℝ) 
  (h1 : side_length = 150) (h2 : distance_between_poles = 30) : 
  ((4 * side_length) / distance_between_poles) = 20 :=
by 
  -- Placeholder to indicate missing proof
  sorry

end number_of_poles_l14_14418


namespace complement_of_M_in_U_l14_14935

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U :
  U \ M = {3, 5, 6} := by
  sorry

end complement_of_M_in_U_l14_14935


namespace part_a_part_b_l14_14459

-- Part a: Prove for specific numbers 2015 and 2017
theorem part_a : ∃ (x y : ℕ), (2015^2 + 2017^2) / 2 = x^2 + y^2 := sorry

-- Part b: Prove for any two different odd natural numbers
theorem part_b (a b : ℕ) (h1 : a ≠ b) (h2 : a % 2 = 1) (h3 : b % 2 = 1) :
  ∃ (x y : ℕ), (a^2 + b^2) / 2 = x^2 + y^2 := sorry

end part_a_part_b_l14_14459


namespace cos_x_when_sin_x_is_given_l14_14600

theorem cos_x_when_sin_x_is_given (x : ℝ) (h : Real.sin x = (Real.sqrt 5) / 5) :
  Real.cos x = -(Real.sqrt 20) / 5 :=
sorry

end cos_x_when_sin_x_is_given_l14_14600


namespace circle_area_l14_14945

theorem circle_area (C : ℝ) (hC : C = 31.4) : 
  ∃ (A : ℝ), A = 246.49 / π := 
by
  sorry -- proof not required

end circle_area_l14_14945


namespace math_problem_l14_14634

variables (x y z w p q : ℕ)
variables (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) (w_pos : 0 < w)

theorem math_problem
  (h1 : x^3 = y^2)
  (h2 : z^4 = w^3)
  (h3 : z - x = 22)
  (hx : x = p^2)
  (hy : y = p^3)
  (hz : z = q^3)
  (hw : w = q^4) : w - y = q^4 - p^3 :=
sorry

end math_problem_l14_14634


namespace f_bound_l14_14646

-- Define the function f(n) representing the number of representations of n as a sum of powers of 2
noncomputable def f (n : ℕ) : ℕ := 
-- f is defined as described in the problem, implementation skipped here
sorry

-- Propose to prove the main inequality for all n ≥ 3
theorem f_bound (n : ℕ) (h : n ≥ 3) : 2 ^ (n^2 / 4) < f (2 ^ n) ∧ f (2 ^ n) < 2 ^ (n^2 / 2) :=
sorry

end f_bound_l14_14646


namespace total_money_earned_l14_14551

def earning_per_question : ℝ := 0.2
def questions_per_survey : ℕ := 10
def surveys_on_monday : ℕ := 3
def surveys_on_tuesday : ℕ := 4

theorem total_money_earned :
  earning_per_question * (questions_per_survey * (surveys_on_monday + surveys_on_tuesday)) = 14 := by
  sorry

end total_money_earned_l14_14551


namespace exponent_combination_l14_14592

theorem exponent_combination (a : ℝ) (m n : ℕ) (h₁ : a^m = 3) (h₂ : a^n = 4) :
  a^(2 * m + 3 * n) = 576 :=
by
  sorry

end exponent_combination_l14_14592


namespace equivalent_proposition_l14_14138

variable (M : Set α) (m n : α)

theorem equivalent_proposition :
  (m ∈ M → n ∉ M) ↔ (n ∈ M → m ∉ M) := by
  sorry

end equivalent_proposition_l14_14138


namespace probability_longer_piece_l14_14990

theorem probability_longer_piece {x y : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) :
  (∃ (p : ℝ), p = 2 / (x * y + 1)) :=
by
  sorry

end probability_longer_piece_l14_14990


namespace solution_set_of_inequality_l14_14126

theorem solution_set_of_inequality (x : ℝ) : x^2 - |x| - 2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 := by
  sorry

end solution_set_of_inequality_l14_14126


namespace area_contained_by_graph_l14_14836

theorem area_contained_by_graph (x y : ℝ) :
  (|x + y| + |x - y| ≤ 6) → (36 = 36) := by
  sorry

end area_contained_by_graph_l14_14836


namespace pyramid_volume_is_232_l14_14308

noncomputable def pyramid_volume (length : ℝ) (width : ℝ) (slant_height : ℝ) : ℝ :=
  (1 / 3) * (length * width) * (Real.sqrt ((slant_height)^2 - ((length / 2)^2 + (width / 2)^2)))

theorem pyramid_volume_is_232 :
  pyramid_volume 5 10 15 = 232 := 
by
  sorry

end pyramid_volume_is_232_l14_14308


namespace q_at_1_is_zero_l14_14877

-- Define the function q : ℝ → ℝ
-- The conditions imply q(1) = 0
axiom q : ℝ → ℝ

-- Given that (1, 0) is on the graph of y = q(x)
axiom q_condition : q 1 = 0

-- Prove q(1) = 0 given the condition that (1, 0) is on the graph
theorem q_at_1_is_zero : q 1 = 0 :=
by
  exact q_condition

end q_at_1_is_zero_l14_14877


namespace B_alone_finishes_in_19_point_5_days_l14_14172

-- Define the conditions
def is_half_good(A B : ℝ) : Prop := A = 1 / 2 * B
def together_finish_in_13_days(A B : ℝ) : Prop := (A + B) * 13 = 1

-- Define the statement
theorem B_alone_finishes_in_19_point_5_days (A B : ℝ) (h1 : is_half_good A B) (h2 : together_finish_in_13_days A B) :
  B * 19.5 = 1 :=
by
  sorry

end B_alone_finishes_in_19_point_5_days_l14_14172


namespace new_average_amount_l14_14320

theorem new_average_amount (A : ℝ) (H : A = 14) (new_amount : ℝ) (H1 : new_amount = 56) : 
  ((7 * A + new_amount) / 8) = 19.25 :=
by
  rw [H, H1]
  norm_num

end new_average_amount_l14_14320


namespace parabola_vertex_coordinates_l14_14407

theorem parabola_vertex_coordinates :
  ∃ (h k : ℝ), (∀ (x : ℝ), (y = (x - h)^2 + k) = (y = (x-1)^2 + 2)) ∧ h = 1 ∧ k = 2 :=
by
  sorry

end parabola_vertex_coordinates_l14_14407


namespace volume_of_stone_l14_14170

theorem volume_of_stone 
  (width length initial_height final_height : ℕ)
  (h_width : width = 15)
  (h_length : length = 20)
  (h_initial_height : initial_height = 10)
  (h_final_height : final_height = 15)
  : (width * length * final_height - width * length * initial_height = 1500) :=
by
  sorry

end volume_of_stone_l14_14170


namespace total_cost_is_correct_l14_14045

def gravel_cost_per_cubic_foot : ℝ := 8
def discount_rate : ℝ := 0.10
def volume_in_cubic_yards : ℝ := 8
def conversion_factor : ℝ := 27

-- The initial cost for the given volume of gravel in cubic feet
noncomputable def initial_cost : ℝ := gravel_cost_per_cubic_foot * (volume_in_cubic_yards * conversion_factor)

-- The discount amount
noncomputable def discount_amount : ℝ := initial_cost * discount_rate

-- Total cost after applying discount
noncomputable def total_cost_after_discount : ℝ := initial_cost - discount_amount

theorem total_cost_is_correct : total_cost_after_discount = 1555.20 :=
sorry

end total_cost_is_correct_l14_14045


namespace problem_statement_l14_14243

def f (x : ℤ) : ℤ := 2 * x ^ 2 + 3 * x - 1

theorem problem_statement : f (f 3) = 1429 := by
  sorry

end problem_statement_l14_14243


namespace find_m_l14_14598

-- Definitions based on conditions in the problem
def f (x : ℝ) := 4 * x + 7

-- Theorem statement to prove m = 3/4 given the conditions
theorem find_m (m : ℝ) :
  (∀ x : ℝ, f (1/2 * x - 1) = 2 * x + 3) →
  f (m - 1) = 6 →
  m = 3 / 4 :=
by
  -- Proof should go here
  sorry

end find_m_l14_14598


namespace solve_oranges_problem_find_plans_and_max_profit_l14_14851

theorem solve_oranges_problem :
  ∃ (a b : ℕ), 15 * a + 20 * b = 430 ∧ 10 * a + 8 * b = 212 ∧ a = 10 ∧ b = 14 := by
    sorry

theorem find_plans_and_max_profit (a b : ℕ) (h₁ : 15 * a + 20 * b = 430) (h₂ : 10 * a + 8 * b = 212) (ha : a = 10) (hb : b = 14) :
  ∃ (x : ℕ), 58 ≤ x ∧ x ≤ 60 ∧ (10 * x + 14 * (100 - x) ≥ 1160) ∧ (10 * x + 14 * (100 - x) ≤ 1168) ∧ (1000 - 4 * x = 768) :=
    sorry

end solve_oranges_problem_find_plans_and_max_profit_l14_14851


namespace counterexample_not_prime_implies_prime_l14_14975

theorem counterexample_not_prime_implies_prime (n : ℕ) (h₁ : ¬Nat.Prime n) (h₂ : n = 27) : ¬Nat.Prime (n - 2) :=
by
  sorry

end counterexample_not_prime_implies_prime_l14_14975


namespace arithmetic_example_l14_14081

theorem arithmetic_example : 4 * (9 - 6) - 8 = 4 := by
  sorry

end arithmetic_example_l14_14081


namespace count_four_digit_integers_with_1_or_7_l14_14501

/-- 
The total number of four-digit integers with at least one digit being 1 or 7 is 5416.
-/
theorem count_four_digit_integers_with_1_or_7 : 
  let all_four_digit_integers := 9000
  let without_1_or_7 := 7 * 8 * 8 * 8
  let with_1_or_7 := all_four_digit_integers - without_1_or_7
  with_1_or_7 = 5416
:= by
  let all_four_digit_integers := 9000
  let without_1_or_7 := 7 * 8 * 8 * 8
  let with_1_or_7 := all_four_digit_integers - without_1_or_7
  show with_1_or_7 = 5416
  sorry

end count_four_digit_integers_with_1_or_7_l14_14501


namespace number_of_pencils_l14_14659

variable (P L : ℕ)

-- Conditions
def condition1 : Prop := P / L = 5 / 6
def condition2 : Prop := L = P + 5

-- Statement to prove
theorem number_of_pencils (h1 : condition1 P L) (h2 : condition2 P L) : L = 30 :=
  sorry

end number_of_pencils_l14_14659


namespace max_value_expr_l14_14657

variable (x y z : ℝ)

theorem max_value_expr (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (∃ a, ∀ x y z, (a = (x*y + y*z) / (x^2 + y^2 + z^2)) ∧ a ≤ (Real.sqrt 2) / 2) ∧
  (∃ x' y' z', (x' > 0) ∧ (y' > 0) ∧ (z' > 0) ∧ ((x'*y' + y'*z') / (x'^2 + y'^2 + z'^2) = (Real.sqrt 2) / 2)) :=
by
  sorry

end max_value_expr_l14_14657


namespace perfect_square_append_100_digits_l14_14705

-- Define the number X consisting of 99 nines

def X : ℕ := (10^99 - 1)

theorem perfect_square_append_100_digits :
  ∃ n : ℕ, X * 10^100 ≤ n^2 ∧ n^2 < X * 10^100 + 10^100 :=
by 
  sorry

end perfect_square_append_100_digits_l14_14705


namespace nathan_ate_total_gumballs_l14_14725

-- Define the constants and variables based on the conditions
def gumballs_small : Nat := 5
def gumballs_medium : Nat := 12
def gumballs_large : Nat := 20
def small_packages : Nat := 4
def medium_packages : Nat := 3
def large_packages : Nat := 2

-- The total number of gumballs Nathan ate
def total_gumballs : Nat := (small_packages * gumballs_small) + (medium_packages * gumballs_medium) + (large_packages * gumballs_large)

-- The theorem to prove
theorem nathan_ate_total_gumballs : total_gumballs = 96 :=
by
  unfold total_gumballs
  sorry

end nathan_ate_total_gumballs_l14_14725


namespace sufficient_not_necessary_condition_l14_14814

theorem sufficient_not_necessary_condition (x : ℝ) : (x ≥ 3 → (x - 2) ≥ 0) ∧ ((x - 2) ≥ 0 → x ≥ 3) = false :=
by
  sorry

end sufficient_not_necessary_condition_l14_14814


namespace system_of_equations_solution_l14_14348

theorem system_of_equations_solution (b : ℝ) :
  (∀ (a : ℝ), ∃ (x y : ℝ), (x - 1)^2 + y^2 = 1 ∧ a * x + y = a * b) ↔ 0 ≤ b ∧ b ≤ 2 :=
by
  sorry

end system_of_equations_solution_l14_14348


namespace part1_part2_l14_14338

variables (a x : ℝ)

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (|x - 1| ≤ 2) ∧ ((x + 3) / (x - 2) ≥ 0)

-- Part 1
theorem part1 (h_a : a = 1) (h_p : p a x) (h_q : q x) : 2 < x ∧ x < 3 := sorry

-- Part 2
theorem part2 (h_suff : ∀ x, q x → p a x) : 1 < a ∧ a ≤ 2 := sorry

end part1_part2_l14_14338


namespace negation_of_existence_l14_14788

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ ∀ x : ℝ, x^2 + 2 * a * x + a > 0 :=
by
  sorry

end negation_of_existence_l14_14788


namespace positive_x_condition_l14_14496

theorem positive_x_condition (x : ℝ) (h : x > 0 ∧ (0.01 * x * x = 9)) : x = 30 :=
sorry

end positive_x_condition_l14_14496


namespace sum_lent_eq_1100_l14_14974

def interest_rate : ℚ := 6 / 100

def period : ℕ := 8

def interest_amount (P : ℚ) : ℚ :=
  period * interest_rate * P

def total_interest_eq_principal_minus_572 (P: ℚ) : Prop :=
  interest_amount P = P - 572

theorem sum_lent_eq_1100 : ∃ P : ℚ, total_interest_eq_principal_minus_572 P ∧ P = 1100 :=
by
  use 1100
  sorry

end sum_lent_eq_1100_l14_14974


namespace no_roots_in_interval_l14_14369

theorem no_roots_in_interval (a : ℝ) (x : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h_eq: a ^ x + a ^ (-x) = 2 * a) : x < -1 ∨ x > 1 :=
sorry

end no_roots_in_interval_l14_14369


namespace sum_of_repeating_decimals_l14_14509

-- Definitions for periodic decimals
def repeating_five := 5 / 9
def repeating_seven := 7 / 9

-- Theorem statement
theorem sum_of_repeating_decimals : (repeating_five + repeating_seven) = 4 / 3 :=
by
  -- Placeholder for the proof
  sorry

end sum_of_repeating_decimals_l14_14509


namespace tangent_circle_radius_l14_14281

theorem tangent_circle_radius (O A B C : ℝ) (r1 r2 : ℝ) :
  (O = 5) →
  (abs (A - B) = 8) →
  (C = (2 * A + B) / 3) →
  r1 = 8 / 9 ∨ r2 = 32 / 9 :=
sorry

end tangent_circle_radius_l14_14281


namespace hot_dogs_remainder_l14_14136

theorem hot_dogs_remainder :
  let n := 16789537
  let d := 5
  n % d = 2 :=
by
  sorry

end hot_dogs_remainder_l14_14136


namespace smallest_number_of_eggs_l14_14715

-- Define the conditions given in the problem
def total_containers (c : ℕ) : ℕ := 15 * c - 3

-- Prove that given the conditions, the smallest number of eggs you could have is 162
theorem smallest_number_of_eggs (h : ∃ c : ℕ, total_containers c > 150) : ∃ c : ℕ, total_containers c = 162 :=
by
  sorry

end smallest_number_of_eggs_l14_14715


namespace fraction_simplified_l14_14191

-- Define the fraction function
def fraction (n : ℕ) := (21 * n + 4, 14 * n + 3)

-- Define the gcd function to check if fractions are simplified.
def is_simplified (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Main theorem
theorem fraction_simplified (n : ℕ) : is_simplified (21 * n + 4) (14 * n + 3) :=
by
  -- Rest of the proof
  sorry

end fraction_simplified_l14_14191


namespace factor_polynomial_l14_14181

def Polynomial_Factorization (x : ℝ) : Prop := 
  let P := x^2 - 6*x + 9 - 64*x^4
  P = (8*x^2 + x - 3) * (-8*x^2 + x - 3)

theorem factor_polynomial : ∀ x : ℝ, Polynomial_Factorization x :=
by 
  intro x
  unfold Polynomial_Factorization
  sorry

end factor_polynomial_l14_14181


namespace find_x_l14_14215

variable {a b x : ℝ}
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(2 * b) = a^b * x^b)

theorem find_x (h₀ : b ≠ 0) (h₁ : (3 * a)^(2 * b) = a^b * x^b) : x = 9 * a :=
by
  sorry

end find_x_l14_14215


namespace mia_has_110_l14_14827

def darwin_money : ℕ := 45
def mia_money (darwin_money : ℕ) : ℕ := 2 * darwin_money + 20

theorem mia_has_110 :
  mia_money darwin_money = 110 := sorry

end mia_has_110_l14_14827


namespace remainder_of_n_mod_9_eq_5_l14_14728

-- Definitions of the variables and conditions
variables (a b c n : ℕ)

-- The given conditions as assumptions
def conditions : Prop :=
  a + b + c = 63 ∧
  a = c + 22 ∧
  n = 2 * a + 3 * b + 4 * c

-- The proof statement that needs to be proven
theorem remainder_of_n_mod_9_eq_5 (h : conditions a b c n) : n % 9 = 5 := 
  sorry

end remainder_of_n_mod_9_eq_5_l14_14728


namespace train_speed_kmh_l14_14040

variable (length_of_train_meters : ℕ) (time_to_cross_seconds : ℕ)

theorem train_speed_kmh (h1 : length_of_train_meters = 50) (h2 : time_to_cross_seconds = 6) :
  (length_of_train_meters * 3600) / (time_to_cross_seconds * 1000) = 30 :=
by
  sorry

end train_speed_kmh_l14_14040


namespace solve_ordered_pair_l14_14815

theorem solve_ordered_pair : ∃ (x y : ℚ), 3*x - 24*y = 3 ∧ x - 3*y = 4 ∧ x = 29/5 ∧ y = 3/5 := by
  sorry

end solve_ordered_pair_l14_14815


namespace total_donation_l14_14610

theorem total_donation : 2 + 6 + 2 + 8 = 18 := 
by sorry

end total_donation_l14_14610


namespace divisible_by_five_solution_exists_l14_14918

theorem divisible_by_five_solution_exists
  (a b c d : ℤ)
  (h₀ : ∃ k : ℤ, d = 5 * k + d % 5 ∧ d % 5 ≠ 0)
  (h₁ : ∃ n : ℤ, (a * n^3 + b * n^2 + c * n + d) % 5 = 0) :
  ∃ m : ℤ, (a + b * m + c * m^2 + d * m^3) % 5 = 0 := 
sorry

end divisible_by_five_solution_exists_l14_14918


namespace maximum_possible_angle_Z_l14_14025

theorem maximum_possible_angle_Z (X Y Z : ℝ) (h1 : Z ≤ Y) (h2 : Y ≤ X) (h3 : 2 * X = 6 * Z) (h4 : X + Y + Z = 180) : Z = 36 :=
by
  sorry

end maximum_possible_angle_Z_l14_14025


namespace trigonometric_identity_l14_14778

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  1 + Real.sin α * Real.cos α = 7 / 5 :=
by
  sorry

end trigonometric_identity_l14_14778


namespace complex_fraction_value_l14_14492

theorem complex_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 7 / 4 :=
sorry

end complex_fraction_value_l14_14492


namespace line_intersects_circle_l14_14889

variable (x0 y0 R : ℝ)

theorem line_intersects_circle (h : x0^2 + y0^2 > R^2) :
  ∃ (x y : ℝ), (x^2 + y^2 = R^2) ∧ (x0 * x + y0 * y = R^2) :=
sorry

end line_intersects_circle_l14_14889


namespace max_bishops_on_chessboard_l14_14252

theorem max_bishops_on_chessboard : ∃ n : ℕ, n = 14 ∧ (∃ k : ℕ, n * n = k^2) := 
by {
  sorry
}

end max_bishops_on_chessboard_l14_14252


namespace B_is_not_15_percent_less_than_A_l14_14278

noncomputable def A (B : ℝ) : ℝ := 1.15 * B

theorem B_is_not_15_percent_less_than_A (B : ℝ) (h : B > 0) : A B ≠ 0.85 * (A B) :=
by
  unfold A
  suffices 1.15 * B ≠ 0.85 * (1.15 * B) by
    intro h1
    exact this h1
  sorry

end B_is_not_15_percent_less_than_A_l14_14278


namespace more_roses_than_orchids_l14_14184

-- Definitions
def roses_now : Nat := 12
def orchids_now : Nat := 2

-- Theorem statement
theorem more_roses_than_orchids : (roses_now - orchids_now) = 10 := by
  sorry

end more_roses_than_orchids_l14_14184


namespace min_value_of_quadratic_l14_14343

theorem min_value_of_quadratic (x y z : ℝ) 
  (h1 : x + 2 * y - 5 * z = 3)
  (h2 : x - 2 * y - z = -5) : 
  ∃ z' : ℝ,  x = 3 * z' - 1 ∧ y = z' + 2 ∧ (11 * z' * z' - 2 * z' + 5 = (54 : ℝ) / 11) :=
sorry

end min_value_of_quadratic_l14_14343


namespace jake_balloons_bought_l14_14433

theorem jake_balloons_bought (B : ℕ) (h : 6 = (2 + B) + 1) : B = 3 :=
by
  -- proof omitted
  sorry

end jake_balloons_bought_l14_14433


namespace winner_C_l14_14304

noncomputable def votes_A : ℕ := 4500
noncomputable def votes_B : ℕ := 7000
noncomputable def votes_C : ℕ := 12000
noncomputable def votes_D : ℕ := 8500
noncomputable def votes_E : ℕ := 3500

noncomputable def total_votes : ℕ := votes_A + votes_B + votes_C + votes_D + votes_E

noncomputable def percentage (votes : ℕ) : ℚ :=
   (votes : ℚ) / (total_votes : ℚ) * 100

noncomputable def percentage_A : ℚ := percentage votes_A
noncomputable def percentage_B : ℚ := percentage votes_B
noncomputable def percentage_C : ℚ := percentage votes_C
noncomputable def percentage_D : ℚ := percentage votes_D
noncomputable def percentage_E : ℚ := percentage votes_E

theorem winner_C : (percentage_C = 33.803) := 
sorry

end winner_C_l14_14304


namespace five_ones_make_100_l14_14951

noncomputable def concatenate (a b c : Nat) : Nat :=
  a * 100 + b * 10 + c

theorem five_ones_make_100 :
  let one := 1
  let x := concatenate one one one -- 111
  let y := concatenate one one 0 / 10 -- 11, concatenation of 1 and 1 treated as 110, divided by 10
  x - y = 100 :=
by
  sorry

end five_ones_make_100_l14_14951


namespace count_natural_numbers_perfect_square_l14_14021

theorem count_natural_numbers_perfect_square :
  ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ (n1^2 - 19 * n1 + 91) = m^2 ∧ (n2^2 - 19 * n2 + 91) = k^2 ∧
  ∀ n : ℕ, (n^2 - 19 * n + 91) = p^2 → n = n1 ∨ n = n2 := sorry

end count_natural_numbers_perfect_square_l14_14021


namespace max_ballpoint_pens_l14_14804

def ballpoint_pen_cost : ℕ := 10
def gel_pen_cost : ℕ := 30
def fountain_pen_cost : ℕ := 60
def total_pens : ℕ := 20
def total_cost : ℕ := 500

theorem max_ballpoint_pens : ∃ (x y z : ℕ), 
  x + y + z = total_pens ∧ 
  ballpoint_pen_cost * x + gel_pen_cost * y + fountain_pen_cost * z = total_cost ∧ 
  1 ≤ x ∧ 
  1 ≤ y ∧
  1 ≤ z ∧
  ∀ x', ((∃ y' z', x' + y' + z' = total_pens ∧ 
                    ballpoint_pen_cost * x' + gel_pen_cost * y' + fountain_pen_cost * z' = total_cost ∧ 
                    1 ≤ x' ∧ 
                    1 ≤ y' ∧
                    1 ≤ z') → x' ≤ x) :=
  sorry

end max_ballpoint_pens_l14_14804


namespace log_change_of_base_log_change_of_base_with_b_l14_14831

variable {a b x : ℝ}
variable (h₁ : 0 < a ∧ a ≠ 1)
variable (h₂ : 0 < b ∧ b ≠ 1)
variable (h₃ : 0 < x)

theorem log_change_of_base (h₁ : 0 < a ∧ a ≠ 1) (h₂ : 0 < b ∧ b ≠ 1) (h₃ : 0 < x) : 
  Real.log x / Real.log a = Real.log x / Real.log b := by
  sorry

theorem log_change_of_base_with_b (h₁ : 0 < a ∧ a ≠ 1) (h₂ : 0 < b ∧ b ≠ 1) : 
  Real.log b / Real.log a = 1 / Real.log a := by
  sorry

end log_change_of_base_log_change_of_base_with_b_l14_14831


namespace congruence_solution_exists_l14_14463

theorem congruence_solution_exists {p n a : ℕ} (hp : Prime p) (hn : n % p ≠ 0) (ha : a % p ≠ 0)
  (hx : ∃ x : ℕ, x^n % p = a % p) :
  ∀ r : ℕ, ∃ x : ℕ, x^n % (p^(r + 1)) = a % (p^(r + 1)) :=
by
  intros r
  sorry

end congruence_solution_exists_l14_14463


namespace total_pieces_of_chicken_needed_l14_14420

def friedChickenDinnerPieces := 8
def chickenPastaPieces := 2
def barbecueChickenPieces := 4
def grilledChickenSaladPieces := 1

def friedChickenDinners := 4
def chickenPastaOrders := 8
def barbecueChickenOrders := 5
def grilledChickenSaladOrders := 6

def totalChickenPiecesNeeded :=
  (friedChickenDinnerPieces * friedChickenDinners) +
  (chickenPastaPieces * chickenPastaOrders) +
  (barbecueChickenPieces * barbecueChickenOrders) +
  (grilledChickenSaladPieces * grilledChickenSaladOrders)

theorem total_pieces_of_chicken_needed : totalChickenPiecesNeeded = 74 := by
  sorry

end total_pieces_of_chicken_needed_l14_14420


namespace cards_dealt_to_people_l14_14712

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end cards_dealt_to_people_l14_14712


namespace find_original_number_l14_14068

theorem find_original_number (n a b: ℤ) 
  (h1 : n > 1000) 
  (h2 : n + 79 = a^2) 
  (h3 : n + 204 = b^2) 
  (h4 : b^2 - a^2 = 125) : 
  n = 3765 := 
by 
  sorry

end find_original_number_l14_14068


namespace sin_double_angle_values_l14_14285

theorem sin_double_angle_values (α : ℝ) (hα : 0 < α ∧ α < π) (h : 3 * (Real.cos α)^2 = Real.sin ((π / 4) - α)) :
  Real.sin (2 * α) = 1 ∨ Real.sin (2 * α) = -17 / 18 :=
by
  sorry

end sin_double_angle_values_l14_14285


namespace stream_speed_l14_14800

theorem stream_speed (C S : ℝ) 
    (h1 : C - S = 8) 
    (h2 : C + S = 12) : 
    S = 2 :=
sorry

end stream_speed_l14_14800


namespace smallest_four_digit_palindrome_div7_eq_1661_l14_14124

theorem smallest_four_digit_palindrome_div7_eq_1661 :
  ∃ (A B : ℕ), (A == 1 ∨ A == 3 ∨ A == 5 ∨ A == 7 ∨ A == 9) ∧
  (1000 ≤ 1100 * A + 11 * B ∧ 1100 * A + 11 * B < 10000) ∧
  (1100 * A + 11 * B) % 7 = 0 ∧
  (1100 * A + 11 * B) = 1661 :=
by
  sorry

end smallest_four_digit_palindrome_div7_eq_1661_l14_14124


namespace circle_polar_equation_l14_14240

-- Definitions and conditions
def circle_equation_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * y = 0

def polar_coordinates (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem to be proven
theorem circle_polar_equation (ρ θ : ℝ) :
  (∀ x y : ℝ, circle_equation_cartesian x y → 
  polar_coordinates ρ θ x y) → ρ = 2 * Real.sin θ :=
by
  -- This is a placeholder for the proof
  sorry

end circle_polar_equation_l14_14240


namespace sum_a4_a6_l14_14773

variable (a : ℕ → ℝ) (d : ℝ)
variable (h_arith : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
variable (h_sum : a 2 + a 3 + a 7 + a 8 = 8)

theorem sum_a4_a6 : a 4 + a 6 = 4 :=
by
  sorry

end sum_a4_a6_l14_14773


namespace distance_traveled_l14_14746

def velocity (t : ℝ) : ℝ := t^2 + 1

theorem distance_traveled :
  (∫ t in (0:ℝ)..(3:ℝ), velocity t) = 12 :=
by
  simp [velocity]
  sorry

end distance_traveled_l14_14746


namespace stadium_length_in_yards_l14_14976

theorem stadium_length_in_yards (length_in_feet : ℕ) (conversion_factor : ℕ) : ℕ :=
    length_in_feet / conversion_factor

example : stadium_length_in_yards 240 3 = 80 :=
by sorry

end stadium_length_in_yards_l14_14976


namespace determine_digit_I_l14_14425

theorem determine_digit_I (F I V E T H R N : ℕ) (hF : F = 8) (hE_odd : E = 1 ∨ E = 3 ∨ E = 5 ∨ E = 7 ∨ E = 9)
  (h_diff : F ≠ I ∧ F ≠ V ∧ F ≠ E ∧ F ≠ T ∧ F ≠ H ∧ F ≠ R ∧ F ≠ N 
             ∧ I ≠ V ∧ I ≠ E ∧ I ≠ T ∧ I ≠ H ∧ I ≠ R ∧ I ≠ N 
             ∧ V ≠ E ∧ V ≠ T ∧ V ≠ H ∧ V ≠ R ∧ V ≠ N 
             ∧ E ≠ T ∧ E ≠ H ∧ E ≠ R ∧ E ≠ N 
             ∧ T ≠ H ∧ T ≠ R ∧ T ≠ N 
             ∧ H ≠ R ∧ H ≠ N 
             ∧ R ≠ N)
  (h_verify_sum : (10^3 * 8 + 10^2 * I + 10 * V + E) + (10^4 * T + 10^3 * H + 10^2 * R + 11 * E) = 10^3 * N + 10^2 * I + 10 * N + E) :
  I = 4 := 
sorry

end determine_digit_I_l14_14425


namespace union_of_sets_l14_14911

def setA : Set ℝ := { x | -5 ≤ x ∧ x < 1 }
def setB : Set ℝ := { x | x ≤ 2 }

theorem union_of_sets : setA ∪ setB = { x | x ≤ 2 } :=
by sorry

end union_of_sets_l14_14911


namespace find_a_l14_14685

theorem find_a (a : ℝ) :
  (∃! x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0) ↔ a = 1 ∨ a = 5/3 :=
by
  sorry

end find_a_l14_14685


namespace combined_vacations_and_classes_l14_14871

-- Define the conditions
def Kelvin_classes : ℕ := 90
def Grant_vacations : ℕ := 4 * Kelvin_classes

-- The Lean 4 statement proving the combined total of vacations and classes
theorem combined_vacations_and_classes : Kelvin_classes + Grant_vacations = 450 := by
  sorry

end combined_vacations_and_classes_l14_14871


namespace breaststroke_speed_correct_l14_14269

-- Defining the given conditions
def total_distance : ℕ := 500
def front_crawl_speed : ℕ := 45
def front_crawl_time : ℕ := 8
def total_time : ℕ := 12

-- Definition of the breaststroke speed given the conditions
def breaststroke_speed : ℕ :=
  let front_crawl_distance := front_crawl_speed * front_crawl_time
  let breaststroke_distance := total_distance - front_crawl_distance
  let breaststroke_time := total_time - front_crawl_time
  breaststroke_distance / breaststroke_time

-- Theorem to prove the breaststroke speed is 35 yards per minute
theorem breaststroke_speed_correct : breaststroke_speed = 35 :=
  sorry

end breaststroke_speed_correct_l14_14269


namespace sum_of_squares_l14_14531

theorem sum_of_squares (x : ℚ) (h : x + 2 * x + 3 * x = 14) : 
  (x^2 + (2 * x)^2 + (3 * x)^2) = 686 / 9 :=
by
  sorry

end sum_of_squares_l14_14531


namespace failed_in_hindi_percentage_l14_14020

/-- In an examination, a specific percentage of students failed in Hindi (H%), 
45% failed in English, and 20% failed in both. We know that 40% passed in both subjects. 
Prove that 35% students failed in Hindi. --/
theorem failed_in_hindi_percentage : 
  ∀ (H E B P : ℕ),
    (E = 45) → (B = 20) → (P = 40) → (100 - P = H + E - B) → H = 35 := by
  intros H E B P hE hB hP h
  sorry

end failed_in_hindi_percentage_l14_14020


namespace math_problem_l14_14987

noncomputable def f : ℝ → ℝ := sorry

theorem math_problem (h_decreasing : ∀ x y : ℝ, 2 < x → x < y → f y < f x)
  (h_even : ∀ x : ℝ, f (-x + 2) = f (x + 2)) :
  f 2 < f 3 ∧ f 3 < f 0 ∧ f 0 < f (-1) :=
by
  sorry

end math_problem_l14_14987


namespace vents_per_zone_l14_14413

theorem vents_per_zone (total_cost : ℝ) (number_of_zones : ℝ) (cost_per_vent : ℝ) (h_total_cost : total_cost = 20000) (h_zones : number_of_zones = 2) (h_cost_per_vent : cost_per_vent = 2000) : 
  (total_cost / cost_per_vent) / number_of_zones = 5 :=
by 
  sorry

end vents_per_zone_l14_14413


namespace find_number_l14_14844

theorem find_number (x : ℝ) (h : 0.5 * x = 0.25 * x + 2) : x = 8 :=
by
  sorry

end find_number_l14_14844


namespace smallest_a_l14_14371

theorem smallest_a 
  (a : ℤ) (P : ℤ → ℤ) 
  (h_pos : 0 < a) 
  (hP1 : P 1 = a) (hP5 : P 5 = a) (hP7 : P 7 = a) (hP9 : P 9 = a) 
  (hP2 : P 2 = -a) (hP4 : P 4 = -a) (hP6 : P 6 = -a) (hP8 : P 8 = -a) : 
  a ≥ 336 :=
by
  sorry

end smallest_a_l14_14371


namespace number_of_distinct_intersections_l14_14913

theorem number_of_distinct_intersections :
  (∃ x y : ℝ, 9 * x^2 + 16 * y^2 = 16 ∧ 16 * x^2 + 9 * y^2 = 9) →
  (∀ x y₁ y₂ : ℝ, 9 * x^2 + 16 * y₁^2 = 16 ∧ 16 * x^2 + 9 * y₁^2 = 9 ∧
    9 * x^2 + 16 * y₂^2 = 16 ∧ 16 * x^2 + 9 * y₂^2 = 9 → y₁ = y₂) →
  (∃! p : ℝ × ℝ, 9 * p.1^2 + 16 * p.2^2 = 16 ∧ 16 * p.1^2 + 9 * p.2^2 = 9) :=
by
  sorry

end number_of_distinct_intersections_l14_14913


namespace answer_is_p_and_q_l14_14554

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l14_14554


namespace range_of_x_l14_14018

theorem range_of_x (a b x : ℝ) (h : a ≠ 0) 
  (ineq : |a + b| + |a - b| ≥ |a| * |x - 2|) : 
  0 ≤ x ∧ x ≤ 4 :=
  sorry

end range_of_x_l14_14018


namespace intersection_singleton_one_l14_14078

-- Define sets A and B according to the given conditions
def setA : Set ℤ := { x | 0 < x ∧ x < 4 }
def setB : Set ℤ := { x | (x+1)*(x-2) < 0 }

-- Statement to prove A ∩ B = {1}
theorem intersection_singleton_one : setA ∩ setB = {1} :=
by 
  sorry

end intersection_singleton_one_l14_14078


namespace manufacturer_price_l14_14415

theorem manufacturer_price :
  ∃ M : ℝ, 
    (∃ R : ℝ, 
      R = 1.15 * M ∧
      ∃ D : ℝ, 
        D = 0.85 * R ∧
        R - D = 57.5) ∧
    M = 333.33 := 
by
  sorry

end manufacturer_price_l14_14415


namespace cistern_total_wet_surface_area_l14_14559

/-- Given a cistern with length 6 meters, width 4 meters, and water depth 1.25 meters,
    the total area of the wet surface is 49 square meters. -/
theorem cistern_total_wet_surface_area
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (h_length : length = 6) (h_width : width = 4) (h_depth : depth = 1.25) :
  (length * width) + 2 * (length * depth) + 2 * (width * depth) = 49 :=
by {
  -- Proof goes here
  sorry
}

end cistern_total_wet_surface_area_l14_14559


namespace root_interval_l14_14211

def f (x : ℝ) : ℝ := 5 * x - 7

theorem root_interval : ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- Proof steps should be here
  sorry

end root_interval_l14_14211


namespace yield_percentage_of_stock_l14_14926

noncomputable def annual_dividend (par_value : ℝ) : ℝ := 0.21 * par_value
noncomputable def market_price : ℝ := 210
noncomputable def yield_percentage (annual_dividend : ℝ) (market_price : ℝ) : ℝ :=
  (annual_dividend / market_price) * 100

theorem yield_percentage_of_stock (par_value : ℝ)
  (h_par_value : par_value = 100) :
  yield_percentage (annual_dividend par_value) market_price = 10 :=
by
  sorry

end yield_percentage_of_stock_l14_14926


namespace ratio_of_areas_inequality_l14_14183

theorem ratio_of_areas_inequality (a x m : ℝ) (h1 : a > 0) (h2 : x > 0) (h3 : x < a) :
  m = (3 * x^2 - 3 * a * x + a^2) / a^2 →
  (1 / 4 ≤ m ∧ m < 1) :=
sorry

end ratio_of_areas_inequality_l14_14183


namespace initial_number_of_observations_l14_14781

theorem initial_number_of_observations (n : ℕ) 
  (initial_mean : ℝ := 100) 
  (wrong_obs : ℝ := 75) 
  (corrected_obs : ℝ := 50) 
  (corrected_mean : ℝ := 99.075) 
  (h1 : (n:ℝ) * initial_mean = n * corrected_mean + wrong_obs - corrected_obs) 
  (h2 : n = (25 : ℝ) / 0.925) 
  : n = 27 := 
sorry

end initial_number_of_observations_l14_14781


namespace two_a_sq_minus_six_b_plus_one_l14_14702

theorem two_a_sq_minus_six_b_plus_one (a b : ℝ) (h : a^2 - 3 * b = 5) : 2 * a^2 - 6 * b + 1 = 11 := by
  sorry

end two_a_sq_minus_six_b_plus_one_l14_14702


namespace smallest_side_is_10_l14_14071

noncomputable def smallest_side_of_triangle (x : ℝ) : ℝ :=
    let side1 := 10
    let side2 := 3 * x + 6
    let side3 := x + 5
    min side1 (min side2 side3)

theorem smallest_side_is_10 (x : ℝ) (h : 10 + (3 * x + 6) + (x + 5) = 60) : 
    smallest_side_of_triangle x = 10 :=
by
    sorry

end smallest_side_is_10_l14_14071


namespace number_drawn_from_3rd_group_l14_14653

theorem number_drawn_from_3rd_group {n k : ℕ} (pop_size : ℕ) (sample_size : ℕ) 
  (drawn_from_group : ℕ → ℕ) (group_id : ℕ) (num_in_13th_group : ℕ) : 
  pop_size = 160 → 
  sample_size = 20 → 
  (∀ i, 1 ≤ i ∧ i ≤ sample_size → ∃ j, group_id = i ∧ 
    (j = (i - 1) * (pop_size / sample_size) + drawn_from_group 1)) → 
  num_in_13th_group = 101 → 
  drawn_from_group 3 = 21 := 
by
  intros hp hs hg h13
  sorry

end number_drawn_from_3rd_group_l14_14653


namespace figure_100_squares_l14_14055

def f (n : ℕ) : ℕ := n^3 + 2 * n^2 + 2 * n + 1

theorem figure_100_squares : f 100 = 1020201 :=
by
  -- The proof will go here
  sorry

end figure_100_squares_l14_14055


namespace matrix_B_pow48_l14_14691

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 0, 2], ![0, -2, 0]]

theorem matrix_B_pow48 :
  B ^ 48 = ![![0, 0, 0], ![0, 16^12, 0], ![0, 0, 16^12]] :=
by sorry

end matrix_B_pow48_l14_14691


namespace number_of_ways_to_represent_5030_l14_14111

theorem number_of_ways_to_represent_5030 :
  let even := {x : ℕ | x % 2 = 0}
  let in_range := {x : ℕ | x ≤ 98}
  let valid_b := even ∩ in_range
  ∃ (M : ℕ), M = 150 ∧ ∀ (b3 b2 b1 b0 : ℕ), 
    b3 ∈ valid_b ∧ b2 ∈ valid_b ∧ b1 ∈ valid_b ∧ b0 ∈ valid_b →
    5030 = b3 * 10 ^ 3 + b2 * 10 ^ 2 + b1 * 10 + b0 → 
    M = 150 :=
  sorry

end number_of_ways_to_represent_5030_l14_14111


namespace total_items_children_carry_l14_14623

theorem total_items_children_carry 
  (pieces_per_pizza : ℕ) (number_of_fourthgraders : ℕ) (pizza_per_fourthgrader : ℕ) 
  (pepperoni_per_pizza : ℕ) (mushrooms_per_pizza : ℕ) (olives_per_pizza : ℕ) 
  (total_pizzas : ℕ) (total_pieces_of_pizza : ℕ) (total_pepperoni : ℕ) (total_mushrooms : ℕ) 
  (total_olives : ℕ) (total_toppings : ℕ) (total_items : ℕ) : 
  pieces_per_pizza = 6 →
  number_of_fourthgraders = 10 →
  pizza_per_fourthgrader = 20 →
  pepperoni_per_pizza = 5 →
  mushrooms_per_pizza = 3 →
  olives_per_pizza = 8 →
  total_pizzas = number_of_fourthgraders * pizza_per_fourthgrader →
  total_pieces_of_pizza = total_pizzas * pieces_per_pizza →
  total_pepperoni = total_pizzas * pepperoni_per_pizza →
  total_mushrooms = total_pizzas * mushrooms_per_pizza →
  total_olives = total_pizzas * olives_per_pizza →
  total_toppings = total_pepperoni + total_mushrooms + total_olives →
  total_items = total_pieces_of_pizza + total_toppings →
  total_items = 4400 :=
by
  sorry

end total_items_children_carry_l14_14623


namespace algebraic_expression_value_l14_14934

theorem algebraic_expression_value (x y : ℝ) (h1 : x + 2 * y = 4) (h2 : x - 2 * y = -1) :
  x^2 - 4 * y^2 + 1 = -3 := by
  sorry

end algebraic_expression_value_l14_14934


namespace candy_problem_l14_14834

variable (total_pieces_eaten : ℕ) (pieces_from_sister : ℕ) (pieces_from_neighbors : ℕ)

theorem candy_problem
  (h1 : total_pieces_eaten = 18)
  (h2 : pieces_from_sister = 13)
  (h3 : total_pieces_eaten = pieces_from_sister + pieces_from_neighbors) :
  pieces_from_neighbors = 5 := by
  -- Add proof here
  sorry

end candy_problem_l14_14834


namespace investment_three_years_ago_l14_14277

noncomputable def initial_investment (final_amount : ℝ) : ℝ :=
  final_amount / (1.08 ^ 3)

theorem investment_three_years_ago :
  abs (initial_investment 439.23 - 348.68) < 0.01 :=
by
  sorry

end investment_three_years_ago_l14_14277


namespace trip_time_is_approximate_l14_14687

noncomputable def total_distance : ℝ := 620
noncomputable def half_distance : ℝ := total_distance / 2
noncomputable def speed1 : ℝ := 70
noncomputable def speed2 : ℝ := 85
noncomputable def time1 : ℝ := half_distance / speed1
noncomputable def time2 : ℝ := half_distance / speed2
noncomputable def total_time : ℝ := time1 + time2

theorem trip_time_is_approximate :
  abs (total_time - 8.0757) < 0.0001 :=
sorry

end trip_time_is_approximate_l14_14687


namespace helga_shoes_l14_14658

theorem helga_shoes :
  ∃ (S : ℕ), 7 + S + 0 + 2 * (7 + S) = 48 ∧ (S - 7 = 2) :=
by
  sorry

end helga_shoes_l14_14658


namespace infinite_solutions_2n_3n_square_n_multiple_of_40_infinite_solutions_general_l14_14771

open Nat

theorem infinite_solutions_2n_3n_square :
  ∃ᶠ n : ℤ in at_top, ∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2 :=
sorry

theorem n_multiple_of_40 :
  ∀ n : ℤ, (∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2) → (40 ∣ n) :=
sorry

theorem infinite_solutions_general (m : ℕ) (hm : 0 < m) :
  ∃ᶠ n : ℤ in at_top, ∃ a b : ℤ, m * n + 1 = a^2 ∧ (m + 1) * n + 1 = b^2 :=
sorry

end infinite_solutions_2n_3n_square_n_multiple_of_40_infinite_solutions_general_l14_14771


namespace m_cubed_plus_m_inv_cubed_l14_14037

theorem m_cubed_plus_m_inv_cubed (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 1 = 971 :=
sorry

end m_cubed_plus_m_inv_cubed_l14_14037


namespace minimum_k_value_l14_14536

theorem minimum_k_value (a b k : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∀ a b, (1 / a + 1 / b + k / (a + b)) ≥ 0) : k ≥ -4 :=
sorry

end minimum_k_value_l14_14536


namespace radius_of_smaller_circle_l14_14542

open Real

-- Definitions based on the problem conditions
def large_circle_radius : ℝ := 10
def pattern := "square"

-- Statement of the problem in Lean 4
theorem radius_of_smaller_circle :
  ∀ (r : ℝ), (large_circle_radius = 10) → (pattern = "square") → r = 5 * sqrt 2 →  ∃ r, r = 5 * sqrt 2 :=
by
  sorry

end radius_of_smaller_circle_l14_14542


namespace number_of_winning_scores_l14_14004

-- Define the problem conditions
variable (n : ℕ) (team1_scores team2_scores : Finset ℕ)

-- Define the total number of runners
def total_runners := 12

-- Define the sum of placements
def sum_placements : ℕ := (total_runners * (total_runners + 1)) / 2

-- Define the threshold for the winning score
def winning_threshold : ℕ := sum_placements / 2

-- Define the minimum score for a team
def min_score : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Prove that the number of different possible winning scores is 19
theorem number_of_winning_scores : 
  Finset.card (Finset.range (winning_threshold + 1) \ Finset.range min_score) = 19 :=
by
  sorry -- Proof to be filled in

end number_of_winning_scores_l14_14004


namespace upward_shift_of_parabola_l14_14435

variable (k : ℝ) -- Define k as a real number representing the vertical shift

def original_function (x : ℝ) : ℝ := -x^2 -- Define the original function

def shifted_function (x : ℝ) : ℝ := original_function x + 2 -- Define the shifted function by 2 units upwards

theorem upward_shift_of_parabola (x : ℝ) : shifted_function x = -x^2 + k :=
by
  sorry

end upward_shift_of_parabola_l14_14435


namespace find_number_l14_14689

theorem find_number (N : ℝ) (h : (0.47 * N - 0.36 * 1412) + 66 = 6) : N = 953.87 :=
  sorry

end find_number_l14_14689


namespace inequality_m_le_minus3_l14_14502

theorem inequality_m_le_minus3 (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x^2 - 4 * x ≥ m) → m ≤ -3 :=
by
  sorry

end inequality_m_le_minus3_l14_14502


namespace find_x_l14_14458

theorem find_x
  (x : ℝ)
  (h : 5^29 * x^15 = 2 * 10^29) :
  x = 4 :=
by
  sorry

end find_x_l14_14458


namespace problem_a_problem_c_problem_d_l14_14198

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l14_14198


namespace sum_of_x_and_y_l14_14726

theorem sum_of_x_and_y (x y : ℕ) (hxpos : 0 < x) (hypos : 1 < y) (hxy : x^y < 500) (hmax : ∀ (a b : ℕ), 0 < a → 1 < b → a^b < 500 → a^b ≤ x^y) : x + y = 24 := 
sorry

end sum_of_x_and_y_l14_14726


namespace powderman_distance_when_hears_explosion_l14_14515

noncomputable def powderman_speed_yd_per_s : ℝ := 10
noncomputable def blast_time_s : ℝ := 45
noncomputable def sound_speed_ft_per_s : ℝ := 1080
noncomputable def powderman_speed_ft_per_s : ℝ := 30

noncomputable def distance_powderman (t : ℝ) : ℝ := powderman_speed_ft_per_s * t
noncomputable def distance_sound (t : ℝ) : ℝ := sound_speed_ft_per_s * (t - blast_time_s)

theorem powderman_distance_when_hears_explosion :
  ∃ t, t > blast_time_s ∧ distance_powderman t = distance_sound t ∧ (distance_powderman t) / 3 = 463 :=
sorry

end powderman_distance_when_hears_explosion_l14_14515


namespace milk_production_l14_14033

theorem milk_production (a b c d e f : ℕ) (h₁ : a > 0) (h₂ : c > 0) (h₃ : f > 0) : 
  ((d * e * b * f) / (100 * a * c)) = (d * e * b * f / (100 * a * c)) :=
by
  sorry

end milk_production_l14_14033


namespace geometric_sequence_a8_l14_14148

theorem geometric_sequence_a8 {a : ℕ → ℝ} (h1 : a 1 * a 3 = 4) (h9 : a 9 = 256) :
  a 8 = 128 ∨ a 8 = -128 :=
sorry

end geometric_sequence_a8_l14_14148


namespace problem_statement_l14_14787

def op (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem problem_statement : ((op 7 4) - 12) * 5 = 105 := by
  sorry

end problem_statement_l14_14787


namespace page_numbers_sum_l14_14768

theorem page_numbers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 136080) : n + (n + 1) + (n + 2) = 144 :=
by
  sorry

end page_numbers_sum_l14_14768


namespace tangent_line_at_x1_f_nonnegative_iff_l14_14806

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (x-1) * Real.log x - m * (x+1)

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + (x-1) / x - m

theorem tangent_line_at_x1 (m : ℝ) (h : m = 1) :
  ∀ x y : ℝ, f x 1 = y → (x = 1) → x + y + 1 = 0 :=
sorry

theorem f_nonnegative_iff (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) ↔ m ≤ 0 :=
sorry

end tangent_line_at_x1_f_nonnegative_iff_l14_14806


namespace hat_price_after_discounts_l14_14498

-- Defining initial conditions
def initial_price : ℝ := 15
def first_discount_percent : ℝ := 0.25
def second_discount_percent : ℝ := 0.50

-- Defining the expected final price after applying both discounts
def expected_final_price : ℝ := 5.625

-- Lean statement to prove the final price after both discounts is as expected
theorem hat_price_after_discounts : 
  let first_reduced_price := initial_price * (1 - first_discount_percent)
  let second_reduced_price := first_reduced_price * (1 - second_discount_percent)
  second_reduced_price = expected_final_price := sorry

end hat_price_after_discounts_l14_14498


namespace sum_of_areas_of_triangles_in_cube_l14_14852

theorem sum_of_areas_of_triangles_in_cube : 
  let m := 48
  let n := 4608
  let p := 576
  m + n + p = 5232 := by 
    sorry

end sum_of_areas_of_triangles_in_cube_l14_14852


namespace hyperbola_eccentricity_l14_14719

-- Define the hyperbola and the condition of the asymptote passing through (2,1)
def hyperbola (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧
               (a ≠ 0 ∧ b ≠ 0) ∧
               (x, y) = (2, 1)

-- Define the eccentricity of the hyperbola
def eccentricity (a b e : ℝ) : Prop :=
  a^2 + b^2 = (b * e)^2

theorem hyperbola_eccentricity (a b e : ℝ) 
  (hx : hyperbola a b)
  (ha : a = 2 * b)
  (ggt: (a^2 = 4 * b^2)) :
  eccentricity a b e → e = (Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_eccentricity_l14_14719


namespace max_value_on_interval_l14_14617

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_on_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f x ≤ 5 :=
by
  sorry

end max_value_on_interval_l14_14617


namespace correct_exp_operation_l14_14505

theorem correct_exp_operation (a : ℝ) : (a^2 * a = a^3) := 
by
  -- Leave the proof as an exercise
  sorry

end correct_exp_operation_l14_14505


namespace division_of_sums_and_products_l14_14605

theorem division_of_sums_and_products (a b c : ℕ) (h_a : a = 7) (h_b : b = 5) (h_c : c = 3) :
  (a^3 + b^3 + c^3) / (a^2 - a * b + b^2 - b * c + c^2) = 15 := by
  -- proofs go here
  sorry

end division_of_sums_and_products_l14_14605


namespace trapezium_height_l14_14711

-- Defining the lengths of the parallel sides and the area of the trapezium
def a : ℝ := 28
def b : ℝ := 18
def area : ℝ := 345

-- Defining the distance between the parallel sides to be proven
def h : ℝ := 15

-- The theorem that proves the distance between the parallel sides
theorem trapezium_height :
  (1 / 2) * (a + b) * h = area :=
by
  sorry

end trapezium_height_l14_14711


namespace bacon_suggestion_l14_14881

theorem bacon_suggestion (x y : ℕ) (h1 : x = 479) (h2 : y = x + 10) : y = 489 := 
by {
  sorry
}

end bacon_suggestion_l14_14881


namespace hyperbola_equiv_l14_14870

-- The existing hyperbola
def hyperbola1 (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- The new hyperbola with same asymptotes passing through (2, 2) should have this form
def hyperbola2 (x y : ℝ) : Prop := (x^2 / 3 - y^2 / 12 = 1)

theorem hyperbola_equiv (x y : ℝ) :
  (hyperbola1 2 2) →
  (y^2 / 4 - x^2 / 4 = -3) →
  (hyperbola2 x y) :=
by
  intros h1 h2
  sorry

end hyperbola_equiv_l14_14870


namespace deck_width_l14_14355

theorem deck_width (w : ℝ) : 
  (10 + 2 * w) * (12 + 2 * w) = 360 → w = 4 := 
by 
  sorry

end deck_width_l14_14355


namespace ninth_term_arith_seq_l14_14583

theorem ninth_term_arith_seq (a d : ℤ) (h1 : a + 2 * d = 25) (h2 : a + 5 * d = 31) : a + 8 * d = 37 :=
sorry

end ninth_term_arith_seq_l14_14583


namespace largest_square_area_with_4_interior_lattice_points_l14_14219

/-- 
A point (x, y) in the plane is called a lattice point if both x and y are integers.
The largest square that contains exactly four lattice points solely in its interior
has an area of 9.
-/
theorem largest_square_area_with_4_interior_lattice_points : 
  ∃ s : ℝ, ∀ (x y : ℤ), 
  (1 ≤ x ∧ x < s ∧ 1 ≤ y ∧ y < s) → s^2 = 9 := 
sorry

end largest_square_area_with_4_interior_lattice_points_l14_14219


namespace nat_representation_l14_14161

theorem nat_representation (k : ℕ) : ∃ n r : ℕ, (r = 0 ∨ r = 1 ∨ r = 2) ∧ k = 3 * n + r :=
by
  sorry

end nat_representation_l14_14161


namespace find_range_g_l14_14123

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + abs x

theorem find_range_g :
  {x : ℝ | g (2 * x - 1) < g 3} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end find_range_g_l14_14123


namespace sum_nine_terms_of_arithmetic_sequence_l14_14628

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2

theorem sum_nine_terms_of_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_of_first_n_terms a S)
  (h3 : a 5 = 7) :
  S 9 = 63 := by
  sorry

end sum_nine_terms_of_arithmetic_sequence_l14_14628


namespace product_of_coordinates_of_D_l14_14915

theorem product_of_coordinates_of_D (Mx My Cx Cy Dx Dy : ℝ) (M : (Mx, My) = (4, 8)) (C : (Cx, Cy) = (5, 4)) 
  (midpoint : (Mx, My) = ((Cx + Dx) / 2, (Cy + Dy) / 2)) : (Dx * Dy) = 36 := 
by
  sorry

end product_of_coordinates_of_D_l14_14915


namespace common_ratio_geometric_sequence_l14_14983

variables {a : ℕ → ℝ} -- 'a' is a sequence of positive real numbers
variable {q : ℝ} -- 'q' is the common ratio of the geometric sequence

-- Definition of a geometric sequence with common ratio 'q'
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Condition from the problem statement
def condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  2 * a 5 - 3 * a 4 = 2 * a 3

-- Main theorem: If the sequence {a_n} is a geometric sequence with positive terms and satisfies the condition, 
-- then the common ratio q = 2
theorem common_ratio_geometric_sequence :
  (∀ n, 0 < a n) → geometric_sequence a q → condition a q → q = 2 :=
by
  intro h_pos h_geom h_cond
  sorry

end common_ratio_geometric_sequence_l14_14983


namespace ferry_speed_difference_l14_14493

variable (v_P v_Q d_P d_Q t_P t_Q x : ℝ)

-- Defining the constants and conditions provided in the problem
axiom h1 : v_P = 8 
axiom h2 : t_P = 2 
axiom h3 : d_P = t_P * v_P 
axiom h4 : d_Q = 3 * d_P 
axiom h5 : t_Q = t_P + 2
axiom h6 : d_Q = v_Q * t_Q 
axiom h7 : x = v_Q - v_P 

-- The theorem that corresponds to the solution
theorem ferry_speed_difference : x = 4 := by
  sorry

end ferry_speed_difference_l14_14493


namespace max_value_of_f_l14_14384

noncomputable def f (x : ℝ) : ℝ := 3^x - 9^x

theorem max_value_of_f : ∃ x : ℝ, f x = 1 / 4 := sorry

end max_value_of_f_l14_14384


namespace sum_diff_l14_14235

-- Define the lengths of the ropes
def shortest_rope_length := 80
def ratio_shortest := 4
def ratio_middle := 5
def ratio_longest := 6

-- Use the given ratio to find the common multiple x.
def x := shortest_rope_length / ratio_shortest

-- Find the lengths of the other ropes
def middle_rope_length := ratio_middle * x
def longest_rope_length := ratio_longest * x

-- Define the sum of the longest and shortest ropes
def sum_of_longest_and_shortest := longest_rope_length + shortest_rope_length

-- Define the difference between the sum of the longest and shortest rope and the middle rope
def difference := sum_of_longest_and_shortest - middle_rope_length

-- Theorem statement
theorem sum_diff : difference = 100 := by
  sorry

end sum_diff_l14_14235


namespace impossible_transformation_l14_14677

variable (G : Type) [Group G]

/-- Initial word represented by 2003 'a's followed by 'b' --/
def initial_word := "aaa...ab"

/-- Transformed word represented by 'b' followed by 2003 'a's --/
def transformed_word := "baaa...a"

/-- Hypothetical group relations derived from transformations --/
axiom aba_to_b (a b : G) : (a * b * a = b)
axiom bba_to_a (a b : G) : (b * b * a = a)

/-- Impossible transformation proof --/
theorem impossible_transformation (a b : G) : 
  (initial_word = transformed_word) → False := by
  sorry

end impossible_transformation_l14_14677


namespace find_k_l14_14767

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 + 2 * x + 1) / (k * x - 1)

theorem find_k (k : ℝ) : (∀ x : ℝ, f k (f k x) = x) ↔ k = -2 :=
  sorry

end find_k_l14_14767


namespace travelers_on_liner_l14_14377

theorem travelers_on_liner (a : ℕ) : 
  250 ≤ a ∧ a ≤ 400 ∧ a % 15 = 7 ∧ a % 25 = 17 → a = 292 ∨ a = 367 :=
by
  sorry

end travelers_on_liner_l14_14377


namespace final_people_amount_l14_14826

def initial_people : ℕ := 250
def people_left1 : ℕ := 35
def people_joined1 : ℕ := 20
def percentage_left : ℕ := 10
def groups_joined : ℕ := 4
def group_size : ℕ := 15

theorem final_people_amount :
  let intermediate_people1 := initial_people - people_left1;
  let intermediate_people2 := intermediate_people1 + people_joined1;
  let people_left2 := (intermediate_people2 * percentage_left) / 100;
  let rounded_people_left2 := people_left2;
  let intermediate_people3 := intermediate_people2 - rounded_people_left2;
  let total_new_join := groups_joined * group_size;
  let final_people := intermediate_people3 + total_new_join;
  final_people = 272 :=
by sorry

end final_people_amount_l14_14826


namespace probability_of_two_non_defective_pens_l14_14478

-- Definitions for conditions from the problem
def total_pens : ℕ := 16
def defective_pens : ℕ := 3
def selected_pens : ℕ := 2
def non_defective_pens : ℕ := total_pens - defective_pens

-- Function to calculate probability of drawing non-defective pens
noncomputable def probability_no_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  (non_defective_pens / total_pens) * ((non_defective_pens - 1) / (total_pens - 1))

-- Theorem stating the correct answer
theorem probability_of_two_non_defective_pens : 
  probability_no_defective total_pens defective_pens selected_pens = 13 / 20 :=
by
  sorry

end probability_of_two_non_defective_pens_l14_14478


namespace MrsHiltReadTotalChapters_l14_14118

-- Define the number of books and chapters per book
def numberOfBooks : ℕ := 4
def chaptersPerBook : ℕ := 17

-- Define the total number of chapters Mrs. Hilt read
def totalChapters (books : ℕ) (chapters : ℕ) : ℕ := books * chapters

-- The main statement to be proved
theorem MrsHiltReadTotalChapters : totalChapters numberOfBooks chaptersPerBook = 68 := by
  sorry

end MrsHiltReadTotalChapters_l14_14118


namespace number_of_cut_red_orchids_l14_14679

variable (initial_red_orchids added_red_orchids final_red_orchids : ℕ)

-- Conditions
def initial_red_orchids_in_vase (initial_red_orchids : ℕ) : Prop :=
  initial_red_orchids = 9

def final_red_orchids_in_vase (final_red_orchids : ℕ) : Prop :=
  final_red_orchids = 15

-- Proof statement
theorem number_of_cut_red_orchids (initial_red_orchids added_red_orchids final_red_orchids : ℕ)
  (h1 : initial_red_orchids_in_vase initial_red_orchids) 
  (h2 : final_red_orchids_in_vase final_red_orchids) :
  final_red_orchids = initial_red_orchids + added_red_orchids → added_red_orchids = 6 := by
  simp [initial_red_orchids_in_vase, final_red_orchids_in_vase] at *
  sorry

end number_of_cut_red_orchids_l14_14679


namespace range_of_a_l14_14373

-- Define the function f(x) = x^2 - 3x
def f (x : ℝ) : ℝ := x^2 - 3 * x

-- Define the interval as a closed interval from -1 to 1
def interval : Set ℝ := Set.Icc (-1) (1)

-- State the main proposition
theorem range_of_a (a : ℝ) :
  (∃ x ∈ interval, -x^2 + 3 * x + a > 0) ↔ a > -2 :=
by
  sorry

end range_of_a_l14_14373


namespace calculate_expression_l14_14388

variable (x : ℝ)

theorem calculate_expression : (1/2 * x^3)^2 = 1/4 * x^6 := 
by 
  sorry

end calculate_expression_l14_14388


namespace total_miles_walked_l14_14168

def weekly_group_walk_miles : ℕ := 3 * 6

def Jamie_additional_walk_miles_per_week : ℕ := 2 * 6
def Sue_additional_walk_miles_per_week : ℕ := 1 * 6 -- half of Jamie's additional walk
def Laura_additional_walk_miles_per_week : ℕ := 1 * 3 -- 1 mile every two days for 6 days
def Melissa_additional_walk_miles_per_week : ℕ := 2 * 2 -- 2 miles every three days for 6 days
def Katie_additional_walk_miles_per_week : ℕ := 1 * 6

def Jamie_weekly_miles : ℕ := weekly_group_walk_miles + Jamie_additional_walk_miles_per_week
def Sue_weekly_miles : ℕ := weekly_group_walk_miles + Sue_additional_walk_miles_per_week
def Laura_weekly_miles : ℕ := weekly_group_walk_miles + Laura_additional_walk_miles_per_week
def Melissa_weekly_miles : ℕ := weekly_group_walk_miles + Melissa_additional_walk_miles_per_week
def Katie_weekly_miles : ℕ := weekly_group_walk_miles + Katie_additional_walk_miles_per_week

def weeks_in_month : ℕ := 4

def Jamie_monthly_miles : ℕ := Jamie_weekly_miles * weeks_in_month
def Sue_monthly_miles : ℕ := Sue_weekly_miles * weeks_in_month
def Laura_monthly_miles : ℕ := Laura_weekly_miles * weeks_in_month
def Melissa_monthly_miles : ℕ := Melissa_weekly_miles * weeks_in_month
def Katie_monthly_miles : ℕ := Katie_weekly_miles * weeks_in_month

def total_monthly_miles : ℕ :=
  Jamie_monthly_miles + Sue_monthly_miles + Laura_monthly_miles + Melissa_monthly_miles + Katie_monthly_miles

theorem total_miles_walked (month_has_30_days : Prop) : total_monthly_miles = 484 :=
by
  unfold total_monthly_miles
  unfold Jamie_monthly_miles Sue_monthly_miles Laura_monthly_miles Melissa_monthly_miles Katie_monthly_miles
  unfold Jamie_weekly_miles Sue_weekly_miles Laura_weekly_miles Melissa_weekly_miles Katie_weekly_miles
  unfold weekly_group_walk_miles Jamie_additional_walk_miles_per_week Sue_additional_walk_miles_per_week Laura_additional_walk_miles_per_week Melissa_additional_walk_miles_per_week Katie_additional_walk_miles_per_week
  unfold weeks_in_month
  sorry

end total_miles_walked_l14_14168


namespace num_five_ruble_coins_l14_14162

def total_coins := 25
def c1 := 25 - 16
def c2 := 25 - 19
def c10 := 25 - 20

theorem num_five_ruble_coins : (total_coins - (c1 + c2 + c10)) = 5 := by
  sorry

end num_five_ruble_coins_l14_14162


namespace find_m_values_l14_14094

theorem find_m_values (m : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (2, 2) ∧ B = (m, 0) ∧ 
   ∃ r R : ℝ, r = 1 ∧ R = 3 ∧ 
   ∃ d : ℝ, d = abs (dist A B) ∧ d = (R + r)) →
  (m = 2 - 2 * Real.sqrt 3 ∨ m = 2 + 2 * Real.sqrt 3) := 
sorry

end find_m_values_l14_14094


namespace number_of_white_balls_l14_14428

theorem number_of_white_balls (a : ℕ) (h1 : 3 + a ≠ 0) (h2 : (3 : ℚ) / (3 + a) = 3 / 7) : a = 4 :=
sorry

end number_of_white_balls_l14_14428


namespace min_ab_l14_14946

theorem min_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1 / a + 1 / b = 1) : ab = 4 :=
  sorry

end min_ab_l14_14946


namespace maximal_length_sequence_l14_14861

theorem maximal_length_sequence :
  ∃ (a : ℕ → ℤ) (n : ℕ), (∀ i, 1 ≤ i → i + 6 ≤ n → (a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) + a (i + 6) > 0)) ∧ 
                          (∀ j, 1 ≤ j → j + 10 ≤ n → (a j + a (j + 1) + a (j + 2) + a (j + 3) + a (j + 4) + a (j + 5) + a (j + 6) + a (j + 7) + a (j + 8) + a (j + 9) + a (j + 10) < 0)) ∧ 
                          n = 16 :=
sorry

end maximal_length_sequence_l14_14861


namespace prime_solution_exists_l14_14337

theorem prime_solution_exists (p : ℕ) (hp : Nat.Prime p) : ∃ x y z : ℤ, x^2 + y^2 + (p:ℤ) * z = 2003 := 
by 
  sorry

end prime_solution_exists_l14_14337


namespace height_of_box_l14_14792

-- Define box dimensions
def box_length := 6
def box_width := 6

-- Define spherical radii
def radius_large := 3
def radius_small := 2

-- Define coordinates
def box_volume (h : ℝ) : Prop :=
  ∃ (z : ℝ), z = 2 + Real.sqrt 23 ∧ 
  z + radius_large = h

theorem height_of_box (h : ℝ) : box_volume h ↔ h = 5 + Real.sqrt 23 := by
  sorry

end height_of_box_l14_14792


namespace parallel_lines_implies_slope_l14_14396

theorem parallel_lines_implies_slope (a : ℝ) :
  (∀ (x y: ℝ), ax + 2 * y = 0) ∧ (∀ (x y: ℝ), x + y = 1) → (a = 2) :=
by
  sorry

end parallel_lines_implies_slope_l14_14396


namespace find_min_value_l14_14027

theorem find_min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : a + b = 2) : 
  (1 / (2 * a)) + (2 / (b - 1)) ≥ 9 / 2 :=
by
  sorry

end find_min_value_l14_14027


namespace remainder_of_sums_modulo_l14_14143

theorem remainder_of_sums_modulo :
  (2 * (8735 + 8736 + 8737 + 8738 + 8739)) % 11 = 8 :=
by
  sorry

end remainder_of_sums_modulo_l14_14143


namespace Mario_savings_percentage_l14_14992

theorem Mario_savings_percentage 
  (P : ℝ) -- Normal price of a single ticket 
  (h_campaign : 5 * P = 3 * P) -- Campaign condition: 5 tickets for the price of 3
  : (2 * P) / (5 * P) * 100 = 40 := 
by
  -- Below this, we would write the actual automated proof, but we leave it as sorry.
  sorry

end Mario_savings_percentage_l14_14992


namespace mitchell_pencils_l14_14876

/-- Mitchell and Antonio have a combined total of 54 pencils.
Mitchell has 6 more pencils than Antonio. -/
theorem mitchell_pencils (A M : ℕ) 
  (h1 : M = A + 6)
  (h2 : M + A = 54) : M = 30 :=
by
  sorry

end mitchell_pencils_l14_14876


namespace enter_exit_ways_correct_l14_14541

-- Defining the problem conditions
def num_entrances := 4

-- Defining the problem question and answer
def enter_exit_ways (n : Nat) : Nat := n * (n - 1)

-- Statement: Prove the number of different ways to enter and exit is 12
theorem enter_exit_ways_correct : enter_exit_ways num_entrances = 12 := by
  -- Proof
  sorry

end enter_exit_ways_correct_l14_14541


namespace reciprocal_of_neg_three_l14_14540

theorem reciprocal_of_neg_three : (1:ℝ) / (-3:ℝ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg_three_l14_14540


namespace freddy_total_call_cost_l14_14147

def lm : ℕ := 45
def im : ℕ := 31
def lc : ℝ := 0.05
def ic : ℝ := 0.25

theorem freddy_total_call_cost : lm * lc + im * ic = 10.00 := by
  sorry

end freddy_total_call_cost_l14_14147


namespace isosceles_triangle_perimeter_l14_14762

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangle_perimeter :
  ∃ (a b c : ℕ), is_isosceles a b c ∧ ((a = 3 ∧ b = 3 ∧ c = 4 ∧ a + b + c = 10) ∨ (a = 3 ∧ b = 4 ∧ c = 4 ∧ a + b + c = 11)) :=
by
  sorry

end isosceles_triangle_perimeter_l14_14762


namespace canoe_total_weight_calculation_canoe_maximum_weight_limit_l14_14937

def canoe_max_people : ℕ := 8
def people_with_pets_ratio : ℚ := 3 / 4
def adult_weight : ℚ := 150
def child_weight : ℚ := adult_weight / 2
def dog_weight : ℚ := adult_weight / 3
def cat1_weight : ℚ := adult_weight / 10
def cat2_weight : ℚ := adult_weight / 8

def canoe_capacity_with_pets : ℚ := people_with_pets_ratio * canoe_max_people

def total_weight_adults_and_children : ℚ := 4 * adult_weight + 2 * child_weight
def total_weight_pets : ℚ := dog_weight + cat1_weight + cat2_weight
def total_weight : ℚ := total_weight_adults_and_children + total_weight_pets

def max_weight_limit : ℚ := canoe_max_people * adult_weight

theorem canoe_total_weight_calculation :
  total_weight = 833 + 3 / 4 := by
  sorry

theorem canoe_maximum_weight_limit :
  max_weight_limit = 1200 := by
  sorry

end canoe_total_weight_calculation_canoe_maximum_weight_limit_l14_14937


namespace solve_system_eq_l14_14760

theorem solve_system_eq (x1 x2 x3 x4 x5 : ℝ) :
  (x3 + x4 + x5)^5 = 3 * x1 ∧
  (x4 + x5 + x1)^5 = 3 * x2 ∧
  (x5 + x1 + x2)^5 = 3 * x3 ∧
  (x1 + x2 + x3)^5 = 3 * x4 ∧
  (x2 + x3 + x4)^5 = 3 * x5 →
  (x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨
  (x1 = 1/3 ∧ x2 = 1/3 ∧ x3 = 1/3 ∧ x4 = 1/3 ∧ x5 = 1/3) ∨
  (x1 = -1/3 ∧ x2 = -1/3 ∧ x3 = -1/3 ∧ x4 = -1/3 ∧ x5 = -1/3) :=
by
  sorry

end solve_system_eq_l14_14760


namespace solution_of_equation_l14_14864

theorem solution_of_equation (a : ℝ) : (∃ x : ℝ, x = 4 ∧ (a * x - 3 = 4 * x + 1)) → a = 5 :=
by
  sorry

end solution_of_equation_l14_14864


namespace dino_dolls_count_l14_14465

theorem dino_dolls_count (T : ℝ) (H : 0.7 * T = 140) : T = 200 :=
sorry

end dino_dolls_count_l14_14465


namespace probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l14_14883

-- Definitions and conditions from the problem
def success_probability_each_segment : ℚ := 3 / 4
def num_segments : ℕ := 4

-- Correct answers from the solution
def prob_both_success : ℚ := 9 / 16
def expected_successful_segments : ℚ := 3
def cond_prob_given_3_successful : ℚ := 3 / 4

theorem probability_both_segments_successful :
  (success_probability_each_segment * success_probability_each_segment) = prob_both_success :=
by
  sorry

theorem expected_number_of_successful_segments :
  (num_segments * success_probability_each_segment) = expected_successful_segments :=
by
  sorry

theorem probability_given_3_successful :
  let prob_M := 4 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  let prob_NM := 3 * (success_probability_each_segment^3 * (1 - success_probability_each_segment))
  (prob_NM / prob_M) = cond_prob_given_3_successful :=
by
  sorry

end probability_both_segments_successful_expected_number_of_successful_segments_probability_given_3_successful_l14_14883


namespace expected_yolks_correct_l14_14193

-- Define the conditions
def total_eggs : ℕ := 15
def double_yolk_eggs : ℕ := 5
def triple_yolk_eggs : ℕ := 3
def single_yolk_eggs : ℕ := total_eggs - double_yolk_eggs - triple_yolk_eggs
def extra_yolk_prob : ℝ := 0.10

-- Define the expected number of yolks calculation
noncomputable def expected_yolks : ℝ :=
  (single_yolk_eggs * 1) + 
  (double_yolk_eggs * 2) + 
  (triple_yolk_eggs * 3) + 
  (double_yolk_eggs * extra_yolk_prob) + 
  (triple_yolk_eggs * extra_yolk_prob)

-- State that the expected number of total yolks is 26.8
theorem expected_yolks_correct : expected_yolks = 26.8 := by
  -- solution would go here
  sorry

end expected_yolks_correct_l14_14193


namespace has_two_distinct_roots_and_ordered_l14_14469

-- Define the context and the conditions of the problem.
variables (a b c : ℝ) (h : a < b) (h2 : b < c)

-- Define the quadratic function derived from the problem.
def quadratic (x : ℝ) : ℝ :=
  (x - a) * (x - b) + (x - a) * (x - c) + (x - b) * (x - c)

-- State the main theorem.
theorem has_two_distinct_roots_and_ordered:
  ∃ x1 x2 : ℝ, quadratic a b c x1 = 0 ∧ quadratic a b c x2 = 0 ∧ a < x1 ∧ x1 < b ∧ b < x2 ∧ x2 < c :=
sorry

end has_two_distinct_roots_and_ordered_l14_14469


namespace am_gm_inequality_l14_14776

-- Definitions of the variables and hypotheses
variables {a b : ℝ}

-- The theorem statement
theorem am_gm_inequality (h : a * b > 0) : a / b + b / a ≥ 2 :=
sorry

end am_gm_inequality_l14_14776


namespace unique_solution_quadratic_l14_14669

theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃ x, q * x^2 - 18 * x + 8 = 0 ∧ ∀ y, q * y^2 - 18 * y + 8 = 0 → y = x) →
  q = 81 / 8 :=
by
  sorry

end unique_solution_quadratic_l14_14669


namespace smallest_four_digit_multiple_of_17_l14_14692

theorem smallest_four_digit_multiple_of_17 : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ 17 ∣ n ∧ ∀ m, m ≥ 1000 ∧ m < 10000 ∧ 17 ∣ m → n ≤ m := 
by
  use 1003
  sorry

end smallest_four_digit_multiple_of_17_l14_14692


namespace inequality_solution_set_inequality_proof_l14_14757

def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

theorem inequality_solution_set :
  ∀ x : ℝ, -2 < f x ∧ f x < 0 ↔ -1/2 < x ∧ x < 1/2 :=
by
  sorry

theorem inequality_proof (m n : ℝ) (h_m : -1/2 < m ∧ m < 1/2) (h_n : -1/2 < n ∧ n < 1/2) :
  |1 - 4 * m * n| > 2 * |m - n| :=
by
  sorry

end inequality_solution_set_inequality_proof_l14_14757


namespace rectangle_area_y_l14_14142

theorem rectangle_area_y (y : ℝ) (h_y_pos : y > 0)
  (h_area : (3 * y = 21)) : y = 7 :=
by
  sorry

end rectangle_area_y_l14_14142


namespace xyz_inequality_l14_14051

theorem xyz_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z ≥ 1) :
    (x^4 + y) * (y^4 + z) * (z^4 + x) ≥ (x + y^2) * (y + z^2) * (z + x^2) :=
by
  sorry

end xyz_inequality_l14_14051


namespace smallest_integer_y_l14_14825

theorem smallest_integer_y (y : ℤ) : (5 : ℝ) / 8 < (y : ℝ) / 17 → y = 11 := by
  sorry

end smallest_integer_y_l14_14825


namespace circumradius_geq_3_times_inradius_l14_14579

-- Define the variables representing the circumradius and inradius
variables {R r : ℝ}

-- Assume the conditions that R is the circumradius and r is the inradius of a tetrahedron
def tetrahedron_circumradius (R : ℝ) : Prop := true
def tetrahedron_inradius (r : ℝ) : Prop := true

-- State the theorem
theorem circumradius_geq_3_times_inradius (hR : tetrahedron_circumradius R) (hr : tetrahedron_inradius r) : R ≥ 3 * r :=
sorry

end circumradius_geq_3_times_inradius_l14_14579


namespace ratio_of_sums_l14_14061

theorem ratio_of_sums (a b c : ℚ) (h1 : b / a = 2) (h2 : c / b = 3) : (a + b) / (b + c) = 3 / 8 := 
  sorry

end ratio_of_sums_l14_14061


namespace distinct_digits_sum_l14_14422

theorem distinct_digits_sum (A B C D G : ℕ) (AB CD GGG : ℕ)
  (h1: AB = 10 * A + B)
  (h2: CD = 10 * C + D)
  (h3: GGG = 111 * G)
  (h4: AB * CD = GGG)
  (h5: A ≠ B)
  (h6: A ≠ C)
  (h7: A ≠ D)
  (h8: A ≠ G)
  (h9: B ≠ C)
  (h10: B ≠ D)
  (h11: B ≠ G)
  (h12: C ≠ D)
  (h13: C ≠ G)
  (h14: D ≠ G)
  (hA: A < 10)
  (hB: B < 10)
  (hC: C < 10)
  (hD: D < 10)
  (hG: G < 10)
  : A + B + C + D + G = 17 := sorry

end distinct_digits_sum_l14_14422


namespace tan_identity_equality_l14_14535

theorem tan_identity_equality
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 :=
by
  sorry

end tan_identity_equality_l14_14535


namespace degree_of_monomial_3ab_l14_14383

variable (a b : ℕ)

def monomialDegree (x y : ℕ) : ℕ :=
  x + y

theorem degree_of_monomial_3ab : monomialDegree 1 1 = 2 :=
by
  sorry

end degree_of_monomial_3ab_l14_14383


namespace ratio_problem_l14_14358

theorem ratio_problem (c d : ℚ) (h1 : c / d = 4) (h2 : c = 15 - 3 * d) : d = 15 / 7 := by
  sorry

end ratio_problem_l14_14358


namespace sum_of_fractions_l14_14782

-- Definition of the fractions given as conditions
def frac1 := 2 / 10
def frac2 := 4 / 40
def frac3 := 6 / 60
def frac4 := 8 / 30

-- Statement of the theorem to prove
theorem sum_of_fractions : frac1 + frac2 + frac3 + frac4 = 2 / 3 := by
  sorry

end sum_of_fractions_l14_14782


namespace complement_U_A_eq_two_l14_14538

open Set

universe u

def U : Set ℕ := { x | x ≥ 2 }
def A : Set ℕ := { x | x^2 ≥ 5 }
def comp_U_A : Set ℕ := U \ A

theorem complement_U_A_eq_two : comp_U_A = {2} :=
by 
  sorry

end complement_U_A_eq_two_l14_14538


namespace cost_difference_proof_l14_14560

-- Define the cost per copy at print shop X
def cost_per_copy_X : ℝ := 1.25

-- Define the cost per copy at print shop Y
def cost_per_copy_Y : ℝ := 2.75

-- Define the number of copies
def number_of_copies : ℝ := 60

-- Define the total cost at print shop X
def total_cost_X : ℝ := cost_per_copy_X * number_of_copies

-- Define the total cost at print shop Y
def total_cost_Y : ℝ := cost_per_copy_Y * number_of_copies

-- Define the difference in cost between print shop Y and print shop X
def cost_difference : ℝ := total_cost_Y - total_cost_X

-- The theorem statement proving the cost difference is $90
theorem cost_difference_proof : cost_difference = 90 := by
  sorry

end cost_difference_proof_l14_14560


namespace solve_for_x_l14_14326

theorem solve_for_x (x : ℝ) (h : 2 * x + 10 = (1 / 2) * (5 * x + 30)) : x = -10 :=
sorry

end solve_for_x_l14_14326


namespace sqrt_sum_eq_eight_l14_14552

theorem sqrt_sum_eq_eight :
  Real.sqrt (24 - 8 * Real.sqrt 3) + Real.sqrt (24 + 8 * Real.sqrt 3) = 8 := by
  sorry

end sqrt_sum_eq_eight_l14_14552


namespace roots_sum_and_product_l14_14464

theorem roots_sum_and_product (p q : ℝ) (h_sum : p / 3 = 9) (h_prod : q / 3 = 24) : p + q = 99 :=
by
  -- We are given h_sum: p / 3 = 9
  -- We are given h_prod: q / 3 = 24
  -- We need to prove p + q = 99
  sorry

end roots_sum_and_product_l14_14464


namespace percent_republicans_voting_for_A_l14_14453

theorem percent_republicans_voting_for_A (V : ℝ) (percent_Democrats : ℝ) 
  (percent_Republicans : ℝ) (percent_D_voting_for_A : ℝ) 
  (percent_total_voting_for_A : ℝ) (R : ℝ) 
  (h1 : percent_Democrats = 0.60)
  (h2 : percent_Republicans = 0.40)
  (h3 : percent_D_voting_for_A = 0.85)
  (h4 : percent_total_voting_for_A = 0.59) :
  R = 0.2 :=
by 
  sorry

end percent_republicans_voting_for_A_l14_14453


namespace simplify_expression_evaluate_expression_at_neg1_evaluate_expression_at_2_l14_14533

theorem simplify_expression (a : ℤ) (h1 : -2 < a) (h2 : a ≤ 2) (h3 : a ≠ 0) (h4 : a ≠ 1) :
  (a - (2 * a - 1) / a) / ((a - 1) / a) = a - 1 :=
by
  sorry

theorem evaluate_expression_at_neg1 (h : (-1 : ℤ) ≠ 0) (h' : (-1 : ℤ) ≠ 1) : 
  (-1 - (2 * (-1) - 1) / (-1)) / ((-1 - 1) / (-1)) = -2 :=
by
  sorry

theorem evaluate_expression_at_2 (h : (2 : ℤ) ≠ 0) (h' : (2 : ℤ) ≠ 1) : 
  (2 - (2 * 2 - 1) / 2) / ((2 - 1) / 2) = 1 :=
by
  sorry

end simplify_expression_evaluate_expression_at_neg1_evaluate_expression_at_2_l14_14533


namespace female_muscovy_ducks_l14_14486

theorem female_muscovy_ducks :
  let total_ducks := 40
  let muscovy_percentage := 0.5
  let female_muscovy_percentage := 0.3
  let muscovy_ducks := total_ducks * muscovy_percentage
  let female_muscovy_ducks := muscovy_ducks * female_muscovy_percentage
  female_muscovy_ducks = 6 :=
by
  sorry

end female_muscovy_ducks_l14_14486


namespace marley_fruits_l14_14354

theorem marley_fruits 
    (louis_oranges : ℕ := 5) (louis_apples : ℕ := 3)
    (samantha_oranges : ℕ := 8) (samantha_apples : ℕ := 7)
    (marley_oranges : ℕ := 2 * louis_oranges)
    (marley_apples : ℕ := 3 * samantha_apples) :
    marley_oranges + marley_apples = 31 := by
  sorry

end marley_fruits_l14_14354


namespace sum_of_remaining_two_scores_l14_14030

open Nat

theorem sum_of_remaining_two_scores :
  ∃ x y : ℕ, x + y = 160 ∧ (65 + 75 + 85 + 95 + x + y) / 6 = 80 :=
by
  sorry

end sum_of_remaining_two_scores_l14_14030


namespace distance_from_point_to_line_l14_14744

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def cartesian_distance_to_line (point : ℝ × ℝ) (y_line : ℝ) : ℝ :=
  abs (point.snd - y_line)

theorem distance_from_point_to_line
  (ρ θ : ℝ)
  (h_point : ρ = 2 ∧ θ = Real.pi / 6)
  (h_line : ∀ θ, (3 : ℝ) = ρ * Real.sin θ) :
  cartesian_distance_to_line (polar_to_cartesian ρ θ) 3 = 2 :=
  sorry

end distance_from_point_to_line_l14_14744


namespace sufficient_but_not_necessary_condition_l14_14957

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a = 2 → |a| = 2) ∧ (|a| = 2 → a = 2 ∨ a = -2) :=
by
  sorry

end sufficient_but_not_necessary_condition_l14_14957


namespace parker_savings_l14_14041

-- Define the costs of individual items and meals
def burger_cost : ℝ := 5
def fries_cost : ℝ := 3
def drink_cost : ℝ := 3
def special_meal_cost : ℝ := 9.5
def kids_burger_cost : ℝ := 3
def kids_fries_cost : ℝ := 2
def kids_drink_cost : ℝ := 2
def kids_meal_cost : ℝ := 5

-- Define the number of meals Mr. Parker buys
def adult_meals : ℕ := 2
def kids_meals : ℕ := 2

-- Define the total cost of individual items for adults and children
def total_individual_cost_adults : ℝ :=
  adult_meals * (burger_cost + fries_cost + drink_cost)

def total_individual_cost_children : ℝ :=
  kids_meals * (kids_burger_cost + kids_fries_cost + kids_drink_cost)

-- Define the total cost of meal deals
def total_meals_cost : ℝ :=
  adult_meals * special_meal_cost + kids_meals * kids_meal_cost

-- Define the total cost of individual items for both adults and children
def total_individual_cost : ℝ :=
  total_individual_cost_adults + total_individual_cost_children

-- Define the savings
def savings : ℝ := total_individual_cost - total_meals_cost

theorem parker_savings : savings = 7 :=
by
  sorry

end parker_savings_l14_14041


namespace geometric_series_sum_l14_14517

theorem geometric_series_sum :
  2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2))))))))) = 2046 := 
by sorry

end geometric_series_sum_l14_14517


namespace time_to_cross_same_direction_l14_14431

-- Defining the conditions
def speed_train1 : ℝ := 60 -- kmph
def speed_train2 : ℝ := 40 -- kmph
def time_opposite_directions : ℝ := 10.000000000000002 -- seconds 
def relative_speed_opposite_directions : ℝ := speed_train1 + speed_train2 -- 100 kmph
def relative_speed_same_direction : ℝ := speed_train1 - speed_train2 -- 20 kmph

-- Defining the proof statement
theorem time_to_cross_same_direction : 
  (time_opposite_directions * (relative_speed_opposite_directions / relative_speed_same_direction)) = 50 :=
by
  sorry

end time_to_cross_same_direction_l14_14431


namespace arithmetic_sequence_problem_l14_14156

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (d : ℚ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 4 + 1 / 2 * a 7 + a 10 = 10) : a 3 + a 11 = 8 :=
sorry

end arithmetic_sequence_problem_l14_14156


namespace sales_ratio_l14_14902

def large_price : ℕ := 60
def small_price : ℕ := 30
def last_month_large_paintings : ℕ := 8
def last_month_small_paintings : ℕ := 4
def this_month_sales : ℕ := 1200

theorem sales_ratio :
  (this_month_sales : ℕ) = 2 * (last_month_large_paintings * large_price + last_month_small_paintings * small_price) :=
by
  -- We will just state the proof steps as sorry.
  sorry

end sales_ratio_l14_14902


namespace modules_count_l14_14620

theorem modules_count (x y: ℤ) (hx: 10 * x + 35 * y = 450) (hy: x + y = 11) : y = 10 :=
by
  sorry

end modules_count_l14_14620


namespace arithmetic_progression_condition_l14_14675

theorem arithmetic_progression_condition
  (a b c : ℝ) : ∃ (A B : ℤ), A ≠ 0 ∧ B ≠ 0 ∧ (b - a) * B = (c - b) * A := 
by {
  sorry
}

end arithmetic_progression_condition_l14_14675


namespace girls_try_out_l14_14327

-- Given conditions
variables (boys callBacks didNotMakeCut : ℕ)
variable (G : ℕ)

-- Define the conditions
def conditions : Prop := 
  boys = 14 ∧ 
  callBacks = 2 ∧ 
  didNotMakeCut = 21 ∧ 
  G + boys = callBacks + didNotMakeCut

-- The statement of the proof
theorem girls_try_out (h : conditions boys callBacks didNotMakeCut G) : G = 9 :=
by
  sorry

end girls_try_out_l14_14327


namespace solve_absolute_value_eq_l14_14195

theorem solve_absolute_value_eq (x : ℝ) : |x - 5| = 3 * x - 2 ↔ x = 7 / 4 :=
sorry

end solve_absolute_value_eq_l14_14195


namespace find_x_l14_14905

theorem find_x (x y: ℤ) (h1: x + 2 * y = 12) (h2: y = 3) : x = 6 := by
  sorry

end find_x_l14_14905


namespace sin_alpha_minus_pi_over_6_l14_14264

open Real

theorem sin_alpha_minus_pi_over_6 (α : ℝ) (h : sin (α + π / 6) + 2 * sin (α / 2) ^ 2 = 1 - sqrt 2 / 2) : 
  sin (α - π / 6) = -sqrt 2 / 2 :=
sorry

end sin_alpha_minus_pi_over_6_l14_14264


namespace quadratic_coefficients_l14_14510

theorem quadratic_coefficients (x1 x2 p q : ℝ)
  (h1 : x1 - x2 = 5)
  (h2 : x1 ^ 3 - x2 ^ 3 = 35) :
  (x1 + x2 = -p ∧ x1 * x2 = q ∧ (p = 1 ∧ q = -6) ∨ 
   x1 + x2 = p ∧ x1 * x2 = q ∧ (p = -1 ∧ q = -6)) :=
by
  sorry

end quadratic_coefficients_l14_14510


namespace doubled_marks_new_average_l14_14908

theorem doubled_marks_new_average (avg_marks : ℝ) (num_students : ℕ) (h_avg : avg_marks = 36) (h_num : num_students = 12) : 2 * avg_marks = 72 :=
by
  sorry

end doubled_marks_new_average_l14_14908


namespace solution_l14_14019

theorem solution (y : ℚ) (h : (1/3 : ℚ) + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solution_l14_14019


namespace inequality_xyz_l14_14253

theorem inequality_xyz (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) : 
    x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := 
    sorry

end inequality_xyz_l14_14253


namespace speed_of_current_l14_14116

-- Define the context and variables
variables (m c : ℝ)
-- State the conditions
variables (h1 : m + c = 12) (h2 : m - c = 8)

-- State the goal which is to prove the speed of the current
theorem speed_of_current : c = 2 :=
by
  sorry

end speed_of_current_l14_14116


namespace evaluate_expression_l14_14105

def g (x : ℝ) := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 2 + 2 * g (-4) = 177 := by
  sorry

end evaluate_expression_l14_14105


namespace calculate_gfg3_l14_14312

def f (x : ℕ) : ℕ := 2 * x + 4
def g (x : ℕ) : ℕ := 5 * x + 2

theorem calculate_gfg3 : g (f (g 3)) = 192 := by
  sorry

end calculate_gfg3_l14_14312


namespace prob_B_hired_is_3_4_prob_at_least_two_hired_l14_14182

-- Definitions for the conditions
def prob_A_hired : ℚ := 2 / 3
def prob_neither_A_nor_B_hired : ℚ := 1 / 12
def prob_B_and_C_hired : ℚ := 3 / 8

-- Targets to prove
theorem prob_B_hired_is_3_4 (P_A_hired : ℚ) (P_neither_A_nor_B_hired : ℚ) (P_B_and_C_hired : ℚ)
    (P_A_hired_eq : P_A_hired = prob_A_hired)
    (P_neither_A_nor_B_hired_eq : P_neither_A_nor_B_hired = prob_neither_A_nor_B_hired)
    (P_B_and_C_hired_eq : P_B_and_C_hired = prob_B_and_C_hired)
    : ∃ x y : ℚ, y = 1 / 2 ∧ x = 3 / 4 :=
by
  sorry
  
theorem prob_at_least_two_hired (P_A_hired : ℚ) (P_B_hired : ℚ) (P_C_hired : ℚ)
    (P_A_hired_eq : P_A_hired = prob_A_hired)
    (P_B_hired_eq : P_B_hired = 3 / 4)
    (P_C_hired_eq : P_C_hired = 1 / 2)
    : (P_A_hired * P_B_hired * P_C_hired) + 
      ((1 - P_A_hired) * P_B_hired * P_C_hired) + 
      (P_A_hired * (1 - P_B_hired) * P_C_hired) + 
      (P_A_hired * P_B_hired * (1 - P_C_hired)) = 2 / 3 :=
by
  sorry

end prob_B_hired_is_3_4_prob_at_least_two_hired_l14_14182


namespace arithmetic_sequence_a5_eq_6_l14_14737

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_a5_eq_6 (h_arith : is_arithmetic_sequence a_n) (h_sum : a_n 2 + a_n 8 = 12) : a_n 5 = 6 :=
by
  sorry

end arithmetic_sequence_a5_eq_6_l14_14737


namespace minimum_students_for_same_vote_l14_14255

theorem minimum_students_for_same_vote (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 2) :
  ∃ m, m = 46 ∧ ∀ (students : Finset (Finset ℕ)), students.card = m → 
    (∃ s1 s2, s1 ≠ s2 ∧ s1.card = k ∧ s2.card = k ∧ s1 ⊆ (Finset.range n) ∧ s2 ⊆ (Finset.range n) ∧ s1 = s2) :=
by 
  sorry

end minimum_students_for_same_vote_l14_14255


namespace find_principal_l14_14766

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * ((1 + r / n) ^ (n * t))

theorem find_principal
  (A : ℝ) (r : ℝ) (n t : ℕ)
  (hA : A = 4410)
  (hr : r = 0.05)
  (hn : n = 1)
  (ht : t = 2) :
  ∃ (P : ℝ), compound_interest P r n t = A ∧ P = 4000 :=
by
  sorry

end find_principal_l14_14766


namespace positive_value_of_A_l14_14734

theorem positive_value_of_A (A : ℝ) :
  (A ^ 2 + 7 ^ 2 = 200) → A = Real.sqrt 151 :=
by
  intros h
  sorry

end positive_value_of_A_l14_14734


namespace predicted_sales_volume_l14_14233

-- Define the linear regression equation
def regression_equation (x : ℝ) : ℝ := 2 * x + 60

-- Use the given condition x = 34
def temperature_value : ℝ := 34

-- State the theorem that the predicted sales volume is 128
theorem predicted_sales_volume : regression_equation temperature_value = 128 :=
by
  sorry

end predicted_sales_volume_l14_14233


namespace jane_ate_four_pieces_l14_14179

def total_pieces : ℝ := 12.0
def num_people : ℝ := 3.0
def pieces_per_person : ℝ := 4.0

theorem jane_ate_four_pieces :
  total_pieces / num_people = pieces_per_person := 
  by
    sorry

end jane_ate_four_pieces_l14_14179


namespace annulus_area_sufficient_linear_element_l14_14922

theorem annulus_area_sufficient_linear_element (R r : ℝ) (hR : R > 0) (hr : r > 0) (hrR : r < R):
  (∃ d : ℝ, d = R - r ∨ d = R + r) → ∃ A : ℝ, A = π * (R ^ 2 - r ^ 2) :=
by
  sorry

end annulus_area_sufficient_linear_element_l14_14922


namespace sparrows_among_non_pigeons_l14_14151

theorem sparrows_among_non_pigeons (perc_sparrows perc_pigeons perc_parrots perc_crows : ℝ)
  (h_sparrows : perc_sparrows = 0.40)
  (h_pigeons : perc_pigeons = 0.20)
  (h_parrots : perc_parrots = 0.15)
  (h_crows : perc_crows = 0.25) :
  (perc_sparrows / (1 - perc_pigeons) * 100) = 50 :=
by
  sorry

end sparrows_among_non_pigeons_l14_14151


namespace range_of_a_l14_14959

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x = 1 → ¬ ((x + 1) / (x + a) < 2))) ↔ -1 ≤ a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l14_14959


namespace hotel_room_friends_distribution_l14_14301

theorem hotel_room_friends_distribution 
    (rooms : ℕ)
    (friends : ℕ)
    (min_friends_per_room : ℕ)
    (max_friends_per_room : ℕ)
    (unique_ways : ℕ) :
    rooms = 6 →
    friends = 10 →
    min_friends_per_room = 1 →
    max_friends_per_room = 3 →
    unique_ways = 1058400 :=
by
  intros h_rooms h_friends h_min_friends h_max_friends
  sorry

end hotel_room_friends_distribution_l14_14301


namespace find_d_l14_14914

theorem find_d (d : ℝ) (h : 4 * (3.6 * 0.48 * 2.50) / (d * 0.09 * 0.5) = 3200.0000000000005) : d = 0.3 :=
by
  sorry

end find_d_l14_14914


namespace infinite_geometric_series_second_term_l14_14344

theorem infinite_geometric_series_second_term (a r S : ℝ) (h1 : r = 1 / 4) (h2 : S = 16) (h3 : S = a / (1 - r)) : a * r = 3 := 
sorry

end infinite_geometric_series_second_term_l14_14344


namespace trapezium_distance_parallel_sides_l14_14014

theorem trapezium_distance_parallel_sides
  (l1 l2 area : ℝ) (h : ℝ)
  (h_area : area = (1 / 2) * (l1 + l2) * h)
  (hl1 : l1 = 30)
  (hl2 : l2 = 12)
  (h_area_val : area = 336) :
  h = 16 :=
by
  sorry

end trapezium_distance_parallel_sides_l14_14014


namespace students_in_both_band_and_chorus_l14_14104

-- Definitions for conditions
def total_students : ℕ := 300
def students_in_band : ℕ := 100
def students_in_chorus : ℕ := 120
def students_in_band_or_chorus : ℕ := 195

-- Theorem: Prove the number of students in both band and chorus
theorem students_in_both_band_and_chorus : ℕ :=
  students_in_band + students_in_chorus - students_in_band_or_chorus

example : students_in_both_band_and_chorus = 25 := by
  sorry

end students_in_both_band_and_chorus_l14_14104


namespace volleyballs_basketballs_difference_l14_14549

variable (V B : ℕ)

theorem volleyballs_basketballs_difference :
  (V + B = 14) →
  (4 * V + 5 * B = 60) →
  V - B = 6 :=
by
  intros h1 h2
  sorry

end volleyballs_basketballs_difference_l14_14549


namespace P_2017_P_eq_4_exists_P_minus_P_succ_gt_50_l14_14516

-- Assume the definition of sum of digits of n and count of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum  -- Sum of digits in base 10 representation

def num_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).length  -- Number of digits in base 10 representation

def P (n : ℕ) : ℕ :=
  sum_of_digits n + num_of_digits n

-- Problem (a)
theorem P_2017 : P 2017 = 14 :=
sorry

-- Problem (b)
theorem P_eq_4 :
  {n : ℕ | P n = 4} = {3, 11, 20, 100} :=
sorry

-- Problem (c)
theorem exists_P_minus_P_succ_gt_50 : 
  ∃ n : ℕ, P n - P (n + 1) > 50 :=
sorry

end P_2017_P_eq_4_exists_P_minus_P_succ_gt_50_l14_14516


namespace value_of_x_squared_plus_inverse_squared_l14_14113

theorem value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x + 1 / x = 8) : x^2 + 1 / x^2 = 62 := 
sorry

end value_of_x_squared_plus_inverse_squared_l14_14113


namespace range_of_omega_l14_14404

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ (a b : ℝ), a ≠ b ∧ 0 ≤ a ∧ a ≤ π/2 ∧ 0 ≤ b ∧ b ≤ π/2 ∧ f ω a + f ω b = 4) ↔ 5 ≤ ω ∧ ω < 9 :=
sorry

end range_of_omega_l14_14404


namespace find_common_ratio_l14_14648

noncomputable def geometric_series (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a_1 * (1 - q^n) / (1 - q)

theorem find_common_ratio (a_1 : ℝ) (q : ℝ) (n : ℕ) (S_n : ℕ → ℝ)
  (h1 : ∀ n, S_n n = geometric_series a_1 q n)
  (h2 : S_n 3 = (2 * a_1 + a_1 * q) / 2)
  : q = -1/2 :=
  sorry

end find_common_ratio_l14_14648


namespace area_of_smallest_square_containing_circle_l14_14780

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 7) : ∃ s, s = 14 ∧ s * s = 196 :=
by
  sorry

end area_of_smallest_square_containing_circle_l14_14780


namespace players_odd_sum_probability_l14_14060

theorem players_odd_sum_probability :
  let tiles := (1:ℕ) :: (2:ℕ) :: (3:ℕ) :: (4:ℕ) :: (5:ℕ) :: (6:ℕ) :: (7:ℕ) :: (8:ℕ) :: (9:ℕ) :: (10:ℕ) :: (11:ℕ) :: []
  let m := 1
  let n := 26
  m + n = 27 :=
by
  sorry

end players_odd_sum_probability_l14_14060


namespace solve_quadratic_l14_14334

theorem solve_quadratic : ∀ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 → x = 1 :=
by
  intros x h
  sorry

end solve_quadratic_l14_14334


namespace total_spent_l14_14324

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_friends : ℕ := 4

theorem total_spent : (original_price * (1 - discount_rate) * number_of_friends) = 40 := by
  sorry

end total_spent_l14_14324


namespace alice_needs_7_fills_to_get_3_cups_l14_14410

theorem alice_needs_7_fills_to_get_3_cups (needs : ℚ) (cup_size : ℚ) (has : ℚ) :
  needs = 3 ∧ cup_size = 1 / 3 ∧ has = 2 / 3 →
  (needs - has) / cup_size = 7 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end alice_needs_7_fills_to_get_3_cups_l14_14410


namespace consecutive_ints_product_div_6_l14_14029

theorem consecutive_ints_product_div_6 (n : ℤ) : (n * (n + 1) * (n + 2)) % 6 = 0 := 
sorry

end consecutive_ints_product_div_6_l14_14029


namespace inequality_ab_sum_eq_five_l14_14363

noncomputable def inequality_solution (a b : ℝ) : Prop :=
  (∀ x : ℝ, (x < 1) → (x < a) → (x > b) ∨ (x > 4) → (x < a) → (x > b))

theorem inequality_ab_sum_eq_five (a b : ℝ) 
  (h : inequality_solution a b) : a + b = 5 :=
sorry

end inequality_ab_sum_eq_five_l14_14363


namespace pen_price_l14_14221

theorem pen_price (x y : ℝ) (h1 : 2 * x + 3 * y = 49) (h2 : 3 * x + y = 49) : x = 14 :=
by
  -- Proof required here
  sorry

end pen_price_l14_14221


namespace required_total_money_l14_14196

def bundle_count := 100
def number_of_bundles := 10
def bill_5_value := 5
def bill_10_value := 10
def bill_20_value := 20

-- Sum up the total money required to fill the machine
theorem required_total_money : 
  (bundle_count * bill_5_value * number_of_bundles) + 
  (bundle_count * bill_10_value * number_of_bundles) + 
  (bundle_count * bill_20_value * number_of_bundles) = 35000 := 
by 
  sorry

end required_total_money_l14_14196


namespace bluegrass_percentage_l14_14260

theorem bluegrass_percentage (rx : ℝ) (ry : ℝ) (f : ℝ) (rm : ℝ) (wx : ℝ) (wy : ℝ) (B : ℝ) :
  rx = 0.4 →
  ry = 0.25 →
  f = 0.75 →
  rm = 0.35 →
  wx = 0.6667 →
  wy = 0.3333 →
  (wx * rx + wy * ry = rm) →
  B = 1.0 - rx →
  B = 0.6 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bluegrass_percentage_l14_14260


namespace george_total_payment_in_dollars_l14_14619
noncomputable def total_cost_in_dollars : ℝ := 
  let sandwich_cost : ℝ := 4
  let juice_cost : ℝ := 2 * sandwich_cost * 0.9
  let coffee_cost : ℝ := sandwich_cost / 2
  let milk_cost : ℝ := 0.75 * (sandwich_cost + juice_cost)
  let milk_cost_dollars : ℝ := milk_cost * 1.2
  let chocolate_bar_cost_pounds : ℝ := 3
  let chocolate_bar_cost_dollars : ℝ := chocolate_bar_cost_pounds * 1.25
  let total_euros_in_items : ℝ := 2 * sandwich_cost + juice_cost + coffee_cost
  let total_euros_to_dollars : ℝ := total_euros_in_items * 1.2
  total_euros_to_dollars + milk_cost_dollars + chocolate_bar_cost_dollars

theorem george_total_payment_in_dollars : total_cost_in_dollars = 38.07 := by
  sorry

end george_total_payment_in_dollars_l14_14619


namespace count_int_values_not_satisfying_ineq_l14_14513

theorem count_int_values_not_satisfying_ineq :
  ∃ (s : Finset ℤ), (∀ x ∈ s, 3 * x^2 + 14 * x + 8 ≤ 17) ∧ (s.card = 10) :=
by
  sorry

end count_int_values_not_satisfying_ineq_l14_14513


namespace intersection_A_B_l14_14491

-- define the set A
def A : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ 3 * x - y = 7 }

-- define the set B
def B : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ 2 * x + y = 3 }

-- Prove the intersection
theorem intersection_A_B :
  A ∩ B = { (2, -1) } :=
by
  -- We will insert the proof here
  sorry

end intersection_A_B_l14_14491


namespace major_axis_of_ellipse_l14_14250

structure Ellipse :=
(center : ℝ × ℝ)
(tangent_y_axis : Bool)
(tangent_y_eq_3 : Bool)
(focus_1 : ℝ × ℝ)
(focus_2 : ℝ × ℝ)

noncomputable def major_axis_length (e : Ellipse) : ℝ :=
  2 * (e.focus_1.2 - e.center.2)

theorem major_axis_of_ellipse : 
  ∀ (e : Ellipse), 
    e.center = (3, 0) ∧
    e.tangent_y_axis = true ∧
    e.tangent_y_eq_3 = true ∧
    e.focus_1 = (3, 2 + Real.sqrt 2) ∧
    e.focus_2 = (3, -2 - Real.sqrt 2) →
      major_axis_length e = 4 + 2 * Real.sqrt 2 :=
by
  intro e
  intro h
  sorry

end major_axis_of_ellipse_l14_14250


namespace length_of_field_l14_14997

variable (l w : ℝ)

theorem length_of_field : 
  (l = 2 * w) ∧ (8 * 8 = 64) ∧ ((8 * 8) = (1 / 50) * l * w) → l = 80 :=
by
  sorry

end length_of_field_l14_14997


namespace map_distance_ratio_l14_14075

theorem map_distance_ratio (actual_distance_km : ℕ) (map_distance_cm : ℕ) (h1 : actual_distance_km = 6) (h2 : map_distance_cm = 20) : map_distance_cm / (actual_distance_km * 100000) = 1 / 30000 :=
by
  -- Proof goes here
  sorry

end map_distance_ratio_l14_14075


namespace sum_of_smallest_integers_l14_14180

theorem sum_of_smallest_integers (x y : ℕ) (h1 : ∃ x, x > 0 ∧ (∃ n : ℕ, 720 * x = n^2) ∧ (∀ m : ℕ, m > 0 ∧ (∃ k : ℕ, 720 * m = k^2) → x ≤ m))
  (h2 : ∃ y, y > 0 ∧ (∃ p : ℕ, 720 * y = p^4) ∧ (∀ q : ℕ, q > 0 ∧ (∃ r : ℕ, 720 * q = r^4) → y ≤ q)) :
  x + y = 1130 := 
sorry

end sum_of_smallest_integers_l14_14180


namespace remainder_276_l14_14984

theorem remainder_276 (y : ℤ) (k : ℤ) (hk : y = 23 * k + 19) : y % 276 = 180 :=
sorry

end remainder_276_l14_14984


namespace find_digit_D_l14_14988

theorem find_digit_D (A B C D : ℕ)
  (h_add : 100 + 10 * A + B + 100 * C + 10 * A + A = 100 * D + 10 * A + B)
  (h_sub : 100 + 10 * A + B - (100 * C + 10 * A + A) = 100 + 10 * A) :
  D = 1 :=
by
  -- Since we're skipping the proof and focusing on the statement only
  sorry

end find_digit_D_l14_14988


namespace at_least_one_angle_not_greater_than_60_l14_14035

theorem at_least_one_angle_not_greater_than_60 (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (hSum : A + B + C = 180) : false :=
by
  sorry

end at_least_one_angle_not_greater_than_60_l14_14035


namespace total_wood_gathered_l14_14718

def pieces_per_sack := 20
def number_of_sacks := 4

theorem total_wood_gathered : pieces_per_sack * number_of_sacks = 80 := 
by 
  sorry

end total_wood_gathered_l14_14718


namespace ratio_of_fractions_l14_14032

theorem ratio_of_fractions (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) : 
  (1 / 5 * x) / (1 / 6 * y) = 0.72 :=
sorry

end ratio_of_fractions_l14_14032


namespace parabola_equation_l14_14175

theorem parabola_equation (vertex focus : ℝ × ℝ) 
  (h_vertex : vertex = (0, 0)) 
  (h_focus_line : ∃ x y : ℝ, focus = (x, y) ∧ x - y + 2 = 0) 
  (h_symmetry_axis : ∃ axis : ℝ × ℝ → ℝ, ∀ p : ℝ × ℝ, axis p = 0): 
  ∃ k : ℝ, k > 0 ∧ (∀ x y : ℝ, y^2 = -8*x ∨ x^2 = 8*y) :=
by {
  sorry
}

end parabola_equation_l14_14175


namespace muffin_is_twice_as_expensive_as_banana_l14_14387

variable (m b : ℚ)
variable (h1 : 4 * m + 10 * b = 3 * m + 5 * b + 12)
variable (h2 : 3 * m + 5 * b = S)

theorem muffin_is_twice_as_expensive_as_banana (h1 : 4 * m + 10 * b = 3 * m + 5 * b + 12) : m = 2 * b :=
by
  sorry

end muffin_is_twice_as_expensive_as_banana_l14_14387


namespace number_of_girls_l14_14048

theorem number_of_girls (B G : ℕ) (ratio_condition : B = G / 2) (total_condition : B + G = 90) : 
  G = 60 := 
by
  -- This is the problem statement, with conditions and required result.
  sorry

end number_of_girls_l14_14048


namespace students_not_reading_l14_14024

theorem students_not_reading (total_girls : ℕ) (total_boys : ℕ)
  (frac_girls_reading : ℚ) (frac_boys_reading : ℚ)
  (h1 : total_girls = 12) (h2 : total_boys = 10)
  (h3 : frac_girls_reading = 5 / 6) (h4 : frac_boys_reading = 4 / 5) :
  let girls_not_reading := total_girls - total_girls * frac_girls_reading
  let boys_not_reading := total_boys - total_boys * frac_boys_reading
  let total_not_reading := girls_not_reading + boys_not_reading
  total_not_reading = 4 := sorry

end students_not_reading_l14_14024


namespace first_number_in_proportion_is_correct_l14_14637

-- Define the proportion condition
def proportion_condition (a x : ℝ) : Prop := a / x = 5 / 11

-- Define the given known value for x
def x_value : ℝ := 1.65

-- Define the correct answer for a
def correct_a : ℝ := 0.75

-- The theorem to prove
theorem first_number_in_proportion_is_correct :
  ∀ a : ℝ, proportion_condition a x_value → a = correct_a := by
  sorry

end first_number_in_proportion_is_correct_l14_14637


namespace swapped_digit_number_l14_14270

theorem swapped_digit_number (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  10 * b + a = new_number :=
sorry

end swapped_digit_number_l14_14270


namespace PetesOriginalNumber_l14_14805

-- Define the context and problem
theorem PetesOriginalNumber (x : ℤ) (h : 3 * (2 * x + 12) = 90) : x = 9 :=
by
  -- proof goes here
  sorry

end PetesOriginalNumber_l14_14805


namespace chloe_pawn_loss_l14_14351

theorem chloe_pawn_loss (sophia_lost : ℕ) (total_left : ℕ) (total_initial : ℕ) (each_start : ℕ) (sophia_initial : ℕ) :
  sophia_lost = 5 → total_left = 10 → each_start = 8 → total_initial = 16 → sophia_initial = 8 →
  ∃ (chloe_lost : ℕ), chloe_lost = 1 :=
by
  sorry

end chloe_pawn_loss_l14_14351


namespace inequality_div_c_squared_l14_14259

theorem inequality_div_c_squared (a b c : ℝ) (h : a > b) : (a / (c^2 + 1) > b / (c^2 + 1)) :=
by
  sorry

end inequality_div_c_squared_l14_14259


namespace find_k_l14_14591

theorem find_k (x y k : ℤ) 
  (h1 : 2 * x - y = 5 * k + 6) 
  (h2 : 4 * x + 7 * y = k) 
  (h3 : x + y = 2023) : 
  k = 2022 := 
  by 
    sorry

end find_k_l14_14591


namespace tangent_function_range_l14_14789

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + 1
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x

theorem tangent_function_range {a : ℝ} :
  (∃ (m : ℝ), 4 * m^3 - 3 * a * m^2 + 6 = 0) ↔ a > 2 * Real.sqrt 33 :=
sorry -- proof omitted

end tangent_function_range_l14_14789


namespace proposition_not_hold_for_4_l14_14770

variable (P : ℕ → Prop)

axiom induction_step (k : ℕ) (hk : k > 0) : P k → P (k + 1)
axiom base_case : ¬ P 5

theorem proposition_not_hold_for_4 : ¬ P 4 :=
sorry

end proposition_not_hold_for_4_l14_14770


namespace green_pairs_count_l14_14238

variable (blueShirtedStudents : Nat)
variable (yellowShirtedStudents : Nat)
variable (greenShirtedStudents : Nat)
variable (totalStudents : Nat)
variable (totalPairs : Nat)
variable (blueBluePairs : Nat)

def green_green_pairs (blueShirtedStudents yellowShirtedStudents greenShirtedStudents totalStudents totalPairs blueBluePairs : Nat) : Nat := 
  greenShirtedStudents / 2

theorem green_pairs_count
  (h1 : blueShirtedStudents = 70)
  (h2 : yellowShirtedStudents = 80)
  (h3 : greenShirtedStudents = 50)
  (h4 : totalStudents = 200)
  (h5 : totalPairs = 100)
  (h6 : blueBluePairs = 30) : 
  green_green_pairs blueShirtedStudents yellowShirtedStudents greenShirtedStudents totalStudents totalPairs blueBluePairs = 25 := by
  sorry

end green_pairs_count_l14_14238


namespace crushing_load_calculation_l14_14002

theorem crushing_load_calculation (T H : ℝ) (L : ℝ) 
  (h1 : L = 40 * T^5 / H^3) 
  (h2 : T = 3) 
  (h3 : H = 6) : 
  L = 45 := 
by sorry

end crushing_load_calculation_l14_14002


namespace train_usual_time_l14_14972

theorem train_usual_time (S T_new T : ℝ) (h_speed : T_new = 7 / 6 * T) (h_delay : T_new = T + 1 / 6) : T = 1 := by
  sorry

end train_usual_time_l14_14972


namespace cone_base_radius_l14_14867

theorem cone_base_radius (slant_height : ℝ) (central_angle_deg : ℝ) (r : ℝ) 
  (h1 : slant_height = 6) 
  (h2 : central_angle_deg = 120) 
  (h3 : 2 * π * slant_height * (central_angle_deg / 360) = 4 * π) 
  : r = 2 := by
  sorry

end cone_base_radius_l14_14867


namespace CarriageSharingEquation_l14_14165

theorem CarriageSharingEquation (x : ℕ) :
  (x / 3 + 2 = (x - 9) / 2) ↔
  (3 * ((x - 9) / 2) + 2 * 3 = x / 3 + 2) ∧ 
  (2 * ((x - 9) / 2) + 9 = x ∨ 2 * ((x - 9) / 2) + 9 < x) ∧ 
  (x / 3 + 2 < 3 * (x / 2) + 2 * 2 ∨ x / 3 + 2 = 3 * (x / 2) + 2 * 2) :=
sorry

end CarriageSharingEquation_l14_14165


namespace jill_arrives_30_minutes_before_jack_l14_14031

theorem jill_arrives_30_minutes_before_jack
    (d : ℝ) (s_jill : ℝ) (s_jack : ℝ) (t_diff : ℝ)
    (h_d : d = 2)
    (h_s_jill : s_jill = 12)
    (h_s_jack : s_jack = 3)
    (h_t_diff : t_diff = 30) :
    ((d / s_jack) * 60 - (d / s_jill) * 60) = t_diff :=
by
  sorry

end jill_arrives_30_minutes_before_jack_l14_14031


namespace distance_interval_l14_14904

theorem distance_interval (d : ℝ) (h₁ : ¬ (d ≥ 8)) (h₂ : ¬ (d ≤ 6)) (h₃ : ¬ (d ≤ 3)) : 6 < d ∧ d < 8 := by
  sorry

end distance_interval_l14_14904


namespace min_moves_to_reset_counters_l14_14721

theorem min_moves_to_reset_counters (f : Fin 28 -> Nat) (h_initial : ∀ i, 1 ≤ f i ∧ f i ≤ 2017) :
  ∃ k, k = 11 ∧ ∀ g : Fin 28 -> Nat, (∀ i, f i = 0) :=
by
  sorry

end min_moves_to_reset_counters_l14_14721


namespace system1_solution_l14_14146

theorem system1_solution (x y : ℝ) (h1 : 2 * x - y = 1) (h2 : 7 * x - 3 * y = 4) : x = 1 ∧ y = 1 :=
by sorry

end system1_solution_l14_14146


namespace auntie_em_can_park_l14_14106

-- Define the conditions as formal statements in Lean
def parking_lot_spaces : ℕ := 20
def cars_arriving : ℕ := 14
def suv_adjacent_spaces : ℕ := 2

-- Define the total number of ways to park 14 cars in 20 spaces
def total_ways_to_park : ℕ := Nat.choose parking_lot_spaces cars_arriving
-- Define the number of unfavorable configurations where the SUV cannot park
def unfavorable_configs : ℕ := Nat.choose (parking_lot_spaces - suv_adjacent_spaces + 1) (parking_lot_spaces - cars_arriving)

-- Final probability calculation
def probability_park_suv : ℚ := 1 - (unfavorable_configs / total_ways_to_park)

-- Mathematically equivalent statement to be proved
theorem auntie_em_can_park : probability_park_suv = 850 / 922 :=
by sorry

end auntie_em_can_park_l14_14106


namespace good_oranges_per_month_l14_14590

/-- Salaria has 50% of tree A and 50% of tree B, totaling to 10 trees.
    Tree A gives 10 oranges a month and 60% are good.
    Tree B gives 15 oranges a month and 1/3 are good.
    Prove that the total number of good oranges Salaria gets per month is 55. -/
theorem good_oranges_per_month 
  (total_trees : ℕ) 
  (percent_tree_A : ℝ) 
  (percent_tree_B : ℝ) 
  (oranges_tree_A : ℕ)
  (good_percent_A : ℝ)
  (oranges_tree_B : ℕ)
  (good_ratio_B : ℝ)
  (H1 : total_trees = 10)
  (H2 : percent_tree_A = 0.5)
  (H3 : percent_tree_B = 0.5)
  (H4 : oranges_tree_A = 10)
  (H5 : good_percent_A = 0.6)
  (H6 : oranges_tree_B = 15)
  (H7 : good_ratio_B = 1/3)
  : (total_trees * percent_tree_A * oranges_tree_A * good_percent_A) + 
    (total_trees * percent_tree_B * oranges_tree_B * good_ratio_B) = 55 := 
  by 
    sorry

end good_oranges_per_month_l14_14590


namespace sum_of_ages_l14_14519

-- Define the variables
variables (a b c : ℕ)

-- Define the conditions
def condition1 := a = 16 + b + c
def condition2 := a^2 = 1632 + (b + c)^2

-- Define the theorem to prove the question
theorem sum_of_ages : condition1 a b c → condition2 a b c → a + b + c = 102 := 
by 
  intros h1 h2
  sorry

end sum_of_ages_l14_14519


namespace xyz_not_divisible_by_3_l14_14080

theorem xyz_not_divisible_by_3 (x y z : ℕ) (h1 : x % 2 = 1) (h2 : y % 2 = 1) (h3 : z % 2 = 1) 
  (h4 : Nat.gcd (Nat.gcd x y) z = 1) (h5 : (x^2 + y^2 + z^2) % (x + y + z) = 0) : 
  (x + y + z - 2) % 3 ≠ 0 :=
by
  sorry

end xyz_not_divisible_by_3_l14_14080


namespace find_ab_l14_14152

theorem find_ab (a b c : ℤ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) (h3 : a + b + c = 21) : a * b = 10 := 
sorry

end find_ab_l14_14152


namespace number_of_solutions_l14_14567

theorem number_of_solutions : ∃! (xy : ℕ × ℕ), (xy.1 ^ 2 - xy.2 ^ 2 = 91 ∧ xy.1 > 0 ∧ xy.2 > 0) := sorry

end number_of_solutions_l14_14567


namespace new_monthly_savings_l14_14119

-- Definitions based on conditions
def monthly_salary := 4166.67
def initial_savings_percent := 0.20
def expense_increase_percent := 0.10

-- Calculations
def initial_savings := initial_savings_percent * monthly_salary
def initial_expenses := (1 - initial_savings_percent) * monthly_salary
def increased_expenses := initial_expenses + expense_increase_percent * initial_expenses
def new_savings := monthly_salary - increased_expenses

-- Lean statement to prove the question equals the answer given conditions
theorem new_monthly_savings :
  new_savings = 499.6704 := 
by
  sorry

end new_monthly_savings_l14_14119


namespace fraction_of_difference_l14_14613

theorem fraction_of_difference (A_s A_l : ℝ) (h_total : A_s + A_l = 500) (h_smaller : A_s = 225) :
  (A_l - A_s) / ((A_s + A_l) / 2) = 1 / 5 :=
by
  -- Proof goes here
  sorry

end fraction_of_difference_l14_14613


namespace common_difference_of_arithmetic_sequence_l14_14046

/--
Given an arithmetic sequence {a_n}, the sum of the first n terms is S_n,
a_3 and a_7 are the two roots of the equation 2x^2 - 12x + c = 0,
and S_{13} = c.
Prove that the common difference of the sequence {a_n} satisfies d = -3/2 or d = -7/4.
-/
theorem common_difference_of_arithmetic_sequence 
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (c : ℚ)
  (h1 : ∃ a_3 a_7, (2 * a_3^2 - 12 * a_3 + c = 0) ∧ (2 * a_7^2 - 12 * a_7 + c = 0))
  (h2 : S 13 = c) :
  ∃ d : ℚ, d = -3/2 ∨ d = -7/4 :=
sorry

end common_difference_of_arithmetic_sequence_l14_14046


namespace max_value_l14_14155

-- Define the vector types
structure Vector2 where
  x : ℝ
  y : ℝ

-- Define the properties given in the problem
def a_is_unit_vector (a : Vector2) : Prop :=
  a.x^2 + a.y^2 = 1

def a_plus_b (a b : Vector2) : Prop :=
  a.x + b.x = 3 ∧ a.y + b.y = 4

-- Define dot product for the vectors
def dot_product (a b : Vector2) : ℝ :=
  a.x * b.x + a.y * b.y

-- The theorem statement
theorem max_value (a b : Vector2) (h1 : a_is_unit_vector a) (h2 : a_plus_b a b) :
  ∃ m, m = 5 ∧ ∀ c : ℝ, |1 + dot_product a b| ≤ m :=
  sorry

end max_value_l14_14155


namespace num_words_with_consonant_l14_14247

-- Definitions
def letters : List Char := ['A', 'B', 'C', 'D', 'E']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D']

-- Total number of 4-letter words without restrictions
def total_words : Nat := 5 ^ 4

-- Number of 4-letter words with only vowels
def vowels_only_words : Nat := 2 ^ 4

-- Number of 4-letter words with at least one consonant
def words_with_consonant : Nat := total_words - vowels_only_words

theorem num_words_with_consonant : words_with_consonant = 609 := by
  -- Add proof steps
  sorry

end num_words_with_consonant_l14_14247


namespace last_even_distribution_l14_14967

theorem last_even_distribution (n : ℕ) (h : n = 590490) :
  ∃ k : ℕ, (k ≤ n ∧ (n = 3^k + 3^k + 3^k) ∧ (∀ m : ℕ, m < k → ¬(n = 3^m + 3^m + 3^m))) ∧ k = 1 := 
by 
  sorry

end last_even_distribution_l14_14967


namespace plane_equation_through_points_perpendicular_l14_14374

theorem plane_equation_through_points_perpendicular {M N : ℝ × ℝ × ℝ} (hM : M = (2, -1, 4)) (hN : N = (3, 2, -1)) :
  ∃ A B C d : ℝ, (∀ x y z : ℝ, A * x + B * y + C * z + d = 0 ↔ (x, y, z) = M ∨ (x, y, z) = N ∧ A + B + C = 0) ∧
  (4, -3, -1, -7) = (A, B, C, d) := 
sorry

end plane_equation_through_points_perpendicular_l14_14374


namespace seq_bn_arithmetic_seq_an_formula_sum_an_terms_l14_14759

-- (1) Prove that the sequence {b_n} is an arithmetic sequence
theorem seq_bn_arithmetic (a : ℕ → ℕ) (b : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 2 * a n + 2^n)
  (h3 : ∀ n, b n = a n / 2^(n - 1)) :
  ∀ n, b (n + 1) - b n = 1 := by
  sorry

-- (2) Find the general formula for the sequence {a_n}
theorem seq_an_formula (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 2 * a n + 2^n)
  (h3 : ∀ n, b n = a n / 2^(n - 1)) :
  ∀ n, a n = n * 2^(n - 1) := by
  sorry

-- (3) Find the sum of the first n terms of the sequence {a_n}
theorem sum_an_terms (a : ℕ → ℕ) (S : ℕ → ℤ) (h1 : ∀ n, a n = n * 2^(n - 1)) :
  ∀ n, S n = (n - 1) * 2^n + 1 := by
  sorry

end seq_bn_arithmetic_seq_an_formula_sum_an_terms_l14_14759


namespace total_books_sold_amount_l14_14176

def num_fiction_books := 60
def num_non_fiction_books := 84
def num_children_books := 42

def fiction_books_sold := 3 / 4 * num_fiction_books
def non_fiction_books_sold := 5 / 6 * num_non_fiction_books
def children_books_sold := 2 / 3 * num_children_books

def price_fiction := 5
def price_non_fiction := 7
def price_children := 3

def total_amount_fiction := fiction_books_sold * price_fiction
def total_amount_non_fiction := non_fiction_books_sold * price_non_fiction
def total_amount_children := children_books_sold * price_children

def total_amount_received := total_amount_fiction + total_amount_non_fiction + total_amount_children

theorem total_books_sold_amount :
  total_amount_received = 799 :=
sorry

end total_books_sold_amount_l14_14176


namespace roots_quadratic_sum_l14_14869

theorem roots_quadratic_sum (a b : ℝ) (h1 : (-2) + (-(1/4)) = -b/a)
  (h2 : -2 * (-(1/4)) = -2/a) : a + b = -13 := by
  sorry

end roots_quadratic_sum_l14_14869


namespace rectangular_field_area_l14_14313

noncomputable def a : ℝ := 14
noncomputable def c : ℝ := 17
noncomputable def b := Real.sqrt (c^2 - a^2)
noncomputable def area := a * b

theorem rectangular_field_area : area = 14 * Real.sqrt 93 := by
  sorry

end rectangular_field_area_l14_14313


namespace maximum_possible_savings_is_63_l14_14426

-- Definitions of the conditions
def doughnut_price := 8
def doughnut_discount_2 := 14
def doughnut_discount_4 := 26

def croissant_price := 10
def croissant_discount_3 := 28
def croissant_discount_5 := 45

def muffin_price := 6
def muffin_discount_2 := 11
def muffin_discount_6 := 30

-- Quantities to purchase
def doughnut_qty := 20
def croissant_qty := 15
def muffin_qty := 18

-- Prices calculated from quantities
def total_price_without_discount :=
  doughnut_qty * doughnut_price + croissant_qty * croissant_price + muffin_qty * muffin_price

def total_price_with_discount :=
  5 * doughnut_discount_4 + 3 * croissant_discount_5 + 3 * muffin_discount_6

def maximum_savings := total_price_without_discount - total_price_with_discount

theorem maximum_possible_savings_is_63 : maximum_savings = 63 := by
  -- Proof to be filled in
  sorry

end maximum_possible_savings_is_63_l14_14426


namespace volume_of_defined_region_l14_14448

noncomputable def volume_of_region (x y z : ℝ) : ℝ :=
if x + y ≤ 5 ∧ z ≤ 5 ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x ≤ 2 then 15 else 0

theorem volume_of_defined_region :
  ∀ (x y z : ℝ),
  (0 ≤ x) → (0 ≤ y) → (0 ≤ z) → (x ≤ 2) →
  (|x + y + z| + |x + y - z| ≤ 10) →
  volume_of_region x y z = 15 :=
sorry

end volume_of_defined_region_l14_14448


namespace certain_number_divides_expression_l14_14586

theorem certain_number_divides_expression : 
  ∃ m : ℕ, (∃ n : ℕ, n = 6 ∧ m ∣ (11 * n - 1)) ∧ m = 65 := 
by
  sorry

end certain_number_divides_expression_l14_14586


namespace initial_profit_percentage_l14_14167

-- Definitions of conditions
variables {x y : ℝ} (h1 : y > x) (h2 : 2 * y - x = 1.4 * x)

-- Proof statement in Lean
theorem initial_profit_percentage (x y : ℝ) (h1 : y > x) (h2 : 2 * y - x = 1.4 * x) :
  ((y - x) / x) * 100 = 20 :=
by sorry

end initial_profit_percentage_l14_14167


namespace Emily_spent_28_dollars_l14_14741

theorem Emily_spent_28_dollars :
  let roses_cost := 4
  let daisies_cost := 3
  let tulips_cost := 5
  let lilies_cost := 6
  let roses_qty := 2
  let daisies_qty := 3
  let tulips_qty := 1
  let lilies_qty := 1
  (roses_qty * roses_cost) + (daisies_qty * daisies_cost) + (tulips_qty * tulips_cost) + (lilies_qty * lilies_cost) = 28 :=
by
  sorry

end Emily_spent_28_dollars_l14_14741


namespace cost_of_book_sold_at_loss_l14_14226

theorem cost_of_book_sold_at_loss:
  ∃ (C1 C2 : ℝ), 
    C1 + C2 = 490 ∧ 
    C1 * 0.85 = C2 * 1.19 ∧ 
    C1 = 285.93 :=
by
  sorry

end cost_of_book_sold_at_loss_l14_14226


namespace tan_identity_proof_l14_14220

theorem tan_identity_proof
  (α β : ℝ)
  (h₁ : Real.tan (α + β) = 3)
  (h₂ : Real.tan (α + π / 4) = -3) :
  Real.tan (β - π / 4) = -3 / 4 := 
sorry

end tan_identity_proof_l14_14220


namespace prism_volume_l14_14546

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : a * c = 84) : a * b * c = 1572 :=
by
  sorry

end prism_volume_l14_14546


namespace rope_fold_length_l14_14943

theorem rope_fold_length (L : ℝ) (hL : L = 1) :
  (L / 2 / 2 / 2) = (1 / 8) :=
by
  -- proof steps here
  sorry

end rope_fold_length_l14_14943


namespace coordinates_of_B_l14_14547
open Real

-- Define the conditions given in the problem
def A : ℝ × ℝ := (1, 6)
def d : ℝ := 4

-- Define the properties of the solution given the conditions
theorem coordinates_of_B (B : ℝ × ℝ) :
  (B = (-3, 6) ∨ B = (5, 6)) ↔
  (B.2 = A.2 ∧ (B.1 = A.1 - d ∨ B.1 = A.1 + d)) :=
by
  sorry

end coordinates_of_B_l14_14547


namespace part1_part2_l14_14034

-- Part 1
noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (1/3) * a * x ^ 3 - (1/2) * x ^ 2

noncomputable def f' (x a : ℝ) : ℝ := x * Real.exp x - a * x ^ 2 - x

noncomputable def g (x a : ℝ) : ℝ := f' x a / x

theorem part1 (a : ℝ) (h : a > 0) : g a a > 0 := by
  sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∃ x, f' x a = 0) : a > 0 := by
  sorry

end part1_part2_l14_14034


namespace find_positive_value_of_X_l14_14402

-- define the relation X # Y
def rel (X Y : ℝ) : ℝ := X^2 + Y^2

theorem find_positive_value_of_X (X : ℝ) (h : rel X 7 = 250) : X = Real.sqrt 201 :=
by
  sorry

end find_positive_value_of_X_l14_14402


namespace three_times_greater_than_two_l14_14127

theorem three_times_greater_than_two (x : ℝ) : 3 * x - 2 > 0 → 3 * x > 2 :=
by
  sorry

end three_times_greater_than_two_l14_14127


namespace recommended_apps_l14_14853

namespace RogerPhone

-- Let's define the conditions.
def optimalApps : ℕ := 50
def currentApps (R : ℕ) : ℕ := 2 * R
def appsToDelete : ℕ := 20

-- Defining the problem as a theorem.
theorem recommended_apps (R : ℕ) (h1 : 2 * R = optimalApps + appsToDelete) : R = 35 := by
  sorry

end RogerPhone

end recommended_apps_l14_14853


namespace penultimate_digit_of_quotient_l14_14397

theorem penultimate_digit_of_quotient :
  (4^1994 + 7^1994) / 10 % 10 = 1 :=
by
  sorry

end penultimate_digit_of_quotient_l14_14397


namespace average_nums_correct_l14_14808

def nums : List ℕ := [55, 48, 507, 2, 684, 42]

theorem average_nums_correct :
  (List.sum nums) / (nums.length) = 223 := by
  sorry

end average_nums_correct_l14_14808


namespace three_digit_int_one_less_than_lcm_mult_l14_14209

theorem three_digit_int_one_less_than_lcm_mult : 
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ (n + 1) % Nat.lcm (Nat.lcm (Nat.lcm 3 5) 7) 9 = 0 :=
sorry

end three_digit_int_one_less_than_lcm_mult_l14_14209


namespace solve_inequality_l14_14133

theorem solve_inequality (x : ℝ) (h : 5 * x - 12 ≤ 2 * (4 * x - 3)) : x ≥ -2 :=
sorry

end solve_inequality_l14_14133


namespace ratio_of_volumes_l14_14368

variables (A B : ℚ)

theorem ratio_of_volumes 
  (h1 : (3/8) * A = (5/8) * B) :
  A / B = 5 / 3 :=
sorry

end ratio_of_volumes_l14_14368


namespace area_of_triangle_l14_14177

noncomputable def triangle_area (AB AC θ : ℝ) : ℝ := 
  0.5 * AB * AC * Real.sin θ

theorem area_of_triangle (AB AC : ℝ) (θ : ℝ) (hAB : AB = 1) (hAC : AC = 2) (hθ : θ = 2 * Real.pi / 3) :
  triangle_area AB AC θ = 3 * Real.sqrt 3 / 14 :=
by
  rw [triangle_area, hAB, hAC, hθ]
  sorry

end area_of_triangle_l14_14177


namespace quadratic_has_real_root_for_any_t_l14_14276

theorem quadratic_has_real_root_for_any_t (s : ℝ) :
  (∀ t : ℝ, ∃ x : ℝ, s * x^2 + t * x + s - 1 = 0) ↔ (0 < s ∧ s ≤ 1) :=
by
  sorry

end quadratic_has_real_root_for_any_t_l14_14276


namespace number_of_shelves_l14_14607

theorem number_of_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h_total_books : total_books = 14240) (h_books_per_shelf : books_per_shelf = 8) : total_books / books_per_shelf = 1780 :=
by 
  -- Proof goes here.
  sorry

end number_of_shelves_l14_14607


namespace sin_cos_value_l14_14366

theorem sin_cos_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : (Real.sin x) * (Real.cos x) = 4 / 17 := by
  sorry

end sin_cos_value_l14_14366


namespace max_cos_y_cos_x_l14_14932

noncomputable def max_cos_sum : ℝ :=
  1 + (Real.sqrt (2 + Real.sqrt 2)) / 2

theorem max_cos_y_cos_x
  (x y : ℝ)
  (h1 : Real.sin y + Real.sin x + Real.cos (3 * x) = 0)
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) + Real.cos (2 * x)) :
  ∃ (x y : ℝ), Real.cos y + Real.cos x = max_cos_sum :=
sorry

end max_cos_y_cos_x_l14_14932


namespace B_work_time_l14_14084

noncomputable def workRateA (W : ℝ): ℝ := W / 14
noncomputable def combinedWorkRate (W : ℝ): ℝ := W / 10

theorem B_work_time (W : ℝ) :
  ∃ T : ℝ, (W / T) = (combinedWorkRate W) - (workRateA W) ∧ T = 35 :=
by {
  use 35,
  sorry
}

end B_work_time_l14_14084


namespace max_d_n_is_one_l14_14210

open Int

/-- The sequence definition -/
def seq (n : ℕ) : ℤ := 100 + n^3

/-- The definition of d_n -/
def d_n (n : ℕ) : ℤ := gcd (seq n) (seq (n + 1))

/-- The theorem stating the maximum value of d_n for positive integers is 1 -/
theorem max_d_n_is_one : ∀ (n : ℕ), 1 ≤ n → d_n n = 1 := by
  sorry

end max_d_n_is_one_l14_14210


namespace smallest_x_for_multiple_l14_14101

theorem smallest_x_for_multiple 
  (x : ℕ) (h₁ : ∀ m : ℕ, 450 * x = 800 * m) 
  (h₂ : ∀ y : ℕ, (∀ m : ℕ, 450 * y = 800 * m) → x ≤ y) : 
  x = 16 := 
sorry

end smallest_x_for_multiple_l14_14101


namespace range_of_a_l14_14810

theorem range_of_a (a : ℝ) (x : ℤ) (h1 : ∀ x, x > 0 → ⌊(x + a) / 3⌋ = 2) : a < 8 :=
sorry

end range_of_a_l14_14810


namespace range_of_a_l14_14503

open Set

variable (a : ℝ)

def P(a : ℝ) : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0

def Q(a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (hP : P a) (hQ : Q a) : a ≤ -2 ∨ a = 1 := sorry

end range_of_a_l14_14503


namespace tiffany_bags_on_monday_l14_14953

theorem tiffany_bags_on_monday : 
  ∃ M : ℕ, M = 8 ∧ ∃ T : ℕ, T = 7 ∧ M = T + 1 :=
by
  sorry

end tiffany_bags_on_monday_l14_14953


namespace solve_diff_eq_l14_14129

def solution_of_diff_eq (x y : ℝ) (y' : ℝ → ℝ) : Prop :=
  (x + y) * y' x = 1

def initial_condition (y x : ℝ) : Prop :=
  y = 0 ∧ x = -1

theorem solve_diff_eq (x : ℝ) (y : ℝ) (y' : ℝ → ℝ) (h1 : initial_condition y x) (h2 : solution_of_diff_eq x y y') :
  y = -(x + 1) :=
by 
  sorry

end solve_diff_eq_l14_14129


namespace arithmetic_progression_product_l14_14716

theorem arithmetic_progression_product (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (b : ℕ), (a * (a + d) * (a + 2 * d) * (a + 3 * d) * (a + 4 * d) = b ^ 2008) :=
by
  sorry

end arithmetic_progression_product_l14_14716


namespace average_cars_given_per_year_l14_14500

/-- Definition of initial conditions and the proposition -/
def initial_cars : ℕ := 3500
def final_cars : ℕ := 500
def years : ℕ := 60

theorem average_cars_given_per_year : (initial_cars - final_cars) / years = 50 :=
by
  sorry

end average_cars_given_per_year_l14_14500


namespace paint_for_smaller_statues_l14_14954

open Real

theorem paint_for_smaller_statues :
  ∀ (paint_needed : ℝ) (height_big_statue height_small_statue : ℝ) (num_small_statues : ℝ),
  height_big_statue = 10 → height_small_statue = 2 → paint_needed = 5 → num_small_statues = 200 →
  (paint_needed / (height_big_statue / height_small_statue) ^ 2) * num_small_statues = 40 :=
by
  intros paint_needed height_big_statue height_small_statue num_small_statues
  intros h_big_height h_small_height h_paint_needed h_num_small
  rw [h_big_height, h_small_height, h_paint_needed, h_num_small]
  sorry

end paint_for_smaller_statues_l14_14954


namespace new_cost_percentage_l14_14892

variable (t b : ℝ)

-- Define the original cost
def original_cost : ℝ := t * b ^ 4

-- Define the new cost when b is doubled
def new_cost : ℝ := t * (2 * b) ^ 4

-- The theorem statement
theorem new_cost_percentage (t b : ℝ) : new_cost t b = 16 * original_cost t b := 
by
  -- Proof steps are skipped
  sorry

end new_cost_percentage_l14_14892


namespace largest_fraction_l14_14838

theorem largest_fraction :
  let frac1 := (5 : ℚ) / 12
  let frac2 := (7 : ℚ) / 15
  let frac3 := (23 : ℚ) / 45
  let frac4 := (89 : ℚ) / 178
  let frac5 := (199 : ℚ) / 400
  frac3 > frac1 ∧ frac3 > frac2 ∧ frac3 > frac4 ∧ frac3 > frac5 :=
by
  let frac1 := (5 : ℚ) / 12
  let frac2 := (7 : ℚ) / 15
  let frac3 := (23 : ℚ) / 45
  let frac4 := (89 : ℚ) / 178
  let frac5 := (199 : ℚ) / 400
  sorry

end largest_fraction_l14_14838


namespace greatest_power_of_2_divides_l14_14671

-- Define the conditions as Lean definitions.
def a : ℕ := 15
def b : ℕ := 3
def n : ℕ := 600

-- Define the theorem statement based on the conditions and correct answer.
theorem greatest_power_of_2_divides (x : ℕ) (y : ℕ) (k : ℕ) (h₁ : x = a) (h₂ : y = b) (h₃ : k = n) :
  ∃ m : ℕ, (x^k - y^k) % (2^1200) = 0 ∧ ¬ ∃ m' : ℕ, m' > m ∧ (x^k - y^k) % (2^m') = 0 := sorry

end greatest_power_of_2_divides_l14_14671


namespace subtraction_property_l14_14998

theorem subtraction_property : (12.56 - (5.56 - 2.63)) = (12.56 - 5.56 + 2.63) := 
by 
  sorry

end subtraction_property_l14_14998


namespace sin_6phi_l14_14449

theorem sin_6phi (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * (Real.sqrt 8)) / 5) : 
  Real.sin (6 * φ) = -198 * Real.sqrt 2 / 15625 :=
by
  sorry

end sin_6phi_l14_14449


namespace investor_share_purchase_price_l14_14231

theorem investor_share_purchase_price 
  (dividend_rate : ℝ) 
  (face_value : ℝ) 
  (roi : ℝ) 
  (purchase_price : ℝ)
  (h1 : dividend_rate = 0.125)
  (h2 : face_value = 60)
  (h3 : roi = 0.25)
  (h4 : 0.25 = (0.125 * 60) / purchase_price) 
  : purchase_price = 30 := 
sorry

end investor_share_purchase_price_l14_14231


namespace books_added_after_lunch_l14_14314

-- Definitions for the given conditions
def initial_books : Int := 100
def books_borrowed_lunch : Int := 50
def books_remaining_lunch : Int := initial_books - books_borrowed_lunch
def books_borrowed_evening : Int := 30
def books_remaining_evening : Int := 60

-- Let X be the number of books added after lunchtime
variable (X : Int)

-- The proof goal in Lean statement
theorem books_added_after_lunch (h : books_remaining_lunch + X - books_borrowed_evening = books_remaining_evening) :
  X = 40 := by
  sorry

end books_added_after_lunch_l14_14314


namespace range_of_z_l14_14189

theorem range_of_z (x y : ℝ) (h : x^2 + 2 * x * y + 4 * y^2 = 6) :
  4 ≤ x^2 + 4 * y^2 ∧ x^2 + 4 * y^2 ≤ 12 :=
by
  sorry

end range_of_z_l14_14189


namespace alexandra_magazines_l14_14574

noncomputable def magazines (bought_on_friday : ℕ) (bought_on_saturday : ℕ) (times_friday : ℕ) (chewed_up : ℕ) : ℕ :=
  bought_on_friday + bought_on_saturday + times_friday * bought_on_friday - chewed_up

theorem alexandra_magazines :
  ∀ (bought_on_friday bought_on_saturday times_friday chewed_up : ℕ),
      bought_on_friday = 8 → 
      bought_on_saturday = 12 → 
      times_friday = 4 → 
      chewed_up = 4 →
      magazines bought_on_friday bought_on_saturday times_friday chewed_up = 48 :=
by
  intros
  sorry

end alexandra_magazines_l14_14574


namespace weighted_average_plants_per_hour_l14_14043

theorem weighted_average_plants_per_hour :
  let heath_carrot_plants_100 := 100 * 275
  let heath_carrot_plants_150 := 150 * 325
  let heath_total_plants := heath_carrot_plants_100 + heath_carrot_plants_150
  let heath_total_time := 10 + 20
  
  let jake_potato_plants_50 := 50 * 300
  let jake_potato_plants_100 := 100 * 400
  let jake_total_plants := jake_potato_plants_50 + jake_potato_plants_100
  let jake_total_time := 12 + 18

  let total_plants := heath_total_plants + jake_total_plants
  let total_time := heath_total_time + jake_total_time
  let weighted_average := total_plants / total_time
  weighted_average = 2187.5 :=
by
  sorry

end weighted_average_plants_per_hour_l14_14043


namespace distance_NYC_to_DC_l14_14564

noncomputable def horse_speed := 10 -- miles per hour
noncomputable def travel_time := 24 -- hours

theorem distance_NYC_to_DC : horse_speed * travel_time = 240 := by
  sorry

end distance_NYC_to_DC_l14_14564


namespace parabola_min_area_l14_14924

-- Definition of the parabola C with vertex at the origin and focus on the positive y-axis
-- (Conditions 1 and 2)
def parabola_eq (x y : ℝ) : Prop := x^2 = 6 * y

-- Line l defined by mx + y - 3/2 = 0 (Condition 3)
def line_eq (m x y : ℝ) : Prop := m * x + y - 3 / 2 = 0

-- Formal statement combining all conditions to prove the equivalent Lean statement
theorem parabola_min_area :
  (∀ x y : ℝ, parabola_eq x y ↔ x^2 = 6 * y) ∧
  (∀ m x y : ℝ, line_eq m x y ↔ m * x + y - 3 / 2 = 0) →
  (parabola_eq 0 0) ∧ (∃ y > 0, parabola_eq 0 y ∧ line_eq 0 0 (y/2) ∧ y = 3 / 2) ∧
  ∀ A B P : ℝ, line_eq 0 A B ∧ line_eq 0 B P ∧ A^2 + B^2 > 0 → 
  ∃ min_S : ℝ, min_S = 9 :=
by
  sorry

end parabola_min_area_l14_14924


namespace eval_polynomial_at_3_l14_14893

theorem eval_polynomial_at_3 : (3 : ℤ) ^ 3 + (3 : ℤ) ^ 2 + 3 + 1 = 40 := by
  sorry

end eval_polynomial_at_3_l14_14893


namespace find_g_9_l14_14879

-- Define the function g
def g (a b c x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 7

-- Given conditions
variables (a b c : ℝ)

-- g(-9) = 9
axiom h : g a b c (-9) = 9

-- Prove g(9) = -23
theorem find_g_9 : g a b c 9 = -23 :=
by
  sorry

end find_g_9_l14_14879


namespace electrical_appliance_supermarket_l14_14305

-- Define the known quantities and conditions
def purchase_price_A : ℝ := 140
def purchase_price_B : ℝ := 100
def week1_sales_A : ℕ := 4
def week1_sales_B : ℕ := 3
def week1_revenue : ℝ := 1250
def week2_sales_A : ℕ := 5
def week2_sales_B : ℕ := 5
def week2_revenue : ℝ := 1750
def total_units : ℕ := 50
def budget : ℝ := 6500
def profit_goal : ℝ := 2850

-- Define the unknown selling prices
noncomputable def selling_price_A : ℝ := 200
noncomputable def selling_price_B : ℝ := 150

-- Define the constraints
def cost_constraint (m : ℕ) : Prop := 140 * m + 100 * (50 - m) ≤ 6500
def profit_exceeds_goal (m : ℕ) : Prop := (200 - 140) * m + (150 - 100) * (50 - m) > 2850

-- The main theorem stating the results
theorem electrical_appliance_supermarket :
  (4 * selling_price_A + 3 * selling_price_B = week1_revenue)
  ∧ (5 * selling_price_A + 5 * selling_price_B = week2_revenue)
  ∧ (∃ m : ℕ, m ≤ 37 ∧ cost_constraint m)
  ∧ (∃ m : ℕ, m > 35 ∧ m ≤ 37 ∧ profit_exceeds_goal m) :=
sorry

end electrical_appliance_supermarket_l14_14305


namespace find_natural_numbers_l14_14333

theorem find_natural_numbers (n : ℕ) :
  (∀ k : ℕ, k^2 + ⌊ (n : ℝ) / (k^2 : ℝ) ⌋ ≥ 1991) ∧
  (∃ k_0 : ℕ, k_0^2 + ⌊ (n : ℝ) / (k_0^2 : ℝ) ⌋ < 1992) ↔
  990208 ≤ n ∧ n ≤ 991231 :=
by sorry

end find_natural_numbers_l14_14333


namespace total_weight_of_hay_bales_l14_14212

theorem total_weight_of_hay_bales
  (initial_bales : Nat) (weight_per_initial_bale : Nat)
  (total_bales_now : Nat) (weight_per_new_bale : Nat) : 
  (initial_bales = 73 ∧ weight_per_initial_bale = 45 ∧ 
   total_bales_now = 96 ∧ weight_per_new_bale = 50) →
  (73 * 45 + (96 - 73) * 50 = 4435) :=
by
  sorry

end total_weight_of_hay_bales_l14_14212


namespace product_of_numbers_l14_14082

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x^3 + y^3 = 9450) : x * y = -585 :=
  sorry

end product_of_numbers_l14_14082


namespace second_quadrant_implies_value_of_m_l14_14539

theorem second_quadrant_implies_value_of_m (m : ℝ) : 4 - m < 0 → m = 5 := by
  intro h
  have ineq : m > 4 := by
    linarith
  sorry

end second_quadrant_implies_value_of_m_l14_14539


namespace least_possible_number_of_coins_in_jar_l14_14678

theorem least_possible_number_of_coins_in_jar (n : ℕ) : 
  (n % 7 = 3) → (n % 4 = 1) → (n % 6 = 5) → n = 17 :=
by
  sorry

end least_possible_number_of_coins_in_jar_l14_14678


namespace minimum_value_l14_14694

noncomputable def minimum_y_over_2x_plus_1_over_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : ℝ :=
  (y / (2 * x)) + (1 / y)

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) :
  minimum_y_over_2x_plus_1_over_y x y hx hy h = 2 + Real.sqrt 2 :=
sorry

end minimum_value_l14_14694


namespace prove_positive_a_l14_14840

variable (a b c n : ℤ)
variable (p : ℤ → ℤ)

-- Conditions given in the problem
def quadratic_polynomial (x : ℤ) : ℤ := a*x^2 + b*x + c

def condition_1 : Prop := a ≠ 0
def condition_2 : Prop := n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))

-- Proof goal
theorem prove_positive_a (h1 : a ≠ 0) (h2 : n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))) :
  0 < a :=
by
  sorry

end prove_positive_a_l14_14840


namespace number_of_integer_solutions_l14_14732

theorem number_of_integer_solutions : ∃ (n : ℕ), n = 120 ∧ ∀ (x y z : ℤ), x * y * z = 2008 → n = 120 :=
by
  sorry

end number_of_integer_solutions_l14_14732


namespace power_function_increasing_l14_14865

   theorem power_function_increasing (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x < y → x^a < y^a) : 0 < a :=
   by
   sorry
   
end power_function_increasing_l14_14865


namespace find_fx_when_x_positive_l14_14882

def isOddFunction {α : Type} [AddGroup α] [Neg α] (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

variable (f : ℝ → ℝ)
variable (h_odd : isOddFunction f)
variable (h_neg : ∀ x : ℝ, x < 0 → f x = -x^2 + x)

theorem find_fx_when_x_positive : ∀ x : ℝ, x > 0 → f x = x^2 + x :=
by
  sorry

end find_fx_when_x_positive_l14_14882


namespace find_N_mod_inverse_l14_14709

-- Definitions based on given conditions
def A := 111112
def B := 142858
def M := 1000003
def AB : Nat := (A * B) % M
def N := 513487

-- Statement to prove
theorem find_N_mod_inverse : (711812 * N) % M = 1 := by
  -- Proof skipped as per instruction
  sorry

end find_N_mod_inverse_l14_14709


namespace net_change_is_minus_0_19_l14_14330

-- Define the yearly change factors as provided in the conditions
def yearly_changes : List ℚ := [6/5, 11/10, 7/10, 4/5, 11/10]

-- Compute the net change over the five years
def net_change (changes : List ℚ) : ℚ :=
  changes.foldl (λ acc x => acc * x) 1 - 1

-- Define the target value for the net change
def target_net_change : ℚ := -19 / 100

-- The theorem to prove the net change calculated matches the target net change
theorem net_change_is_minus_0_19 : net_change yearly_changes = target_net_change :=
  by
    sorry

end net_change_is_minus_0_19_l14_14330


namespace min_value_expression_l14_14772

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 3) :
  ∃ (M : ℝ), M = (2 : ℝ) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → ((y / x) + (3 / (y + 1)) ≥ M)) :=
by
  use 2
  sorry

end min_value_expression_l14_14772


namespace cos_pi_minus_2alpha_l14_14012

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (Real.pi / 2 - α) = Real.sqrt 2 / 3) : 
  Real.cos (Real.pi - 2 * α) = -5 / 9 := by
  sorry

end cos_pi_minus_2alpha_l14_14012


namespace no_valid_n_exists_l14_14730

theorem no_valid_n_exists :
  ¬ ∃ n : ℕ, 219 ≤ n ∧ n ≤ 2019 ∧ ∃ x y : ℕ, 
    1 ≤ x ∧ x < n ∧ n < y ∧ (∀ k : ℕ, k ≤ n → k ≠ x ∧ k ≠ x+1 → y % k = 0) := 
by {
  sorry
}

end no_valid_n_exists_l14_14730


namespace percentage_increase_l14_14360

theorem percentage_increase :
  let original_employees := 852
  let new_employees := 1065
  let increase := new_employees - original_employees
  let percentage := (increase.toFloat / original_employees.toFloat) * 100
  percentage = 25 := 
by 
  sorry

end percentage_increase_l14_14360


namespace circle_radius_twice_value_l14_14362

theorem circle_radius_twice_value (r_x r_y v : ℝ) (h1 : π * r_x^2 = π * r_y^2)
  (h2 : 2 * π * r_x = 12 * π) (h3 : r_y = 2 * v) : v = 3 := by
  sorry

end circle_radius_twice_value_l14_14362


namespace melanie_average_speed_l14_14611

theorem melanie_average_speed
  (bike_distance run_distance total_time : ℝ)
  (h_bike : bike_distance = 15)
  (h_run : run_distance = 5)
  (h_time : total_time = 4) :
  (bike_distance + run_distance) / total_time = 5 :=
by
  sorry

end melanie_average_speed_l14_14611


namespace log_sqrt_7_of_343sqrt7_l14_14405

noncomputable def log_sqrt_7 (y : ℝ) : ℝ := 
  Real.log y / Real.log (Real.sqrt 7)

theorem log_sqrt_7_of_343sqrt7 : log_sqrt_7 (343 * Real.sqrt 7) = 4 :=
by
  sorry

end log_sqrt_7_of_343sqrt7_l14_14405


namespace parallel_lines_m_values_l14_14888

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (m-2) * x - y - 1 = 0) ∧ (∀ x y : ℝ, 3 * x - m * y = 0) → 
  (m = -1 ∨ m = 3) :=
by
  sorry

end parallel_lines_m_values_l14_14888


namespace length_sawed_off_l14_14697

-- Define the lengths as constants
def original_length : ℝ := 8.9
def final_length : ℝ := 6.6

-- State the property to be proven
theorem length_sawed_off : original_length - final_length = 2.3 := by
  sorry

end length_sawed_off_l14_14697


namespace angle_z_value_l14_14704

theorem angle_z_value
  (ABC BAC : ℝ)
  (h1 : ABC = 70)
  (h2 : BAC = 50)
  (h3 : ∀ BCA : ℝ, BCA + ABC + BAC = 180) :
  ∃ z : ℝ, z = 30 :=
by
  sorry

end angle_z_value_l14_14704


namespace discount_percentage_correct_l14_14110

-- Define the problem parameters as variables
variables (sale_price marked_price : ℝ) (discount_percentage : ℝ)

-- Provide the conditions from the problem
def conditions : Prop :=
  sale_price = 147.60 ∧ marked_price = 180

-- State the problem: Prove the discount percentage is 18%
theorem discount_percentage_correct (h : conditions sale_price marked_price) : 
  discount_percentage = 18 :=
by
  sorry

end discount_percentage_correct_l14_14110


namespace students_more_than_pets_l14_14280

theorem students_more_than_pets :
  let students_per_classroom := 15
  let rabbits_per_classroom := 1
  let guinea_pigs_per_classroom := 3
  let number_of_classrooms := 6
  let total_students := students_per_classroom * number_of_classrooms
  let total_pets := (rabbits_per_classroom + guinea_pigs_per_classroom) * number_of_classrooms
  total_students - total_pets = 66 :=
by
  sorry

end students_more_than_pets_l14_14280


namespace three_zeros_condition_l14_14005

noncomputable def f (ω : ℝ) (x : ℝ) := Real.sin (ω * x) + Real.cos (ω * x)

theorem three_zeros_condition (ω : ℝ) (hω : ω > 0) :
  (∃ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ 2 * Real.pi ∧
  f ω x1 = 0 ∧ f ω x2 = 0 ∧ f ω x3 = 0) →
  (∀ ω, (11 / 8 : ℝ) ≤ ω ∧ ω < (15 / 8 : ℝ) ∧
  (∀ x, f ω x = 0 ↔ x = (5 * Real.pi) / (4 * ω))) :=
sorry

end three_zeros_condition_l14_14005


namespace find_n_l14_14790

-- Define x and y
def x : ℕ := 3
def y : ℕ := 1

-- Define n based on the given expression.
def n : ℕ := x - y^(x - (y + 1))

-- State the theorem
theorem find_n : n = 2 := by
  sorry

end find_n_l14_14790


namespace total_quantities_l14_14950

theorem total_quantities (n S S₃ S₂ : ℕ) (h₁ : S = 6 * n) (h₂ : S₃ = 4 * 3) (h₃ : S₂ = 33 * 2) (h₄ : S = S₃ + S₂) : n = 13 :=
by
  sorry

end total_quantities_l14_14950


namespace find_ab_solutions_l14_14999

theorem find_ab_solutions (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (h1 : (a + 1) ∣ (a ^ 3 * b - 1))
  (h2 : (b - 1) ∣ (b ^ 3 * a + 1)) : 
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) :=
sorry

end find_ab_solutions_l14_14999


namespace meetings_percentage_l14_14856

/-- Define the total work day in hours -/
def total_work_day_hours : ℕ := 10

/-- Define the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 60 -- 1 hour = 60 minutes

/-- Define the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 75

/-- Define the break duration in minutes -/
def break_minutes : ℕ := 30

/-- Define the effective work minutes -/
def effective_work_minutes : ℕ := (total_work_day_hours * 60) - break_minutes

/-- Define the total meeting minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- The percentage of the effective work day spent in meetings -/
def percent_meetings : ℕ := (total_meeting_minutes * 100) / effective_work_minutes

theorem meetings_percentage : percent_meetings = 24 := by
  sorry

end meetings_percentage_l14_14856


namespace find_x_coordinate_l14_14847

theorem find_x_coordinate (m b x y : ℝ) (h1: m = 4) (h2: b = 100) (h3: y = 300) (line_eq: y = m * x + b) : x = 50 :=
by {
  sorry
}

end find_x_coordinate_l14_14847


namespace sherry_needs_bananas_l14_14076

/-
Conditions:
- Sherry wants to make 99 loaves.
- Her recipe makes enough batter for 3 loaves.
- The recipe calls for 1 banana per batch of 3 loaves.

Question:
- How many bananas does Sherry need?

Equivalent Proof Problem:
- Prove that given the conditions, the number of bananas needed is 33.
-/

def total_loaves : ℕ := 99
def loaves_per_batch : ℕ := 3
def bananas_per_batch : ℕ := 1

theorem sherry_needs_bananas :
  (total_loaves / loaves_per_batch) * bananas_per_batch = 33 :=
sorry

end sherry_needs_bananas_l14_14076


namespace wood_rope_equations_l14_14248

theorem wood_rope_equations (x y : ℝ) (h1 : y - x = 4.5) (h2 : 0.5 * y = x - 1) :
  (y - x = 4.5) ∧ (0.5 * y = x - 1) :=
by
  sorry

end wood_rope_equations_l14_14248


namespace bowling_ball_weight_l14_14857

def weight_of_canoe : ℕ := 32
def weight_of_canoes (n : ℕ) := n * weight_of_canoe
def weight_of_bowling_balls (n : ℕ) := 128

theorem bowling_ball_weight :
  (128 / 5 : ℚ) = (weight_of_bowling_balls 5 / 5 : ℚ) :=
by
  -- Theorems and calculations would typically be carried out here
  sorry

end bowling_ball_weight_l14_14857


namespace bushes_needed_l14_14063

theorem bushes_needed
  (num_sides : ℕ) (side_length : ℝ) (bush_fill : ℝ) (total_length : ℝ) (num_bushes : ℕ) :
  num_sides = 3 ∧ side_length = 16 ∧ bush_fill = 4 ∧ total_length = num_sides * side_length ∧ num_bushes = total_length / bush_fill →
  num_bushes = 12 := by
  sorry

end bushes_needed_l14_14063


namespace ice_cream_eaten_l14_14819

variables (f : ℝ)

theorem ice_cream_eaten (h : f + 0.25 = 3.5) : f = 3.25 :=
sorry

end ice_cream_eaten_l14_14819


namespace area_ratio_of_squares_l14_14824

theorem area_ratio_of_squares (s t : ℝ) (h : 4 * s = 4 * (4 * t)) : (s ^ 2) / (t ^ 2) = 16 :=
by
  sorry

end area_ratio_of_squares_l14_14824


namespace train_speed_l14_14121

-- Define the conditions as given in part (a)
def train_length : ℝ := 160
def crossing_time : ℝ := 6

-- Define the statement to prove
theorem train_speed :
  train_length / crossing_time = 26.67 :=
by
  sorry

end train_speed_l14_14121


namespace sum_even_102_to_600_l14_14169

def sum_first_50_even : ℕ := 2550
def sum_even_602_to_700 : ℕ := 32550

theorem sum_even_102_to_600 : sum_even_602_to_700 - sum_first_50_even = 30000 :=
by
  -- The given sum of the first 50 positive even integers is 2550
  have h1 : sum_first_50_even = 2550 := by rfl
  
  -- The given sum of the even integers from 602 to 700 inclusive is 32550
  have h2 : sum_even_602_to_700 = 32550 := by rfl
  
  -- Therefore, the sum of the even integers from 102 to 600 is:
  have h3 : sum_even_602_to_700 - sum_first_50_even = 32550 - 2550 := by
    rw [h1, h2]
  
  -- Calculate the result
  exact h3

end sum_even_102_to_600_l14_14169


namespace tunnel_length_proof_l14_14056

variable (train_length : ℝ) (train_speed : ℝ) (time_in_tunnel : ℝ)

noncomputable def tunnel_length (train_length train_speed time_in_tunnel : ℝ) : ℝ :=
  (train_speed / 60) * time_in_tunnel - train_length

theorem tunnel_length_proof 
  (h_train_length : train_length = 2) 
  (h_train_speed : train_speed = 30) 
  (h_time_in_tunnel : time_in_tunnel = 4) : 
  tunnel_length 2 30 4 = 2 := by
    simp [tunnel_length, h_train_length, h_train_speed, h_time_in_tunnel]
    norm_num
    sorry

end tunnel_length_proof_l14_14056


namespace sum_of_distinct_selections_is_34_l14_14328

-- Define a 4x4 grid filled sequentially from 1 to 16
def grid : List (List ℕ) := [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16]
]

-- Define a type for selections from the grid ensuring distinct rows and columns.
structure Selection where
  row : ℕ
  col : ℕ
  h_row : row < 4
  h_col : col < 4

-- Define the sum of any selection of 4 numbers from distinct rows and columns in the grid.
def sum_of_selection (selections : List Selection) : ℕ :=
  if h : List.length selections = 4 then
    List.sum (List.map (λ sel => (grid.get! sel.row).get! sel.col) selections)
  else 0

-- The main theorem
theorem sum_of_distinct_selections_is_34 (selections : List Selection) 
  (h_distinct_rows : List.Nodup (List.map (λ sel => sel.row) selections))
  (h_distinct_cols : List.Nodup (List.map (λ sel => sel.col) selections)) :
  sum_of_selection selections = 34 :=
by
  -- Proof is omitted
  sorry

end sum_of_distinct_selections_is_34_l14_14328


namespace hypotenuse_unique_l14_14001

theorem hypotenuse_unique (a b : ℝ) (h: ∃ x : ℝ, x^2 = a^2 + b^2 ∧ x > 0) : 
  ∃! c : ℝ, c^2 = a^2 + b^2 :=
sorry

end hypotenuse_unique_l14_14001


namespace compute_binom_product_l14_14845

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.choose k

/-- The main theorem to prove -/
theorem compute_binom_product : binomial 10 3 * binomial 8 3 = 6720 :=
by
  sorry

end compute_binom_product_l14_14845


namespace truck_gas_consumption_l14_14885

theorem truck_gas_consumption :
  ∀ (initial_gasoline total_distance remaining_gasoline : ℝ),
    initial_gasoline = 12 →
    total_distance = (2 * 5 + 2 + 2 * 2 + 6) →
    remaining_gasoline = 2 →
    (initial_gasoline - remaining_gasoline) ≠ 0 →
    (total_distance / (initial_gasoline - remaining_gasoline)) = 2.2 :=
by
  intros initial_gasoline total_distance remaining_gasoline
  intro h_initial_gas h_total_distance h_remaining_gas h_non_zero
  sorry

end truck_gas_consumption_l14_14885


namespace distance_walked_by_friend_P_l14_14615

def trail_length : ℝ := 33
def speed_ratio : ℝ := 1.20

theorem distance_walked_by_friend_P (v t d_P : ℝ) 
  (h1 : t = 33 / (2.20 * v)) 
  (h2 : d_P = 1.20 * v * t) 
  : d_P = 18 := by
  sorry

end distance_walked_by_friend_P_l14_14615


namespace find_pq_of_orthogonal_and_equal_magnitudes_l14_14661

noncomputable def vec_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
noncomputable def vec_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_pq_of_orthogonal_and_equal_magnitudes
    (p q : ℝ)
    (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
    (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
    (p, q) = (-29/12, 43/12) :=
by {
  sorry
}

end find_pq_of_orthogonal_and_equal_magnitudes_l14_14661


namespace quadratic_rewriting_l14_14791

theorem quadratic_rewriting (b n : ℝ) (h₁ : 0 < n)
  (h₂ : ∀ x : ℝ, x^2 + b*x + 72 = (x + n)^2 + 20) :
  b = 4 * Real.sqrt 13 :=
by
  sorry

end quadratic_rewriting_l14_14791


namespace find_k_series_sum_l14_14099

theorem find_k_series_sum :
  (∃ k : ℝ, 5 + ∑' n : ℕ, ((5 + (n + 1) * k) / 5^n.succ) = 10) →
  k = 12 :=
sorry

end find_k_series_sum_l14_14099


namespace find_breadth_l14_14323

theorem find_breadth (p l : ℕ) (h_p : p = 600) (h_l : l = 100) (h_perimeter : p = 2 * (l + b)) : b = 200 :=
by
  sorry

end find_breadth_l14_14323


namespace larger_number_l14_14612

theorem larger_number (x y : ℕ) (h₁ : x + y = 27) (h₂ : x - y = 5) : x = 16 :=
by sorry

end larger_number_l14_14612


namespace intersection_M_N_l14_14190

/-- Define the set M as pairs (x, y) such that x + y = 2. -/
def M : Set (ℝ × ℝ) := { p | p.1 + p.2 = 2 }

/-- Define the set N as pairs (x, y) such that x - y = 2. -/
def N : Set (ℝ × ℝ) := { p | p.1 - p.2 = 2 }

/-- The intersection of sets M and N is the single point (2, 0). -/
theorem intersection_M_N : M ∩ N = { (2, 0) } :=
by
  sorry

end intersection_M_N_l14_14190


namespace geometric_sequence_relation_l14_14764

theorem geometric_sequence_relation (a b c : ℝ) (r : ℝ)
  (h1 : -2 * r = a)
  (h2 : a * r = b)
  (h3 : b * r = c)
  (h4 : c * r = -8) :
  b = -4 ∧ a * c = 16 := by
  sorry

end geometric_sequence_relation_l14_14764


namespace men_in_hotel_l14_14050

theorem men_in_hotel (n : ℕ) (A : ℝ) (h1 : 8 * 3 = 24)
  (h2 : A = 32.625 / n)
  (h3 : 24 + (A + 5) = 32.625) :
  n = 9 := 
  by
  sorry

end men_in_hotel_l14_14050


namespace prove_additional_minutes_needed_l14_14206

-- Assume the given conditions as definitions in Lean 4
def number_of_classmates := 30
def initial_gathering_time := 120   -- in minutes (2 hours)
def time_per_flower := 10           -- in minutes
def flowers_lost := 3

-- Calculate the flowers gathered initially
def initial_flowers_gathered := initial_gathering_time / time_per_flower

-- Calculate flowers remaining after loss
def flowers_remaining := initial_flowers_gathered - flowers_lost

-- Calculate additional flowers needed
def additional_flowers_needed := number_of_classmates - flowers_remaining

-- Therefore, calculate the additional minutes required to gather the remaining flowers
def additional_minutes_needed := additional_flowers_needed * time_per_flower

theorem prove_additional_minutes_needed :
  additional_minutes_needed = 210 :=
by 
  unfold additional_minutes_needed additional_flowers_needed flowers_remaining initial_flowers_gathered
  sorry

end prove_additional_minutes_needed_l14_14206


namespace all_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_l14_14072

theorem all_palindromes_divisible_by_11 : 
  (∀ a b : ℕ, 1 <= a ∧ a <= 9 ∧ 0 <= b ∧ b <= 9 →
    (1001 * a + 110 * b) % 11 = 0 ) := sorry

theorem probability_palindrome_divisible_by_11 : 
  (∀ (palindromes : ℕ → Prop), 
  (∀ n, palindromes n ↔ ∃ (a b : ℕ), 
  1 <= a ∧ a <= 9 ∧ 0 <= b ∧ b <= 9 ∧ 
  n = 1001 * a + 110 * b) → 
  (∀ n, palindromes n → n % 11 = 0) →
  ∃ p : ℝ, p = 1) := sorry

end all_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_l14_14072


namespace tetrahedron_area_theorem_l14_14488

noncomputable def tetrahedron_faces_areas_and_angles
  (a b c d : ℝ) (α β γ : ℝ) : Prop :=
  d^2 = a^2 + b^2 + c^2 - 2 * a * b * Real.cos γ - 2 * b * c * Real.cos α - 2 * c * a * Real.cos β

theorem tetrahedron_area_theorem
  (a b c d : ℝ) (α β γ : ℝ) :
  tetrahedron_faces_areas_and_angles a b c d α β γ :=
sorry

end tetrahedron_area_theorem_l14_14488


namespace quadratic_real_roots_l14_14424

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k^2 * x^2 - (2 * k + 1) * x + 1 = 0 ∧ ∃ x2 : ℝ, k^2 * x2^2 - (2 * k + 1) * x2 + 1 = 0)
  ↔ (k ≥ -1/4 ∧ k ≠ 0) := 
by 
  sorry

end quadratic_real_roots_l14_14424


namespace line_intersection_l14_14696

-- Definitions for the parametric lines
def line1 (t : ℝ) : ℝ × ℝ := (3 + t, 2 * t)
def line2 (u : ℝ) : ℝ × ℝ := (-1 + 3 * u, 4 - u)

-- Statement that expresses the intersection point condition
theorem line_intersection :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (30 / 7, 18 / 7) :=
by
  sorry

end line_intersection_l14_14696


namespace volume_relation_l14_14751

variable {x y z V : ℝ}

theorem volume_relation
  (top_area : x * y = A)
  (side_area : y * z = B)
  (volume : x * y * z = V) :
  (y * z) * (x * y * z)^2 = z^3 * V := by
  sorry

end volume_relation_l14_14751


namespace average_marks_l14_14993

-- Define the conditions
variables (M P C : ℕ)
axiom condition1 : M + P = 30
axiom condition2 : C = P + 20

-- Define the target statement
theorem average_marks : (M + C) / 2 = 25 :=
by
  sorry

end average_marks_l14_14993


namespace find_N_l14_14526

theorem find_N (x y N : ℝ) (h1 : 2 * x + y = N) (h2 : x + 2 * y = 5) (h3 : (x + y) / 3 = 1) : N = 4 :=
by
  have h4 : x + y = 3 := by
    linarith [h3]
  have h5 : y = 3 - x := by
    linarith [h4]
  have h6 : x + 2 * (3 - x) = 5 := by
    linarith [h2, h5]
  have h7 : x = 1 := by
    linarith [h6]
  have h8 : y = 2 := by
    linarith [h4, h7]
  have h9 : 2 * x + y = 4 := by
    linarith [h7, h8]
  linarith [h1, h9]

end find_N_l14_14526


namespace girls_count_in_leos_class_l14_14462

def leo_class_girls_count (g b : ℕ) :=
  (g / b = 3 / 4) ∧ (g + b = 35) → g = 15

theorem girls_count_in_leos_class (g b : ℕ) :
  leo_class_girls_count g b :=
by
  sorry

end girls_count_in_leos_class_l14_14462


namespace first_class_product_rate_l14_14370

theorem first_class_product_rate
  (total_products : ℕ)
  (pass_rate : ℝ)
  (first_class_rate_among_qualified : ℝ)
  (pass_rate_correct : pass_rate = 0.95)
  (first_class_rate_correct : first_class_rate_among_qualified = 0.2) :
  (first_class_rate_among_qualified * pass_rate : ℝ) = 0.19 :=
by
  rw [pass_rate_correct, first_class_rate_correct]
  norm_num


end first_class_product_rate_l14_14370


namespace total_interest_at_tenth_year_l14_14057

-- Define the conditions for the simple interest problem
variables (P R T : ℝ)

-- Given conditions in the problem
def initial_condition : Prop := (P * R * 10) / 100 = 800
def trebled_principal_condition : Prop := (3 * P * R * 5) / 100 = 1200

-- Statement to prove
theorem total_interest_at_tenth_year (h1 : initial_condition P R) (h2 : trebled_principal_condition P R) :
  (800 + 1200) = 2000 := by
  sorry

end total_interest_at_tenth_year_l14_14057


namespace tank_base_length_width_difference_l14_14357

variable (w l h : ℝ)

theorem tank_base_length_width_difference :
  (l = 5 * w) →
  (h = (1/2) * w) →
  (l * w * h = 3600) →
  (|l - w - 45.24| < 0.01) := 
by
  sorry

end tank_base_length_width_difference_l14_14357


namespace weight_removed_l14_14086

-- Definitions for the given conditions
def weight_sugar : ℕ := 16
def weight_salt : ℕ := 30
def new_combined_weight : ℕ := 42

-- The proof problem statement
theorem weight_removed : (weight_sugar + weight_salt) - new_combined_weight = 4 := by
  -- Proof will be provided here
  sorry

end weight_removed_l14_14086


namespace find_b_l14_14451

theorem find_b (b p : ℝ) (h_factor : ∃ k : ℝ, 3 * (x^3 : ℝ) + b * x + 9 = (x^2 + p * x + 3) * (k * x + k)) :
  b = -6 :=
by
  obtain ⟨k, h_eq⟩ := h_factor
  sorry

end find_b_l14_14451


namespace ladder_of_twos_l14_14599

theorem ladder_of_twos (n : ℕ) (h : n ≥ 3) : 
  ∃ N_n : ℕ, N_n = 2 ^ (n - 3) :=
by
  sorry

end ladder_of_twos_l14_14599


namespace seq_diff_five_consec_odd_avg_55_l14_14303

theorem seq_diff_five_consec_odd_avg_55 {a b c d e : ℤ} 
    (h1: a % 2 = 1) (h2: b % 2 = 1) (h3: c % 2 = 1) (h4: d % 2 = 1) (h5: e % 2 = 1)
    (h6: b = a + 2) (h7: c = a + 4) (h8: d = a + 6) (h9: e = a + 8)
    (avg_5_seq : (a + b + c + d + e) / 5 = 55) : 
    e - a = 8 := 
by
    -- proof part can be skipped with sorry
    sorry

end seq_diff_five_consec_odd_avg_55_l14_14303


namespace ExpandedOHaraTripleValue_l14_14900

/-- Define an Expanded O'Hara triple -/
def isExpandedOHaraTriple (a b x : ℕ) : Prop :=
  2 * (Nat.sqrt a + Nat.sqrt b) = x

/-- Prove that for given a=64 and b=49, x is equal to 30 if (a, b, x) is an Expanded O'Hara triple -/
theorem ExpandedOHaraTripleValue (a b x : ℕ) (ha : a = 64) (hb : b = 49) (h : isExpandedOHaraTriple a b x) : x = 30 :=
by
  sorry

end ExpandedOHaraTripleValue_l14_14900


namespace find_N_l14_14649

theorem find_N : ∃ (N : ℤ), N > 0 ∧ (36^2 * 60^2 = 30^2 * N^2) ∧ (N = 72) :=
by
  sorry

end find_N_l14_14649


namespace largest_even_number_l14_14994

theorem largest_even_number (x : ℤ) (h1 : 3 * x + 6 = (x + (x + 2) + (x + 4)) / 3 + 44) : 
  x + 4 = 24 := 
by 
  sorry

end largest_even_number_l14_14994


namespace eight_packets_weight_l14_14406

variable (weight_per_can : ℝ)
variable (weight_per_packet : ℝ)

-- Conditions
axiom h1 : weight_per_can = 1
axiom h2 : 3 * weight_per_can = 8 * weight_per_packet
axiom h3 : weight_per_packet = 6 * weight_per_can

-- Question to be proved: 8 packets weigh 12 kg
theorem eight_packets_weight : 8 * weight_per_packet = 12 :=
by 
  -- Proof would go here
  sorry

end eight_packets_weight_l14_14406


namespace probability_dmitry_before_anatoly_l14_14895

theorem probability_dmitry_before_anatoly (m : ℝ) (non_neg_m : 0 < m) :
  let volume_prism := (m^3) / 2
  let volume_tetrahedron := (m^3) / 3
  let probability := volume_tetrahedron / volume_prism
  probability = (2 : ℝ) / 3 :=
by
  sorry

end probability_dmitry_before_anatoly_l14_14895


namespace penthouse_floors_l14_14636

theorem penthouse_floors (R P : ℕ) (h1 : R + P = 23) (h2 : 12 * R + 2 * P = 256) : P = 2 :=
by
  sorry

end penthouse_floors_l14_14636


namespace sum_of_roots_l14_14466

theorem sum_of_roots (x : ℝ) : (x - 4)^2 = 16 → x = 8 ∨ x = 0 := by
  intro h
  have h1 : x - 4 = 4 ∨ x - 4 = -4 := by
    sorry
  cases h1
  case inl h2 =>
    rw [h2] at h
    exact Or.inl (by linarith)
  case inr h2 =>
    rw [h2] at h
    exact Or.inr (by linarith)

end sum_of_roots_l14_14466


namespace unique_square_friendly_l14_14577

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k^2 = n

def is_square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, is_perfect_square (m^2 + 18 * m + c)

theorem unique_square_friendly :
  ∃! c : ℤ, is_square_friendly c ∧ c = 81 := 
sorry

end unique_square_friendly_l14_14577


namespace speed_ratio_l14_14016

variable (v1 v2 : ℝ) -- Speeds of A and B respectively
variable (dA dB : ℝ) -- Distances to destinations A and B respectively

-- Conditions:
-- 1. Both reach their destinations in 1 hour
def condition_1 : Prop := dA = v1 ∧ dB = v2

-- 2. When they swap destinations, A takes 35 minutes more to reach B's destination
def condition_2 : Prop := dB / v1 = dA / v2 + 35 / 60

-- Given these conditions, prove that the ratio of v1 to v2 is 3
theorem speed_ratio (h1 : condition_1 v1 v2 dA dB) (h2 : condition_2 v1 v2 dA dB) : v1 = 3 * v2 :=
sorry

end speed_ratio_l14_14016


namespace solution_set_inequality_l14_14727

theorem solution_set_inequality (x : ℝ) : (x-3) * (x-1) > 0 → (x < 1 ∨ x > 3) :=
by sorry

end solution_set_inequality_l14_14727


namespace probability_exceeds_175_l14_14854

theorem probability_exceeds_175 (P_lt_160 : ℝ) (P_160_to_175 : ℝ) (h : ℝ) :
  P_lt_160 = 0.2 → P_160_to_175 = 0.5 → 1 - P_lt_160 - P_160_to_175 = 0.3 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end probability_exceeds_175_l14_14854


namespace cube_root_of_neg_eight_squared_is_neg_four_l14_14178

-- Define the value of -8^2
def neg_eight_squared : ℤ := -8^2

-- Define what it means for a number to be the cube root of another number
def is_cube_root (a b : ℤ) : Prop := a^3 = b

-- The desired proof statement
theorem cube_root_of_neg_eight_squared_is_neg_four :
  neg_eight_squared = -64 → is_cube_root (-4) neg_eight_squared :=
by
  sorry

end cube_root_of_neg_eight_squared_is_neg_four_l14_14178


namespace least_value_expression_l14_14749

theorem least_value_expression (x : ℝ) (h : x < -2) :
  2 * x < x ∧ 2 * x < x + 2 ∧ 2 * x < (1 / 2) * x ∧ 2 * x < x - 2 :=
by
  sorry

end least_value_expression_l14_14749


namespace complete_triangles_l14_14618

noncomputable def possible_placements_count : Nat :=
  sorry

theorem complete_triangles {a b c : Nat} :
  (1 + 2 + 4 + 10 + a + b + c) = 23 →
  ∃ (count : Nat), count = 4 := 
by
  sorry

end complete_triangles_l14_14618


namespace sufficient_necessary_condition_l14_14461

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * a * x^3 + (1 / 2) * a * x^2 - 2 * a * x + 2 * a + 1

theorem sufficient_necessary_condition (a : ℝ) :
  (-6 / 5 < a ∧ a < -3 / 16) ↔
  (∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧
   (∃ c₁ c₂ : ℝ, deriv (f a) c₁ = 0 ∧ deriv (f a) c₂ = 0 ∧
   deriv (deriv (f a)) c₁ < 0 ∧ deriv (deriv (f a)) c₂ > 0 ∧
   f a c₁ > 0 ∧ f a c₂ < 0)) := sorry

end sufficient_necessary_condition_l14_14461


namespace complete_square_eq_l14_14062

theorem complete_square_eq (x : ℝ) : x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  have : x^2 - 2 * x = 5 := by linarith
  have : x^2 - 2 * x + 1 = 6 := by linarith
  exact eq_of_sub_eq_zero (by linarith)

end complete_square_eq_l14_14062


namespace log_function_domain_l14_14639

noncomputable def domain_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Set ℝ :=
  { x : ℝ | x < a }

theorem log_function_domain (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x, x ∈ domain_of_log_function a h1 h2 ↔ x < a :=
by
  sorry

end log_function_domain_l14_14639


namespace percent_increase_in_sales_l14_14399

theorem percent_increase_in_sales (sales_this_year : ℕ) (sales_last_year : ℕ) (percent_increase : ℚ) :
  sales_this_year = 400 ∧ sales_last_year = 320 → percent_increase = 25 :=
by
  sorry

end percent_increase_in_sales_l14_14399


namespace dishwasher_manager_wage_ratio_l14_14797

theorem dishwasher_manager_wage_ratio
  (chef_wage dishwasher_wage manager_wage : ℝ)
  (h1 : chef_wage = 1.22 * dishwasher_wage)
  (h2 : dishwasher_wage = r * manager_wage)
  (h3 : manager_wage = 8.50)
  (h4 : chef_wage = manager_wage - 3.315) :
  r = 0.5 :=
sorry

end dishwasher_manager_wage_ratio_l14_14797


namespace find_x_l14_14750

theorem find_x (P0 P1 P2 P3 P4 P5 : ℝ) (y : ℝ) (h1 : P1 = P0 * 1.10)
                                      (h2 : P2 = P1 * 0.85)
                                      (h3 : P3 = P2 * 1.20)
                                      (h4 : P4 = P3 * (1 - x/100))
                                      (h5 : y = 0.15)
                                      (h6 : P5 = P4 * 1.15)
                                      (h7 : P5 = P0) : x = 23 :=
sorry

end find_x_l14_14750


namespace slope_probability_l14_14919

def line_equation (a x y : ℝ) : Prop := a * x + 2 * y - 3 = 0

def in_interval (a : ℝ) : Prop := -5 ≤ a ∧ a ≤ 4

def slope_not_less_than_1 (a : ℝ) : Prop := - a / 2 ≥ 1

noncomputable def probability_slope_not_less_than_1 : ℝ :=
  (2 - (-5)) / (4 - (-5))

theorem slope_probability :
  ∀ (a : ℝ), in_interval a → slope_not_less_than_1 a → probability_slope_not_less_than_1 = 1 / 3 :=
by
  intros a h_in h_slope
  sorry

end slope_probability_l14_14919


namespace circle_standard_equation_l14_14145

theorem circle_standard_equation (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = (x - 1 + y - 1)^2) ∧
  (∀ x y : ℝ, (x - a)^2 + (y - 1)^2 = (x - 1 + y + 2)^2) →
  (∃ x y : ℝ, (x - 2) ^ 2 + (y - 1) ^ 2 = 2) :=
sorry

end circle_standard_equation_l14_14145


namespace evaluate_expression_l14_14322

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/2) (hz : z = 8) : 
  x^3 * y^4 * z = 1/128 := 
by
  sorry

end evaluate_expression_l14_14322


namespace replaced_person_weight_l14_14859

theorem replaced_person_weight (W : ℝ) (increase : ℝ) (new_weight : ℝ) (average_increase : ℝ) (number_of_persons : ℕ) :
  average_increase = 2.5 →
  new_weight = 70 →
  number_of_persons = 8 →
  increase = number_of_persons * average_increase →
  W + increase = W - replaced_weight + new_weight →
  replaced_weight = 50 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end replaced_person_weight_l14_14859


namespace f_inequality_solution_set_l14_14244

noncomputable
def f : ℝ → ℝ := sorry

axiom f_at_1 : f 1 = 1
axiom f_deriv : ∀ x : ℝ, deriv f x < 1/3

theorem f_inequality_solution_set :
  {x : ℝ | f (x^2) > (x^2 / 3) + 2 / 3} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end f_inequality_solution_set_l14_14244


namespace sum_of_possible_k_l14_14553

theorem sum_of_possible_k (a b c k : ℂ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a / (2 - b) = k) (h5 : b / (3 - c) = k) (h6 : c / (4 - a) = k) :
  k = 1 ∨ k = -1 ∨ k = -2 → k = 1 + (-1) + (-2) :=
by
  sorry

end sum_of_possible_k_l14_14553


namespace find_unknown_numbers_l14_14775

def satisfies_condition1 (A B : ℚ) : Prop := 
  0.05 * A = 0.20 * 650 + 0.10 * B

def satisfies_condition2 (A B : ℚ) : Prop := 
  A + B = 4000

def satisfies_condition3 (B C : ℚ) : Prop := 
  C = 2 * B

def satisfies_condition4 (A B C D : ℚ) : Prop := 
  A + B + C = 0.40 * D

theorem find_unknown_numbers (A B C D : ℚ) :
  satisfies_condition1 A B → satisfies_condition2 A B →
  satisfies_condition3 B C → satisfies_condition4 A B C D →
  A = 3533 + 1/3 ∧ B = 466 + 2/3 ∧ C = 933 + 1/3 ∧ D = 12333 + 1/3 :=
by
  sorry

end find_unknown_numbers_l14_14775


namespace total_weekly_pay_proof_l14_14262

-- Define the weekly pay for employees X and Y
def weekly_pay_employee_y : ℝ := 260
def weekly_pay_employee_x : ℝ := 1.2 * weekly_pay_employee_y

-- Definition of total weekly pay
def total_weekly_pay : ℝ := weekly_pay_employee_x + weekly_pay_employee_y

-- Theorem stating the total weekly pay equals 572
theorem total_weekly_pay_proof : total_weekly_pay = 572 := by
  sorry

end total_weekly_pay_proof_l14_14262


namespace find_point_symmetric_about_y_axis_l14_14485

def point := ℤ × ℤ

def symmetric_about_y_axis (A B : point) : Prop :=
  B.1 = -A.1 ∧ B.2 = A.2

theorem find_point_symmetric_about_y_axis (A B : point) 
  (hA : A = (-5, 2)) 
  (hSym : symmetric_about_y_axis A B) : 
  B = (5, 2) := 
by
  -- We declare the proof but omit the steps for this exercise.
  sorry

end find_point_symmetric_about_y_axis_l14_14485


namespace least_real_number_K_l14_14622

theorem least_real_number_K (x y z K : ℝ) (h_cond1 : -2 ≤ x ∧ x ≤ 2) (h_cond2 : -2 ≤ y ∧ y ≤ 2) (h_cond3 : -2 ≤ z ∧ z ≤ 2) (h_eq : x^2 + y^2 + z^2 + x * y * z = 4) :
  (∀ x y z : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2 ∧ -2 ≤ z ∧ z ≤ 2 ∧ x^2 + y^2 + z^2 + x * y * z = 4 → z * (x * z + y * z + y) / (x * y + y^2 + z^2 + 1) ≤ K) → K = 4 / 3 :=
by
  sorry

end least_real_number_K_l14_14622


namespace sum_of_ages_is_20_l14_14054

-- Given conditions
variables (age_kiana age_twin : ℕ)
axiom product_of_ages : age_kiana * age_twin * age_twin = 162

-- Required proof
theorem sum_of_ages_is_20 : age_kiana + age_twin + age_twin = 20 :=
sorry

end sum_of_ages_is_20_l14_14054


namespace mowing_field_l14_14829

theorem mowing_field (x : ℝ) 
  (h1 : 1 / 84 + 1 / x = 1 / 21) : 
  x = 28 := 
sorry

end mowing_field_l14_14829


namespace hydrogen_atoms_in_compound_l14_14284

theorem hydrogen_atoms_in_compound : 
  ∀ (Al_weight O_weight H_weight : ℕ) (total_weight : ℕ) (num_Al num_O num_H : ℕ),
  Al_weight = 27 →
  O_weight = 16 →
  H_weight = 1 →
  total_weight = 78 →
  num_Al = 1 →
  num_O = 3 →
  (num_Al * Al_weight + num_O * O_weight + num_H * H_weight = total_weight) →
  num_H = 3 := 
by
  intros
  sorry

end hydrogen_atoms_in_compound_l14_14284


namespace charlie_widgets_difference_l14_14522

theorem charlie_widgets_difference (w t : ℕ) (hw : w = 3 * t) :
  w * t - ((w + 6) * (t - 3)) = 3 * t + 18 :=
by
  sorry

end charlie_widgets_difference_l14_14522


namespace total_team_cost_correct_l14_14717

variable (jerseyCost shortsCost socksCost cleatsCost waterBottleCost : ℝ)
variable (numPlayers : ℕ)
variable (discountThreshold discountRate salesTaxRate : ℝ)

noncomputable def totalTeamCost : ℝ :=
  let totalCostPerPlayer := jerseyCost + shortsCost + socksCost + cleatsCost + waterBottleCost
  let totalCost := totalCostPerPlayer * numPlayers
  let discount := if totalCost > discountThreshold then totalCost * discountRate else 0
  let discountedTotal := totalCost - discount
  let tax := discountedTotal * salesTaxRate
  let finalCost := discountedTotal + tax
  finalCost

theorem total_team_cost_correct :
  totalTeamCost 25 15.20 6.80 40 12 25 500 0.10 0.07 = 2383.43 := by
  sorry

end total_team_cost_correct_l14_14717


namespace exists_square_divisible_by_12_between_100_and_200_l14_14693

theorem exists_square_divisible_by_12_between_100_and_200 : 
  ∃ x : ℕ, (∃ y : ℕ, x = y * y) ∧ (12 ∣ x) ∧ (100 ≤ x ∧ x ≤ 200) ∧ x = 144 :=
by
  sorry

end exists_square_divisible_by_12_between_100_and_200_l14_14693


namespace sin_3pi_div_2_eq_neg_1_l14_14786

theorem sin_3pi_div_2_eq_neg_1 : Real.sin (3 * Real.pi / 2) = -1 := by
  sorry

end sin_3pi_div_2_eq_neg_1_l14_14786


namespace oblique_area_l14_14944

theorem oblique_area (side_length : ℝ) (A_ratio : ℝ) (S_original : ℝ) (S_oblique : ℝ) 
  (h1 : side_length = 1) 
  (h2 : A_ratio = (Real.sqrt 2) / 4) 
  (h3 : S_original = side_length ^ 2) 
  (h4 : S_oblique / S_original = A_ratio) : 
  S_oblique = (Real.sqrt 2) / 4 :=
by 
  sorry

end oblique_area_l14_14944


namespace smallest_other_number_l14_14186

theorem smallest_other_number (x : ℕ)  (h_pos : 0 < x) (n : ℕ)
  (h_gcd : Nat.gcd 60 n = x + 3)
  (h_lcm : Nat.lcm 60 n = x * (x + 3)) :
  n = 45 :=
sorry

end smallest_other_number_l14_14186


namespace arithmetic_sequence_a6_value_l14_14731

theorem arithmetic_sequence_a6_value
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 14) :
  a 6 = 2 :=
by
  sorry

end arithmetic_sequence_a6_value_l14_14731


namespace min_value_of_inverse_sum_l14_14092

noncomputable def min_value (a b : ℝ) := ¬(1 ≤ a + 2*b)

theorem min_value_of_inverse_sum (a b : ℝ) (h : a + 2 * b = 1) (h_nonneg : 0 < a ∧ 0 < b) :
  (1 / a + 2 / b) ≥ 9 :=
sorry

end min_value_of_inverse_sum_l14_14092


namespace variance_transformed_is_8_l14_14049

variables {n : ℕ} (x : Fin n → ℝ)

-- Given: the variance of x₁, x₂, ..., xₙ is 2.
def variance_x (x : Fin n → ℝ) : ℝ := sorry

axiom variance_x_is_2 : variance_x x = 2

-- Variance of 2 * x₁ + 3, 2 * x₂ + 3, ..., 2 * xₙ + 3
def variance_transformed (x : Fin n → ℝ) : ℝ :=
  variance_x (fun i => 2 * x i + 3)

-- Prove that the variance is 8.
theorem variance_transformed_is_8 : variance_transformed x = 8 :=
  sorry

end variance_transformed_is_8_l14_14049


namespace smallest_positive_integer_congruence_l14_14102

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 0 < x ∧ x < 17 ∧ (3 * x ≡ 14 [MOD 17]) := sorry

end smallest_positive_integer_congruence_l14_14102


namespace choir_members_correct_l14_14850

def choir_members_condition (n : ℕ) : Prop :=
  150 < n ∧ n < 250 ∧ 
  n % 3 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3

theorem choir_members_correct : ∃ n, choir_members_condition n ∧ (n = 195 ∨ n = 219) :=
by {
  sorry
}

end choir_members_correct_l14_14850


namespace find_center_radius_l14_14345

noncomputable def circle_center_radius (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y - 6 = 0 → 
  ∃ (h k r : ℝ), (x + 1) * (x + 1) + (y - 2) * (y - 2) = r ∧ h = -1 ∧ k = 2 ∧ r = 11

theorem find_center_radius :
  circle_center_radius x y :=
sorry

end find_center_radius_l14_14345


namespace smallest_n_division_l14_14207

-- Lean statement equivalent to the mathematical problem
theorem smallest_n_division (n : ℕ) (hn : n ≥ 3) : 
  (∃ (s : Finset ℕ), (∀ m ∈ s, 3 ≤ m ∧ m ≤ 2006) ∧ s.card = n - 2) ↔ n = 3 := 
sorry

end smallest_n_division_l14_14207


namespace radius_of_circle_l14_14095

-- Define the polar coordinates equation
def polar_circle (ρ θ : ℝ) : Prop := ρ = 6 * Real.cos θ

-- Define the conversion to Cartesian coordinates and the circle equation
def cartesian_circle (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 9

-- Prove that given the polar coordinates equation, the radius of the circle is 3
theorem radius_of_circle : ∀ (ρ θ : ℝ), polar_circle ρ θ → ∃ r, r = 3 := by
  sorry

end radius_of_circle_l14_14095


namespace maximum_value_expression_maximum_value_expression_achieved_l14_14447

theorem maximum_value_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  (1 / (x^2 - 4 * x + 9) + 1 / (y^2 - 4 * y + 9) + 1 / (z^2 - 4 * z + 9)) ≤ 7 / 18 :=
sorry

theorem maximum_value_expression_achieved :
  (1 / (0^2 - 4 * 0 + 9) + 1 / (0^2 - 4 * 0 + 9) + 1 / (1^2 - 4 * 1 + 9)) = 7 / 18 :=
sorry

end maximum_value_expression_maximum_value_expression_achieved_l14_14447


namespace select_four_person_committee_l14_14290

open Nat

theorem select_four_person_committee 
  (n : ℕ)
  (h1 : (n * (n - 1) * (n - 2)) / 6 = 21) 
  : (n = 9) → Nat.choose n 4 = 126 :=
by
  sorry

end select_four_person_committee_l14_14290


namespace total_stops_is_seven_l14_14347

-- Definitions of conditions
def initial_stops : ℕ := 3
def additional_stops : ℕ := 4

-- Statement to be proved
theorem total_stops_is_seven : initial_stops + additional_stops = 7 :=
by {
  -- this is a placeholder for the proof
  sorry
}

end total_stops_is_seven_l14_14347


namespace max_expression_l14_14511

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem max_expression (x y : ℝ) (h : x + y = 5) :
  max_value x y ≤ 6084 / 17 :=
sorry

end max_expression_l14_14511


namespace correct_conclusion_l14_14955

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 2 else n * 2^n

theorem correct_conclusion (n : ℕ) (h₁ : ∀ k : ℕ, k > 0 → a_n (k + 1) - 2 * a_n k = 2^(k + 1)) :
  a_n n = n * 2 ^ n :=
by
  sorry

end correct_conclusion_l14_14955


namespace find_a_l14_14393

theorem find_a (a r s : ℚ) (h1 : a = r^2) (h2 : 20 = 2 * r * s) (h3 : 9 = s^2) : a = 100 / 9 := by
  sorry

end find_a_l14_14393


namespace slant_height_of_cone_l14_14894

theorem slant_height_of_cone
  (r : ℝ) (CSA : ℝ) (l : ℝ)
  (hr : r = 14)
  (hCSA : CSA = 1539.3804002589986) :
  CSA = Real.pi * r * l → l = 35 := 
sorry

end slant_height_of_cone_l14_14894


namespace find_angle_B_l14_14878

theorem find_angle_B
  (a : ℝ) (c : ℝ) (A B C : ℝ)
  (h1 : a = 5 * Real.sqrt 2)
  (h2 : c = 10)
  (h3 : A = π / 6) -- 30 degrees in radians
  (h4 : A + B + C = π) -- sum of angles in a triangle
  : B = 7 * π / 12 ∨ B = π / 12 := -- 105 degrees or 15 degrees in radians
sorry

end find_angle_B_l14_14878


namespace common_ratio_geometric_sequence_l14_14585

-- Definition of a geometric sequence and given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_geo : is_geometric_sequence a)
  (h_a2 : a 2 = 2) (h_a5 : a 5 = 1 / 4) : q = 1 / 2 :=
by 
  sorry

end common_ratio_geometric_sequence_l14_14585


namespace range_of_a_l14_14699

def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def B (a : ℝ) : Set ℝ := { x | x ≥ a }

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ≤ -2 :=
by
  sorry

end range_of_a_l14_14699


namespace exist_pairwise_distinct_gcd_l14_14839

theorem exist_pairwise_distinct_gcd (S : Set ℕ) (h_inf : S.Infinite) 
  (h_gcd : ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ gcd a b ≠ gcd c d) :
  ∃ x y z : ℕ, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ gcd x y = gcd y z ∧ gcd y z ≠ gcd z x := 
by sorry

end exist_pairwise_distinct_gcd_l14_14839


namespace sqrt_x_minus_2_meaningful_in_reals_l14_14214

theorem sqrt_x_minus_2_meaningful_in_reals (x : ℝ) : (∃ (y : ℝ), y * y = x - 2) → x ≥ 2 :=
by
  sorry

end sqrt_x_minus_2_meaningful_in_reals_l14_14214


namespace find_integer_n_l14_14225

open Int

theorem find_integer_n (n a b : ℤ) :
  (4 * n + 1 = a^2) ∧ (9 * n + 1 = b^2) → n = 0 := by
sorry

end find_integer_n_l14_14225


namespace max_value_of_e_n_l14_14602

def b (n : ℕ) : ℕ := (8^n - 1) / 7
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem max_value_of_e_n : ∀ n : ℕ, e n = 1 := 
by
  sorry

end max_value_of_e_n_l14_14602


namespace average_side_length_of_squares_l14_14596

theorem average_side_length_of_squares (a b c : ℕ) (h₁ : a = 36) (h₂ : b = 64) (h₃ : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 26 / 3 := by
  sorry

end average_side_length_of_squares_l14_14596


namespace smallest_natural_greater_than_12_l14_14223

def smallest_greater_than (n : ℕ) : ℕ := n + 1

theorem smallest_natural_greater_than_12 : smallest_greater_than 12 = 13 :=
by
  sorry

end smallest_natural_greater_than_12_l14_14223


namespace angle_between_hour_and_minute_hand_at_5_oclock_l14_14059

theorem angle_between_hour_and_minute_hand_at_5_oclock : 
  let degrees_in_circle := 360
  let hours_in_clock := 12
  let angle_per_hour := degrees_in_circle / hours_in_clock
  let hour_hand_position := 5
  let minute_hand_position := 0
  let angle := (hour_hand_position - minute_hand_position) * angle_per_hour
  angle = 150 :=
by sorry

end angle_between_hour_and_minute_hand_at_5_oclock_l14_14059


namespace algebra_square_formula_l14_14529

theorem algebra_square_formula (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
sorry

end algebra_square_formula_l14_14529


namespace find_x_value_l14_14122

/-- Defining the conditions given in the problem -/
structure HenrikhConditions where
  x : ℕ
  walking_time_per_block : ℕ := 60
  bicycle_time_per_block : ℕ := 20
  skateboard_time_per_block : ℕ := 40
  added_time_walking_over_bicycle : ℕ := 480
  added_time_walking_over_skateboard : ℕ := 240

/-- Defining a hypothesis based on the conditions -/
noncomputable def henrikh (c : HenrikhConditions) : Prop :=
  c.walking_time_per_block * c.x = c.bicycle_time_per_block * c.x + c.added_time_walking_over_bicycle ∧
  c.walking_time_per_block * c.x = c.skateboard_time_per_block * c.x + c.added_time_walking_over_skateboard

/-- The theorem to be proved -/
theorem find_x_value (c : HenrikhConditions) (h : henrikh c) : c.x = 12 := by
  sorry

end find_x_value_l14_14122


namespace findNumberOfItemsSoldByStoreA_l14_14765

variable (P x : ℝ) -- P is the price of the product, x is the number of items Store A sells

-- Total sales amount for Store A (in yuan)
def totalSalesA := P * x = 7200

-- Total sales amount for Store B (in yuan)
def totalSalesB := 0.8 * P * (x + 15) = 7200

-- Same price in both stores
def samePriceInBothStores := (P > 0)

-- Proof Problem Statement
theorem findNumberOfItemsSoldByStoreA (storeASellsAtListedPrice : totalSalesA P x)
  (storeBSells15MoreItemsAndAt80PercentPrice : totalSalesB P x)
  (priceIsPositive : samePriceInBothStores P) :
  x = 60 :=
sorry

end findNumberOfItemsSoldByStoreA_l14_14765


namespace intersection_eq_l14_14754

-- Universal set and its sets M and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 > 9}
def N : Set ℝ := {x | -1 < x ∧ x < 4}
def complement_N : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}

-- Prove the intersection
theorem intersection_eq :
  M ∩ complement_N = {x | x < -3 ∨ x ≥ 4} :=
by
  sorry

end intersection_eq_l14_14754


namespace average_after_19_innings_is_23_l14_14419

-- Definitions for the conditions given in the problem
variables {A : ℝ} -- Let A be the average score before the 19th inning

-- Conditions: The cricketer scored 95 runs in the 19th inning and his average increased by 4 runs.
def total_runs_after_18_innings (A : ℝ) : ℝ := 18 * A
def total_runs_after_19th_inning (A : ℝ) : ℝ := total_runs_after_18_innings A + 95
def new_average_after_19_innings (A : ℝ) : ℝ := A + 4

-- The statement of the problem as a Lean theorem
theorem average_after_19_innings_is_23 :
  (18 * A + 95) / 19 = A + 4 → A = 19 → (A + 4) = 23 :=
by
  intros hA h_avg_increased
  sorry

end average_after_19_innings_is_23_l14_14419


namespace find_theta_interval_l14_14518

theorem find_theta_interval (θ : ℝ) (x : ℝ) :
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (0 ≤ x ∧ x ≤ 1) →
  (∀ k, k = 0.5 → x^2 * Real.sin θ - k * x * (1 - x) + (1 - x)^2 * Real.cos θ ≥ 0) ↔
  (0 ≤ θ ∧ θ ≤ π / 12) ∨ (23 * π / 12 ≤ θ ∧ θ ≤ 2 * π) := 
sorry

end find_theta_interval_l14_14518


namespace term_position_in_sequence_l14_14245

theorem term_position_in_sequence (n : ℕ) (h1 : n > 0) (h2 : 3 * n + 1 = 40) : n = 13 :=
by
  sorry

end term_position_in_sequence_l14_14245


namespace smallest_part_2340_division_l14_14266

theorem smallest_part_2340_division :
  ∃ (A B C : ℕ), (A + B + C = 2340) ∧ 
                 (A / 5 = B / 7) ∧ 
                 (B / 7 = C / 11) ∧ 
                 (A = 510) :=
by 
  sorry

end smallest_part_2340_division_l14_14266


namespace income_is_10000_l14_14390

theorem income_is_10000 (x : ℝ) (h : 10 * x = 8 * x + 2000) : 10 * x = 10000 := by
  have h1 : 2 * x = 2000 := by
    linarith
  have h2 : x = 1000 := by
    linarith
  linarith

end income_is_10000_l14_14390


namespace T_sum_correct_l14_14743

-- Defining the sequence T_n
def T (n : ℕ) : ℤ := 
(-1)^n * 2 * n + (-1)^(n + 1) * n

-- Values to compute
def n1 : ℕ := 27
def n2 : ℕ := 43
def n3 : ℕ := 60

-- Sum of particular values
def T_sum : ℤ := T n1 + T n2 + T n3

-- Placeholder value until actual calculation
def expected_sum : ℤ := -42 -- Replace with the correct calculated result

theorem T_sum_correct : T_sum = expected_sum := sorry

end T_sum_correct_l14_14743


namespace inequality_geq_l14_14432

theorem inequality_geq (t : ℝ) (n : ℕ) (ht : t ≥ 1/2) : 
  t^(2*n) ≥ (t-1)^(2*n) + (2*t-1)^n := 
sorry

end inequality_geq_l14_14432


namespace dentist_age_is_32_l14_14543

-- Define the conditions
def one_sixth_of_age_8_years_ago_eq_one_tenth_of_age_8_years_hence (x : ℕ) : Prop :=
  (x - 8) / 6 = (x + 8) / 10

-- State the theorem
theorem dentist_age_is_32 : ∃ x : ℕ, one_sixth_of_age_8_years_ago_eq_one_tenth_of_age_8_years_hence x ∧ x = 32 :=
by
  sorry

end dentist_age_is_32_l14_14543


namespace consecutive_integers_sum_l14_14733

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 380) : x + (x + 1) = 39 := by
  sorry

end consecutive_integers_sum_l14_14733


namespace crayons_lost_l14_14286

theorem crayons_lost (initial_crayons ending_crayons : ℕ) (h_initial : initial_crayons = 253) (h_ending : ending_crayons = 183) : (initial_crayons - ending_crayons) = 70 :=
by
  sorry

end crayons_lost_l14_14286


namespace range_of_k_l14_14237

noncomputable def quadratic_inequality (k : ℝ) := 
  ∀ x : ℝ, 2 * k * x^2 + k * x - (3 / 8) < 0

theorem range_of_k (k : ℝ) :
  (quadratic_inequality k) → -3 < k ∧ k < 0 := sorry

end range_of_k_l14_14237


namespace find_missing_fraction_l14_14287

theorem find_missing_fraction :
  ∃ (x : ℚ), (1/2 + -5/6 + 1/5 + 1/4 + -9/20 + -9/20 + x = 9/20) :=
  by
  sorry

end find_missing_fraction_l14_14287


namespace R2_area_l14_14416

-- Definitions for the conditions
def R1_side1 : ℝ := 4
def R1_area : ℝ := 16
def R2_diagonal : ℝ := 10
def similar_rectangles (R1 R2 : ℝ × ℝ) : Prop := (R1.fst / R1.snd = R2.fst / R2.snd)

-- Main theorem
theorem R2_area {a b : ℝ} 
  (R1_side1 : a = 4)
  (R1_area : a * a = 16) 
  (R2_diagonal : b = 10)
  (h : similar_rectangles (a, a) (b / (10 / (2 : ℝ)), b / (10 / (2 : ℝ)))) : 
  b * b / (2 : ℝ) = 50 :=
by
  sorry

end R2_area_l14_14416


namespace find_length_of_train_l14_14708

noncomputable def speed_kmhr : ℝ := 30
noncomputable def time_seconds : ℝ := 9
noncomputable def conversion_factor : ℝ := 5 / 18
noncomputable def speed_ms : ℝ := speed_kmhr * conversion_factor
noncomputable def length_train : ℝ := speed_ms * time_seconds

theorem find_length_of_train : length_train = 74.97 := 
by
  sorry

end find_length_of_train_l14_14708


namespace differential_savings_is_4830_l14_14949

-- Defining the conditions
def initial_tax_rate : ℝ := 0.42
def new_tax_rate : ℝ := 0.28
def annual_income : ℝ := 34500

-- Defining the calculation of tax before and after the tax rate change
def tax_before : ℝ := annual_income * initial_tax_rate
def tax_after : ℝ := annual_income * new_tax_rate

-- Defining the differential savings
def differential_savings : ℝ := tax_before - tax_after

-- Statement asserting that the differential savings is $4830
theorem differential_savings_is_4830 : differential_savings = 4830 := by sorry

end differential_savings_is_4830_l14_14949


namespace perimeter_paper_count_l14_14444

theorem perimeter_paper_count (n : Nat) (h : n = 10) : 
  let top_side := n
  let right_side := n - 1
  let bottom_side := n - 1
  let left_side := n - 2
  top_side + right_side + bottom_side + left_side = 36 :=
by
  sorry

end perimeter_paper_count_l14_14444


namespace wyatt_headmaster_duration_l14_14784

def duration_of_wyatt_job (start_month end_month total_months : ℕ) : Prop :=
  start_month <= end_month ∧ total_months = end_month - start_month + 1

theorem wyatt_headmaster_duration : duration_of_wyatt_job 3 12 9 :=
by
  sorry

end wyatt_headmaster_duration_l14_14784


namespace arithmetic_sum_S11_l14_14632

noncomputable def Sn_sum (a1 an n : ℕ) : ℕ := n * (a1 + an) / 2

theorem arithmetic_sum_S11 (a1 a9 a8 a5 a11 : ℕ) (h1 : Sn_sum a1 a9 9 = 54)
    (h2 : Sn_sum a1 a8 8 - Sn_sum a1 a5 5 = 30) : Sn_sum a1 a11 11 = 88 := by
  sorry

end arithmetic_sum_S11_l14_14632


namespace smallest_value_a2_b2_c2_l14_14868

theorem smallest_value_a2_b2_c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 3 * b + 4 * c = 120) : 
  a^2 + b^2 + c^2 ≥ 14400 / 29 :=
by sorry

end smallest_value_a2_b2_c2_l14_14868


namespace expenditure_increase_36_percent_l14_14088

theorem expenditure_increase_36_percent
  (m : ℝ) -- mass of the bread
  (p_bread : ℝ) -- price of the bread
  (p_crust : ℝ) -- price of the crust
  (h1 : p_crust = 1.2 * p_bread) -- condition: crust is 20% more expensive
  (h2 : p_crust = 1.2 * p_bread) -- condition: crust is 20% more expensive
  (h3 : ∃ (m_crust : ℝ), m_crust = 0.75 * m) -- condition: crust is 25% lighter in weight
  (h4 : ∃ (m_consumed_bread : ℝ), m_consumed_bread = 0.85 * m) -- condition: 15% of bread dries out
  (h5 : ∃ (m_consumed_crust : ℝ), m_consumed_crust = 0.75 * m) -- condition: crust is consumed completely
  : (17 / 15) * (1.2 : ℝ) = 1.36 := 
by sorry

end expenditure_increase_36_percent_l14_14088


namespace solve_triple_l14_14291

theorem solve_triple (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c + a * b + c = a^3) : 
  (b = a - 1 ∧ c = a) ∨ (b = 1 ∧ c = a * (a - 1)) :=
by 
  sorry

end solve_triple_l14_14291


namespace total_amount_paid_l14_14427

theorem total_amount_paid (sales_tax : ℝ) (tax_rate : ℝ) (cost_tax_free_items : ℝ) : 
  sales_tax = 1.28 → tax_rate = 0.08 → cost_tax_free_items = 12.72 → 
  (sales_tax / tax_rate + sales_tax + cost_tax_free_items) = 30.00 :=
by
  intros h1 h2 h3
  -- Proceed with the proof using h1, h2, and h3
  sorry

end total_amount_paid_l14_14427


namespace digit_after_decimal_l14_14910

theorem digit_after_decimal (n : ℕ) : (n = 123) → (123 % 12 ≠ 0) → (123 % 12 = 3) → (∃ d : ℕ, d = 1 ∧ (43 / 740 : ℚ)^123 = 0 + d / 10^(123)) := 
by
    intros h₁ h₂ h₃
    sorry

end digit_after_decimal_l14_14910


namespace find_positive_integers_l14_14340

theorem find_positive_integers (a b : ℕ) (h1 : a > 1) (h2 : b ∣ (a - 1)) (h3 : (2 * a + 1) ∣ (5 * b - 3)) : a = 10 ∧ b = 9 :=
sorry

end find_positive_integers_l14_14340


namespace lillian_candies_total_l14_14471

variable (initial_candies : ℕ)
variable (candies_given_by_father : ℕ)

theorem lillian_candies_total (initial_candies : ℕ) (candies_given_by_father : ℕ) :
  initial_candies = 88 →
  candies_given_by_father = 5 →
  initial_candies + candies_given_by_father = 93 :=
by
  intros
  sorry

end lillian_candies_total_l14_14471


namespace ashley_loan_least_months_l14_14820

theorem ashley_loan_least_months (t : ℕ) (principal : ℝ) (interest_rate : ℝ) (triple_principal : ℝ) : 
  principal = 1500 ∧ interest_rate = 0.06 ∧ triple_principal = 3 * principal → 
  1.06^t > triple_principal → t = 20 :=
by
  intro h h2
  sorry

end ashley_loan_least_months_l14_14820


namespace sum_of_digits_palindrome_l14_14090

theorem sum_of_digits_palindrome 
  (r : ℕ) 
  (h1 : r ≤ 36) 
  (x p q : ℕ) 
  (h2 : 2 * q = 5 * p) 
  (h3 : x = p * r^3 + p * r^2 + q * r + q) 
  (h4 : ∃ (a b c : ℕ), (x * x = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a)) : 
  (2 * (a + b + c) = 36) := 
sorry

end sum_of_digits_palindrome_l14_14090


namespace probability_of_event_A_l14_14107

def total_balls : ℕ := 10
def white_balls : ℕ := 7
def black_balls : ℕ := 3

def event_A : Prop := (black_balls / total_balls) * (white_balls / (total_balls - 1)) = 7 / 30

theorem probability_of_event_A : event_A := by
  sorry

end probability_of_event_A_l14_14107


namespace geometric_sequence_common_ratio_is_2_l14_14353

variable {a : ℕ → ℝ} (h : ∀ n : ℕ, a n * a (n + 1) = 4 ^ n)

theorem geometric_sequence_common_ratio_is_2 : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_is_2_l14_14353


namespace ticket_cost_l14_14257

theorem ticket_cost (a : ℝ)
  (h1 : ∀ c : ℝ, c = a / 3)
  (h2 : 3 * a + 5 * (a / 3) = 27.75) :
  6 * a + 9 * (a / 3) = 53.52 := 
sorry

end ticket_cost_l14_14257


namespace base_salary_l14_14576

theorem base_salary {B : ℝ} {C : ℝ} :
  (B + 200 * C = 2000) → 
  (B + 200 * 15 = 4000) → 
  B = 1000 :=
by
  sorry

end base_salary_l14_14576


namespace larger_of_two_numbers_l14_14204

-- Define necessary conditions
def hcf : ℕ := 23
def factor1 : ℕ := 11
def factor2 : ℕ := 12
def lcm : ℕ := hcf * factor1 * factor2

-- Define the problem statement in Lean
theorem larger_of_two_numbers : ∃ (a b : ℕ), a = hcf * factor1 ∧ b = hcf * factor2 ∧ max a b = 276 := by
  sorry

end larger_of_two_numbers_l14_14204


namespace expression_evaluation_l14_14236

theorem expression_evaluation : 4 * 10 + 5 * 11 + 12 * 4 + 4 * 9 = 179 :=
by
  sorry

end expression_evaluation_l14_14236


namespace roots_of_cubic_eq_l14_14714

theorem roots_of_cubic_eq (r s t p q : ℝ) (h1 : r + s + t = p) (h2 : r * s + r * t + s * t = q) 
(h3 : r * s * t = r) : r^2 + s^2 + t^2 = p^2 - 2 * q := 
by 
  sorry

end roots_of_cubic_eq_l14_14714


namespace smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2_l14_14443

theorem smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2 :
  ∀ (x y z w : ℝ),
    x = Real.sqrt 3 →
    y = -1 / 3 →
    z = -2 →
    w = 0 →
    (x > 1) ∧ (y < 0) ∧ (z < 0) ∧ (|y| = 1 / 3) ∧ (|z| = 2) ∧ (w = 0) →
    min (min (min x y) z) w = z :=
by
  intros x y z w hx hy hz hw hcond
  sorry

end smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2_l14_14443


namespace ellipse_equation_constants_l14_14650

noncomputable def ellipse_parametric_eq (t : ℝ) : ℝ × ℝ :=
  ((3 * (Real.sin t - 2)) / (3 - Real.cos t),
  (4 * (Real.cos t - 4)) / (3 - Real.cos t))

theorem ellipse_equation_constants :
  ∃ (A B C D E F : ℤ), ∀ (x y : ℝ),
  ((∃ t : ℝ, (x, y) = ellipse_parametric_eq t) → (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) ∧
  (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd A B) C) D) E) F = 1) ∧
  (|A| + |B| + |C| + |D| + |E| + |F| = 2502) :=
sorry

end ellipse_equation_constants_l14_14650


namespace tiles_needed_l14_14587

/--
A rectangular swimming pool is 20m long, 8m wide, and 1.5m deep. 
Each tile used to cover the pool has a side length of 2dm. 
We need to prove the number of tiles required to cover the bottom and all four sides of the pool.
-/
theorem tiles_needed (pool_length pool_width pool_depth : ℝ) (tile_side : ℝ) 
  (h1 : pool_length = 20) (h2 : pool_width = 8) (h3 : pool_depth = 1.5) 
  (h4 : tile_side = 0.2) : 
  (pool_length * pool_width + 2 * pool_length * pool_depth + 2 * pool_width * pool_depth) / (tile_side * tile_side) = 6100 :=
by
  sorry

end tiles_needed_l14_14587


namespace custom_op_evaluation_l14_14703

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem custom_op_evaluation : custom_op 4 2 = 26 := 
by 
  sorry

end custom_op_evaluation_l14_14703


namespace xy_value_l14_14273

theorem xy_value (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = -5) : x + y = 1 := 
sorry

end xy_value_l14_14273


namespace moles_of_CO2_formed_l14_14837

variables (CH4 O2 C2H2 CO2 H2O : Type)
variables (nCH4 nO2 nC2H2 nCO2 : ℕ)
variables (reactsCompletely : Prop)

-- Balanced combustion equations
axiom combustion_methane : ∀ (mCH4 mO2 mCO2 mH2O : ℕ), mCH4 = 1 → mO2 = 2 → mCO2 = 1 → mH2O = 2 → Prop
axiom combustion_acetylene : ∀ (aC2H2 aO2 aCO2 aH2O : ℕ), aC2H2 = 2 → aO2 = 5 → aCO2 = 4 → aH2O = 2 → Prop

-- Given conditions
axiom conditions :
  nCH4 = 3 ∧ nO2 = 6 ∧ nC2H2 = 2 ∧ reactsCompletely

-- Prove the number of moles of CO2 formed
theorem moles_of_CO2_formed : 
  (nCH4 = 3 ∧ nO2 = 6 ∧ nC2H2 = 2 ∧ reactsCompletely) →
  nCO2 = 3
:= by
  intros h
  sorry

end moles_of_CO2_formed_l14_14837


namespace smallest_a1_l14_14963

noncomputable def is_sequence (a : ℕ → ℝ) : Prop :=
∀ n > 1, a n = 7 * a (n - 1) - 2 * n

noncomputable def is_positive_sequence (a : ℕ → ℝ) : Prop :=
∀ n > 0, a n > 0

theorem smallest_a1 (a : ℕ → ℝ)
  (h_seq : is_sequence a)
  (h_pos : is_positive_sequence a) :
  a 1 ≥ 13 / 18 :=
sorry

end smallest_a1_l14_14963


namespace train_speed_before_accident_l14_14898

theorem train_speed_before_accident (d v : ℝ) (hv_pos : v > 0) (hd_pos : d > 0) :
  (d / ((3/4) * v) - d / v = 35 / 60) ∧
  (d - 24) / ((3/4) * v) - (d - 24) / v = 25 / 60 → 
  v = 64 :=
by
  sorry

end train_speed_before_accident_l14_14898


namespace log_expression_simplifies_to_one_l14_14409

theorem log_expression_simplifies_to_one :
  (Real.log 5)^2 + Real.log 50 * Real.log 2 = 1 :=
by 
  sorry

end log_expression_simplifies_to_one_l14_14409


namespace bellas_goal_product_l14_14572

theorem bellas_goal_product (g1 g2 g3 g4 g5 g6 : ℕ) (g7 g8 : ℕ) 
  (h1 : g1 = 5) 
  (h2 : g2 = 3) 
  (h3 : g3 = 2) 
  (h4 : g4 = 4)
  (h5 : g5 = 1) 
  (h6 : g6 = 6)
  (h7 : g7 < 10)
  (h8 : (g1 + g2 + g3 + g4 + g5 + g6 + g7) % 7 = 0) 
  (h9 : g8 < 10)
  (h10 : (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) % 8 = 0) :
  g7 * g8 = 28 :=
by 
  sorry

end bellas_goal_product_l14_14572


namespace ratio_major_minor_is_15_4_l14_14817

-- Define the given conditions
def main_characters : ℕ := 5
def minor_characters : ℕ := 4
def minor_character_pay : ℕ := 15000
def total_payment : ℕ := 285000

-- Define the total pay to minor characters
def minor_total_pay : ℕ := minor_characters * minor_character_pay

-- Define the total pay to major characters
def major_total_pay : ℕ := total_payment - minor_total_pay

-- Define the ratio computation
def ratio_major_minor : ℕ × ℕ := (major_total_pay / 15000, minor_total_pay / 15000)

-- State the theorem
theorem ratio_major_minor_is_15_4 : ratio_major_minor = (15, 4) :=
by
  -- Proof goes here
  sorry

end ratio_major_minor_is_15_4_l14_14817


namespace no_function_satisfies_inequality_l14_14341

theorem no_function_satisfies_inequality (f : ℝ → ℝ) :
  ¬ ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y| :=
sorry

end no_function_satisfies_inequality_l14_14341


namespace no_perfect_power_l14_14740

theorem no_perfect_power (n m : ℕ) (hn : 0 < n) (hm : 1 < m) : 102 ^ 1991 + 103 ^ 1991 ≠ n ^ m := 
sorry

end no_perfect_power_l14_14740


namespace power_difference_expression_l14_14903

theorem power_difference_expression : 
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * (30^1001) :=
by
  sorry

end power_difference_expression_l14_14903


namespace hypotenuse_length_l14_14512

-- Definitions derived from conditions
def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Proposed theorem
theorem hypotenuse_length (a c : ℝ) 
  (h1 : is_isosceles_right_triangle a a c) 
  (h2 : perimeter a a c = 8 + 8 * Real.sqrt 2) :
  c = 4 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l14_14512


namespace eval_expression_in_second_quadrant_l14_14438

theorem eval_expression_in_second_quadrant (α : ℝ) (h1 : π/2 < α ∧ α < π) (h2 : Real.sin α > 0) (h3 : Real.cos α < 0) :
  (Real.sin α / Real.cos α) * Real.sqrt (1 / (Real.sin α) ^ 2 - 1) = -1 :=
by
  sorry

end eval_expression_in_second_quadrant_l14_14438


namespace square_area_of_equal_perimeter_l14_14676

theorem square_area_of_equal_perimeter 
  (side_length_triangle : ℕ) (side_length_square : ℕ) (perimeter_square : ℕ)
  (h1 : side_length_triangle = 20)
  (h2 : perimeter_square = 3 * side_length_triangle)
  (h3 : 4 * side_length_square = perimeter_square) :
  side_length_square ^ 2 = 225 := 
by
  sorry

end square_area_of_equal_perimeter_l14_14676


namespace journey_distance_l14_14173

theorem journey_distance :
  ∃ D : ℝ, ((D / 2) / 21 + (D / 2) / 24 = 10) ∧ D = 224 :=
by
  use 224
  sorry

end journey_distance_l14_14173


namespace shirt_ratio_l14_14288

theorem shirt_ratio
  (A B S : ℕ)
  (h1 : A = 6 * B)
  (h2 : B = 3)
  (h3 : S = 72) :
  S / A = 4 :=
by
  sorry

end shirt_ratio_l14_14288


namespace carter_total_drum_sticks_l14_14468

def sets_per_show_used := 5
def sets_per_show_tossed := 6
def nights := 30

theorem carter_total_drum_sticks : 
  (sets_per_show_used + sets_per_show_tossed) * nights = 330 := by
  sorry

end carter_total_drum_sticks_l14_14468


namespace height_large_cylinder_is_10_l14_14710

noncomputable def height_large_cylinder : ℝ :=
  let V_small := 13.5 * Real.pi
  let factor := 74.07407407407408
  let V_large := 100 * Real.pi
  factor * V_small / V_large

theorem height_large_cylinder_is_10 :
  height_large_cylinder = 10 :=
by
  sorry

end height_large_cylinder_is_10_l14_14710


namespace correct_time_after_2011_minutes_l14_14858

def time_2011_minutes_after_midnight : String :=
  "2011 minutes after midnight on January 1, 2011 is January 2 at 9:31AM"

theorem correct_time_after_2011_minutes :
  time_2011_minutes_after_midnight = "2011 minutes after midnight on January 1, 2011 is January 2 at 9:31AM" :=
sorry

end correct_time_after_2011_minutes_l14_14858


namespace find_third_coaster_speed_l14_14645

theorem find_third_coaster_speed
  (s1 s2 s4 s5 avg_speed n : ℕ)
  (hs1 : s1 = 50)
  (hs2 : s2 = 62)
  (hs4 : s4 = 70)
  (hs5 : s5 = 40)
  (havg_speed : avg_speed = 59)
  (hn : n = 5) : 
  ∃ s3 : ℕ, s3 = 73 :=
by
  sorry

end find_third_coaster_speed_l14_14645


namespace compare_f_m_plus_2_l14_14933

theorem compare_f_m_plus_2 (a : ℝ) (ha : a > 0) (m : ℝ) 
  (hf : (a * m^2 + 2 * a * m + 1) < 0) : 
  (a * (m + 2)^2 + 2 * a * (m + 2) + 1) > 1 :=
sorry

end compare_f_m_plus_2_l14_14933


namespace max_range_f_plus_2g_l14_14403

noncomputable def max_val_of_f_plus_2g (f g : ℝ → ℝ) (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5) (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2) : ℝ :=
  9

theorem max_range_f_plus_2g (f g : ℝ → ℝ) (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5) (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2) :
  ∃ (a b : ℝ), (-3 ≤ a ∧ a ≤ 5) ∧ (-8 ≤ b ∧ b ≤ 4) ∧ b = 9 := 
sorry

end max_range_f_plus_2g_l14_14403


namespace weighted_average_inequality_l14_14785

variable (x y z : ℝ)
variable (h1 : x < y) (h2 : y < z)

theorem weighted_average_inequality :
  (4 * z + x + y) / 6 > (x + y + 2 * z) / 4 :=
by
  sorry

end weighted_average_inequality_l14_14785


namespace common_ratio_of_infinite_geometric_series_l14_14931

theorem common_ratio_of_infinite_geometric_series 
  (a b : ℚ) 
  (h1 : a = 8 / 10) 
  (h2 : b = -6 / 15) 
  (h3 : b = a * r) : 
  r = -1 / 2 :=
by
  -- The proof goes here
  sorry

end common_ratio_of_infinite_geometric_series_l14_14931


namespace sum_abc_l14_14098

noncomputable def f (a b c : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then a * x + 3
  else if x = 0 then a * b
  else b * x^2 + c

theorem sum_abc (a b c : ℕ) (h1 : f a b c 2 = 7) (h2 : f a b c 0 = 6) (h3 : f a b c (-1) = 8) :
  a + b + c = 10 :=
by {
  sorry
}

end sum_abc_l14_14098


namespace ticket_price_divisors_count_l14_14307

theorem ticket_price_divisors_count :
  ∃ (x : ℕ), (36 % x = 0) ∧ (60 % x = 0) ∧ (Nat.divisors (Nat.gcd 36 60)).card = 6 := 
by
  sorry

end ticket_price_divisors_count_l14_14307


namespace train_pass_station_time_l14_14961

-- Define the lengths of the train and station
def length_train : ℕ := 250
def length_station : ℕ := 200

-- Define the speed of the train in km/hour
def speed_kmh : ℕ := 36

-- Convert the speed to meters per second
def speed_mps : ℕ := speed_kmh * 1000 / 3600

-- Calculate the total distance the train needs to cover
def total_distance : ℕ := length_train + length_station

-- Define the expected time to pass the station
def expected_time : ℕ := 45

-- State the theorem that needs to be proven
theorem train_pass_station_time :
  total_distance / speed_mps = expected_time := by
  sorry

end train_pass_station_time_l14_14961


namespace sum_of_roots_l14_14667

noncomputable def equation (x : ℝ) := 2 * (x^2 + 1 / x^2) - 3 * (x + 1 / x) = 1

theorem sum_of_roots (r s : ℝ) (hr : equation r) (hs : equation s) (hne : r ≠ s) :
  r + s = -5 / 2 :=
sorry

end sum_of_roots_l14_14667


namespace seven_pow_l14_14843

theorem seven_pow (k : ℕ) (h : 7 ^ k = 2) : 7 ^ (4 * k + 2) = 784 :=
by 
  sorry

end seven_pow_l14_14843


namespace arccos_zero_l14_14153

theorem arccos_zero : Real.arccos 0 = Real.pi / 2 := 
by 
  sorry

end arccos_zero_l14_14153


namespace inequality_with_equality_condition_l14_14200

variable {a b c d : ℝ}

theorem inequality_with_equality_condition (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : 
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧ 
  ((a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d) := sorry

end inequality_with_equality_condition_l14_14200


namespace product_of_roots_l14_14482

variable {x1 x2 : ℝ}

theorem product_of_roots (hx1 : x1 * Real.log x1 = 2006) (hx2 : x2 * Real.exp x2 = 2006) : x1 * x2 = 2006 :=
sorry

end product_of_roots_l14_14482


namespace distance_between_A_and_B_l14_14548

def rowing_speed_still_water : ℝ := 10
def round_trip_time : ℝ := 5
def stream_speed : ℝ := 2

theorem distance_between_A_and_B : 
  ∃ x : ℝ, 
    (x / (rowing_speed_still_water - stream_speed) + x / (rowing_speed_still_water + stream_speed) = round_trip_time) 
    ∧ x = 24 :=
sorry

end distance_between_A_and_B_l14_14548


namespace ratio_of_x_to_y_l14_14752

-- Defining the given condition
def ratio_condition (x y : ℝ) : Prop :=
  (3 * x - 2 * y) / (2 * x + y) = 3 / 5

-- The theorem to be proven
theorem ratio_of_x_to_y (x y : ℝ) (h : ratio_condition x y) : x / y = 13 / 9 :=
by
  sorry

end ratio_of_x_to_y_l14_14752


namespace find_index_l14_14629

-- Declaration of sequence being arithmetic with first term 1 and common difference 3
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 1 + (n - 1) * 3

-- The theorem to be proven
theorem find_index (a : ℕ → ℤ) (h1 : arithmetic_sequence a) (h2 : a 672 = 2014) : 672 = 672 :=
by 
  sorry

end find_index_l14_14629


namespace max_S_value_l14_14251

noncomputable def max_S (A C : ℝ) [DecidableEq ℝ] : ℝ :=
  if h : 0 < A ∧ A < 2 * Real.pi / 3 ∧ A + C = 2 * Real.pi / 3 then
    (Real.sqrt 3 / 6) * Real.sin (2 * A - Real.pi / 3) + (Real.sqrt 3 / 12)
  else
    0

theorem max_S_value :
  ∃ (A C : ℝ), A + C = 2 * Real.pi / 3 ∧
    (S = (Real.sqrt 3 / 3) * Real.sin A * Real.sin C) ∧
    (max_S A C = Real.sqrt 3 / 4) := 
sorry

end max_S_value_l14_14251


namespace matrix_power_difference_l14_14886

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 4;
     0, 1]

theorem matrix_power_difference :
  B^30 - 3 * B^29 = !![-2, 0;
                       0,  2] := 
by
  sorry

end matrix_power_difference_l14_14886


namespace extreme_points_l14_14134

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem extreme_points (a b : ℝ) 
  (h1 : 3*(-2)^2 + 2*a*(-2) + b = 0) 
  (h2 : 3*(4)^2 + 2*a*(4) + b = 0) : 
  a - b = 21 :=
by sorry

end extreme_points_l14_14134


namespace jorge_acres_l14_14700

theorem jorge_acres (A : ℕ) (H1 : A = 60) 
    (H2 : ∀ acres, acres / 3 = 60 / 3 ∧ 2 * (acres / 3) = 2 * (60 / 3)) 
    (H3 : ∀ good_yield_per_acre, good_yield_per_acre = 400) 
    (H4 : ∀ clay_yield_per_acre, clay_yield_per_acre = 200) 
    (H5 : ∀ total_yield, total_yield = (2 * (A / 3) * 400 + (A / 3) * 200)) 
    : total_yield = 20000 :=
by 
  sorry

end jorge_acres_l14_14700


namespace dyslexian_alphabet_size_l14_14017

theorem dyslexian_alphabet_size (c v : ℕ) (h1 : (c * v * c * v * c + v * c * v * c * v) = 4800) : c + v = 12 :=
by
  sorry

end dyslexian_alphabet_size_l14_14017


namespace find_real_solutions_l14_14445

theorem find_real_solutions (x : ℝ) : 
  x^4 + (3 - x)^4 = 130 ↔ x = 1.5 + Real.sqrt 1.5 ∨ x = 1.5 - Real.sqrt 1.5 :=
sorry

end find_real_solutions_l14_14445


namespace ratio_hooper_bay_to_other_harbors_l14_14103

-- Definitions based on conditions
def other_harbors_lobster : ℕ := 80
def total_lobster : ℕ := 480
def combined_other_harbors_lobster := 2 * other_harbors_lobster
def hooper_bay_lobster := total_lobster - combined_other_harbors_lobster

-- The theorem to prove
theorem ratio_hooper_bay_to_other_harbors : hooper_bay_lobster / combined_other_harbors_lobster = 2 :=
by
  sorry

end ratio_hooper_bay_to_other_harbors_l14_14103


namespace percentage_of_boys_currently_l14_14866

theorem percentage_of_boys_currently (B G : ℕ) (h1 : B + G = 50) (h2 : B + 50 = 95) : (B / 50) * 100 = 90 := by
  sorry

end percentage_of_boys_currently_l14_14866


namespace count_four_digit_multiples_of_5_l14_14044

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l14_14044


namespace cos_theta_value_l14_14642

noncomputable def coefficient_x2 (θ : ℝ) : ℝ := Nat.choose 5 2 * (Real.cos θ)^2
noncomputable def coefficient_x3 : ℝ := Nat.choose 4 3 * (5 / 4 : ℝ)^3

theorem cos_theta_value (θ : ℝ) (h : coefficient_x2 θ = coefficient_x3) : 
  Real.cos θ = (Real.sqrt 2)/2 ∨ Real.cos θ = -(Real.sqrt 2)/2 := 
by sorry

end cos_theta_value_l14_14642


namespace proof_problem_l14_14325

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, (y = 2^x - 1) ∧ (x ≤ 2)}

-- Define the complement of set A in U
def complement_A : Set ℝ := Set.compl A

-- Define the intersection of complement_A and B
def complement_A_inter_B : Set ℝ := complement_A ∩ B

-- State the theorem
theorem proof_problem : complement_A_inter_B = {x | (-1 < x) ∧ (x ≤ 2)} :=
by
  sorry

end proof_problem_l14_14325


namespace simplification_l14_14429

theorem simplification (b : ℝ) : 3 * b * (3 * b^3 + 2 * b) - 2 * b^2 = 9 * b^4 + 4 * b^2 :=
by
  sorry

end simplification_l14_14429


namespace george_blocks_l14_14015

theorem george_blocks (num_boxes : ℕ) (blocks_per_box : ℕ) (total_blocks : ℕ) :
  num_boxes = 2 → blocks_per_box = 6 → total_blocks = num_boxes * blocks_per_box → total_blocks = 12 := by
  intros h_num_boxes h_blocks_per_box h_blocks_equal
  rw [h_num_boxes, h_blocks_per_box] at h_blocks_equal
  exact h_blocks_equal

end george_blocks_l14_14015


namespace new_boxes_of_markers_l14_14901

theorem new_boxes_of_markers (initial_markers new_markers_per_box total_markers : ℕ) 
    (h_initial : initial_markers = 32) 
    (h_new_markers_per_box : new_markers_per_box = 9)
    (h_total : total_markers = 86) :
  (total_markers - initial_markers) / new_markers_per_box = 6 := 
by
  sorry

end new_boxes_of_markers_l14_14901


namespace stick_length_l14_14232

theorem stick_length (x : ℕ) (h1 : 2 * x + (2 * x - 1) = 14) : x = 3 := sorry

end stick_length_l14_14232


namespace molecular_weight_is_correct_l14_14821

noncomputable def molecular_weight_of_compound : ℝ :=
  3 * 39.10 + 2 * 51.996 + 7 * 15.999 + 4 * 1.008 + 1 * 14.007

theorem molecular_weight_is_correct : molecular_weight_of_compound = 351.324 := 
by
  sorry

end molecular_weight_is_correct_l14_14821


namespace find_number_l14_14508

theorem find_number (x : ℝ) (h : 0.50 * x = 0.30 * 50 + 13) : x = 56 :=
by
  sorry

end find_number_l14_14508


namespace percentage_loss_is_correct_l14_14408

noncomputable def initial_cost : ℝ := 300
noncomputable def selling_price : ℝ := 255
noncomputable def loss : ℝ := initial_cost - selling_price
noncomputable def percentage_loss : ℝ := (loss / initial_cost) * 100

theorem percentage_loss_is_correct :
  percentage_loss = 15 :=
sorry

end percentage_loss_is_correct_l14_14408


namespace largest_divisor_of_square_difference_l14_14174

theorem largest_divisor_of_square_difference (m n : ℤ) (hm : m % 2 = 0) (hn : n % 2 = 0) (h : n < m) : 
  ∃ d, ∀ m n, (m % 2 = 0) → (n % 2 = 0) → (n < m) → d ∣ (m^2 - n^2) ∧ ∀ k, (∀ m n, (m % 2 = 0) → (n % 2 = 0) → (n < m) → k ∣ (m^2 - n^2)) → k ≤ d :=
sorry

end largest_divisor_of_square_difference_l14_14174


namespace evaluate_g_at_5_l14_14507

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem evaluate_g_at_5 : g 5 = 15 :=
by
    -- proof steps here
    sorry

end evaluate_g_at_5_l14_14507


namespace abs_neg_two_l14_14923

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end abs_neg_two_l14_14923


namespace stratified_sampling_grade11_l14_14267

noncomputable def g10 : ℕ := 500
noncomputable def total_students : ℕ := 1350
noncomputable def g10_sample : ℕ := 120
noncomputable def ratio : ℚ := g10_sample / g10
noncomputable def g11 : ℕ := 450
noncomputable def g12 : ℕ := g11 - 50

theorem stratified_sampling_grade11 :
  g10 + g11 + g12 = total_students →
  (g10_sample / g10) = ratio →
  sample_g11 = g11 * ratio →
  sample_g11 = 108 :=
by
  sorry

end stratified_sampling_grade11_l14_14267


namespace probability_of_odd_sum_rows_columns_l14_14065

open BigOperators

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def probability_odd_sums : ℚ :=
  let even_arrangements := factorial 4
  let odd_positions := factorial 12
  let total_arrangements := factorial 16
  (even_arrangements * odd_positions : ℚ) / total_arrangements

theorem probability_of_odd_sum_rows_columns :
  probability_odd_sums = 1 / 1814400 :=
by
  sorry

end probability_of_odd_sum_rows_columns_l14_14065


namespace alternating_students_count_l14_14565

theorem alternating_students_count :
  let num_male := 4
  let num_female := 5
  let arrangements := Nat.factorial num_female * Nat.factorial num_male
  arrangements = 2880 :=
by
  sorry

end alternating_students_count_l14_14565


namespace remainder_when_divided_by_r_minus_2_l14_14656

-- Define polynomial p(r)
def p (r : ℝ) : ℝ := r ^ 11 - 3

-- The theorem stating the problem
theorem remainder_when_divided_by_r_minus_2 : p 2 = 2045 := by
  sorry

end remainder_when_divided_by_r_minus_2_l14_14656


namespace percentage_who_do_not_have_job_of_choice_have_university_diploma_l14_14774

theorem percentage_who_do_not_have_job_of_choice_have_university_diploma :
  ∀ (total_population university_diploma job_of_choice no_diploma_job_of_choice : ℝ),
    total_population = 100 →
    job_of_choice = 40 →
    no_diploma_job_of_choice = 10 →
    university_diploma = 48 →
    ((university_diploma - (job_of_choice - no_diploma_job_of_choice)) / (total_population - job_of_choice)) * 100 = 30 :=
by
  intros total_population university_diploma job_of_choice no_diploma_job_of_choice h1 h2 h3 h4
  sorry

end percentage_who_do_not_have_job_of_choice_have_university_diploma_l14_14774


namespace min_value_inv_sum_l14_14474

open Real

theorem min_value_inv_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  3 ≤ (1 / x) + (1 / y) + (1 / z) :=
sorry

end min_value_inv_sum_l14_14474


namespace inverse_function_passes_through_point_l14_14640

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1)

theorem inverse_function_passes_through_point {a : ℝ} (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f a (-1) = 1) :
  f a⁻¹ 1 = -1 :=
sorry

end inverse_function_passes_through_point_l14_14640


namespace final_pen_count_l14_14887

theorem final_pen_count
  (initial_pens : ℕ := 7) 
  (mike_given_pens : ℕ := 22) 
  (doubled_pens : ℕ := 2)
  (sharon_given_pens : ℕ := 19) :
  let total_after_mike := initial_pens + mike_given_pens
  let total_after_cindy := total_after_mike * doubled_pens
  let final_count := total_after_cindy - sharon_given_pens
  final_count = 39 :=
by
  sorry

end final_pen_count_l14_14887


namespace total_toys_per_week_l14_14603

def toys_per_day := 1100
def working_days_per_week := 5

theorem total_toys_per_week : toys_per_day * working_days_per_week = 5500 :=
by
  sorry

end total_toys_per_week_l14_14603


namespace gasoline_tank_capacity_l14_14873

-- Given conditions
def initial_fraction_full := 5 / 6
def used_gallons := 15
def final_fraction_full := 2 / 3

-- Mathematical problem statement in Lean 4
theorem gasoline_tank_capacity (x : ℝ)
  (initial_full : initial_fraction_full * x = 5 / 6 * x)
  (final_full : initial_fraction_full * x - used_gallons = final_fraction_full * x) :
  x = 90 := by
  sorry

end gasoline_tank_capacity_l14_14873


namespace ajith_rana_meet_l14_14265

/--
Ajith and Rana walk around a circular course 115 km in circumference, starting together from the same point.
Ajith walks at 4 km/h, and Rana walks at 5 km/h in the same direction.
Prove that they will meet after 115 hours.
-/
theorem ajith_rana_meet 
  (course_circumference : ℕ)
  (ajith_speed : ℕ)
  (rana_speed : ℕ)
  (relative_speed : ℕ)
  (time : ℕ)
  (start_point : Point)
  (ajith : Person)
  (rana : Person)
  (walk_in_same_direction : Prop)
  (start_time : ℕ)
  (meet_time : ℕ) :
  course_circumference = 115 →
  ajith_speed = 4 →
  rana_speed = 5 →
  relative_speed = rana_speed - ajith_speed →
  time = course_circumference / relative_speed →
  meet_time = start_time + time →
  meet_time = 115 :=
by
  sorry

end ajith_rana_meet_l14_14265


namespace production_today_is_correct_l14_14673

theorem production_today_is_correct (n : ℕ) (P : ℕ) (T : ℕ) (average_daily_production : ℕ) (new_average_daily_production : ℕ) (h1 : n = 3) (h2 : average_daily_production = 70) (h3 : new_average_daily_production = 75) (h4 : P = n * average_daily_production) (h5 : P + T = (n + 1) * new_average_daily_production) : T = 90 :=
by
  sorry

end production_today_is_correct_l14_14673


namespace cyclic_sum_inequality_l14_14707

theorem cyclic_sum_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ( (b + c - a)^2 / (a^2 + (b + c)^2) +
    (c + a - b)^2 / (b^2 + (c + a)^2) +
    (a + b - c)^2 / (c^2 + (a + b)^2) ) ≥ 3 / 5 :=
  sorry

end cyclic_sum_inequality_l14_14707


namespace calculate_force_l14_14802

noncomputable def force_on_dam (ρ g a b h : ℝ) : ℝ :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

theorem calculate_force : force_on_dam 1000 10 4.8 7.2 3.0 = 252000 := 
  by 
  sorry

end calculate_force_l14_14802


namespace Billy_current_age_l14_14981

variable (B : ℕ)

theorem Billy_current_age 
  (h1 : ∃ B, 4 * B - B = 12) : B = 4 := by
  sorry

end Billy_current_age_l14_14981


namespace total_cost_correct_l14_14917

-- Define the conditions
def total_employees : ℕ := 300
def emp_12_per_hour : ℕ := 200
def emp_14_per_hour : ℕ := 40
def emp_17_per_hour : ℕ := total_employees - emp_12_per_hour - emp_14_per_hour

def wage_12_per_hour : ℕ := 12
def wage_14_per_hour : ℕ := 14
def wage_17_per_hour : ℕ := 17

def hours_per_shift : ℕ := 8

-- Define the cost calculations
def cost_12 : ℕ := emp_12_per_hour * wage_12_per_hour * hours_per_shift
def cost_14 : ℕ := emp_14_per_hour * wage_14_per_hour * hours_per_shift
def cost_17 : ℕ := emp_17_per_hour * wage_17_per_hour * hours_per_shift

def total_cost : ℕ := cost_12 + cost_14 + cost_17

-- The theorem to be proved
theorem total_cost_correct :
  total_cost = 31840 :=
by
  sorry

end total_cost_correct_l14_14917


namespace train_pass_time_l14_14356

-- Definitions based on the conditions
def train_length : ℕ := 280  -- train length in meters
def train_speed_kmh : ℕ := 72  -- train speed in km/hr
noncomputable def train_speed_ms : ℚ := (train_speed_kmh * 5 / 18)  -- train speed in m/s

-- Theorem statement
theorem train_pass_time : (train_length / train_speed_ms) = 14 := by
  sorry

end train_pass_time_l14_14356


namespace simple_interest_years_l14_14275

variable (P R T : ℕ)
variable (deltaI : ℕ := 400)
variable (P_value : P = 800)

theorem simple_interest_years 
  (h : (800 * (R + 5) * T / 100) = (800 * R * T / 100) + 400) :
  T = 10 :=
by sorry

end simple_interest_years_l14_14275


namespace body_diagonal_length_l14_14484

theorem body_diagonal_length (a b c : ℝ) (h1 : a * b = 6) (h2 : a * c = 8) (h3 : b * c = 12) :
  (a^2 + b^2 + c^2 = 29) :=
by
  sorry

end body_diagonal_length_l14_14484


namespace largest_multiple_l14_14575

theorem largest_multiple (a b limit : ℕ) (ha : a = 3) (hb : b = 5) (h_limit : limit = 800) : 
  ∃ (n : ℕ), (lcm a b) * n < limit ∧ (lcm a b) * (n + 1) ≥ limit ∧ (lcm a b) * n = 795 := 
by 
  sorry

end largest_multiple_l14_14575


namespace set_contains_difference_of_elements_l14_14777

variable {A : Set Int}

axiom cond1 (a : Int) (ha : a ∈ A) : 2 * a ∈ A
axiom cond2 (a b : Int) (ha : a ∈ A) (hb : b ∈ A) : a + b ∈ A

theorem set_contains_difference_of_elements 
  (a b : Int) (ha : a ∈ A) (hb : b ∈ A) : a - b ∈ A := by
  sorry

end set_contains_difference_of_elements_l14_14777


namespace price_increase_decrease_l14_14862

theorem price_increase_decrease (P : ℝ) (x : ℝ) (h : P > 0) :
  (P * (1 + x / 100) * (1 - x / 100) = 0.64 * P) → (x = 60) :=
by
  sorry

end price_increase_decrease_l14_14862


namespace probability_reach_3_1_in_8_steps_l14_14595

theorem probability_reach_3_1_in_8_steps :
  let m := 35
  let n := 2048
  let q := m / n
  ∃ (m n : ℕ), (Nat.gcd m n = 1) ∧ (q = 35 / 2048) ∧ (m + n = 2083) := by
  sorry

end probability_reach_3_1_in_8_steps_l14_14595


namespace arithmetic_sequence_middle_term_l14_14609

theorem arithmetic_sequence_middle_term 
  (a b c d e : ℕ) 
  (h_seq : a = 23 ∧ e = 53 ∧ (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d)) :
  c = 38 :=
by
  sorry

end arithmetic_sequence_middle_term_l14_14609


namespace stationery_store_sales_l14_14532

theorem stationery_store_sales :
  let price_pencil_eraser := 0.8
  let price_regular_pencil := 0.5
  let price_short_pencil := 0.4
  let num_pencil_eraser := 200
  let num_regular_pencil := 40
  let num_short_pencil := 35
  (num_pencil_eraser * price_pencil_eraser) +
  (num_regular_pencil * price_regular_pencil) +
  (num_short_pencil * price_short_pencil) = 194 :=
by
  sorry

end stationery_store_sales_l14_14532


namespace determine_a_values_l14_14779

theorem determine_a_values (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = { x | abs x = 1 }) 
  (hB : B = { x | a * x = 1 }) 
  (h_superset : A ⊇ B) :
  a = -1 ∨ a = 0 ∨ a = 1 :=
sorry

end determine_a_values_l14_14779


namespace angle_is_50_l14_14985

-- Define the angle, supplement, and complement
def angle (x : ℝ) := x
def supplement (x : ℝ) := 180 - x
def complement (x : ℝ) := 90 - x
def condition (x : ℝ) := supplement x = 3 * (complement x) + 10

theorem angle_is_50 :
  ∃ x : ℝ, condition x ∧ x = 50 :=
by
  -- Here we show the existence of x that satisfies the condition and is equal to 50
  sorry

end angle_is_50_l14_14985


namespace opposite_sides_line_l14_14874

theorem opposite_sides_line (m : ℝ) : 
  (2 * 1 + 3 + m) * (2 * -4 + -2 + m) < 0 ↔ -5 < m ∧ m < 10 :=
by sorry

end opposite_sides_line_l14_14874


namespace bus_speed_excluding_stoppages_l14_14499

variable (v : ℝ)

-- Given conditions
def speed_including_stoppages := 45 -- kmph
def stoppage_time_ratio := 1/6 -- 10 minutes per hour is 1/6 of the time

-- Prove that the speed excluding stoppages is 54 kmph
theorem bus_speed_excluding_stoppages (h1 : speed_including_stoppages = 45) 
                                      (h2 : stoppage_time_ratio = 1/6) : 
                                      v = 54 := by
  sorry

end bus_speed_excluding_stoppages_l14_14499


namespace expected_value_is_correct_l14_14561

noncomputable def expected_winnings : ℚ :=
  (5/12 : ℚ) * 2 + (1/3 : ℚ) * 0 + (1/6 : ℚ) * (-2) + (1/12 : ℚ) * 10

theorem expected_value_is_correct : expected_winnings = 4 / 3 := 
by 
  -- Complex calculations skipped for brevity
  sorry

end expected_value_is_correct_l14_14561


namespace largest_remainder_division_by_11_l14_14112

theorem largest_remainder_division_by_11 (A B C : ℕ) (h : A = 11 * B + C) (hC : 0 ≤ C ∧ C < 11) : C ≤ 10 :=
  sorry

end largest_remainder_division_by_11_l14_14112


namespace solve_xyz_eq_x_plus_y_l14_14202

theorem solve_xyz_eq_x_plus_y (x y z : ℕ) (h1 : x * y * z = x + y) (h2 : x ≤ y) : (x = 2 ∧ y = 2 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 2) :=
by {
    sorry -- The actual proof goes here
}

end solve_xyz_eq_x_plus_y_l14_14202


namespace pairs_of_real_numbers_l14_14594

theorem pairs_of_real_numbers (a b : ℝ) (h : ∀ (n : ℕ), n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)) :
  a = 0 ∨ b = 0 ∨ a = b ∨ (∃ m n : ℤ, a = (m : ℝ) ∧ b = (n : ℝ)) :=
by
  sorry

end pairs_of_real_numbers_l14_14594


namespace dogs_running_l14_14801

theorem dogs_running (total_dogs playing_with_toys barking not_doing_anything running : ℕ)
  (h1 : total_dogs = 88)
  (h2 : playing_with_toys = total_dogs / 2)
  (h3 : barking = total_dogs / 4)
  (h4 : not_doing_anything = 10)
  (h5 : running = total_dogs - playing_with_toys - barking - not_doing_anything) :
  running = 12 :=
sorry

end dogs_running_l14_14801


namespace speed_in_still_water_l14_14846

-- Defining the terms as given conditions in the problem
def speed_downstream (v_m v_s : ℝ) : ℝ := v_m + v_s
def speed_upstream (v_m v_s : ℝ) : ℝ := v_m - v_s

-- Given conditions translated into Lean definitions
def downstream_condition : Prop :=
  ∃ (v_m v_s : ℝ), speed_downstream v_m v_s = 7

def upstream_condition : Prop :=
  ∃ (v_m v_s : ℝ), speed_upstream v_m v_s = 4

-- The problem statement to prove
theorem speed_in_still_water : 
  downstream_condition ∧ upstream_condition → ∃ v_m : ℝ, v_m = 5.5 :=
by 
  intros
  sorry

end speed_in_still_water_l14_14846


namespace intersection_M_N_l14_14272

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_M_N : M ∩ N = Set.Ico 1 3 := 
by
  sorry

end intersection_M_N_l14_14272


namespace find_frac_a_b_c_l14_14067

theorem find_frac_a_b_c (a b c : ℝ) (h1 : a = 2 * b) (h2 : a^2 + b^2 = c^2) : (a + b) / c = (3 * Real.sqrt 5) / 5 :=
by
  sorry

end find_frac_a_b_c_l14_14067


namespace find_a_l14_14625

theorem find_a (a b c : ℝ) (h1 : b = 15) (h2 : c = 5)
  (h3 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) 
  (result : a * 15 * 5 = 2) : a = 6 := by 
  sorry

end find_a_l14_14625


namespace intersection_of_A_and_complement_of_B_l14_14382

noncomputable def U : Set ℝ := Set.univ

noncomputable def A : Set ℝ := { x : ℝ | 2^x * (x - 2) < 1 }
noncomputable def B : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (1 - x) }
noncomputable def B_complement : Set ℝ := { x : ℝ | x ≥ 1 }

theorem intersection_of_A_and_complement_of_B :
  A ∩ B_complement = { x : ℝ | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_complement_of_B_l14_14382


namespace find_velocity_l14_14378

variable (k V : ℝ)
variable (P A : ℕ)

theorem find_velocity (k_eq : k = 1 / 200) 
  (initial_cond : P = 4 ∧ A = 2 ∧ V = 20) 
  (new_cond : P = 16 ∧ A = 4) : 
  V = 20 * Real.sqrt 2 :=
by
  sorry

end find_velocity_l14_14378


namespace nominal_rate_of_interest_annual_l14_14973

theorem nominal_rate_of_interest_annual (EAR nominal_rate : ℝ) (n : ℕ) (h1 : EAR = 0.0816) (h2 : n = 2) : 
  nominal_rate = 0.0796 :=
by 
  sorry

end nominal_rate_of_interest_annual_l14_14973


namespace dragon_2023_first_reappearance_l14_14007

theorem dragon_2023_first_reappearance :
  let cycle_letters := 6
  let cycle_digits := 4
  Nat.lcm cycle_letters cycle_digits = 12 :=
by
  rfl -- since LCM of 6 and 4 directly calculates to 12

end dragon_2023_first_reappearance_l14_14007


namespace average_headcount_is_correct_l14_14436

/-- The student headcount data for the specified semesters -/
def student_headcount : List ℕ := [11700, 10900, 11500, 10500, 11600, 10700, 11300]

noncomputable def average_headcount : ℕ :=
  (student_headcount.sum) / student_headcount.length

theorem average_headcount_is_correct : average_headcount = 11029 := by
  sorry

end average_headcount_is_correct_l14_14436


namespace perpendicular_vectors_find_a_l14_14641

theorem perpendicular_vectors_find_a
  (a : ℝ)
  (m : ℝ × ℝ := (1, 2))
  (n : ℝ × ℝ := (a, -1))
  (h : m.1 * n.1 + m.2 * n.2 = 0) :
  a = 2 := 
sorry

end perpendicular_vectors_find_a_l14_14641


namespace leftover_potatoes_l14_14069

theorem leftover_potatoes (fries_per_potato : ℕ) (total_potatoes : ℕ) (required_fries : ℕ)
    (h1 : fries_per_potato = 25) (h2 : total_potatoes = 15) (h3 : required_fries = 200) :
    (total_potatoes - required_fries / fries_per_potato) = 7 :=
sorry

end leftover_potatoes_l14_14069


namespace coefficient_of_determination_l14_14897

-- Define the observations and conditions for the problem
def observations (n : ℕ) := 
  {x : ℕ → ℝ // ∃ b a : ℝ, ∀ i : ℕ, i < n → ∃ y_i : ℝ, y_i = b * x i + a}

/-- 
  Given a set of observations (x_1, y_1), (x_2, y_2), ..., (x_n, y_n) 
  that satisfies the equation y_i = bx_i + a for i = 1, 2, ..., n, 
  prove that the coefficient of determination R² is 1.
-/
theorem coefficient_of_determination (n : ℕ) (obs : observations n) : 
  ∃ R_squared : ℝ, R_squared = 1 :=
sorry

end coefficient_of_determination_l14_14897


namespace find_n_l14_14421

theorem find_n : (∃ n : ℕ, 2^3 * 8^3 = 2^(2 * n)) ↔ n = 6 :=
by
  sorry

end find_n_l14_14421


namespace find_t_l14_14159

-- Given: (1) g(x) = x^5 + px^4 + qx^3 + rx^2 + sx + t with all roots being negative integers
--        (2) p + q + r + s + t = 3024
-- Prove: t = 1600

noncomputable def poly (x : ℝ) (p q r s t : ℝ) := 
  x^5 + p*x^4 + q*x^3 + r*x^2 + s*x + t

theorem find_t
  (p q r s t : ℝ)
  (roots_neg_int : ∀ root, root ∈ [-s1, -s2, -s3, -s4, -s5] → (root : ℤ) < 0)
  (sum_coeffs : p + q + r + s + t = 3024)
  (poly_1_eq : poly 1 p q r s t = 3025) :
  t = 1600 := 
sorry

end find_t_l14_14159


namespace remainder_when_divided_by_296_and_37_l14_14626

theorem remainder_when_divided_by_296_and_37 (N : ℤ) (k : ℤ)
  (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_when_divided_by_296_and_37_l14_14626


namespace tetrahedron_fourth_face_possibilities_l14_14160

theorem tetrahedron_fourth_face_possibilities :
  ∃ (S : Set String), S = {"right-angled triangle", "acute-angled triangle", "isosceles triangle", "isosceles right-angled triangle", "equilateral triangle"} :=
sorry

end tetrahedron_fourth_face_possibilities_l14_14160


namespace factory_fills_boxes_per_hour_l14_14558

theorem factory_fills_boxes_per_hour
  (colors_per_box : ℕ)
  (crayons_per_color : ℕ)
  (total_crayons : ℕ)
  (hours : ℕ)
  (crayons_per_hour := total_crayons / hours)
  (crayons_per_box := colors_per_box * crayons_per_color)
  (boxes_per_hour := crayons_per_hour / crayons_per_box) :
  colors_per_box = 4 →
  crayons_per_color = 2 →
  total_crayons = 160 →
  hours = 4 →
  boxes_per_hour = 5 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end factory_fills_boxes_per_hour_l14_14558


namespace relationship_a_b_l14_14194

-- Definitions of the two quadratic equations having a single common root
def has_common_root (a b : ℝ) : Prop :=
  ∃ t : ℝ, (t^2 + a * t + b = 0) ∧ (t^2 + b * t + a = 0)

-- Theorem stating the relationship between a and b
theorem relationship_a_b (a b : ℝ) (h : has_common_root a b) : a ≠ b → a + b + 1 = 0 :=
by sorry

end relationship_a_b_l14_14194
