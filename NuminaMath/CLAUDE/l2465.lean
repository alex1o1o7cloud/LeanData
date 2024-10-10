import Mathlib

namespace assembly_line_production_rate_l2465_246530

/-- Assembly line production problem -/
theorem assembly_line_production_rate 
  (initial_rate : ℝ) 
  (initial_order : ℝ) 
  (second_order : ℝ) 
  (average_output : ℝ) 
  (h1 : initial_rate = 20)
  (h2 : initial_order = 60)
  (h3 : second_order = 60)
  (h4 : average_output = 30) :
  let total_cogs := initial_order + second_order
  let total_time := total_cogs / average_output
  let initial_time := initial_order / initial_rate
  let second_time := total_time - initial_time
  second_order / second_time = 60 := by
  sorry

end assembly_line_production_rate_l2465_246530


namespace unique_number_exists_l2465_246510

theorem unique_number_exists : ∃! n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  n / sum = 2 * diff ∧ n % sum = 80 ∧ n = 220080 := by
  sorry

end unique_number_exists_l2465_246510


namespace travel_time_calculation_l2465_246537

theorem travel_time_calculation (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 540)
  (h2 : speed1 = 45)
  (h3 : speed2 = 30) :
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 15 := by
  sorry

end travel_time_calculation_l2465_246537


namespace other_class_size_l2465_246574

theorem other_class_size (avg_zits_other : ℝ) (avg_zits_jones : ℝ) 
  (zits_diff : ℕ) (jones_kids : ℕ) :
  avg_zits_other = 5 →
  avg_zits_jones = 6 →
  zits_diff = 67 →
  jones_kids = 32 →
  ∃ other_kids : ℕ, 
    (jones_kids : ℝ) * avg_zits_jones = 
    (other_kids : ℝ) * avg_zits_other + zits_diff ∧
    other_kids = 25 := by
  sorry

end other_class_size_l2465_246574


namespace loss_per_metre_cloth_l2465_246546

/-- Calculates the loss per metre of cloth for a shopkeeper. -/
theorem loss_per_metre_cloth (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) 
  (h1 : total_metres = 300)
  (h2 : total_selling_price = 9000)
  (h3 : cost_price_per_metre = 36) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 6 := by
  sorry

#check loss_per_metre_cloth

end loss_per_metre_cloth_l2465_246546


namespace collection_distribution_l2465_246518

def karl_stickers : ℕ := 25
def karl_cards : ℕ := 15
def karl_keychains : ℕ := 5
def karl_stamps : ℕ := 10

def ryan_stickers : ℕ := karl_stickers + 20
def ryan_cards : ℕ := karl_cards - 10
def ryan_keychains : ℕ := karl_keychains + 2
def ryan_stamps : ℕ := karl_stamps

def ben_stickers : ℕ := ryan_stickers - 10
def ben_cards : ℕ := ryan_cards / 2
def ben_keychains : ℕ := karl_keychains * 2
def ben_stamps : ℕ := karl_stamps + 5

def total_items : ℕ := karl_stickers + karl_cards + karl_keychains + karl_stamps +
                       ryan_stickers + ryan_cards + ryan_keychains + ryan_stamps +
                       ben_stickers + ben_cards + ben_keychains + ben_stamps

def num_collectors : ℕ := 4

theorem collection_distribution :
  total_items = 184 ∧ total_items % num_collectors = 0 ∧ total_items / num_collectors = 46 := by
  sorry

end collection_distribution_l2465_246518


namespace subtraction_result_l2465_246594

theorem subtraction_result : 2.43 - 1.2 = 1.23 := by
  sorry

end subtraction_result_l2465_246594


namespace tommy_truck_count_l2465_246593

/-- The number of trucks Tommy saw -/
def num_trucks : ℕ := 12

/-- The number of cars Tommy saw -/
def num_cars : ℕ := 13

/-- The total number of wheels Tommy saw -/
def total_wheels : ℕ := 100

/-- The number of wheels per vehicle -/
def wheels_per_vehicle : ℕ := 4

theorem tommy_truck_count :
  num_trucks * wheels_per_vehicle + num_cars * wheels_per_vehicle = total_wheels :=
by sorry

end tommy_truck_count_l2465_246593


namespace expression_evaluation_l2465_246556

theorem expression_evaluation : (28 / (8 - 3 + 2)) * (4 - 1) = 12 := by
  sorry

end expression_evaluation_l2465_246556


namespace perfect_square_divisibility_l2465_246581

theorem perfect_square_divisibility (a b : ℕ+) 
  (h : (2 * a * b) ∣ (a^2 + b^2 - a)) : 
  ∃ k : ℕ+, a = k^2 := by
  sorry

end perfect_square_divisibility_l2465_246581


namespace stock_price_problem_l2465_246569

theorem stock_price_problem (initial_price : ℝ) : 
  let day1 := initial_price * (1 + 1/10)
  let day2 := day1 * (1 - 1/11)
  let day3 := day2 * (1 + 1/12)
  let day4 := day3 * (1 - 1/13)
  day4 = 5000 → initial_price = 5000 := by
sorry

end stock_price_problem_l2465_246569


namespace complex_zero_of_polynomial_l2465_246504

def is_valid_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℤ), ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + (P 0)

theorem complex_zero_of_polynomial 
  (P : ℝ → ℝ) 
  (h_valid : is_valid_polynomial P) 
  (h_zero1 : P 3 = 0) 
  (h_zero2 : P (-1) = 0) : 
  P (3/2) = (15 : ℝ)/4 :=
sorry

end complex_zero_of_polynomial_l2465_246504


namespace sqrt_of_sixteen_l2465_246542

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_sixteen_l2465_246542


namespace trajectory_of_m_l2465_246590

/-- The trajectory of point M given conditions on triangle MAB -/
theorem trajectory_of_m (x y : ℝ) (hx : x ≠ 3 ∧ x ≠ -3) (hy : y ≠ 0) :
  (y / (x + 3)) * (y / (x - 3)) = 4 →
  x^2 / 9 - y^2 / 36 = 1 :=
by sorry

end trajectory_of_m_l2465_246590


namespace inverse_proportion_solution_l2465_246514

/-- Given two inversely proportional quantities, this is their constant product. -/
def InverseProportionConstant (x y : ℝ) : ℝ := x * y

theorem inverse_proportion_solution (a b c : ℝ → ℝ) :
  (∀ x y, InverseProportionConstant (a x) (b x) = InverseProportionConstant (a y) (b y)) →
  (∀ x y, InverseProportionConstant (b x) (c x) = InverseProportionConstant (b y) (c y)) →
  a 1 = 40 →
  b 1 = 5 →
  c 2 = 10 →
  b 2 = 7 →
  c 3 = 5.6 →
  a 3 = 16 :=
by sorry

end inverse_proportion_solution_l2465_246514


namespace eliminate_denominators_l2465_246500

theorem eliminate_denominators (x : ℝ) : 
  ((x + 1) / 2 + 1 = x / 3) ↔ (3 * (x + 1) + 6 = 2 * x) := by sorry

end eliminate_denominators_l2465_246500


namespace jigsaw_puzzle_pieces_l2465_246520

/-- The number of pieces in Luke's jigsaw puzzle -/
def P : ℕ := sorry

/-- The fraction of pieces remaining after each day -/
def remaining_pieces (day : ℕ) : ℚ :=
  match day with
  | 0 => 1
  | 1 => 0.9
  | 2 => 0.72
  | 3 => 0.504
  | _ => 0

theorem jigsaw_puzzle_pieces :
  P = 1000 ∧
  remaining_pieces 1 = 0.9 ∧
  remaining_pieces 2 = 0.72 ∧
  remaining_pieces 3 = 0.504 ∧
  (remaining_pieces 3 * P : ℚ) = 504 :=
by sorry

end jigsaw_puzzle_pieces_l2465_246520


namespace cubic_polynomial_special_case_l2465_246503

-- Define a cubic polynomial
def cubic_polynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d

-- Define the theorem
theorem cubic_polynomial_special_case (p : ℝ → ℝ) 
  (h_cubic : cubic_polynomial p)
  (h1 : p 1 = 1)
  (h2 : p 2 = 1/8)
  (h3 : p 3 = 1/27) :
  p 4 = 1/576 := by
  sorry

end cubic_polynomial_special_case_l2465_246503


namespace smallest_number_l2465_246587

theorem smallest_number (a b c d : ℝ) (ha : a = Real.sqrt 2) (hb : b = 0) (hc : c = -1) (hd : d = 2) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end smallest_number_l2465_246587


namespace roses_in_vase_l2465_246552

/-- The number of roses in the vase initially -/
def initial_roses : ℕ := 9

/-- The number of orchids in the vase initially -/
def initial_orchids : ℕ := 6

/-- The number of orchids in the vase now -/
def current_orchids : ℕ := 13

/-- The difference between the number of orchids and roses in the vase now -/
def orchid_rose_difference : ℕ := 10

/-- The number of roses in the vase now -/
def current_roses : ℕ := 3

theorem roses_in_vase :
  current_roses = current_orchids - orchid_rose_difference := by
  sorry

end roses_in_vase_l2465_246552


namespace sine_cosine_simplification_l2465_246521

theorem sine_cosine_simplification (x y : ℝ) :
  Real.sin (x + y) * Real.cos y - Real.cos (x + y) * Real.sin y = Real.sin x := by
  sorry

end sine_cosine_simplification_l2465_246521


namespace max_at_one_implies_c_equals_three_l2465_246526

/-- The function f(x) defined as x(x-c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) with respect to x --/
def f' (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem max_at_one_implies_c_equals_three (c : ℝ) :
  (∀ x : ℝ, f c x ≤ f c 1) →
  (f' c 1 = 0) →
  c = 3 := by
  sorry

end max_at_one_implies_c_equals_three_l2465_246526


namespace expansion_terms_imply_n_equals_10_l2465_246505

theorem expansion_terms_imply_n_equals_10 (x a : ℝ) (n : ℕ) :
  (n.choose 1 : ℝ) * x^(n - 1) * a = 210 →
  (n.choose 2 : ℝ) * x^(n - 2) * a^2 = 840 →
  (n.choose 3 : ℝ) * x^(n - 3) * a^3 = 2520 →
  n = 10 := by
sorry

end expansion_terms_imply_n_equals_10_l2465_246505


namespace tetrahedron_PQRS_volume_l2465_246592

/-- The volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (a b c d e f : ℝ) : ℝ :=
  (1 / 6) * Real.sqrt (
    a^2 * b^2 * c^2 + a^2 * d^2 * e^2 + b^2 * d^2 * f^2 + c^2 * e^2 * f^2
    - a^2 * (d^2 * e^2 + d^2 * f^2 + e^2 * f^2)
    - b^2 * (c^2 * e^2 + c^2 * f^2 + e^2 * f^2)
    - c^2 * (b^2 * d^2 + b^2 * f^2 + d^2 * f^2)
    - d^2 * (a^2 * e^2 + a^2 * f^2 + e^2 * f^2)
    - e^2 * (a^2 * d^2 + a^2 * f^2 + d^2 * f^2)
    - f^2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2)
  )

theorem tetrahedron_PQRS_volume :
  let PQ : ℝ := 6
  let PR : ℝ := 4
  let PS : ℝ := (12 / 5) * Real.sqrt 2
  let QR : ℝ := 3
  let QS : ℝ := 4
  let RS : ℝ := (12 / 5) * Real.sqrt 5
  tetrahedronVolume PQ PR PS QR QS RS = 24 / 5 := by
  sorry

end tetrahedron_PQRS_volume_l2465_246592


namespace max_profit_theorem_l2465_246557

/-- Represents the flour factory problem --/
structure FlourFactory where
  totalWorkers : ℕ
  flourPerWorker : ℕ
  noodlesPerWorker : ℕ
  flourPricePerKg : ℚ
  noodlesPricePerKg : ℚ

/-- Calculates the daily profit based on the number of workers processing noodles --/
def dailyProfit (factory : FlourFactory) (noodleWorkers : ℕ) : ℚ :=
  let flourWorkers := factory.totalWorkers - noodleWorkers
  let flourProfit := (factory.flourPerWorker * flourWorkers : ℕ) * factory.flourPricePerKg
  let noodleProfit := (factory.noodlesPerWorker * noodleWorkers : ℕ) * factory.noodlesPricePerKg
  flourProfit + noodleProfit

/-- Theorem stating the maximum profit and optimal worker allocation --/
theorem max_profit_theorem (factory : FlourFactory) 
    (h1 : factory.totalWorkers = 20)
    (h2 : factory.flourPerWorker = 600)
    (h3 : factory.noodlesPerWorker = 400)
    (h4 : factory.flourPricePerKg = 1/5)
    (h5 : factory.noodlesPricePerKg = 3/5) :
    ∃ (optimalNoodleWorkers : ℕ),
      optimalNoodleWorkers = 12 ∧ 
      dailyProfit factory optimalNoodleWorkers = 384/5 ∧
      ∀ (n : ℕ), n ≤ factory.totalWorkers → 
        dailyProfit factory n ≤ dailyProfit factory optimalNoodleWorkers :=
  sorry


end max_profit_theorem_l2465_246557


namespace half_work_days_l2465_246517

/-- Represents the number of days it takes for the larger group to complete half the work -/
def days_for_half_work (original_days : ℕ) (efficiency_ratio : ℚ) : ℚ :=
  original_days / (2 * (1 + 2 * efficiency_ratio))

/-- Theorem stating that under the given conditions, it takes 4 days for the larger group to complete half the work -/
theorem half_work_days :
  days_for_half_work 20 (3/4) = 4 := by sorry

end half_work_days_l2465_246517


namespace grapes_remainder_l2465_246524

theorem grapes_remainder (josiah katelyn liam basket_size : ℕ) 
  (h_josiah : josiah = 54)
  (h_katelyn : katelyn = 67)
  (h_liam : liam = 29)
  (h_basket : basket_size = 15) : 
  (josiah + katelyn + liam) % basket_size = 0 := by
sorry

end grapes_remainder_l2465_246524


namespace inequality_and_equality_condition_l2465_246509

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) ∧ 
  (Real.sqrt ((a + c) * (b + d)) = Real.sqrt (a * b) + Real.sqrt (c * d) ↔ a * d = b * c) :=
by sorry

end inequality_and_equality_condition_l2465_246509


namespace imaginary_part_of_complex_fraction_l2465_246560

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i^2 = -1 →
  Complex.im (2 * i^3 / (i - 1)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l2465_246560


namespace jack_and_jill_speed_l2465_246583

/-- The speed of Jack and Jill's walk, given their conditions -/
theorem jack_and_jill_speed :
  ∀ x : ℝ,
  let jack_speed := x^2 - 13*x - 30
  let jill_distance := x^2 - 5*x - 84
  let jill_time := x + 7
  let jill_speed := jill_distance / jill_time
  (jack_speed = jill_speed) → (jack_speed = 6) :=
by
  sorry

end jack_and_jill_speed_l2465_246583


namespace calculate_second_oil_price_l2465_246516

/-- Given a mixture of two oils, calculate the price of the second oil -/
theorem calculate_second_oil_price (volume1 volume2 price1 price_mixture : ℝ) 
  (h1 : volume1 = 10)
  (h2 : volume2 = 5)
  (h3 : price1 = 55)
  (h4 : price_mixture = 58.67) : 
  ∃ (price2 : ℝ), price2 = 66.01 ∧ 
  (volume1 * price1 + volume2 * price2) / (volume1 + volume2) = price_mixture := by
  sorry

#check calculate_second_oil_price

end calculate_second_oil_price_l2465_246516


namespace intersection_point_satisfies_system_l2465_246544

/-- Given two linear functions and their intersection point, prove that the point satisfies a specific system of equations -/
theorem intersection_point_satisfies_system (a b : ℝ) : 
  (∃ x y : ℝ, y = 3 * x + 6 ∧ y = 2 * x - 4 ∧ x = a ∧ y = b) →
  (3 * a - b = -6 ∧ 2 * a - b - 4 = 0) := by
  sorry

end intersection_point_satisfies_system_l2465_246544


namespace amanda_candy_bars_l2465_246591

theorem amanda_candy_bars :
  let initial_candy_bars : ℕ := 7
  let first_giveaway : ℕ := 3
  let bought_candy_bars : ℕ := 30
  let second_giveaway_multiplier : ℕ := 4
  
  let remaining_after_first := initial_candy_bars - first_giveaway
  let second_giveaway := first_giveaway * second_giveaway_multiplier
  let remaining_after_second := bought_candy_bars - second_giveaway
  let total_kept := remaining_after_first + remaining_after_second

  total_kept = 22 := by
  sorry

end amanda_candy_bars_l2465_246591


namespace vector_problem_solution_l2465_246579

def vector_problem (a b : ℝ × ℝ) : Prop :=
  a ≠ (0, 0) ∧ b ≠ (0, 0) ∧
  a + b = (-3, 6) ∧ a - b = (-3, 2) →
  a.1^2 + a.2^2 - (b.1^2 + b.2^2) = 21

theorem vector_problem_solution :
  ∀ a b : ℝ × ℝ, vector_problem a b :=
by
  sorry

end vector_problem_solution_l2465_246579


namespace anne_weight_is_67_l2465_246576

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference between Anne's and Douglas's weights in pounds -/
def weight_difference : ℕ := 15

/-- Anne's weight in pounds -/
def anne_weight : ℕ := douglas_weight + weight_difference

theorem anne_weight_is_67 : anne_weight = 67 := by
  sorry

end anne_weight_is_67_l2465_246576


namespace bills_found_l2465_246572

def initial_amount : ℕ := 75
def final_amount : ℕ := 135
def bill_value : ℕ := 20

theorem bills_found : (final_amount - initial_amount) / bill_value = 3 := by
  sorry

end bills_found_l2465_246572


namespace total_price_of_hats_l2465_246535

theorem total_price_of_hats (total_hats : ℕ) (green_hats : ℕ) (blue_cost : ℕ) (green_cost : ℕ) :
  total_hats = 85 →
  green_hats = 40 →
  blue_cost = 6 →
  green_cost = 7 →
  (total_hats - green_hats) * blue_cost + green_hats * green_cost = 550 :=
by
  sorry

end total_price_of_hats_l2465_246535


namespace right_triangle_hypotenuse_l2465_246533

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a + b + c = 40 →
  (1/2) * a * b = 24 →
  a^2 + b^2 = c^2 →
  c = 18.8 := by
  sorry

end right_triangle_hypotenuse_l2465_246533


namespace pencils_leftover_l2465_246523

theorem pencils_leftover : 76394821 % 7 = 6 := by
  sorry

end pencils_leftover_l2465_246523


namespace total_coins_l2465_246584

def total_value : ℚ := 71
def value_20_paise : ℚ := 20 / 100
def value_25_paise : ℚ := 25 / 100
def num_20_paise : ℕ := 260

theorem total_coins : ∃ (num_25_paise : ℕ), 
  (num_20_paise : ℚ) * value_20_paise + (num_25_paise : ℚ) * value_25_paise = total_value ∧
  num_20_paise + num_25_paise = 336 :=
sorry

end total_coins_l2465_246584


namespace three_number_problem_l2465_246564

def is_solution (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 406 ∧
  ∃ p : ℕ, Nat.Prime p ∧ p > 2 ∧ p ∣ a ∧ p ∣ b ∧ p ∣ c ∧
    Nat.Prime (a / p) ∧ Nat.Prime (b / p) ∧ Nat.Prime (c / p)

theorem three_number_problem :
  ∃ a b c : ℕ, is_solution a b c ∧
    ((a = 14 ∧ b = 21 ∧ c = 371) ∨
     (a = 14 ∧ b = 91 ∧ c = 301) ∨
     (a = 14 ∧ b = 133 ∧ c = 259) ∨
     (a = 58 ∧ b = 145 ∧ c = 203)) :=
sorry

end three_number_problem_l2465_246564


namespace rectangle_area_l2465_246532

theorem rectangle_area (x y : ℝ) (h1 : y = (7/3) * x) (h2 : 2 * (x + y) = 40) : x * y = 84 := by
  sorry

end rectangle_area_l2465_246532


namespace not_equal_to_seven_fifths_l2465_246582

theorem not_equal_to_seven_fifths : ∃ x : ℚ, x ≠ 7/5 ∧
  (x = 1 + 3/8) ∧
  (14/10 = 7/5) ∧
  (1 + 2/5 = 7/5) ∧
  (1 + 6/15 = 7/5) ∧
  (1 + 28/20 = 7/5) :=
by
  sorry

end not_equal_to_seven_fifths_l2465_246582


namespace scientific_notation_2150000_l2465_246558

theorem scientific_notation_2150000 : 
  ∃ (a : ℝ) (n : ℤ), 2150000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof would go here
  sorry

end scientific_notation_2150000_l2465_246558


namespace constant_white_sectors_neighboring_minutes_l2465_246547

/-- Represents the color of a sector -/
inductive Color
| White
| Red

/-- Represents the state of the circle -/
def CircleState := Vector Color 1000

/-- Repaints 500 consecutive sectors in the circle -/
def repaint (state : CircleState) (start : Fin 1000) : CircleState :=
  sorry

/-- Counts the number of white sectors in the circle -/
def countWhite (state : CircleState) : Nat :=
  sorry

theorem constant_white_sectors_neighboring_minutes 
  (initial : CircleState)
  (repaints : ℕ → Fin 1000) 
  (n : ℕ) :
  (countWhite (repaint (repaint (repaint initial (repaints (n-1))) (repaints n)) (repaints (n+1))) = 
   countWhite (repaint (repaint initial (repaints (n-1))) (repaints n))) →
  ((countWhite (repaint (repaint initial (repaints (n-1))) (repaints n)) = 
    countWhite (repaint initial (repaints (n-1))))
   ∨ 
   (countWhite (repaint (repaint (repaint initial (repaints (n-1))) (repaints n)) (repaints (n+1))) = 
    countWhite (repaint (repaint initial (repaints (n-1))) (repaints n)))) :=
  sorry

#check constant_white_sectors_neighboring_minutes

end constant_white_sectors_neighboring_minutes_l2465_246547


namespace closest_point_l2465_246548

def u (s : ℝ) : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 3 + 6*s
  | 1 => -2 + 4*s
  | 2 => 4 + 2*s

def b : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 1
  | 1 => 7
  | 2 => 6

def direction_vector : Fin 3 → ℝ := λ i =>
  match i with
  | 0 => 6
  | 1 => 4
  | 2 => 2

theorem closest_point (s : ℝ) :
  (∀ t : ℝ, ‖u s - b‖ ≤ ‖u t - b‖) ↔ s = 1/2 := by
  sorry

end closest_point_l2465_246548


namespace two_phase_tournament_matches_l2465_246567

/-- Calculate the number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of matches in a two-phase tennis tournament -/
theorem two_phase_tournament_matches : 
  roundRobinMatches 10 + roundRobinMatches 5 = 55 := by
  sorry

end two_phase_tournament_matches_l2465_246567


namespace budget_allocation_l2465_246571

theorem budget_allocation (microphotonics : ℝ) (home_electronics : ℝ) (food_additives : ℝ) 
  (genetically_modified : ℝ) (basic_astrophysics_degrees : ℝ) 
  (h1 : microphotonics = 13)
  (h2 : home_electronics = 24)
  (h3 : food_additives = 15)
  (h4 : genetically_modified = 29)
  (h5 : basic_astrophysics_degrees = 39.6) :
  100 - (microphotonics + home_electronics + food_additives + genetically_modified + 
    (basic_astrophysics_degrees / 360 * 100)) = 8 := by
  sorry

end budget_allocation_l2465_246571


namespace eight_integer_lengths_l2465_246525

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- Counts the number of distinct integer lengths of line segments 
    from vertex E to points on the hypotenuse DF -/
def count_integer_lengths (t : RightTriangle) : ℕ :=
  sorry

/-- The main theorem stating that for the specific triangle, 
    there are exactly 8 distinct integer lengths -/
theorem eight_integer_lengths :
  ∃ (t : RightTriangle), t.de = 24 ∧ t.ef = 25 ∧ count_integer_lengths t = 8 :=
by sorry

end eight_integer_lengths_l2465_246525


namespace xiao_gang_steps_for_one_kilocalorie_l2465_246512

/-- The number of steps Xiao Gang walks for 1 kilocalorie of energy -/
def xiao_gang_steps : ℕ := 30

/-- The number of steps Xiao Qiong walks for 1 kilocalorie of energy -/
def xiao_qiong_steps : ℕ := xiao_gang_steps + 15

/-- The total steps Xiao Gang walks for a certain amount of energy -/
def xiao_gang_total_steps : ℕ := 9000

/-- The total steps Xiao Qiong walks for the same amount of energy as Xiao Gang -/
def xiao_qiong_total_steps : ℕ := 13500

theorem xiao_gang_steps_for_one_kilocalorie :
  xiao_gang_steps = 30 ∧
  xiao_qiong_steps = xiao_gang_steps + 15 ∧
  xiao_gang_total_steps * xiao_qiong_steps = xiao_qiong_total_steps * xiao_gang_steps :=
by sorry

end xiao_gang_steps_for_one_kilocalorie_l2465_246512


namespace absolute_value_inequality_l2465_246598

theorem absolute_value_inequality (x : ℝ) : 
  |2*x - 1| - x ≥ 2 ↔ x ≥ 3 ∨ x ≤ -1/3 := by
  sorry

end absolute_value_inequality_l2465_246598


namespace max_area_right_triangle_l2465_246589

/-- Given two positive real numbers a and b, the area of a triangle with sides a and b
    is maximized when these sides are perpendicular. -/
theorem max_area_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ θ : ℝ, 0 < θ ∧ θ < π →
    (1/2) * a * b * Real.sin θ ≤ (1/2) * a * b :=
by sorry

end max_area_right_triangle_l2465_246589


namespace arrange_five_and_three_books_l2465_246513

/-- The number of ways to arrange two types of indistinguishable objects in a row -/
def arrange_books (n m : ℕ) : ℕ := Nat.choose (n + m) n

/-- Theorem: Arranging 5 copies of one book and 3 copies of another book results in 56 ways -/
theorem arrange_five_and_three_books : arrange_books 5 3 = 56 := by
  sorry

end arrange_five_and_three_books_l2465_246513


namespace vector_sum_and_scale_l2465_246575

theorem vector_sum_and_scale :
  let v1 : Fin 2 → ℝ := ![5, -3]
  let v2 : Fin 2 → ℝ := ![-4, 9]
  v1 + 2 • v2 = ![-3, 15] := by
sorry

end vector_sum_and_scale_l2465_246575


namespace aunt_wang_money_proof_l2465_246541

/-- The price of apples per kilogram -/
def apple_price : ℝ := 5

/-- The amount of money Aunt Wang has -/
def aunt_wang_money : ℝ := 10.9

/-- Proves that Aunt Wang has 10.9 yuan given the problem conditions -/
theorem aunt_wang_money_proof :
  (2.5 * apple_price - aunt_wang_money = 1.6) ∧
  (aunt_wang_money - 2 * apple_price = 0.9) →
  aunt_wang_money = 10.9 :=
by
  sorry

#check aunt_wang_money_proof

end aunt_wang_money_proof_l2465_246541


namespace inequality_proof_l2465_246561

theorem inequality_proof (a b : ℝ) : a^2 + a*b + b^2 - 3*(a + b - 1) ≥ 0 := by
  sorry

end inequality_proof_l2465_246561


namespace system_of_equations_solution_l2465_246555

theorem system_of_equations_solution :
  ∀ x y : ℝ,
  (4 * x - 2) / (5 * x - 5) = 3 / 4 →
  x + y = 3 →
  x = -7 ∧ y = 10 := by
sorry

end system_of_equations_solution_l2465_246555


namespace sequence_sum_product_l2465_246588

theorem sequence_sum_product (n : ℕ+) : 
  let S : ℕ+ → ℚ := λ k => k / (k + 1)
  S n * S (n + 1) = 3/4 → n = 6 := by
sorry

end sequence_sum_product_l2465_246588


namespace geometric_sequence_formula_l2465_246580

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 2 →
  (∀ n : ℕ, a (n + 2)^2 + 4 * a n^2 = 4 * a (n + 1)^2) →
  ∀ n : ℕ, a n = 2^((n + 1) / 2 : ℝ) :=
by sorry

end geometric_sequence_formula_l2465_246580


namespace trapezoid_equal_area_segment_l2465_246566

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  base_difference : longer_base = shorter_base + 50
  midpoint_segment : ℝ
  midpoint_area_ratio : midpoint_segment = shorter_base + 25
  equal_area_segment : ℝ
  area_ratio_condition : 
    2 * (height / 3 * (shorter_base + midpoint_segment)) = 
    (2 * height / 3) * (midpoint_segment + longer_base)

/-- The main theorem to be proved -/
theorem trapezoid_equal_area_segment (t : Trapezoid) : 
  Int.floor (t.equal_area_segment ^ 2 / 50) = 78 := by
  sorry


end trapezoid_equal_area_segment_l2465_246566


namespace quadratic_equation_solution_l2465_246553

theorem quadratic_equation_solution (b : ℚ) : 
  ((-4 : ℚ)^2 + b * (-4 : ℚ) - 45 = 0) → b = -29/4 := by
  sorry

end quadratic_equation_solution_l2465_246553


namespace quadratic_equation_1_l2465_246595

theorem quadratic_equation_1 (x : ℝ) : 5 * x^2 = 40 * x → x = 0 ∨ x = 8 := by
  sorry

#check quadratic_equation_1

end quadratic_equation_1_l2465_246595


namespace sum_of_roots_equals_five_l2465_246538

theorem sum_of_roots_equals_five : 
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, x ≠ 3 ∧ (x^3 - 3*x^2 - 12*x) / (x - 3) = 6) ∧ 
    (∀ x ∉ S, x = 3 ∨ (x^3 - 3*x^2 - 12*x) / (x - 3) ≠ 6) ∧
    (Finset.sum S id = 5) :=
by sorry

end sum_of_roots_equals_five_l2465_246538


namespace function_inequality_implies_k_value_l2465_246549

/-- The function f(x) = x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- The theorem stating that if f(x) ≤ kx for all x in (1,5], then k = 36/5 -/
theorem function_inequality_implies_k_value (k : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x ≤ 5 → f x ≤ k * x) → k = 36/5 := by
  sorry

end function_inequality_implies_k_value_l2465_246549


namespace phone_plan_charge_equality_l2465_246502

/-- Represents the per-minute charge for plan B -/
def plan_b_charge : ℝ := 0.20

/-- Represents the fixed charge for the first 9 minutes in plan A -/
def plan_a_fixed_charge : ℝ := 0.60

/-- Represents the duration (in minutes) where both plans charge the same amount -/
def equal_charge_duration : ℝ := 3

theorem phone_plan_charge_equality :
  plan_a_fixed_charge = plan_b_charge * equal_charge_duration := by
  sorry

#check phone_plan_charge_equality

end phone_plan_charge_equality_l2465_246502


namespace quadratic_decreasing_range_l2465_246543

/-- Given a quadratic function y = (x-m)^2 - 1, if y decreases as x increases when x ≤ 3, then m ≥ 3 -/
theorem quadratic_decreasing_range (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≤ 3 ∧ x₂ ≤ 3 ∧ x₁ < x₂ → (x₁ - m)^2 - 1 > (x₂ - m)^2 - 1) →
  m ≥ 3 := by
sorry

end quadratic_decreasing_range_l2465_246543


namespace total_cards_l2465_246508

theorem total_cards (initial_cards added_cards : ℕ) 
  (h1 : initial_cards = 4)
  (h2 : added_cards = 3) :
  initial_cards + added_cards = 7 := by
sorry

end total_cards_l2465_246508


namespace negation_of_proposition_l2465_246511

theorem negation_of_proposition (a b c : ℝ) :
  (¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3)) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) :=
by sorry

end negation_of_proposition_l2465_246511


namespace fraction_sum_equivalence_l2465_246573

theorem fraction_sum_equivalence (a b c : ℝ) 
  (h : a / (35 - a) + b / (75 - b) + c / (85 - c) = 5) :
  7 / (35 - a) + 15 / (75 - b) + 17 / (85 - c) = 8 / 5 := by
  sorry

end fraction_sum_equivalence_l2465_246573


namespace investment_growth_l2465_246529

/-- Calculates the future value of an investment -/
def future_value (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_growth (principal : ℝ) (rate : ℝ) (time : ℕ) (future_amount : ℝ) 
  (h1 : principal = 376889.02)
  (h2 : rate = 0.06)
  (h3 : time = 8)
  (h4 : future_amount = 600000) :
  future_value principal rate time = future_amount := by
  sorry

end investment_growth_l2465_246529


namespace pascal_triangle_46th_row_45th_number_l2465_246501

theorem pascal_triangle_46th_row_45th_number : 
  let n : ℕ := 46  -- The row number (0-indexed)
  let k : ℕ := 44  -- The position in the row (0-indexed)
  Nat.choose n k = 1035 := by
sorry

end pascal_triangle_46th_row_45th_number_l2465_246501


namespace vector_difference_l2465_246597

/-- Given two vectors AB and AC in 2D space, prove that BC is their difference -/
theorem vector_difference (AB AC : ℝ × ℝ) (h1 : AB = (2, -1)) (h2 : AC = (-4, 1)) :
  AC - AB = (-6, 2) := by
  sorry

end vector_difference_l2465_246597


namespace hill_climbing_speed_l2465_246515

/-- Proves that given a round trip journey with specified conditions, 
    the average speed while going up is 2.25 km/h -/
theorem hill_climbing_speed 
  (time_up : ℝ) 
  (time_down : ℝ) 
  (avg_speed_total : ℝ) 
  (h1 : time_up = 4) 
  (h2 : time_down = 2) 
  (h3 : avg_speed_total = 3) : 
  (avg_speed_total * (time_up + time_down) / 2) / time_up = 2.25 := by
  sorry

end hill_climbing_speed_l2465_246515


namespace sharp_composition_l2465_246551

def sharp (N : ℕ) : ℕ := 3 * N + 2

theorem sharp_composition : sharp (sharp (sharp 6)) = 188 := by
  sorry

end sharp_composition_l2465_246551


namespace no_perfect_cube_pair_l2465_246534

theorem no_perfect_cube_pair : ¬ ∃ (a b : ℤ), 
  (∃ (x : ℤ), a^5 * b + 3 = x^3) ∧ (∃ (y : ℤ), a * b^5 + 3 = y^3) := by
  sorry

end no_perfect_cube_pair_l2465_246534


namespace cubic_polynomial_integer_root_l2465_246596

theorem cubic_polynomial_integer_root
  (p q : ℚ)
  (h1 : ∃ x : ℝ, x^3 + p*x + q = 0 ∧ x = 2 - Real.sqrt 5)
  (h2 : ∃ n : ℤ, n^3 + p*n + q = 0) :
  ∃ n : ℤ, n^3 + p*n + q = 0 ∧ n = -4 := by
sorry

end cubic_polynomial_integer_root_l2465_246596


namespace uncle_jerry_tomatoes_l2465_246527

/-- The number of tomatoes Uncle Jerry reaped yesterday -/
def yesterday_tomatoes : ℕ := 120

/-- The additional number of tomatoes Uncle Jerry reaped today compared to yesterday -/
def additional_today : ℕ := 50

/-- The total number of tomatoes Uncle Jerry reaped over two days -/
def total_tomatoes : ℕ := yesterday_tomatoes + (yesterday_tomatoes + additional_today)

theorem uncle_jerry_tomatoes :
  total_tomatoes = 290 :=
sorry

end uncle_jerry_tomatoes_l2465_246527


namespace square_area_from_hexagon_wire_l2465_246599

/-- The area of a square formed from a wire that previously made a regular hexagon --/
theorem square_area_from_hexagon_wire (hexagon_side : ℝ) : 
  hexagon_side = 4 → (6 * hexagon_side / 4) ^ 2 = 36 := by
  sorry

end square_area_from_hexagon_wire_l2465_246599


namespace angle_sum_around_point_l2465_246559

theorem angle_sum_around_point (x : ℝ) : 
  (6 * x + 3 * x + x + 5 * x = 360) → x = 24 := by
  sorry

end angle_sum_around_point_l2465_246559


namespace eight_by_eight_unfolds_to_nine_l2465_246507

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a folded and cut grid -/
structure FoldedCutGrid :=
  (original : Grid)
  (folded_size : ℕ)
  (cut : Bool)

/-- Counts the number of parts after unfolding a cut grid -/
def count_parts (g : FoldedCutGrid) : ℕ :=
  sorry

/-- Theorem stating that an 8x8 grid folded to 1x1 and cut unfolds into 9 parts -/
theorem eight_by_eight_unfolds_to_nine :
  ∀ (g : FoldedCutGrid),
    g.original.size = 8 →
    g.folded_size = 1 →
    g.cut = true →
    count_parts g = 9 :=
  sorry

end eight_by_eight_unfolds_to_nine_l2465_246507


namespace two_numbers_problem_l2465_246506

theorem two_numbers_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 ∧ y = 17 := by
  sorry

end two_numbers_problem_l2465_246506


namespace weight_replacement_l2465_246519

theorem weight_replacement (n : ℕ) (average_increase : ℝ) (new_weight : ℝ) :
  n = 8 →
  average_increase = 1.5 →
  new_weight = 77 →
  ∃ old_weight : ℝ,
    old_weight = new_weight - n * average_increase ∧
    old_weight = 65 :=
by sorry

end weight_replacement_l2465_246519


namespace total_cost_is_60_l2465_246578

/-- The cost of a set of school supplies -/
structure SchoolSupplies where
  notebook : ℕ
  pen : ℕ
  ruler : ℕ
  pencil : ℕ

/-- The conditions given in the problem -/
structure Conditions where
  supplies : SchoolSupplies
  notebook_pencil_ruler_cost : supplies.notebook + supplies.pencil + supplies.ruler = 47
  notebook_ruler_pen_cost : supplies.notebook + supplies.ruler + supplies.pen = 58
  pen_pencil_cost : supplies.pen + supplies.pencil = 15

/-- The theorem to be proved -/
theorem total_cost_is_60 (c : Conditions) : 
  c.supplies.notebook + c.supplies.pen + c.supplies.ruler + c.supplies.pencil = 60 := by
  sorry

#check total_cost_is_60

end total_cost_is_60_l2465_246578


namespace arc_length_of_curve_l2465_246554

noncomputable def f (x : ℝ) : ℝ := -Real.arccos x + Real.sqrt (1 - x^2) + 1

theorem arc_length_of_curve (a b : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ 9/16) :
  ∫ x in a..b, Real.sqrt (1 + (((1 - x) / Real.sqrt (1 - x^2))^2)) = 1 / Real.sqrt 2 :=
sorry

end arc_length_of_curve_l2465_246554


namespace total_population_is_56000_l2465_246540

/-- The total population of Boise, Seattle, and Lake View --/
def total_population (boise seattle lakeview : ℕ) : ℕ :=
  boise + seattle + lakeview

/-- Theorem: Given the conditions, the total population of the three cities is 56000 --/
theorem total_population_is_56000 :
  ∀ (boise seattle lakeview : ℕ),
    boise = (3 * seattle) / 5 →
    lakeview = seattle + 4000 →
    lakeview = 24000 →
    total_population boise seattle lakeview = 56000 := by
  sorry

end total_population_is_56000_l2465_246540


namespace new_basis_from_original_l2465_246522

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem new_basis_from_original
  (a b c : V)
  (h : LinearIndependent ℝ ![a, b, c])
  (hspan : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![a + b, a - c, b] ∧
  Submodule.span ℝ {a + b, a - c, b} = ⊤ :=
sorry

end new_basis_from_original_l2465_246522


namespace sum_of_quadratic_roots_sum_of_solutions_is_neg_nine_l2465_246528

theorem sum_of_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_solutions_is_neg_nine :
  let a : ℝ := -3
  let b : ℝ := -27
  let c : ℝ := 54
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -9 :=
by sorry

end sum_of_quadratic_roots_sum_of_solutions_is_neg_nine_l2465_246528


namespace substitution_elimination_l2465_246565

/-- Given a system of linear equations in two variables x and y,
    prove that the equation obtained by eliminating y using substitution
    is equivalent to the given result. -/
theorem substitution_elimination (x y : ℝ) :
  (y = x + 3 ∧ 2 * x - y = 5) → (2 * x - x - 3 = 5) := by
  sorry

end substitution_elimination_l2465_246565


namespace parabola_vertex_correct_l2465_246586

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y = (x + 2)^2 + 1

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (-2, 1)

-- Theorem stating that the given equation represents a parabola with the specified vertex
theorem parabola_vertex_correct :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex ↔ x = -2 ∧ y = 1 :=
sorry

end parabola_vertex_correct_l2465_246586


namespace existence_of_n_for_prime_p_l2465_246562

theorem existence_of_n_for_prime_p (p : ℕ) (hp : Prime p) : ∃ n : ℕ, p ∣ (2022^n - n) := by
  sorry

end existence_of_n_for_prime_p_l2465_246562


namespace reconstruction_possible_iff_odd_l2465_246536

/-- A regular n-gon with numbers assigned to its vertices and center -/
structure NumberedPolygon (n : ℕ) where
  vertex_numbers : Fin n → ℕ
  center_number : ℕ

/-- The set of triples formed by connecting the center to all vertices -/
def triples (p : NumberedPolygon n) : Finset (Finset ℕ) :=
  sorry

/-- A function that attempts to reconstruct the original numbers -/
def reconstruct (t : Finset (Finset ℕ)) : Option (NumberedPolygon n) :=
  sorry

theorem reconstruction_possible_iff_odd (n : ℕ) :
  (∀ p : NumberedPolygon n, ∃! q : NumberedPolygon n, triples p = triples q) ↔ Odd n :=
sorry

end reconstruction_possible_iff_odd_l2465_246536


namespace quiz_ranking_l2465_246568

theorem quiz_ranking (F E H G : ℝ) 
  (nonneg : F ≥ 0 ∧ E ≥ 0 ∧ H ≥ 0 ∧ G ≥ 0)
  (sum_equal : E + G = F + H)
  (sum_equal_swap : F + E = H + G)
  (george_higher : G > E + F) :
  G > E ∧ G > H ∧ E = H ∧ E > F ∧ H > F := by
  sorry

end quiz_ranking_l2465_246568


namespace last_three_average_l2465_246570

theorem last_three_average (list : List ℝ) (h1 : list.length = 7) 
  (h2 : list.sum / 7 = 62) (h3 : (list.take 4).sum / 4 = 58) : 
  (list.drop 4).sum / 3 = 202 / 3 := by
  sorry

end last_three_average_l2465_246570


namespace quadratic_coefficient_relation_l2465_246531

/-- Given two quadratic equations and their root relationships, prove the relation between their coefficients -/
theorem quadratic_coefficient_relation (a b c d : ℝ) (α β : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 ↔ x = α ∨ x = β) →
  (∀ x, x^2 + c*x + d = 0 ↔ x = α^2 + 1 ∨ x = β^2 + 1) →
  c = -a^2 + 2*b - 2 :=
by sorry

end quadratic_coefficient_relation_l2465_246531


namespace marked_angle_is_fifteen_degrees_l2465_246545

-- Define the figure with three squares
structure ThreeSquaresFigure where
  -- Angles are represented in degrees
  angle_C1OC2 : ℝ
  angle_A2OA3 : ℝ
  angle_C3OA1 : ℝ

-- Define the properties of the figure
def is_valid_three_squares_figure (f : ThreeSquaresFigure) : Prop :=
  f.angle_C1OC2 = 30 ∧
  f.angle_A2OA3 = 45 ∧
  -- Additional properties of squares (all right angles)
  -- Assuming O is the center where all squares meet
  ∃ (angle_C1OA1 angle_C2OA2 angle_C3OA3 : ℝ),
    angle_C1OA1 = 90 ∧ angle_C2OA2 = 90 ∧ angle_C3OA3 = 90

-- Theorem statement
theorem marked_angle_is_fifteen_degrees (f : ThreeSquaresFigure) 
  (h : is_valid_three_squares_figure f) : f.angle_C3OA1 = 15 := by
  sorry

end marked_angle_is_fifteen_degrees_l2465_246545


namespace eighteen_wheel_truck_toll_l2465_246550

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculate_axles (total_wheels : ℕ) (front_axle_wheels : ℕ) (other_axle_wheels : ℕ) : ℕ :=
  1 + (total_wheels - front_axle_wheels) / other_axle_wheels

/-- Calculates the toll for a truck given the number of axles -/
def calculate_toll (axles : ℕ) : ℚ :=
  0.50 + 0.50 * (axles - 2)

theorem eighteen_wheel_truck_toll :
  let total_wheels : ℕ := 18
  let front_axle_wheels : ℕ := 2
  let other_axle_wheels : ℕ := 4
  let axles := calculate_axles total_wheels front_axle_wheels other_axle_wheels
  calculate_toll axles = 2 := by sorry

end eighteen_wheel_truck_toll_l2465_246550


namespace angle_bisector_locus_l2465_246577

/-- An angle with vertex A and sides AB and AC -/
structure Angle (A B C : ℝ × ℝ) : Prop where
  nondegenerate : A ≠ B ∧ A ≠ C

/-- A ray from point P through point Q -/
def Ray (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X | ∃ t : ℝ, t ≥ 0 ∧ X = P + t • (Q - P)}

/-- A line segment between two points -/
def Segment (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X | ∃ t : ℝ, 0 < t ∧ t < 1 ∧ X = P + t • (Q - P)}

/-- Perpendicular from a point to a line -/
def Perpendicular (P Q R : ℝ × ℝ) (X : ℝ × ℝ) : Prop :=
  (X - P) • (Q - R) = 0 ∧ ∃ t : ℝ, X = P + t • (Q - R)

/-- The locus of points for the angle bisector problem -/
def AngleBisectorLocus (A O B C K L : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {M | (M ∈ Ray A O \ {A, O}) ∨ (M ∈ Segment K L \ {K, L})}

theorem angle_bisector_locus
  (A O B C K L : ℝ × ℝ)
  (h_angle : Angle A B C)
  (h_bisector : O ∈ Ray A O \ {A})
  (h_perp_K : Perpendicular O K B A)
  (h_perp_L : Perpendicular O L C A)
  (M : ℝ × ℝ) :
  M ∈ AngleBisectorLocus A O B C K L ↔
    ((M ∈ Ray A O ∧ M ≠ A ∧ M ≠ O) ∨ (M ∈ Segment K L ∧ M ≠ K ∧ M ≠ L)) :=
by sorry

end angle_bisector_locus_l2465_246577


namespace inventory_and_profit_calculation_l2465_246539

/-- Represents the inventory and financial data of a supermarket --/
structure Supermarket where
  total_cost : ℕ
  cost_A : ℕ
  cost_B : ℕ
  price_A : ℕ
  price_B : ℕ

/-- Calculates the number of items A and B, and the total profit --/
def calculate_inventory_and_profit (s : Supermarket) : ℕ × ℕ × ℕ :=
  let items_A := 150
  let items_B := 90
  let profit := items_A * (s.price_A - s.cost_A) + items_B * (s.price_B - s.cost_B)
  (items_A, items_B, profit)

/-- Theorem stating the correct inventory and profit calculation --/
theorem inventory_and_profit_calculation (s : Supermarket) 
  (h1 : s.total_cost = 6000)
  (h2 : s.cost_A = 22)
  (h3 : s.cost_B = 30)
  (h4 : s.price_A = 29)
  (h5 : s.price_B = 40) :
  calculate_inventory_and_profit s = (150, 90, 1950) :=
by
  sorry

#eval calculate_inventory_and_profit ⟨6000, 22, 30, 29, 40⟩

end inventory_and_profit_calculation_l2465_246539


namespace question_mark_value_l2465_246563

theorem question_mark_value (question_mark : ℝ) : 
  question_mark * 240 = 173 * 240 → question_mark = 173 := by
  sorry

end question_mark_value_l2465_246563


namespace doll_ratio_is_two_to_one_l2465_246585

/-- Given the number of Dina's dolls -/
def dinas_dolls : ℕ := 60

/-- Given the fraction of Ivy's dolls that are collectors editions -/
def ivy_collectors_fraction : ℚ := 2/3

/-- Given the number of Ivy's collectors edition dolls -/
def ivy_collectors : ℕ := 20

/-- Calculate the total number of Ivy's dolls -/
def ivys_dolls : ℕ := ivy_collectors * 3 / 2

/-- The ratio of Dina's dolls to Ivy's dolls -/
def doll_ratio : ℚ := dinas_dolls / ivys_dolls

theorem doll_ratio_is_two_to_one : doll_ratio = 2 := by sorry

end doll_ratio_is_two_to_one_l2465_246585
