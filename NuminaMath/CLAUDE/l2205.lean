import Mathlib

namespace median_inequality_l2205_220562

/-- Given a triangle ABC with sides a, b, c and medians s_a, s_b, s_c,
    if a < (b+c)/2, then s_a > (s_b + s_c)/2 -/
theorem median_inequality (a b c s_a s_b s_c : ℝ) 
    (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
    (h_medians : s_a = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2) ∧
                 s_b = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2) ∧
                 s_c = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2))
    (h_cond : a < (b + c) / 2) :
  s_a > (s_b + s_c) / 2 := by
  sorry

end median_inequality_l2205_220562


namespace jasons_leg_tattoos_l2205_220576

theorem jasons_leg_tattoos (jason_arm_tattoos : ℕ) (adam_tattoos : ℕ) :
  jason_arm_tattoos = 2 →
  adam_tattoos = 23 →
  ∃ (jason_leg_tattoos : ℕ),
    adam_tattoos = 2 * (2 * jason_arm_tattoos + 2 * jason_leg_tattoos) + 3 ∧
    jason_leg_tattoos = 3 :=
by sorry

end jasons_leg_tattoos_l2205_220576


namespace problem_figure_area_l2205_220544

/-- A figure composed of square segments -/
structure SegmentedFigure where
  /-- The number of segments along one side of the square -/
  segments_per_side : ℕ
  /-- The length of each segment in cm -/
  segment_length : ℝ

/-- The area of a SegmentedFigure in cm² -/
def area (figure : SegmentedFigure) : ℝ :=
  (figure.segments_per_side * figure.segment_length) ^ 2

/-- The specific figure from the problem -/
def problem_figure : SegmentedFigure :=
  { segments_per_side := 3
  , segment_length := 3 }

theorem problem_figure_area :
  area problem_figure = 81 := by sorry

end problem_figure_area_l2205_220544


namespace garbage_collection_l2205_220598

theorem garbage_collection (D : ℝ) : 
  (∃ (Dewei Zane : ℝ), 
    Dewei = D - 2 ∧ 
    Zane = 4 * Dewei ∧ 
    Zane = 62) → 
  D = 17.5 := by
sorry

end garbage_collection_l2205_220598


namespace sum_of_largest_and_smallest_abc_l2205_220586

def is_valid_abc (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100 = 2) ∧ (n % 10 = 7)

def largest_abc : ℕ := 297
def smallest_abc : ℕ := 207

theorem sum_of_largest_and_smallest_abc :
  is_valid_abc largest_abc ∧
  is_valid_abc smallest_abc ∧
  (∀ n : ℕ, is_valid_abc n → smallest_abc ≤ n ∧ n ≤ largest_abc) ∧
  largest_abc + smallest_abc = 504 :=
sorry

end sum_of_largest_and_smallest_abc_l2205_220586


namespace jack_recycling_earnings_l2205_220569

/-- The amount Jack gets per bottle in dollars -/
def bottle_amount : ℚ := sorry

/-- The amount Jack gets per can in dollars -/
def can_amount : ℚ := 5 / 100

/-- The number of bottles Jack recycled -/
def num_bottles : ℕ := 80

/-- The number of cans Jack recycled -/
def num_cans : ℕ := 140

/-- The total amount Jack made in dollars -/
def total_amount : ℚ := 15

theorem jack_recycling_earnings :
  bottle_amount * num_bottles + can_amount * num_cans = total_amount ∧
  bottle_amount = 1 / 10 := by sorry

end jack_recycling_earnings_l2205_220569


namespace false_proposition_l2205_220504

-- Define proposition p
def p : Prop := ∃ x : ℝ, (Real.cos x)^2 - (Real.sin x)^2 = 7

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp x > 0

-- Theorem statement
theorem false_proposition : ¬(¬p ∧ ¬q) := by
  sorry

end false_proposition_l2205_220504


namespace farmer_land_usage_l2205_220521

theorem farmer_land_usage (beans wheat corn total : ℕ) : 
  beans + wheat + corn = total →
  5 * wheat = 2 * beans →
  2 * corn = beans →
  corn = 376 →
  total = 1034 := by
sorry

end farmer_land_usage_l2205_220521


namespace discounted_price_calculation_l2205_220501

theorem discounted_price_calculation (original_price discount_percentage : ℝ) 
  (h1 : original_price = 975)
  (h2 : discount_percentage = 20) : 
  original_price * (1 - discount_percentage / 100) = 780 := by
  sorry

end discounted_price_calculation_l2205_220501


namespace largest_prime_with_special_form_l2205_220509

theorem largest_prime_with_special_form :
  ∀ p : ℕ, Prime p →
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p = (b / 2 : ℚ) * Real.sqrt ((a - b : ℚ) / (a + b))) →
    p ≤ 5 :=
by sorry

end largest_prime_with_special_form_l2205_220509


namespace egyptian_fraction_odd_divisor_l2205_220537

theorem egyptian_fraction_odd_divisor (n : ℕ) (h_n : n > 1) (h_odd : Odd n) :
  (∃ x y : ℕ, (4 : ℚ) / n = 1 / x + 1 / y) ↔
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∃ k : ℕ, p = 4 * k - 1) :=
by sorry

end egyptian_fraction_odd_divisor_l2205_220537


namespace zhang_qiujian_problem_l2205_220577

theorem zhang_qiujian_problem (x y : ℤ) : 
  (x + 10 - (y - 10) = 5 * (y - 10) ∧ x - 10 = y + 10) ↔
  (x = y + 10 ∧ 
   x + 10 - (y - 10) = 5 * (y - 10) ∧ 
   x - 10 = y + 10) :=
by sorry

end zhang_qiujian_problem_l2205_220577


namespace power_sum_equality_l2205_220563

theorem power_sum_equality : (-1)^45 + 2^(3^2 + 5^2 - 4^2) = 262143 := by
  sorry

end power_sum_equality_l2205_220563


namespace quadratic_function_k_l2205_220584

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_function_k (a b c : ℤ) : 
  (f a b c 1 = 0) →
  (60 < f a b c 6 ∧ f a b c 6 < 70) →
  (120 < f a b c 9 ∧ f a b c 9 < 130) →
  (∃ k : ℤ, 10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1)) →
  (∃ k : ℤ, 10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1) ∧ k = 4) :=
by sorry

end quadratic_function_k_l2205_220584


namespace quadratic_theorem_l2205_220572

/-- Quadratic function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The maximum value of f occurs at x = 2 -/
def has_max_at_2 (a b c : ℝ) : Prop :=
  ∀ x, f a b c x ≤ f a b c 2

/-- The maximum value of f is 7 -/
def max_value_is_7 (a b c : ℝ) : Prop :=
  f a b c 2 = 7

/-- f passes through the point (0, -7) -/
def passes_through_0_neg7 (a b c : ℝ) : Prop :=
  f a b c 0 = -7

theorem quadratic_theorem (a b c : ℝ) 
  (h1 : has_max_at_2 a b c)
  (h2 : max_value_is_7 a b c)
  (h3 : passes_through_0_neg7 a b c) :
  f a b c 5 = -24.5 := by sorry

end quadratic_theorem_l2205_220572


namespace paperboy_delivery_ways_l2205_220541

/-- Represents the number of valid delivery sequences for n houses -/
def E : ℕ → ℕ
  | 0 => 0  -- No houses, no deliveries
  | 1 => 2  -- For one house, two options: deliver or not
  | 2 => 4  -- For two houses, all combinations are valid
  | 3 => 8  -- E_3 = E_2 + E_1 + 2
  | n + 4 => E (n + 3) + E (n + 2) + E (n + 1)

/-- The problem statement -/
theorem paperboy_delivery_ways : E 12 = 1854 := by
  sorry

end paperboy_delivery_ways_l2205_220541


namespace tuesday_to_monday_ratio_l2205_220560

def monday_fabric : ℕ := 20
def fabric_cost : ℕ := 2
def wednesday_ratio : ℚ := 1/4
def total_earnings : ℕ := 140

theorem tuesday_to_monday_ratio :
  ∃ (tuesday_fabric : ℕ),
    (monday_fabric * fabric_cost + 
     tuesday_fabric * fabric_cost + 
     (wednesday_ratio * tuesday_fabric) * fabric_cost = total_earnings) ∧
    (tuesday_fabric = monday_fabric) := by
  sorry

end tuesday_to_monday_ratio_l2205_220560


namespace intersection_distance_l2205_220564

theorem intersection_distance : ∃ (p1 p2 : ℝ × ℝ),
  (p1.1^2 + p1.2 = 12 ∧ p1.1 + p1.2 = 8) ∧
  (p2.1^2 + p2.2 = 12 ∧ p2.1 + p2.2 = 8) ∧
  p1 ≠ p2 ∧
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 34 := by
  sorry

end intersection_distance_l2205_220564


namespace sum_and_equal_numbers_l2205_220542

theorem sum_and_equal_numbers (a b c : ℚ) 
  (sum_eq : a + b + c = 108)
  (equal_after_changes : a + 8 = b - 4 ∧ b - 4 = 6 * c) :
  b = 724 / 13 := by
sorry

end sum_and_equal_numbers_l2205_220542


namespace min_c_value_l2205_220526

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! (x y : ℝ), 2*x + y = 2019 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1010 :=
sorry

end min_c_value_l2205_220526


namespace master_bedroom_size_l2205_220591

theorem master_bedroom_size (total_area guest_area master_area combined_area : ℝ) 
  (h1 : total_area = 2300)
  (h2 : combined_area = 1000)
  (h3 : guest_area = (1/4) * master_area)
  (h4 : total_area = combined_area + guest_area + master_area) :
  master_area = 1040 := by
sorry

end master_bedroom_size_l2205_220591


namespace ski_lift_time_l2205_220516

theorem ski_lift_time (ski_down_time : ℝ) (num_trips : ℕ) (total_time : ℝ) 
  (h1 : ski_down_time = 5)
  (h2 : num_trips = 6)
  (h3 : total_time = 120) : 
  (total_time - num_trips * ski_down_time) / num_trips = 15 := by
sorry

end ski_lift_time_l2205_220516


namespace log_equation_equals_zero_l2205_220547

-- Define the logarithm function with base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_equation_equals_zero : 2 * log5 10 + log5 0.25 = 0 := by sorry

end log_equation_equals_zero_l2205_220547


namespace mitzi_amusement_park_money_l2205_220500

def ticket_cost : ℕ := 30
def food_cost : ℕ := 13
def tshirt_cost : ℕ := 23
def remaining_money : ℕ := 9

theorem mitzi_amusement_park_money :
  ticket_cost + food_cost + tshirt_cost + remaining_money = 75 := by
  sorry

end mitzi_amusement_park_money_l2205_220500


namespace train_carriage_seats_l2205_220556

theorem train_carriage_seats : 
  ∀ (seats_per_carriage : ℕ),
  (3 * 4 * (seats_per_carriage + 10) = 420) →
  seats_per_carriage = 25 := by
sorry

end train_carriage_seats_l2205_220556


namespace jesses_room_length_l2205_220582

theorem jesses_room_length (area : ℝ) (width : ℝ) (h1 : area = 12.0) (h2 : width = 8) :
  area / width = 1.5 := by
  sorry

end jesses_room_length_l2205_220582


namespace tims_bodyguard_cost_l2205_220589

/-- Calculate the total weekly cost for bodyguards --/
def total_weekly_cost (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

/-- Prove that the total weekly cost for Tim's bodyguards is $2240 --/
theorem tims_bodyguard_cost :
  total_weekly_cost 2 20 8 7 = 2240 := by
  sorry

end tims_bodyguard_cost_l2205_220589


namespace jason_remaining_cards_l2205_220525

def initial_cards : ℕ := 3
def cards_bought : ℕ := 2

theorem jason_remaining_cards : initial_cards - cards_bought = 1 := by
  sorry

end jason_remaining_cards_l2205_220525


namespace feed_has_greatest_value_l2205_220566

/-- The value of a letter in the alphabet (A to F) -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | _ => 0

/-- The value of a word, which is the sum of its letter values -/
def word_value (w : String) : ℕ :=
  w.data.map letter_value |>.sum

/-- The list of words to compare -/
def words : List String := ["BEEF", "FADE", "FEED", "FACE", "DEAF"]

theorem feed_has_greatest_value :
  ∀ w ∈ words, word_value "FEED" ≥ word_value w :=
by sorry

end feed_has_greatest_value_l2205_220566


namespace inscribed_cube_volume_in_specific_pyramid_l2205_220517

/-- A pyramid with a regular hexagonal base and isosceles triangular lateral faces -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_height : ℝ

/-- A cube inscribed in a hexagonal pyramid -/
structure InscribedCube where
  pyramid : HexagonalPyramid
  -- Each vertex of the cube is either on the base or touches a point on the lateral faces

/-- The volume of an inscribed cube in a hexagonal pyramid -/
def inscribed_cube_volume (cube : InscribedCube) : ℝ :=
  sorry

theorem inscribed_cube_volume_in_specific_pyramid :
  ∀ (cube : InscribedCube),
    cube.pyramid.base_side_length = 2 →
    cube.pyramid.lateral_face_height = 3 →
    inscribed_cube_volume cube = 2 * Real.sqrt 2 :=
by sorry

end inscribed_cube_volume_in_specific_pyramid_l2205_220517


namespace power_eight_sum_ratio_l2205_220583

theorem power_eight_sum_ratio (x y k : ℝ) 
  (h : (x^2 + y^2)/(x^2 - y^2) + (x^2 - y^2)/(x^2 + y^2) = k) :
  (x^8 + y^8)/(x^8 - y^8) + (x^8 - y^8)/(x^8 + y^8) = (k^4 + 24*k^2 + 16)/(4*k^3 + 16*k) :=
by sorry

end power_eight_sum_ratio_l2205_220583


namespace sin_cos_identity_l2205_220597

theorem sin_cos_identity (α : ℝ) : (Real.sin α - Real.cos α)^2 + Real.sin (2 * α) = 1 := by
  sorry

end sin_cos_identity_l2205_220597


namespace max_profit_and_break_even_l2205_220539

/-- Revenue function (in ten thousand yuan) -/
def R (x : ℝ) : ℝ := 5 * x - x^2

/-- Cost function (in ten thousand yuan) -/
def C (x : ℝ) : ℝ := 0.5 + 0.25 * x

/-- Profit function (in ten thousand yuan) -/
def profit (x : ℝ) : ℝ := R x - C x

/-- Annual demand in hundreds of units -/
def annual_demand : ℝ := 5

theorem max_profit_and_break_even :
  ∃ (max_profit_units : ℝ) (break_even_lower break_even_upper : ℝ),
    (∀ x, 0 ≤ x → x ≤ annual_demand → profit x ≤ profit max_profit_units) ∧
    (max_profit_units = 4.75) ∧
    (break_even_lower = 0.1) ∧
    (break_even_upper = 48) ∧
    (∀ x, break_even_lower ≤ x → x ≤ break_even_upper → profit x ≥ 0) :=
  sorry

end max_profit_and_break_even_l2205_220539


namespace investment_interest_proof_l2205_220538

/-- Calculates the interest earned on an investment with annual compounding -/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the interest earned on a $500 investment at 2% annual rate for 3 years is approximately $30.60 -/
theorem investment_interest_proof :
  let principal := 500
  let rate := 0.02
  let years := 3
  abs (interest_earned principal rate years - 30.60) < 0.01 := by
  sorry

end investment_interest_proof_l2205_220538


namespace sine_function_amplitude_l2205_220554

theorem sine_function_amplitude (a b : ℝ) (ha : a < 0) (hb : b > 0) :
  (∀ x, |a * Real.sin (b * x)| ≤ 3) ∧ (∃ x, |a * Real.sin (b * x)| = 3) → a = -3 := by
  sorry

end sine_function_amplitude_l2205_220554


namespace work_distribution_l2205_220532

theorem work_distribution (total_work : ℝ) (h1 : total_work > 0) : 
  let top_20_percent_work := 0.8 * total_work
  let remaining_work := total_work - top_20_percent_work
  let next_20_percent_work := 0.25 * remaining_work
  ∃ (work_40_percent : ℝ), work_40_percent ≥ top_20_percent_work + next_20_percent_work ∧ 
                            work_40_percent / total_work ≥ 0.85 := by
  sorry

end work_distribution_l2205_220532


namespace max_a_value_l2205_220534

-- Define the quadratic polynomial f(x) = x^2 + ax + b
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem max_a_value (a b : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a b y = f a b x + y) →
  a ≤ (1/2 : ℝ) :=
sorry

end max_a_value_l2205_220534


namespace correct_average_l2205_220594

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 19 ∧ incorrect_num = 26 ∧ correct_num = 76 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 24 :=
by sorry

end correct_average_l2205_220594


namespace packetB_height_day10_l2205_220543

/-- Represents the growth rate of sunflowers --/
structure GrowthRate where
  x : ℝ  -- number of days since planting
  y : ℝ  -- daily average sunlight exposure (hours)
  W : ℝ  -- combined effect of competition and weather (0-10 scale)

/-- Calculates the growth rate for Packet A sunflowers --/
def growthRateA (r : GrowthRate) : ℝ := 2 * r.x + r.y - 0.1 * r.W

/-- Calculates the growth rate for Packet B sunflowers --/
def growthRateB (r : GrowthRate) : ℝ := 3 * r.x - r.y + 0.2 * r.W

/-- Theorem stating the height of Packet B sunflowers on day 10 --/
theorem packetB_height_day10 (r : GrowthRate) 
  (h1 : r.x = 10)
  (h2 : r.y = 6)
  (h3 : r.W = 5)
  (h4 : ∃ (hA hB : ℝ), hA = 192 ∧ hA = 1.2 * hB) :
  ∃ (hB : ℝ), hB = 160 := by
  sorry


end packetB_height_day10_l2205_220543


namespace hyperbola_eccentricity_l2205_220578

/-- Given a hyperbola and a line intersecting it, proves that the eccentricity is √2 under specific conditions -/
theorem hyperbola_eccentricity (a b k m : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (A B N : ℝ × ℝ), 
    -- The line y = kx + m intersects the hyperbola at A and B
    (A.1^2 / a^2 - A.2^2 / b^2 = 1 ∧ A.2 = k * A.1 + m) ∧
    (B.1^2 / a^2 - B.2^2 / b^2 = 1 ∧ B.2 = k * B.1 + m) ∧
    -- A and B are where the asymptotes intersect the line
    (A.2 = -b/a * A.1 ∨ A.2 = b/a * A.1) ∧
    (B.2 = -b/a * B.1 ∨ B.2 = b/a * B.1) ∧
    -- N is on both lines
    (N.2 = k * N.1 + m) ∧
    (N.2 = 1/k * N.1) ∧
    -- N is the midpoint of AB
    (N.1 = (A.1 + B.1) / 2 ∧ N.2 = (A.2 + B.2) / 2)) →
  -- The eccentricity of the hyperbola is √2
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 2 :=
by sorry

end hyperbola_eccentricity_l2205_220578


namespace spherical_cap_height_theorem_l2205_220527

/-- The height of a spherical cap -/
def spherical_cap_height (R : ℝ) (c : ℝ) : Set ℝ :=
  {h | h = 2*R*(c-1)/c ∨ h = 2*R*(c-2)/(c-1)}

/-- Theorem: The height of a spherical cap with radius R, whose surface area is c times 
    the area of its circular base (c > 1), is either 2R(c-1)/c or 2R(c-2)/(c-1) -/
theorem spherical_cap_height_theorem (R c : ℝ) (hR : R > 0) (hc : c > 1) :
  ∃ h ∈ spherical_cap_height R c,
    (∃ S_cap S_base : ℝ, 
      S_cap = c * S_base ∧
      ((S_cap = 2 * π * R * h ∧ S_base = π * (2*R*h - h^2)) ∨
       (S_cap = 2 * π * R * h + π * (2*R*h - h^2) ∧ S_base = π * (2*R*h - h^2)))) :=
by
  sorry

end spherical_cap_height_theorem_l2205_220527


namespace function_property_l2205_220567

theorem function_property (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x - f y) - f (f x) = -f y - 1) →
  (∀ x : ℤ, f x = x + 1) :=
by sorry

end function_property_l2205_220567


namespace polyhedron_volume_l2205_220523

/-- The volume of a polyhedron formed by a cube and a tetrahedron -/
theorem polyhedron_volume (cube_side : ℝ) (tetra_base_area : ℝ) (tetra_height : ℝ) :
  cube_side = 2 →
  tetra_base_area = 2 →
  tetra_height = 2 →
  cube_side ^ 3 + (1/3) * tetra_base_area * tetra_height = 28/3 := by
  sorry

end polyhedron_volume_l2205_220523


namespace sin_2x_minus_pi_4_increasing_l2205_220545

open Real

theorem sin_2x_minus_pi_4_increasing (k : ℤ) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → 
  x₁ ∈ Set.Ioo (- π/8 + k*π) (3*π/8 + k*π) → 
  x₂ ∈ Set.Ioo (- π/8 + k*π) (3*π/8 + k*π) → 
  sin (2*x₁ - π/4) < sin (2*x₂ - π/4) := by
sorry

end sin_2x_minus_pi_4_increasing_l2205_220545


namespace m_value_in_set_union_l2205_220574

def A (m : ℝ) : Set ℝ := {2, m}
def B (m : ℝ) : Set ℝ := {1, m^2}

theorem m_value_in_set_union (m : ℝ) :
  A m ∪ B m = {1, 2, 3, 9} → m = 3 := by
  sorry

end m_value_in_set_union_l2205_220574


namespace positive_number_equality_l2205_220570

theorem positive_number_equality (x : ℝ) (h1 : x > 0) : 
  (2 / 3) * x = (144 / 216) * (1 / x) → x = 1 := by
  sorry

end positive_number_equality_l2205_220570


namespace third_of_ten_given_metaphorical_quarter_l2205_220514

-- Define the metaphorical relationship
def metaphorical_quarter (x : ℚ) : ℚ := x / 5

-- Define the actual third
def actual_third (x : ℚ) : ℚ := x / 3

-- Theorem statement
theorem third_of_ten_given_metaphorical_quarter :
  metaphorical_quarter 20 = 4 → actual_third 10 = 8/3 :=
by
  sorry

end third_of_ten_given_metaphorical_quarter_l2205_220514


namespace square_root_of_1_5625_l2205_220558

theorem square_root_of_1_5625 : Real.sqrt 1.5625 = 1.25 := by
  sorry

end square_root_of_1_5625_l2205_220558


namespace complement_of_union_l2205_220515

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union :
  (A ∪ B)ᶜ = {0, 2, 4} := by sorry

end complement_of_union_l2205_220515


namespace alfonso_daily_earnings_l2205_220573

def helmet_cost : ℕ := 340
def savings : ℕ := 40
def days_per_week : ℕ := 5
def total_weeks : ℕ := 10

def total_working_days : ℕ := days_per_week * total_weeks

def additional_savings_needed : ℕ := helmet_cost - savings

theorem alfonso_daily_earnings :
  additional_savings_needed / total_working_days = 6 :=
by sorry

end alfonso_daily_earnings_l2205_220573


namespace inequality_proof_l2205_220599

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ ≥ a₂ ∧ a₂ ≥ a₃) (hb : b₁ ≥ b₂ ∧ b₂ ≥ b₃) : 
  3 * (a₁ * b₁ + a₂ * b₂ + a₃ * b₃) ≥ (a₁ + a₂ + a₃) * (b₁ + b₂ + b₃) := by
  sorry

end inequality_proof_l2205_220599


namespace set_operations_l2205_220552

def A : Set ℝ := {x | x < -2 ∨ x > 5}
def B : Set ℝ := {x | 4 ≤ x ∧ x ≤ 6}

theorem set_operations :
  (Aᶜ : Set ℝ) = {x | -2 ≤ x ∧ x ≤ 5} ∧
  (Bᶜ : Set ℝ) = {x | x < 4 ∨ x > 6} ∧
  (A ∩ B : Set ℝ) = {x | 5 < x ∧ x ≤ 6} ∧
  ((A ∪ B)ᶜ : Set ℝ) = {x | -2 ≤ x ∧ x < 4} := by
  sorry

end set_operations_l2205_220552


namespace fermat_point_distance_sum_l2205_220524

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (12, 0)
def C : ℝ × ℝ := (3, 5)
def P : ℝ × ℝ := (5, 3)

theorem fermat_point_distance_sum :
  let AP := distance A.1 A.2 P.1 P.2
  let BP := distance B.1 B.2 P.1 P.2
  let CP := distance C.1 C.2 P.1 P.2
  AP + BP + CP = Real.sqrt 34 + Real.sqrt 58 + 2 * Real.sqrt 2 ∧
  (1 : ℕ) + (1 : ℕ) + (2 : ℕ) = 4 := by
  sorry

end fermat_point_distance_sum_l2205_220524


namespace temperature_difference_l2205_220507

def highest_temp : Int := 9
def lowest_temp : Int := -1

theorem temperature_difference : highest_temp - lowest_temp = 10 := by
  sorry

end temperature_difference_l2205_220507


namespace rational_function_value_at_two_l2205_220520

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_cubic : ∃ a b c d : ℝ, ∀ x, q x = a * x^3 + b * x^2 + c * x + d
  asymptote_neg_four : q (-4) = 0
  asymptote_one : q 1 = 0
  passes_origin : p 0 = 0 ∧ q 0 ≠ 0
  passes_neg_one_neg_two : p (-1) / q (-1) = -2

/-- The main theorem -/
theorem rational_function_value_at_two (f : RationalFunction) : f.p 2 / f.q 2 = 8 := by
  sorry

end rational_function_value_at_two_l2205_220520


namespace manager_team_selection_l2205_220550

theorem manager_team_selection : Nat.choose 10 6 = 210 := by
  sorry

end manager_team_selection_l2205_220550


namespace smallest_four_digit_divisible_by_55_l2205_220531

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_55 :
  ∀ n : ℕ, is_four_digit n → is_divisible_by n 55 → n ≥ 1100 :=
by sorry

end smallest_four_digit_divisible_by_55_l2205_220531


namespace number_of_divisors_of_30_l2205_220557

theorem number_of_divisors_of_30 : Nat.card {d : ℕ | d > 0 ∧ 30 % d = 0} = 8 := by
  sorry

end number_of_divisors_of_30_l2205_220557


namespace seating_arrangement_l2205_220595

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def total_arrangements (n : ℕ) : ℕ := factorial n

def restricted_arrangements (n : ℕ) (k : ℕ) : ℕ := 
  factorial (n - k + 1) * factorial k

theorem seating_arrangement (n : ℕ) (k : ℕ) 
  (h1 : n = 8) (h2 : k = 4) : 
  total_arrangements n - restricted_arrangements n k = 37440 := by
  sorry

end seating_arrangement_l2205_220595


namespace coefficient_sum_equals_15625_l2205_220505

theorem coefficient_sum_equals_15625 (b₆ b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 15625 := by
sorry

end coefficient_sum_equals_15625_l2205_220505


namespace newberg_airport_passengers_l2205_220528

theorem newberg_airport_passengers (on_time late : ℕ) 
  (h1 : on_time = 14507) 
  (h2 : late = 213) : 
  on_time + late = 14620 := by
sorry

end newberg_airport_passengers_l2205_220528


namespace solution_of_equation_l2205_220530

theorem solution_of_equation :
  let f (x : ℝ) := 
    8 / (Real.sqrt (x - 10) - 10) + 
    2 / (Real.sqrt (x - 10) - 5) + 
    9 / (Real.sqrt (x - 10) + 5) + 
    16 / (Real.sqrt (x - 10) + 10)
  ∀ x : ℝ, f x = 0 ↔ x = 1841 / 121 ∨ x = 190 / 9 := by
  sorry

end solution_of_equation_l2205_220530


namespace product_of_decimals_l2205_220510

theorem product_of_decimals : (0.4 : ℝ) * 0.6 = 0.24 := by
  sorry

end product_of_decimals_l2205_220510


namespace sqrt_neg_two_squared_l2205_220502

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by sorry

end sqrt_neg_two_squared_l2205_220502


namespace tray_pieces_count_l2205_220588

def tray_length : ℕ := 24
def tray_width : ℕ := 20
def piece_length : ℕ := 3
def piece_width : ℕ := 2

theorem tray_pieces_count : 
  (tray_length * tray_width) / (piece_length * piece_width) = 80 :=
sorry

end tray_pieces_count_l2205_220588


namespace fibonacci_like_sequence_l2205_220565

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) + a n

def increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem fibonacci_like_sequence (a : ℕ → ℕ) 
  (h1 : sequence_property a) 
  (h2 : increasing_sequence a)
  (h3 : a 7 = 120) : 
  a 8 = 194 := by
sorry

end fibonacci_like_sequence_l2205_220565


namespace shaded_cells_after_five_minutes_l2205_220533

/-- Represents the state of the grid at a given minute -/
def GridState := Nat → Nat → Bool

/-- The initial state of the grid with a 1 × 5 shaded rectangle -/
def initial_state : GridState := sorry

/-- The rule for shading cells in the next minute -/
def shade_rule (state : GridState) : GridState := sorry

/-- The state of the grid after n minutes -/
def state_after (n : Nat) : GridState := sorry

/-- Counts the number of shaded cells in a given state -/
def count_shaded (state : GridState) : Nat := sorry

/-- The main theorem: after 5 minutes, 105 cells are shaded -/
theorem shaded_cells_after_five_minutes :
  count_shaded (state_after 5) = 105 := by sorry

end shaded_cells_after_five_minutes_l2205_220533


namespace sufficient_condition_implies_range_of_a_l2205_220518

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a^2 + 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 3 * (a + 1) * x + 2 * (3 * a + 1) ≤ 0}

-- Define the range of a
def RangeOfA : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3} ∪ {-1}

-- State the theorem
theorem sufficient_condition_implies_range_of_a (a : ℝ) :
  A a ⊆ B a → a ∈ RangeOfA := by sorry

end sufficient_condition_implies_range_of_a_l2205_220518


namespace david_is_seven_l2205_220508

/-- David's age in years -/
def david_age : ℕ := sorry

/-- Yuan's age in years -/
def yuan_age : ℕ := sorry

/-- Yuan is 7 years older than David -/
axiom yuan_older : yuan_age = david_age + 7

/-- Yuan is twice David's age -/
axiom yuan_twice : yuan_age = 2 * david_age

theorem david_is_seven : david_age = 7 := by sorry

end david_is_seven_l2205_220508


namespace coffee_shop_sales_l2205_220551

theorem coffee_shop_sales (teas : ℕ) (lattes : ℕ) : 
  teas = 6 → lattes = 4 * teas + 8 → lattes = 32 := by
  sorry

end coffee_shop_sales_l2205_220551


namespace car_trading_profit_l2205_220513

/-- Calculates the profit percentage for a car trading scenario -/
theorem car_trading_profit (original_price : ℝ) (h : original_price > 0) :
  let trader_buy_price := original_price * (1 - 0.2)
  let dealer_buy_price := trader_buy_price * (1 + 0.3)
  let customer_buy_price := dealer_buy_price * (1 + 0.5)
  let trader_final_price := customer_buy_price * (1 - 0.1)
  let profit := trader_final_price - trader_buy_price
  let profit_percentage := (profit / original_price) * 100
  profit_percentage = 60.4 := by
sorry


end car_trading_profit_l2205_220513


namespace chessboard_touching_squares_probability_l2205_220548

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Checks if two squares are touching -/
def are_touching (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row = s2.row ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.col = s2.col ∧ s1.row.val + 1 = s2.row.val) ∨
  (s1.col = s2.col ∧ s1.row.val = s2.row.val + 1) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val = s2.col.val + 1)

/-- Checks if two squares are the same color -/
def same_color (s1 s2 : Square) : Prop :=
  (s1.row.val + s1.col.val) % 2 = (s2.row.val + s2.col.val) % 2

theorem chessboard_touching_squares_probability :
  ∀ (s1 s2 : Square), s1 ≠ s2 → are_touching s1 s2 → ¬(same_color s1 s2) :=
by sorry

end chessboard_touching_squares_probability_l2205_220548


namespace hare_wins_by_10_meters_l2205_220561

-- Define the race parameters
def race_duration : ℕ := 50
def hare_initial_speed : ℕ := 12
def hare_later_speed : ℕ := 1
def tortoise_speed : ℕ := 3

-- Define the function to calculate the hare's distance
def hare_distance (initial_time : ℕ) : ℕ :=
  (initial_time * hare_initial_speed) + ((race_duration - initial_time) * hare_later_speed)

-- Define the function to calculate the tortoise's distance
def tortoise_distance : ℕ := race_duration * tortoise_speed

-- Theorem statement
theorem hare_wins_by_10_meters :
  ∃ (initial_time : ℕ), initial_time < race_duration ∧ 
  hare_distance initial_time = tortoise_distance + 10 :=
sorry

end hare_wins_by_10_meters_l2205_220561


namespace total_students_l2205_220592

def line_up (students_between : ℕ) (right_of_hoseok : ℕ) (left_of_yoongi : ℕ) : ℕ :=
  2 + students_between + right_of_hoseok + left_of_yoongi

theorem total_students :
  line_up 5 9 6 = 22 :=
by sorry

end total_students_l2205_220592


namespace unique_solution_l2205_220506

theorem unique_solution : ∃! (x : ℕ+), (1 : ℕ)^(x.val + 2) + 2^(x.val + 1) + 3^(x.val - 1) + 4^x.val = 1170 ∧ x = 5 := by
  sorry

end unique_solution_l2205_220506


namespace fruit_orders_eq_six_l2205_220575

/-- Represents the types of fruit in the basket -/
inductive Fruit
  | Apple
  | Peach
  | Pear

/-- The number of fruits in the basket -/
def basket_size : Nat := 3

/-- The number of chances to draw -/
def draw_chances : Nat := 2

/-- Calculates the number of different orders of fruit that can be drawn -/
def fruit_orders : Nat :=
  basket_size * (basket_size - 1)

theorem fruit_orders_eq_six :
  fruit_orders = 6 :=
sorry

end fruit_orders_eq_six_l2205_220575


namespace rajan_income_l2205_220585

/-- Represents the financial situation of two individuals -/
structure FinancialSituation where
  income_ratio : Rat
  expenditure_ratio : Rat
  savings : ℕ

/-- Calculates the income based on the given financial situation -/
def calculate_income (fs : FinancialSituation) : ℕ :=
  sorry

/-- Theorem stating that given the specific financial situation, Rajan's income is 7000 -/
theorem rajan_income (fs : FinancialSituation) 
  (h1 : fs.income_ratio = 7/6)
  (h2 : fs.expenditure_ratio = 6/5)
  (h3 : fs.savings = 1000) :
  calculate_income fs = 7000 := by
  sorry

end rajan_income_l2205_220585


namespace final_sum_after_operations_l2205_220519

theorem final_sum_after_operations (x y T : ℝ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 := by
  sorry

end final_sum_after_operations_l2205_220519


namespace additive_function_is_linear_l2205_220546

theorem additive_function_is_linear (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y) :
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x :=
sorry

end additive_function_is_linear_l2205_220546


namespace division_remainder_problem_l2205_220522

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1375 →
  L = 1632 →
  L = 6 * S + R →
  R < S →
  R = 90 :=
by sorry

end division_remainder_problem_l2205_220522


namespace remainder_theorem_l2205_220559

-- Define the polynomial f(r) = r^15 - 3
def f (r : ℝ) : ℝ := r^15 - 3

-- Theorem statement
theorem remainder_theorem (r : ℝ) : 
  (f r) % (r - 2) = 32765 := by
  sorry

end remainder_theorem_l2205_220559


namespace chessboard_nail_configuration_l2205_220536

/-- Represents a point on the chessboard --/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Checks if three points are collinear --/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- A configuration of 16 points on the chessboard --/
def Configuration := Fin 16 → Point

/-- Predicate to check if a configuration is valid --/
def valid_configuration (config : Configuration) : Prop :=
  (∀ i j k : Fin 16, i ≠ j → j ≠ k → i ≠ k → ¬collinear (config i) (config j) (config k))

theorem chessboard_nail_configuration :
  ∃ (config : Configuration), valid_configuration config :=
sorry

end chessboard_nail_configuration_l2205_220536


namespace complex_number_purely_imaginary_l2205_220549

theorem complex_number_purely_imaginary (a : ℝ) : 
  (a = -1) ↔ (∃ (t : ℝ), (1 + I) / (1 + a * I) = t * I) :=
sorry

end complex_number_purely_imaginary_l2205_220549


namespace cookies_packs_l2205_220590

theorem cookies_packs (total packs_cake packs_chocolate : ℕ) 
  (h1 : total = 42)
  (h2 : packs_cake = 22)
  (h3 : packs_chocolate = 16) :
  total - packs_cake - packs_chocolate = 4 := by
  sorry

end cookies_packs_l2205_220590


namespace min_k_for_inequality_l2205_220581

theorem min_k_for_inequality (k : ℝ) : 
  (∀ x : ℝ, x > 0 → k * x ≥ (Real.sin x) / (2 + Real.cos x)) ↔ k ≥ 1/3 :=
by sorry

end min_k_for_inequality_l2205_220581


namespace tan_pi_fourth_plus_alpha_l2205_220571

theorem tan_pi_fourth_plus_alpha (α : Real) (h : Real.tan α = 2) : 
  Real.tan (π/4 + α) = -3 := by
  sorry

end tan_pi_fourth_plus_alpha_l2205_220571


namespace bagel_cost_proof_l2205_220535

/-- The cost of a dozen bagels when bought together -/
def dozen_cost : ℝ := 24

/-- The amount saved per bagel when buying a dozen -/
def savings_per_bagel : ℝ := 0.25

/-- The number of bagels in a dozen -/
def dozen : ℕ := 12

/-- The individual cost of a bagel -/
def individual_cost : ℝ := 2.25

theorem bagel_cost_proof :
  individual_cost = (dozen_cost + dozen * savings_per_bagel) / dozen :=
by sorry

end bagel_cost_proof_l2205_220535


namespace largest_term_binomial_sequence_l2205_220579

theorem largest_term_binomial_sequence (k : ℕ) :
  k ≤ 1992 →
  k * Nat.choose 1992 k ≤ 997 * Nat.choose 1992 997 :=
by sorry

end largest_term_binomial_sequence_l2205_220579


namespace compare_two_point_five_and_sqrt_six_l2205_220503

theorem compare_two_point_five_and_sqrt_six :
  2.5 > Real.sqrt 6 := by
  sorry

end compare_two_point_five_and_sqrt_six_l2205_220503


namespace no_rational_roots_for_odd_coefficients_l2205_220555

theorem no_rational_roots_for_odd_coefficients (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  ¬∃ (x : ℚ), x^2 + 2*↑p*x + 2*↑q = 0 := by
  sorry

end no_rational_roots_for_odd_coefficients_l2205_220555


namespace homologous_pair_from_both_parents_l2205_220553

/-- Represents a chromosome in a human cell -/
structure Chromosome where
  parent : Bool  -- true for paternal, false for maternal

/-- Represents a pair of homologous chromosomes -/
structure HomologousPair where
  chromosome1 : Chromosome
  chromosome2 : Chromosome

/-- Represents a human cell -/
structure HumanCell where
  chromosomePairs : List HomologousPair

/-- Axiom: Humans reproduce sexually -/
axiom human_sexual_reproduction : True

/-- Axiom: Fertilization involves fusion of sperm and egg cells -/
axiom fertilization_fusion : True

/-- Axiom: Meiosis occurs in formation of reproductive cells -/
axiom meiosis_in_reproduction : True

/-- Axiom: Zygote chromosome count is restored to somatic cell count -/
axiom zygote_chromosome_restoration : True

/-- Axiom: Half of zygote chromosomes from sperm, half from egg -/
axiom zygote_chromosome_origin : True

/-- Theorem: Each pair of homologous chromosomes is provided by both parents -/
theorem homologous_pair_from_both_parents (cell : HumanCell) : 
  ∀ pair ∈ cell.chromosomePairs, pair.chromosome1.parent ≠ pair.chromosome2.parent := by
  sorry


end homologous_pair_from_both_parents_l2205_220553


namespace circle_symmetric_points_line_l2205_220511

/-- Circle with center (-1, 3) and radius 3 -/
def Circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9

/-- Line with equation x + my + 4 = 0 -/
def Line (m x y : ℝ) : Prop := x + m * y + 4 = 0

/-- Two points are symmetric with respect to a line -/
def SymmetricPoints (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  Line m ((P.1 + Q.1) / 2) ((P.2 + Q.2) / 2)

theorem circle_symmetric_points_line (m : ℝ) :
  (∃ P Q : ℝ × ℝ, Circle P.1 P.2 ∧ Circle Q.1 Q.2 ∧ SymmetricPoints P Q m) →
  m = -1 := by
  sorry

end circle_symmetric_points_line_l2205_220511


namespace mans_age_to_sons_age_ratio_l2205_220587

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 30 years older than his son and the son's current age is 28 years. -/
theorem mans_age_to_sons_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 28 →
    man_age = son_age + 30 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end mans_age_to_sons_age_ratio_l2205_220587


namespace die_probability_l2205_220593

/-- The number of times the die is tossed -/
def n : ℕ := 30

/-- The number of faces on the die -/
def faces : ℕ := 6

/-- The number of favorable outcomes before the first six -/
def favorable_before : ℕ := 3

/-- Probability of the event: at least one six appears, and no five or four appears before the first six -/
def prob_event : ℚ :=
  1 / 3

theorem die_probability :
  prob_event = (favorable_before ^ (n - 1) * (2 ^ n - 1)) / (faces ^ n) :=
sorry

end die_probability_l2205_220593


namespace ninth_term_value_l2205_220580

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧  -- Definition of arithmetic sequence
  (a 5 + a 7 = 16) ∧                         -- Given condition
  (a 3 = 4)                                  -- Given condition

/-- Theorem stating the value of the 9th term -/
theorem ninth_term_value (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 9 = 12 := by
  sorry

end ninth_term_value_l2205_220580


namespace condition_for_f_sum_positive_l2205_220540

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem condition_for_f_sum_positive :
  ∀ (a b : ℝ), (a + b > 0 ↔ f a + f b > 0) := by sorry

end condition_for_f_sum_positive_l2205_220540


namespace value_of_a_l2205_220596

theorem value_of_a (x y a : ℝ) 
  (h1 : 3^x = a) 
  (h2 : 5^y = a) 
  (h3 : 1/x + 1/y = 2) : 
  a = Real.sqrt 15 := by
  sorry

end value_of_a_l2205_220596


namespace nabla_problem_l2205_220529

-- Define the operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_problem : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end nabla_problem_l2205_220529


namespace vector_dot_product_result_l2205_220512

theorem vector_dot_product_result :
  let a : ℝ × ℝ := (Real.cos (45 * π / 180), Real.sin (45 * π / 180))
  let b : ℝ × ℝ := (Real.cos (15 * π / 180), Real.sin (15 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = Real.sqrt 3 / 2 :=
by sorry

end vector_dot_product_result_l2205_220512


namespace square_of_negative_two_times_a_cubed_l2205_220568

theorem square_of_negative_two_times_a_cubed (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end square_of_negative_two_times_a_cubed_l2205_220568
