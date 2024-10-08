import Mathlib

namespace evaluate_expression_l107_107243

theorem evaluate_expression : 
  1 + 2 / (3 + 4 / (5 + 6 / 7)) = 233 / 151 := 
by 
  sorry

end evaluate_expression_l107_107243


namespace range_of_m_l107_107590

theorem range_of_m (m : ℝ) : (∃ x : ℝ, y = (m-2)*x + m ∧ x > 0 ∧ y > 0) ∧ 
                              (∃ x : ℝ, y = (m-2)*x + m ∧ x < 0 ∧ y > 0) ∧ 
                              (∃ x : ℝ, y = (m-2)*x + m ∧ x > 0 ∧ y < 0) ↔ 0 < m ∧ m < 2 :=
by sorry

end range_of_m_l107_107590


namespace isosceles_triangle_perimeter_l107_107908

-- Definitions of the conditions
def is_isosceles (a b : ℕ) : Prop :=
  a = b

def has_side_lengths (a b : ℕ) (c : ℕ) : Prop :=
  true

-- The statement to be proved
theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h₁ : is_isosceles a b) (h₂ : has_side_lengths a b c) :
  (a + b + c = 16 ∨ a + b + c = 17) :=
sorry

end isosceles_triangle_perimeter_l107_107908


namespace range_of_x0_l107_107545

noncomputable def point_on_circle_and_line (x0 : ℝ) (y0 : ℝ) : Prop :=
(x0^2 + y0^2 = 1) ∧ (3 * x0 + 2 * y0 = 4)

theorem range_of_x0 
  (x0 : ℝ) (y0 : ℝ) 
  (h1 : 3 * x0 + 2 * y0 = 4)
  (h2 : ∃ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A ≠ B) ∧ (A + B = (x0, y0))) :
  0 < x0 ∧ x0 < 24 / 13 :=
sorry

end range_of_x0_l107_107545


namespace intersection_of_complements_l107_107979

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := U \ A

theorem intersection_of_complements :
  U = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
  A = {0, 1, 3, 5, 8} →
  B = {2, 4, 5, 6, 8} →
  (complement U A ∩ complement U B) = {7, 9} :=
by
  intros hU hA hB
  sorry

end intersection_of_complements_l107_107979


namespace calculate_f_of_f_of_f_l107_107599

def f (x : ℤ) : ℤ := 5 * x - 4

theorem calculate_f_of_f_of_f (h : f (f (f 3)) = 251) : f (f (f 3)) = 251 := 
by sorry

end calculate_f_of_f_of_f_l107_107599


namespace total_area_is_8_units_l107_107950

-- Let s be the side length of the original square and x be the leg length of each isosceles right triangle
variables (s x : ℕ)

-- The side length of the smaller square is 8 units
axiom smaller_square_length : s - 2 * x = 8

-- The area of one isosceles right triangle
def area_triangle : ℕ := x * x / 2

-- There are four triangles
def total_area_triangles : ℕ := 4 * area_triangle x

-- The aim is to prove that the total area of the removed triangles is 8 square units
theorem total_area_is_8_units : total_area_triangles x = 8 :=
sorry

end total_area_is_8_units_l107_107950


namespace health_risk_factor_prob_l107_107587

noncomputable def find_p_q_sum (p q: ℕ) : ℕ :=
if h1 : p.gcd q = 1 then
  31
else 
  sorry

theorem health_risk_factor_prob (p q : ℕ) (h1 : p.gcd q = 1) 
                                (h2 : (p : ℚ) / q = 5 / 26) :
  find_p_q_sum p q = 31 :=
sorry

end health_risk_factor_prob_l107_107587


namespace squirrel_acorns_l107_107055

theorem squirrel_acorns :
  ∀ (total_acorns : ℕ)
    (first_month_percent second_month_percent third_month_percent : ℝ)
    (first_month_consumed second_month_consumed third_month_consumed : ℝ),
    total_acorns = 500 →
    first_month_percent = 0.40 →
    second_month_percent = 0.30 →
    third_month_percent = 0.30 →
    first_month_consumed = 0.20 →
    second_month_consumed = 0.25 →
    third_month_consumed = 0.15 →
    let first_month_acorns := total_acorns * first_month_percent
    let second_month_acorns := total_acorns * second_month_percent
    let third_month_acorns := total_acorns * third_month_percent
    let remaining_first_month := first_month_acorns - (first_month_consumed * first_month_acorns)
    let remaining_second_month := second_month_acorns - (second_month_consumed * second_month_acorns)
    let remaining_third_month := third_month_acorns - (third_month_consumed * third_month_acorns)
    remaining_first_month + remaining_second_month + remaining_third_month = 400 := 
by
  intros 
    total_acorns
    first_month_percent second_month_percent third_month_percent
    first_month_consumed second_month_consumed third_month_consumed
    h_total
    h_first_percent
    h_second_percent
    h_third_percent
    h_first_consumed
    h_second_consumed
    h_third_consumed
  let first_month_acorns := total_acorns * first_month_percent
  let second_month_acorns := total_acorns * second_month_percent
  let third_month_acorns := total_acorns * third_month_percent
  let remaining_first_month := first_month_acorns - (first_month_consumed * first_month_acorns)
  let remaining_second_month := second_month_acorns - (second_month_consumed * second_month_acorns)
  let remaining_third_month := third_month_acorns - (third_month_consumed * third_month_acorns)
  sorry

end squirrel_acorns_l107_107055


namespace probability_of_event_a_l107_107538

-- Given conditions and question
variables (a b : Prop)
variables (p : Prop → ℝ)

-- Given conditions
axiom p_a : p a = 4 / 5
axiom p_b : p b = 2 / 5
axiom p_a_and_b_given : p (a ∧ b) = 0.32
axiom independent_a_b : p (a ∧ b) = p a * p b

-- The proof statement we need to prove: p a = 0.8
theorem probability_of_event_a :
  p a = 0.8 :=
sorry

end probability_of_event_a_l107_107538


namespace min_bottles_needed_l107_107317

theorem min_bottles_needed (bottle_size : ℕ) (min_ounces : ℕ) (n : ℕ) 
  (h1 : bottle_size = 15) 
  (h2 : min_ounces = 195) 
  (h3 : 15 * n >= 195) : n = 13 :=
sorry

end min_bottles_needed_l107_107317


namespace div_246_by_73_sum_9999_999_99_9_prod_25_29_4_l107_107645

-- Define the division of 246 by 73
theorem div_246_by_73 :
  246 / 73 = 3 + 27 / 73 :=
sorry

-- Define the sum calculation
theorem sum_9999_999_99_9 :
  9999 + 999 + 99 + 9 = 11106 :=
sorry

-- Define the product calculation
theorem prod_25_29_4 :
  25 * 29 * 4 = 2900 :=
sorry

end div_246_by_73_sum_9999_999_99_9_prod_25_29_4_l107_107645


namespace paul_tickets_left_l107_107667

theorem paul_tickets_left (initial_tickets : ℕ) (spent_tickets : ℕ) (remaining_tickets : ℕ) :
  initial_tickets = 11 → spent_tickets = 3 → remaining_tickets = initial_tickets - spent_tickets → remaining_tickets = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end paul_tickets_left_l107_107667


namespace purely_imaginary_z_point_on_line_z_l107_107989

-- Proof problem for (I)
theorem purely_imaginary_z (a : ℝ) (z : ℂ) (h : z = Complex.mk 0 (a+2)) 
: a = 2 :=
sorry

-- Proof problem for (II)
theorem point_on_line_z (a : ℝ) (x y : ℝ) (h1 : x = a^2-4) (h2 : y = a+2) (h3 : x + 2*y + 1 = 0) 
: a = -1 :=
sorry

end purely_imaginary_z_point_on_line_z_l107_107989


namespace total_blocks_needed_l107_107759

theorem total_blocks_needed (length height : ℕ) (block_height : ℕ) (block1_length block2_length : ℕ)
                            (height_blocks : height = 8) (length_blocks : length = 102)
                            (block_height_cond : block_height = 1)
                            (block_lengths : block1_length = 2 ∧ block2_length = 1)
                            (staggered_cond : True) (even_ends : True) :
  ∃ total_blocks, total_blocks = 416 := 
  sorry

end total_blocks_needed_l107_107759


namespace min_distance_from_circle_to_line_l107_107982

-- Define the circle and line conditions
def is_on_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def line (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- The theorem to prove
theorem min_distance_from_circle_to_line (x y : ℝ) (h : is_on_circle x y) : 
  ∃ m_dist : ℝ, m_dist = 2 :=
by
  -- Place holder proof
  sorry

end min_distance_from_circle_to_line_l107_107982


namespace community_cleaning_children_l107_107339

theorem community_cleaning_children (total_members adult_men_ratio adult_women_ratio : ℕ) 
(h_total : total_members = 2000)
(h_men_ratio : adult_men_ratio = 30) 
(h_women_ratio : adult_women_ratio = 2) :
  (total_members - (adult_men_ratio * total_members / 100 + 
  adult_women_ratio * (adult_men_ratio * total_members / 100))) = 200 :=
by
  sorry

end community_cleaning_children_l107_107339


namespace find_k_l107_107941

noncomputable def geometric_series_sum (k : ℝ) (h : k > 1) : ℝ :=
  ∑' n, ((7 * n - 2) / k ^ n)

theorem find_k (k : ℝ) (h : k > 1)
  (series_sum : geometric_series_sum k h = 18 / 5) :
  k = 3.42 :=
by
  sorry

end find_k_l107_107941


namespace rain_on_tuesday_l107_107785

/-- Let \( R_M \) be the event that a county received rain on Monday. -/
def RM : Prop := sorry

/-- Let \( R_T \) be the event that a county received rain on Tuesday. -/
def RT : Prop := sorry

/-- Let \( R_{MT} \) be the event that a county received rain on both Monday and Tuesday. -/
def RMT : Prop := RM ∧ RT

/-- The probability that a county received rain on Monday is 0.62. -/
def prob_RM : ℝ := 0.62

/-- The probability that a county received rain on both Monday and Tuesday is 0.44. -/
def prob_RMT : ℝ := 0.44

/-- The probability that no rain fell on either day is 0.28. -/
def prob_no_rain : ℝ := 0.28

/-- The probability that a county received rain on at least one of the days is 0.72. -/
def prob_at_least_one_day : ℝ := 1 - prob_no_rain

/-- The probability that a county received rain on Tuesday is 0.54. -/
theorem rain_on_tuesday : (prob_at_least_one_day = prob_RM + x - prob_RMT) → (x = 0.54) :=
by
  intros h
  sorry

end rain_on_tuesday_l107_107785


namespace circle_circumference_ratio_l107_107835

theorem circle_circumference_ratio (q r p : ℝ) (hq : p = q + r) : 
  (2 * Real.pi * q + 2 * Real.pi * r) / (2 * Real.pi * p) = 1 :=
by
  sorry

end circle_circumference_ratio_l107_107835


namespace circle_parabola_intersection_l107_107074

theorem circle_parabola_intersection (b : ℝ) :
  (∃ c r, ∀ x y : ℝ, y = (5 / 12) * x^2 → ((x - c)^2 + (y - b)^2 = r^2) ∧ 
   (y = (5 / 12) * x + b → ((x - c)^2 + (y - b)^2 = r^2))) → b = 169 / 60 :=
by
  sorry

end circle_parabola_intersection_l107_107074


namespace ice_cubes_per_cup_l107_107064

theorem ice_cubes_per_cup (total_ice_cubes number_of_cups : ℕ) (h1 : total_ice_cubes = 30) (h2 : number_of_cups = 6) : 
  total_ice_cubes / number_of_cups = 5 := 
by
  sorry

end ice_cubes_per_cup_l107_107064


namespace radio_advertiser_savings_l107_107214

def total_store_price : ℚ := 299.99
def ad_payment : ℚ := 55.98
def payments_count : ℚ := 5
def shipping_handling : ℚ := 12.99

def total_ad_price : ℚ := payments_count * ad_payment + shipping_handling

def savings_in_dollars : ℚ := total_store_price - total_ad_price
def savings_in_cents : ℚ := savings_in_dollars * 100

theorem radio_advertiser_savings :
  savings_in_cents = 710 := by
  sorry

end radio_advertiser_savings_l107_107214


namespace total_books_from_library_l107_107383

def initialBooks : ℕ := 54
def additionalBooks : ℕ := 23

theorem total_books_from_library : initialBooks + additionalBooks = 77 := by
  sorry

end total_books_from_library_l107_107383


namespace one_div_a_plus_one_div_b_l107_107454

theorem one_div_a_plus_one_div_b (a b : ℝ) (h₀ : a ≠ b) (ha : a^2 - 3 * a + 2 = 0) (hb : b^2 - 3 * b + 2 = 0) :
  1 / a + 1 / b = 3 / 2 :=
by
  -- Proof goes here
  sorry

end one_div_a_plus_one_div_b_l107_107454


namespace Uki_earnings_l107_107647

theorem Uki_earnings (cupcake_price cookie_price biscuit_price : ℝ) 
                     (cupcake_count cookie_count biscuit_count : ℕ)
                     (days : ℕ) :
  cupcake_price = 1.50 →
  cookie_price = 2 →
  biscuit_price = 1 →
  cupcake_count = 20 →
  cookie_count = 10 →
  biscuit_count = 20 →
  days = 5 →
  (days : ℝ) * (cupcake_price * (cupcake_count : ℝ) + cookie_price * (cookie_count : ℝ) + biscuit_price * (biscuit_count : ℝ)) = 350 := 
by
  sorry

end Uki_earnings_l107_107647


namespace interest_rate_simple_and_compound_l107_107241

theorem interest_rate_simple_and_compound (P T: ℝ) (SI CI R: ℝ) 
  (simple_interest_eq: SI = (P * R * T) / 100)
  (compound_interest_eq: CI = P * ((1 + R / 100) ^ T - 1)) 
  (hP : P = 3000) (hT : T = 2) (hSI : SI = 300) (hCI : CI = 307.50) :
  R = 5 :=
by
  sorry

end interest_rate_simple_and_compound_l107_107241


namespace solution_set_inequality_l107_107188

theorem solution_set_inequality (x : ℝ) : ((x - 1) * (x + 2) < 0) ↔ (-2 < x ∧ x < 1) := by
  sorry

end solution_set_inequality_l107_107188


namespace number_of_refuels_needed_l107_107322

noncomputable def fuelTankCapacity : ℕ := 50
noncomputable def distanceShanghaiHarbin : ℕ := 2560
noncomputable def fuelConsumptionRate : ℕ := 8
noncomputable def safetyFuel : ℕ := 6

theorem number_of_refuels_needed
  (fuelTankCapacity : ℕ)
  (distanceShanghaiHarbin : ℕ)
  (fuelConsumptionRate : ℕ)
  (safetyFuel : ℕ) :
  (fuelTankCapacity = 50) →
  (distanceShanghaiHarbin = 2560) →
  (fuelConsumptionRate = 8) →
  (safetyFuel = 6) →
  ∃ n : ℕ, n = 4 := by
  sorry

end number_of_refuels_needed_l107_107322


namespace correct_pair_has_integer_distance_l107_107098

-- Define the pairs of (x, y)
def pairs : List (ℕ × ℕ) :=
  [(88209, 90288), (82098, 89028), (28098, 89082), (90882, 28809)]

-- Define the property: a pair (x, y) has the distance √(x^2 + y^2) as an integer
def is_integer_distance_pair (x y : ℕ) : Prop :=
  ∃ (n : ℕ), n * n = x * x + y * y

-- Translate the problem to the proof: Prove (88209, 90288) satisfies the given property
theorem correct_pair_has_integer_distance :
  is_integer_distance_pair 88209 90288 :=
by
  sorry

end correct_pair_has_integer_distance_l107_107098


namespace hotel_rooms_count_l107_107737

theorem hotel_rooms_count
  (TotalLamps : ℕ) (TotalChairs : ℕ) (TotalBedSheets : ℕ)
  (LampsPerRoom : ℕ) (ChairsPerRoom : ℕ) (BedSheetsPerRoom : ℕ) :
  TotalLamps = 147 → 
  TotalChairs = 84 → 
  TotalBedSheets = 210 → 
  LampsPerRoom = 7 → 
  ChairsPerRoom = 4 → 
  BedSheetsPerRoom = 10 →
  (TotalLamps / LampsPerRoom = 21) ∧ 
  (TotalChairs / ChairsPerRoom = 21) ∧ 
  (TotalBedSheets / BedSheetsPerRoom = 21) :=
by
  intros
  sorry

end hotel_rooms_count_l107_107737


namespace sufficient_but_not_necessary_condition_l107_107923

variable (x : ℝ)

def p : Prop := (x - 1) / (x + 2) ≥ 0
def q : Prop := (x - 1) * (x + 2) ≥ 0

theorem sufficient_but_not_necessary_condition : (p x) → (q x) ∧ ¬((q x) → (p x)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l107_107923


namespace years_since_mothers_death_l107_107827

noncomputable def jessica_age_at_death (x : ℕ) : ℕ := 40 - x
noncomputable def mother_age_at_death (x : ℕ) : ℕ := 2 * jessica_age_at_death x

theorem years_since_mothers_death (x : ℕ) : mother_age_at_death x + x = 70 ↔ x = 10 :=
by
  sorry

end years_since_mothers_death_l107_107827


namespace volume_ratio_of_cones_l107_107552

theorem volume_ratio_of_cones (R : ℝ) (hR : 0 < R) :
  let circumference := 2 * Real.pi * R
  let sector1_circumference := (2 / 3) * circumference
  let sector2_circumference := (1 / 3) * circumference
  let r1 := sector1_circumference / (2 * Real.pi)
  let r2 := sector2_circumference / (2 * Real.pi)
  let s := R
  let h1 := Real.sqrt (R^2 - r1^2)
  let h2 := Real.sqrt (R^2 - r2^2)
  let V1 := (Real.pi * r1^2 * h1) / 3
  let V2 := (Real.pi * r2^2 * h2) / 3
  V1 / V2 = Real.sqrt 10 := 
by
  sorry

end volume_ratio_of_cones_l107_107552


namespace percentage_reduction_l107_107900

theorem percentage_reduction (S P : ℝ) (h : S - (P / 100) * S = S / 2) : P = 50 :=
by
  sorry

end percentage_reduction_l107_107900


namespace area_of_triangle_CM_N_l107_107404

noncomputable def triangle_area (a : ℝ) : ℝ :=
  let M := (a / 2, a, a)
  let N := (a, a / 2, a)
  let MN := Real.sqrt ((a - a / 2) ^ 2 + (a / 2 - a) ^ 2)
  let CK := Real.sqrt (a ^ 2 + (a * Real.sqrt 2 / 4) ^ 2)
  (1/2) * MN * CK

theorem area_of_triangle_CM_N 
  (a : ℝ) :
  (a > 0) →
  triangle_area a = (3 * a^2) / 8 :=
by
  intro h
  -- Proof will go here.
  sorry

end area_of_triangle_CM_N_l107_107404


namespace water_pump_calculation_l107_107056

-- Define the given initial conditions
variables (f h j g k l m : ℕ)

-- Provide the correctly calculated answer
theorem water_pump_calculation (hf : f > 0) (hg : g > 0) (hk : k > 0) (hm : m > 0) : 
  (k * l * m * j * h) / (10000 * f * g) = (k * (j * h / (f * g)) * l * m) / 10000 := 
sorry

end water_pump_calculation_l107_107056


namespace problem_part1_problem_part2_l107_107205

variable {θ m : ℝ}
variable {h₀ : θ ∈ Ioo 0 (Real.pi / 2)}
variable {h₁ : Real.sin θ + Real.cos θ = (Real.sqrt 3 + 1) / 2}
variable {h₂ : Real.sin θ * Real.cos θ = m / 2}

theorem problem_part1 :
  (Real.sin θ / (1 - 1 / Real.tan θ) + Real.cos θ / (1 - Real.tan θ)) = (Real.sqrt 3 + 1) / 2 :=
sorry

theorem problem_part2 :
  m = Real.sqrt 3 / 2 ∧ (θ = Real.pi / 6 ∨ θ = Real.pi / 3) :=
sorry

end problem_part1_problem_part2_l107_107205


namespace smallest_c_plus_d_l107_107364

theorem smallest_c_plus_d :
  ∃ (c d : ℕ), (8 * c + 3 = 3 * d + 8) ∧ c + d = 27 :=
by
  sorry

end smallest_c_plus_d_l107_107364


namespace number_of_ways_to_partition_22_as_triangle_pieces_l107_107351

theorem number_of_ways_to_partition_22_as_triangle_pieces : 
  (∃ (a b c : ℕ), a + b + c = 22 ∧ a + b > c ∧ a + c > b ∧ b + c > a) → 
  ∃! (count : ℕ), count = 10 :=
by sorry

end number_of_ways_to_partition_22_as_triangle_pieces_l107_107351


namespace expected_value_of_win_is_2_5_l107_107969

noncomputable def expected_value_of_win : ℚ := 
  (1/6) * (6 - 1) + (1/6) * (6 - 2) + (1/6) * (6 - 3) + 
  (1/6) * (6 - 4) + (1/6) * (6 - 5) + (1/6) * (6 - 6)

theorem expected_value_of_win_is_2_5 : expected_value_of_win = 5 / 2 := 
by
  -- Proof steps will go here
  sorry

end expected_value_of_win_is_2_5_l107_107969


namespace last_digit_base5_89_l107_107182

theorem last_digit_base5_89 : 
  ∃ (b : ℕ), (89 : ℕ) = b * 5 + 4 :=
by
  -- The theorem above states that there exists an integer b, such that when we compute 89 in base 5, 
  -- its last digit is 4.
  sorry

end last_digit_base5_89_l107_107182


namespace range_of_a_l107_107570

variable (a : ℝ)

def p := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0
def r := -1 ≤ a ∧ a ≤ 1 ∨ a > 3

theorem range_of_a
  (h₀ : p a ∨ q a)
  (h₁ : ¬ (p a ∧ q a)) :
  r a :=
sorry

end range_of_a_l107_107570


namespace div_relation_l107_107988

theorem div_relation (a b d : ℝ) (h1 : a / b = 3) (h2 : b / d = 2 / 5) : d / a = 5 / 6 := by
  sorry

end div_relation_l107_107988


namespace leibo_orange_price_l107_107266

variable (x y m : ℝ)

theorem leibo_orange_price :
  (3 * x + 2 * y = 78) ∧ (2 * x + 3 * y = 72) ∧ (18 * m + 12 * (100 - m) ≤ 1440) → (x = 18) ∧ (y = 12) ∧ (m ≤ 40) :=
by
  intros h
  sorry

end leibo_orange_price_l107_107266


namespace quadrilateral_identity_l107_107090

theorem quadrilateral_identity 
  {A B C D : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (AC : ℝ) (BD : ℝ)
  (angle_A : ℝ) (angle_C : ℝ) 
  (h_angle_sum : angle_A + angle_C = 120)
  : (AC * BD)^2 = (AB * CD)^2 + (BC * AD)^2 + AB * BC * CD * DA := 
by {
  sorry
}

end quadrilateral_identity_l107_107090


namespace four_xyz_value_l107_107847

theorem four_xyz_value (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 4 * x * y * z = 48 := by
  sorry

end four_xyz_value_l107_107847


namespace real_roots_condition_l107_107307

theorem real_roots_condition (a : ℝ) (h : a ≠ -1) : 
    (∃ x : ℝ, x^2 + a * x + (a + 1)^2 = 0) ↔ a ∈ Set.Icc (-2 : ℝ) (-2 / 3) :=
sorry

end real_roots_condition_l107_107307


namespace midpoint_of_segment_l107_107913

theorem midpoint_of_segment (A B : (ℤ × ℤ)) (hA : A = (12, 3)) (hB : B = (-8, -5)) :
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = -1 :=
by
  sorry

end midpoint_of_segment_l107_107913


namespace find_difference_of_a_b_l107_107970

noncomputable def a_b_are_relative_prime_and_positive (a b : ℕ) (hab_prime : Nat.gcd a b = 1) (ha_pos : a > 0) (hb_pos : b > 0) (h_gt : a > b) : Prop :=
  a ^ 3 - b ^ 3 = (131 / 5) * (a - b) ^ 3

theorem find_difference_of_a_b (a b : ℕ) 
  (hab_prime : Nat.gcd a b = 1) 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (h_gt : a > b) 
  (h_eq : (a ^ 3 - b ^ 3 : ℚ) / (a - b) ^ 3 = 131 / 5) : 
  a - b = 7 :=
  sorry

end find_difference_of_a_b_l107_107970


namespace remainder_of_p_l107_107041

theorem remainder_of_p (p : ℤ) (h1 : p = 35 * 17 + 10) : p % 35 = 10 := 
  sorry

end remainder_of_p_l107_107041


namespace janes_score_l107_107614

theorem janes_score (jane_score tom_score : ℕ) (h1 : jane_score = tom_score + 50) (h2 : (jane_score + tom_score) / 2 = 90) :
  jane_score = 115 :=
sorry

end janes_score_l107_107614


namespace line_of_symmetry_is_x_eq_0_l107_107411

variable (f : ℝ → ℝ)

theorem line_of_symmetry_is_x_eq_0 :
  (∀ y, f (10 + y) = f (10 - y)) → ( ∃ l, l = 0 ∧ ∀ x,  f (10 + l + x) = f (10 + l - x)) := 
by
  sorry

end line_of_symmetry_is_x_eq_0_l107_107411


namespace wrapping_paper_area_l107_107173

variable {l w h : ℝ}

theorem wrapping_paper_area (hl : 0 < l) (hw : 0 < w) (hh : 0 < h) :
  (4 * l * h + 2 * l * h + 2 * w * h) = 6 * l * h + 2 * w * h :=
  sorry

end wrapping_paper_area_l107_107173


namespace largest_r_in_subset_l107_107684

theorem largest_r_in_subset (A : Finset ℕ) (hA : A.card = 500) : 
  ∃ (B C : Finset ℕ), B ⊆ A ∧ C ⊆ A ∧ (B ∩ C).card ≥ 100 := sorry

end largest_r_in_subset_l107_107684


namespace power_calculation_l107_107951

theorem power_calculation :
  ((8^5 / 8^3) * 4^6) = 262144 := by
  sorry

end power_calculation_l107_107951


namespace min_value_of_1_over_a_plus_2_over_b_l107_107601

theorem min_value_of_1_over_a_plus_2_over_b (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  (1 / a + 2 / b) ≥ 9 := 
sorry

end min_value_of_1_over_a_plus_2_over_b_l107_107601


namespace scheduling_arrangements_correct_l107_107268

-- Define the set of employees
inductive Employee
| A | B | C | D | E | F deriving DecidableEq

open Employee

-- Define the days of the festival
inductive Day
| May31 | June1 | June2 deriving DecidableEq

open Day

def canWork (e : Employee) (d : Day) : Prop :=
match e, d with
| A, May31 => False
| B, June2 => False
| _, _ => True

def schedulingArrangements : ℕ :=
  -- Calculations go here, placeholder for now
  sorry

theorem scheduling_arrangements_correct : schedulingArrangements = 42 := 
  sorry

end scheduling_arrangements_correct_l107_107268


namespace calculate_total_weight_AlBr3_l107_107449

-- Definitions for the atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_Br : ℝ := 79.90

-- Definition for the molecular weight of AlBr3
def molecular_weight_AlBr3 : ℝ := atomic_weight_Al + 3 * atomic_weight_Br

-- Number of moles
def number_of_moles : ℝ := 5

-- Total weight of 5 moles of AlBr3
def total_weight_5_moles_AlBr3 : ℝ := molecular_weight_AlBr3 * number_of_moles

-- Desired result
def expected_total_weight : ℝ := 1333.40

-- Statement to prove that total_weight_5_moles_AlBr3 equals the expected total weight
theorem calculate_total_weight_AlBr3 :
  total_weight_5_moles_AlBr3 = expected_total_weight :=
sorry

end calculate_total_weight_AlBr3_l107_107449


namespace find_value_of_A_l107_107477

theorem find_value_of_A (x y A : ℝ)
  (h1 : 2^x = A)
  (h2 : 7^(2*y) = A)
  (h3 : 1 / x + 2 / y = 2) : 
  A = 7 * Real.sqrt 2 := 
sorry

end find_value_of_A_l107_107477


namespace cos_555_value_l107_107523

noncomputable def cos_555_equals_neg_sqrt6_add_sqrt2_div4 : Prop :=
  (Real.cos 555 = -((Real.sqrt 6 + Real.sqrt 2) / 4))

theorem cos_555_value : cos_555_equals_neg_sqrt6_add_sqrt2_div4 :=
  by sorry

end cos_555_value_l107_107523


namespace original_price_of_dinosaur_model_l107_107777

-- Define the conditions
theorem original_price_of_dinosaur_model
  (P : ℝ) -- original price of each model
  (kindergarten_models : ℝ := 2)
  (elementary_models : ℝ := 2 * kindergarten_models)
  (total_models : ℝ := kindergarten_models + elementary_models)
  (reduction_percentage : ℝ := 0.05)
  (discounted_price : ℝ := P * (1 - reduction_percentage))
  (total_paid : ℝ := total_models * discounted_price)
  (total_paid_condition : total_paid = 570) :
  P = 100 :=
by
  sorry

end original_price_of_dinosaur_model_l107_107777


namespace profit_at_original_price_l107_107337

theorem profit_at_original_price (x : ℝ) (h : 0.8 * x = 1.2) : x - 1 = 0.5 :=
by
  sorry

end profit_at_original_price_l107_107337


namespace ab_div_c_eq_one_l107_107825

theorem ab_div_c_eq_one (A B C : ℕ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (hne1 : A ≠ B) (hne2 : A ≠ C) (hne3 : B ≠ C) :
  (1 - 1 / (6 + 1 / (6 + 1 / 6)) = 1 / (A + 1 / (B + 1 / 1))) → (A + B) / C = 1 :=
by sorry

end ab_div_c_eq_one_l107_107825


namespace budget_left_equals_16_l107_107231

def initial_budget : ℤ := 200
def expense_shirt : ℤ := 30
def expense_pants : ℤ := 46
def expense_coat : ℤ := 38
def expense_socks : ℤ := 11
def expense_belt : ℤ := 18
def expense_shoes : ℤ := 41

def total_expenses : ℤ := 
  expense_shirt + expense_pants + expense_coat + expense_socks + expense_belt + expense_shoes

def budget_left : ℤ := initial_budget - total_expenses

theorem budget_left_equals_16 : 
  budget_left = 16 := by
  sorry

end budget_left_equals_16_l107_107231


namespace parabola_c_value_l107_107634

theorem parabola_c_value :
  ∃ a b c : ℝ, (∀ y : ℝ, 4 = a * (3 : ℝ)^2 + b * 3 + c ∧ 2 = a * 5^2 + b * 5 + c ∧ c = -1 / 2) :=
by
  sorry

end parabola_c_value_l107_107634


namespace missing_number_geometric_sequence_l107_107113

theorem missing_number_geometric_sequence : 
  ∃ (x : ℤ), (x = 162) ∧ 
  (x = 54 * 3 ∧ 
  486 = x * 3 ∧ 
  ∀ a b : ℤ, (b = 2 * 3) ∧ 
              (a = 2 * 3) ∧ 
              (18 = b * 3) ∧ 
              (54 = 18 * 3) ∧ 
              (54 * 3 = x)) := 
by sorry

end missing_number_geometric_sequence_l107_107113


namespace number_of_polynomials_satisfying_P_neg1_eq_neg12_l107_107928

noncomputable def count_polynomials_satisfying_condition : ℕ := 
  sorry

theorem number_of_polynomials_satisfying_P_neg1_eq_neg12 :
  count_polynomials_satisfying_condition = 455 := 
  sorry

end number_of_polynomials_satisfying_P_neg1_eq_neg12_l107_107928


namespace pure_imaginary_complex_l107_107333

theorem pure_imaginary_complex (m : ℝ) (i : ℂ) (h : i^2 = -1) :
    (∃ (y : ℂ), (2 - m * i) / (1 + i) = y * i) ↔ m = 2 :=
by
  sorry

end pure_imaginary_complex_l107_107333


namespace even_function_condition_iff_l107_107794

theorem even_function_condition_iff (m : ℝ) :
    (∀ x : ℝ, (m * 2^x + 2^(-x)) = (m * 2^(-x) + 2^x)) ↔ (m = 1) :=
by
  sorry

end even_function_condition_iff_l107_107794


namespace board_arithmetic_impossibility_l107_107489

theorem board_arithmetic_impossibility :
  ¬ (∃ (a b : ℕ), a ≡ 0 [MOD 7] ∧ b ≡ 1 [MOD 7] ∧ (a * b + a^3 + b^3) = 2013201420152016) := 
    sorry

end board_arithmetic_impossibility_l107_107489


namespace find_a_l107_107259

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 0 then a ^ x - 1 else 2 * x ^ 2

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ m n : ℝ, f a m ≤ f a n ↔ m ≤ n)
  (h4 : f a a = 5 * a - 2) : a = 2 :=
sorry

end find_a_l107_107259


namespace possible_k_value_l107_107302

theorem possible_k_value (a n k : ℕ) (h1 : n > 1) (h2 : 10^(n-1) ≤ a ∧ a < 10^n)
    (h3 : b = a * (10^n + 1)) (h4 : k = b / a^2) (h5 : b = a * 10 ^n + a) :
  k = 7 := 
sorry

end possible_k_value_l107_107302


namespace tenth_pair_in_twentieth_row_l107_107991

noncomputable def pair_in_row (n k : ℕ) : ℕ × ℕ :=
  if k = 0 ∨ k > n then (0, 0) else (k, n + 1 - k)

theorem tenth_pair_in_twentieth_row : pair_in_row 20 10 = (10, 11) := by
  sorry

end tenth_pair_in_twentieth_row_l107_107991


namespace triangle_inequality_l107_107434

theorem triangle_inequality (a b c : ℝ) (h : a < b + c) : a^2 - b^2 - c^2 - 2*b*c < 0 := by
  sorry

end triangle_inequality_l107_107434


namespace expression_simplifies_to_zero_l107_107367

theorem expression_simplifies_to_zero (x y : ℝ) (h : x = 2024) :
    5 * (x ^ 3 - 3 * x ^ 2 * y - 2 * x * y ^ 2) -
    3 * (x ^ 3 - 5 * x ^ 2 * y + 2 * y ^ 3) +
    2 * (-x ^ 3 + 5 * x * y ^ 2 + 3 * y ^ 3) = 0 :=
by {
    sorry
}

end expression_simplifies_to_zero_l107_107367


namespace product_12_3460_l107_107232

theorem product_12_3460 : 12 * 3460 = 41520 :=
by
  sorry

end product_12_3460_l107_107232


namespace volume_of_rectangular_box_l107_107104

theorem volume_of_rectangular_box 
  (l w h : ℝ)
  (h1 : l * w = 30)
  (h2 : w * h = 20)
  (h3 : l * h = 12) : 
  l * w * h = 60 :=
sorry

end volume_of_rectangular_box_l107_107104


namespace power_of_11_in_expression_l107_107184

-- Define the mathematical context
def prime_factors_count (n : ℕ) (a b c : ℕ) : ℕ :=
  n + a + b

-- Given conditions
def count_factors_of_2 : ℕ := 22
def count_factors_of_7 : ℕ := 5
def total_prime_factors : ℕ := 29

-- Theorem stating that power of 11 in the expression is 2
theorem power_of_11_in_expression : 
  ∃ n : ℕ, prime_factors_count n count_factors_of_2 count_factors_of_7 = total_prime_factors ∧ n = 2 :=
by
  sorry

end power_of_11_in_expression_l107_107184


namespace number_divisible_by_75_l107_107793

def is_two_digit (x : ℕ) := x >= 10 ∧ x < 100

theorem number_divisible_by_75 {a b : ℕ} (h1 : a * b = 35) (h2 : is_two_digit (10 * a + b)) : (10 * a + b) % 75 = 0 :=
sorry

end number_divisible_by_75_l107_107793


namespace geom_seq_general_term_sum_geometric_arithmetic_l107_107525

noncomputable def a_n (n : ℕ) : ℕ := 2^n
def b_n (n : ℕ) : ℕ := 2*n - 1

theorem geom_seq_general_term (a : ℕ → ℕ) (a1 : a 1 = 2)
  (a2 : a 3 = (a 2) + 4) : ∀ n, a n = a_n n :=
by
  sorry

theorem sum_geometric_arithmetic (a b : ℕ → ℕ) 
  (a_def : ∀ n, a n = 2 ^ n) (b_def : ∀ n, b n = 2 * n - 1) : 
  ∀ n, (Finset.range n).sum (λ i => (a (i + 1) + b (i + 1))) = 2^(n+1) + n^2 - 2 :=
by
  sorry

end geom_seq_general_term_sum_geometric_arithmetic_l107_107525


namespace question_d_l107_107092

variable {x a : ℝ}

theorem question_d (h1 : x < a) (h2 : a < 0) : x^3 > a * x ∧ a * x < 0 :=
  sorry

end question_d_l107_107092


namespace calc_154_1836_minus_54_1836_l107_107742

-- Statement of the problem in Lean 4
theorem calc_154_1836_minus_54_1836 : 154 * 1836 - 54 * 1836 = 183600 :=
by
  sorry

end calc_154_1836_minus_54_1836_l107_107742


namespace geometric_sequence_fraction_l107_107061

variable (a_1 : ℝ) (q : ℝ)

theorem geometric_sequence_fraction (h : q = 2) :
  (2 * a_1 + a_1 * q) / (2 * (a_1 * q^2) + a_1 * q^3) = 1 / 4 :=
by sorry

end geometric_sequence_fraction_l107_107061


namespace Ramsey_number_bound_l107_107303

noncomputable def Ramsey_number (k : ℕ) : ℕ := sorry

theorem Ramsey_number_bound (k : ℕ) (h : k ≥ 3) : Ramsey_number k > 2^(k / 2) := sorry

end Ramsey_number_bound_l107_107303


namespace wuyang_math_total_participants_l107_107791

theorem wuyang_math_total_participants :
  ∀ (x : ℕ), 
  95 * (x + 5) = 75 * (x + 3 + 10) → 
  2 * (x + x + 8) + 9 = 125 :=
by
  intro x h
  sorry

end wuyang_math_total_participants_l107_107791


namespace neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0_l107_107203

theorem neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0 :
  ¬ (∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ ∀ x : ℝ, x ≤ 0 → x^2 - x > 0 :=
by
    sorry

end neg_prop_of_exists_x_gt_0_xsq_sub_x_leq_0_l107_107203


namespace tan_alpha_eq_cos_two_alpha_plus_quarter_pi_sin_beta_eq_l107_107461

-- Definitions
variables {α β : ℝ}

-- Condition: 0 < α < π / 2
def valid_alpha (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Condition: sin α = 4 / 5
def sin_alpha (α : ℝ) : Prop := Real.sin α = 4 / 5

-- Condition: 0 < β < π / 2
def valid_beta (β : ℝ) : Prop := 0 < β ∧ β < Real.pi / 2

-- Condition: cos (α + β) = -1 / 2
def cos_alpha_add_beta (α β : ℝ) : Prop := Real.cos (α + β) = - 1 / 2

/-- Proofs begin -/
-- Proof for tan α = 4 / 3 given 0 < α < π / 2 and sin α = 4 / 5
theorem tan_alpha_eq (α : ℝ) (h_valid : valid_alpha α) (h_sin : sin_alpha α) : Real.tan α = 4 / 3 := 
  sorry

-- Proof for cos (2α + π / 4) = -31√2 / 50 given 0 < α < π / 2 and sin α = 4 / 5
theorem cos_two_alpha_plus_quarter_pi (α : ℝ) (h_valid : valid_alpha α) (h_sin : sin_alpha α) : 
  Real.cos (2 * α + Real.pi / 4) = -31 * Real.sqrt 2 / 50 := 
  sorry

-- Proof for sin β = 4 + 3√3 / 10 given 0 < α < π / 2, sin α = 4 / 5, 0 < β < π / 2 and cos (α + β) = -1 / 2
theorem sin_beta_eq (α β : ℝ) (h_validα : valid_alpha α) (h_sinα : sin_alpha α) 
  (h_validβ : valid_beta β) (h_cosαβ : cos_alpha_add_beta α β) : Real.sin β = 4 + 3 * Real.sqrt 3 / 10 := 
  sorry

end tan_alpha_eq_cos_two_alpha_plus_quarter_pi_sin_beta_eq_l107_107461


namespace valid_transformation_b_l107_107149

theorem valid_transformation_b (a b : ℚ) : ((-a - b) / (a + b) = -1) := sorry

end valid_transformation_b_l107_107149


namespace david_trip_distance_l107_107784

theorem david_trip_distance (t : ℝ) (d : ℝ) : 
  (40 * (t + 1) = d) →
  (d - 40 = 60 * (t - 0.75)) →
  d = 130 := 
by
  intro h1 h2
  sorry

end david_trip_distance_l107_107784


namespace roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells_l107_107558

theorem roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells 
  (k n : ℕ) (h_k : k = 4) (h_n : n = 100)
  (shift_rule : ∀ (m : ℕ), m ≤ n → 
    ∃ (chips_moved : ℕ), chips_moved = 1 ∧ chips_moved ≤ m) 
  : ∃ m, m ≤ n ∧ m = 50 := 
by
  sorry

end roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells_l107_107558


namespace every_real_has_cube_root_l107_107537

theorem every_real_has_cube_root : ∀ y : ℝ, ∃ x : ℝ, x^3 = y := 
by
  sorry

end every_real_has_cube_root_l107_107537


namespace consecutive_odd_numbers_first_l107_107018

theorem consecutive_odd_numbers_first :
  ∃ x : ℤ, 11 * x = 3 * (x + 4) + 4 * (x + 2) + 16 ∧ x = 9 :=
by 
  sorry

end consecutive_odd_numbers_first_l107_107018


namespace inequality_proof_l107_107039

-- Define the main theorem to be proven.
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c - a) + b^2 * (a + c - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end inequality_proof_l107_107039


namespace binomial_product_l107_107749

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 6) = 4 * x ^ 2 - 21 * x - 18 := 
sorry

end binomial_product_l107_107749


namespace f_f_five_eq_five_l107_107842

-- Define the function and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Hypotheses
axiom h1 : ∀ x : ℝ, f (x + 2) = -f x
axiom h2 : f 1 = -5

-- Theorem to prove
theorem f_f_five_eq_five : f (f 5) = 5 :=
sorry

end f_f_five_eq_five_l107_107842


namespace exists_infinitely_many_triples_l107_107674

theorem exists_infinitely_many_triples :
  ∀ n : ℕ, ∃ (a b c : ℕ), a^2 + b^2 + c^2 + 2016 = a * b * c :=
sorry

end exists_infinitely_many_triples_l107_107674


namespace solve_for_p_l107_107043

def cubic_eq_has_natural_roots (p : ℝ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  5*(a:ℝ)^3 - 5*(p + 1)*(a:ℝ)^2 + (71*p - 1)*(a:ℝ) + 1 = 66*p ∧
  5*(b:ℝ)^3 - 5*(p + 1)*(b:ℝ)^2 + (71*p - 1)*(b:ℝ) + 1 = 66*p ∧
  5*(c:ℝ)^3 - 5*(p + 1)*(c:ℝ)^2 + (71*p - 1)*(c:ℝ) + 1 = 66*p

theorem solve_for_p : ∀ (p : ℝ), cubic_eq_has_natural_roots p → p = 76 :=
by
  sorry

end solve_for_p_l107_107043


namespace mrs_hilt_total_payment_l107_107875

noncomputable def total_hotdogs : ℕ := 12
noncomputable def cost_first_4 : ℝ := 4 * 0.60
noncomputable def cost_next_5 : ℝ := 5 * 0.75
noncomputable def cost_last_3 : ℝ := 3 * 0.90
noncomputable def total_cost : ℝ := cost_first_4 + cost_next_5 + cost_last_3

theorem mrs_hilt_total_payment : total_cost = 8.85 := by
  -- proof goes here
  sorry

end mrs_hilt_total_payment_l107_107875


namespace range_of_a_l107_107426

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a_seq n = a + n - 1)
  (h2 : ∀ n : ℕ, b n = (1 + a_seq n) / a_seq n)
  (h3 : ∀ n : ℕ, n > 0 → b n ≤ b 5) :
  -4 < a ∧ a < -3 :=
by
  sorry

end range_of_a_l107_107426


namespace minimum_total_trips_l107_107192

theorem minimum_total_trips :
  ∃ (x y : ℕ), (31 * x + 32 * y = 5000) ∧ (x + y = 157) :=
by
  sorry

end minimum_total_trips_l107_107192


namespace simplify_expression_l107_107129

theorem simplify_expression (w : ℝ) :
  3 * w + 4 - 2 * w - 5 + 6 * w + 7 - 3 * w - 9 = 4 * w - 3 :=
by 
  sorry

end simplify_expression_l107_107129


namespace problem_statement_l107_107177

theorem problem_statement (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + 6 = 104 := by
  sorry

end problem_statement_l107_107177


namespace brochures_per_box_l107_107475

theorem brochures_per_box (total_brochures : ℕ) (boxes : ℕ) 
  (htotal : total_brochures = 5000) (hboxes : boxes = 5) : 
  (1000 / 5000 : ℚ) = 1 / 5 := 
by sorry

end brochures_per_box_l107_107475


namespace original_population_l107_107495

theorem original_population (P : ℕ) (h1 : 0.1 * (P : ℝ) + 0.2 * (0.9 * P) = 4500) : P = 6250 :=
sorry

end original_population_l107_107495


namespace find_constants_C_and_A_l107_107666

theorem find_constants_C_and_A :
  ∃ (C A : ℚ), (C * x + 7 - 17)/(x^2 - 9 * x + 20) = A / (x - 4) + 2 / (x - 5) ∧ B = 7 ∧ C = 12/5 ∧ A = 2/5 := sorry

end find_constants_C_and_A_l107_107666


namespace total_bathing_suits_l107_107242

theorem total_bathing_suits 
  (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ)
  (ha : a = 8500) (hb : b = 12750) (hc : c = 5900) (hd : d = 7250) (he : e = 1100) :
  a + b + c + d + e = 35500 :=
by
  sorry

end total_bathing_suits_l107_107242


namespace battery_current_l107_107633

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l107_107633


namespace john_paid_more_l107_107997

-- Define the required variables
def original_price : ℝ := 84.00000000000009
def discount_rate : ℝ := 0.10
def tip_rate : ℝ := 0.15

-- Define John and Jane's payments
def discounted_price : ℝ := original_price * (1 - discount_rate)
def johns_tip : ℝ := tip_rate * original_price
def johns_total_payment : ℝ := original_price + johns_tip
def janes_tip : ℝ := tip_rate * discounted_price
def janes_total_payment : ℝ := discounted_price + janes_tip

-- Calculate the difference
def payment_difference : ℝ := johns_total_payment - janes_total_payment

-- Statement to prove the payment difference equals $9.66
theorem john_paid_more : payment_difference = 9.66 := by
  sorry

end john_paid_more_l107_107997


namespace inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared_l107_107249

theorem inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared 
  (a b c : ℝ)
  (h_sum : a + b + c = 0)
  (d : ℝ) 
  (h_d : d = max (abs a) (max (abs b) (abs c))) : 
  abs ((1 + a) * (1 + b) * (1 + c)) ≥ 1 - d^2 :=
by 
  sorry

end inequality_1_plus_a_1_plus_b_1_plus_c_geq_1_minus_d_squared_l107_107249


namespace bob_walks_more_l107_107574

def street_width : ℝ := 30
def length_side1 : ℝ := 500
def length_side2 : ℝ := 300

def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

def alice_perimeter : ℝ := perimeter (length_side1 + 2 * street_width) (length_side2 + 2 * street_width)
def bob_perimeter : ℝ := perimeter (length_side1 + 4 * street_width) (length_side2 + 4 * street_width)

theorem bob_walks_more :
  bob_perimeter - alice_perimeter = 240 :=
by
  sorry

end bob_walks_more_l107_107574


namespace square_of_1005_l107_107179

theorem square_of_1005 : (1005 : ℕ)^2 = 1010025 := 
  sorry

end square_of_1005_l107_107179


namespace tetrahedron_point_choice_l107_107728

-- Definitions
variables (h s1 s2 : ℝ) -- h, s1, s2 are positive real numbers
variables (A B C : ℝ)  -- A, B, C can be points in space

-- Hypothetical tetrahedron face areas and height
def height_condition (D : ℝ) : Prop := -- D is a point in space
  ∃ (D_height : ℝ), D_height = h

def area_ACD_condition (D : ℝ) : Prop := 
  ∃ (area_ACD : ℝ), area_ACD = s1

def area_BCD_condition (D : ℝ) : Prop := 
  ∃ (area_BCD : ℝ), area_BCD = s2

-- The main theorem
theorem tetrahedron_point_choice : 
  ∃ D, height_condition h D ∧ area_ACD_condition s1 D ∧ area_BCD_condition s2 D :=
sorry

end tetrahedron_point_choice_l107_107728


namespace total_voters_l107_107880

theorem total_voters (x : ℝ)
  (h1 : 0.35 * x + 80 = (0.35 * x + 80) + 0.65 * x - (0.65 * x - 0.45 * (x + 80)))
  (h2 : 0.45 * (x + 80) = 0.65 * x) : 
  x + 80 = 260 := by
  -- We'll provide the proof here
  sorry

end total_voters_l107_107880


namespace sum_of_integers_l107_107927

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : 
  x + y = 32 :=
by
  sorry

end sum_of_integers_l107_107927


namespace solve_abs_eq_l107_107258

theorem solve_abs_eq (x : ℝ) (h : |x + 2| = |x - 3|) : x = 1 / 2 :=
sorry

end solve_abs_eq_l107_107258


namespace triangle_angles_21_equal_triangles_around_square_l107_107256

theorem triangle_angles_21_equal_triangles_around_square
    (theta alpha beta gamma : ℝ)
    (h1 : 4 * theta + 90 = 360)
    (h2 : alpha + beta + 90 = 180)
    (h3 : alpha + beta + gamma = 180)
    (h4 : gamma + 90 = 180)
    : theta = 67.5 ∧ alpha = 67.5 ∧ beta = 22.5 ∧ gamma = 90 :=
by
  sorry

end triangle_angles_21_equal_triangles_around_square_l107_107256


namespace shekar_biology_marks_l107_107051

theorem shekar_biology_marks (M S SS E A n B : ℕ) 
  (hM : M = 76)
  (hS : S = 65)
  (hSS : SS = 82)
  (hE : E = 67)
  (hA : A = 73)
  (hn : n = 5)
  (hA_eq : A = (M + S + SS + E + B) / n) : 
  B = 75 := 
by
  rw [hM, hS, hSS, hE, hn, hA] at hA_eq
  sorry

end shekar_biology_marks_l107_107051


namespace fewest_students_possible_l107_107261

theorem fewest_students_possible (N : ℕ) :
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) ↔ N = 59 :=
by
  sorry

end fewest_students_possible_l107_107261


namespace russia_is_one_third_bigger_l107_107165

theorem russia_is_one_third_bigger (U : ℝ) (Canada Russia : ℝ) 
  (h1 : Canada = 1.5 * U) (h2 : Russia = 2 * U) : 
  (Russia - Canada) / Canada = 1 / 3 :=
by
  sorry

end russia_is_one_third_bigger_l107_107165


namespace problem_solution_l107_107688

noncomputable def f (A B : ℝ) (x : ℝ) : ℝ := A + B / x + x

theorem problem_solution (A B : ℝ) :
  ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 →
  (x * f A B (x + 1 / y) + y * f A B y + y / x = y * f A B (y + 1 / x) + x * f A B x + x / y) :=
by
  sorry

end problem_solution_l107_107688


namespace emma_additional_miles_l107_107162

theorem emma_additional_miles :
  ∀ (initial_distance : ℝ) (initial_speed : ℝ) (additional_speed : ℝ) (desired_avg_speed : ℝ) (total_distance : ℝ) (additional_distance : ℝ),
    initial_distance = 20 →
    initial_speed = 40 →
    additional_speed = 70 →
    desired_avg_speed = 60 →
    total_distance = initial_distance + additional_distance →
    (total_distance / ((initial_distance / initial_speed) + (additional_distance / additional_speed))) = desired_avg_speed →
    additional_distance = 70 :=
by
  intros initial_distance initial_speed additional_speed desired_avg_speed total_distance additional_distance
  intros h1 h2 h3 h4 h5 h6
  sorry

end emma_additional_miles_l107_107162


namespace triangular_25_l107_107368

-- Defining the formula for the n-th triangular number.
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Stating that the 25th triangular number is 325.
theorem triangular_25 : triangular 25 = 325 :=
  by
    -- We don't prove it here, so we simply state it requires a proof.
    sorry

end triangular_25_l107_107368


namespace solve_system_of_equations_l107_107270

theorem solve_system_of_equations : 
  ∃ (x y : ℚ), 4 * x - 3 * y = -2 ∧ 5 * x + 2 * y = 8 ∧ x = 20 / 23 ∧ y = 42 / 23 :=
by
  sorry

end solve_system_of_equations_l107_107270


namespace m_minus_t_value_l107_107107

-- Define the sum of squares of the odd integers from 1 to 215
def sum_squares_odds (n : ℕ) : ℕ := n * (4 * n^2 - 1) / 3

-- Define the sum of squares of the even integers from 2 to 100
def sum_squares_evens (n : ℕ) : ℕ := 2 * n * (n + 1) * (2 * n + 1) / 3

-- Number of odd terms from 1 to 215
def odd_terms_count : ℕ := (215 - 1) / 2 + 1

-- Number of even terms from 2 to 100
def even_terms_count : ℕ := (100 - 2) / 2 + 1

-- Define m and t
def m : ℕ := sum_squares_odds odd_terms_count
def t : ℕ := sum_squares_evens even_terms_count

-- Prove that m - t = 1507880
theorem m_minus_t_value : m - t = 1507880 :=
by
  -- calculations to verify the proof will be here, but are omitted for now
  sorry

end m_minus_t_value_l107_107107


namespace final_surface_area_l107_107171

noncomputable def surface_area (total_cubes remaining_cubes cube_surface removed_internal_surface : ℕ) : ℕ :=
  (remaining_cubes * cube_surface) + (remaining_cubes * removed_internal_surface)

theorem final_surface_area :
  surface_area 64 55 54 6 = 3300 :=
by
  sorry

end final_surface_area_l107_107171


namespace equation_of_line_bisecting_chord_l107_107108

theorem equation_of_line_bisecting_chord
  (P : ℝ × ℝ) 
  (A B : ℝ × ℝ)
  (P_bisects_AB : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (P_on_ellipse : 3 * P.1^2 + 4 * P.2^2 = 24)
  (A_on_ellipse : 3 * A.1^2 + 4 * A.2^2 = 24)
  (B_on_ellipse : 3 * B.1^2 + 4 * B.2^2 = 24) :
  ∃ (a b c : ℝ), a * P.2 + b * P.1 + c = 0 ∧ a = 2 ∧ b = -3 ∧ c = 7 :=
by 
  sorry

end equation_of_line_bisecting_chord_l107_107108


namespace age_difference_l107_107833

def A := 10
def B := 8
def C := B / 2
def total_age (A B C : ℕ) : Prop := A + B + C = 22

theorem age_difference (A B C : ℕ) (hB : B = 8) (hC : B = 2 * C) (h_total : total_age A B C) : A - B = 2 := by
  sorry

end age_difference_l107_107833


namespace Exponent_Equality_l107_107180

theorem Exponent_Equality : 2^8 * 2^32 = 256^5 :=
by
  sorry

end Exponent_Equality_l107_107180


namespace inverse_function_properties_l107_107361

theorem inverse_function_properties {f : ℝ → ℝ} 
  (h_monotonic_decreasing : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 3 → f x2 < f x1)
  (h_range : ∀ y : ℝ, 4 ≤ y ∧ y ≤ 7 ↔ ∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ y = f x)
  (h_inverse_exists : ∃ g : ℝ → ℝ, ∀ x : ℝ, f (g x) = x ∧ g (f x) = x) :
  ∃ g : ℝ → ℝ, (∀ y1 y2 : ℝ, 4 ≤ y1 ∧ y1 < y2 ∧ y2 ≤ 7 → g y2 < g y1) ∧ (∀ y : ℝ, 4 ≤ y ∧ y ≤ 7 → g y ≤ 3) :=
sorry

end inverse_function_properties_l107_107361


namespace survey_blue_percentage_l107_107015

-- Conditions
def red (r : ℕ) := r = 70
def blue (b : ℕ) := b = 80
def green (g : ℕ) := g = 50
def yellow (y : ℕ) := y = 70
def orange (o : ℕ) := o = 30

-- Total responses sum
def total_responses (r b g y o : ℕ) := r + b + g + y + o = 300

-- Percentage of blue respondents
def blue_percentage (b total : ℕ) := (b : ℚ) / total * 100 = 26 + 2/3

-- Theorem statement
theorem survey_blue_percentage (r b g y o : ℕ) (H_red : red r) (H_blue : blue b) (H_green : green g) (H_yellow : yellow y) (H_orange : orange o) (H_total : total_responses r b g y o) : blue_percentage b 300 :=
by {
  sorry
}

end survey_blue_percentage_l107_107015


namespace baskets_delivered_l107_107511

theorem baskets_delivered 
  (peaches_per_basket : ℕ := 25)
  (boxes : ℕ := 8)
  (peaches_per_box : ℕ := 15)
  (peaches_eaten : ℕ := 5)
  (peaches_in_boxes := boxes * peaches_per_box) 
  (total_peaches := peaches_in_boxes + peaches_eaten) : 
  total_peaches / peaches_per_basket = 5 :=
by
  sorry

end baskets_delivered_l107_107511


namespace diameter_other_endpoint_l107_107251

def center : ℝ × ℝ := (1, -2)
def endpoint1 : ℝ × ℝ := (4, 3)
def expected_endpoint2 : ℝ × ℝ := (7, -7)

theorem diameter_other_endpoint (c : ℝ × ℝ) (e1 e2 : ℝ × ℝ) (h₁ : c = center) (h₂ : e1 = endpoint1) : e2 = expected_endpoint2 :=
by
  sorry

end diameter_other_endpoint_l107_107251


namespace initial_bucket_capacity_l107_107376

theorem initial_bucket_capacity (x : ℕ) (h1 : x - 3 = 2) : x = 5 := sorry

end initial_bucket_capacity_l107_107376


namespace no_geometric_progression_l107_107135

theorem no_geometric_progression (r s t : ℕ) (h1 : r < s) (h2 : s < t) :
  ¬ ∃ (b : ℂ), (3^r - 2^r) * b^(s - r) = 3^s - 2^s ∧ (3^s - 2^s) * b^(t - s) = 3^t - 2^t := by
  sorry

end no_geometric_progression_l107_107135


namespace marbles_given_to_joan_l107_107087

def mary_original_marbles : ℝ := 9.0
def mary_marbles_left : ℝ := 6.0

theorem marbles_given_to_joan :
  mary_original_marbles - mary_marbles_left = 3 := 
by
  sorry

end marbles_given_to_joan_l107_107087


namespace jerry_showers_l107_107566

variable (water_allowance : ℕ) (drinking_cooking : ℕ) (water_per_shower : ℕ) (pool_length : ℕ) 
  (pool_width : ℕ) (pool_height : ℕ) (gallons_per_cubic_foot : ℕ)

/-- Jerry can take 15 showers in July given the conditions. -/
theorem jerry_showers :
  water_allowance = 1000 →
  drinking_cooking = 100 →
  water_per_shower = 20 →
  pool_length = 10 →
  pool_width = 10 →
  pool_height = 6 →
  gallons_per_cubic_foot = 1 →
  (water_allowance - (drinking_cooking + (pool_length * pool_width * pool_height) * gallons_per_cubic_foot)) / water_per_shower = 15 :=
by
  intros h_water_allowance h_drinking_cooking h_water_per_shower h_pool_length h_pool_width h_pool_height h_gallons_per_cubic_foot
  sorry

end jerry_showers_l107_107566


namespace solve_for_y_l107_107331

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end solve_for_y_l107_107331


namespace sequence_inequality_l107_107110

theorem sequence_inequality (a : ℕ → ℕ) (strictly_increasing : ∀ n, a n < a (n + 1))
  (sum_condition : ∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) :
  ∀ n, a n ≤ n^2 :=
by sorry

end sequence_inequality_l107_107110


namespace total_books_l107_107419

-- Define the conditions
def books_per_shelf : ℕ := 9
def mystery_shelves : ℕ := 6
def picture_shelves : ℕ := 2

-- The proof problem statement
theorem total_books : 
  (mystery_shelves * books_per_shelf) + 
  (picture_shelves * books_per_shelf) = 72 := 
sorry

end total_books_l107_107419


namespace b_2023_value_l107_107907

noncomputable def seq (b : ℕ → ℝ) : Prop := 
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem b_2023_value (b : ℕ → ℝ) (h1 : seq b) (h2 : b 1 = 2 + Real.sqrt 5) (h3 : b 1984 = 12 + Real.sqrt 5) : 
  b 2023 = -4/3 + 10 * Real.sqrt 5 / 3 :=
sorry

end b_2023_value_l107_107907


namespace arrangement_plans_l107_107036

-- Definition of the problem conditions
def numChineseTeachers : ℕ := 2
def numMathTeachers : ℕ := 4
def numTeachersPerSchool : ℕ := 3

-- Definition of the problem statement
theorem arrangement_plans
  (c : ℕ) (m : ℕ) (s : ℕ)
  (h1 : numChineseTeachers = c)
  (h2 : numMathTeachers = m)
  (h3 : numTeachersPerSchool = s)
  (h4 : ∀ a b : ℕ, a + b = numChineseTeachers → a = 1 ∧ b = 1)
  (h5 : ∀ a b : ℕ, a + b = numMathTeachers → a = 2 ∧ b = 2) :
  (c * (1 / 2 * m * (m - 1) / 2)) = 12 :=
sorry

end arrangement_plans_l107_107036


namespace behavior_on_neg_interval_l107_107865

variable (f : ℝ → ℝ)

-- condition 1: f is an odd function
def odd_function : Prop :=
  ∀ x, f (-x) = -f x

-- condition 2: f is increasing on [3, 7]
def increasing_3_7 : Prop :=
  ∀ x y, (3 ≤ x ∧ x < y ∧ y ≤ 7) → f x < f y

-- condition 3: minimum value of f on [3, 7] is 5
def minimum_3_7 : Prop :=
  ∃ a, 3 ≤ a ∧ a ≤ 7 ∧ f a = 5

-- Use the above conditions to prove the required property on [-7, -3].
theorem behavior_on_neg_interval 
  (h1 : odd_function f) 
  (h2 : increasing_3_7 f) 
  (h3 : minimum_3_7 f) : 
  (∀ x y, (-7 ≤ x ∧ x < y ∧ y ≤ -3) → f x < f y) 
  ∧ ∀ x, -7 ≤ x ∧ x ≤ -3 → f x ≤ -5 :=
sorry

end behavior_on_neg_interval_l107_107865


namespace complement_intersection_l107_107731

open Set

variable (R : Type) [LinearOrderedField R]

def A : Set R := {x | |x| < 1}
def B : Set R := {y | ∃ x, y = 2^x + 1}
def complement_A : Set R := {x | x ≤ -1 ∨ x ≥ 1}

theorem complement_intersection (x : R) : 
  x ∈ (complement_A R) ∩ B R ↔ x > 1 :=
by
  sorry

end complement_intersection_l107_107731


namespace earnings_per_day_correct_l107_107797

-- Given conditions
variable (total_earned : ℕ) (days : ℕ) (earnings_per_day : ℕ)

-- Specify the given values from the conditions
def given_conditions : Prop :=
  total_earned = 165 ∧ days = 5 ∧ total_earned = days * earnings_per_day

-- Statement of the problem: proving the earnings per day
theorem earnings_per_day_correct (h : given_conditions total_earned days earnings_per_day) : 
  earnings_per_day = 33 :=
by
  sorry

end earnings_per_day_correct_l107_107797


namespace find_F_l107_107678

theorem find_F (C F : ℝ) 
  (h1 : C = 7 / 13 * (F - 40))
  (h2 : C = 26) :
  F = 88.2857 :=
by
  sorry

end find_F_l107_107678


namespace hall_area_l107_107579

theorem hall_area (L : ℝ) (B : ℝ) (A : ℝ) (h1 : B = (2/3) * L) (h2 : L = 60) (h3 : A = L * B) : A = 2400 := 
by 
sorry

end hall_area_l107_107579


namespace number_of_rel_prime_to_21_in_range_l107_107146

def is_rel_prime (a b : ℕ) : Prop := gcd a b = 1

noncomputable def count_rel_prime_in_range (a b g : ℕ) : ℕ :=
  ((b - a + 1) : ℕ) - ((b / 3 - (a - 1) / 3) + (b / 7 - (a - 1) / 7) - (b / 21 - (a - 1) / 21))

theorem number_of_rel_prime_to_21_in_range :
  count_rel_prime_in_range 11 99 21 = 51 :=
by 
  sorry

end number_of_rel_prime_to_21_in_range_l107_107146


namespace problem_l107_107347

theorem problem 
  (a : ℝ) 
  (h_a : ∀ x : ℝ, |x + 1| - |2 - x| ≤ a ∧ a ≤ |x + 1| + |2 - x|)
  {m n : ℝ} 
  (h_mn : m > n) 
  (h_n : n > 0)
  (h: a = 3) 
  : 2 * m + 1 / (m^2 - 2 * m * n + n^2) ≥ 2 * n + a :=
by
  sorry

end problem_l107_107347


namespace complex_number_z_l107_107508

theorem complex_number_z (i : ℂ) (z : ℂ) (hi : i * i = -1) (h : 2 * i / z = 1 - i) : z = -1 + i :=
by
  sorry

end complex_number_z_l107_107508


namespace percentage_difference_l107_107748

theorem percentage_difference : 0.70 * 100 - 0.60 * 80 = 22 := 
by
  sorry

end percentage_difference_l107_107748


namespace gcd_35_91_840_l107_107327

theorem gcd_35_91_840 : Nat.gcd (Nat.gcd 35 91) 840 = 7 :=
by
  sorry

end gcd_35_91_840_l107_107327


namespace astronaut_revolutions_l107_107713

theorem astronaut_revolutions (n : ℤ) (R : ℝ) (hn : n > 2) :
    ∃ k : ℤ, k = n - 1 := 
sorry

end astronaut_revolutions_l107_107713


namespace area_of_rectangle_l107_107228

noncomputable def area_proof : ℝ :=
  let a := 294
  let b := 147
  let c := 3
  a + b * Real.sqrt c

theorem area_of_rectangle (ABCD : ℝ × ℝ) (E : ℝ) (F : ℝ) (BE : ℝ) (AB' : ℝ) : 
  BE = 21 ∧ BE = 2 * CF → AB' = 7 → 
  (ABCD.1 * ABCD.2 = 294 + 147 * Real.sqrt 3 ∧ (294 + 147 + 3 = 444)) :=
sorry

end area_of_rectangle_l107_107228


namespace parallel_lines_no_intersection_l107_107122

theorem parallel_lines_no_intersection (k : ℝ) :
  (∀ t s : ℝ, 
    ∃ (a b : ℝ), (a, b) = (1, -3) + t • (2, 5) ∧ (a, b) = (-4, 2) + s • (3, k)) → 
  k = 15 / 2 :=
by
  sorry

end parallel_lines_no_intersection_l107_107122


namespace box_volume_l107_107657

theorem box_volume (x y z : ℕ) 
  (h1 : 2 * x + 2 * y = 26)
  (h2 : x + z = 10)
  (h3 : y + z = 7) :
  x * y * z = 80 :=
by
  sorry

end box_volume_l107_107657


namespace spider_paths_l107_107265

theorem spider_paths : (Nat.choose (7 + 3) 3) = 210 := 
by
  sorry

end spider_paths_l107_107265


namespace ferris_wheel_seats_l107_107360

variable (total_people : ℕ) (people_per_seat : ℕ)

theorem ferris_wheel_seats (h1 : total_people = 18) (h2 : people_per_seat = 9) : total_people / people_per_seat = 2 := by
  sorry

end ferris_wheel_seats_l107_107360


namespace solution_set_correct_l107_107463

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then 2^(-x) - 4 else 2^(x) - 4

theorem solution_set_correct : 
  (∀ x, f x = f |x|) → 
  (∀ x, f x = 2^(-x) - 4 ∨ f x = 2^(x) - 4) → 
  { x | f (x - 2) > 0 } = { x | x < 0 ∨ x > 4 } :=
by
  intro h1 h2
  sorry

end solution_set_correct_l107_107463


namespace inverse_negative_exchange_l107_107879

theorem inverse_negative_exchange (f1 f2 f3 f4 : ℝ → ℝ) (hx1 : ∀ x, f1 x = x - (1/x))
  (hx2 : ∀ x, f2 x = x + (1/x)) (hx3 : ∀ x, f3 x = Real.log x)
  (hx4 : ∀ x, f4 x = if 0 < x ∧ x < 1 then x else if x = 1 then 0 else -(1/x)) :
  (∀ x, f1 (1/x) = -f1 x) ∧ (∀ x, f2 (1/x) = -f2 x) ∧ (∀ x, f3 (1/x) = -f3 x) ∧
  (∀ x, f4 (1/x) = -f4 x) ↔ True := by 
  sorry

end inverse_negative_exchange_l107_107879


namespace max_rectangle_area_l107_107796

-- Lean statement for the proof problem

theorem max_rectangle_area (x : ℝ) (y : ℝ) (h1 : 2 * x + 2 * y = 24) : ∃ A : ℝ, A = 36 :=
by
  -- Definitions for perimeter and area
  let P := 2 * x + 2 * y
  let A := x * y

  -- Conditions
  have h1 : P = 24 := h1

  -- Setting maximum area and completing the proof
  sorry

end max_rectangle_area_l107_107796


namespace cut_out_square_possible_l107_107096

/-- 
Formalization of cutting out eight \(2 \times 1\) rectangles from an \(8 \times 8\) 
checkered board, and checking if it is always possible to cut out a \(2 \times 2\) square
from the remaining part of the board.
-/
theorem cut_out_square_possible :
  ∀ (board : ℕ) (rectangles : ℕ), (board = 64) ∧ (rectangles = 8) → (4 ∣ board) →
  ∃ (remaining_squares : ℕ), (remaining_squares = 48) ∧ 
  (∃ (square_size : ℕ), (square_size = 4) ∧ (remaining_squares ≥ square_size)) :=
by {
  sorry
}

end cut_out_square_possible_l107_107096


namespace johns_balance_at_end_of_first_year_l107_107992

theorem johns_balance_at_end_of_first_year (initial_deposit interest_first_year : ℝ) 
  (h1 : initial_deposit = 5000) 
  (h2 : interest_first_year = 500) :
  initial_deposit + interest_first_year = 5500 :=
by
  rw [h1, h2]
  norm_num

end johns_balance_at_end_of_first_year_l107_107992


namespace cos_angle_of_vectors_l107_107562

variables (a b : EuclideanSpace ℝ (Fin 2))

theorem cos_angle_of_vectors (h1 : ‖a‖ = 2) (h2 : ‖b‖ = 1) (h3 : ‖a - b‖ = 2) :
  (inner a b) / (‖a‖ * ‖b‖) = 1/4 :=
by
  sorry

end cos_angle_of_vectors_l107_107562


namespace value_of_x_plus_y_l107_107010

variable {x y : ℝ}

theorem value_of_x_plus_y (h1 : 1 / x + 1 / y = 1) (h2 : 1 / x - 1 / y = 9) : x + y = -1 / 20 := 
sorry

end value_of_x_plus_y_l107_107010


namespace max_silver_coins_l107_107744

theorem max_silver_coins (n : ℕ) : (n < 150) ∧ (n % 15 = 3) → n = 138 :=
by
  sorry

end max_silver_coins_l107_107744


namespace candies_bought_friday_l107_107193

-- Definitions based on the given conditions
def candies_bought_tuesday : ℕ := 3
def candies_bought_thursday : ℕ := 5
def candies_left (c : ℕ) : Prop := c = 4
def candies_eaten (c : ℕ) : Prop := c = 6

-- Theorem to prove the number of candies bought on Friday
theorem candies_bought_friday (c_left c_eaten : ℕ) (h_left : candies_left c_left) (h_eaten : candies_eaten c_eaten) : 
  (10 - (candies_bought_tuesday + candies_bought_thursday) = 2) :=
  by
    sorry

end candies_bought_friday_l107_107193


namespace three_times_x_not_much_different_from_two_l107_107961

theorem three_times_x_not_much_different_from_two (x : ℝ) :
  3 * x - 2 ≤ -1 := 
sorry

end three_times_x_not_much_different_from_two_l107_107961


namespace theater_ticket_sales_l107_107080

theorem theater_ticket_sales 
  (total_tickets : ℕ) (price_adult_ticket : ℕ) (price_senior_ticket : ℕ) (senior_tickets_sold : ℕ) 
  (Total_tickets_condition : total_tickets = 510)
  (Price_adult_ticket_condition : price_adult_ticket = 21)
  (Price_senior_ticket_condition : price_senior_ticket = 15)
  (Senior_tickets_sold_condition : senior_tickets_sold = 327) : 
  (183 * 21 + 327 * 15 = 8748) :=
by
  sorry

end theater_ticket_sales_l107_107080


namespace profit_percentage_l107_107994

theorem profit_percentage (purchase_price sell_price : ℝ) (h1 : purchase_price = 600) (h2 : sell_price = 624) :
  ((sell_price - purchase_price) / purchase_price) * 100 = 4 := by
  sorry

end profit_percentage_l107_107994


namespace difference_between_largest_and_smallest_l107_107480

def largest_number := 9765310
def smallest_number := 1035679
def expected_difference := 8729631
def digits := [3, 9, 6, 0, 5, 1, 7]

theorem difference_between_largest_and_smallest :
  (largest_number - smallest_number) = expected_difference :=
sorry

end difference_between_largest_and_smallest_l107_107480


namespace fractional_equation_no_solution_l107_107559

theorem fractional_equation_no_solution (x : ℝ) (h1 : x ≠ 3) : (2 - x) / (x - 3) ≠ 1 + 1 / (3 - x) :=
by
  sorry

end fractional_equation_no_solution_l107_107559


namespace discount_amount_l107_107094

/-- Suppose Maria received a 25% discount on DVDs, and she paid $120.
    The discount she received is $40. -/
theorem discount_amount (P : ℝ) (h : 0.75 * P = 120) : P - 120 = 40 := 
sorry

end discount_amount_l107_107094


namespace students_in_dexters_high_school_l107_107229

variables (D S N : ℕ)

theorem students_in_dexters_high_school :
  (D = 4 * S) ∧
  (D + S + N = 3600) ∧
  (N = S - 400) →
  D = 8000 / 3 := 
sorry

end students_in_dexters_high_school_l107_107229


namespace tea_sale_price_correct_l107_107402

noncomputable def cost_price (weight: ℕ) (unit_price: ℕ) : ℕ := weight * unit_price
noncomputable def desired_profit (cost: ℕ) (percentage: ℕ) : ℕ := cost * percentage / 100
noncomputable def sale_price (cost: ℕ) (profit: ℕ) : ℕ := cost + profit
noncomputable def sale_price_per_kg (total_sale_price: ℕ) (weight: ℕ) : ℚ := total_sale_price / weight

theorem tea_sale_price_correct :
  ∀ (weight_A weight_B weight_C weight_D cost_per_kg_A cost_per_kg_B cost_per_kg_C cost_per_kg_D
     profit_percent_A profit_percent_B profit_percent_C profit_percent_D : ℕ),

  weight_A = 80 →
  weight_B = 20 →
  weight_C = 50 →
  weight_D = 30 →
  cost_per_kg_A = 15 →
  cost_per_kg_B = 20 →
  cost_per_kg_C = 25 →
  cost_per_kg_D = 30 →
  profit_percent_A = 25 →
  profit_percent_B = 30 →
  profit_percent_C = 20 →
  profit_percent_D = 15 →
  
  sale_price_per_kg (sale_price (cost_price weight_A cost_per_kg_A) (desired_profit (cost_price weight_A cost_per_kg_A) profit_percent_A)) weight_A = 18.75 →
  sale_price_per_kg (sale_price (cost_price weight_B cost_per_kg_B) (desired_profit (cost_price weight_B cost_per_kg_B) profit_percent_B)) weight_B = 26 →
  sale_price_per_kg (sale_price (cost_price weight_C cost_per_kg_C) (desired_profit (cost_price weight_C cost_per_kg_C) profit_percent_C)) weight_C = 30 →
  sale_price_per_kg (sale_price (cost_price weight_D cost_per_kg_D) (desired_profit (cost_price weight_D cost_per_kg_D) profit_percent_D)) weight_D = 34.5 :=
by
  intros
  sorry

end tea_sale_price_correct_l107_107402


namespace income_expenses_opposite_l107_107262

def income_denotation (income : Int) : Int := income

theorem income_expenses_opposite :
  income_denotation 5 = 5 →
  income_denotation (-5) = -5 :=
by
  intro h
  sorry

end income_expenses_opposite_l107_107262


namespace sara_spent_on_movies_l107_107479

def cost_of_movie_tickets : ℝ := 2 * 10.62
def cost_of_rented_movie : ℝ := 1.59
def cost_of_purchased_movie : ℝ := 13.95

theorem sara_spent_on_movies :
  cost_of_movie_tickets + cost_of_rented_movie + cost_of_purchased_movie = 36.78 := by
  sorry

end sara_spent_on_movies_l107_107479


namespace blue_balls_unchanged_l107_107521

def initial_red_balls : ℕ := 3
def initial_blue_balls : ℕ := 2
def initial_yellow_balls : ℕ := 5
def added_yellow_balls : ℕ := 4

theorem blue_balls_unchanged :
  initial_blue_balls = 2 := by
  sorry

end blue_balls_unchanged_l107_107521


namespace double_given_number_l107_107977

def given_number : ℝ := 1.2 * 10^6

def double_number (x: ℝ) : ℝ := x * 2

theorem double_given_number : double_number given_number = 2.4 * 10^6 :=
by sorry

end double_given_number_l107_107977


namespace discount_percentage_l107_107497

theorem discount_percentage
  (number_of_fandoms : ℕ)
  (tshirts_per_fandom : ℕ)
  (price_per_shirt : ℝ)
  (tax_rate : ℝ)
  (total_paid : ℝ)
  (total_expected_price_with_discount_without_tax : ℝ)
  (total_expected_price_without_discount : ℝ)
  (discount_amount : ℝ)
  (discount_percentage : ℝ) :

  number_of_fandoms = 4 ∧
  tshirts_per_fandom = 5 ∧
  price_per_shirt = 15 ∧
  tax_rate = 10 / 100 ∧
  total_paid = 264 ∧
  total_expected_price_with_discount_without_tax = total_paid / (1 + tax_rate) ∧
  total_expected_price_without_discount = number_of_fandoms * tshirts_per_fandom * price_per_shirt ∧
  discount_amount = total_expected_price_without_discount - total_expected_price_with_discount_without_tax ∧
  discount_percentage = (discount_amount / total_expected_price_without_discount) * 100 ->

  discount_percentage = 20 :=
sorry

end discount_percentage_l107_107497


namespace frustum_surface_area_l107_107878

noncomputable def total_surface_area_of_frustum
  (R r h : ℝ) : ℝ :=
  let s := Real.sqrt (h^2 + (R - r)^2)
  let A_lateral := Real.pi * (R + r) * s
  let A_top := Real.pi * r^2
  let A_bottom := Real.pi * R^2
  A_lateral + A_top + A_bottom

theorem frustum_surface_area :
  total_surface_area_of_frustum 8 2 5 = 10 * Real.pi * Real.sqrt 61 + 68 * Real.pi :=
  sorry

end frustum_surface_area_l107_107878


namespace one_third_of_1206_is_100_5_percent_of_400_l107_107315

theorem one_third_of_1206_is_100_5_percent_of_400 : (1206 / 3) / 400 * 100 = 100.5 := by
  sorry

end one_third_of_1206_is_100_5_percent_of_400_l107_107315


namespace perimeter_is_22_l107_107583

-- Definitions based on the conditions
def side_lengths : List ℕ := [2, 3, 2, 6, 2, 4, 3]

-- Statement of the problem
theorem perimeter_is_22 : side_lengths.sum = 22 := 
  sorry

end perimeter_is_22_l107_107583


namespace escalator_steps_l107_107082

theorem escalator_steps
  (steps_ascending : ℤ)
  (steps_descending : ℤ)
  (ascend_units_time : ℤ)
  (descend_units_time : ℤ)
  (speed_ratio : ℤ)
  (equation : ((steps_ascending : ℚ) / (1 + (ascend_units_time : ℚ))) = ((steps_descending : ℚ) / ((descend_units_time : ℚ) * speed_ratio)) )
  (solution_x : (125 * 0.6 = 75)) : 
  (steps_ascending * (1 + 0.6 : ℚ) = 120) :=
by
  sorry

end escalator_steps_l107_107082


namespace equilateral_triangle_dot_product_l107_107870

noncomputable def dot_product_sum (a b c : ℝ) := 
  a * b + b * c + c * a

theorem equilateral_triangle_dot_product 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A = 1)
  (h2 : B = 1)
  (h3 : C = 1)
  (h4 : a = 1)
  (h5 : b = 1)
  (h6 : c = 1) :
  dot_product_sum a b c = 1 / 2 :=
by 
  sorry

end equilateral_triangle_dot_product_l107_107870


namespace intersection_point_exists_l107_107812

def line_l (x y : ℝ) : Prop := 2 * x + y = 10
def line_l_prime (x y : ℝ) : Prop := x - 2 * y + 10 = 0
def passes_through (x y : ℝ) (p : ℝ × ℝ) : Prop := p.2 = y ∧ 2 * p.1 - 10 = x

theorem intersection_point_exists :
  ∃ p : ℝ × ℝ, line_l p.1 p.2 ∧ line_l_prime p.1 p.2 ∧ passes_through p.1 p.2 (-10, 0) :=
sorry

end intersection_point_exists_l107_107812


namespace max_value_f_on_0_4_l107_107116

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_f_on_0_4 : ∃ (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (4 : ℝ)), ∀ (y : ℝ), y ∈ Set.Icc (0 : ℝ) (4 : ℝ) → f y ≤ f x ∧ f x = 1 / Real.exp 1 :=
by
  sorry

end max_value_f_on_0_4_l107_107116


namespace max_sum_of_diagonals_l107_107160

theorem max_sum_of_diagonals (a b : ℝ) (h_side : a^2 + b^2 = 25) (h_bounds1 : 2 * a ≤ 6) (h_bounds2 : 2 * b ≥ 6) : 2 * (a + b) = 14 :=
sorry

end max_sum_of_diagonals_l107_107160


namespace circle_radius_zero_l107_107013

-- Define the given circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 4 * x + y^2 - 6 * y + 13 = 0

-- The proof problem statement
theorem circle_radius_zero : ∀ (x y : ℝ), circle_eq x y → 0 = 0 :=
by
  sorry

end circle_radius_zero_l107_107013


namespace no_non_degenerate_triangle_l107_107071

theorem no_non_degenerate_triangle 
  (a b c : ℕ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ c) 
  (h3 : a ≠ c) 
  (h4 : Nat.gcd a (Nat.gcd b c) = 1) 
  (h5 : a ∣ (b - c) * (b - c)) 
  (h6 : b ∣ (a - c) * (a - c)) 
  (h7 : c ∣ (a - b) * (a - b)) : 
  ¬ (a < b + c ∧ b < a + c ∧ c < a + b) := 
sorry

end no_non_degenerate_triangle_l107_107071


namespace sum_of_drawn_vegetable_oil_and_fruits_vegetables_l107_107729

-- Definitions based on conditions
def varieties_of_grains : ℕ := 40
def varieties_of_vegetable_oil : ℕ := 10
def varieties_of_animal_products : ℕ := 30
def varieties_of_fruits_vegetables : ℕ := 20
def total_sample_size : ℕ := 20

def sampling_fraction : ℚ := total_sample_size / (varieties_of_grains + varieties_of_vegetable_oil + varieties_of_animal_products + varieties_of_fruits_vegetables)

def expected_drawn_vegetable_oil : ℚ := varieties_of_vegetable_oil * sampling_fraction
def expected_drawn_fruits_vegetables : ℚ := varieties_of_fruits_vegetables * sampling_fraction

-- The theorem to be proved
theorem sum_of_drawn_vegetable_oil_and_fruits_vegetables : 
  expected_drawn_vegetable_oil + expected_drawn_fruits_vegetables = 6 := 
by 
  -- Placeholder for proof
  sorry

end sum_of_drawn_vegetable_oil_and_fruits_vegetables_l107_107729


namespace molecular_weight_C7H6O2_l107_107274

noncomputable def molecular_weight_one_mole (w_9moles : ℕ) (m_9moles : ℕ) : ℕ :=
  m_9moles / w_9moles

theorem molecular_weight_C7H6O2 :
  molecular_weight_one_mole 9 1098 = 122 := by
  sorry

end molecular_weight_C7H6O2_l107_107274


namespace connections_in_computer_lab_l107_107813

theorem connections_in_computer_lab (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 := by
  sorry

end connections_in_computer_lab_l107_107813


namespace largest_A_divisible_by_8_l107_107464

theorem largest_A_divisible_by_8 (A B C : ℕ) (h1 : A = 8 * B + C) (h2 : B = C) (h3 : C < 8) : A ≤ 9 * 7 :=
by sorry

end largest_A_divisible_by_8_l107_107464


namespace abs_neg_sqrt_six_l107_107933

noncomputable def abs_val (x : ℝ) : ℝ :=
  if x < 0 then -x else x

theorem abs_neg_sqrt_six : abs_val (- Real.sqrt 6) = Real.sqrt 6 := by
  -- Proof goes here
  sorry

end abs_neg_sqrt_six_l107_107933


namespace carla_paints_120_square_feet_l107_107914

def totalWork : ℕ := 360
def ratioAlex : ℕ := 3
def ratioBen : ℕ := 5
def ratioCarla : ℕ := 4
def ratioTotal : ℕ := ratioAlex + ratioBen + ratioCarla
def workPerPart : ℕ := totalWork / ratioTotal
def carlasWork : ℕ := ratioCarla * workPerPart

theorem carla_paints_120_square_feet : carlasWork = 120 := by
  sorry

end carla_paints_120_square_feet_l107_107914


namespace sum_integers_neg40_to_60_l107_107014

theorem sum_integers_neg40_to_60 : (Finset.range (60 + 41)).sum (fun i => i - 40) = 1010 := by
  sorry

end sum_integers_neg40_to_60_l107_107014


namespace vasechkin_result_l107_107856

theorem vasechkin_result (x : ℕ) (h : (x / 2 * 7) - 1001 = 7) : (x / 8) ^ 2 - 1001 = 295 :=
by
  sorry

end vasechkin_result_l107_107856


namespace range_of_m_l107_107235

theorem range_of_m (m : ℝ) (x : ℝ) : (∀ x, (1 - m) * x = 2 - 3 * x → x > 0) ↔ m < 4 :=
by
  sorry

end range_of_m_l107_107235


namespace side_length_of_S2_is_1001_l107_107398

-- Definitions and Conditions
variables (R1 R2 : Type) (S1 S2 S3 : Type)
variables (r s : ℤ)
variables (h_total_width : 2 * r + 3 * s = 4422)
variables (h_total_height : 2 * r + s = 2420)

theorem side_length_of_S2_is_1001 (R1 R2 S1 S2 S3 : Type) (r s : ℤ)
  (h_total_width : 2 * r + 3 * s = 4422)
  (h_total_height : 2 * r + s = 2420) : s = 1001 :=
by
  sorry -- proof to be provided

end side_length_of_S2_is_1001_l107_107398


namespace find_pq_l107_107694

-- Define the constants function for the given equation and form
noncomputable def quadratic_eq (p q r : ℤ) : (ℤ × ℤ × ℤ) :=
(2*p*q, p^2 + 2*p*q + q^2 + r, q*q + r)

-- Define the theorem we want to prove
theorem find_pq (p q r: ℤ) (h : quadratic_eq 2 q r = (8, -24, -56)) : pq = -12 :=
by sorry

end find_pq_l107_107694


namespace solution_set_a_neg5_solution_set_general_l107_107340

theorem solution_set_a_neg5 (x : ℝ) : (-5 * x^2 + 3 * x + 2 > 0) ↔ (-2/5 < x ∧ x < 1) := 
sorry

theorem solution_set_general (a x : ℝ) : 
  (ax^2 + (a + 3) * x + 3 > 0) ↔
  ((0 < a ∧ a < 3 ∧ (x < -1 ∨ x > -3/a)) ∨ 
   (a = 3 ∧ x ≠ -1) ∨ 
   (a > 3 ∧ (x < -1 ∨ x > -3/a)) ∨ 
   (a = 0 ∧ x > -1) ∨ 
   (a < 0 ∧ -1 < x ∧ x < -3/a)) := 
sorry

end solution_set_a_neg5_solution_set_general_l107_107340


namespace multiple_of_other_number_l107_107745

theorem multiple_of_other_number (S L k : ℤ) (h₁ : S = 18) (h₂ : L = k * S - 3) (h₃ : S + L = 51) : k = 2 :=
by
  sorry

end multiple_of_other_number_l107_107745


namespace largest_side_of_triangle_l107_107983

theorem largest_side_of_triangle (x y Δ c : ℕ)
  (h1 : (x + 2 * Δ / x = y + 2 * Δ / y))
  (h2 : x = 60)
  (h3 : y = 63) :
  c = 87 :=
sorry

end largest_side_of_triangle_l107_107983


namespace three_digit_numbers_distinct_base_l107_107534

theorem three_digit_numbers_distinct_base (b : ℕ) (h : (b - 1) ^ 2 * (b - 2) = 250) : b = 8 :=
sorry

end three_digit_numbers_distinct_base_l107_107534


namespace find_minimum_r_l107_107430

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem find_minimum_r (r : ℕ) (h_pos : r > 0) (h_perfect : is_perfect_square (4^3 + 4^r + 4^4)) : r = 4 :=
sorry

end find_minimum_r_l107_107430


namespace maximum_matches_l107_107287

theorem maximum_matches (A B C : ℕ) (h1 : A > B) (h2 : B > C) 
    (h3 : A ≥ B + 10) (h4 : B ≥ C + 10) (h5 : B + C > A) : 
    A + B + C - 1 ≤ 62 :=
sorry

end maximum_matches_l107_107287


namespace range_of_m_l107_107133

theorem range_of_m (m : ℝ) (h : (m^2 + m) ^ (3 / 5) ≤ (3 - m) ^ (3 / 5)) : 
  -3 ≤ m ∧ m ≤ 1 :=
by { sorry }

end range_of_m_l107_107133


namespace closest_point_on_parabola_to_line_is_l107_107669

-- Definitions of the parabola and the line
def parabola (x : ℝ) : ℝ := 4 * x^2
def line (x : ℝ) : ℝ := 4 * x - 5

-- Prove that the point on the parabola that is closest to the line is (1/2, 1)
theorem closest_point_on_parabola_to_line_is (x y : ℝ) :
  parabola x = y ∧ (∀ (x' y' : ℝ), parabola x' = y' -> (line x - y)^2 >= (line x' - y')^2) ->
  (x, y) = (1/2, 1) :=
by
  sorry

end closest_point_on_parabola_to_line_is_l107_107669


namespace chocolate_distribution_l107_107281

theorem chocolate_distribution :
  let total_chocolate := 60 / 7
  let piles := 5
  let eaten_piles := 1
  let friends := 2
  let one_pile := total_chocolate / piles
  let remaining_chocolate := total_chocolate - eaten_piles * one_pile
  let chocolate_per_friend := remaining_chocolate / friends
  chocolate_per_friend = 24 / 7 :=
by
  sorry

end chocolate_distribution_l107_107281


namespace book_club_boys_count_l107_107254

theorem book_club_boys_count (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : B + (1 / 3 : ℝ) * G = 18) :
  B = 12 :=
by
  have h3 : 3 • B + G = 54 := sorry
  have h4 : 3 • B + G - (B + G) = 54 - 30 := sorry
  have h5 : 2 • B = 24 := sorry
  have h6 : B = 12 := sorry
  exact h6

end book_club_boys_count_l107_107254


namespace recreation_percentage_l107_107710

theorem recreation_percentage (W : ℝ) (hW : W > 0) :
  (0.40 * W) / (0.15 * W) * 100 = 267 := by
  sorry

end recreation_percentage_l107_107710


namespace probability_more_heads_than_tails_l107_107572

-- Define the total number of outcomes when flipping 10 coins
def total_outcomes : ℕ := 2 ^ 10

-- Define the number of ways to get exactly 5 heads out of 10 flips (combination)
def combinations_5_heads : ℕ := Nat.choose 10 5

-- Define the probability of getting exactly 5 heads out of 10 flips
def probability_5_heads := (combinations_5_heads : ℚ) / total_outcomes

-- Define the probability of getting more heads than tails (x)
def probability_more_heads := (1 - probability_5_heads) / 2

-- Theorem stating the probability of getting more heads than tails
theorem probability_more_heads_than_tails : probability_more_heads = 193 / 512 :=
by
  -- Proof skipped using sorry
  sorry

end probability_more_heads_than_tails_l107_107572


namespace problem1_problem2_l107_107986

-- Problem 1
theorem problem1 : 2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / Real.sqrt 2 = (3 * Real.sqrt 2) / 2 :=
by sorry

-- Problem 2
theorem problem2 : (Real.sqrt 3 - Real.sqrt 2)^2 + (Real.sqrt 8 - Real.sqrt 3) * (2 * Real.sqrt 2 + Real.sqrt 3) = 10 - 2 * Real.sqrt 6 :=
by sorry

end problem1_problem2_l107_107986


namespace time_after_1750_minutes_is_1_10_pm_l107_107448

def add_minutes_to_time (hours : Nat) (minutes : Nat) : Nat × Nat :=
  let total_minutes := hours * 60 + minutes
  (total_minutes / 60, total_minutes % 60)

def time_after_1750_minutes (current_hour : Nat) (current_minute : Nat) : Nat × Nat :=
  let (new_hour, new_minute) := add_minutes_to_time current_hour current_minute
  let final_hour := (new_hour + 1750 / 60) % 24
  let final_minute := (new_minute + 1750 % 60) % 60
  (final_hour, final_minute)

theorem time_after_1750_minutes_is_1_10_pm : 
  time_after_1750_minutes 8 0 = (13, 10) :=
by {
  sorry
}

end time_after_1750_minutes_is_1_10_pm_l107_107448


namespace tom_and_eva_children_count_l107_107554

theorem tom_and_eva_children_count (karen_donald_children : ℕ)
  (total_legs_in_pool : ℕ) (people_not_in_pool : ℕ) 
  (total_legs_each_person : ℕ) (karen_donald : ℕ) (tom_eva : ℕ) 
  (total_people_in_pool : ℕ) (total_people : ℕ) :
  karen_donald_children = 6 ∧ total_legs_in_pool = 16 ∧ people_not_in_pool = 6 ∧ total_legs_each_person = 2 ∧
  karen_donald = 2 ∧ tom_eva = 2 ∧ total_people_in_pool = total_legs_in_pool / total_legs_each_person ∧ 
  total_people = total_people_in_pool + people_not_in_pool ∧ 
  total_people - (karen_donald + karen_donald_children + tom_eva) = 4 :=
by
  intros
  sorry

end tom_and_eva_children_count_l107_107554


namespace percentage_gain_second_week_l107_107152

variables (initial_investment final_value after_first_week_value gain_percentage first_week_gain second_week_gain second_week_gain_percentage : ℝ)

def pima_investment (initial_investment: ℝ) (first_week_gain_percentage: ℝ) : ℝ :=
  initial_investment * (1 + first_week_gain_percentage)

def second_week_investment (initial_investment first_week_gain_percentage second_week_gain_percentage : ℝ) : ℝ :=
  initial_investment * (1 + first_week_gain_percentage) * (1 + second_week_gain_percentage)

theorem percentage_gain_second_week
  (initial_investment : ℝ)
  (first_week_gain_percentage : ℝ)
  (final_value : ℝ)
  (h1: initial_investment = 400)
  (h2: first_week_gain_percentage = 0.25)
  (h3: final_value = 750) :
  second_week_gain_percentage = 0.5 :=
by
  let after_first_week_value := pima_investment initial_investment first_week_gain_percentage
  let second_week_gain := final_value - after_first_week_value
  let second_week_gain_percentage := second_week_gain / after_first_week_value * 100
  sorry

end percentage_gain_second_week_l107_107152


namespace total_distance_traveled_l107_107292

noncomputable def travel_distance : ℝ :=
  1280 * Real.sqrt 2 + 640 * Real.sqrt (2 + Real.sqrt 2) + 640

theorem total_distance_traveled :
  let n := 8
  let r := 40
  let theta := 2 * Real.pi / n
  let d_2arcs := 2 * r * Real.sin (theta)
  let d_3arcs := r * (2 + Real.sqrt (2))
  let d_4arcs := 2 * r
  (8 * (4 * d_2arcs + 2 * d_3arcs + d_4arcs)) = travel_distance := by
  sorry

end total_distance_traveled_l107_107292


namespace crazy_silly_school_movie_count_l107_107100

theorem crazy_silly_school_movie_count
  (books : ℕ) (read_books : ℕ) (watched_movies : ℕ) (diff_books_movies : ℕ)
  (total_books : books = 8) 
  (read_movie_count : watched_movies = 19)
  (read_book_count : read_books = 16)
  (book_movie_diff : watched_movies = read_books + diff_books_movies)
  (diff_value : diff_books_movies = 3) :
  ∃ M, M ≥ 19 :=
by
  sorry

end crazy_silly_school_movie_count_l107_107100


namespace least_number_of_cookies_l107_107944

theorem least_number_of_cookies (c : ℕ) :
  (c % 6 = 5) ∧ (c % 8 = 7) ∧ (c % 9 = 6) → c = 23 :=
by
  sorry

end least_number_of_cookies_l107_107944


namespace arithmetic_sequence_sum_l107_107063

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the conditions
def a_5 := a 5
def a_6 := a 6
def a_7 := a 7

axiom cond1 : a_5 = 11
axiom cond2 : a_6 = 17
axiom cond3 : a_7 = 23

noncomputable def sum_first_four_terms : ℤ :=
  a 1 + a 2 + a 3 + a 4

theorem arithmetic_sequence_sum :
  a_5 = 11 → a_6 = 17 → a_7 = 23 → sum_first_four_terms a = -16 :=
by
  intros h5 h6 h7
  sorry

end arithmetic_sequence_sum_l107_107063


namespace number_of_lines_dist_l107_107472

theorem number_of_lines_dist {A B : ℝ × ℝ} (hA : A = (3, 0)) (hB : B = (0, 4)) : 
  ∃ n : ℕ, n = 3 ∧
  ∀ l : ℝ → ℝ → Prop, 
  (∀ p : ℝ × ℝ, l p.1 p.2 → p ≠ A → dist A p = 2) ∧ 
  (∀ p : ℝ × ℝ, l p.1 p.2 → p ≠ B → dist B p = 3) → n = 3 := 
by sorry

end number_of_lines_dist_l107_107472


namespace length_MN_proof_l107_107561

-- Declare a noncomputable section to avoid computational requirements
noncomputable section

-- Define the quadrilateral ABCD with given sides
structure Quadrilateral :=
  (BC AD AB CD : ℕ)
  (BC_AD_parallel : Prop)

-- Define a theorem to calculate the length MN
theorem length_MN_proof (ABCD : Quadrilateral) 
  (M N : ℝ) (BisectorsIntersect_M : Prop) (BisectorsIntersect_N : Prop) : 
  ABCD.BC = 26 → ABCD.AD = 5 → ABCD.AB = 10 → ABCD.CD = 17 → 
  (MN = 2 ↔ (BC + AD - AB - CD) / 2 = 2) :=
by
  sorry

end length_MN_proof_l107_107561


namespace factorize_expression_l107_107044

theorem factorize_expression (a : ℝ) : a^2 + 5 * a = a * (a + 5) :=
sorry

end factorize_expression_l107_107044


namespace figure_50_squares_l107_107412

open Nat

noncomputable def g (n : ℕ) : ℕ := 2 * n ^ 2 + 5 * n + 2

theorem figure_50_squares : g 50 = 5252 :=
by
  sorry

end figure_50_squares_l107_107412


namespace ratio_female_to_total_l107_107379

theorem ratio_female_to_total:
  ∃ (F : ℕ), (6 + 7 * F - 9 = (6 + 7 * F) - 9) ∧ 
             (7 * F - 9 = 67 / 100 * ((6 + 7 * F) - 9)) → 
             F = 3 ∧ 6 = 6 → 
             1 / F = 2 / 6 :=
by sorry

end ratio_female_to_total_l107_107379


namespace sin_2017pi_over_6_l107_107222

theorem sin_2017pi_over_6 : Real.sin (2017 * Real.pi / 6) = 1 / 2 := 
by 
  -- Proof to be filled in later
  sorry

end sin_2017pi_over_6_l107_107222


namespace find_k_l107_107577

-- Define the lines l1 and l2
def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x - y - 2 = 0

-- Define the fact that the quadrilateral formed by l1, l2, and the positive halves of the axes
-- has a circumscribed circle.
def has_circumscribed_circle (k : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), line1 x1 y1 ∧ line2 k x2 y2 ∧
  x1 > 0 ∧ y1 > 0 ∧ x2 > 0 ∧ y2 > 0 ∧
  (x1 - x2 = 0 ∨ y1 - y2 = 0) ∧
  (x1 = 0 ∨ y1 = 0 ∨ x2 = 0 ∨ y2 = 0)

-- The statement we need to prove
theorem find_k : ∀ k : ℝ, has_circumscribed_circle k → k = 3 := by
  sorry

end find_k_l107_107577


namespace determine_x_l107_107721

theorem determine_x
  (w : ℤ) (z : ℤ) (y : ℤ) (x : ℤ)
  (h₁ : w = 90)
  (h₂ : z = w + 25)
  (h₃ : y = z + 12)
  (h₄ : x = y + 7) : x = 134 :=
by
  sorry

end determine_x_l107_107721


namespace determine_set_of_integers_for_ratio_l107_107635

def arithmetic_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n / T n = (31 * n + 101) / (n + 3)

def ratio_is_integer (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, a n / b n = k

theorem determine_set_of_integers_for_ratio (a b : ℕ → ℕ) (S T : ℕ → ℕ) :
  arithmetic_sequences a b S T →
  {n : ℕ | ratio_is_integer a b n} = {1, 3} :=
sorry

end determine_set_of_integers_for_ratio_l107_107635


namespace train_length_l107_107643

theorem train_length (L V : ℝ) 
  (h1 : L = V * 110) 
  (h2 : L + 700 = V * 180) : 
  L = 1100 :=
by
  sorry

end train_length_l107_107643


namespace percentage_increase_of_x_l107_107030

theorem percentage_increase_of_x (C x y : ℝ) (P : ℝ) (h1 : x * y = C) (h2 : (x * (1 + P / 100)) * (y * (5 / 6)) = C) :
  P = 20 :=
by
  sorry

end percentage_increase_of_x_l107_107030


namespace plane_through_point_contains_line_l107_107644

-- Definitions from conditions
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

def passes_through (p : Point) (plane : Point → Prop) : Prop :=
  plane p

def contains_line (line : ℝ → Point) (plane : Point → Prop) : Prop :=
  ∀ t, plane (line t)

def line_eq (t : ℝ) : Point :=
  ⟨4 * t + 2, -6 * t - 3, 2 * t + 4⟩

def plane_eq (A B C D : ℝ) (p : Point) : Prop :=
  A * p.x + B * p.y + C * p.z + D = 0

theorem plane_through_point_contains_line :
  ∃ (A B C D : ℝ), 1 < A ∧ gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1 ∧
  passes_through ⟨1, 2, -3⟩ (plane_eq A B C D) ∧
  contains_line line_eq (plane_eq A B C D) ∧ 
  (∃ (k : ℝ), 3 * k = A ∧ k = 1 / 3 ∧ B = k * 1 ∧ C = k * (-3) ∧ D = k * 2) :=
sorry

end plane_through_point_contains_line_l107_107644


namespace simplify_expression_l107_107921

theorem simplify_expression (x : ℝ) : 
  (x^2 + 2 * x + 3) / 4 + (3 * x - 5) / 6 = (3 * x^2 + 12 * x - 1) / 12 := 
by
  sorry

end simplify_expression_l107_107921


namespace no_real_roots_of_quadratic_l107_107488

theorem no_real_roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = 1 ∧ c = 1) :
  (b^2 - 4 * a * c < 0) → ¬∃ x : ℝ, a * x^2 + b * x + c = 0 := by
  sorry

end no_real_roots_of_quadratic_l107_107488


namespace required_folders_l107_107707

def pencil_cost : ℝ := 0.5
def folder_cost : ℝ := 0.9
def pencil_count : ℕ := 24
def total_cost : ℝ := 30

theorem required_folders : ∃ (folders : ℕ), folders = 20 ∧ 
  (pencil_count * pencil_cost + folders * folder_cost = total_cost) :=
sorry

end required_folders_l107_107707


namespace negation_of_all_exp_monotonic_l107_107897

theorem negation_of_all_exp_monotonic :
  ¬ (∀ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f x < f y) → (∃ g : ℝ → ℝ, ∃ x y : ℝ, x < y ∧ g x ≥ g y)) :=
sorry

end negation_of_all_exp_monotonic_l107_107897


namespace Larry_wins_probability_l107_107476

noncomputable def probability_Larry_wins (p_larry: ℚ) (p_paul: ℚ): ℚ :=
  let q_larry := 1 - p_larry
  let q_paul := 1 - p_paul
  p_larry / (1 - q_larry * q_paul)

theorem Larry_wins_probability:
  probability_Larry_wins (1/3 : ℚ) (1/2 : ℚ) = (2/5 : ℚ) :=
by {
  sorry
}

end Larry_wins_probability_l107_107476


namespace total_strings_needed_l107_107195

def basses := 3
def strings_per_bass := 4
def guitars := 2 * basses
def strings_per_guitar := 6
def eight_string_guitars := guitars - 3
def strings_per_eight_string_guitar := 8

theorem total_strings_needed :
  (basses * strings_per_bass) + (guitars * strings_per_guitar) + (eight_string_guitars * strings_per_eight_string_guitar) = 72 := by
  sorry

end total_strings_needed_l107_107195


namespace kids_tubing_and_rafting_l107_107022

theorem kids_tubing_and_rafting 
  (total_kids : ℕ) 
  (one_fourth_tubing : ℕ)
  (half_rafting : ℕ)
  (h1 : total_kids = 40)
  (h2 : one_fourth_tubing = total_kids / 4)
  (h3 : half_rafting = one_fourth_tubing / 2) :
  half_rafting = 5 :=
by
  sorry

end kids_tubing_and_rafting_l107_107022


namespace base4_last_digit_390_l107_107677

theorem base4_last_digit_390 : 
  (Nat.digits 4 390).head! = 2 := sorry

end base4_last_digit_390_l107_107677


namespace correct_calculation_only_A_l107_107586

-- Definitions of the expressions
def exprA (a : ℝ) : Prop := 3 * a + 2 * a = 5 * a
def exprB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def exprC (a : ℝ) : Prop := 3 * a * 2 * a = 6 * a
def exprD (a : ℝ) : Prop := 3 * a / (2 * a) = (3 / 2) * a

-- The theorem stating that only exprA is correct
theorem correct_calculation_only_A (a : ℝ) :
  exprA a ∧ ¬exprB a ∧ ¬exprC a ∧ ¬exprD a :=
by
  sorry

end correct_calculation_only_A_l107_107586


namespace cos_pi_minus_alpha_correct_l107_107213

noncomputable def cos_pi_minus_alpha (α : ℝ) (P : ℝ × ℝ) : ℝ :=
  let x := P.1
  let y := P.2
  let h := Real.sqrt (x^2 + y^2)
  let cos_alpha := x / h
  let cos_pi_minus_alpha := -cos_alpha
  cos_pi_minus_alpha

theorem cos_pi_minus_alpha_correct :
  cos_pi_minus_alpha α (-1, 2) = Real.sqrt 5 / 5 :=
by
  sorry

end cos_pi_minus_alpha_correct_l107_107213


namespace school_count_l107_107006

theorem school_count (n : ℕ) (h1 : 2 * n - 1 = 69) (h2 : n < 76) (h3 : n > 29) : (2 * n - 1) / 3 = 23 :=
by
  sorry

end school_count_l107_107006


namespace neg_p_l107_107800

open Set

-- Definitions of sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

def A : Set ℤ := {x | is_odd x}
def B : Set ℤ := {x | is_even x}

-- Proposition p
def p : Prop := ∀ x ∈ A, 2 * x ∈ B

-- Negation of the proposition p
theorem neg_p : ¬p ↔ ∃ x ∈ A, ¬(2 * x ∈ B) := sorry

end neg_p_l107_107800


namespace remainder_three_n_l107_107598

theorem remainder_three_n (n : ℤ) (h : n % 7 = 1) : (3 * n) % 7 = 3 :=
by
  sorry

end remainder_three_n_l107_107598


namespace linear_correlation_test_l107_107700

theorem linear_correlation_test (n1 n2 n3 n4 : ℕ) (r1 r2 r3 r4 : ℝ) :
  n1 = 10 ∧ r1 = 0.9533 →
  n2 = 15 ∧ r2 = 0.3012 →
  n3 = 17 ∧ r3 = 0.9991 →
  n4 = 3  ∧ r4 = 0.9950 →
  abs r1 > abs r2 ∧ abs r3 > abs r4 →
  (abs r1 > abs r2 → abs r1 > abs r4) →
  (abs r3 > abs r2 → abs r3 > abs r4) →
  abs r1 ≠ abs r2 →
  abs r3 ≠ abs r4 →
  true := 
sorry

end linear_correlation_test_l107_107700


namespace transportation_cost_l107_107839

-- Definitions for the conditions
def number_of_original_bags : ℕ := 80
def weight_of_original_bag : ℕ := 50
def total_cost_original : ℕ := 6000

def scale_factor_bags : ℕ := 3
def scale_factor_weight : ℚ := 3 / 5

-- Derived quantities
def number_of_new_bags : ℕ := scale_factor_bags * number_of_original_bags
def weight_of_new_bag : ℚ := scale_factor_weight * weight_of_original_bag
def cost_per_original_bag : ℚ := total_cost_original / number_of_original_bags
def cost_per_new_bag : ℚ := cost_per_original_bag * (weight_of_new_bag / weight_of_original_bag)

-- Final cost calculation
def total_cost_new : ℚ := number_of_new_bags * cost_per_new_bag

-- The statement that needs to be proved
theorem transportation_cost : total_cost_new = 10800 := sorry

end transportation_cost_l107_107839


namespace area_of_midpoint_quadrilateral_l107_107885

theorem area_of_midpoint_quadrilateral (length width : ℝ) (h_length : length = 15) (h_width : width = 8) :
  let A := (0, width / 2)
  let B := (length / 2, 0)
  let C := (length, width / 2)
  let D := (length / 2, width)
  let mid_quad_area := (length / 2) * (width / 2)
  mid_quad_area = 30 :=
by
  simp [h_length, h_width]
  sorry

end area_of_midpoint_quadrilateral_l107_107885


namespace complex_number_equality_l107_107640

open Complex

theorem complex_number_equality (u v : ℂ) 
  (h1 : 3 * abs (u + 1) * abs (v + 1) ≥ abs (u * v + 5 * u + 5 * v + 1))
  (h2 : abs (u + v) = abs (u * v + 1)) : 
  u = 1 ∨ v = 1 :=
sorry

end complex_number_equality_l107_107640


namespace abs_neg_two_l107_107359

def absolute_value (x : Int) : Int :=
  if x >= 0 then x else -x

theorem abs_neg_two : absolute_value (-2) = 2 := 
by 
  sorry

end abs_neg_two_l107_107359


namespace inverse_of_f_l107_107727

def f (x : ℝ) : ℝ := 7 - 3 * x

noncomputable def f_inv (x : ℝ) : ℝ := (7 - x) / 3

theorem inverse_of_f : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x :=
by
  intros
  sorry

end inverse_of_f_l107_107727


namespace complement_of_M_in_U_l107_107845

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 4, 6}
def complement_U_M : Set ℕ := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = complement_U_M :=
by
  sorry

end complement_of_M_in_U_l107_107845


namespace alyssa_picked_42_l107_107381

variable (totalPears nancyPears : ℕ)
variable (total_picked : totalPears = 59)
variable (nancy_picked : nancyPears = 17)

theorem alyssa_picked_42 (h1 : totalPears = 59) (h2 : nancyPears = 17) :
  totalPears - nancyPears = 42 :=
by
  sorry

end alyssa_picked_42_l107_107381


namespace exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1_l107_107671

theorem exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1 (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) :
  (∃ a : ℤ, a^2 ≡ -1 [ZMOD p]) ↔ p % 4 = 1 :=
sorry

end exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1_l107_107671


namespace movie_attendance_l107_107102

theorem movie_attendance (total_seats : ℕ) (empty_seats : ℕ) (h1 : total_seats = 750) (h2 : empty_seats = 218) :
  total_seats - empty_seats = 532 := by
  sorry

end movie_attendance_l107_107102


namespace hyperbola_foci_distance_l107_107433

theorem hyperbola_foci_distance (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 9) :
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 34 := 
by
  sorry

end hyperbola_foci_distance_l107_107433


namespace correct_option_C_l107_107857

noncomputable def question := "Which of the following operations is correct?"
noncomputable def option_A := (-2)^2
noncomputable def option_B := (-2)^3
noncomputable def option_C := (-1/2)^3
noncomputable def option_D := (-7/3)^3
noncomputable def correct_answer := -1/8

theorem correct_option_C :
  option_C = correct_answer := by
  sorry

end correct_option_C_l107_107857


namespace line_intersects_circle_l107_107478

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, y = k * (x - 1) ∧ x^2 + y^2 = 1 :=
by
  sorry

end line_intersects_circle_l107_107478


namespace triangle_perimeter_l107_107362

variable (r A p : ℝ)

-- Define the conditions from the problem
def inradius (r : ℝ) := r = 3
def area (A : ℝ) := A = 30
def perimeter (A r p : ℝ) := A = r * (p / 2)

-- The theorem stating the problem
theorem triangle_perimeter (h1 : inradius r) (h2 : area A) (h3 : perimeter A r p) : p = 20 := 
by
  -- Proof is provided by the user, so we skip it with sorry
  sorry

end triangle_perimeter_l107_107362


namespace range_of_a_l107_107124

variables (m a x y : ℝ)

def p (m a : ℝ) : Prop := m^2 + 12 * a^2 < 7 * a * m ∧ a > 0

def ellipse (m x y : ℝ) : Prop := (x^2)/(m-1) + (y^2)/(2-m) = 1

def q (m : ℝ) (x y : ℝ) : Prop := ellipse m x y ∧ 1 < m ∧ m < 3/2

theorem range_of_a :
  (∃ m, p m a → (∀ x y, q m x y)) → (1/3 ≤ a ∧ a ≤ 3/8) :=
sorry

end range_of_a_l107_107124


namespace pretzels_count_l107_107424

-- Define the number of pretzels
def pretzels : ℕ := 64

-- Given conditions
def goldfish (P : ℕ) : ℕ := 4 * P
def suckers : ℕ := 32
def kids : ℕ := 16
def items_per_kid : ℕ := 22
def total_items (P : ℕ) : ℕ := P + goldfish P + suckers

-- The theorem to prove
theorem pretzels_count : total_items pretzels = kids * items_per_kid := by
  sorry

end pretzels_count_l107_107424


namespace race_total_people_l107_107131

theorem race_total_people (b t : ℕ) 
(h1 : b = t + 15) 
(h2 : 3 * t = 2 * b + 15) : 
b + t = 105 := 
sorry

end race_total_people_l107_107131


namespace neg_ten_plus_three_l107_107514

theorem neg_ten_plus_three :
  -10 + 3 = -7 := by
  sorry

end neg_ten_plus_three_l107_107514


namespace bound_c_n_l107_107037

theorem bound_c_n (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) :
  (a 1 = 4) →
  (∀ n, a (n + 1) = a n * (a n - 1)) →
  (∀ n, 2^b n = a n) →
  (∀ n, 2^(n - c n) = b n) →
  ∃ (m M : ℝ), (m = 0) ∧ (M = 1) ∧ ∀ n > 0, m ≤ c n ∧ c n ≤ M :=
by
  intro h1 h2 h3 h4
  use 0
  use 1
  sorry

end bound_c_n_l107_107037


namespace graph_crosses_x_axis_at_origin_l107_107065

-- Let g(x) be a quadratic function defined as ax^2 + bx
def g (a b x : ℝ) : ℝ := a * x^2 + b * x

-- Define the conditions a ≠ 0 and b ≠ 0
axiom a_ne_0 (a : ℝ) : a ≠ 0
axiom b_ne_0 (b : ℝ) : b ≠ 0

-- The problem statement
theorem graph_crosses_x_axis_at_origin (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  ∃ x : ℝ, g a b x = 0 ∧ ∀ x', g a b x' = 0 → x' = 0 ∨ x' = -b / a :=
sorry

end graph_crosses_x_axis_at_origin_l107_107065


namespace square_areas_l107_107244

theorem square_areas (z : ℂ) 
  (h1 : ¬ (2 : ℂ) * z^2 = z)
  (h2 : ¬ (3 : ℂ) * z^3 = z)
  (sz : (3 * z^3 - z) = (I * (2 * z^2 - z)) ∨ (3 * z^3 - z) = (-I * (2 * z^2 - z))) :
  ∃ (areas : Finset ℝ), areas = {85, 4500} :=
by {
  sorry
}

end square_areas_l107_107244


namespace family_children_count_l107_107217

theorem family_children_count (x y : ℕ) 
  (sister_condition : x = y - 1) 
  (brother_condition : y = 2 * (x - 1)) : 
  x + y = 7 := 
sorry

end family_children_count_l107_107217


namespace area_bounded_by_circles_and_x_axis_l107_107798

/--
Circle C has its center at (5, 5) and radius 5 units.
Circle D has its center at (15, 5) and radius 5 units.
Prove that the area of the region bounded by these circles
and the x-axis is 50 - 25 * π square units.
-/
theorem area_bounded_by_circles_and_x_axis :
  let C_center := (5, 5)
  let D_center := (15, 5)
  let radius := 5
  (2 * (radius * radius) * π / 2) + (10 * radius) = 50 - 25 * π :=
sorry

end area_bounded_by_circles_and_x_axis_l107_107798


namespace consecutive_ints_square_l107_107286

theorem consecutive_ints_square (a b : ℤ) (h : b = a + 1) : 
  a^2 + b^2 + (a * b)^2 = (a * b + 1)^2 := 
by sorry

end consecutive_ints_square_l107_107286


namespace weight_of_b_l107_107995

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45) 
  (h2 : (a + b) / 2 = 41) 
  (h3 : (b + c) / 2 = 43) 
  : b = 33 :=
by
  sorry

end weight_of_b_l107_107995


namespace correct_option_is_C_l107_107420

-- Definitions for given conditions
def optionA (x y : ℝ) : Prop := 3 * x + 3 * y = 6 * x * y
def optionB (x y : ℝ) : Prop := 4 * x * y^2 - 5 * x * y^2 = -1
def optionC (x : ℝ) : Prop := -2 * (x - 3) = -2 * x + 6
def optionD (a : ℝ) : Prop := 2 * a + a = 3 * a^2

-- The proof statement to show that Option C is the correct calculation
theorem correct_option_is_C (x y a : ℝ) : 
  ¬ optionA x y ∧ ¬ optionB x y ∧ optionC x ∧ ¬ optionD a :=
by
  -- Proof not required, using sorry to compile successfully
  sorry

end correct_option_is_C_l107_107420


namespace arithmetic_seq_fifth_term_l107_107772

theorem arithmetic_seq_fifth_term (x y : ℝ) 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 = 2 * x^2 + 3 * y^2) 
  (h2 : a2 = x^2 + 2 * y^2) 
  (h3 : a3 = 2 * x^2 - y^2) 
  (h4 : a4 = x^2 - y^2) 
  (d : ℝ) 
  (hd : d = -x^2 - y^2) 
  (h_arith: ∀ i j k : ℕ, i < j ∧ j < k → a2 - a1 = d ∧ a3 - a2 = d ∧ a4 - a3 = d) : 
  a4 + d = -2 * y^2 := 
by 
  sorry

end arithmetic_seq_fifth_term_l107_107772


namespace cube_root_eval_l107_107704

noncomputable def cube_root_nested (N : ℝ) : ℝ := (N * (N * (N * (N)))) ^ (1/81)

theorem cube_root_eval (N : ℝ) (h : N > 1) : 
  cube_root_nested N = N ^ (40 / 81) := 
sorry

end cube_root_eval_l107_107704


namespace original_deck_total_l107_107473

theorem original_deck_total (b y : ℕ) 
    (h1 : (b : ℚ) / (b + y) = 2 / 5)
    (h2 : (b : ℚ) / (b + y + 6) = 5 / 14) :
    b + y = 50 := by
  sorry

end original_deck_total_l107_107473


namespace work_problem_l107_107519

theorem work_problem (W : ℝ) (A_rate : ℝ) (AB_rate : ℝ) : A_rate = W / 14 ∧ AB_rate = W / 10 → 1 / (AB_rate - A_rate) = 35 :=
by
  sorry

end work_problem_l107_107519


namespace fraction_zero_l107_107556

theorem fraction_zero (x : ℝ) (h : (x^2 - 1) / (x + 1) = 0) : x = 1 := 
sorry

end fraction_zero_l107_107556


namespace division_rounded_nearest_hundredth_l107_107903

theorem division_rounded_nearest_hundredth :
  Float.round (285 * 387 / (981^2) * 100) / 100 = 0.11 :=
by
  sorry

end division_rounded_nearest_hundredth_l107_107903


namespace propositions_false_l107_107751

structure Plane :=
(is_plane : Prop)

structure Line :=
(in_plane : Plane → Prop)

def is_parallel (p1 p2 : Plane) : Prop := sorry
def is_perpendicular (p1 p2 : Plane) : Prop := sorry
def line_parallel (l1 l2 : Line) : Prop := sorry
def line_perpendicular (l1 l2 : Line) : Prop := sorry

variable (α β : Plane)
variable (l m : Line)

axiom α_neq_β : α ≠ β
axiom l_in_α : l.in_plane α
axiom m_in_β : m.in_plane β

theorem propositions_false :
  ¬(is_parallel α β → line_parallel l m) ∧ 
  ¬(line_perpendicular l m → is_perpendicular α β) := 
sorry

end propositions_false_l107_107751


namespace area_ratio_of_squares_l107_107963

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * (4 * b) = 4 * a) : (a * a) / (b * b) = 16 :=
by
  sorry

end area_ratio_of_squares_l107_107963


namespace factor_difference_of_squares_l107_107099

theorem factor_difference_of_squares (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := 
by
  sorry

end factor_difference_of_squares_l107_107099


namespace vertex_and_maximum_l107_107854

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := -3 * x^2 + 6 * x - 9

-- Prove that the vertex of the parabola quadratic is (1, -6) and it is a maximum point
theorem vertex_and_maximum :
  (∃ x y : ℝ, (quadratic x = y) ∧ (x = 1) ∧ (y = -6)) ∧
  (∀ x : ℝ, quadratic x ≤ quadratic 1) :=
sorry

end vertex_and_maximum_l107_107854


namespace cone_sector_central_angle_l107_107858

noncomputable def base_radius := 1
noncomputable def slant_height := 2
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def arc_length (r : ℝ) := circumference r
noncomputable def central_angle (l : ℝ) (s : ℝ) := l / s

theorem cone_sector_central_angle : central_angle (arc_length base_radius) slant_height = Real.pi := 
by 
  -- Here we acknowledge that the proof would go, but it is left out as per instructions.
  sorry

end cone_sector_central_angle_l107_107858


namespace area_of_field_l107_107571

-- Definitions based on the conditions
def length_uncovered (L : ℝ) := L = 20
def fencing_required (W : ℝ) (L : ℝ) := 2 * W + L = 76

-- Statement of the theorem to be proved
theorem area_of_field (L W : ℝ) (hL : length_uncovered L) (hF : fencing_required W L) : L * W = 560 := by
  sorry

end area_of_field_l107_107571


namespace janes_stick_shorter_than_sarahs_l107_107392

theorem janes_stick_shorter_than_sarahs :
  ∀ (pat_length jane_length pat_dirt sarah_factor : ℕ),
    pat_length = 30 →
    jane_length = 22 →
    pat_dirt = 7 →
    sarah_factor = 2 →
    (sarah_factor * (pat_length - pat_dirt)) - jane_length = 24 :=
by
  intros pat_length jane_length pat_dirt sarah_factor h1 h2 h3 h4
  -- sorry skips the proof
  sorry

end janes_stick_shorter_than_sarahs_l107_107392


namespace inequality_solution_set_l107_107342

theorem inequality_solution_set :
  {x : ℝ | (x / (x ^ 2 - 8 * x + 15) ≥ 2) ∧ (x ^ 2 - 8 * x + 15 ≠ 0)} =
  {x : ℝ | (5 / 2 ≤ x ∧ x < 3) ∨ (5 < x ∧ x ≤ 6)} :=
by
  -- The proof is omitted
  sorry

end inequality_solution_set_l107_107342


namespace eccentricity_range_of_ellipse_l107_107389

theorem eccentricity_range_of_ellipse 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (P : ℝ × ℝ) (hP_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (h_foci_relation : ∀(θ₁ θ₂ : ℝ), a / (Real.sin θ₁) = c / (Real.sin θ₂)) :
  ∃ (e : ℝ), e = c / a ∧ (Real.sqrt 2 - 1 < e ∧ e < 1) := 
sorry

end eccentricity_range_of_ellipse_l107_107389


namespace total_time_to_complete_work_l107_107170

-- Definitions based on conditions
variable (W : ℝ) -- W is the total work
variable (Mahesh_days : ℝ := 35) -- Mahesh can complete the work in 35 days
variable (Mahesh_working_days : ℝ := 20) -- Mahesh works for 20 days
variable (Rajesh_days : ℝ := 30) -- Rajesh finishes the remaining work in 30 days

-- Proof statement
theorem total_time_to_complete_work : Mahesh_working_days + Rajesh_days = 50 :=
by
  sorry

end total_time_to_complete_work_l107_107170


namespace exists_pair_distinct_integers_l107_107070

theorem exists_pair_distinct_integers :
  ∃ (a b : ℤ), a ≠ b ∧ (a / 2015 + b / 2016 = (2015 + 2016) / (2015 * 2016)) :=
by
  -- Constructing the proof or using sorry to skip it if not needed here
  sorry

end exists_pair_distinct_integers_l107_107070


namespace additional_lollipops_needed_l107_107150

theorem additional_lollipops_needed
  (kids : ℕ) (initial_lollipops : ℕ) (min_lollipops : ℕ) (max_lollipops : ℕ)
  (total_kid_with_lollipops : ∀ k, ∃ n, min_lollipops ≤ n ∧ n ≤ max_lollipops ∧ k = n ∨ k = n + 1 )
  (divisible_by_kids : (min_lollipops + max_lollipops) % kids = 0)
  (min_lollipops_eq : min_lollipops = 42)
  (kids_eq : kids = 42)
  (initial_lollipops_eq : initial_lollipops = 650)
  : ∃ additional_lollipops, (n : ℕ) = 42 → additional_lollipops = 1975 := 
by sorry

end additional_lollipops_needed_l107_107150


namespace original_number_is_13_l107_107934

theorem original_number_is_13 (x : ℝ) (h : 3 * (2 * x + 7) = 99) : x = 13 :=
sorry

end original_number_is_13_l107_107934


namespace porter_l107_107168

def previous_sale_amount : ℕ := 9000

def recent_sale_price (previous_sale_amount : ℕ) : ℕ :=
  5 * previous_sale_amount - 1000

theorem porter's_recent_sale : recent_sale_price previous_sale_amount = 44000 :=
by
  sorry

end porter_l107_107168


namespace problem1_l107_107762

theorem problem1 (a : ℝ) 
    (circle_eqn : ∀ (x y : ℝ), x^2 + y^2 - 2*a*x + a = 0)
    (line_eqn : ∀ (x y : ℝ), a*x + y + 1 = 0)
    (chord_length : ∀ (x y : ℝ), (ax + y + 1 = 0) ∧ (x^2 + y^2 - 2*a*x + a = 0)  -> ((x - x')^2 + (y - y')^2 = 4)) : 
    a = -2 := sorry

end problem1_l107_107762


namespace right_triangle_area_l107_107706

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := 1 / 2 * a * b

theorem right_triangle_area {a b : ℝ} 
  (h1 : a + b = 4) 
  (h2 : a^2 + b^2 = 14) : 
  area_of_right_triangle a b = 1 / 2 :=
by 
  sorry

end right_triangle_area_l107_107706


namespace check_3x5_board_cannot_be_covered_l107_107410

/-- Define the concept of a checkerboard with a given number of rows and columns. -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Define the number of squares on a checkerboard. -/
def num_squares (cb : Checkerboard) : ℕ :=
  cb.rows * cb.cols

/-- Define whether a board can be completely covered by dominoes. -/
def can_be_covered_by_dominoes (cb : Checkerboard) : Prop :=
  (num_squares cb) % 2 = 0

/-- Instantiate the specific checkerboard scenarios. -/
def board_3x4 := Checkerboard.mk 3 4
def board_3x5 := Checkerboard.mk 3 5
def board_4x4 := Checkerboard.mk 4 4
def board_4x5 := Checkerboard.mk 4 5
def board_6x3 := Checkerboard.mk 6 3

/-- Statement to prove which board cannot be covered completely by dominoes. -/
theorem check_3x5_board_cannot_be_covered : ¬ can_be_covered_by_dominoes board_3x5 :=
by
  /- We leave out the proof steps here as requested. -/
  sorry

end check_3x5_board_cannot_be_covered_l107_107410


namespace inverse_variation_l107_107125

theorem inverse_variation (a : ℕ) (b : ℝ) (h : a * b = 400) (h₀ : a = 3200) : b = 0.125 :=
by sorry

end inverse_variation_l107_107125


namespace remainder_3_45_plus_4_mod_5_l107_107740

theorem remainder_3_45_plus_4_mod_5 :
  (3 ^ 45 + 4) % 5 = 2 := 
by {
  sorry
}

end remainder_3_45_plus_4_mod_5_l107_107740


namespace range_and_period_range_of_m_l107_107924

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos (x + Real.pi / 3) * (Real.sin (x + Real.pi / 3) - Real.sqrt 3 * Real.cos (x + Real.pi / 3))

theorem range_and_period (x : ℝ) :
  (Set.range f = Set.Icc (-2 - Real.sqrt 3) (2 - Real.sqrt 3)) ∧ (∀ x, f (x + Real.pi) = f x) := sorry

theorem range_of_m (x m : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi / 6) (h2 : m * (f x + Real.sqrt 3) + 2 = 0) :
  m ∈ Set.Icc (- 2 * Real.sqrt 3 / 3) (-1) := sorry

end range_and_period_range_of_m_l107_107924


namespace range_of_a_l107_107829

theorem range_of_a :
  (∀ t : ℝ, 0 < t ∧ t ≤ 2 → (t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2)) →
  (2 / 13 ≤ a ∧ a ≤ 1) :=
by
  intro h
  -- Proof of the theorem goes here
  sorry

end range_of_a_l107_107829


namespace option_c_correct_l107_107421

theorem option_c_correct (α x1 x2 : ℝ) (hα1 : 0 < α) (hα2 : α < π) (hx1 : 0 < x1) (hx2 : x1 < x2) : 
  (x2 / x1) ^ Real.sin α > 1 :=
by
  sorry

end option_c_correct_l107_107421


namespace marys_garbage_bill_is_correct_l107_107079

noncomputable def calculate_garbage_bill :=
  let weekly_trash_bin_cost := 2 * 10
  let weekly_recycling_bin_cost := 1 * 5
  let weekly_green_waste_bin_cost := 1 * 3
  let total_weekly_cost := weekly_trash_bin_cost + weekly_recycling_bin_cost + weekly_green_waste_bin_cost
  let monthly_bin_cost := total_weekly_cost * 4
  let base_monthly_cost := monthly_bin_cost + 15
  let discount := base_monthly_cost * 0.18
  let discounted_cost := base_monthly_cost - discount
  let fines := 20 + 10
  discounted_cost + fines

theorem marys_garbage_bill_is_correct :
  calculate_garbage_bill = 134.14 := 
  by {
  sorry
  }

end marys_garbage_bill_is_correct_l107_107079


namespace quadratic_complete_square_l107_107313

theorem quadratic_complete_square : 
  ∃ d e : ℝ, ((x^2 - 16*x + 15) = ((x + d)^2 + e)) ∧ (d + e = -57) := by
  sorry

end quadratic_complete_square_l107_107313


namespace find_a_value_l107_107484

theorem find_a_value (a x y : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : a * x - 3 * y = 3) : a = 6 :=
by
  rw [h1, h2] at h3 -- Substitute x and y values into the equation
  sorry -- The proof is omitted as per instructions.

end find_a_value_l107_107484


namespace overall_percentage_favoring_new_tool_l107_107949

theorem overall_percentage_favoring_new_tool (teachers students : ℕ) 
  (favor_teachers favor_students : ℚ) 
  (surveyed_teachers surveyed_students : ℕ) : 
  surveyed_teachers = 200 → 
  surveyed_students = 800 → 
  favor_teachers = 0.4 → 
  favor_students = 0.75 → 
  ( ( (favor_teachers * surveyed_teachers) + (favor_students * surveyed_students) ) / (surveyed_teachers + surveyed_students) ) * 100 = 68 := 
by 
  sorry

end overall_percentage_favoring_new_tool_l107_107949


namespace sum_of_four_terms_l107_107058

theorem sum_of_four_terms (a d : ℕ) (h1 : a + d > a) (h2 : a + 2 * d > a + d)
  (h3 : (a + 2 * d) * (a + 2 * d) = (a + d) * (a + 3 * d)) (h4 : (a + 3 * d) - a = 30) :
  a + (a + d) + (a + 2 * d) + (a + 3 * d) = 129 :=
sorry

end sum_of_four_terms_l107_107058


namespace cows_and_sheep_bushels_l107_107603

theorem cows_and_sheep_bushels (bushels_per_chicken: Int) (total_bushels: Int) (num_chickens: Int) 
  (bushels_chickens: Int) (bushels_cows_sheep: Int) (num_cows: Int) (num_sheep: Int):
  bushels_per_chicken = 3 ∧ total_bushels = 35 ∧ num_chickens = 7 ∧
  bushels_chickens = num_chickens * bushels_per_chicken ∧ bushels_chickens = 21 ∧ bushels_cows_sheep = total_bushels - bushels_chickens → 
  bushels_cows_sheep = 14 := by
  sorry

end cows_and_sheep_bushels_l107_107603


namespace scientific_notation_316000000_l107_107057

theorem scientific_notation_316000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 316000000 = a * 10 ^ n ∧ a = 3.16 ∧ n = 8 :=
by
  -- Proof would be here
  sorry

end scientific_notation_316000000_l107_107057


namespace largest_k_exists_l107_107167

theorem largest_k_exists (n : ℕ) (h : n ≥ 4) : 
  ∃ k : ℕ, (∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n → (c - b) ≥ k ∧ (b - a) ≥ k ∧ (a + b ≥ c + 1)) ∧ 
  (k = (n - 1) / 3) :=
  sorry

end largest_k_exists_l107_107167


namespace count_4_digit_numbers_with_property_l107_107547

noncomputable def count_valid_4_digit_numbers : ℕ :=
  let valid_units (t : ℕ) : List ℕ := List.filter (λ u => u ≥ 3 * t) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  let choices_for_tu : ℕ := (List.length (valid_units 0)) + (List.length (valid_units 1)) + (List.length (valid_units 2))
  choices_for_tu * 9 * 9

theorem count_4_digit_numbers_with_property : count_valid_4_digit_numbers = 1701 := by
  sorry

end count_4_digit_numbers_with_property_l107_107547


namespace population_of_missing_village_eq_945_l107_107385

theorem population_of_missing_village_eq_945
  (pop1 pop2 pop3 pop4 pop5 pop6 : ℕ)
  (avg_pop total_population missing_population : ℕ)
  (h1 : pop1 = 803)
  (h2 : pop2 = 900)
  (h3 : pop3 = 1100)
  (h4 : pop4 = 1023)
  (h5 : pop5 = 980)
  (h6 : pop6 = 1249)
  (h_avg : avg_pop = 1000)
  (h_total_population : total_population = avg_pop * 7)
  (h_missing_population : missing_population = total_population - (pop1 + pop2 + pop3 + pop4 + pop5 + pop6)) :
  missing_population = 945 :=
by {
  -- Here would go the proof steps if needed
  sorry 
}

end population_of_missing_village_eq_945_l107_107385


namespace guests_did_not_come_l107_107824

theorem guests_did_not_come 
  (total_cookies : ℕ) 
  (prepared_guests : ℕ) 
  (cookies_per_guest : ℕ) 
  (total_cookies_eq : total_cookies = 18) 
  (prepared_guests_eq : prepared_guests = 10)
  (cookies_per_guest_eq : cookies_per_guest = 18) 
  (total_cookies_computation : total_cookies = cookies_per_guest) :
  prepared_guests - total_cookies / cookies_per_guest = 9 :=
by
  sorry

end guests_did_not_come_l107_107824


namespace value_of_y_square_plus_inverse_square_l107_107350

variable {y : ℝ}
variable (h : 35 = y^4 + 1 / y^4)

theorem value_of_y_square_plus_inverse_square (h : 35 = y^4 + 1 / y^4) : y^2 + 1 / y^2 = Real.sqrt 37 := 
sorry

end value_of_y_square_plus_inverse_square_l107_107350


namespace find_a5_a7_l107_107703

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom h1 : a 1 + a 3 = 2
axiom h2 : a 3 + a 5 = 4

theorem find_a5_a7 (a : ℕ → ℤ) (d : ℤ) (h_seq : is_arithmetic_sequence a d)
  (h1 : a 1 + a 3 = 2) (h2 : a 3 + a 5 = 4) : a 5 + a 7 = 6 :=
sorry

end find_a5_a7_l107_107703


namespace right_triangle_perimeter_l107_107701

theorem right_triangle_perimeter 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_area : 1/2 * 30 * b = 180)
  (h_pythagorean : c^2 = 30^2 + b^2)
  : a + b + c = 42 + 2 * Real.sqrt 261 :=
sorry

end right_triangle_perimeter_l107_107701


namespace marcia_oranges_l107_107675

noncomputable def averageCost
  (appleCost bananaCost orangeCost : ℝ) 
  (numApples numBananas numOranges : ℝ) : ℝ :=
  (numApples * appleCost + numBananas * bananaCost + numOranges * orangeCost) /
  (numApples + numBananas + numOranges)

theorem marcia_oranges : 
  ∀ (appleCost bananaCost orangeCost avgCost : ℝ) 
  (numApples numBananas numOranges : ℝ),
  appleCost = 2 → 
  bananaCost = 1 → 
  orangeCost = 3 → 
  numApples = 12 → 
  numBananas = 4 → 
  avgCost = 2 → 
  averageCost appleCost bananaCost orangeCost numApples numBananas numOranges = avgCost → 
  numOranges = 4 :=
by 
  intros appleCost bananaCost orangeCost avgCost numApples numBananas numOranges
         h1 h2 h3 h4 h5 h6 h7
  sorry

end marcia_oranges_l107_107675


namespace find_integer_solutions_l107_107314

theorem find_integer_solutions (k : ℕ) (hk : k > 1) : 
  ∃ x y : ℤ, y^k = x^2 + x ↔ (k = 2 ∧ (x = 0 ∨ x = -1)) ∨ (k > 2 ∧ y^k ≠ x^2 + x) :=
by
  sorry

end find_integer_solutions_l107_107314


namespace remainder_7_pow_150_mod_12_l107_107711

theorem remainder_7_pow_150_mod_12 :
  (7^150) % 12 = 1 := sorry

end remainder_7_pow_150_mod_12_l107_107711


namespace functional_equation_to_linear_l107_107955

-- Define that f satisfies the Cauchy functional equation
variable (f : ℕ → ℝ)
axiom cauchy_eq (x y : ℕ) : f (x + y) = f x + f y

-- The theorem we want to prove
theorem functional_equation_to_linear (h : ∀ n k : ℕ, f (n * k) = n * f k) : ∃ a : ℝ, ∀ n : ℕ, f n = a * n :=
by
  sorry

end functional_equation_to_linear_l107_107955


namespace equation_of_line_AB_l107_107157

theorem equation_of_line_AB 
  (x y : ℝ)
  (passes_through_P : (4 - 1)^2 + (1 - 0)^2 = 1)     
  (circle_eq : (x - 1)^2 + y^2 = 1) :
  3 * x + y - 4 = 0 :=
sorry

end equation_of_line_AB_l107_107157


namespace bound_on_f_l107_107277

theorem bound_on_f 
  (f : ℝ → ℝ) 
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ 1) 
  (h_zeros : f 0 = 0 ∧ f 1 = 0)
  (h_condition : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ x1 ≠ x2 → |f x2 - f x1| < |x2 - x1|) 
  : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 → |f x2 - f x1| < 1/2 :=
by
  sorry

end bound_on_f_l107_107277


namespace flight_duration_l107_107491

theorem flight_duration :
  ∀ (h m : ℕ),
  3 * 60 + 42 = 15 * 60 + 57 →
  0 < m ∧ m < 60 →
  h + m = 18 :=
by
  intros h m h_def hm_bound
  sorry

end flight_duration_l107_107491


namespace mary_total_nickels_l107_107948

-- Definitions for the conditions
def initial_nickels := 7
def dad_nickels := 5
def mom_nickels := 3 * dad_nickels
def chore_nickels := 2

-- The proof problem statement
theorem mary_total_nickels : 
  initial_nickels + dad_nickels + mom_nickels + chore_nickels = 29 := 
by
  sorry

end mary_total_nickels_l107_107948


namespace geometric_sequence_second_term_l107_107792

theorem geometric_sequence_second_term (a r : ℕ) (h1 : a = 5) (h2 : a * r^4 = 1280) : a * r = 20 :=
by
  sorry

end geometric_sequence_second_term_l107_107792


namespace games_in_tournament_l107_107542

def single_elimination_games (n : Nat) : Nat :=
  n - 1

theorem games_in_tournament : single_elimination_games 24 = 23 := by
  sorry

end games_in_tournament_l107_107542


namespace system_exactly_two_solutions_l107_107615

theorem system_exactly_two_solutions (a : ℝ) : 
  (∃ x y : ℝ, |y + x + 8| + |y - x + 8| = 16 ∧ (|x| - 15)^2 + (|y| - 8)^2 = a) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, |y₁ + x₁ + 8| + |y₁ - x₁ + 8| = 16 ∧ (|x₁| - 15)^2 + (|y₁| - 8)^2 = a → 
                      |y₂ + x₂ + 8| + |y₂ - x₂ + 8| = 16 ∧ (|x₂| - 15)^2 + (|y₂| - 8)^2 = a → 
                      x₁ = x₂ ∧ y₁ = y₂) → 
  (a = 49 ∨ a = 289) :=
sorry

end system_exactly_two_solutions_l107_107615


namespace sequence_period_16_l107_107011

theorem sequence_period_16 (a : ℝ) (h : a > 0) 
  (u : ℕ → ℝ) (h1 : u 1 = a) (h2 : ∀ n, u (n + 1) = -1 / (u n + 1)) : 
  u 16 = a :=
sorry

end sequence_period_16_l107_107011


namespace ryan_chinese_learning_hours_l107_107081

theorem ryan_chinese_learning_hours
    (hours_per_day : ℕ) 
    (days : ℕ) 
    (h1 : hours_per_day = 4) 
    (h2 : days = 6) : 
    hours_per_day * days = 24 := 
by 
    sorry

end ryan_chinese_learning_hours_l107_107081


namespace line_through_points_l107_107105

theorem line_through_points (m b: ℝ) 
  (h1: ∃ m, ∀ x y : ℝ, ((x, y) = (1, 3) ∨ (x, y) = (3, 7)) → y = m * x + b) 
  (h2: ∀ x y : ℝ, ((x, y) = (1, 3) ∨ (x, y) = (3, 7)) → y = m * x + b):
  m + b = 3 :=
by
  -- proof goes here
  sorry

end line_through_points_l107_107105


namespace range_of_m_l107_107802

theorem range_of_m (m : ℝ) (h : 0 < m)
  (subset_cond : ∀ x y : ℝ, x - 4 ≤ 0 → y ≥ 0 → mx - y ≥ 0 → (x - 2)^2 + (y - 2)^2 ≤ 8) :
  m ≤ 1 :=
sorry

end range_of_m_l107_107802


namespace stock_increase_l107_107546

theorem stock_increase (x : ℝ) (h₁ : x > 0) :
  (1.25 * (0.85 * x) - x) / x * 100 = 6.25 :=
by 
  -- {proof steps would go here}
  sorry

end stock_increase_l107_107546


namespace original_peaches_l107_107807

theorem original_peaches (picked: ℕ) (current: ℕ) (initial: ℕ) : 
  picked = 52 → 
  current = 86 → 
  initial = current - picked → 
  initial = 34 := 
by intros h1 h2 h3
   subst h1
   subst h2
   subst h3
   simp

end original_peaches_l107_107807


namespace poly_at_2_eq_0_l107_107032

def poly (x : ℝ) : ℝ := x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

theorem poly_at_2_eq_0 : poly 2 = 0 := by
  sorry

end poly_at_2_eq_0_l107_107032


namespace division_decimal_l107_107145

theorem division_decimal (x : ℝ) (h : x = 0.3333): 12 / x = 36 :=
  by
    sorry

end division_decimal_l107_107145


namespace evaluation_of_expression_l107_107980

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l107_107980


namespace total_cost_for_doughnuts_l107_107062

theorem total_cost_for_doughnuts
  (num_students : ℕ)
  (num_chocolate : ℕ)
  (num_glazed : ℕ)
  (price_chocolate : ℕ)
  (price_glazed : ℕ)
  (H1 : num_students = 25)
  (H2 : num_chocolate = 10)
  (H3 : num_glazed = 15)
  (H4 : price_chocolate = 2)
  (H5 : price_glazed = 1) :
  num_chocolate * price_chocolate + num_glazed * price_glazed = 35 :=
by
  -- Proof steps would go here
  sorry

end total_cost_for_doughnuts_l107_107062


namespace second_smallest_perimeter_l107_107967

theorem second_smallest_perimeter (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → 
  (a + b + c = 12) :=
by
  sorry

end second_smallest_perimeter_l107_107967


namespace necessary_but_not_sufficient_condition_l107_107717

variables {a b c : ℝ × ℝ}

def nonzero_vector (v : ℝ × ℝ) : Prop := v ≠ (0, 0)

theorem necessary_but_not_sufficient_condition (ha : nonzero_vector a) (hb : nonzero_vector b) (hc : nonzero_vector c) :
  (a.1 * (b.1 - c.1) + a.2 * (b.2 - c.2) = 0) ↔ (b = c) :=
sorry

end necessary_but_not_sufficient_condition_l107_107717


namespace prob_2022_2023_l107_107925

theorem prob_2022_2023 (n : ℤ) (h : (n - 2022)^2 + (2023 - n)^2 = 1) : (n - 2022) * (2023 - n) = 0 :=
sorry

end prob_2022_2023_l107_107925


namespace tan_theta_condition_l107_107679

open Real

theorem tan_theta_condition (k : ℤ) : 
  (∃ θ : ℝ, θ = 2 * k * π + π / 4 ∧ tan θ = 1) ∧ ¬ (∀ θ : ℝ, tan θ = 1 → ∃ k : ℤ, θ = 2 * k * π + π / 4) :=
by sorry

end tan_theta_condition_l107_107679


namespace tiles_difference_l107_107019

-- Definitions based on given conditions
def initial_blue_tiles : Nat := 20
def initial_green_tiles : Nat := 10
def first_border_green_tiles : Nat := 24
def second_border_green_tiles : Nat := 36

-- Problem statement
theorem tiles_difference :
  initial_green_tiles + first_border_green_tiles + second_border_green_tiles - initial_blue_tiles = 50 :=
by
  sorry

end tiles_difference_l107_107019


namespace min_value_expression_l107_107609

theorem min_value_expression (x : ℝ) (h : x > 3) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 21) ∧ 
           (∀ z : ℝ, (z = (x + 18) / Real.sqrt (x - 3)) → y ≤ z) := 
sorry

end min_value_expression_l107_107609


namespace paige_folders_l107_107140

-- Definitions derived from the conditions
def initial_files : Nat := 27
def deleted_files : Nat := 9
def files_per_folder : Nat := 6

-- Define the remaining files after deletion
def remaining_files : Nat := initial_files - deleted_files

-- The theorem: Prove that the number of folders is 3
theorem paige_folders : remaining_files / files_per_folder = 3 := by
  sorry

end paige_folders_l107_107140


namespace triangle_range_condition_l107_107024

def triangle_side_range (x : ℝ) : Prop :=
  (1 < x) ∧ (x < 17)

theorem triangle_range_condition (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 8) → (b = 9) → triangle_side_range x :=
by
  intros h1 h2
  dsimp [triangle_side_range]
  sorry

end triangle_range_condition_l107_107024


namespace find_x_squared_plus_inv_squared_l107_107692

noncomputable def x : ℝ := sorry

theorem find_x_squared_plus_inv_squared (h : x^4 + 1 / x^4 = 240) : x^2 + 1 / x^2 = Real.sqrt 242 := by
  sorry

end find_x_squared_plus_inv_squared_l107_107692


namespace part_1_solution_set_part_2_a_range_l107_107646

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part_1_solution_set (a : ℝ) (h : a = 4) :
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
by
  sorry

theorem part_2_a_range :
  {a : ℝ | ∀ x : ℝ, f x a ≥ 4} = {a : ℝ | a ≤ -3 ∨ a ≥ 5} :=
by
  sorry

end part_1_solution_set_part_2_a_range_l107_107646


namespace solve_for_x_l107_107034

theorem solve_for_x (x : ℝ) (h1 : 1 - x^2 = 0) (h2 : x ≠ 1) : x = -1 := 
by 
  sorry

end solve_for_x_l107_107034


namespace range_of_a_l107_107202

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * x^3 + (1 / 2) * x^2 + 2 * a * x

theorem range_of_a (a : ℝ) :
  (∀ x > (2 / 3), (deriv (f a)) x > 0) → a > -(1 / 9) :=
by
  sorry

end range_of_a_l107_107202


namespace total_books_correct_l107_107269

-- Define the number of books each person has
def booksKeith : Nat := 20
def booksJason : Nat := 21
def booksMegan : Nat := 15

-- Define the total number of books they have together
def totalBooks : Nat := booksKeith + booksJason + booksMegan

-- Prove that the total number of books is 56
theorem total_books_correct : totalBooks = 56 := by
  sorry

end total_books_correct_l107_107269


namespace fourth_term_correct_l107_107431

def fourth_term_sequence : Nat :=
  4^0 + 4^1 + 4^2 + 4^3

theorem fourth_term_correct : fourth_term_sequence = 85 :=
by
  sorry

end fourth_term_correct_l107_107431


namespace simplify_expression_correct_l107_107894

def simplify_expression (i : ℂ) (h : i ^ 2 = -1) : ℂ :=
  3 * (4 - 2 * i) + 2 * i * (3 - i)

theorem simplify_expression_correct (i : ℂ) (h : i ^ 2 = -1) : simplify_expression i h = 14 := 
by
  sorry

end simplify_expression_correct_l107_107894


namespace g_prime_positive_l107_107860

noncomputable def f (a x : ℝ) := a * x - a * x ^ 2 - Real.log x

noncomputable def g (a x : ℝ) := -2 * (a * x - a * x ^ 2 - Real.log x) - (2 * a + 1) * x ^ 2 + a * x

def g_zero (a x1 x2 : ℝ) := g a x1 = 0 ∧ g a x2 = 0

def x1_x2_condition (x1 x2 : ℝ) := x1 < x2 ∧ x2 < 4 * x1

theorem g_prime_positive (a x1 x2 : ℝ) (h1 : g_zero a x1 x2) (h2 : x1_x2_condition x1 x2) :
  (deriv (g a) ((2 * x1 + x2) / 3)) > 0 := by
  sorry

end g_prime_positive_l107_107860


namespace find_salary_january_l107_107441

noncomputable section
open Real

def average_salary_jan_to_apr (J F M A : ℝ) : Prop := 
  (J + F + M + A) / 4 = 8000

def average_salary_feb_to_may (F M A May : ℝ) : Prop := 
  (F + M + A + May) / 4 = 9500

def may_salary_value (May : ℝ) : Prop := 
  May = 6500

theorem find_salary_january : 
  ∀ J F M A May, 
    average_salary_jan_to_apr J F M A → 
    average_salary_feb_to_may F M A May → 
    may_salary_value May → 
    J = 500 :=
by
  intros J F M A May h1 h2 h3
  sorry

end find_salary_january_l107_107441


namespace passing_percentage_l107_107166

theorem passing_percentage
  (marks_obtained : ℕ)
  (marks_failed_by : ℕ)
  (max_marks : ℕ)
  (h_marks_obtained : marks_obtained = 92)
  (h_marks_failed_by : marks_failed_by = 40)
  (h_max_marks : max_marks = 400) :
  (marks_obtained + marks_failed_by) / max_marks * 100 = 33 := 
by
  sorry

end passing_percentage_l107_107166


namespace kekai_ratio_l107_107437

/-
Kekai sells 5 shirts at $1 each,
5 pairs of pants at $3 each,
and he has $10 left after giving some money to his parents.
Our goal is to prove the ratio of the money Kekai gives to his parents
to the total money he earns from selling his clothes is 1:2.
-/

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def shirt_price : ℕ := 1
def pants_price : ℕ := 3
def money_left : ℕ := 10

def total_earnings : ℕ := (shirts_sold * shirt_price) + (pants_sold * pants_price)
def money_given_to_parents : ℕ := total_earnings - money_left
def ratio (a b : ℕ) := (a / Nat.gcd a b, b / Nat.gcd a b)

theorem kekai_ratio : ratio money_given_to_parents total_earnings = (1, 2) :=
  by
    sorry

end kekai_ratio_l107_107437


namespace inequality_true_l107_107591

theorem inequality_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
by
  sorry

end inequality_true_l107_107591


namespace min_value_of_3x_plus_4y_is_5_l107_107021

theorem min_value_of_3x_plus_4y_is_5 :
  ∀ (x y : ℝ), 0 < x → 0 < y → (3 / x + 1 / y = 5) → (∃ (b : ℝ), b = 3 * x + 4 * y ∧ ∀ (x y : ℝ), 0 < x → 0 < y → (3 / x + 1 / y = 5) → 3 * x + 4 * y ≥ b) :=
by
  intro x y x_pos y_pos h_eq
  let b := 5
  use b
  simp [b]
  sorry

end min_value_of_3x_plus_4y_is_5_l107_107021


namespace determine_common_ratio_l107_107569

-- Definition of geometric sequence and sum of first n terms
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_geometric_sequence (a : ℕ → ℝ) : ℕ → ℝ
  | 0       => a 0
  | (n + 1) => a (n + 1) + sum_geometric_sequence a n

-- Main theorem
theorem determine_common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : is_geometric_sequence a q)
  (h3 : ∀ n, S n = sum_geometric_sequence a n)
  (h4 : 3 * (S 2 + a 2 + a 1 * q^2) = 8 * a 1 * q + 5 * a 1) :
  q = 2 :=
by 
  sorry

end determine_common_ratio_l107_107569


namespace bisectors_form_inscribed_quadrilateral_l107_107954

noncomputable def angle_sum_opposite_bisectors {α β γ δ : ℝ} (a_bisector b_bisector c_bisector d_bisector : ℝ)
  (cond : α + β + γ + δ = 360) : Prop :=
  (a_bisector + b_bisector + c_bisector + d_bisector) = 180

theorem bisectors_form_inscribed_quadrilateral
  {α β γ δ : ℝ} (convex_quad : α + β + γ + δ = 360) :
  ∃ a_bisector b_bisector c_bisector d_bisector : ℝ,
  angle_sum_opposite_bisectors a_bisector b_bisector c_bisector d_bisector convex_quad := 
sorry

end bisectors_form_inscribed_quadrilateral_l107_107954


namespace gcd_154_308_462_l107_107310

theorem gcd_154_308_462 : Nat.gcd (Nat.gcd 154 308) 462 = 154 := by
  sorry

end gcd_154_308_462_l107_107310


namespace max_digit_sum_in_24_hour_format_l107_107309

theorem max_digit_sum_in_24_hour_format : 
  ∃ t : ℕ × ℕ, (0 ≤ t.fst ∧ t.fst < 24 ∧ 0 ≤ t.snd ∧ t.snd < 60 ∧ (t.fst / 10 + t.fst % 10 + t.snd / 10 + t.snd % 10 = 24)) :=
sorry

end max_digit_sum_in_24_hour_format_l107_107309


namespace fraction_playing_in_field_l107_107178

def class_size : ℕ := 50
def students_painting : ℚ := 3/5
def students_left_in_classroom : ℕ := 10

theorem fraction_playing_in_field :
  (class_size - students_left_in_classroom - students_painting * class_size) / class_size = 1/5 :=
by
  sorry

end fraction_playing_in_field_l107_107178


namespace pow_mod_3_225_l107_107427

theorem pow_mod_3_225 :
  (3 ^ 225) % 11 = 1 :=
by
  -- Given condition from problem:
  have h : 3 ^ 5 % 11 = 1 := by norm_num
  -- Proceed to prove based on this condition
  sorry

end pow_mod_3_225_l107_107427


namespace correct_statements_about_f_l107_107450

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem correct_statements_about_f : 
  (∀ x, (f x) ≤ (f e)) ∧ (f e = 1 / e) ∧ 
  (∀ x, (f x = 0) → x = 1) ∧ 
  (f 2 < f π ∧ f π < f 3) :=
by
  sorry

end correct_statements_about_f_l107_107450


namespace cost_of_second_type_of_rice_is_22_l107_107442

noncomputable def cost_second_type_of_rice (c1 : ℝ) (w1 : ℝ) (w2 : ℝ) (avg : ℝ) (total_weight : ℝ) : ℝ :=
  ((total_weight * avg) - (w1 * c1)) / w2

theorem cost_of_second_type_of_rice_is_22 :
  cost_second_type_of_rice 16 8 4 18 12 = 22 :=
by
  sorry

end cost_of_second_type_of_rice_is_22_l107_107442


namespace coefficients_of_quadratic_function_l107_107191

-- Define the quadratic function.
def quadratic_function (x : ℝ) : ℝ :=
  2 * (x - 3) ^ 2 + 2

-- Define the expected expanded form.
def expanded_form (x : ℝ) : ℝ :=
  2 * x ^ 2 - 12 * x + 20

-- State the proof problem.
theorem coefficients_of_quadratic_function :
  ∀ (x : ℝ), quadratic_function x = expanded_form x := by
  sorry

end coefficients_of_quadratic_function_l107_107191


namespace group_age_analysis_l107_107966

theorem group_age_analysis (total_members : ℕ) (average_age : ℝ) (zero_age_members : ℕ) 
  (h1 : total_members = 50) (h2 : average_age = 5) (h3 : zero_age_members = 10) :
  let total_age := total_members * average_age
  let non_zero_members := total_members - zero_age_members
  let non_zero_average_age := total_age / non_zero_members
  non_zero_members = 40 ∧ non_zero_average_age = 6.25 :=
by
  let total_age := total_members * average_age
  let non_zero_members := total_members - zero_age_members
  let non_zero_average_age := total_age / non_zero_members
  have h_non_zero_members : non_zero_members = 40 := by sorry
  have h_non_zero_average_age : non_zero_average_age = 6.25 := by sorry
  exact ⟨h_non_zero_members, h_non_zero_average_age⟩

end group_age_analysis_l107_107966


namespace lower_limit_of_a_l107_107664

theorem lower_limit_of_a (a b : ℤ) (h_a : a < 26) (h_b1 : b > 14) (h_b2 : b < 31) (h_ineq : (4 : ℚ) / 3 ≤ a / b) : 
  20 ≤ a :=
by
  sorry

end lower_limit_of_a_l107_107664


namespace intersection_empty_implies_m_leq_neg1_l107_107255

theorem intersection_empty_implies_m_leq_neg1 (m : ℝ) :
  (∀ (x y: ℝ), (x < m) → (y = x^2 + 2*x) → y < -1) →
  m ≤ -1 :=
by
  intro h
  sorry

end intersection_empty_implies_m_leq_neg1_l107_107255


namespace complex_exp_sum_l107_107887

def w : ℂ := sorry  -- We define w as a complex number, satisfying the given condition.

theorem complex_exp_sum (h : w^2 - w + 1 = 0) : 
  w^97 + w^98 + w^99 + w^100 + w^101 + w^102 = -2 + 2 * w :=
by
  sorry

end complex_exp_sum_l107_107887


namespace village_population_rate_l107_107428

noncomputable def population_change_X (initial_X : ℕ) (decrease_rate : ℕ) (years : ℕ) : ℕ :=
  initial_X - decrease_rate * years

noncomputable def population_change_Y (initial_Y : ℕ) (increase_rate : ℕ) (years : ℕ) : ℕ :=
  initial_Y + increase_rate * years

theorem village_population_rate (initial_X decrease_rate initial_Y years result : ℕ) 
  (h1 : initial_X = 70000) (h2 : decrease_rate = 1200) 
  (h3 : initial_Y = 42000) (h4 : years = 14) 
  (h5 : initial_X - decrease_rate * years = initial_Y + result * years) 
  : result = 800 :=
  sorry

end village_population_rate_l107_107428


namespace largest_good_number_is_576_smallest_bad_number_is_443_l107_107458

def is_good_number (M : ℕ) : Prop :=
  ∃ (a b c d : ℤ), M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

def largest_good_number : ℕ := 576

def smallest_bad_number : ℕ := 443

theorem largest_good_number_is_576 : ∀ M : ℕ, is_good_number M → M ≤ 576 := 
by
  sorry

theorem smallest_bad_number_is_443 : ∀ M : ℕ, ¬ is_good_number M → 443 ≤ M :=
by
  sorry

end largest_good_number_is_576_smallest_bad_number_is_443_l107_107458


namespace circle_center_distance_travelled_l107_107683

theorem circle_center_distance_travelled :
  ∀ (r : ℝ) (a b c : ℝ), r = 2 ∧ a = 9 ∧ b = 12 ∧ c = 15 → (a^2 + b^2 = c^2) → 
  ∃ (d : ℝ), d = 24 :=
by
  intros r a b c h1 h2
  sorry

end circle_center_distance_travelled_l107_107683


namespace remainder_when_divided_by_15_l107_107306

theorem remainder_when_divided_by_15 (c d : ℤ) (h1 : c % 60 = 47) (h2 : d % 45 = 14) : (c + d) % 15 = 1 :=
  sorry

end remainder_when_divided_by_15_l107_107306


namespace at_least_one_ge_one_l107_107563

theorem at_least_one_ge_one (x y : ℝ) (h : x + y ≥ 2) : x ≥ 1 ∨ y ≥ 1 :=
sorry

end at_least_one_ge_one_l107_107563


namespace gcd_9011_4379_l107_107844

def a : ℕ := 9011
def b : ℕ := 4379

theorem gcd_9011_4379 : Nat.gcd a b = 1 := by
  sorry

end gcd_9011_4379_l107_107844


namespace first_train_length_correct_l107_107817

noncomputable def length_of_first_train : ℝ :=
  let speed_first_train := 90 * 1000 / 3600  -- converting to m/s
  let speed_second_train := 72 * 1000 / 3600 -- converting to m/s
  let relative_speed := speed_first_train + speed_second_train
  let distance_apart := 630
  let length_second_train := 200
  let time_to_meet := 13.998880089592832
  let distance_covered := relative_speed * time_to_meet
  let total_distance := distance_apart
  let length_first_train := total_distance - length_second_train
  length_first_train

theorem first_train_length_correct :
  length_of_first_train = 430 :=
by
  -- Place for the proof steps
  sorry

end first_train_length_correct_l107_107817


namespace problem_statement_l107_107985

noncomputable def a : ℚ := 18 / 11
noncomputable def c : ℚ := -30 / 11

theorem problem_statement (a b c : ℚ) (h1 : b / a = 4)
    (h2 : b = 18 - 7 * a) (h3 : c = 2 * a - 6):
    a = 18 / 11 ∧ c = -30 / 11 :=
by
  sorry

end problem_statement_l107_107985


namespace find_N_l107_107176

theorem find_N (
    A B : ℝ) (N : ℕ) (r : ℝ) (hA : A = N * π * r^2 / 2) 
    (hB : B = (π * r^2 / 2) * (N^2 - N)) 
    (ratio : A / B = 1 / 18) : 
    N = 19 :=
by
  sorry

end find_N_l107_107176


namespace solution_inequality_l107_107354

theorem solution_inequality
  (a a' b b' c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a' ≠ 0)
  (h₃ : (c - b) / a > (c - b') / a') :
  (c - b') / a' < (c - b) / a :=
by
  sorry

end solution_inequality_l107_107354


namespace average_mark_is_correct_l107_107680

-- Define the maximum score in the exam
def max_score := 1100

-- Define the percentages scored by Amar, Bhavan, Chetan, and Deepak
def score_percentage_amar := 64 / 100
def score_percentage_bhavan := 36 / 100
def score_percentage_chetan := 44 / 100
def score_percentage_deepak := 52 / 100

-- Calculate the actual scores based on percentages
def score_amar := score_percentage_amar * max_score
def score_bhavan := score_percentage_bhavan * max_score
def score_chetan := score_percentage_chetan * max_score
def score_deepak := score_percentage_deepak * max_score

-- Define the total score
def total_score := score_amar + score_bhavan + score_chetan + score_deepak

-- Define the number of students
def number_of_students := 4

-- Define the average score
def average_score := total_score / number_of_students

-- The theorem to prove that the average score is 539
theorem average_mark_is_correct : average_score = 539 := by
  -- Proof skipped
  sorry

end average_mark_is_correct_l107_107680


namespace find_unique_p_l107_107998

theorem find_unique_p (p : ℝ) (h1 : p ≠ 0) : (∀ x : ℝ, p * x^2 - 10 * x + 2 = 0 → p = 12.5) :=
by sorry

end find_unique_p_l107_107998


namespace angle_in_third_quadrant_l107_107197

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α * Real.cos α > 0) (h2 : Real.sin α * Real.tan α < 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end angle_in_third_quadrant_l107_107197


namespace no_solution_range_of_a_l107_107859

noncomputable def range_of_a : Set ℝ := {a | ∀ x : ℝ, ¬(abs (x - 1) + abs (x - 2) ≤ a^2 + a + 1)}

theorem no_solution_range_of_a :
  range_of_a = {a | -1 < a ∧ a < 0} :=
by
  sorry

end no_solution_range_of_a_l107_107859


namespace geometric_seq_arith_mean_l107_107004

theorem geometric_seq_arith_mean 
  (b : ℕ → ℝ) 
  (r : ℝ) 
  (b_geom : ∀ n, b (n + 1) = r * b n)
  (h_arith_mean : b 9 = (3 + 5) / 2) :
  b 1 * b 17 = 16 :=
by
  sorry

end geometric_seq_arith_mean_l107_107004


namespace incorrect_statement_l107_107984

theorem incorrect_statement (p q : Prop) (hp : ¬ p) (hq : q) : ¬ (¬ q) :=
by
  sorry

end incorrect_statement_l107_107984


namespace olympiad_divisors_l107_107696

theorem olympiad_divisors :
  {n : ℕ | n > 0 ∧ n ∣ (1998 + n)} = {n : ℕ | n > 0 ∧ n ∣ 1998} :=
by {
  sorry
}

end olympiad_divisors_l107_107696


namespace parabola_min_value_incorrect_statement_l107_107596

theorem parabola_min_value_incorrect_statement
  (m : ℝ)
  (A B : ℝ × ℝ)
  (P Q : ℝ × ℝ)
  (parabola : ℝ → ℝ)
  (on_parabola : ∀ (x : ℝ), parabola x = x^2 - 2*m*x + m^2 - 9)
  (A_intersects_x_axis : A.2 = 0)
  (B_intersects_x_axis : B.2 = 0)
  (A_on_parabola : parabola A.1 = A.2)
  (B_on_parabola : parabola B.1 = B.2)
  (P_on_parabola : parabola P.1 = P.2)
  (Q_on_parabola : parabola Q.1 = Q.2)
  (P_coordinates : P = (m + 1, parabola (m + 1)))
  (Q_coordinates : Q = (m - 3, parabola (m - 3))) :
  ∃ (min_y : ℝ), min_y = -9 ∧ min_y ≠ m^2 - 9 := 
sorry

end parabola_min_value_incorrect_statement_l107_107596


namespace permutation_equals_power_l107_107531

-- Definition of permutation with repetition
def permutation_with_repetition (n k : ℕ) : ℕ := n ^ k

-- Theorem to prove
theorem permutation_equals_power (n k : ℕ) : permutation_with_repetition n k = n ^ k :=
by
  sorry

end permutation_equals_power_l107_107531


namespace relationship_between_a_and_b_l107_107257

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1

-- Given conditions
variables (b a : ℝ)
variables (hx : 0 < b) (ha : 0 < a)
variables (x : ℝ) (hb : |x - 1| < b) (hf : |f x - 4| < a)

-- The theorem statement
theorem relationship_between_a_and_b
  (hf_x : ∀ x : ℝ, |x - 1| < b -> |f x - 4| < a) :
  a - 3 * b ≥ 0 :=
sorry

end relationship_between_a_and_b_l107_107257


namespace rhombus_perimeter_is_80_l107_107280

-- Definitions of the conditions
def rhombus_diagonals_ratio : Prop := ∃ (d1 d2 : ℝ), d1 / d2 = 3 / 4 ∧ d1 + d2 = 56

-- The goal is to prove that given the conditions, the perimeter of the rhombus is 80
theorem rhombus_perimeter_is_80 (h : rhombus_diagonals_ratio) : ∃ (p : ℝ), p = 80 :=
by
  sorry  -- The actual proof steps would go here

end rhombus_perimeter_is_80_l107_107280


namespace intersection_of_sets_l107_107532

-- Definitions from the conditions.
def A := { x : ℝ | x^2 - 2 * x ≤ 0 }
def B := { x : ℝ | x > 1 }

-- The proof problem statement.
theorem intersection_of_sets :
  A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } :=
sorry

end intersection_of_sets_l107_107532


namespace expression_inside_absolute_value_l107_107459

theorem expression_inside_absolute_value (E : ℤ) (x : ℤ) (h1 : x = 10) (h2 : 30 - |E| = 26) :
  E = 4 ∨ E = -4 := 
by
  sorry

end expression_inside_absolute_value_l107_107459


namespace smallest_N_l107_107838

theorem smallest_N (l m n : ℕ) (N : ℕ) (h1 : N = l * m * n) (h2 : (l - 1) * (m - 1) * (n - 1) = 300) : 
  N = 462 :=
sorry

end smallest_N_l107_107838


namespace find_omega_l107_107515

theorem find_omega (ω : Real) (h : ∀ x : Real, (1 / 2) * Real.cos (ω * x - (Real.pi / 6)) = (1 / 2) * Real.cos (ω * (x + Real.pi) - (Real.pi / 6))) : ω = 2 ∨ ω = -2 :=
by
  sorry

end find_omega_l107_107515


namespace algebraic_expression_value_l107_107763

theorem algebraic_expression_value (x : ℝ) (hx : x = Real.sqrt 7 + 1) :
  (x^2 / (x - 3) - 2 * x / (x - 3)) / (x / (x - 3)) = Real.sqrt 7 - 1 :=
by
  sorry

end algebraic_expression_value_l107_107763


namespace lines_intersect_l107_107353

theorem lines_intersect (m : ℝ) : ∃ (x y : ℝ), 3 * x + 2 * y + m = 0 ∧ (m^2 + 1) * x - 3 * y - 3 * m = 0 := 
by {
  sorry
}

end lines_intersect_l107_107353


namespace find_difference_square_l107_107755

theorem find_difference_square (x y c b : ℝ) (h1 : x * y = c^2) (h2 : (1 / x^2) + (1 / y^2) = b * c) : 
  (x - y)^2 = b * c^4 - 2 * c^2 := 
by sorry

end find_difference_square_l107_107755


namespace a_18_value_l107_107958

variable (a : ℕ → ℚ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 2
axiom a_rec (n : ℕ) (hn : 2 ≤ n) : 2 * n * a n = (n - 1) * a (n - 1) + (n + 1) * a (n + 1)

theorem a_18_value : a 18 = 26 / 9 :=
sorry

end a_18_value_l107_107958


namespace initial_men_count_l107_107936

theorem initial_men_count (M : ℕ) (P : ℝ) 
  (h1 : P = M * 12) 
  (h2 : P = (M + 300) * 9.662337662337663) :
  M = 1240 :=
sorry

end initial_men_count_l107_107936


namespace pizza_cost_difference_l107_107003

theorem pizza_cost_difference :
  let p := 12 -- Cost of plain pizza
  let m := 3 -- Cost of mushrooms
  let o := 4 -- Cost of olives
  let s := 12 -- Total number of slices
  (m + o + p) / s * 10 - (m + o + p) / s * 2 = 12.67 :=
by
  sorry

end pizza_cost_difference_l107_107003


namespace work_completion_l107_107470

theorem work_completion (a b : ℕ) (hab : a = 2 * b) (hwork_together : (1/a + 1/b) = 1/8) : b = 24 := by
  sorry

end work_completion_l107_107470


namespace contrapositive_example_l107_107264

theorem contrapositive_example (x : ℝ) :
  (x < -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
sorry

end contrapositive_example_l107_107264


namespace store_A_more_advantageous_l107_107517

theorem store_A_more_advantageous (x : ℕ) (h : x > 5) : 
  6000 + 4500 * (x - 1) < 4800 * x := 
by 
  sorry

end store_A_more_advantageous_l107_107517


namespace sqrt_product_simplification_l107_107806

-- Define the main problem
theorem sqrt_product_simplification : Real.sqrt 18 * Real.sqrt 72 = 36 := 
by
  sorry

end sqrt_product_simplification_l107_107806


namespace simplified_expression_value_at_4_l107_107391

theorem simplified_expression (x : ℝ) (h : x ≠ 5) : (x^2 - 3*x - 10) / (x - 5) = x + 2 := 
sorry

theorem value_at_4 : (4 : ℝ)^2 - 3*4 - 10 / (4 - 5) = 6 := 
sorry

end simplified_expression_value_at_4_l107_107391


namespace simplify_expression_l107_107164

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) : 
  (2 / y^2 - y⁻¹) = (2 - y) / y^2 :=
by sorry

end simplify_expression_l107_107164


namespace added_amount_correct_l107_107138

theorem added_amount_correct (n x : ℕ) (h1 : n = 20) (h2 : 1/2 * n + x = 15) :
  x = 5 :=
by
  sorry

end added_amount_correct_l107_107138


namespace gcd_max_value_l107_107548

theorem gcd_max_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) : 
  ∃ d, d = Nat.gcd a b ∧ d = 504 :=
by
  sorry

end gcd_max_value_l107_107548


namespace probability_age_less_than_20_l107_107758

theorem probability_age_less_than_20 (total : ℕ) (ages_gt_30 : ℕ) (ages_lt_20 : ℕ) 
    (h1 : total = 150) (h2 : ages_gt_30 = 90) (h3 : ages_lt_20 = total - ages_gt_30) :
    (ages_lt_20 : ℚ) / total = 2 / 5 :=
by
  simp [h1, h2, h3]
  sorry

end probability_age_less_than_20_l107_107758


namespace number_of_diagonals_in_decagon_l107_107050

-- Definition of the problem condition: a polygon with n = 10 sides
def n : ℕ := 10

-- Theorem stating the number of diagonals in a regular decagon
theorem number_of_diagonals_in_decagon : (n * (n - 3)) / 2 = 35 :=
by
  -- Proof steps will go here
  sorry

end number_of_diagonals_in_decagon_l107_107050


namespace tangent_line_eqn_l107_107543

noncomputable def f (x : ℝ) : ℝ := 5 * x + Real.log x

theorem tangent_line_eqn : ∀ x y : ℝ, (x, y) = (1, f 1) → 6 * x - y - 1 = 0 := 
by
  intro x y h
  sorry

end tangent_line_eqn_l107_107543


namespace fraction_eq_zero_implies_x_eq_one_l107_107084

theorem fraction_eq_zero_implies_x_eq_one (x : ℝ) (h1 : (x - 1) = 0) (h2 : (x - 5) ≠ 0) : x = 1 :=
sorry

end fraction_eq_zero_implies_x_eq_one_l107_107084


namespace cost_of_other_disc_l107_107650

theorem cost_of_other_disc (x : ℝ) (total_spent : ℝ) (num_discs : ℕ) (num_850_discs : ℕ) (price_850 : ℝ) 
    (total_cost : total_spent = 93) (num_bought : num_discs = 10) (num_850 : num_850_discs = 6) (price_per_850 : price_850 = 8.50) 
    (total_cost_850 : num_850_discs * price_850 = 51) (remaining_discs_cost : total_spent - 51 = 42) (remaining_discs : num_discs - num_850_discs = 4) :
    total_spent = num_850_discs * price_850 + (num_discs - num_850_discs) * x → x = 10.50 :=
by
  sorry

end cost_of_other_disc_l107_107650


namespace aquarium_final_volume_l107_107503

theorem aquarium_final_volume :
  let length := 4
  let width := 6
  let height := 3
  let total_volume := length * width * height
  let initial_volume := total_volume / 2
  let spilled_volume := initial_volume / 2
  let remaining_volume := initial_volume - spilled_volume
  let final_volume := remaining_volume * 3
  final_volume = 54 :=
by sorry

end aquarium_final_volume_l107_107503


namespace interest_difference_l107_107375

theorem interest_difference (P R T : ℝ) (SI : ℝ) (Diff : ℝ) :
  P = 250 ∧ R = 4 ∧ T = 8 ∧ SI = (P * R * T) / 100 ∧ Diff = P - SI → Diff = 170 :=
by sorry

end interest_difference_l107_107375


namespace discount_calculation_l107_107271

-- Definitions based on the given conditions
def cost_magazine : Float := 0.85
def cost_pencil : Float := 0.50
def amount_spent : Float := 1.00

-- Define the total cost before discount
def total_cost_before_discount : Float := cost_magazine + cost_pencil

-- Goal: Prove that the discount is $0.35
theorem discount_calculation : total_cost_before_discount - amount_spent = 0.35 := by
  -- Proof (to be filled in later)
  sorry

end discount_calculation_l107_107271


namespace man_l107_107471

-- Define the man's rowing speed in still water, the speed of the current, the downstream speed and headwind reduction.
def v : Real := 17.5
def speed_current : Real := 4.5
def speed_downstream : Real := 22
def headwind_reduction : Real := 1.5

-- Define the man's speed against the current and headwind.
def speed_against_current_headwind := v - speed_current - headwind_reduction

-- The statement to prove. 
theorem man's_speed_against_current_and_headwind :
  speed_against_current_headwind = 11.5 := by
  -- Using the conditions (which are already defined in lean expressions above), we can end the proof here.
  sorry

end man_l107_107471


namespace range_of_m_l107_107957

variable (m : ℝ) -- variable m in the real numbers

-- Definition of proposition p
def p : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

-- Definition of proposition q
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- The theorem statement with the given conditions
theorem range_of_m (h : p m ∧ q m) : -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l107_107957


namespace salon_fingers_l107_107676

theorem salon_fingers (clients non_clients total_fingers cost_per_client total_earnings : Nat)
  (h1 : cost_per_client = 20)
  (h2 : total_earnings = 200)
  (h3 : total_fingers = 210)
  (h4 : non_clients = 11)
  (h_clients : clients = total_earnings / cost_per_client)
  (h_people : total_fingers / 10 = clients + non_clients) :
  10 = total_fingers / (clients + non_clients) :=
by
  sorry

end salon_fingers_l107_107676


namespace letters_into_mailboxes_l107_107999

theorem letters_into_mailboxes (n m : ℕ) (h1 : n = 3) (h2 : m = 5) : m^n = 125 :=
by
  rw [h1, h2]
  exact rfl

end letters_into_mailboxes_l107_107999


namespace power_equality_l107_107248

theorem power_equality (p : ℕ) : 16^10 = 4^p → p = 20 :=
by
  intro h
  -- proof goes here
  sorry

end power_equality_l107_107248


namespace color_property_l107_107843

theorem color_property (k : ℕ) (h : k ≥ 1) : k = 1 ∨ k = 2 :=
by
  sorry

end color_property_l107_107843


namespace insert_arithmetic_sequence_l107_107871

theorem insert_arithmetic_sequence (d a b : ℤ) 
  (h1 : (-1) + 3 * d = 8) 
  (h2 : a = (-1) + d) 
  (h3 : b = a + d) : 
  a = 2 ∧ b = 5 := by
  sorry

end insert_arithmetic_sequence_l107_107871


namespace mac_runs_faster_than_apple_l107_107017

theorem mac_runs_faster_than_apple :
  let Apple_speed := 3 -- miles per hour
  let Mac_speed := 4 -- miles per hour
  let Distance := 24 -- miles
  let Apple_time := Distance / Apple_speed -- hours
  let Mac_time := Distance / Mac_speed -- hours
  let Time_difference := (Apple_time - Mac_time) * 60 -- converting hours to minutes
  Time_difference = 120 := by
  sorry

end mac_runs_faster_than_apple_l107_107017


namespace standard_equation_of_parabola_l107_107358

theorem standard_equation_of_parabola (x : ℝ) (y : ℝ) (directrix : ℝ) (eq_directrix : directrix = 1) :
  y^2 = -4 * x :=
sorry

end standard_equation_of_parabola_l107_107358


namespace geometric_sequence_common_ratio_l107_107335

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : 2 * a 0 + a 1 = a 2)
  : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l107_107335


namespace sum_of_squares_l107_107418

theorem sum_of_squares (n : ℕ) (x : ℕ) (h1 : (x + 1)^3 - x^3 = n^2) (h2 : n > 0) : ∃ a b : ℕ, n = a^2 + b^2 :=
by
  sorry

end sum_of_squares_l107_107418


namespace product_three_power_l107_107652

theorem product_three_power (w : ℕ) (hW : w = 132) (hProd : ∃ (k : ℕ), 936 * w = 2^5 * 11^2 * k) : 
  ∃ (n : ℕ), (936 * w) = (2^5 * 11^2 * (3^3 * n)) :=
by 
  sorry

end product_three_power_l107_107652


namespace product_value_l107_107046

theorem product_value (x : ℝ) (h : (Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8)) : (6 + x) * (21 - x) = 1369 / 4 :=
by
  sorry

end product_value_l107_107046


namespace bin_sum_sub_eq_l107_107163

-- Define binary numbers
def b1 := 0b101110  -- binary 101110_2
def b2 := 0b10101   -- binary 10101_2
def b3 := 0b111000  -- binary 111000_2
def b4 := 0b110101  -- binary 110101_2
def b5 := 0b11101   -- binary 11101_2

-- Define the theorem
theorem bin_sum_sub_eq : ((b1 + b2) - (b3 - b4) + b5) = 0b1011101 := by
  sorry

end bin_sum_sub_eq_l107_107163


namespace movies_left_to_watch_l107_107407

theorem movies_left_to_watch (total_movies watched_movies : Nat) (h_total : total_movies = 12) (h_watched : watched_movies = 6) : total_movies - watched_movies = 6 :=
by
  sorry

end movies_left_to_watch_l107_107407


namespace combined_tax_rate_is_correct_l107_107384

noncomputable def combined_tax_rate (john_income : ℝ) (ingrid_income : ℝ) (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) : ℝ :=
  let john_tax := john_tax_rate * john_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let total_tax := john_tax + ingrid_tax
  let total_income := john_income + ingrid_income
  total_tax / total_income

theorem combined_tax_rate_is_correct :
  combined_tax_rate 56000 72000 0.30 0.40 = 0.35625 := 
by
  sorry

end combined_tax_rate_is_correct_l107_107384


namespace exponent_identity_l107_107299

theorem exponent_identity (m : ℕ) : 5 ^ m = 5 * (25 ^ 4) * (625 ^ 3) ↔ m = 21 := by
  sorry

end exponent_identity_l107_107299


namespace max_value_of_expression_l107_107864

-- Define the variables and constraints
variables {a b c d : ℤ}
variables (S : finset ℤ) (a_val b_val c_val d_val : ℤ)

axiom h1 : S = {0, 1, 2, 4, 5}
axiom h2 : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S
axiom h3 : ∀ x ∈ S, x = a → x ≠ b ∧ x ≠ c ∧ x ≠ d
axiom h4 : ∀ x ∈ S, x = b → x ≠ a ∧ x ≠ c ∧ x ≠ d
axiom h5 : ∀ x ∈ S, x = c → x ≠ a ∧ x ≠ b ∧ x ≠ d
axiom h6 : ∀ x ∈ S, x = d → x ≠ a ∧ x ≠ b ∧ x ≠ c

-- The main theorem to be proven
theorem max_value_of_expression : (∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
  (∀ x ∈ S, (x = a → x ≠ b ∧ x ≠ c ∧ x ≠ d) ∧ 
             (x = b → x ≠ a ∧ x ≠ c ∧ x ≠ d) ∧ 
             (x = c → x ≠ a ∧ x ≠ b ∧ x ≠ d) ∧ 
             (x = d → x ≠ a ∧ x ≠ b ∧ x ≠ c)) ∧
  (c * a^b - d = 20)) :=
sorry

end max_value_of_expression_l107_107864


namespace total_tweets_l107_107873

-- Conditions and Definitions
def tweets_happy_per_minute := 18
def tweets_hungry_per_minute := 4
def tweets_reflection_per_minute := 45
def minutes_each_period := 20

-- Proof Problem Statement
theorem total_tweets : 
  (minutes_each_period * tweets_happy_per_minute) + 
  (minutes_each_period * tweets_hungry_per_minute) + 
  (minutes_each_period * tweets_reflection_per_minute) = 1340 :=
by
  sorry

end total_tweets_l107_107873


namespace prop_logic_example_l107_107828

theorem prop_logic_example (p q : Prop) (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
by {
  sorry
}

end prop_logic_example_l107_107828


namespace rainfall_on_tuesday_is_correct_l107_107627

-- Define the total days in a week
def days_in_week : ℕ := 7

-- Define the average rainfall for the whole week
def avg_rainfall : ℝ := 3.0

-- Define the total rainfall for the week
def total_rainfall : ℝ := avg_rainfall * days_in_week

-- Define a proposition that states rainfall on Tuesday equals 10.5 cm
def rainfall_on_tuesday (T : ℝ) : Prop :=
  T = 10.5

-- Prove that the rainfall on Tuesday is 10.5 cm given the conditions
theorem rainfall_on_tuesday_is_correct : rainfall_on_tuesday (total_rainfall / 2) :=
by
  sorry

end rainfall_on_tuesday_is_correct_l107_107627


namespace find_f_5_l107_107853

theorem find_f_5 : 
  ∀ (f : ℝ → ℝ) (y : ℝ), 
  (∀ x, f x = 2 * x ^ 2 + y) ∧ f 2 = 60 -> f 5 = 102 :=
by
  sorry

end find_f_5_l107_107853


namespace intersection_S_T_eq_T_l107_107126

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l107_107126


namespace solve_thought_of_number_l107_107697

def thought_of_number (x : ℝ) : Prop :=
  (x / 6) + 5 = 17

theorem solve_thought_of_number :
  ∃ x, thought_of_number x ∧ x = 72 :=
by
  sorry

end solve_thought_of_number_l107_107697


namespace students_juice_count_l107_107653

theorem students_juice_count (students chose_water chose_juice : ℕ) 
  (h1 : chose_water = 140) 
  (h2 : (25 : ℚ) / 100 * (students : ℚ) = chose_juice)
  (h3 : (70 : ℚ) / 100 * (students : ℚ) = chose_water) : 
  chose_juice = 50 :=
by 
  sorry

end students_juice_count_l107_107653


namespace find_digits_sum_l107_107557

theorem find_digits_sum (a b c : Nat) (ha : 0 <= a ∧ a <= 9) (hb : 0 <= b ∧ b <= 9) 
  (hc : 0 <= c ∧ c <= 9) 
  (h1 : 2 * a = c) 
  (h2 : b = b) : 
  a + b + c = 11 :=
  sorry

end find_digits_sum_l107_107557


namespace tan_of_sin_in_interval_l107_107345

theorem tan_of_sin_in_interval (α : ℝ) (h1 : Real.sin α = 4 / 5) (h2 : 0 < α ∧ α < Real.pi) :
  Real.tan α = 4 / 3 ∨ Real.tan α = -4 / 3 :=
  sorry

end tan_of_sin_in_interval_l107_107345


namespace adela_numbers_l107_107974

theorem adela_numbers (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a - b)^2 = a^2 - b^2 - 4038) :
  (a = 2020 ∧ b = 1) ∨ (a = 2020 ∧ b = 2019) ∨ (a = 676 ∧ b = 3) ∨ (a = 676 ∧ b = 673) :=
sorry

end adela_numbers_l107_107974


namespace first_to_receive_10_pieces_l107_107953

-- Definitions and conditions
def children := [1, 2, 3, 4, 5, 6, 7, 8]
def distribution_cycle := [1, 3, 6, 8, 3, 5, 8, 2, 5, 7, 2, 4, 7, 1, 4, 6]

def count_occurrences (n : ℕ) (lst : List ℕ) : ℕ :=
  lst.count n

-- Theorem
theorem first_to_receive_10_pieces : ∃ k, k = 3 ∧ count_occurrences k distribution_cycle = 2 :=
by
  sorry

end first_to_receive_10_pieces_l107_107953


namespace new_oranges_added_l107_107487

def initial_oranges : Nat := 31
def thrown_away_oranges : Nat := 9
def final_oranges : Nat := 60
def remaining_oranges : Nat := initial_oranges - thrown_away_oranges
def new_oranges (initial_oranges thrown_away_oranges final_oranges : Nat) : Nat := 
  final_oranges - (initial_oranges - thrown_away_oranges)

theorem new_oranges_added :
  new_oranges initial_oranges thrown_away_oranges final_oranges = 38 := by
  sorry

end new_oranges_added_l107_107487


namespace regular_polygon_sides_l107_107575

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l107_107575


namespace bowling_ball_weight_l107_107101

theorem bowling_ball_weight (b k : ℝ) (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 :=
by
  sorry

end bowling_ball_weight_l107_107101


namespace checkered_fabric_cost_l107_107284

variable (P : ℝ) (cost_per_yard : ℝ) (total_yards : ℕ)
variable (x : ℝ) (C : ℝ)

theorem checkered_fabric_cost :
  P = 45 ∧ cost_per_yard = 7.50 ∧ total_yards = 16 →
  C = cost_per_yard * (total_yards - x) →
  7.50 * (16 - x) = 45 →
  C = 75 :=
by
  intro h1 h2 h3
  sorry

end checkered_fabric_cost_l107_107284


namespace Karlson_max_candies_l107_107595

theorem Karlson_max_candies (f : Fin 25 → ℕ) (g : Fin 25 → Fin 25 → ℕ) :
  (∀ i, f i = 1) →
  (∀ i j, g i j = f i * f j) →
  (∃ (S : ℕ), S = 300) :=
by
  intros h1 h2
  sorry

end Karlson_max_candies_l107_107595


namespace number_of_foals_l107_107539

theorem number_of_foals (t f : ℕ) (h1 : t + f = 11) (h2 : 2 * t + 4 * f = 30) : f = 4 :=
by
  sorry

end number_of_foals_l107_107539


namespace complement_intersection_l107_107216

theorem complement_intersection (A B U : Set ℕ) (hA : A = {4, 5, 7}) (hB : B = {3, 4, 7, 8}) (hU : U = A ∪ B) :
  U \ (A ∩ B) = {3, 5, 8} :=
by
  sorry

end complement_intersection_l107_107216


namespace maximize_expression_l107_107594

theorem maximize_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 29 :=
by
  sorry

end maximize_expression_l107_107594


namespace range_of_x_l107_107610

theorem range_of_x (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
  sorry

end range_of_x_l107_107610


namespace max_difference_l107_107151

theorem max_difference (U V W X Y Z : ℕ) (hUVW : U ≠ V ∧ V ≠ W ∧ U ≠ W)
    (hXYZ : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) (digits_UVW : 1 ≤ U ∧ U ≤ 9 ∧ 1 ≤ V ∧ V ≤ 9 ∧ 1 ≤ W ∧ W ≤ 9)
    (digits_XYZ : 1 ≤ X ∧ X ≤ 9 ∧ 1 ≤ Y ∧ Y ≤ 9 ∧ 1 ≤ Z ∧ Z ≤ 9) :
    U * 100 + V * 10 + W = 987 → X * 100 + Y * 10 + Z = 123 → (U * 100 + V * 10 + W) - (X * 100 + Y * 10 + Z) = 864 :=
by
  sorry

end max_difference_l107_107151


namespace f_2019_is_zero_l107_107035

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_non_negative
  (x : ℝ) : 0 ≤ f x

axiom f_satisfies_condition
  (a b c : ℝ) : f (a^3) + f (b^3) + f (c^3) = 3 * f a * f b * f c

axiom f_one_not_one : f 1 ≠ 1

theorem f_2019_is_zero : f 2019 = 0 := 
  sorry

end f_2019_is_zero_l107_107035


namespace find_a_values_l107_107008

def setA : Set ℝ := {-1, 1/2, 1}
def setB (a : ℝ) : Set ℝ := {x | a * x^2 = 1 ∧ a ≥ 0}

def full_food (A B : Set ℝ) : Prop := A ⊆ B ∨ B ⊆ A
def partial_food (A B : Set ℝ) : Prop := (∃ x, x ∈ A ∧ x ∈ B) ∧ ¬(A ⊆ B ∨ B ⊆ A)

theorem find_a_values :
  ∀ a : ℝ, full_food setA (setB a) ∨ partial_food setA (setB a) ↔ a = 0 ∨ a = 1 ∨ a = 4 := 
by
  sorry

end find_a_values_l107_107008


namespace total_handshakes_is_72_l107_107705

-- Define the conditions
def number_of_players_per_team := 6
def number_of_teams := 2
def number_of_referees := 3

-- Define the total number of players
def total_players := number_of_teams * number_of_players_per_team

-- Define the total number of handshakes between players of different teams
def team_handshakes := number_of_players_per_team * number_of_players_per_team

-- Define the total number of handshakes between players and referees
def player_referee_handshakes := total_players * number_of_referees

-- Define the total number of handshakes
def total_handshakes := team_handshakes + player_referee_handshakes

-- Prove that the total number of handshakes is 72
theorem total_handshakes_is_72 : total_handshakes = 72 := by
  sorry

end total_handshakes_is_72_l107_107705


namespace sally_paid_peaches_l107_107334

def total_spent : ℝ := 23.86
def amount_spent_on_cherries : ℝ := 11.54
def amount_spent_on_peaches_after_coupon : ℝ := total_spent - amount_spent_on_cherries

theorem sally_paid_peaches : amount_spent_on_peaches_after_coupon = 12.32 :=
by 
  -- The actual proof will involve concrete calculation here.
  -- For now, we skip it with sorry.
  sorry

end sally_paid_peaches_l107_107334


namespace solve_inequality_2_star_x_l107_107811

theorem solve_inequality_2_star_x :
  ∀ x : ℝ, 
  6 < (2 * x - 2 - x + 3) ∧ (2 * x - 2 - x + 3) < 7 ↔ 5 < x ∧ x < 6 :=
by sorry

end solve_inequality_2_star_x_l107_107811


namespace price_reduction_l107_107218

theorem price_reduction (p0 p1 p2 : ℝ) (H0 : p0 = 1) (H1 : p1 = 1.25 * p0) (H2 : p2 = 1.1 * p0) :
  ∃ x : ℝ, p2 = p1 * (1 - x / 100) ∧ x = 12 :=
  sorry

end price_reduction_l107_107218


namespace mean_of_six_numbers_l107_107509

theorem mean_of_six_numbers (sum_of_six: ℚ) (H: sum_of_six = 3 / 4) : sum_of_six / 6 = 1 / 8 := by
  sorry

end mean_of_six_numbers_l107_107509


namespace probability_heads_9_tails_at_least_2_l107_107672

noncomputable def probability_exactly_nine_heads : ℚ :=
  let total_outcomes := 2 ^ 12
  let successful_outcomes := Nat.choose 12 9
  successful_outcomes / total_outcomes

theorem probability_heads_9_tails_at_least_2 (n : ℕ) (h : n = 12) :
  n = 12 → probability_exactly_nine_heads = 55 / 1024 := by
  intros h
  sorry

end probability_heads_9_tails_at_least_2_l107_107672


namespace number_is_280_l107_107756

theorem number_is_280 (x : ℝ) (h : x / 5 + 4 = x / 4 - 10) : x = 280 := 
by 
  sorry

end number_is_280_l107_107756


namespace determine_k_l107_107660

variable (x y z k : ℝ)

theorem determine_k (h1 : 7 / (x + y) = k / (x + z)) (h2 : k / (x + z) = 11 / (z - y)) : k = 18 := 
by 
  sorry

end determine_k_l107_107660


namespace determine_a_l107_107103

theorem determine_a (r s a : ℝ) (h1 : r^2 = a) (h2 : 2 * r * s = 16) (h3 : s^2 = 16) : a = 4 :=
by {
  sorry
}

end determine_a_l107_107103


namespace set_equality_l107_107576

open Set

namespace Proof

variables (U M N : Set ℕ) 
variables (U_univ : U = {1, 2, 3, 4, 5, 6})
variables (M_set : M = {2, 3})
variables (N_set : N = {1, 3})

theorem set_equality :
  {4, 5, 6} = (U \ M) ∩ (U \ N) :=
by
  rw [U_univ, M_set, N_set]
  sorry

end Proof

end set_equality_l107_107576


namespace vec_eq_l107_107352

def a : ℝ × ℝ := (-1, 0)
def b : ℝ × ℝ := (0, 2)

theorem vec_eq : (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2) = (-2, -6) := by
  sorry

end vec_eq_l107_107352


namespace percent_more_proof_l107_107770

-- Define the conditions
def y := 150
def x := 120
def is_percent_more (y x p : ℕ) : Prop := y = (1 + p / 100) * x

-- The proof problem statement
theorem percent_more_proof : ∃ p : ℕ, is_percent_more y x p ∧ p = 25 := by
  sorry

end percent_more_proof_l107_107770


namespace cube_face_area_l107_107127

-- Definition for the condition of the cube's surface area
def cube_surface_area (s : ℝ) : Prop := s = 36

-- Definition stating a cube has 6 faces
def cube_faces : ℝ := 6

-- The target proposition to prove
theorem cube_face_area (s : ℝ) (area_of_one_face : ℝ) (h1 : cube_surface_area s) (h2 : cube_faces = 6) : area_of_one_face = s / 6 :=
by
  sorry

end cube_face_area_l107_107127


namespace correct_weight_misread_l107_107687

theorem correct_weight_misread (initial_avg correct_avg : ℝ) (num_boys : ℕ) (misread_weight : ℝ)
  (h_initial : initial_avg = 58.4) (h_correct : correct_avg = 58.85) (h_num_boys : num_boys = 20)
  (h_misread_weight : misread_weight = 56) :
  ∃ x : ℝ, x = 65 :=
by
  sorry

end correct_weight_misread_l107_107687


namespace probability_collinear_dots_l107_107990

theorem probability_collinear_dots 
  (rows : ℕ) (cols : ℕ) (total_dots : ℕ) (collinear_sets : ℕ) (total_ways : ℕ) : 
  rows = 5 → cols = 4 → total_dots = 20 → collinear_sets = 20 → total_ways = 4845 → 
  (collinear_sets : ℚ) / total_ways = 4 / 969 :=
by
  intros hrows hcols htotal_dots hcollinear_sets htotal_ways
  sorry

end probability_collinear_dots_l107_107990


namespace new_mean_after_adding_eleven_l107_107754

theorem new_mean_after_adding_eleven (nums : List ℝ) (h_len : nums.length = 15) (h_avg : (nums.sum / 15) = 40) :
  ((nums.map (λ x => x + 11)).sum / 15) = 51 := by
  sorry

end new_mean_after_adding_eleven_l107_107754


namespace remainder_gx12_div_gx_l107_107414

-- Definition of the polynomial g(x)
def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Theorem stating the problem
theorem remainder_gx12_div_gx : ∀ x : ℂ, (g (x^12)) % (g x) = 6 := by
  sorry

end remainder_gx12_div_gx_l107_107414


namespace smallest_n_l107_107820

-- Define the costs.
def cost_red := 10 * 8  -- = 80
def cost_green := 18 * 12  -- = 216
def cost_blue := 20 * 15  -- = 300
def cost_yellow (n : Nat) := 24 * n

-- Define the LCM of the costs.
def LCM_cost : Nat := Nat.lcm (Nat.lcm cost_red cost_green) cost_blue

-- Problem statement: Prove that the smallest value of n such that 24 * n is the LCM of the candy costs is 150.
theorem smallest_n : ∃ n : Nat, cost_yellow n = LCM_cost ∧ n = 150 := 
by {
  -- This part is just a placeholder; the proof steps are omitted.
  sorry
}

end smallest_n_l107_107820


namespace units_digit_24_pow_4_plus_42_pow_4_l107_107445

theorem units_digit_24_pow_4_plus_42_pow_4 : 
    (24^4 + 42^4) % 10 = 2 :=
by
  sorry

end units_digit_24_pow_4_plus_42_pow_4_l107_107445


namespace find_g3_l107_107033

noncomputable def g : ℝ → ℝ := sorry

theorem find_g3 (h : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = x) : g 3 = 1 :=
sorry

end find_g3_l107_107033


namespace project_completion_time_l107_107230

theorem project_completion_time (rate_a rate_b rate_c : ℝ) (total_work : ℝ) (quit_time : ℝ) 
  (ha : rate_a = 1 / 20) 
  (hb : rate_b = 1 / 30) 
  (hc : rate_c = 1 / 40) 
  (htotal : total_work = 1)
  (hquit : quit_time = 18) : 
  ∃ T : ℝ, T = 18 :=
by {
  sorry
}

end project_completion_time_l107_107230


namespace sqrt_49_mul_sqrt_25_l107_107658

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end sqrt_49_mul_sqrt_25_l107_107658


namespace graph_not_in_first_quadrant_l107_107560

theorem graph_not_in_first_quadrant (a b : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) 
  (h_not_in_first_quadrant : ∀ x : ℝ, a^x + b - 1 ≤ 0) : 
  0 < a ∧ a < 1 ∧ b ≤ 0 :=
sorry

end graph_not_in_first_quadrant_l107_107560


namespace monster_perimeter_correct_l107_107764

noncomputable def monster_perimeter (radius : ℝ) (central_angle_missing : ℝ) : ℝ :=
  let full_circle_circumference := 2 * radius * Real.pi
  let arc_length := (1 - central_angle_missing / 360) * full_circle_circumference
  arc_length + 2 * radius

theorem monster_perimeter_correct :
  monster_perimeter 2 90 = 3 * Real.pi + 4 :=
by
  -- The proof would go here
  sorry

end monster_perimeter_correct_l107_107764


namespace lines_not_form_triangle_l107_107444

theorem lines_not_form_triangle {m : ℝ} :
  (∀ x y : ℝ, 2 * x - 3 * y + 1 ≠ 0 → 4 * x + 3 * y + 5 ≠ 0 → mx - y - 1 ≠ 0) →
  (m = -4 / 3 ∨ m = 2 / 3 ∨ m = 4 / 3) :=
sorry

end lines_not_form_triangle_l107_107444


namespace min_w_value_l107_107516

def w (x y : ℝ) : ℝ := 3 * x^2 + 5 * y^2 + 12 * x - 10 * y + 45

theorem min_w_value : ∀ x y : ℝ, (w x y) ≥ 28 ∧ (∃ x y : ℝ, (w x y) = 28) :=
by
  sorry

end min_w_value_l107_107516


namespace determineFinalCounts_l107_107699

structure FruitCounts where
  plums : ℕ
  oranges : ℕ
  apples : ℕ
  pears : ℕ
  cherries : ℕ

def initialCounts : FruitCounts :=
  { plums := 10, oranges := 8, apples := 12, pears := 6, cherries := 0 }

def givenAway : FruitCounts :=
  { plums := 4, oranges := 3, apples := 5, pears := 0, cherries := 0 }

def receivedFromSam : FruitCounts :=
  { plums := 2, oranges := 0, apples := 0, pears := 1, cherries := 0 }

def receivedFromBrother : FruitCounts :=
  { plums := 0, oranges := 1, apples := 2, pears := 0, cherries := 0 }

def receivedFromNeighbor : FruitCounts :=
  { plums := 0, oranges := 0, apples := 0, pears := 3, cherries := 2 }

def finalCounts (initial given receivedSam receivedBrother receivedNeighbor : FruitCounts) : FruitCounts :=
  { plums := initial.plums - given.plums + receivedSam.plums,
    oranges := initial.oranges - given.oranges + receivedBrother.oranges,
    apples := initial.apples - given.apples + receivedBrother.apples,
    pears := initial.pears - given.pears + receivedSam.pears + receivedNeighbor.pears,
    cherries := initial.cherries - given.cherries + receivedNeighbor.cherries }

theorem determineFinalCounts :
  finalCounts initialCounts givenAway receivedFromSam receivedFromBrother receivedFromNeighbor =
  { plums := 8, oranges := 6, apples := 9, pears := 10, cherries := 2 } :=
by
  sorry

end determineFinalCounts_l107_107699


namespace range_of_a_l107_107500

-- Definitions and theorems
theorem range_of_a (a : ℝ) : 
  (∀ (x y z : ℝ), x + y + z = 1 → abs (a - 2) ≤ x^2 + 2*y^2 + 3*z^2) → (16 / 11 ≤ a ∧ a ≤ 28 / 11) := 
by
  sorry

end range_of_a_l107_107500


namespace sequence_first_term_eq_three_l107_107968

theorem sequence_first_term_eq_three
  (a : ℕ → ℕ)
  (h_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h_nz : ∀ n : ℕ, 0 < a n)
  (h_a11 : a 11 = 157) :
  a 1 = 3 :=
sorry

end sequence_first_term_eq_three_l107_107968


namespace angle_C_in_triangle_l107_107153

theorem angle_C_in_triangle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A + B = 115) : C = 65 := 
by 
  sorry

end angle_C_in_triangle_l107_107153


namespace common_chord_length_of_two_circles_l107_107814

-- Define the equations of the circles C1 and C2
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 2 * y - 4 = 0
def circle2 (x y : ℝ) : Prop := (x + 3 / 2)^2 + (y - 3 / 2)^2 = 11 / 2

-- The theorem stating the length of the common chord
theorem common_chord_length_of_two_circles :
  ∃ l : ℝ, (∀ (x y : ℝ), circle1 x y ↔ circle2 x y) → l = 2 :=
by simp [circle1, circle2]; sorry

end common_chord_length_of_two_circles_l107_107814


namespace sequence_general_term_l107_107619

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, 2 * a n = 3 * a (n + 1)) ∧ 
  (a 2 * a 5 = 8 / 27) ∧ 
  (∀ n, 0 < a n)

theorem sequence_general_term (a : ℕ → ℝ) (h : sequence_condition a) : 
  ∀ n, a n = (2 / 3)^(n - 2) :=
by 
  sorry

end sequence_general_term_l107_107619


namespace find_uv_non_integer_l107_107413

noncomputable def q (x y : ℝ) (b : ℕ → ℝ) := 
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3

theorem find_uv_non_integer (b : ℕ → ℝ) 
  (h0 : q 0 0 b = 0) 
  (h1 : q 1 0 b = 0) 
  (h2 : q (-1) 0 b = 0) 
  (h3 : q 0 1 b = 0) 
  (h4 : q 0 (-1) b = 0) 
  (h5 : q 1 1 b = 0) 
  (h6 : q 1 (-1) b = 0) 
  (h7 : q 3 3 b = 0) : 
  ∃ u v : ℝ, q u v b = 0 ∧ u = 17/19 ∧ v = 18/19 := 
  sorry

end find_uv_non_integer_l107_107413


namespace new_student_weight_l107_107736

theorem new_student_weight (W_new : ℝ) (W : ℝ) (avg_decrease : ℝ) (num_students : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_decrease = 5 → old_weight = 86 → num_students = 8 →
  W_new = W - old_weight + new_weight → W_new = W - avg_decrease * num_students →
  new_weight = 46 :=
by
  intros avg_decrease_eq old_weight_eq num_students_eq W_new_eq avg_weight_decrease_eq
  rw [avg_decrease_eq, old_weight_eq, num_students_eq] at *
  sorry

end new_student_weight_l107_107736


namespace gcd_153_119_l107_107930

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_l107_107930


namespace circle_center_coordinates_l107_107513

theorem circle_center_coordinates :
  ∀ x y, (x^2 + y^2 - 4 * x - 2 * y - 5 = 0) → (x, y) = (2, 1) :=
by
  sorry

end circle_center_coordinates_l107_107513


namespace smallest_number_is_neg1_l107_107260

-- Defining the list of numbers
def numbers := [0, -1, 1, 2]

-- Theorem statement to prove that the smallest number in the list is -1
theorem smallest_number_is_neg1 :
  ∀ x ∈ numbers, x ≥ -1 := 
sorry

end smallest_number_is_neg1_l107_107260


namespace discount_percentage_l107_107981

theorem discount_percentage (original_price sale_price : ℕ) (h₁ : original_price = 1200) (h₂ : sale_price = 1020) : 
  ((original_price - sale_price) * 100 / original_price : ℝ) = 15 :=
by
  sorry

end discount_percentage_l107_107981


namespace onions_total_l107_107743

theorem onions_total (Sara : ℕ) (Sally : ℕ) (Fred : ℕ)
  (hSara : Sara = 4) (hSally : Sally = 5) (hFred : Fred = 9) :
  Sara + Sally + Fred = 18 :=
by
  sorry

end onions_total_l107_107743


namespace min_value_of_a_l107_107815

theorem min_value_of_a (a : ℝ) : (∀ x, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_value_of_a_l107_107815


namespace ratio_AB_AD_l107_107779

theorem ratio_AB_AD (a x y : ℝ) (h1 : 0.3 * a^2 = 0.7 * x * y) (h2 : y = a / 10) : x / y = 43 :=
by
  sorry

end ratio_AB_AD_l107_107779


namespace solve_system_I_solve_system_II_l107_107766

theorem solve_system_I (x y : ℝ) (h1 : y = x + 3) (h2 : x - 2 * y + 12 = 0) : x = 6 ∧ y = 9 :=
by
  sorry

theorem solve_system_II (x y : ℝ) (h1 : 4 * (x - y - 1) = 3 * (1 - y) - 2) (h2 : x / 2 + y / 3 = 2) : x = 2 ∧ y = 3 :=
by
  sorry

end solve_system_I_solve_system_II_l107_107766


namespace determine_b_l107_107239

def imaginary_unit : Type := {i : ℂ // i^2 = -1}

theorem determine_b (i : imaginary_unit) (b : ℝ) : 
  (2 - i.val) * 4 * i.val = 4 - b * i.val → b = -8 :=
by
  sorry

end determine_b_l107_107239


namespace mps_to_kmph_conversion_l107_107663

/-- Define the conversion factor from meters per second to kilometers per hour. -/
def mps_to_kmph : ℝ := 3.6

/-- Define the speed in meters per second. -/
def speed_mps : ℝ := 5

/-- Define the converted speed in kilometers per hour. -/
def speed_kmph : ℝ := 18

/-- Statement asserting the conversion from meters per second to kilometers per hour. -/
theorem mps_to_kmph_conversion : speed_mps * mps_to_kmph = speed_kmph := by 
  sorry

end mps_to_kmph_conversion_l107_107663


namespace min_value_of_m_l107_107993

theorem min_value_of_m (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  a^2 + b^2 + c^2 ≥ 3 :=
sorry

end min_value_of_m_l107_107993


namespace copper_to_zinc_ratio_l107_107215

theorem copper_to_zinc_ratio (total_weight_brass : ℝ) (weight_zinc : ℝ) (weight_copper : ℝ) 
  (h1 : total_weight_brass = 100) (h2 : weight_zinc = 70) (h3 : weight_copper = total_weight_brass - weight_zinc) : 
  weight_copper / weight_zinc = 3 / 7 :=
by
  sorry

end copper_to_zinc_ratio_l107_107215


namespace product_fraction_simplification_l107_107320

theorem product_fraction_simplification : 
  (1^4 - 1) / (1^4 + 1) * (2^4 - 1) / (2^4 + 1) * (3^4 - 1) / (3^4 + 1) *
  (4^4 - 1) / (4^4 + 1) * (5^4 - 1) / (5^4 + 1) * (6^4 - 1) / (6^4 + 1) *
  (7^4 - 1) / (7^4 + 1) = 50 := 
  sorry

end product_fraction_simplification_l107_107320


namespace average_apples_sold_per_day_l107_107490

theorem average_apples_sold_per_day (boxes_sold : ℕ) (days : ℕ) (apples_per_box : ℕ) (H1 : boxes_sold = 12) (H2 : days = 4) (H3 : apples_per_box = 25) : (boxes_sold * apples_per_box) / days = 75 :=
by {
  -- Based on given conditions, the total apples sold is 12 * 25 = 300.
  -- Dividing by the number of days, 300 / 4 gives us 75 apples/day.
  -- The proof is omitted as instructed.
  sorry
}

end average_apples_sold_per_day_l107_107490


namespace angle_measure_l107_107486

theorem angle_measure (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : P = 206 :=
by
  sorry

end angle_measure_l107_107486


namespace leap_day_2040_is_friday_l107_107289

def leap_day_day_of_week (start_year : ℕ) (start_day : ℕ) (end_year : ℕ) : ℕ :=
  let num_years := end_year - start_year
  let num_leap_years := (num_years + 4) / 4 -- number of leap years including start and end year
  let total_days := 365 * (num_years - num_leap_years) + 366 * num_leap_years
  let day_of_week := (total_days % 7 + start_day) % 7
  day_of_week

theorem leap_day_2040_is_friday :
  leap_day_day_of_week 2008 5 2040 = 5 := 
  sorry

end leap_day_2040_is_friday_l107_107289


namespace number_of_friends_l107_107159

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def envelopes_left : ℕ := 22

theorem number_of_friends :
  ((total_envelopes - envelopes_left) / envelopes_per_friend) = 5 := by
  sorry

end number_of_friends_l107_107159


namespace compare_exponent_inequality_l107_107573

theorem compare_exponent_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 :=
sorry

end compare_exponent_inequality_l107_107573


namespace perpendicular_line_equation_l107_107047

theorem perpendicular_line_equation 
  (p : ℝ × ℝ)
  (L1 : ℝ → ℝ → Prop)
  (L2 : ℝ → ℝ → ℝ → Prop) 
  (hx : p = (1, -1)) 
  (hL1 : ∀ x y, L1 x y ↔ 3 * x - 2 * y = 0) 
  (hL2 : ∀ x y m, L2 x y m ↔ 2 * x + 3 * y + m = 0) :
  ∃ m : ℝ, L2 (p.1) (p.2) m ∧ 2 * p.1 + 3 * p.2 + m = 0 :=
by
  sorry

end perpendicular_line_equation_l107_107047


namespace square_side_increase_l107_107447

variable (s : ℝ)  -- original side length of the square.
variable (p : ℝ)  -- percentage increase of the side length.

theorem square_side_increase (h1 : (s * (1 + p / 100))^2 = 1.21 * s^2) : p = 10 := 
by
  sorry

end square_side_increase_l107_107447


namespace population_increase_l107_107761

-- Define the problem conditions
def average_birth_rate := (6 + 10) / 2 / 2  -- the average number of births per second
def average_death_rate := (4 + 8) / 2 / 2  -- the average number of deaths per second
def net_migration_day := 500  -- net migration inflow during the day
def net_migration_night := -300  -- net migration outflow during the night

-- Define the number of seconds in a day
def seconds_in_a_day := 24 * 3600

-- Define the net increase due to births and deaths
def net_increase_births_deaths := (average_birth_rate - average_death_rate) * seconds_in_a_day

-- Define the total net migration
def total_net_migration := net_migration_day + net_migration_night

-- Define the total population net increase
def total_population_net_increase :=
  net_increase_births_deaths + total_net_migration

-- The theorem to be proved
theorem population_increase (h₁ : average_birth_rate = 4)
                           (h₂ : average_death_rate = 3)
                           (h₃ : seconds_in_a_day = 86400) :
  total_population_net_increase = 86600 := by
  sorry

end population_increase_l107_107761


namespace promotional_savings_l107_107943

noncomputable def y (x : ℝ) : ℝ :=
if x ≤ 500 then x
else if x ≤ 1000 then 500 + 0.8 * (x - 500)
else 500 + 400 + 0.5 * (x - 1000)

theorem promotional_savings (payment : ℝ) (hx : y 2400 = 1600) : 2400 - payment = 800 :=
by sorry

end promotional_savings_l107_107943


namespace garrett_bought_peanut_granola_bars_l107_107200

def garrett_granola_bars (t o : ℕ) (h_t : t = 14) (h_o : o = 6) : ℕ :=
  t - o

theorem garrett_bought_peanut_granola_bars : garrett_granola_bars 14 6 rfl rfl = 8 :=
  by
    unfold garrett_granola_bars
    rw [Nat.sub_eq_of_eq_add]
    sorry

end garrett_bought_peanut_granola_bars_l107_107200


namespace gp_values_l107_107884

theorem gp_values (p : ℝ) (hp : 0 < p) :
  let a := -p - 12
  let b := 2 * Real.sqrt p
  let c := p - 5
  (b / a = c / b) ↔ p = 4 :=
by
  sorry

end gp_values_l107_107884


namespace union_M_N_l107_107816

def M : Set ℕ := {1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by
  sorry

end union_M_N_l107_107816


namespace miriam_cleaning_room_time_l107_107976

theorem miriam_cleaning_room_time
  (laundry_time : Nat := 30)
  (bathroom_time : Nat := 15)
  (homework_time : Nat := 40)
  (total_time : Nat := 120) :
  ∃ room_time : Nat, laundry_time + bathroom_time + homework_time + room_time = total_time ∧
                  room_time = 35 := by
  sorry

end miriam_cleaning_room_time_l107_107976


namespace count_four_digit_numbers_with_thousands_digit_one_l107_107862

theorem count_four_digit_numbers_with_thousands_digit_one : 
  ∃ N : ℕ, N = 1000 ∧ (∀ n : ℕ, 1000 ≤ n ∧ n < 2000 → (n / 1000 = 1)) :=
sorry

end count_four_digit_numbers_with_thousands_digit_one_l107_107862


namespace find_n_interval_l107_107452

theorem find_n_interval :
  ∃ n : ℕ, n < 1000 ∧
  (∃ ghijkl : ℕ, (ghijkl < 999999) ∧ (ghijkl * n = 999999 * ghijkl)) ∧
  (∃ mnop : ℕ, (mnop < 9999) ∧ (mnop * (n + 5) = 9999 * mnop)) ∧
  151 ≤ n ∧ n ≤ 300 :=
sorry

end find_n_interval_l107_107452


namespace first_day_is_wednesday_l107_107630

theorem first_day_is_wednesday (day22_wednesday : ∀ n, n = 22 → (n = 22 → "Wednesday" = "Wednesday")) :
  ∀ n, n = 1 → (n = 1 → "Wednesday" = "Wednesday") :=
by
  sorry

end first_day_is_wednesday_l107_107630


namespace values_of_x_for_f_l107_107874

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem values_of_x_for_f (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_monotonically_increasing_on_nonneg f) : 
  (∀ x : ℝ, f (2*x - 1) < f 3 ↔ (-1 < x ∧ x < 2)) :=
by
  sorry

end values_of_x_for_f_l107_107874


namespace cannot_sum_to_nine_l107_107752

def sum_pairs (a b c d : ℕ) : List ℕ :=
  [a + b, c + d, a + c, b + d, a + d, b + c]

theorem cannot_sum_to_nine :
  ∀ (a b c d : ℕ), a ≠ 5 ∧ b ≠ 6 ∧ c ≠ 5 ∧ d ≠ 6 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a + b ≠ 11 ∧ a + c ≠ 11 ∧ a + d ≠ 11 ∧ b + c ≠ 11 ∧ b + d ≠ 11 ∧ c + d ≠ 11 →
  ¬9 ∈ sum_pairs a b c d :=
by
  intros a b c d h
  sorry

end cannot_sum_to_nine_l107_107752


namespace two_colonies_same_time_l107_107861

def doubles_in_size_every_day (P : ℕ → ℕ) : Prop :=
∀ n, P (n + 1) = 2 * P n

def reaches_habitat_limit_in (f : ℕ → ℕ) (days limit : ℕ) : Prop :=
f days = limit

theorem two_colonies_same_time (P : ℕ → ℕ) (Q : ℕ → ℕ) (limit : ℕ) (days : ℕ)
  (h1 : doubles_in_size_every_day P)
  (h2 : reaches_habitat_limit_in P days limit)
  (h3 : ∀ n, Q n = 2 * P n) :
  reaches_habitat_limit_in Q days limit :=
sorry

end two_colonies_same_time_l107_107861


namespace find_second_number_l107_107938

theorem find_second_number (x y z : ℚ) (h₁ : x + y + z = 150) (h₂ : x = (3 / 4) * y) (h₃ : z = (7 / 5) * y) : 
  y = 1000 / 21 :=
by sorry

end find_second_number_l107_107938


namespace cathy_remaining_money_l107_107290

noncomputable def remaining_money (initial : ℝ) (dad : ℝ) (book : ℝ) (cab_percentage : ℝ) (food_percentage : ℝ) : ℝ :=
  let money_mom := 2 * dad
  let total_money := initial + dad + money_mom
  let remaining_after_book := total_money - book
  let cab_cost := cab_percentage * remaining_after_book
  let food_budget := food_percentage * total_money
  let dinner_cost := 0.5 * food_budget
  remaining_after_book - cab_cost - dinner_cost

theorem cathy_remaining_money :
  remaining_money 12 25 15 0.03 0.4 = 52.44 :=
by
  sorry

end cathy_remaining_money_l107_107290


namespace cylindrical_to_rectangular_l107_107028

theorem cylindrical_to_rectangular (r θ z : ℝ) (h1 : r = 6) (h2 : θ = π / 3) (h3 : z = 2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, 2) := 
by 
  rw [h1, h2, h3]
  sorry

end cylindrical_to_rectangular_l107_107028


namespace rabbit_carrot_count_l107_107866

theorem rabbit_carrot_count
  (r h : ℕ)
  (hr : r = h - 3)
  (eq_carrots : 4 * r = 5 * h) :
  4 * r = 36 :=
by
  sorry

end rabbit_carrot_count_l107_107866


namespace smallest_common_multiple_l107_107279

theorem smallest_common_multiple : Nat.lcm 18 35 = 630 := by
  sorry

end smallest_common_multiple_l107_107279


namespace black_circles_count_l107_107929

theorem black_circles_count (a1 d n : ℕ) (h1 : a1 = 2) (h2 : d = 1) (h3 : n = 16) :
  (n * (a1 + (n - 1) * d) / 2) + n ≤ 160 :=
by
  rw [h1, h2, h3]
  -- Here we will carry out the arithmetic to prove the statement
  sorry

end black_circles_count_l107_107929


namespace sum_equals_1584_l107_107370

-- Let's define the function that computes the sum, according to the pattern
def sumPattern : ℕ → ℝ
  | 0 => 0
  | k + 1 => if (k + 1) % 3 = 0 then - (k + 1) + sumPattern k
             else (k + 1) + sumPattern k

-- This function defines the problem setting and the final expected result
theorem sum_equals_1584 : sumPattern 99 = 1584 := by
  sorry

end sum_equals_1584_l107_107370


namespace john_games_l107_107026

variables (G_f G_g B G G_t : ℕ)

theorem john_games (h1: G_f = 21) (h2: B = 23) (h3: G = 6) 
(h4: G_t = G_f + G_g) (h5: G + B = G_t) : G_g = 8 :=
by sorry

end john_games_l107_107026


namespace power_function_evaluation_l107_107726

theorem power_function_evaluation (f : ℝ → ℝ) (a : ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = (Real.sqrt 2) / 2) :
  f 4 = 1 / 2 := by
  sorry

end power_function_evaluation_l107_107726


namespace geometric_sequence_sum_l107_107076

theorem geometric_sequence_sum (a : ℝ) (q : ℝ) (h1 : a * q^2 + a * q^5 = 6)
  (h2 : a * q^4 + a * q^7 = 9) : a * q^6 + a * q^9 = 27 / 2 :=
by
  sorry

end geometric_sequence_sum_l107_107076


namespace jill_spent_more_l107_107324

def cost_per_ball_red : ℝ := 1.50
def cost_per_ball_yellow : ℝ := 1.25
def cost_per_ball_blue : ℝ := 1.00

def packs_red : ℕ := 5
def packs_yellow : ℕ := 4
def packs_blue : ℕ := 3

def balls_per_pack_red : ℕ := 18
def balls_per_pack_yellow : ℕ := 15
def balls_per_pack_blue : ℕ := 12

def balls_red : ℕ := packs_red * balls_per_pack_red
def balls_yellow : ℕ := packs_yellow * balls_per_pack_yellow
def balls_blue : ℕ := packs_blue * balls_per_pack_blue

def cost_red : ℝ := balls_red * cost_per_ball_red
def cost_yellow : ℝ := balls_yellow * cost_per_ball_yellow
def cost_blue : ℝ := balls_blue * cost_per_ball_blue

def combined_cost_yellow_blue : ℝ := cost_yellow + cost_blue

theorem jill_spent_more : cost_red = combined_cost_yellow_blue + 24 := by
  sorry

end jill_spent_more_l107_107324


namespace range_of_m_F_x2_less_than_x2_minus_1_l107_107618

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (x : ℝ) : ℝ := 3 - 2 / x
noncomputable def T (x m : ℝ) : ℝ := Real.log x - x - 2 * m
noncomputable def F (x m : ℝ) : ℝ := x - m / x - 2 * Real.log x
noncomputable def h (t : ℝ) : ℝ := t - 2 * Real.log t - 1

-- (1)
theorem range_of_m (m : ℝ) (h_intersections : ∃ x y : ℝ, T x m = 0 ∧ T y m = 0 ∧ x ≠ y) :
  m < -1 / 2 := sorry

-- (2)
theorem F_x2_less_than_x2_minus_1 {m : ℝ} (h₀ : 0 < m ∧ m < 1) {x₁ x₂ : ℝ} (h₁ : 0 < x₁ ∧ x₁ < x₂)
  (h₂ : F x₁ m = 0 ∧ F x₂ m = 0) :
  F x₂ m < x₂ - 1 := sorry

end range_of_m_F_x2_less_than_x2_minus_1_l107_107618


namespace trigonometric_identity_l107_107769

variable (θ : ℝ) (h : Real.tan θ = 2)

theorem trigonometric_identity : 
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 := 
sorry

end trigonometric_identity_l107_107769


namespace total_transportation_cost_l107_107465

def weights_in_grams : List ℕ := [300, 450, 600]
def cost_per_kg : ℕ := 15000

def convert_to_kg (w : ℕ) : ℚ :=
  w / 1000

def calculate_cost (weight_in_kg : ℚ) (cost_per_kg : ℕ) : ℚ :=
  weight_in_kg * cost_per_kg

def total_cost (weights_in_grams : List ℕ) (cost_per_kg : ℕ) : ℚ :=
  weights_in_grams.map (λ w => calculate_cost (convert_to_kg w) cost_per_kg) |>.sum

theorem total_transportation_cost :
  total_cost weights_in_grams cost_per_kg = 20250 := by
  sorry

end total_transportation_cost_l107_107465


namespace jacoby_lottery_winning_l107_107822

theorem jacoby_lottery_winning :
  let total_needed := 5000
  let job_earning := 20 * 10
  let cookies_earning := 4 * 24
  let total_earnings_before_lottery := job_earning + cookies_earning
  let after_lottery := total_earnings_before_lottery - 10
  let gift_from_sisters := 500 * 2
  let total_earnings_and_gifts := after_lottery + gift_from_sisters
  let total_so_far := total_needed - 3214
  total_so_far - total_earnings_and_gifts = 500 :=
by
  sorry

end jacoby_lottery_winning_l107_107822


namespace total_weight_of_bottles_l107_107651

variables (P G : ℕ) -- P stands for the weight of a plastic bottle, G stands for the weight of a glass bottle

-- Condition 1: The weight of 3 glass bottles is 600 grams
axiom glass_bottle_weight : 3 * G = 600

-- Condition 2: A glass bottle is 150 grams heavier than a plastic bottle
axiom glass_bottle_heavier : G = P + 150

-- The statement to prove: The total weight of 4 glass bottles and 5 plastic bottles is 1050 grams
theorem total_weight_of_bottles :
  4 * G + 5 * P = 1050 :=
sorry

end total_weight_of_bottles_l107_107651


namespace flyDistanceCeiling_l107_107300

variable (P : ℝ × ℝ × ℝ)
variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- Defining the conditions
def isAtRightAngles (P : ℝ × ℝ × ℝ) : Prop :=
  P = (0, 0, 0)

def distanceFromWall1 (x : ℝ) : Prop :=
  x = 2

def distanceFromWall2 (y : ℝ) : Prop :=
  y = 5

def distanceFromPointP (x y z : ℝ) : Prop :=
  7 = Real.sqrt (x^2 + y^2 + z^2)

-- Proving the distance from the ceiling
theorem flyDistanceCeiling (P : ℝ × ℝ × ℝ) (x y z : ℝ) :
  isAtRightAngles P →
  distanceFromWall1 x →
  distanceFromWall2 y →
  distanceFromPointP x y z →
  z = 2 * Real.sqrt 5 := 
sorry

end flyDistanceCeiling_l107_107300


namespace slope_of_line_l107_107048

theorem slope_of_line : ∀ (x y : ℝ), 4 * y = -6 * x + 12 → ∃ m b : ℝ, y = m * x + b ∧ m = -3 / 2 :=
by 
sorry

end slope_of_line_l107_107048


namespace minimum_loaves_arithmetic_sequence_l107_107328

theorem minimum_loaves_arithmetic_sequence :
  ∃ a d : ℚ, 
    (5 * a = 100) ∧ (3 * a + 3 * d = 7 * (2 * a - 3 * d)) ∧ (a - 2 * d = 5/3) :=
sorry

end minimum_loaves_arithmetic_sequence_l107_107328


namespace age_of_B_l107_107611

-- Define the ages of A and B
variables (A B : ℕ)

-- The conditions given in the problem
def condition1 (a b : ℕ) : Prop := a + 10 = 2 * (b - 10)
def condition2 (a b : ℕ) : Prop := a = b + 9

theorem age_of_B (A B : ℕ) (h1 : condition1 A B) (h2 : condition2 A B) : B = 39 :=
by
  sorry

end age_of_B_l107_107611


namespace find_multiple_of_ron_l107_107000

variable (R_d R_g R_n m : ℕ)

def rodney_can_lift_146 : Prop := R_d = 146
def combined_weight_239 : Prop := R_d + R_g + R_n = 239
def rodney_twice_as_roger : Prop := R_d = 2 * R_g
def roger_seven_less_than_multiple_of_ron : Prop := R_g = m * R_n - 7

theorem find_multiple_of_ron (h1 : rodney_can_lift_146 R_d) 
                             (h2 : combined_weight_239 R_d R_g R_n) 
                             (h3 : rodney_twice_as_roger R_d R_g) 
                             (h4 : roger_seven_less_than_multiple_of_ron R_g R_n m) 
                             : m = 4 :=
by 
    sorry

end find_multiple_of_ron_l107_107000


namespace a_b_product_l107_107848

theorem a_b_product (a b : ℝ) (h1 : 2 * a - b = 1) (h2 : 2 * b - a = 7) : (a + b) * (a - b) = -16 :=
by
  -- The proof would be provided here.
  sorry

end a_b_product_l107_107848


namespace solution_set_of_quadratic_inequality_l107_107466

theorem solution_set_of_quadratic_inequality 
  (a b c x₁ x₂ : ℝ)
  (h1 : a > 0) 
  (h2 : a * x₁^2 + b * x₁ + c = 0)
  (h3 : a * x₂^2 + b * x₂ + c = 0)
  : {x : ℝ | a * x^2 + b * x + c > 0} = ({x : ℝ | x > x₁} ∩ {x : ℝ | x > x₂}) ∪ ({x : ℝ | x < x₁} ∩ {x : ℝ | x < x₂}) :=
sorry

end solution_set_of_quadratic_inequality_l107_107466


namespace initial_apps_l107_107341

-- Define the initial condition stating the number of files Dave had initially
def files_initial : ℕ := 21

-- Define the condition after deletion
def apps_after_deletion : ℕ := 3
def files_after_deletion : ℕ := 7

-- Define the number of files deleted
def files_deleted : ℕ := 14

-- Prove that the initial number of apps Dave had was 3
theorem initial_apps (a : ℕ) (h1 : files_initial = 21) 
(h2 : files_after_deletion = 7) 
(h3 : files_deleted = 14) 
(h4 : a - 3 = 0) : a = 3 :=
by sorry

end initial_apps_l107_107341


namespace frank_total_cans_l107_107896

def total_cans_picked_up (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ) : ℕ :=
  let total_bags := bags_saturday + bags_sunday
  total_bags * cans_per_bag

theorem frank_total_cans : total_cans_picked_up 5 3 5 = 40 := by
  sorry

end frank_total_cans_l107_107896


namespace tenth_term_geometric_sequence_l107_107637

def a := 5
def r := Rat.ofInt 3 / 4
def n := 10

theorem tenth_term_geometric_sequence :
  a * r^(n-1) = Rat.ofInt 98415 / Rat.ofInt 262144 := sorry

end tenth_term_geometric_sequence_l107_107637


namespace fraction_pow_zero_l107_107502

theorem fraction_pow_zero :
  (4310000 / -21550000 : ℝ) ≠ 0 →
  (4310000 / -21550000 : ℝ) ^ 0 = 1 :=
by
  intro h
  sorry

end fraction_pow_zero_l107_107502


namespace find_sum_of_m_and_k_l107_107132

theorem find_sum_of_m_and_k
  (d m k : ℤ)
  (h : (9 * d^2 - 5 * d + m) * (4 * d^2 + k * d - 6) = 36 * d^4 + 11 * d^3 - 59 * d^2 + 10 * d + 12) :
  m + k = -7 :=
by sorry

end find_sum_of_m_and_k_l107_107132


namespace find_a_l107_107210

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a = 0 ↔ 3 * x^4 - 48 = 0) → a = 4 :=
  by
    intros h
    sorry

end find_a_l107_107210


namespace power_function_inequality_l107_107077

theorem power_function_inequality (m : ℕ) (h : m > 0)
  (h_point : (2 : ℝ) ^ (1 / (m ^ 2 + m)) = Real.sqrt 2) :
  m = 1 ∧ ∀ a : ℝ, 1 ≤ a ∧ a < (3 / 2) → 
  (2 - a : ℝ) ^ (1 / (m ^ 2 + m)) > (a - 1 : ℝ) ^ (1 / (m ^ 2 + m)) :=
by
  sorry

end power_function_inequality_l107_107077


namespace largest_angle_bounds_triangle_angles_l107_107273

theorem largest_angle_bounds (A B C : ℝ) (angle_A angle_B angle_C : ℝ)
  (h_triangle : angle_A + angle_B + angle_C = 180)
  (h_tangent : angle_B + 2 * angle_C = 90) :
  90 ≤ angle_A ∧ angle_A < 135 :=
sorry

theorem triangle_angles (A B C : ℝ) (angle_A angle_B angle_C : ℝ)
  (h_triangle : angle_A + angle_B + angle_C = 180)
  (h_tangent_B : angle_B + 2 * angle_C = 90)
  (h_tangent_C : angle_C + 2 * angle_B = 90) :
  angle_A = 120 ∧ angle_B = 30 ∧ angle_C = 30 :=
sorry

end largest_angle_bounds_triangle_angles_l107_107273


namespace jamie_dimes_l107_107443

theorem jamie_dimes (y : ℕ) (h : 5 * y + 10 * y + 25 * y = 1440) : y = 36 :=
by 
  sorry

end jamie_dimes_l107_107443


namespace min_guests_l107_107901

/-- Problem statement:
Given:
1. The total food consumed by all guests is 319 pounds.
2. Each guest consumes no more than 1.5 pounds of meat, 0.3 pounds of vegetables, and 0.2 pounds of dessert.
3. Each guest has equal proportions of meat, vegetables, and dessert.

Prove:
The minimum number of guests such that the total food consumed is less than or equal to 319 pounds is 160.
-/
theorem min_guests (total_food : ℝ) (meat_per_guest : ℝ) (veg_per_guest : ℝ) (dessert_per_guest : ℝ) (G : ℕ) :
  total_food = 319 ∧ meat_per_guest ≤ 1.5 ∧ veg_per_guest ≤ 0.3 ∧ dessert_per_guest ≤ 0.2 ∧
  (meat_per_guest + veg_per_guest + dessert_per_guest = 2.0) →
  G = 160 :=
by
  intros h
  sorry

end min_guests_l107_107901


namespace total_apples_in_stack_l107_107661

theorem total_apples_in_stack:
  let base_layer := 6 * 9
  let layer_2 := 5 * 8
  let layer_3 := 4 * 7
  let layer_4 := 3 * 6
  let layer_5 := 2 * 5
  let layer_6 := 1 * 4
  let top_layer := 2
  base_layer + layer_2 + layer_3 + layer_4 + layer_5 + layer_6 + top_layer = 156 :=
by sorry

end total_apples_in_stack_l107_107661


namespace product_of_special_triplet_l107_107318

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_triangular (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1) / 2

def three_consecutive (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1

theorem product_of_special_triplet :
  ∃ a b c : ℕ, a < b ∧ b < c ∧ c < 20 ∧ three_consecutive a b c ∧
   is_prime a ∧ is_even b ∧ is_triangular c ∧ a * b * c = 2730 :=
sorry

end product_of_special_triplet_l107_107318


namespace Xiaoliang_catches_up_in_h_l107_107801

-- Define the speeds and head start
def speed_Xiaobin : ℝ := 4  -- Xiaobin's speed in km/h
def speed_Xiaoliang : ℝ := 12  -- Xiaoliang's speed in km/h
def head_start : ℝ := 6  -- Xiaobin's head start in hours

-- Define the additional distance Xiaoliang needs to cover
def additional_distance : ℝ := speed_Xiaobin * head_start

-- Define the hourly distance difference between them
def speed_difference : ℝ := speed_Xiaoliang - speed_Xiaobin

-- Prove that Xiaoliang will catch up with Xiaobin in exactly 3 hours
theorem Xiaoliang_catches_up_in_h : (additional_distance / speed_difference) = 3 :=
by
  sorry

end Xiaoliang_catches_up_in_h_l107_107801


namespace reading_proof_l107_107423

noncomputable def reading (arrow_pos : ℝ) : ℝ :=
  if arrow_pos > 9.75 ∧ arrow_pos < 10.0 then 9.95 else 0

theorem reading_proof
  (arrow_pos : ℝ)
  (h0 : 9.75 < arrow_pos)
  (h1 : arrow_pos < 10.0)
  (possible_readings : List ℝ)
  (h2 : possible_readings = [9.80, 9.90, 9.95, 10.0, 9.85]) :
  reading arrow_pos = 9.95 := by
  -- Proof would go here
  sorry

end reading_proof_l107_107423


namespace discriminant_of_polynomial_l107_107059

noncomputable def polynomial_discriminant (a b c : ℚ) : ℚ :=
b^2 - 4 * a * c

theorem discriminant_of_polynomial : polynomial_discriminant 2 (4 - (1/2 : ℚ)) 1 = 17 / 4 :=
by
  sorry

end discriminant_of_polynomial_l107_107059


namespace matinee_ticket_price_l107_107089

theorem matinee_ticket_price
  (M : ℝ)  -- Denote M as the price of a matinee ticket
  (evening_ticket_price : ℝ := 12)  -- Price of an evening ticket
  (ticket_3D_price : ℝ := 20)  -- Price of a 3D ticket
  (matinee_tickets_sold : ℕ := 200)  -- Number of matinee tickets sold
  (evening_tickets_sold : ℕ := 300)  -- Number of evening tickets sold
  (tickets_3D_sold : ℕ := 100)  -- Number of 3D tickets sold
  (total_revenue : ℝ := 6600) -- Total revenue
  (h : matinee_tickets_sold * M + evening_tickets_sold * evening_ticket_price + tickets_3D_sold * ticket_3D_price = total_revenue) :
  M = 5 :=
by
  sorry

end matinee_ticket_price_l107_107089


namespace sufficient_condition_of_square_inequality_l107_107918

variables (a b : ℝ)

theorem sufficient_condition_of_square_inequality (ha : a > 0) (hb : b > 0) (h : a > b) : a^2 > b^2 :=
by {
  sorry
}

end sufficient_condition_of_square_inequality_l107_107918


namespace math_problem_l107_107886

noncomputable def find_min_value (a m n : ℝ) (h_a_pos : a > 0) (h_bn : n = 2 * m + 1 / 2)
  (h_b : -a^2 / 2 + 3 * Real.log a = -1 / 2) : ℝ :=
  (3 * Real.sqrt 5 / 5) ^ 2

theorem math_problem (a m n : ℝ) (h_a_pos : a > 0) (h_bn : n = 2 * m + 1 / 2) :
  ∃ b : ℝ, b = -a^2 / 2 + 3 * Real.log a →
  (a - m) ^ 2 + (b - n) ^ 2 = 9 / 5 :=
by
  sorry

end math_problem_l107_107886


namespace posts_needed_l107_107917

-- Define the main properties
def length_of_side_W_stone_wall := 80
def short_side := 50
def intervals (metres: ℕ) := metres / 10 + 1 

-- Define the conditions
def posts_along_w_stone_wall := intervals length_of_side_W_stone_wall
def posts_along_short_sides := 2 * (intervals short_side - 1)

-- Calculate total posts
def total_posts := posts_along_w_stone_wall + posts_along_short_sides

-- Define the theorem
theorem posts_needed : total_posts = 19 := 
by
  sorry

end posts_needed_l107_107917


namespace division_multiplication_identity_l107_107810

theorem division_multiplication_identity (a b c d : ℕ) (h1 : b = 6) (h2 : c = 2) (h3 : d = 3) :
  a = 120 → 120 * (b / c) * d = 120 := by
  intro h
  rw [h2, h3, h1]
  sorry

end division_multiplication_identity_l107_107810


namespace tenth_equation_sum_of_cubes_l107_107291

theorem tenth_equation_sum_of_cubes :
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3 + 9^3 + 10^3) = 55^2 := 
by sorry

end tenth_equation_sum_of_cubes_l107_107291


namespace points_meet_every_720_seconds_l107_107529

theorem points_meet_every_720_seconds
    (v1 v2 : ℝ) 
    (h1 : v1 - v2 = 1/720) 
    (h2 : (1/v2) - (1/v1) = 10) :
    v1 = 1/80 ∧ v2 = 1/90 :=
by
  sorry

end points_meet_every_720_seconds_l107_107529


namespace kitchen_upgrade_cost_l107_107550

-- Define the number of cabinet knobs and their cost
def num_knobs : ℕ := 18
def cost_per_knob : ℝ := 2.50

-- Define the number of drawer pulls and their cost
def num_pulls : ℕ := 8
def cost_per_pull : ℝ := 4.00

-- Calculate the total cost of the knobs
def total_cost_knobs : ℝ := num_knobs * cost_per_knob

-- Calculate the total cost of the pulls
def total_cost_pulls : ℝ := num_pulls * cost_per_pull

-- Calculate the total cost of the kitchen upgrade
def total_cost : ℝ := total_cost_knobs + total_cost_pulls

-- Theorem statement
theorem kitchen_upgrade_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_cost_l107_107550


namespace find_k_l107_107147

theorem find_k (k : ℝ) : 
  (∃ c1 c2 : ℝ, (2 * c1^2 + 5 * c1 = k) ∧ 
                (2 * c2^2 + 5 * c2 = k) ∧ 
                (c1 > c2) ∧ 
                (c1 - c2 = 5.5)) → 
  k = 12 := 
by
  intros h
  obtain ⟨c1, c2, h1, h2, h3, h4⟩ := h
  sorry

end find_k_l107_107147


namespace base8_units_digit_l107_107867

theorem base8_units_digit (n m : ℕ) (h1 : n = 348) (h2 : m = 27) : 
  (n * m % 8) = 4 := sorry

end base8_units_digit_l107_107867


namespace number_of_merchants_l107_107069

theorem number_of_merchants (x : ℕ) (h : 2 * x^3 = 2662) : x = 11 :=
  sorry

end number_of_merchants_l107_107069


namespace stationery_sales_l107_107493

theorem stationery_sales :
  let pen_percentage : ℕ := 42
  let pencil_percentage : ℕ := 27
  let total_sales_percentage : ℕ := 100
  total_sales_percentage - (pen_percentage + pencil_percentage) = 31 :=
by
  sorry

end stationery_sales_l107_107493


namespace arithmetic_sequence_sum_l107_107371

theorem arithmetic_sequence_sum :
  ∃ (c d e : ℕ), 
  c = 15 + (9 - 3) ∧ 
  d = c + (9 - 3) ∧ 
  e = d + (9 - 3) ∧ 
  c + d + e = 81 :=
by 
  sorry

end arithmetic_sequence_sum_l107_107371


namespace interpretation_of_k5_3_l107_107959

theorem interpretation_of_k5_3 (k : ℕ) (hk : 0 < k) : (k^5)^3 = k^5 * k^5 * k^5 :=
by sorry

end interpretation_of_k5_3_l107_107959


namespace Sam_wins_probability_l107_107507

-- Define the basic probabilities
def prob_hit : ℚ := 2 / 5
def prob_miss : ℚ := 3 / 5

-- Define the desired probability that Sam wins
noncomputable def p : ℚ := 5 / 8

-- The mathematical problem statement in Lean
theorem Sam_wins_probability :
  p = prob_hit + (prob_miss * prob_miss * p) := 
sorry

end Sam_wins_probability_l107_107507


namespace find_common_tangent_sum_constant_l107_107422

theorem find_common_tangent_sum_constant :
  ∃ (a b c : ℕ), (∀ x y : ℚ, y = x^2 + 169/100 → x = y^2 + 49/4 → a * x + b * y = c) ∧
  (Int.gcd (Int.gcd a b) c = 1) ∧
  (a + b + c = 52) :=
sorry

end find_common_tangent_sum_constant_l107_107422


namespace cone_cannot_have_rectangular_projection_l107_107526

def orthographic_projection (solid : Type) : Type := sorry

theorem cone_cannot_have_rectangular_projection :
  (∀ (solid : Type), orthographic_projection solid = Rectangle → solid ≠ Cone) :=
sorry

end cone_cannot_have_rectangular_projection_l107_107526


namespace hyperbolic_identity_l107_107808

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem hyperbolic_identity (x : ℝ) : (ch x) ^ 2 - (sh x) ^ 2 = 1 := 
sorry

end hyperbolic_identity_l107_107808


namespace range_of_m_l107_107708

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ (m > 2 ∨ m < -4) :=
by
  sorry

end range_of_m_l107_107708


namespace square_side_length_false_l107_107294

theorem square_side_length_false (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 8) (h2 : side_length = 4) :
  ¬(4 * side_length = perimeter) :=
by
  sorry

end square_side_length_false_l107_107294


namespace jackson_vacuuming_time_l107_107394

-- Definitions based on the conditions
def hourly_wage : ℕ := 5
def washing_dishes_time : ℝ := 0.5
def cleaning_bathroom_time : ℝ := 3 * washing_dishes_time
def total_earnings : ℝ := 30

-- The total time spent on chores
def total_chore_time (V : ℝ) : ℝ :=
  2 * V + washing_dishes_time + cleaning_bathroom_time

-- The main theorem that needs to be proven
theorem jackson_vacuuming_time :
  ∃ V : ℝ, hourly_wage * total_chore_time V = total_earnings ∧ V = 2 :=
by
  sorry

end jackson_vacuuming_time_l107_107394


namespace fraction_of_new_releases_l107_107060

theorem fraction_of_new_releases (total_books : ℕ) (historical_fiction_percent : ℝ) (historical_new_releases_percent : ℝ) (other_new_releases_percent : ℝ)
  (h1 : total_books = 100)
  (h2 : historical_fiction_percent = 0.4)
  (h3 : historical_new_releases_percent = 0.4)
  (h4 : other_new_releases_percent = 0.2) :
  (historical_fiction_percent * historical_new_releases_percent * total_books) / 
  ((historical_fiction_percent * historical_new_releases_percent * total_books) + ((1 - historical_fiction_percent) * other_new_releases_percent * total_books)) = 4 / 7 :=
by
  have h_books : total_books = 100 := h1
  have h_fiction : historical_fiction_percent = 0.4 := h2
  have h_new_releases : historical_new_releases_percent = 0.4 := h3
  have h_other_new_releases : other_new_releases_percent = 0.2 := h4
  sorry

end fraction_of_new_releases_l107_107060


namespace Tyler_CDs_after_giveaway_and_purchase_l107_107223

theorem Tyler_CDs_after_giveaway_and_purchase :
  (∃ cds_initial cds_giveaway_fraction cds_bought cds_final, 
     cds_initial = 21 ∧ 
     cds_giveaway_fraction = 1 / 3 ∧ 
     cds_bought = 8 ∧ 
     cds_final = cds_initial - (cds_initial * cds_giveaway_fraction) + cds_bought ∧
     cds_final = 22) := 
sorry

end Tyler_CDs_after_giveaway_and_purchase_l107_107223


namespace intersection_height_correct_l107_107482

noncomputable def intersection_height 
  (height_pole_1 height_pole_2 distance : ℝ) : ℝ := 
  let slope_1 := -(height_pole_1 / distance)
  let slope_2 := height_pole_2 / distance
  let y_intercept_1 := height_pole_1
  let y_intercept_2 := 0
  let x_intersection := height_pole_1 / (slope_2 - slope_1)
  let y_intersection := slope_2 * x_intersection + y_intercept_2
  y_intersection

theorem intersection_height_correct 
  : intersection_height 30 90 150 = 22.5 := 
by sorry

end intersection_height_correct_l107_107482


namespace number_of_dogs_l107_107311

theorem number_of_dogs (h1 : 24 = 2 * 2 + 4 * n) : n = 5 :=
by
  sorry

end number_of_dogs_l107_107311


namespace polynomial_coeff_sums_l107_107325

theorem polynomial_coeff_sums (g h : ℤ) (d : ℤ) :
  (7 * d^2 - 3 * d + g) * (3 * d^2 + h * d - 8) = 21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d - 16 →
  g + h = -3 :=
by
  sorry

end polynomial_coeff_sums_l107_107325


namespace least_possible_value_of_b_plus_c_l107_107524

theorem least_possible_value_of_b_plus_c :
  ∃ (b c : ℕ), (b > 0) ∧ (c > 0) ∧ (∃ (r1 r2 : ℝ), r1 - r2 = 30 ∧ 2 * r1 ^ 2 + b * r1 + c = 0 ∧ 2 * r2 ^ 2 + b * r2 + c = 0) ∧ b + c = 126 := 
by
  sorry 

end least_possible_value_of_b_plus_c_l107_107524


namespace smallest_n_for_good_sequence_l107_107732

def is_good_sequence (a : ℕ → ℝ) : Prop :=
   (∃ (a_0 : ℕ), a 0 = a_0) ∧
   (∀ i : ℕ, a (i+1) = 2 * a i + 1 ∨ a (i+1) = a i / (a i + 2)) ∧
   (∃ k : ℕ, a k = 2014)

theorem smallest_n_for_good_sequence : 
  ∀ (a : ℕ → ℝ), is_good_sequence a → ∃ n : ℕ, a n = 2014 ∧ ∀ m : ℕ, m < n → a m ≠ 2014 :=
sorry

end smallest_n_for_good_sequence_l107_107732


namespace work_completion_days_l107_107823

open Real

theorem work_completion_days (days_A : ℝ) (days_B : ℝ) (amount_total : ℝ) (amount_C : ℝ) :
  days_A = 6 ∧ days_B = 8 ∧ amount_total = 5000 ∧ amount_C = 625.0000000000002 →
  (1 / days_A) + (1 / days_B) + (amount_C / amount_total * (1)) = 5 / 12 →
  1 / ((1 / days_A) + (1 / days_B) + (amount_C / amount_total * (1))) = 2.4 :=
  sorry

end work_completion_days_l107_107823


namespace desired_alcohol_percentage_l107_107659

def initial_volume := 6.0
def initial_percentage := 35.0 / 100.0
def added_alcohol := 1.8
def final_volume := initial_volume + added_alcohol
def initial_alcohol := initial_volume * initial_percentage
def final_alcohol := initial_alcohol + added_alcohol
def desired_percentage := (final_alcohol / final_volume) * 100.0

theorem desired_alcohol_percentage : desired_percentage = 50.0 := 
by
  -- Proof would go here, but is omitted as per the instructions
  sorry

end desired_alcohol_percentage_l107_107659


namespace boys_girls_relationship_l107_107481

theorem boys_girls_relationship (b g : ℕ) (h1 : b > 0) (h2 : g > 2) (h3 : ∀ n : ℕ, n < b → (n + 1) + 2 ≤ g) (h4 : b + 2 = g) : b = g - 2 := 
by
  sorry

end boys_girls_relationship_l107_107481


namespace range_of_a_l107_107225

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ax^2 + ax + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_l107_107225


namespace find_b_minus_c_l107_107624

theorem find_b_minus_c (a b c : ℤ) (h : (x^2 + a * x - 3) * (x + 1) = x^3 + b * x^2 + c * x - 3) : b - c = 4 := by
  -- We would normally construct the proof here.
  sorry

end find_b_minus_c_l107_107624


namespace cost_each_side_is_56_l107_107276

-- Define the total cost and number of sides
def total_cost : ℕ := 224
def number_of_sides : ℕ := 4

-- Define the cost per side as the division of total cost by number of sides
def cost_per_side : ℕ := total_cost / number_of_sides

-- The theorem stating the cost per side is 56
theorem cost_each_side_is_56 : cost_per_side = 56 :=
by
  -- Proof would go here
  sorry

end cost_each_side_is_56_l107_107276


namespace find_complex_number_l107_107435

namespace ComplexProof

open Complex

def satisfies_conditions (z : ℂ) : Prop :=
  (z^2).im = 0 ∧ abs (z - I) = 1

theorem find_complex_number (z : ℂ) (h : satisfies_conditions z) : z = 0 ∨ z = 2 * I :=
sorry

end ComplexProof

end find_complex_number_l107_107435


namespace smallest_class_size_l107_107349

theorem smallest_class_size (n : ℕ) (h : 5 * n + 2 > 40) : 5 * n + 2 ≥ 42 :=
by
  sorry

end smallest_class_size_l107_107349


namespace regular_octagon_interior_angle_l107_107564

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l107_107564


namespace lcm_1540_2310_l107_107293

theorem lcm_1540_2310 : Nat.lcm 1540 2310 = 4620 :=
by sorry

end lcm_1540_2310_l107_107293


namespace y_share_per_x_l107_107741

theorem y_share_per_x (total_amount y_share : ℝ) (z_share_per_x : ℝ) 
  (h_total : total_amount = 234)
  (h_y_share : y_share = 54)
  (h_z_share_per_x : z_share_per_x = 0.5) :
  ∃ a : ℝ, (forall x : ℝ, y_share = a * x) ∧ a = 9 / 20 :=
by
  use 9 / 20
  intros
  sorry

end y_share_per_x_l107_107741


namespace no_three_nat_sum_pair_is_pow_of_three_l107_107319

theorem no_three_nat_sum_pair_is_pow_of_three :
  ¬ ∃ (a b c : ℕ) (m n p : ℕ), a + b = 3 ^ m ∧ b + c = 3 ^ n ∧ c + a = 3 ^ p := 
by 
  sorry

end no_three_nat_sum_pair_is_pow_of_three_l107_107319


namespace condition_for_all_real_solutions_l107_107920

theorem condition_for_all_real_solutions (c : ℝ) :
  (∀ x : ℝ, x^2 + x + c > 0) ↔ c > 1 / 4 :=
sorry

end condition_for_all_real_solutions_l107_107920


namespace initial_weight_of_alloy_is_16_l107_107298

variable (Z C : ℝ)
variable (h1 : Z / C = 5 / 3)
variable (h2 : (Z + 8) / C = 3)
variable (A : ℝ := Z + C)

theorem initial_weight_of_alloy_is_16 (h1 : Z / C = 5 / 3) (h2 : (Z + 8) / C = 3) : A = 16 := by
  sorry

end initial_weight_of_alloy_is_16_l107_107298


namespace circumcenter_rational_l107_107377

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end circumcenter_rational_l107_107377


namespace find_x3_l107_107888

noncomputable def x3 : ℝ :=
  Real.log ((2 / 3) + (1 / 3) * Real.exp 2)

theorem find_x3 
  (x1 x2 : ℝ)
  (h1 : x1 = 0)
  (h2 : x2 = 2)
  (A : ℝ × ℝ := (x1, Real.exp x1))
  (B : ℝ × ℝ := (x2, Real.exp x2))
  (C : ℝ × ℝ := ((2 * A.1 + B.1) / 3, (2 * A.2 + B.2) / 3))
  (yC : ℝ := (2 / 3) * A.2 + (1 / 3) * B.2)
  (E : ℝ × ℝ := (x3, yC)) :
  E.1 = Real.log ((2 / 3) + (1 / 3) * Real.exp x2) := sorry

end find_x3_l107_107888


namespace frank_has_3_cookies_l107_107053

-- The definitions and conditions based on the problem statement
def num_cookies_millie : ℕ := 4
def num_cookies_mike : ℕ := 3 * num_cookies_millie
def num_cookies_frank : ℕ := (num_cookies_mike / 2) - 3

-- The theorem stating the question and the correct answer
theorem frank_has_3_cookies : num_cookies_frank = 3 :=
by 
  -- This is where the proof steps would go, but for now we use sorry
  sorry

end frank_has_3_cookies_l107_107053


namespace petya_vasya_equal_again_l107_107899

theorem petya_vasya_equal_again (n : ℤ) (hn : n ≠ 0) :
  ∃ (k m : ℕ), (∃ P V : ℤ, P = n + 10 * k ∧ V = n - 10 * k ∧ 2014 * P * V = n) :=
sorry

end petya_vasya_equal_again_l107_107899


namespace largest_fraction_l107_107622

theorem largest_fraction :
  (∀ (a b : ℚ), a = 2 / 5 → b = 1 / 3 → a < b) ∧  
  (∀ (a c : ℚ), a = 2 / 5 → c = 7 / 15 → a < c) ∧ 
  (∀ (a d : ℚ), a = 2 / 5 → d = 5 / 12 → a < d) ∧ 
  (∀ (a e : ℚ), a = 2 / 5 → e = 3 / 8 → a < e) ∧ 
  (∀ (b c : ℚ), b = 1 / 3 → c = 7 / 15 → b < c) ∧
  (∀ (b d : ℚ), b = 1 / 3 → d = 5 / 12 → b < d) ∧ 
  (∀ (b e : ℚ), b = 1 / 3 → e = 3 / 8 → b < e) ∧ 
  (∀ (c d : ℚ), c = 7 / 15 → d = 5 / 12 → c > d) ∧
  (∀ (c e : ℚ), c = 7 / 15 → e = 3 / 8 → c > e) ∧
  (∀ (d e : ℚ), d = 5 / 12 → e = 3 / 8 → d > e) :=
sorry

end largest_fraction_l107_107622


namespace board_numbers_l107_107308

theorem board_numbers (a b c : ℕ) (h1 : a = 3) (h2 : b = 9) (h3 : c = 15)
    (op : ∀ x y z : ℕ, (x = y + z - t) → true)  -- simplifying the operation representation
    (min_number : ∃ x, x = 2013) : ∃ n m, n = 2019 ∧ m = 2025 := 
sorry

end board_numbers_l107_107308


namespace pieces_of_wood_for_chair_is_correct_l107_107501

-- Define the initial setup and constants
def total_pieces_of_wood := 672
def pieces_of_wood_per_table := 12
def number_of_tables := 24
def number_of_chairs := 48

-- Calculation in the conditions
def pieces_of_wood_used_for_tables := number_of_tables * pieces_of_wood_per_table
def pieces_of_wood_left_for_chairs := total_pieces_of_wood - pieces_of_wood_used_for_tables

-- Question and answer verification
def pieces_of_wood_per_chair := pieces_of_wood_left_for_chairs / number_of_chairs

theorem pieces_of_wood_for_chair_is_correct :
  pieces_of_wood_per_chair = 8 := 
by
  -- Proof omitted
  sorry

end pieces_of_wood_for_chair_is_correct_l107_107501


namespace compare_negatives_l107_107691

theorem compare_negatives : -2 > -3 :=
by
  sorry

end compare_negatives_l107_107691


namespace geometric_progression_vertex_l107_107201

theorem geometric_progression_vertex (a b c d : ℝ) (q : ℝ)
  (h1 : b = 1)
  (h2 : c = 2)
  (h3 : q = c / b)
  (h4 : a = b / q)
  (h5 : d = c * q) :
  a + d = 9 / 2 :=
sorry

end geometric_progression_vertex_l107_107201


namespace max_xy_l107_107895

open Real

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 6 * x + 8 * y = 72) (h4 : x = 2 * y) : 
  x * y = 25.92 := 
sorry

end max_xy_l107_107895


namespace abscissa_of_A_is_5_l107_107632

theorem abscissa_of_A_is_5
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A.1 = A.2 ∧ A.1 > 0)
  (hB : B = (5, 0))
  (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hC : C = ((A.1 + 5) / 2, A.2 / 2))
  (hD : D = (5 / 2, 5 / 2))
  (dot_product_eq : (B.1 - A.1, B.2 - A.2) • (D.1 - C.1, D.2 - C.2) = 0) :
  A.1 = 5 :=
sorry

end abscissa_of_A_is_5_l107_107632


namespace MN_equal_l107_107868

def M : Set ℝ := {x | ∃ (m : ℤ), x = Real.sin ((2 * m - 3) * Real.pi / 6)}
def N : Set ℝ := {y | ∃ (n : ℤ), y = Real.cos (n * Real.pi / 3)}

theorem MN_equal : M = N := by
  sorry

end MN_equal_l107_107868


namespace total_cost_of_books_l107_107348

theorem total_cost_of_books
  (C1 : ℝ)
  (C2 : ℝ)
  (H1 : C1 = 285.8333333333333)
  (H2 : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 2327.5 :=
by
  sorry

end total_cost_of_books_l107_107348


namespace mean_of_six_numbers_l107_107682

theorem mean_of_six_numbers (a b c d e f : ℚ) (h : a + b + c + d + e + f = 1 / 3) :
  (a + b + c + d + e + f) / 6 = 1 / 18 :=
by
  sorry

end mean_of_six_numbers_l107_107682


namespace arrangements_correctness_l107_107387

noncomputable def arrangements_of_groups (total mountaineers : ℕ) (familiar_with_route : ℕ) (required_in_each_group : ℕ) : ℕ :=
  sorry

theorem arrangements_correctness :
  arrangements_of_groups 10 4 2 = 120 :=
sorry

end arrangements_correctness_l107_107387


namespace operation_commutative_operation_associative_l107_107606

def my_operation (a b : ℝ) : ℝ := a * b + a + b

theorem operation_commutative (a b : ℝ) : my_operation a b = my_operation b a := by
  sorry

theorem operation_associative (a b c : ℝ) : my_operation (my_operation a b) c = my_operation a (my_operation b c) := by
  sorry

end operation_commutative_operation_associative_l107_107606


namespace patrick_savings_l107_107336

theorem patrick_savings :
  let bicycle_price := 150
  let saved_money := bicycle_price / 2
  let lent_money := 50
  saved_money - lent_money = 25 := by
  let bicycle_price := 150
  let saved_money := bicycle_price / 2
  let lent_money := 50
  sorry

end patrick_savings_l107_107336


namespace joyce_new_property_is_10_times_larger_l107_107893

theorem joyce_new_property_is_10_times_larger :
  let previous_property := 2
  let suitable_acres := 19
  let pond := 1
  let new_property := suitable_acres + pond
  new_property / previous_property = 10 := by {
    let previous_property := 2
    let suitable_acres := 19
    let pond := 1
    let new_property := suitable_acres + pond
    sorry
  }

end joyce_new_property_is_10_times_larger_l107_107893


namespace ferry_tourists_total_l107_107581

def series_sum (a d n : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem ferry_tourists_total :
  let t_0 := 90
  let d := -2
  let n := 9
  series_sum t_0 d n = 738 :=
by
  sorry

end ferry_tourists_total_l107_107581


namespace interval_is_correct_l107_107212

def total_population : ℕ := 2000
def sample_size : ℕ := 40
def interval_between_segments (N : ℕ) (n : ℕ) : ℕ := N / n

theorem interval_is_correct : interval_between_segments total_population sample_size = 50 :=
by
  sorry

end interval_is_correct_l107_107212


namespace grape_juice_amount_l107_107597

theorem grape_juice_amount (total_juice : ℝ)
  (orange_juice_percent : ℝ) (watermelon_juice_percent : ℝ)
  (orange_juice_amount : ℝ) (watermelon_juice_amount : ℝ)
  (grape_juice_amount : ℝ) :
  orange_juice_percent = 0.25 →
  watermelon_juice_percent = 0.40 →
  total_juice = 200 →
  orange_juice_amount = total_juice * orange_juice_percent →
  watermelon_juice_amount = total_juice * watermelon_juice_percent →
  grape_juice_amount = total_juice - orange_juice_amount - watermelon_juice_amount →
  grape_juice_amount = 70 :=
by
  sorry

end grape_juice_amount_l107_107597


namespace angle_BDE_60_l107_107915

noncomputable def is_isosceles_triangle (A B C : Type) (angle_BAC : ℝ) : Prop :=
angle_BAC = 20

noncomputable def equal_sides (BC BD BE : ℝ) : Prop :=
BC = BD ∧ BD = BE

theorem angle_BDE_60 (A B C D E : Type) (BC BD BE : ℝ) 
  (h1 : is_isosceles_triangle A B C 20) 
  (h2 : equal_sides BC BD BE) : 
  ∃ (angle_BDE : ℝ), angle_BDE = 60 :=
by
  sorry

end angle_BDE_60_l107_107915


namespace ways_to_sum_420_l107_107776

theorem ways_to_sum_420 : 
  (∃ n k : ℕ, n ≥ 2 ∧ 2 * k + n - 1 > 0 ∧ n * (2 * k + n - 1) = 840) → (∃ c, c = 11) :=
by
  sorry

end ways_to_sum_420_l107_107776


namespace total_selling_price_l107_107425

theorem total_selling_price
  (cost1 : ℝ) (cost2 : ℝ) (cost3 : ℝ) 
  (profit_percent1 : ℝ) (profit_percent2 : ℝ) (profit_percent3 : ℝ) :
  cost1 = 600 → cost2 = 450 → cost3 = 750 →
  profit_percent1 = 0.08 → profit_percent2 = 0.10 → profit_percent3 = 0.15 →
  (cost1 * (1 + profit_percent1) + cost2 * (1 + profit_percent2) + cost3 * (1 + profit_percent3)) = 2005.50 :=
by
  intros h1 h2 h3 p1 p2 p3
  simp [h1, h2, h3, p1, p2, p3]
  sorry

end total_selling_price_l107_107425


namespace sum_of_fourth_powers_of_consecutive_integers_l107_107316

-- Definitions based on conditions
def consecutive_squares_sum (x : ℤ) : Prop :=
  (x - 1)^2 + x^2 + (x + 1)^2 = 12246

-- Statement of the problem
theorem sum_of_fourth_powers_of_consecutive_integers (x : ℤ)
  (h : consecutive_squares_sum x) : 
  (x - 1)^4 + x^4 + (x + 1)^4 = 50380802 :=
sorry

end sum_of_fourth_powers_of_consecutive_integers_l107_107316


namespace total_peaches_in_each_basket_l107_107236

-- Define the given conditions
def red_peaches : ℕ := 7
def green_peaches : ℕ := 3

-- State the theorem
theorem total_peaches_in_each_basket : red_peaches + green_peaches = 10 :=
by
  -- Proof goes here, which we skip for now
  sorry

end total_peaches_in_each_basket_l107_107236


namespace logical_equivalence_l107_107580

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ ¬R) → ¬Q) ↔ (Q → (¬P ∨ R)) :=
by
  sorry

end logical_equivalence_l107_107580


namespace max_points_of_intersection_l107_107156

-- Definitions from the conditions
def circles := 2
def lines := 3

-- Define the problem of the greatest intersection number
theorem max_points_of_intersection (c : ℕ) (l : ℕ) (h_c : c = circles) (h_l : l = lines) : 
  (2 + (l * 2 * c) + (l * (l - 1) / 2)) = 17 :=
by
  rw [h_c, h_l]
  -- We have 2 points from circle intersections
  -- 12 points from lines intersections with circles
  -- 3 points from lines intersections with lines
  -- Hence, 2 + 12 + 3 = 17
  exact Eq.refl 17

end max_points_of_intersection_l107_107156


namespace twelve_percent_greater_l107_107005

theorem twelve_percent_greater :
  ∃ x : ℝ, x = 80 + (12 / 100) * 80 := sorry

end twelve_percent_greater_l107_107005


namespace parabola_focus_l107_107143

theorem parabola_focus (a b c : ℝ) (h k : ℝ) (p : ℝ) :
  (a = 4) →
  (b = -4) →
  (c = -3) →
  (h = -b / (2 * a)) →
  (k = a * h ^ 2 + b * h + c) →
  (p = 1 / (4 * a)) →
  (k + p = -4 + 1 / 16) →
  (h, k + p) = (1 / 2, -63 / 16) :=
by
  intros a_eq b_eq c_eq h_eq k_eq p_eq focus_eq
  rw [a_eq, b_eq, c_eq] at *
  sorry

end parabola_focus_l107_107143


namespace game_goal_impossible_l107_107602

-- Definition for initial setup
def initial_tokens : ℕ := 2013
def initial_piles : ℕ := 1

-- Definition for the invariant
def invariant (tokens piles : ℕ) : ℕ := tokens + piles

-- Initial value of the invariant constant
def initial_invariant : ℕ :=
  invariant initial_tokens initial_piles

-- Goal is to check if the final configuration is possible
theorem game_goal_impossible (n : ℕ) :
  (invariant (3 * n) n = initial_invariant) → false :=
by
  -- The invariant states 4n = initial_invariant which is 2014.
  -- Thus, we need to check if 2014 / 4 results in an integer.
  have invariant_expr : 4 * n = 2014 := by sorry
  have n_is_integer : 2014 % 4 = 0 := by sorry
  sorry

end game_goal_impossible_l107_107602


namespace quadratic_inequality_solution_l107_107837

theorem quadratic_inequality_solution
  (x : ℝ) 
  (h1 : ∀ x, x^2 + 2 * x - 3 > 0 ↔ x < -3 ∨ x > 1) :
  (2 * x^2 - 3 * x - 2 < 0) ↔ (-1 / 2 < x ∧ x < 2) :=
by {
  sorry
}

end quadratic_inequality_solution_l107_107837


namespace problem_l107_107144

noncomputable def f (x : ℝ) : ℝ := Real.sin x + x - Real.pi / 4
noncomputable def g (x : ℝ) : ℝ := Real.cos x - x + Real.pi / 4

theorem problem (x1 x2 : ℝ) (hx1 : 0 < x1 ∧ x1 < Real.pi / 2) (hx2 : 0 < x2 ∧ x2 < Real.pi / 2) :
  (∃! x, 0 < x ∧ x < Real.pi / 2 ∧ f x = 0) ∧ (∃! x, 0 < x ∧ x < Real.pi / 2 ∧ g x = 0) →
  x1 + x2 = Real.pi / 2 :=
by
  sorry -- Proof goes here

end problem_l107_107144


namespace remainder_of_2357916_div_8_l107_107685

theorem remainder_of_2357916_div_8 : (2357916 % 8) = 4 := by
  sorry

end remainder_of_2357916_div_8_l107_107685


namespace find_a_l107_107628

theorem find_a (a x y : ℤ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 1) : a = 3 :=
by
  sorry

end find_a_l107_107628


namespace find_m_range_l107_107457

def p (m : ℝ) : Prop := (4 - 4 * m) ≤ 0
def q (m : ℝ) : Prop := (5 - 2 * m) > 1

theorem find_m_range (m : ℝ) (hp_false : ¬ p m) (hq_true : q m) : 1 ≤ m ∧ m < 2 :=
by {
 sorry
}

end find_m_range_l107_107457


namespace least_positive_integer_congruences_l107_107555

theorem least_positive_integer_congruences :
  ∃ n : ℕ, 
    n > 0 ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 2) ∧ 
    (n % 6 = 3) ∧ 
    (n = 57) :=
by
  sorry

end least_positive_integer_congruences_l107_107555


namespace trajectory_eq_ellipse_l107_107207

theorem trajectory_eq_ellipse :
  (∀ M : ℝ × ℝ, (∀ r : ℝ, (M.1 - 4)^2 + M.2^2 = r^2 ∧ (M.1 + 4)^2 + M.2^2 = (10 - r)^2) → false) →
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) :=
by
  sorry

end trajectory_eq_ellipse_l107_107207


namespace loss_of_50_denoted_as_minus_50_l107_107111

def is_profit (x : Int) : Prop :=
  x > 0

def is_loss (x : Int) : Prop :=
  x < 0

theorem loss_of_50_denoted_as_minus_50 : is_loss (-50) :=
  by
    -- proof steps would go here
    sorry

end loss_of_50_denoted_as_minus_50_l107_107111


namespace find_n_l107_107892

theorem find_n (a b c : ℕ) (n : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : n > 2) 
    (h₃ : (a^n + b^n + c^n)^2 = 2 * (a^(2*n) + b^(2*n) + c^(2*n))) : n = 4 := 
sorry

end find_n_l107_107892


namespace round_trip_average_mileage_l107_107224

theorem round_trip_average_mileage 
  (d1 d2 : ℝ) (m1 m2 : ℝ)
  (h1 : d1 = 150) (h2 : d2 = 150)
  (h3 : m1 = 40) (h4 : m2 = 25) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 30.77 :=
by
  sorry

end round_trip_average_mileage_l107_107224


namespace find_A_l107_107960

namespace PolynomialDecomposition

theorem find_A (x A B C : ℝ)
  (h : (x^3 + 2 * x^2 - 17 * x - 30)⁻¹ = A / (x - 5) + B / (x + 2) + C / ((x + 2)^2)) :
  A = 1 / 49 :=
by sorry

end PolynomialDecomposition

end find_A_l107_107960


namespace total_cookies_l107_107702

theorem total_cookies (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) : bags * cookies_per_bag = 703 :=
by
  sorry

end total_cookies_l107_107702


namespace distance_to_lightning_l107_107405

noncomputable def distance_from_lightning (time_delay : ℕ) (speed_of_sound : ℕ) (feet_per_mile : ℕ) : ℚ :=
  (time_delay * speed_of_sound : ℕ) / feet_per_mile

theorem distance_to_lightning (time_delay : ℕ) (speed_of_sound : ℕ) (feet_per_mile : ℕ) :
  time_delay = 12 → speed_of_sound = 1120 → feet_per_mile = 5280 → distance_from_lightning time_delay speed_of_sound feet_per_mile = 2.5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end distance_to_lightning_l107_107405


namespace total_canoes_built_by_april_l107_107366

theorem total_canoes_built_by_april
  (initial : ℕ)
  (production_increase : ℕ → ℕ) 
  (total_canoes : ℕ) :
  initial = 5 →
  (∀ n, production_increase n = 3 * n) →
  total_canoes = initial + production_increase initial + production_increase (production_increase initial) + production_increase (production_increase (production_increase initial)) →
  total_canoes = 200 :=
by
  intros h_initial h_production h_total
  sorry

end total_canoes_built_by_april_l107_107366


namespace initial_investment_l107_107211

variable (P1 P2 π1 π2 : ℝ)

-- Given conditions
axiom h1 : π1 = 100
axiom h2 : π2 = 120

-- Revenue relation after the first transaction
axiom h3 : P2 = P1 + π1

-- Consistent profit relationship across transactions
axiom h4 : π2 = 0.2 * P2

-- To be proved
theorem initial_investment (P1 : ℝ) (h1 : π1 = 100) (h2 : π2 = 120) (h3 : P2 = P1 + π1) (h4 : π2 = 0.2 * P2) :
  P1 = 500 :=
sorry

end initial_investment_l107_107211


namespace f_neg_one_l107_107819

-- Assume the function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Conditions
-- 1. f(x) is odd: f(-x) = -f(x) for all x ∈ ℝ
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- 2. f(x) = 2^x for all x > 0
axiom f_pos : ∀ x : ℝ, x > 0 → f x = 2^x

-- Proof statement to be filled
theorem f_neg_one : f (-1) = -2 := 
by
  sorry

end f_neg_one_l107_107819


namespace base10_to_base8_440_l107_107158

theorem base10_to_base8_440 :
  ∃ k1 k2 k3,
    k1 = 6 ∧
    k2 = 7 ∧
    k3 = 0 ∧
    (440 = k1 * 64 + k2 * 8 + k3) ∧
    (64 = 8^2) ∧
    (8^3 > 440) :=
sorry

end base10_to_base8_440_l107_107158


namespace groups_needed_l107_107714

theorem groups_needed (h_camper_count : 36 > 0) (h_group_limit : 12 > 0) : 
  ∃ x : ℕ, x = 36 / 12 ∧ x = 3 := by
  sorry

end groups_needed_l107_107714


namespace mileage_interval_l107_107399

-- Define the distances driven each day
def d1 : ℕ := 135
def d2 : ℕ := 135 + 124
def d3 : ℕ := 159
def d4 : ℕ := 189

-- Define the total distance driven
def total_distance : ℕ := d1 + d2 + d3 + d4

-- Define the number of intervals (charges)
def number_of_intervals : ℕ := 6

-- Define the expected mileage interval for charging
def expected_interval : ℕ := 124

-- The theorem to prove that the mileage interval is approximately 124 miles
theorem mileage_interval : total_distance / number_of_intervals = expected_interval := by
  sorry

end mileage_interval_l107_107399


namespace hypotenuse_length_l107_107616

def triangle_hypotenuse := ∃ (a b c : ℚ) (x : ℚ), 
  a = 9 ∧ b = 3 * x + 6 ∧ c = x + 15 ∧ 
  a + b + c = 45 ∧ 
  a^2 + b^2 = c^2 ∧ 
  x = 15 / 4 ∧ 
  c = 75 / 4

theorem hypotenuse_length : triangle_hypotenuse :=
sorry

end hypotenuse_length_l107_107616


namespace monthly_rent_is_1300_l107_107849

def shop_length : ℕ := 10
def shop_width : ℕ := 10
def annual_rent_per_square_foot : ℕ := 156

def area_of_shop : ℕ := shop_length * shop_width
def annual_rent_for_shop : ℕ := annual_rent_per_square_foot * area_of_shop

def monthly_rent : ℕ := annual_rent_for_shop / 12

theorem monthly_rent_is_1300 : monthly_rent = 1300 := by
  sorry

end monthly_rent_is_1300_l107_107849


namespace students_in_all_three_workshops_l107_107285

-- Define the students counts and other conditions
def num_students : ℕ := 25
def num_dance : ℕ := 12
def num_chess : ℕ := 15
def num_robotics : ℕ := 11
def num_at_least_two : ℕ := 12

-- Define the proof statement
theorem students_in_all_three_workshops : 
  ∃ c : ℕ, c = 1 ∧ 
    (∃ a b d : ℕ, 
      a + b + c + d = num_at_least_two ∧
      num_students ≥ num_dance + num_chess + num_robotics - a - b - d - 2 * c
    ) := 
by
  sorry

end students_in_all_three_workshops_l107_107285


namespace range_independent_variable_l107_107940

def domain_of_function (x : ℝ) : Prop :=
  x ≥ -1 ∧ x ≠ 0

theorem range_independent_variable (x : ℝ) :
  domain_of_function x ↔ x ≥ -1 ∧ x ≠ 0 :=
by
  sorry

end range_independent_variable_l107_107940


namespace parabola_focus_coordinates_l107_107295

theorem parabola_focus_coordinates :
  (∃ f : ℝ × ℝ, f = (0, 2) ∧ ∀ x y : ℝ, y = (1/8) * x^2 ↔ f = (0, 2)) :=
sorry

end parabola_focus_coordinates_l107_107295


namespace jennifer_money_left_l107_107301

def money_left (initial_amount sandwich_fraction museum_fraction book_fraction : ℚ) : ℚ :=
  initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction)

theorem jennifer_money_left :
  money_left 150 (1/5) (1/6) (1/2) = 20 := by
  -- Proof goes here
  sorry

end jennifer_money_left_l107_107301


namespace solve_equation_1_solve_equation_2_l107_107964

theorem solve_equation_1 (x : ℝ) : x * (x + 2) = 2 * (x + 2) ↔ x = -2 ∨ x = 2 := 
by sorry

theorem solve_equation_2 (x : ℝ) : 3 * x^2 - x - 1 = 0 ↔ x = (1 + Real.sqrt 13) / 6 ∨ x = (1 - Real.sqrt 13) / 6 := 
by sorry

end solve_equation_1_solve_equation_2_l107_107964


namespace probability_cooking_is_one_fourth_l107_107698
-- Import the entirety of Mathlib to bring necessary libraries

-- Definition of total number of courses and favorable outcomes
def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

-- Define the probability of selecting "cooking"
def probability_of_cooking (favorable total : ℕ) : ℚ := favorable / total

-- Theorem stating the probability of selecting "cooking" is 1/4
theorem probability_cooking_is_one_fourth :
  probability_of_cooking favorable_outcomes total_courses = 1/4 := by
  -- Proof placeholder
  sorry

end probability_cooking_is_one_fourth_l107_107698


namespace calc_3_op_2_op_4_op_1_l107_107582

def op (a b : ℕ) : ℕ :=
match a, b with
| 1, 1 => 2 | 1, 2 => 3 | 1, 3 => 4 | 1, 4 => 1
| 2, 1 => 3 | 2, 2 => 1 | 2, 3 => 2 | 2, 4 => 4
| 3, 1 => 4 | 3, 2 => 2 | 3, 3 => 1 | 3, 4 => 3
| 4, 1 => 1 | 4, 2 => 4 | 4, 3 => 3 | 4, 4 => 2
| _, _  => 0 -- default case, though won't be used

theorem calc_3_op_2_op_4_op_1 : op (op 3 2) (op 4 1) = 3 :=
by
  sorry

end calc_3_op_2_op_4_op_1_l107_107582


namespace triangle_concurrency_l107_107272

-- Define Triangle Structure
structure Triangle (α : Type*) :=
(A B C : α)

-- Define Medians, Angle Bisectors, and Altitudes Concurrency Conditions
noncomputable def medians_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry
noncomputable def angle_bisectors_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry
noncomputable def altitudes_concurrent {α : Type*} [MetricSpace α] (T : Triangle α) : Prop := sorry

-- Main Theorem Statement
theorem triangle_concurrency {α : Type*} [MetricSpace α] (T : Triangle α) :
  medians_concurrent T ∧ angle_bisectors_concurrent T ∧ altitudes_concurrent T :=
by 
  -- Proof outline: Prove each concurrency condition
  sorry

end triangle_concurrency_l107_107272


namespace maximum_bunnies_drum_l107_107855

-- Define the conditions as provided in the problem
def drumsticks := ℕ -- Natural number type for simplicity
def drum := ℕ -- Natural number type for simplicity

structure Bunny :=
(drum_size : drum)
(stick_length : drumsticks)

def max_drumming_bunnies (bunnies : List Bunny) : ℕ := 
  -- Actual implementation to find the maximum number of drumming bunnies
  sorry

theorem maximum_bunnies_drum (bunnies : List Bunny) (h_size : bunnies.length = 7) : max_drumming_bunnies bunnies = 6 :=
by
  -- Proof of the theorem
  sorry

end maximum_bunnies_drum_l107_107855


namespace curve_is_circle_l107_107245

theorem curve_is_circle (ρ θ : ℝ) (h : ρ = 5 * Real.sin θ) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), ∀ (x y : ℝ),
  (x, y) = (ρ * Real.cos θ, ρ * Real.sin θ) → 
  (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2 :=
by
  existsi (0, 5 / 2), 5 / 2
  sorry

end curve_is_circle_l107_107245


namespace total_flowers_l107_107965

def number_of_flowers (F : ℝ) : Prop :=
  let vases := (F - 7.0) / 6.0
  vases = 6.666666667

theorem total_flowers : number_of_flowers 47.0 :=
by
  sorry

end total_flowers_l107_107965


namespace triangle_a_c_sin_A_minus_B_l107_107735

theorem triangle_a_c_sin_A_minus_B (a b c : ℝ) (A B C : ℝ):
  a + c = 6 → b = 2 → Real.cos B = 7/9 →
  a = 3 ∧ c = 3 ∧ Real.sin (A - B) = (10 * Real.sqrt 2) / 27 :=
by
  intro h1 h2 h3
  sorry

end triangle_a_c_sin_A_minus_B_l107_107735


namespace kite_cost_l107_107372

variable (initial_amount : ℕ) (cost_frisbee : ℕ) (amount_left : ℕ)

theorem kite_cost (initial_amount : ℕ) (cost_frisbee : ℕ) (amount_left : ℕ) (h_initial_amount : initial_amount = 78) (h_cost_frisbee : cost_frisbee = 9) (h_amount_left : amount_left = 61) : 
  initial_amount - amount_left - cost_frisbee = 8 :=
by
  -- Proof can be completed here
  sorry

end kite_cost_l107_107372


namespace expansion_identity_l107_107528

theorem expansion_identity : 121 + 2 * 11 * 9 + 81 = 400 := by
  sorry

end expansion_identity_l107_107528


namespace ratio_of_inscribed_squares_l107_107782

theorem ratio_of_inscribed_squares (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (hx : x = 60 / 17) (hy : y = 3) :
  x / y = 20 / 17 :=
by
  sorry

end ratio_of_inscribed_squares_l107_107782


namespace total_area_is_71_l107_107916

noncomputable def area_of_combined_regions 
  (PQ QR RS TU : ℕ) 
  (PQRSTU_is_rectangle : true) 
  (right_angles : true): ℕ :=
  let Area_PQRSTU := PQ * QR
  let VU := TU - PQ
  let WT := TU - RS
  let Area_triangle_PVU := (1 / 2) * VU * PQ
  let Area_triangle_RWT := (1 / 2) * WT * RS
  Area_PQRSTU + Area_triangle_PVU + Area_triangle_RWT

theorem total_area_is_71
  (PQ QR RS TU : ℕ) 
  (h1 : PQ = 8)
  (h2 : QR = 6)
  (h3 : RS = 5)
  (h4 : TU = 10)
  (PQRSTU_is_rectangle : true)
  (right_angles : true) :
  area_of_combined_regions PQ QR RS TU PQRSTU_is_rectangle right_angles = 71 :=
by
  -- The proof is omitted as per the instructions
  sorry

end total_area_is_71_l107_107916


namespace problem_1_problem_2_l107_107199

-- Definitions of the given probabilities
def prob_A : ℚ := 2/3
def prob_B : ℚ := 1/4
def prob_C : ℚ := 2/5

-- Independence implies that the probabilities of combined events are products of individual probabilities.
-- To avoid unnecessary complications, we assume independence holds true without proof.
axiom independence : ∀ A B C : Prop, (A ∧ B ∧ C) ↔ (A ∧ B) ∧ C

-- Problem statement for part (1)
theorem problem_1 : prob_A * prob_B * prob_C = 1/15 := by
  sorry

-- Helper definitions for probabilities of not visiting
def not_prob_A : ℚ := 1 - prob_A
def not_prob_B : ℚ := 1 - prob_B
def not_prob_C : ℚ := 1 - prob_C

-- Problem statement for part (2)
theorem problem_2 : (prob_A * not_prob_B * not_prob_C + not_prob_A * prob_B * not_prob_C + not_prob_A * not_prob_B * prob_C) = 9/20 := by
  sorry

end problem_1_problem_2_l107_107199


namespace arithmetic_sequence_max_sum_l107_107681

-- Condition: first term is 23
def a1 : ℤ := 23

-- Condition: common difference is -2
def d : ℤ := -2

-- Sum of the first n terms of the arithmetic sequence
def Sn (n : ℕ) : ℤ := n * a1 + (n * (n - 1)) / 2 * d

-- Problem Statement: Prove the maximum value of Sn(n)
theorem arithmetic_sequence_max_sum : ∃ n : ℕ, Sn n = 144 :=
sorry

end arithmetic_sequence_max_sum_l107_107681


namespace find_m_value_l107_107440

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem find_m_value :
  ∃ m : ℝ, (∀ x ∈ (Set.Icc 0 3), f x m ≤ 1) ∧ (∃ x ∈ (Set.Icc 0 3), f x m = 1) ↔ m = -2 :=
by
  sorry

end find_m_value_l107_107440


namespace max_k_consecutive_sum_2_times_3_pow_8_l107_107625

theorem max_k_consecutive_sum_2_times_3_pow_8 :
  ∃ k : ℕ, 0 < k ∧ 
           (∃ n : ℕ, 2 * 3^8 = (k * (2 * n + k + 1)) / 2) ∧
           (∀ k' : ℕ, (∃ n' : ℕ, 0 < k' ∧ 2 * 3^8 = (k' * (2 * n' + k' + 1)) / 2) → k' ≤ 81) :=
sorry

end max_k_consecutive_sum_2_times_3_pow_8_l107_107625


namespace trigonometric_identity_l107_107219

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ ∈ Set.Ico 0 Real.pi) (hθ2 : Real.cos θ * (Real.sin θ + Real.cos θ) = 1) :
  θ = 0 ∨ θ = Real.pi / 4 :=
sorry

end trigonometric_identity_l107_107219


namespace andrew_bought_mangoes_l107_107904

theorem andrew_bought_mangoes (m : ℕ) 
    (grapes_cost : 6 * 74 = 444) 
    (mangoes_cost : m * 59 = total_mangoes_cost) 
    (total_cost_eq_975 : 444 + total_mangoes_cost = 975) 
    (total_cost := 444 + total_mangoes_cost) 
    (total_mangoes_cost := 59 * m) 
    : m = 9 := 
sorry

end andrew_bought_mangoes_l107_107904


namespace multiplication_of_variables_l107_107738

theorem multiplication_of_variables 
  (a b c d : ℚ)
  (h1 : 3 * a + 2 * b + 4 * c + 6 * d = 48)
  (h2 : 4 * (d + c) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : 2 * c - 2 = d) :
  a * b * c * d = -58735360 / 81450625 := 
sorry

end multiplication_of_variables_l107_107738


namespace johns_profit_is_200_l107_107876

def num_woodburnings : ℕ := 20
def price_per_woodburning : ℕ := 15
def cost_of_wood : ℕ := 100
def total_revenue : ℕ := num_woodburnings * price_per_woodburning
def profit : ℕ := total_revenue - cost_of_wood

theorem johns_profit_is_200 : profit = 200 :=
by
  -- proof steps go here
  sorry

end johns_profit_is_200_l107_107876


namespace range_of_a_plus_b_l107_107725

noncomputable def range_of_sum_of_sides (a b : ℝ) (c : ℝ) : Prop :=
  (2 < a + b ∧ a + b ≤ 4)

theorem range_of_a_plus_b
  (a b c : ℝ)
  (h1 : (2 * (b ^ 2 - (1/2) * a * b) = b ^ 2 + 4 - a ^ 2))
  (h2 : c = 2) :
  range_of_sum_of_sides a b c :=
by
  -- Proof would go here, but it's omitted as per the instructions.
  sorry

end range_of_a_plus_b_l107_107725


namespace soccer_ball_cost_l107_107527

theorem soccer_ball_cost :
  ∃ x y : ℝ, x + y = 100 ∧ 2 * x + 3 * y = 262 ∧ x = 38 :=
by
  sorry

end soccer_ball_cost_l107_107527


namespace contradiction_proof_l107_107330

theorem contradiction_proof (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
sorry

end contradiction_proof_l107_107330


namespace sum_of_squares_consecutive_nat_l107_107246

theorem sum_of_squares_consecutive_nat (n : ℕ) (h : n = 26) : (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2 = 2030 :=
by
  sorry

end sum_of_squares_consecutive_nat_l107_107246


namespace evaluate_expression_l107_107343

noncomputable def M (x y : ℝ) : ℝ := if x < y then y else x
noncomputable def m (x y : ℝ) : ℝ := if x < y then x else y

theorem evaluate_expression
  (p q r s t : ℝ)
  (h1 : p < q)
  (h2 : q < r)
  (h3 : r < s)
  (h4 : s < t)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ t ∧ t ≠ p ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ s ∧ q ≠ t ∧ r ≠ t):
  M (M p (m q r)) (m s (m p t)) = q := 
sorry

end evaluate_expression_l107_107343


namespace nat_digit_problem_l107_107505

theorem nat_digit_problem :
  ∀ n : Nat, (n % 10 = (2016 * (n / 2016)) % 10) → (n = 4032 ∨ n = 8064 ∨ n = 12096 ∨ n = 16128) :=
by
  sorry

end nat_digit_problem_l107_107505


namespace gcd_is_3_l107_107830

def gcd_6273_14593 : ℕ := Nat.gcd 6273 14593

theorem gcd_is_3 : gcd_6273_14593 = 3 :=
by
  sorry

end gcd_is_3_l107_107830


namespace f_monotonic_intervals_f_extreme_values_l107_107221

def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Monotonicity intervals
theorem f_monotonic_intervals (x : ℝ) : 
  (x < -2 → deriv f x > 0) ∧ 
  (-2 < x ∧ x < 2 → deriv f x < 0) ∧ 
  (2 < x → deriv f x > 0) := 
sorry

-- Extreme values
theorem f_extreme_values :
  f (-2) = 16 ∧ f (2) = -16 :=
sorry

end f_monotonic_intervals_f_extreme_values_l107_107221


namespace petyas_square_is_larger_l107_107083

noncomputable def side_petya_square (a b : ℝ) : ℝ :=
  a * b / (a + b)

noncomputable def side_vasya_square (a b : ℝ) : ℝ :=
  a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petyas_square_is_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  side_petya_square a b > side_vasya_square a b := by
  sorry

end petyas_square_is_larger_l107_107083


namespace sum_of_six_terms_l107_107504

theorem sum_of_six_terms (a1 : ℝ) (S4 : ℝ) (d : ℝ) (a1_eq : a1 = 1 / 2) (S4_eq : S4 = 20) :
  S4 = (4 * a1 + (4 * (4 - 1) / 2) * d) → (S4 = 20) →
  (6 * a1 + (6 * (6 - 1) / 2) * d = 48) :=
by
  intros
  sorry

end sum_of_six_terms_l107_107504


namespace problem_1_l107_107136

theorem problem_1 (f : ℝ → ℝ) (hf_mul : ∀ x y : ℝ, f (x * y) = f x + f y) (hf_4 : f 4 = 2) : f (Real.sqrt 2) = 1 / 2 :=
sorry

end problem_1_l107_107136


namespace min_distance_eq_sqrt2_l107_107172

open Real

variables {P Q : ℝ × ℝ}
variables {x y : ℝ}

/-- Given that point P is on the curve y = e^x and point Q is on the curve y = ln x, prove that the minimum value of the distance |PQ| is sqrt(2). -/
theorem min_distance_eq_sqrt2 : 
  (P.2 = exp P.1) ∧ (Q.2 = log Q.1) → (dist P Q) = sqrt 2 :=
by
  sorry

end min_distance_eq_sqrt2_l107_107172


namespace unique_solution_only_a_is_2_l107_107805

noncomputable def unique_solution_inequality (a : ℝ) : Prop :=
  ∀ (p : ℝ → ℝ), (∀ x, 0 ≤ p x ∧ p x ≤ 1 ∧ p x = x^2 - a * x + a) → 
  ∃! x, p x = 1

theorem unique_solution_only_a_is_2 (a : ℝ) (h : unique_solution_inequality a) : a = 2 :=
sorry

end unique_solution_only_a_is_2_l107_107805


namespace arithmetic_prog_triangle_l107_107898

theorem arithmetic_prog_triangle (a b c : ℝ) (h : a < b ∧ b < c ∧ 2 * b = a + c)
    (hα : ∀ t, t = a ↔ t = min a (min b c))
    (hγ : ∀ t, t = c ↔ t = max a (max b c)) :
    3 * (Real.tan (α / 2)) * (Real.tan (γ / 2)) = 1 := sorry

end arithmetic_prog_triangle_l107_107898


namespace li_family_cinema_cost_l107_107240

theorem li_family_cinema_cost :
  let standard_ticket_price := 10
  let child_discount := 0.4
  let senior_discount := 0.3
  let handling_fee := 5
  let num_adults := 2
  let num_children := 1
  let num_seniors := 1
  let child_ticket_price := (1 - child_discount) * standard_ticket_price
  let senior_ticket_price := (1 - senior_discount) * standard_ticket_price
  let total_ticket_cost := num_adults * standard_ticket_price + num_children * child_ticket_price + num_seniors * senior_ticket_price
  let final_cost := total_ticket_cost + handling_fee
  final_cost = 38 :=
by
  sorry

end li_family_cinema_cost_l107_107240


namespace sum_of_acute_angles_l107_107031

theorem sum_of_acute_angles (α β : Real) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h1 : Real.sin α = 2 * Real.sqrt 5 / 5)
    (h2 : Real.sin β = 3 * Real.sqrt 10 / 10) :
    α + β = 3 * Real.pi / 4 :=
sorry

end sum_of_acute_angles_l107_107031


namespace least_integer_with_remainders_l107_107091

theorem least_integer_with_remainders :
  ∃ M : ℕ, 
    M % 6 = 5 ∧
    M % 7 = 6 ∧
    M % 9 = 8 ∧
    M % 10 = 9 ∧
    M % 11 = 10 ∧
    M = 6929 :=
by
  sorry

end least_integer_with_remainders_l107_107091


namespace average_marks_of_first_class_l107_107553

theorem average_marks_of_first_class (n1 n2 : ℕ) (avg2 avg_all : ℝ)
  (h_n1 : n1 = 25) (h_n2 : n2 = 40) (h_avg2 : avg2 = 65) (h_avg_all : avg_all = 59.23076923076923) :
  ∃ (A : ℝ), A = 50 :=
by 
  sorry

end average_marks_of_first_class_l107_107553


namespace original_avg_is_40_l107_107945

noncomputable def original_average (A : ℝ) := (15 : ℝ) * A

noncomputable def new_sum (A : ℝ) := (15 : ℝ) * A + 15 * (15 : ℝ)

theorem original_avg_is_40 (A : ℝ) (h : new_sum A / 15 = 55) :
  A = 40 :=
by sorry

end original_avg_is_40_l107_107945


namespace exactly_three_correct_is_impossible_l107_107987

theorem exactly_three_correct_is_impossible (n : ℕ) (hn : n = 5) (f : Fin n → Fin n) :
  (∃ S : Finset (Fin n), S.card = 3 ∧ ∀ i ∈ S, f i = i) → False :=
by
  intros h
  sorry

end exactly_three_correct_is_impossible_l107_107987


namespace exactly_one_equals_xx_plus_xx_l107_107073

theorem exactly_one_equals_xx_plus_xx (x : ℝ) (hx : x > 0) :
  let expr1 := 2 * x^x
  let expr2 := x^(2*x)
  let expr3 := (2*x)^x
  let expr4 := (2*x)^(2*x)
  (expr1 = x^x + x^x) ∧ (¬(expr2 = x^x + x^x)) ∧ (¬(expr3 = x^x + x^x)) ∧ (¬(expr4 = x^x + x^x)) := 
by
  sorry

end exactly_one_equals_xx_plus_xx_l107_107073


namespace robert_finite_moves_l107_107456

noncomputable def onlyFiniteMoves (numbers : List ℕ) : Prop :=
  ∀ (a b : ℕ), a > b → ∃ (moves : ℕ), moves < numbers.length

theorem robert_finite_moves (numbers : List ℕ) :
  onlyFiniteMoves numbers := sorry

end robert_finite_moves_l107_107456


namespace book_selling_price_l107_107054

def cost_price : ℕ := 225
def profit_percentage : ℚ := 0.20
def selling_price := cost_price + (profit_percentage * cost_price)

theorem book_selling_price :
  selling_price = 270 :=
by
  sorry

end book_selling_price_l107_107054


namespace daily_wage_c_l107_107453

-- Definitions according to the conditions
def days_worked_a : ℕ := 6
def days_worked_b : ℕ := 9
def days_worked_c : ℕ := 4

def ratio_wages : ℕ × ℕ × ℕ := (3, 4, 5)
def total_earning : ℕ := 1628

-- Goal: Prove that the daily wage of c is Rs. 110
theorem daily_wage_c : (5 * (total_earning / (18 + 36 + 20))) = 110 :=
by
  sorry

end daily_wage_c_l107_107453


namespace sarah_reads_100_words_per_page_l107_107753

noncomputable def words_per_page (W_pages : ℕ) (books : ℕ) (hours : ℕ) (pages_per_book : ℕ) (words_per_minute : ℕ) : ℕ :=
  (words_per_minute * 60 * hours) / books / pages_per_book

theorem sarah_reads_100_words_per_page :
  words_per_page 80 6 20 80 40 = 100 := 
sorry

end sarah_reads_100_words_per_page_l107_107753


namespace find_b_l107_107388

theorem find_b (a b c : ℝ) (k₁ k₂ k₃ : ℤ) :
  (a + b) / 2 = 40 ∧
  (b + c) / 2 = 43 ∧
  (a + c) / 2 = 44 ∧
  a + b = 5 * k₁ ∧
  b + c = 5 * k₂ ∧
  a + c = 5 * k₃
  → b = 40 :=
by {
  sorry
}

end find_b_l107_107388


namespace total_working_days_l107_107799

theorem total_working_days 
  (D : ℕ)
  (A : ℝ)
  (B : ℝ)
  (h1 : A * (D - 2) = 80)
  (h2 : B * (D - 5) = 63)
  (h3 : A * (D - 5) = B * (D - 2) + 2) :
  D = 32 := 
sorry

end total_working_days_l107_107799


namespace cost_per_square_meter_l107_107492

noncomputable def costPerSquareMeter 
  (length : ℝ) (breadth : ℝ) (width : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / ((length * width) + (breadth * width) - (width * width))

theorem cost_per_square_meter (H1 : length = 110)
                              (H2 : breadth = 60)
                              (H3 : width = 10)
                              (H4 : total_cost = 4800) : 
  costPerSquareMeter length breadth width total_cost = 3 := 
by
  sorry

end cost_per_square_meter_l107_107492


namespace find_triplets_l107_107730

theorem find_triplets (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a ^ b ∣ b ^ c - 1) ∧ (a ^ c ∣ c ^ b - 1)) ↔ (a = 1 ∨ (b = 1 ∧ c = 1)) :=
by sorry

end find_triplets_l107_107730


namespace square_area_l107_107719

noncomputable def side_length_x (x : ℚ) : Prop :=
5 * x - 20 = 30 - 4 * x

noncomputable def side_length_s : ℚ :=
70 / 9

noncomputable def area_of_square : ℚ :=
(side_length_s)^2

theorem square_area (x : ℚ) (h : side_length_x x) : area_of_square = 4900 / 81 :=
sorry

end square_area_l107_107719


namespace dividend_is_correct_l107_107460

def divisor : ℕ := 17
def quotient : ℕ := 9
def remainder : ℕ := 6

def calculate_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem dividend_is_correct : calculate_dividend divisor quotient remainder = 159 :=
  by sorry

end dividend_is_correct_l107_107460


namespace solution_set_inequality_system_l107_107467

theorem solution_set_inequality_system (x : ℝ) :
  (x - 3 < 2 ∧ 3 * x + 1 ≥ 2 * x) ↔ (-1 ≤ x ∧ x < 5) := by
  sorry

end solution_set_inequality_system_l107_107467


namespace simplify_expression_l107_107161

theorem simplify_expression (x : ℕ) (h : x = 100) :
  (x + 1) * (x - 1) + x * (2 - x) + (x - 1) ^ 2 = 10000 := by
  sorry

end simplify_expression_l107_107161


namespace negation_of_p_range_of_m_if_p_false_l107_107836

open Real

noncomputable def neg_p (m : ℝ) : Prop :=
∀ x : ℝ, x^2 - m*x - m > 0

theorem negation_of_p (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m*x - m ≤ 0) ↔ neg_p m := 
by sorry

theorem range_of_m_if_p_false : 
  (∀ m : ℝ, neg_p m → (-4 < m ∧ m < 0)) :=
by sorry

end negation_of_p_range_of_m_if_p_false_l107_107836


namespace books_finished_l107_107142

theorem books_finished (miles_traveled : ℕ) (miles_per_book : ℕ) (h_travel : miles_traveled = 6760) (h_rate : miles_per_book = 450) : (miles_traveled / miles_per_book) = 15 :=
by {
  -- Proof will be inserted here
  sorry
}

end books_finished_l107_107142


namespace age_difference_l107_107536

/-- The age difference between each child d -/
theorem age_difference (d : ℝ) 
  (h1 : ∃ a b c e : ℝ, d = a ∧ 2*d = b ∧ 3*d = c ∧ 4*d = e)
  (h2 : 12 + (12 - d) + (12 - 2*d) + (12 - 3*d) + (12 - 4*d) = 40) : 
  d = 2 := 
sorry

end age_difference_l107_107536


namespace find_x_l107_107656

def myOperation (x y : ℝ) : ℝ := 2 * x * y

theorem find_x (x : ℝ) (h : myOperation 9 (myOperation 4 x) = 720) : x = 5 :=
by
  sorry

end find_x_l107_107656


namespace value_of_x_is_4_l107_107451

variable {A B C D E F G H P : ℕ}

theorem value_of_x_is_4 (h1 : 5 + A + B = 19)
                        (h2 : A + B + C = 19)
                        (h3 : C + D + E = 19)
                        (h4 : D + E + F = 19)
                        (h5 : F + x + G = 19)
                        (h6 : x + G + H = 19)
                        (h7 : H + P + 10 = 19) :
                        x = 4 :=
by
  sorry

end value_of_x_is_4_l107_107451


namespace find_starting_number_l107_107483

theorem find_starting_number (n : ℕ) (h : ((28 + n) / 2) = 18) : n = 8 :=
sorry

end find_starting_number_l107_107483


namespace sum_a6_to_a9_l107_107771

-- Given definitions and conditions
def sequence_sum (n : ℕ) : ℕ := n^3
def a (n : ℕ) : ℕ := sequence_sum (n + 1) - sequence_sum n

-- Theorem to be proved
theorem sum_a6_to_a9 : a 6 + a 7 + a 8 + a 9 = 604 :=
by sorry

end sum_a6_to_a9_l107_107771


namespace brian_oranges_is_12_l107_107768

-- Define the number of oranges the person has
def person_oranges : Nat := 12

-- Define the number of oranges Brian has, which is zero fewer than the person's oranges
def brian_oranges : Nat := person_oranges - 0

-- The theorem stating that Brian has 12 oranges
theorem brian_oranges_is_12 : brian_oranges = 12 :=
by
  -- Proof is omitted
  sorry

end brian_oranges_is_12_l107_107768


namespace min_value_expression_l107_107186

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) ≥ 18 :=
by
  sorry

end min_value_expression_l107_107186


namespace shaded_region_area_l107_107206

noncomputable def area_of_shaded_region (a b c d : ℝ) (area_rect : ℝ) : ℝ :=
  let dg : ℝ := (a * d) / (c + d)
  let area_triangle : ℝ := 0.5 * dg * b
  area_rect - area_triangle

theorem shaded_region_area :
  area_of_shaded_region 12 5 12 4 (4 * 5) = 85 / 8 :=
by
  simp [area_of_shaded_region]
  sorry

end shaded_region_area_l107_107206


namespace no_quaint_two_digit_integers_l107_107446

theorem no_quaint_two_digit_integers :
  ∀ x : ℕ, 10 ≤ x ∧ x < 100 ∧ (∃ a b : ℕ, x = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) →  ¬(10 * x.div 10 + x % 10 = (x.div 10) + (x % 10)^3) :=
by
  sorry

end no_quaint_two_digit_integers_l107_107446


namespace sum_ratio_l107_107535

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ := 
  a1 * (1 - q^n) / (1 - q)

theorem sum_ratio (a1 q : ℝ) 
  (h : 8 * (a1 * q) + (a1 * q^4) = 0) :
  geometric_sequence_sum a1 q 6 / geometric_sequence_sum a1 q 3 = -7 := 
by
  sorry

end sum_ratio_l107_107535


namespace hyperbola_eccentricity_l107_107722

theorem hyperbola_eccentricity (a b c : ℝ) (h₁ : 2 * a = 16) (h₂ : 2 * b = 12) (h₃ : c = Real.sqrt (a^2 + b^2)) :
  (c / a) = 5 / 4 :=
by
  sorry

end hyperbola_eccentricity_l107_107722


namespace fraction_of_students_speak_foreign_language_l107_107947

noncomputable def students_speak_foreign_language_fraction (M F : ℕ) (h1 : M = F) (m_frac : ℚ) (f_frac : ℚ) : ℚ :=
  ((3 / 5) * M + (2 / 3) * F) / (M + F)

theorem fraction_of_students_speak_foreign_language (M F : ℕ) (h1 : M = F) :
  students_speak_foreign_language_fraction M F h1 (3 / 5) (2 / 3) = 19 / 30 :=
by 
  sorry

end fraction_of_students_speak_foreign_language_l107_107947


namespace ratio_equivalence_l107_107551

theorem ratio_equivalence (x : ℝ) (h : 3 / x = 3 / 16) : x = 16 := 
by
  sorry

end ratio_equivalence_l107_107551


namespace sum_of_altitudes_l107_107183

theorem sum_of_altitudes (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : a^2 + b^2 = c^2) : a + b = 21 :=
by
  -- Using the provided hypotheses, the proof would ensure a + b = 21.
  sorry

end sum_of_altitudes_l107_107183


namespace find_value_of_expression_l107_107520

variable {x : ℝ}

theorem find_value_of_expression (h : x^2 - 2 * x = 3) : 3 * x^2 - 6 * x - 4 = 5 :=
sorry

end find_value_of_expression_l107_107520


namespace find_m_l107_107605

theorem find_m (m : ℤ) (h0 : -90 ≤ m) (h1 : m ≤ 90) (h2 : Real.sin (m * Real.pi / 180) = Real.sin (710 * Real.pi / 180)) : m = -10 :=
sorry

end find_m_l107_107605


namespace find_b_minus_a_l107_107357

noncomputable def rotate_90_counterclockwise (x y xc yc : ℝ) : ℝ × ℝ :=
  (xc + (-(y - yc)), yc + (x - xc))

noncomputable def reflect_about_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem find_b_minus_a (a b : ℝ) :
  let xc := 2
  let yc := 3
  let P := (a, b)
  let P_rotated := rotate_90_counterclockwise a b xc yc
  let P_reflected := reflect_about_y_eq_x P_rotated.1 P_rotated.2
  P_reflected = (4, 1) →
  b - a = 1 :=
by
  intros
  sorry

end find_b_minus_a_l107_107357


namespace discounted_price_is_correct_l107_107097

def marked_price : ℕ := 125
def discount_rate : ℚ := 4 / 100

def calculate_discounted_price (marked_price : ℕ) (discount_rate : ℚ) : ℚ :=
  marked_price - (discount_rate * marked_price)

theorem discounted_price_is_correct :
  calculate_discounted_price marked_price discount_rate = 120 := by
  sorry

end discounted_price_is_correct_l107_107097


namespace limit_for_regular_pay_l107_107872

theorem limit_for_regular_pay 
  (x : ℕ) 
  (regular_pay_rate : ℕ := 3) 
  (overtime_pay_rate : ℕ := 6) 
  (total_pay : ℕ := 186) 
  (overtime_hours : ℕ := 11) 
  (H : 3 * x + (6 * 11) = 186) 
  :
  x = 40 :=
sorry

end limit_for_regular_pay_l107_107872


namespace distance_between_street_lights_l107_107883

theorem distance_between_street_lights :
  ∀ (n : ℕ) (L : ℝ), n = 18 → L = 16.4 → 8 > 0 →
  (L / (8 : ℕ) = 2.05) :=
by
  intros n L h_n h_L h_nonzero
  sorry

end distance_between_street_lights_l107_107883


namespace number_of_terms_in_arithmetic_sequence_l107_107831

/-- Define the conditions. -/
def a : ℕ := 2
def d : ℕ := 5
def a_n : ℕ := 57

/-- Define the proof problem. -/
theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, a_n = a + (n - 1) * d ∧ n = 12 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l107_107831


namespace min_M_inequality_l107_107734

noncomputable def M_min : ℝ := 9 * Real.sqrt 2 / 32

theorem min_M_inequality :
  ∀ (a b c : ℝ),
    abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2))
    ≤ M_min * (a^2 + b^2 + c^2)^2 :=
by
  sorry

end min_M_inequality_l107_107734


namespace polynomial_condition_degree_n_l107_107522

open Polynomial

theorem polynomial_condition_degree_n 
  (P_n : ℤ[X]) (n : ℕ) (hn_pos : 0 < n) (hn_deg : P_n.degree = n) 
  (hx0 : P_n.eval 0 = 0)
  (hx_conditions : ∃ (a : ℤ) (b : Fin n → ℤ), ∀ i, P_n.eval (b i) = n) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := 
sorry

end polynomial_condition_degree_n_l107_107522


namespace apple_price_equals_oranges_l107_107621

theorem apple_price_equals_oranges (A O : ℝ) (H1 : A = 28 * O) (H2 : 45 * A + 60 * O = 1350) (H3 : 30 * A + 40 * O = 900) : A = 28 * O :=
by
  sorry

end apple_price_equals_oranges_l107_107621


namespace geometric_sequence_a_eq_neg4_l107_107395

theorem geometric_sequence_a_eq_neg4 
    (a : ℝ)
    (h : (2 * a + 2) ^ 2 = a * (3 * a + 3)) : 
    a = -4 :=
sorry

end geometric_sequence_a_eq_neg4_l107_107395


namespace arithmetic_sequence_sum_l107_107045

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (S_10_eq : S 10 = 20) (S_20_eq : S 20 = 15) :
  S 30 = -15 :=
by
  sorry

end arithmetic_sequence_sum_l107_107045


namespace distance_between_clocks_centers_l107_107890

variable (M m : ℝ)

theorem distance_between_clocks_centers :
  ∃ (c : ℝ), (|c| = (1/2) * (M + m)) := by
  sorry

end distance_between_clocks_centers_l107_107890


namespace surface_area_combination_l107_107789

noncomputable def smallest_surface_area : ℕ :=
  let s1 := 3
  let s2 := 5
  let s3 := 8
  let surface_area := 6 * (s1 * s1 + s2 * s2 + s3 * s3)
  let overlap_area := (s1 * s1) * 4 + (s2 * s2) * 2 
  surface_area - overlap_area

theorem surface_area_combination :
  smallest_surface_area = 502 :=
by
  -- Proof goes here
  sorry

end surface_area_combination_l107_107789


namespace slices_per_person_eq_three_l107_107429

variables (num_people : ℕ) (slices_per_pizza : ℕ) (num_pizzas : ℕ)

theorem slices_per_person_eq_three (h1 : num_people = 18) (h2 : slices_per_pizza = 9) (h3 : num_pizzas = 6) : 
  (num_pizzas * slices_per_pizza) / num_people = 3 :=
sorry

end slices_per_person_eq_three_l107_107429


namespace cost_price_of_watch_l107_107919

theorem cost_price_of_watch (C : ℝ) (h1 : ∃ C, 0.91 * C + 220 = 1.04 * C) : C = 1692.31 :=
sorry  -- proof to be provided

end cost_price_of_watch_l107_107919


namespace quadratic_function_characterization_l107_107002

variable (f : ℝ → ℝ)

def quadratic_function_satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 = 2) ∧ (∀ x, f (x + 1) - f x = 2 * x - 1)

theorem quadratic_function_characterization
  (hf : quadratic_function_satisfies_conditions f) : 
  (∀ x, f x = x^2 - 2 * x + 2) ∧ 
  (f (-1) = 5) ∧ 
  (f 1 = 1) ∧ 
  (f 2 = 2) := by
sorry

end quadratic_function_characterization_l107_107002


namespace compare_two_sqrt_three_with_three_l107_107780

theorem compare_two_sqrt_three_with_three : 2 * Real.sqrt 3 > 3 :=
sorry

end compare_two_sqrt_three_with_three_l107_107780


namespace problem_distribution_l107_107208

theorem problem_distribution:
  let num_problems := 6
  let num_friends := 15
  (num_friends ^ num_problems) = 11390625 :=
by sorry

end problem_distribution_l107_107208


namespace area_of_triangle_ABC_l107_107636

variable (A : ℝ) -- Area of the triangle ABC
variable (S_heptagon : ℝ) -- Area of the heptagon ADECFGH
variable (S_overlap : ℝ) -- Overlapping area after folding

-- Given conditions
axiom ratio_condition : S_heptagon = (5 / 7) * A
axiom overlap_condition : S_overlap = 8

-- Proof statement
theorem area_of_triangle_ABC :
  A = 28 := by
  sorry

end area_of_triangle_ABC_l107_107636


namespace problem_l107_107592

theorem problem (d r : ℕ) (a b c : ℕ) (ha : a = 1059) (hb : b = 1417) (hc : c = 2312)
  (h1 : d ∣ (b - a)) (h2 : d ∣ (c - a)) (h3 : d ∣ (c - b)) (hd : d > 1)
  (hr : r = a % d):
  d - r = 15 := sorry

end problem_l107_107592


namespace parabola_vertex_below_x_axis_l107_107095

theorem parabola_vertex_below_x_axis (a : ℝ) : (∀ x : ℝ, (x^2 + 2 * x + a < 0)) → a < 1 := 
by
  intro h
  -- proof step here
  sorry

end parabola_vertex_below_x_axis_l107_107095


namespace anna_chargers_l107_107781

theorem anna_chargers (P L: ℕ) (h1: L = 5 * P) (h2: P + L = 24): P = 4 := by
  sorry

end anna_chargers_l107_107781


namespace farmer_randy_total_acres_l107_107821

-- Define the conditions
def acres_per_tractor_per_day : ℕ := 68
def tractors_first_2_days : ℕ := 2
def days_first_period : ℕ := 2
def tractors_next_3_days : ℕ := 7
def days_second_period : ℕ := 3

-- Prove the total acres Farmer Randy needs to plant
theorem farmer_randy_total_acres :
  (tractors_first_2_days * acres_per_tractor_per_day * days_first_period) +
  (tractors_next_3_days * acres_per_tractor_per_day * days_second_period) = 1700 :=
by
  -- Here, we would provide the proof, but in this example, we will use sorry.
  sorry

end farmer_randy_total_acres_l107_107821


namespace lorry_weight_l107_107356

theorem lorry_weight : 
  let empty_lorry_weight := 500
  let apples_weight := 10 * 55
  let oranges_weight := 5 * 45
  let watermelons_weight := 3 * 125
  let firewood_weight := 2 * 75
  let loaded_items_weight := apples_weight + oranges_weight + watermelons_weight + firewood_weight
  let total_weight := empty_lorry_weight + loaded_items_weight
  total_weight = 1800 :=
by 
  sorry

end lorry_weight_l107_107356


namespace scientific_notation_of_600000_l107_107783

theorem scientific_notation_of_600000 : 600000 = 6 * 10^5 :=
by
  sorry

end scientific_notation_of_600000_l107_107783


namespace problem_statement_l107_107787

theorem problem_statement (n m N k : ℕ)
  (h : (n^2 + 1)^(2^k) * (44 * n^3 + 11 * n^2 + 10 * n + 2) = N^m) :
  m = 1 :=
sorry

end problem_statement_l107_107787


namespace work_completion_l107_107304

theorem work_completion (A B C D : ℝ) :
  (A = 1 / 5) →
  (A + C = 2 / 5) →
  (B + C = 1 / 4) →
  (A + D = 1 / 3.6) →
  (B + C + D = 1 / 2) →
  B = 1 / 20 :=
by
  sorry

end work_completion_l107_107304


namespace denmark_pizza_combinations_l107_107670

theorem denmark_pizza_combinations :
  (let cheese_options := 3
   let meat_options := 4
   let vegetable_options := 5
   let invalid_combinations := 1
   let total_combinations := cheese_options * meat_options * vegetable_options
   let valid_combinations := total_combinations - invalid_combinations
   valid_combinations = 59) :=
by
  sorry

end denmark_pizza_combinations_l107_107670


namespace find_first_term_of_sequence_l107_107344

theorem find_first_term_of_sequence
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n+1) = a n + d)
  (h2 : a 0 + a 1 + a 2 = 12)
  (h3 : a 0 * a 1 * a 2 = 48)
  (h4 : ∀ n m, n < m → a n ≤ a m) :
  a 0 = 2 :=
sorry

end find_first_term_of_sequence_l107_107344


namespace two_digit_numbers_solution_l107_107851

theorem two_digit_numbers_solution :
  ∀ (N : ℕ), (∃ (x y : ℕ), (N = 10 * x + y) ∧ (x < 10) ∧ (y < 10) ∧ 4 * x + 2 * y = N / 2) →
    (N = 32 ∨ N = 64 ∨ N = 96) := 
by
  sorry

end two_digit_numbers_solution_l107_107851


namespace possible_arrangements_count_l107_107863

-- Define students as a type
inductive Student
| A | B | C | D | E | F

open Student

-- Define Club as a type
inductive Club
| A | B | C

open Club

-- Define the arrangement constraints
structure Arrangement :=
(assignment : Student → Club)
(club_size : Club → Nat)
(A_and_B_same_club : assignment A = assignment B)
(C_and_D_diff_clubs : assignment C ≠ assignment D)
(club_A_size : club_size A = 3)
(all_clubs_nonempty : ∀ c : Club, club_size c > 0)

-- Define the possible number of arrangements
def arrangement_count (a : Arrangement) : Nat := sorry

-- Theorem stating the number of valid arrangements
theorem possible_arrangements_count : ∃ a : Arrangement, arrangement_count a = 24 := sorry

end possible_arrangements_count_l107_107863


namespace winnie_keeps_lollipops_l107_107238

theorem winnie_keeps_lollipops :
  let cherry := 36
  let wintergreen := 125
  let grape := 8
  let shrimp_cocktail := 241
  let total_lollipops := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 13
  total_lollipops % friends = 7 :=
by
  sorry

end winnie_keeps_lollipops_l107_107238


namespace second_machine_time_l107_107323

theorem second_machine_time (x : ℝ) : 
  (600 / 10) + (1000 / x) = 1000 / 4 ↔ 
  1 / 10 + 1 / x = 1 / 4 :=
by
  sorry

end second_machine_time_l107_107323


namespace find_k_l107_107724

theorem find_k 
  (k : ℝ) 
  (m_eq : ∀ x : ℝ, ∃ y : ℝ, y = 3 * x + 5)
  (n_eq : ∀ x : ℝ, ∃ y : ℝ, y = k * x - 7) 
  (intersection : ∃ x y : ℝ, (y = 3 * x + 5) ∧ (y = k * x - 7) ∧ x = -4 ∧ y = -7) :
  k = 0 :=
by
  sorry

end find_k_l107_107724


namespace max_bricks_truck_can_carry_l107_107128

-- Define the truck's capacity in terms of bags of sand and bricks
def max_sand_bags := 50
def max_bricks := 400
def sand_to_bricks_ratio := 8

-- Define the current number of sand bags already on the truck
def current_sand_bags := 32

-- Define the number of bricks equivalent to a given number of sand bags
def equivalent_bricks (sand_bags: ℕ) := sand_bags * sand_to_bricks_ratio

-- Define the remaining capacity in terms of bags of sand
def remaining_sand_bags := max_sand_bags - current_sand_bags

-- Define the maximum number of additional bricks the truck can carry
def max_additional_bricks := equivalent_bricks remaining_sand_bags

-- Prove the number of additional bricks the truck can carry is 144
theorem max_bricks_truck_can_carry : max_additional_bricks = 144 := by
  sorry

end max_bricks_truck_can_carry_l107_107128


namespace probability_difference_l107_107649

theorem probability_difference (red_marbles black_marbles : ℤ) (h_red : red_marbles = 1500) (h_black : black_marbles = 1500) :
  |(22485 / 44985 : ℚ) - (22500 / 44985 : ℚ)| = 15 / 44985 := 
by {
  sorry
}

end probability_difference_l107_107649


namespace tangent_line_to_circle_l107_107600

-- Definitions derived directly from the conditions
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0
def passes_through_point (l : ℝ → ℝ → Prop) : Prop := l (-1) 6

-- The statement to be proven
theorem tangent_line_to_circle :
  ∃ (l : ℝ → ℝ → Prop), passes_through_point l ∧ 
    ((∀ x y, l x y ↔ 3*x - 4*y + 27 = 0) ∨ 
     (∀ x y, l x y ↔ x + 1 = 0)) :=
sorry

end tangent_line_to_circle_l107_107600


namespace smallest_whole_number_inequality_l107_107187

theorem smallest_whole_number_inequality (x : ℕ) (h : 3 * x + 4 > 11 - 2 * x) : x ≥ 2 :=
sorry

end smallest_whole_number_inequality_l107_107187


namespace father_dig_time_l107_107403

-- Definitions based on the conditions
variable (T : ℕ) -- Time taken by the father to dig the hole in hours
variable (D : ℕ) -- Depth of the hole dug by the father in feet
variable (M : ℕ) -- Depth of the hole dug by Michael in feet

-- Conditions
def father_hole_depth : Prop := D = 4 * T
def michael_hole_depth : Prop := M = 2 * D - 400
def michael_dig_time : Prop := M = 4 * 700

-- The proof statement, proving T = 400 given the conditions
theorem father_dig_time (T D M : ℕ)
  (h1 : father_hole_depth T D)
  (h2 : michael_hole_depth D M)
  (h3 : michael_dig_time M) : T = 400 := 
by
  sorry

end father_dig_time_l107_107403


namespace h_h_three_l107_107795

def h (x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem h_h_three : h (h 3) = 3568 := by
  sorry

end h_h_three_l107_107795


namespace TV_cost_l107_107716

theorem TV_cost (savings_furniture_fraction : ℚ)
                (original_savings : ℝ)
                (spent_on_furniture : ℝ)
                (spent_on_TV : ℝ)
                (hfurniture : savings_furniture_fraction = 2/4)
                (hsavings : original_savings = 600)
                (hspent_furniture : spent_on_furniture = original_savings * savings_furniture_fraction) :
                spent_on_TV = 300 := 
sorry

end TV_cost_l107_107716


namespace find_salary_of_january_l107_107655

variables (J F M A May : ℝ)

theorem find_salary_of_january
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8600)
  (h3 : May = 6500) :
  J = 4100 := 
sorry

end find_salary_of_january_l107_107655


namespace dawn_monthly_savings_l107_107608

variable (annual_income : ℕ)
variable (months : ℕ)
variable (tax_deduction_percent : ℚ)
variable (variable_expense_percent : ℚ)
variable (savings_percent : ℚ)

def calculate_monthly_savings (annual_income months : ℕ) 
    (tax_deduction_percent variable_expense_percent savings_percent : ℚ) : ℚ :=
  let monthly_income := (annual_income : ℚ) / months;
  let after_tax_income := monthly_income * (1 - tax_deduction_percent);
  let after_expenses_income := after_tax_income * (1 - variable_expense_percent);
  after_expenses_income * savings_percent

theorem dawn_monthly_savings : 
    calculate_monthly_savings 48000 12 0.20 0.30 0.10 = 224 := 
  by 
    sorry

end dawn_monthly_savings_l107_107608


namespace mrs_berkeley_A_students_first_class_mrs_berkeley_A_students_extended_class_l107_107506

noncomputable def ratio_of_A_students (total_students_A : ℕ) (A_students_A : ℕ) : ℚ :=
  A_students_A / total_students_A

theorem mrs_berkeley_A_students_first_class :
  ∀ (total_students_A : ℕ) (A_students_A : ℕ) (total_students_B : ℕ),
    total_students_A = 30 →
    A_students_A = 20 →
    total_students_B = 18 →
    (A_students_A / total_students_A) * total_students_B = 12 :=
by
  intros total_students_A A_students_A total_students_B hA1 hA2 hB
  sorry

theorem mrs_berkeley_A_students_extended_class :
  ∀ (total_students_A : ℕ) (A_students_A : ℕ) (total_students_B : ℕ),
    total_students_A = 30 →
    A_students_A = 20 →
    total_students_B = 27 →
    (A_students_A / total_students_A) * total_students_B = 18 :=
by
  intros total_students_A A_students_A total_students_B hA1 hA2 hB
  sorry

end mrs_berkeley_A_students_first_class_mrs_berkeley_A_students_extended_class_l107_107506


namespace find_a_minus_b_l107_107137

theorem find_a_minus_b (a b : ℚ)
  (h1 : 2 = a + b / 2)
  (h2 : 7 = a - b / 2)
  : a - b = 19 / 2 := 
  sorry

end find_a_minus_b_l107_107137


namespace minimize_segment_sum_l107_107068

theorem minimize_segment_sum (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ x y : ℝ, x = Real.sqrt (a * b) ∧ y = Real.sqrt (a * b) ∧ x * y = a * b ∧ x + y = 2 * Real.sqrt (a * b) := 
by
  sorry

end minimize_segment_sum_l107_107068


namespace lowest_score_on_one_of_last_two_tests_l107_107247

-- define conditions
variables (score1 score2 : ℕ) (total_score average desired_score : ℕ)

-- Shauna's scores on the first two tests are 82 and 75
def shauna_score1 := 82
def shauna_score2 := 75

-- Shauna wants to average 85 over 4 tests
def desired_average := 85
def number_of_tests := 4

-- total points needed for desired average
def total_points_needed := desired_average * number_of_tests

-- total points from first two tests
def total_first_two_tests := shauna_score1 + shauna_score2

-- total points needed on last two tests
def points_needed_last_two_tests := total_points_needed - total_first_two_tests

-- Prove the lowest score on one of the last two tests
theorem lowest_score_on_one_of_last_two_tests : 
  (∃ (score3 score4 : ℕ), score3 + score4 = points_needed_last_two_tests ∧ score3 ≤ 100 ∧ score4 ≤ 100 ∧ (score3 ≥ 83 ∨ score4 ≥ 83)) :=
sorry

end lowest_score_on_one_of_last_two_tests_l107_107247


namespace total_profit_l107_107718

-- Definitions based on the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Statement of the theorem
theorem total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end total_profit_l107_107718


namespace imaginary_power_sum_zero_l107_107373

theorem imaginary_power_sum_zero (i : ℂ) (n : ℤ) (h : i^2 = -1) :
  i^(2*n - 3) + i^(2*n - 1) + i^(2*n + 1) + i^(2*n + 3) = 0 :=
by {
  sorry
}

end imaginary_power_sum_zero_l107_107373


namespace john_bought_soap_l107_107774

theorem john_bought_soap (weight_per_bar : ℝ) (cost_per_pound : ℝ) (total_spent : ℝ) (h1 : weight_per_bar = 1.5) (h2 : cost_per_pound = 0.5) (h3 : total_spent = 15) : 
  total_spent / (weight_per_bar * cost_per_pound) = 20 :=
by
  -- The proof would go here
  sorry

end john_bought_soap_l107_107774


namespace union_of_sets_complement_intersection_of_sets_l107_107567

def setA : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def setB : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_sets :
  setA ∪ setB = {x | 2 < x ∧ x < 10} :=
sorry

theorem complement_intersection_of_sets :
  (setAᶜ) ∩ setB = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
sorry

end union_of_sets_complement_intersection_of_sets_l107_107567


namespace quadratic_roots_is_correct_l107_107952

theorem quadratic_roots_is_correct (a b : ℝ) 
    (h1 : a + b = 16) 
    (h2 : a * b = 225) :
    (∀ x, x^2 - 16 * x + 225 = 0 ↔ x = a ∨ x = b) := sorry

end quadratic_roots_is_correct_l107_107952


namespace two_digit_number_representation_l107_107438

theorem two_digit_number_representation (a b : ℕ) (ha : a < 10) (hb : b < 10) : 10 * b + a = d :=
  sorry

end two_digit_number_representation_l107_107438


namespace sum_proper_divisors_243_l107_107106

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 := by
  sorry

end sum_proper_divisors_243_l107_107106


namespace common_tangents_l107_107093

noncomputable def circle1 := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 4 }
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 9 }

theorem common_tangents (h : ∀ p : ℝ × ℝ, p ∈ circle1 → p ∈ circle2) : 
  ∃ tangents : ℕ, tangents = 2 :=
sorry

end common_tangents_l107_107093


namespace triangle_is_isosceles_l107_107282

variable (A B C a b c : ℝ)
variable (sin : ℝ → ℝ)

theorem triangle_is_isosceles (h1 : a * sin A - b * sin B = 0) :
  a = b :=
by
  sorry

end triangle_is_isosceles_l107_107282


namespace num_ways_to_place_balls_in_boxes_l107_107695

theorem num_ways_to_place_balls_in_boxes (num_balls num_boxes : ℕ) (hB : num_balls = 4) (hX : num_boxes = 3) : 
  (num_boxes ^ num_balls) = 81 := by
  rw [hB, hX]
  sorry

end num_ways_to_place_balls_in_boxes_l107_107695


namespace initial_money_amount_l107_107826

theorem initial_money_amount (x : ℕ) (h : x + 16 = 18) : x = 2 := by
  sorry

end initial_money_amount_l107_107826


namespace hyperbola_center_is_correct_l107_107234

theorem hyperbola_center_is_correct :
  ∃ h k : ℝ, (∀ x y : ℝ, ((4 * y + 8)^2 / 16^2) - ((5 * x - 15)^2 / 9^2) = 1 → x - h = 0 ∧ y + k = 0) ∧ h = 3 ∧ k = -2 :=
sorry

end hyperbola_center_is_correct_l107_107234


namespace six_dice_not_same_probability_l107_107227

theorem six_dice_not_same_probability :
  let total_outcomes := 6^6
  let all_same := 6
  let probability_all_same := all_same / total_outcomes
  let probability_not_all_same := 1 - probability_all_same
  probability_not_all_same = 7775 / 7776 :=
by
  sorry

end six_dice_not_same_probability_l107_107227


namespace cans_per_bag_l107_107042

theorem cans_per_bag (bags_on_Saturday bags_on_Sunday total_cans : ℕ) (h_saturday : bags_on_Saturday = 3) (h_sunday : bags_on_Sunday = 4) (h_total : total_cans = 63) :
  (total_cans / (bags_on_Saturday + bags_on_Sunday) = 9) :=
by {
  sorry
}

end cans_per_bag_l107_107042


namespace combined_size_UK_India_US_l107_107715

theorem combined_size_UK_India_US (U : ℝ)
    (Canada : ℝ := 1.5 * U)
    (Russia : ℝ := (1 + 1/3) * Canada)
    (China : ℝ := (1 / 1.7) * Russia)
    (Brazil : ℝ := (2 / 3) * U)
    (Australia : ℝ := (1 / 2) * Brazil)
    (UK : ℝ := 2 * Australia)
    (India : ℝ := (1 / 4) * Russia)
    (India' : ℝ := 6 * UK)
    (h_India : India = India') :
  UK + India = 7 / 6 * U := 
by
  -- Proof details
  sorry

end combined_size_UK_India_US_l107_107715


namespace range_of_m_l107_107025

theorem range_of_m (m : ℝ) :
  (∀ x: ℝ, |x| + |x - 1| > m) ∨ (∀ x y, x < y → (5 - 2 * m)^x ≤ (5 - 2 * m)^y) 
  → ¬ ((∀ x: ℝ, |x| + |x - 1| > m) ∧ (∀ x y, x < y → (5 - 2 * m)^x ≤ (5 - 2 * m)^y)) 
  ↔ (1 ≤ m ∧ m < 2) :=
by
  sorry

end range_of_m_l107_107025


namespace binary_addition_l107_107393

theorem binary_addition :
  0b1101 + 0b101 + 0b1110 + 0b10111 + 0b11000 = 0b11100010 :=
by
  sorry

end binary_addition_l107_107393


namespace solve_quadratic_l107_107593

theorem solve_quadratic (x : ℝ) (h1 : 2 * x ^ 2 = 9 * x - 4) (h2 : x ≠ 4) : 2 * x = 1 :=
by
  -- The proof will go here
  sorry

end solve_quadratic_l107_107593


namespace range_of_m_l107_107401

/-- Define the domain set A where the function f(x) = 1 / sqrt(4 + 3x - x^2) is defined. -/
def A : Set ℝ := {x | -1 < x ∧ x < 4}

/-- Define the range set B where the function g(x) = - x^2 - 2x + 2, with x in [-1, 1], is defined. -/
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

/-- Define the set C in terms of m. -/
def C (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 2}

/-- Prove the range of the real number m such that C ∩ (A ∪ B) = C. -/
theorem range_of_m : {m : ℝ | C m ⊆ A ∪ B} = {m | -1 ≤ m ∧ m < 2} :=
by
  sorry

end range_of_m_l107_107401


namespace value_of_x_l107_107720

theorem value_of_x (x : ℝ) : (9 - x) ^ 2 = x ^ 2 → x = 4.5 :=
by
  sorry

end value_of_x_l107_107720


namespace clothing_price_decrease_l107_107462

theorem clothing_price_decrease (P : ℝ) (h₁ : P > 0) :
  let price_first_sale := (4 / 5) * P
  let price_second_sale := (1 / 2) * P
  let price_difference := price_first_sale - price_second_sale
  let percent_decrease := (price_difference / price_first_sale) * 100
  percent_decrease = 37.5 :=
by
  sorry

end clothing_price_decrease_l107_107462


namespace frequency_of_group_of_samples_l107_107455

def sample_capacity : ℝ := 32
def frequency_rate : ℝ := 0.125

theorem frequency_of_group_of_samples : frequency_rate * sample_capacity = 4 :=
by 
  sorry

end frequency_of_group_of_samples_l107_107455


namespace min_value_eq_six_l107_107946

theorem min_value_eq_six
    (α β : ℝ)
    (k : ℝ)
    (h1 : α^2 + 2 * (k + 3) * α + (k^2 + 3) = 0)
    (h2 : β^2 + 2 * (k + 3) * β + (k^2 + 3) = 0)
    (h3 : (2 * (k + 3))^2 - 4 * (k^2 + 3) ≥ 0) :
    ( (α - 1)^2 + (β - 1)^2 = 6 ) := 
sorry

end min_value_eq_six_l107_107946


namespace adoption_complete_in_7_days_l107_107196

-- Define the initial number of puppies
def initial_puppies := 9

-- Define the number of puppies brought in later
def additional_puppies := 12

-- Define the number of puppies adopted per day
def adoption_rate := 3

-- Define the total number of puppies
def total_puppies : Nat := initial_puppies + additional_puppies

-- Define the number of days required to adopt all puppies
def adoption_days : Nat := total_puppies / adoption_rate

-- Prove that the number of days to adopt all puppies is 7
theorem adoption_complete_in_7_days : adoption_days = 7 := by
  -- The exact implementation of the proof is not necessary,
  -- so we use sorry to skip the proof.
  sorry

end adoption_complete_in_7_days_l107_107196


namespace quad_eq_diagonals_theorem_l107_107130

noncomputable def quad_eq_diagonals (a b c d m n : ℝ) (A C : ℝ) : Prop :=
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C)

theorem quad_eq_diagonals_theorem (a b c d m n A C : ℝ) :
  quad_eq_diagonals a b c d m n A C :=
by
  sorry

end quad_eq_diagonals_theorem_l107_107130


namespace rubble_initial_money_l107_107765

def initial_money (cost_notebook cost_pen : ℝ) (num_notebooks num_pens : ℕ) (money_left : ℝ) : ℝ :=
  (num_notebooks * cost_notebook + num_pens * cost_pen) + money_left

theorem rubble_initial_money :
  initial_money 4 1.5 2 2 4 = 15 :=
by
  sorry

end rubble_initial_money_l107_107765


namespace remaining_speed_l107_107541
open Real

theorem remaining_speed
  (D T : ℝ) (h1 : 40 * (T / 3) = (2 / 3) * D)
  (h2 : (T / 3) * 3 = T) :
  (D / 3) / ((2 * ((2 / 3) * D) / (40) / (3)) * 2 / 3) = 10 :=
by
  sorry

end remaining_speed_l107_107541


namespace distance_to_airport_l107_107607

theorem distance_to_airport:
  ∃ (d t: ℝ), 
    (d = 35 * (t + 1)) ∧
    (d - 35 = 50 * (t - 1.5)) ∧
    d = 210 := 
by 
  sorry

end distance_to_airport_l107_107607


namespace rate_percent_l107_107023

noncomputable def calculate_rate (P: ℝ) : ℝ :=
  let I : ℝ := 320
  let t : ℝ := 2
  I * 100 / (P * t)

theorem rate_percent (P: ℝ) (hP: P > 0) : calculate_rate P = 4 := 
by
  sorry

end rate_percent_l107_107023


namespace final_weight_is_correct_l107_107922

-- Define the initial weight of marble
def initial_weight := 300.0

-- Define the percentage reductions each week
def first_week_reduction := 0.3 * initial_weight
def second_week_reduction := 0.3 * (initial_weight - first_week_reduction)
def third_week_reduction := 0.15 * (initial_weight - first_week_reduction - second_week_reduction)

-- Calculate the final weight of the statue
def final_weight := initial_weight - first_week_reduction - second_week_reduction - third_week_reduction

-- The statement to prove
theorem final_weight_is_correct : final_weight = 124.95 := by
  -- Here would be the proof, which we are omitting
  sorry

end final_weight_is_correct_l107_107922


namespace symmetry_condition_l107_107220

theorem symmetry_condition (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - a| = |(2 - x) + 1| + |(2 - x) - a|) ↔ a = 3 :=
by
  sorry

end symmetry_condition_l107_107220


namespace total_area_expanded_dining_area_l107_107881

noncomputable def expanded_dining_area_total : ℝ :=
  let rectangular_area := 35
  let radius := 4
  let semi_circular_area := (1 / 2) * Real.pi * (radius^2)
  rectangular_area + semi_circular_area

theorem total_area_expanded_dining_area :
  expanded_dining_area_total = 60.13272 := by
  sorry

end total_area_expanded_dining_area_l107_107881


namespace unique_solution_for_star_l107_107767

def star (x y : ℝ) : ℝ := 4 * x - 5 * y + 2 * x * y

theorem unique_solution_for_star :
  ∃! y : ℝ, star 2 y = 5 :=
by
  -- We know the definition of star and we need to verify the condition.
  sorry

end unique_solution_for_star_l107_107767


namespace jordan_no_quiz_probability_l107_107585

theorem jordan_no_quiz_probability (P_quiz : ℚ) (h : P_quiz = 5 / 9) :
  1 - P_quiz = 4 / 9 :=
by
  rw [h]
  exact sorry

end jordan_no_quiz_probability_l107_107585


namespace performance_stability_l107_107085

theorem performance_stability (avg_score : ℝ) (num_shots : ℕ) (S_A S_B : ℝ) 
  (h_avg : num_shots = 10)
  (h_same_avg : avg_score = avg_score) 
  (h_SA : S_A^2 = 0.4) 
  (h_SB : S_B^2 = 2) : 
  (S_A < S_B) :=
by
  sorry

end performance_stability_l107_107085


namespace whiskers_ratio_l107_107935

/-- Four cats live in the old grey house at the end of the road. Their names are Puffy, Scruffy, Buffy, and Juniper.
Puffy has three times more whiskers than Juniper, but a certain ratio as many as Scruffy. Buffy has the same number of whiskers
as the average number of whiskers on the three other cats. Prove that the ratio of Puffy's whiskers to Scruffy's whiskers is 1:2
given Juniper has 12 whiskers and Buffy has 40 whiskers. -/
theorem whiskers_ratio (J B P S : ℕ) (hJ : J = 12) (hB : B = 40) (hP : P = 3 * J) (hAvg : B = (P + S + J) / 3) :
  P / gcd P S = 1 ∧ S / gcd P S = 2 := by
  sorry

end whiskers_ratio_l107_107935


namespace distance_between_closest_points_l107_107530

noncomputable def distance_closest_points :=
  let center1 : ℝ × ℝ := (5, 3)
  let center2 : ℝ × ℝ := (20, 7)
  let radius1 := center1.2  -- radius of first circle is y-coordinate of its center
  let radius2 := center2.2  -- radius of second circle is y-coordinate of its center
  let distance_centers := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance_centers - radius1 - radius2

theorem distance_between_closest_points :
  distance_closest_points = Real.sqrt 241 - 10 :=
sorry

end distance_between_closest_points_l107_107530


namespace maggie_earnings_correct_l107_107565

def subscriptions_sold_to_parents : ℕ := 4
def subscriptions_sold_to_grandfather : ℕ := 1
def subscriptions_sold_to_next_door_neighbor : ℕ := 2
def subscriptions_sold_to_another_neighbor : ℕ := 2 * subscriptions_sold_to_next_door_neighbor
def price_per_subscription : ℕ := 5
def family_bonus_per_subscription : ℕ := 2
def neighbor_bonus_per_subscription : ℕ := 1
def base_bonus_threshold : ℕ := 10
def base_bonus : ℕ := 10
def extra_bonus_per_subscription : ℝ := 0.5

-- Define total subscriptions sold
def total_subscriptions_sold : ℕ := 
  subscriptions_sold_to_parents + subscriptions_sold_to_grandfather + 
  subscriptions_sold_to_next_door_neighbor + subscriptions_sold_to_another_neighbor

-- Define earnings from subscriptions
def earnings_from_subscriptions : ℕ := total_subscriptions_sold * price_per_subscription

-- Define bonuses
def family_bonus : ℕ :=
  (subscriptions_sold_to_parents + subscriptions_sold_to_grandfather) * family_bonus_per_subscription

def neighbor_bonus : ℕ := 
  (subscriptions_sold_to_next_door_neighbor + subscriptions_sold_to_another_neighbor) * neighbor_bonus_per_subscription

def total_bonus : ℕ := family_bonus + neighbor_bonus

-- Define additional boss bonus
def additional_boss_bonus : ℝ := 
  if total_subscriptions_sold > base_bonus_threshold then 
    base_bonus + extra_bonus_per_subscription * (total_subscriptions_sold - base_bonus_threshold) 
  else 0

-- Define total earnings
def total_earnings : ℝ :=
  earnings_from_subscriptions + total_bonus + additional_boss_bonus

-- Theorem statement
theorem maggie_earnings_correct : total_earnings = 81.5 :=
by
  unfold total_earnings
  unfold earnings_from_subscriptions
  unfold total_bonus
  unfold family_bonus
  unfold neighbor_bonus
  unfold additional_boss_bonus
  unfold total_subscriptions_sold
  simp
  norm_cast
  sorry

end maggie_earnings_correct_l107_107565


namespace seedling_costs_and_purchase_l107_107332

variable (cost_A cost_B : ℕ)
variable (m n : ℕ)

-- Conditions
def conditions : Prop :=
  (cost_A = cost_B + 5) ∧ 
  (400 / cost_A = 300 / cost_B)

-- Prove costs and purchase for minimal costs
theorem seedling_costs_and_purchase (cost_A cost_B : ℕ) (m n : ℕ)
  (h1 : conditions cost_A cost_B)
  (h2 : m + n = 150)
  (h3 : m ≥ n / 2)
  : cost_A = 20 ∧ cost_B = 15 ∧ 5 * 50 + 2250 = 2500 
  := by
  sorry

end seedling_costs_and_purchase_l107_107332


namespace find_divisor_l107_107288

theorem find_divisor (d : ℕ) (h : 127 = d * 5 + 2) : d = 25 :=
by 
  -- Given conditions
  -- 127 = d * 5 + 2
  -- We need to prove d = 25
  sorry

end find_divisor_l107_107288


namespace complex_expression_evaluation_l107_107267

theorem complex_expression_evaluation (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^101 + z^102 + z^103 + z^104 + z^105 = -1 := 
sorry

end complex_expression_evaluation_l107_107267


namespace solution_set_equivalence_l107_107400

def solution_set_inequality (x : ℝ) : Prop :=
  abs (x - 1) + abs x < 3

theorem solution_set_equivalence :
  { x : ℝ | solution_set_inequality x } = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_equivalence_l107_107400


namespace necessary_but_not_sufficient_condition_l107_107978

variable {a : Nat → Real} -- Sequence a_n
variable {q : Real} -- Common ratio
variable (a1_pos : a 1 > 0) -- Condition a1 > 0

-- Definition of geometric sequence
def is_geometric_sequence (a : Nat → Real) (q : Real) : Prop :=
  ∀ n : Nat, a (n + 1) = a n * q

-- Definition of increasing sequence
def is_increasing_sequence (a : Nat → Real) : Prop :=
  ∀ n : Nat, a n < a (n + 1)

-- Theorem statement
theorem necessary_but_not_sufficient_condition (a : Nat → Real) (q : Real) (a1_pos : a 1 > 0) :
  is_geometric_sequence a q →
  is_increasing_sequence a →
  q > 0 ∧ ¬(q > 0 → is_increasing_sequence a) := by
  sorry

end necessary_but_not_sufficient_condition_l107_107978


namespace find_number_l107_107760

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 18) : x = 9 :=
sorry

end find_number_l107_107760


namespace binomial_expansion_equality_l107_107662

theorem binomial_expansion_equality (x : ℝ) : 
  (x-1)^4 - 4*x*(x-1)^3 + 6*(x^2)*(x-1)^2 - 4*(x^3)*(x-1)*x^4 = 1 := 
by 
  sorry 

end binomial_expansion_equality_l107_107662


namespace type_C_count_l107_107578

theorem type_C_count (A B C C1 C2 : ℕ) (h1 : A + B + C = 25) (h2 : A + B + C2 = 17) (h3 : B + C2 = 12) (h4 : C2 = 8) (h5: B = 4) (h6: A = 5) : C = 16 :=
by {
  -- Directly use the given hypotheses.
  sorry
}

end type_C_count_l107_107578


namespace upper_limit_of_sixth_powers_l107_107747

theorem upper_limit_of_sixth_powers :
  ∃ b : ℕ, (∀ n : ℕ, (∃ a : ℕ, a^6 = n) ∧ n ≤ b → n = 46656) :=
by
  sorry

end upper_limit_of_sixth_powers_l107_107747


namespace is_rectangle_l107_107996

-- Define the points A, B, C, and D.
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 6)
def C : ℝ × ℝ := (5, 4)
def D : ℝ × ℝ := (2, -2)

-- Define the vectors AB, DC, AD.
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)
def AB := vec A B
def DC := vec D C
def AD := vec A D

-- Function to compute dot product of two vectors.
def dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that quadrilateral ABCD is a rectangle.
theorem is_rectangle : AB = DC ∧ dot AB AD = 0 := by
  sorry

end is_rectangle_l107_107996


namespace fraction_scaled_l107_107549

theorem fraction_scaled (x y : ℝ) :
  ∃ (k : ℝ), (k = 3 * y) ∧ ((5 * x + 3 * y) / (x + 3 * y) = 5 * ((x + (3 * y)) / (x + (3 * y)))) := 
  sorry

end fraction_scaled_l107_107549


namespace jordyn_total_cost_l107_107778

-- Definitions for conditions
def price_cherries : ℝ := 5
def price_olives : ℝ := 7
def number_of_bags : ℕ := 50
def discount_rate : ℝ := 0.10 

-- Define the discounted price function
def discounted_price (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

-- Calculate the total cost for Jordyn
def total_cost (price_cherries price_olives : ℝ) (number_of_bags : ℕ) (discount_rate : ℝ) : ℝ :=
  (number_of_bags * discounted_price price_cherries discount_rate) + 
  (number_of_bags * discounted_price price_olives discount_rate)

-- Prove the final cost
theorem jordyn_total_cost : total_cost price_cherries price_olives number_of_bags discount_rate = 540 := by
  sorry

end jordyn_total_cost_l107_107778


namespace total_students_l107_107906

theorem total_students (a : ℕ) (h1: (71 * ((3480 - 69 * a) / 2) + 69 * (a - (3480 - 69 * a) / 2)) = 3480) : a = 50 :=
by
  -- Proof to be provided here
  sorry

end total_students_l107_107906


namespace midpoint_coordinates_l107_107169

theorem midpoint_coordinates (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 2) (hy1 : y1 = 10) (hx2 : x2 = 6) (hy2 : y2 = 2) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  mx = 4 ∧ my = 6 :=
by
  sorry

end midpoint_coordinates_l107_107169


namespace gcd_of_ratios_l107_107250

noncomputable def gcd_of_two_ratios (A B : ℕ) : ℕ :=
  if h : A % B = 0 then B else gcd B (A % B)

theorem gcd_of_ratios (A B : ℕ) (k : ℕ) (h1 : Nat.lcm A B = 180) (h2 : A = 2 * k) (h3 : B = 3 * k) : gcd_of_two_ratios A B = 30 :=
  by
    sorry

end gcd_of_ratios_l107_107250


namespace xiao_wang_ways_to_make_8_cents_l107_107390

theorem xiao_wang_ways_to_make_8_cents :
  (∃ c1 c2 c5 : ℕ, c1 ≤ 8 ∧ c2 ≤ 4 ∧ c5 ≤ 1 ∧ c1 + 2 * c2 + 5 * c5 = 8) → (number_of_ways_to_make_8_cents = 7) :=
sorry

end xiao_wang_ways_to_make_8_cents_l107_107390


namespace least_number_l107_107971

theorem least_number (n : ℕ) (h1 : n % 38 = 1) (h2 : n % 3 = 1) : n = 115 :=
sorry

end least_number_l107_107971


namespace problem_a_problem_b_l107_107617

variable (α : ℝ)

theorem problem_a (hα : 0 < α ∧ α < π) :
  Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = Real.tan (α / 2) :=
sorry

theorem problem_b (hα : π < α ∧ α < 2 * π) :
  Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = -Real.tan (α / 2) :=
sorry

end problem_a_problem_b_l107_107617


namespace blue_paint_gallons_l107_107118

-- Define the total gallons of paint used
def total_paint_gallons : ℕ := 6689

-- Define the gallons of white paint used
def white_paint_gallons : ℕ := 660

-- Define the corresponding proof problem
theorem blue_paint_gallons : 
  ∀ total white blue : ℕ, total = 6689 → white = 660 → blue = total - white → blue = 6029 := by
  sorry

end blue_paint_gallons_l107_107118


namespace part1_answer1_part1_answer2_part2_answer1_part2_answer2_l107_107631

open Set

def A : Set ℕ := {x | 1 ≤ x ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem part1_answer1 : A ∩ C = {3, 4, 5, 6, 7} :=
by
  sorry

theorem part1_answer2 : A \ B = {5, 6, 7, 8, 9, 10} :=
by
  sorry

theorem part2_answer1 : A \ (B ∪ C) = {8, 9, 10} :=
by 
  sorry

theorem part2_answer2 : A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} :=
by 
  sorry

end part1_answer1_part1_answer2_part2_answer1_part2_answer2_l107_107631


namespace problem_correct_l107_107629

noncomputable def problem := 
  1 - (1 / 2)⁻¹ * Real.sin (60 * Real.pi / 180) + abs (2^0 - Real.sqrt 3) = 0

theorem problem_correct : problem := by
  sorry

end problem_correct_l107_107629


namespace blue_red_area_ratio_l107_107378

theorem blue_red_area_ratio (d1 d2 : ℝ) (h1 : d1 = 2) (h2 : d2 = 6) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let a_red := π * r1^2
  let a_large := π * r2^2
  let a_blue := a_large - a_red
  a_blue / a_red = 8 :=
by
  have r1 := d1 / 2
  have r2 := d2 / 2
  have a_red := π * r1^2
  have a_large := π * r2^2
  have a_blue := a_large - a_red
  sorry

end blue_red_area_ratio_l107_107378


namespace fraction_sum_l107_107840

theorem fraction_sum (x a b : ℕ) (h1 : x = 36 / 99) (h2 : a = 4) (h3 : b = 11) (h4 : Nat.gcd a b = 1) : a + b = 15 :=
by
  sorry

end fraction_sum_l107_107840


namespace elephant_weight_equivalence_l107_107846

-- Define the conditions as variables
def elephants := 1000000000
def buildings := 25000

-- Define the question and expected answer
def expected_answer := 40000

-- State the theorem
theorem elephant_weight_equivalence:
  (elephants / buildings = expected_answer) :=
by
  sorry

end elephant_weight_equivalence_l107_107846


namespace meals_calculation_l107_107690

def combined_meals (k a : ℕ) : ℕ :=
  k + a

theorem meals_calculation :
  ∀ (k a : ℕ), k = 8 → (2 * a = k) → combined_meals k a = 12 :=
  by
    intros k a h1 h2
    rw [h1] at h2
    have ha : a = 4 := by linarith
    rw [h1, ha]
    unfold combined_meals
    sorry

end meals_calculation_l107_107690


namespace phone_prices_purchase_plans_l107_107474

noncomputable def modelA_price : ℝ := 2000
noncomputable def modelB_price : ℝ := 1000

theorem phone_prices :
  (∀ x y : ℝ, (2 * x + y = 5000 ∧ 3 * x + 2 * y = 8000) → x = modelA_price ∧ y = modelB_price) :=
by
    intro x y
    intro h
    have h1 := h.1
    have h2 := h.2
    -- We would provide the detailed proof here
    sorry

theorem purchase_plans :
  (∀ a : ℕ, (4 ≤ a ∧ a ≤ 6) ↔ (24000 ≤ 2000 * a + 1000 * (20 - a) ∧ 2000 * a + 1000 * (20 - a) ≤ 26000)) :=
by
    intro a
    -- We would provide the detailed proof here
    sorry

end phone_prices_purchase_plans_l107_107474


namespace square_side_length_properties_l107_107226

theorem square_side_length_properties (a: ℝ) (h: a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by
  sorry

end square_side_length_properties_l107_107226


namespace smallest_n_exists_l107_107518

theorem smallest_n_exists (n : ℤ) (r : ℝ) : 
  (∃ m : ℤ, m = (↑n + r) ^ 3 ∧ r > 0 ∧ r < 1 / 1000) ∧ n > 0 → n = 19 := 
by sorry

end smallest_n_exists_l107_107518


namespace A_alone_completes_one_work_in_32_days_l107_107852

def amount_of_work_per_day_by_B : ℝ := sorry
def amount_of_work_per_day_by_A : ℝ := 3 * amount_of_work_per_day_by_B
def total_work : ℝ := (amount_of_work_per_day_by_A + amount_of_work_per_day_by_B) * 24

theorem A_alone_completes_one_work_in_32_days :
  total_work = amount_of_work_per_day_by_A * 32 :=
by
  sorry

end A_alone_completes_one_work_in_32_days_l107_107852


namespace recurring_decimal_to_fraction_l107_107956

theorem recurring_decimal_to_fraction (a b : ℕ) (ha : a = 356) (hb : b = 999) (hab_gcd : Nat.gcd a b = 1)
  (x : ℚ) (hx : x = 356 / 999) 
  (hx_recurring : x = {num := 356, den := 999}): a + b = 1355 :=
by
  sorry  -- Proof is not required as per the instructions

end recurring_decimal_to_fraction_l107_107956


namespace example_theorem_l107_107668

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end example_theorem_l107_107668


namespace gold_hammer_weight_l107_107803

theorem gold_hammer_weight (a : ℕ → ℕ) 
  (h_arith_seq : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h_a1 : a 1 = 4) 
  (h_a5 : a 5 = 2) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := 
sorry

end gold_hammer_weight_l107_107803


namespace simplify_complex_div_l107_107972

theorem simplify_complex_div (a b c d : ℝ) (i : ℂ)
  (h1 : (a = 3) ∧ (b = 5) ∧ (c = -2) ∧ (d = 7) ∧ (i = Complex.I)) :
  ((Complex.mk a b) / (Complex.mk c d) = (Complex.mk (29/53) (-31/53))) :=
by
  sorry

end simplify_complex_div_l107_107972


namespace upstream_distance_l107_107639

theorem upstream_distance
  (man_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (effective_downstream_speed: ℝ)
  (stream_speed : ℝ)
  (upstream_time : ℝ)
  (upstream_distance : ℝ):
  man_speed = 7 ∧ downstream_distance = 45 ∧ downstream_time = 5 ∧ effective_downstream_speed = man_speed + stream_speed 
  ∧ effective_downstream_speed * downstream_time = downstream_distance 
  ∧ upstream_time = 5 ∧ upstream_distance = (man_speed - stream_speed) * upstream_time 
  → upstream_distance = 25 :=
by
  sorry

end upstream_distance_l107_107639


namespace pints_of_cider_l107_107750

def pintCider (g : ℕ) (p : ℕ) : ℕ :=
  g / 20 + p / 40

def totalApples (f : ℕ) (h : ℕ) (a : ℕ) : ℕ :=
  f * h * a

theorem pints_of_cider (g p : ℕ) (farmhands : ℕ) (hours : ℕ) (apples_per_hour : ℕ)
  (H1 : g = 1)
  (H2 : p = 2)
  (H3 : farmhands = 6)
  (H4 : hours = 5)
  (H5 : apples_per_hour = 240) :
  pintCider (apples_per_hour * farmhands * hours / 3)
            (apples_per_hour * farmhands * hours * 2 / 3) = 120 :=
by
  sorry

end pints_of_cider_l107_107750


namespace infinite_geometric_series_sum_l107_107746

theorem infinite_geometric_series_sum : 
  (∃ (a r : ℚ), a = 5/4 ∧ r = 1/3) → 
  ∑' n : ℕ, ((5/4 : ℚ) * (1/3 : ℚ) ^ n) = (15/8 : ℚ) :=
by
  sorry

end infinite_geometric_series_sum_l107_107746


namespace jenna_eel_length_l107_107123

theorem jenna_eel_length (J B L : ℝ)
  (h1 : J = (2 / 5) * B)
  (h2 : J = (3 / 7) * L)
  (h3 : J + B + L = 124) : 
  J = 21 := 
sorry

end jenna_eel_length_l107_107123


namespace option_d_correct_l107_107275

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)
def M : Set ℝ := {x | f x = 0}

theorem option_d_correct : ({1, 3} ∪ {2, 3} : Set ℝ) = M := by
  sorry

end option_d_correct_l107_107275


namespace protective_additive_increase_l107_107012

def percentIncrease (old_val new_val : ℕ) : ℚ :=
  (new_val - old_val) / old_val * 100

theorem protective_additive_increase :
  percentIncrease 45 60 = 33.33 := 
sorry

end protective_additive_increase_l107_107012


namespace edges_after_truncation_l107_107141

-- Define a regular tetrahedron with 4 vertices and 6 edges
structure Tetrahedron :=
  (vertices : ℕ)
  (edges : ℕ)

-- Initial regular tetrahedron
def initial_tetrahedron : Tetrahedron :=
  { vertices := 4, edges := 6 }

-- Function to calculate the number of edges after truncating vertices
def truncated_edges (t : Tetrahedron) (vertex_truncations : ℕ) (new_edges_per_vertex : ℕ) : ℕ :=
  vertex_truncations * new_edges_per_vertex

-- Given a regular tetrahedron and the truncation process
def resulting_edges (t : Tetrahedron) (vertex_truncations : ℕ) :=
  truncated_edges t vertex_truncations 3

-- Problem statement: Proving the resulting figure has 12 edges
theorem edges_after_truncation :
  resulting_edges initial_tetrahedron 4 = 12 :=
  sorry

end edges_after_truncation_l107_107141


namespace multiples_six_or_eight_not_both_l107_107283

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l107_107283


namespace smallest_missing_digit_units_place_cube_l107_107329

theorem smallest_missing_digit_units_place_cube :
  ∀ d : Fin 10, ∃ n : ℕ, (n ^ 3) % 10 = d :=
by
  sorry

end smallest_missing_digit_units_place_cube_l107_107329


namespace four_pow_expression_l107_107612

theorem four_pow_expression : 4 ^ (3 ^ 2) / (4 ^ 3) ^ 2 = 64 := by
  sorry

end four_pow_expression_l107_107612


namespace base8_base13_to_base10_sum_l107_107654

-- Definitions for the base 8 and base 13 numbers
def base8_to_base10 (a b c : ℕ) : ℕ := a * 64 + b * 8 + c
def base13_to_base10 (d e f : ℕ) : ℕ := d * 169 + e * 13 + f

-- Constants for the specific numbers in the problem
def num1 := base8_to_base10 5 3 7
def num2 := base13_to_base10 4 12 5

-- The theorem to prove
theorem base8_base13_to_base10_sum : num1 + num2 = 1188 := by
  sorry

end base8_base13_to_base10_sum_l107_107654


namespace emails_in_afternoon_l107_107194

variable (e_m e_t e_a : Nat)
variable (h1 : e_m = 3)
variable (h2 : e_t = 8)

theorem emails_in_afternoon : e_a = 5 :=
by
  -- (Proof steps would go here)
  sorry

end emails_in_afternoon_l107_107194


namespace chickens_and_rabbits_l107_107374

theorem chickens_and_rabbits (c r : ℕ) (h1 : c + r = 15) (h2 : 2 * c + 4 * r = 40) : c = 10 ∧ r = 5 :=
sorry

end chickens_and_rabbits_l107_107374


namespace three_digit_odd_number_is_803_l107_107485

theorem three_digit_odd_number_is_803 :
  ∃ (a b c : ℕ), 0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ c % 2 = 1 ∧
  100 * a + 10 * b + c = 803 ∧ (100 * a + 10 * b + c) / 11 = a^2 + b^2 + c^2 :=
by {
  sorry
}

end three_digit_odd_number_is_803_l107_107485


namespace ratio_of_boys_to_girls_l107_107626

-- Definitions based on the initial conditions
def G : ℕ := 135
def T : ℕ := 351

-- Noncomputable because it involves division which is not always computable
noncomputable def B : ℕ := T - G

-- Main theorem to prove the ratio
theorem ratio_of_boys_to_girls : (B : ℚ) / G = 8 / 5 :=
by
  -- Here would be the proof, skipped with sorry.
  sorry

end ratio_of_boys_to_girls_l107_107626


namespace amy_total_score_correct_l107_107040

def amyTotalScore (points_per_treasure : ℕ) (treasures_first_level : ℕ) (treasures_second_level : ℕ) : ℕ :=
  (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level)

theorem amy_total_score_correct:
  amyTotalScore 4 6 2 = 32 :=
by
  -- Proof goes here
  sorry

end amy_total_score_correct_l107_107040


namespace fraction_of_yard_occupied_by_flower_beds_l107_107540

theorem fraction_of_yard_occupied_by_flower_beds :
  let leg_length := (36 - 26) / 3
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := 36 * 6
  (total_flower_bed_area / yard_area) = 25 / 324
  := by
  let leg_length := (36 - 26) / 3
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 3 * triangle_area
  let yard_area := 36 * 6
  have h1 : leg_length = 10 / 3 := by sorry
  have h2 : triangle_area = (1 / 2) * (10 / 3)^2 := by sorry
  have h3 : total_flower_bed_area = 3 * ((1 / 2) * (10 / 3)^2) := by sorry
  have h4 : yard_area = 216 := by sorry
  have h5 : total_flower_bed_area / yard_area = 25 / 324 := by sorry
  exact h5

end fraction_of_yard_occupied_by_flower_beds_l107_107540


namespace max_profit_l107_107117

noncomputable def fixed_cost := 20000
noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 8 then (1/3) * x^2 + 2 * x else 7 * x + 100 / x - 37
noncomputable def sales_price_per_unit : ℝ := 6
noncomputable def profit (x : ℝ) : ℝ :=
  let revenue := sales_price_per_unit * x
  let cost := fixed_cost / 10000 + variable_cost x
  revenue - cost

theorem max_profit : ∃ x : ℝ, (0 < x) ∧ (15 = profit 10) :=
by {
  sorry
}

end max_profit_l107_107117


namespace sum_ratio_arithmetic_sequence_l107_107038

noncomputable def sum_of_arithmetic_sequence (n : ℕ) : ℝ := sorry

theorem sum_ratio_arithmetic_sequence (S : ℕ → ℝ) (hS : ∀ n, S n = sum_of_arithmetic_sequence n)
  (h_cond : S 3 / S 6 = 1 / 3) :
  S 6 / S 12 = 3 / 10 :=
sorry

end sum_ratio_arithmetic_sequence_l107_107038


namespace professor_D_error_l107_107190

noncomputable def polynomial_calculation_error (n : ℕ) : Prop :=
  ∃ (f : ℝ → ℝ), (∀ i : ℕ, i ≤ n+1 → f i = 2^i) ∧ f (n+2) ≠ 2^(n+2) - n - 3

theorem professor_D_error (n : ℕ) : polynomial_calculation_error n :=
  sorry

end professor_D_error_l107_107190


namespace reciprocal_neg_one_div_2022_l107_107365

theorem reciprocal_neg_one_div_2022 : (1 / (-1 / 2022)) = -2022 :=
by sorry

end reciprocal_neg_one_div_2022_l107_107365


namespace calculate_fraction_l107_107809

theorem calculate_fraction : (10^20 / 50^10) = 2^10 := by
  sorry

end calculate_fraction_l107_107809


namespace molecular_weight_NaClO_l107_107623

theorem molecular_weight_NaClO :
  let Na := 22.99
  let Cl := 35.45
  let O := 16.00
  Na + Cl + O = 74.44 :=
by
  let Na := 22.99
  let Cl := 35.45
  let O := 16.00
  sorry

end molecular_weight_NaClO_l107_107623


namespace least_positive_integer_to_multiple_of_4_l107_107790

theorem least_positive_integer_to_multiple_of_4 : ∃ n : ℕ, n > 0 ∧ ((563 + n) % 4 = 0) ∧ n = 1 := 
by
  sorry

end least_positive_integer_to_multiple_of_4_l107_107790


namespace alpha_in_second_quadrant_l107_107382

theorem alpha_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 > 0 :=
by
  -- Given conditions
  have : Real.sin α > 0 := h1
  have : Real.cos α < 0 := h2
  sorry

end alpha_in_second_quadrant_l107_107382


namespace highest_power_of_2_divides_l107_107086

def a : ℕ := 17
def b : ℕ := 15
def n : ℕ := a^5 - b^5

def highestPowerOf2Divides (k : ℕ) : ℕ :=
  -- Function to find the highest power of 2 that divides k, implementation is omitted
  sorry

theorem highest_power_of_2_divides :
  highestPowerOf2Divides n = 2^5 := by
    sorry

end highest_power_of_2_divides_l107_107086


namespace simplify_rationalize_denominator_l107_107155

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l107_107155


namespace percentage_of_males_l107_107115

theorem percentage_of_males (total_employees males_below_50 males_percentage : ℕ) (h1 : total_employees = 800) (h2 : males_below_50 = 120) (h3 : 40 * males_percentage / 100 = 60 * males_below_50):
  males_percentage = 25 :=
by
  sorry

end percentage_of_males_l107_107115


namespace curve_cartesian_eq_correct_intersection_distances_sum_l107_107078

noncomputable section

def curve_parametric_eqns (θ : ℝ) : ℝ × ℝ := 
  (1 + 3 * Real.cos θ, 3 + 3 * Real.sin θ)

def line_parametric_eqns (t : ℝ) : ℝ × ℝ := 
  (3 + (1/2) * t, 3 + (Real.sqrt 3 / 2) * t)

def curve_cartesian_eq (x y : ℝ) : Prop := 
  (x - 1)^2 + (y - 3)^2 = 9

def point_p : ℝ × ℝ := 
  (3, 3)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem curve_cartesian_eq_correct (θ : ℝ) : 
  curve_cartesian_eq (curve_parametric_eqns θ).1 (curve_parametric_eqns θ).2 := 
by 
  sorry

theorem intersection_distances_sum (t1 t2 : ℝ) 
  (h1 : curve_cartesian_eq (line_parametric_eqns t1).1 (line_parametric_eqns t1).2) 
  (h2 : curve_cartesian_eq (line_parametric_eqns t2).1 (line_parametric_eqns t2).2) : 
  distance point_p (line_parametric_eqns t1) + distance point_p (line_parametric_eqns t2) = 2 * Real.sqrt 3 := 
by 
  sorry

end curve_cartesian_eq_correct_intersection_distances_sum_l107_107078


namespace simplify_expression_l107_107832

theorem simplify_expression (k : ℤ) (c d : ℤ) 
(h1 : (5 * k + 15) / 5 = c * k + d) 
(h2 : ∀ k, d + c * k = k + 3) : 
c / d = 1 / 3 := 
by 
  sorry

end simplify_expression_l107_107832


namespace increased_work_l107_107415

variable (W p : ℕ)

theorem increased_work (hW : W > 0) (hp : p > 0) : 
  (W / (7 * p / 8)) - (W / p) = W / (7 * p) := 
sorry

end increased_work_l107_107415


namespace trig_identity_example_l107_107209

theorem trig_identity_example :
  (Real.sin (43 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) - Real.sin (13 * Real.pi / 180) * Real.cos (43 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end trig_identity_example_l107_107209


namespace no_such_A_exists_l107_107641

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_such_A_exists :
  ¬ ∃ A : ℕ, 0 < A ∧ digit_sum A = 16 ∧ digit_sum (2 * A) = 17 :=
by 
  sorry

end no_such_A_exists_l107_107641


namespace min_colors_rect_condition_l107_107496

theorem min_colors_rect_condition (n : ℕ) (hn : n ≥ 2) :
  ∃ k : ℕ, (∀ (coloring : Fin n → Fin n → Fin k), 
           (∀ i j, coloring i j < k) → 
           (∀ c, ∃ i j, coloring i j = c) →
           (∃ i1 i2 j1 j2, i1 ≠ i2 ∧ j1 ≠ j2 ∧ 
                            coloring i1 j1 ≠ coloring i1 j2 ∧ 
                            coloring i1 j1 ≠ coloring i2 j1 ∧ 
                            coloring i1 j2 ≠ coloring i2 j2 ∧ 
                            coloring i2 j1 ≠ coloring i2 j2)) → 
           k = 2 * n :=
sorry

end min_colors_rect_condition_l107_107496


namespace sum_of_even_numbers_l107_107673

-- Define the sequence of even numbers between 1 and 1001
def even_numbers_sequence (n : ℕ) := 2 * n

-- Conditions
def first_term := 2
def last_term := 1000
def common_difference := 2
def num_terms := 500
def sum_arithmetic_series (n : ℕ) (a l : ℕ) := n * (a + l) / 2

-- Main statement to be proved
theorem sum_of_even_numbers : 
  sum_arithmetic_series num_terms first_term last_term = 250502 := 
by
  sorry

end sum_of_even_numbers_l107_107673


namespace determine_pairs_l107_107397

theorem determine_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  (∃ k : ℕ, k > 0 ∧ (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1)) :=
by
  sorry

end determine_pairs_l107_107397


namespace f_of_1_eq_zero_l107_107409

-- Conditions
variables (f : ℝ → ℝ)
-- f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
-- f is a periodic function with a period of 2
def periodic_function (f : ℝ → ℝ) := ∀ x : ℝ, f (x + 2) = f x

-- Theorem statement
theorem f_of_1_eq_zero {f : ℝ → ℝ} (h1 : odd_function f) (h2 : periodic_function f) : f 1 = 0 :=
by { sorry }

end f_of_1_eq_zero_l107_107409


namespace dan_spent_at_music_store_l107_107712

def cost_of_clarinet : ℝ := 130.30
def cost_of_song_book : ℝ := 11.24
def money_left_in_pocket : ℝ := 12.32
def total_spent : ℝ := 129.22

theorem dan_spent_at_music_store : 
  cost_of_clarinet + cost_of_song_book - money_left_in_pocket = total_spent :=
by
  -- Proof omitted.
  sorry

end dan_spent_at_music_store_l107_107712


namespace total_tickets_l107_107911

theorem total_tickets (R K : ℕ) (hR : R = 12) (h_income : 2 * R + (9 / 2) * K = 60) : R + K = 20 :=
sorry

end total_tickets_l107_107911


namespace solve_for_a_plus_b_l107_107112

theorem solve_for_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, (-1 < x ∧ x < 1 / 3) → ax^2 + bx + 1 > 0) →
  a * (-3) + b = -5 :=
by
  intro h
  -- Here we can use the proofs provided in the solution steps.
  sorry

end solve_for_a_plus_b_l107_107112


namespace probability_of_red_ball_is_correct_l107_107432

noncomputable def probability_of_drawing_red_ball (white_balls : ℕ) (red_balls : ℕ) :=
  let total_balls := white_balls + red_balls
  let favorable_outcomes := red_balls
  (favorable_outcomes : ℚ) / total_balls

theorem probability_of_red_ball_is_correct :
  probability_of_drawing_red_ball 5 2 = 2 / 7 :=
by
  sorry

end probability_of_red_ball_is_correct_l107_107432


namespace paint_needed_to_buy_l107_107939

def total_paint := 333
def existing_paint := 157

theorem paint_needed_to_buy : total_paint - existing_paint = 176 := by
  sorry

end paint_needed_to_buy_l107_107939


namespace brian_breath_proof_l107_107148

def breath_holding_time (initial_time: ℕ) (week1_factor: ℝ) (week2_factor: ℝ) 
  (missed_days: ℕ) (missed_decrease: ℝ) (week3_factor: ℝ): ℝ := by
  let week1_time := initial_time * week1_factor
  let hypothetical_week2_time := week1_time * (1 + week2_factor)
  let missed_decrease_total := week1_time * missed_decrease * missed_days
  let effective_week2_time := hypothetical_week2_time - missed_decrease_total
  let final_time := effective_week2_time * (1 + week3_factor)
  exact final_time

theorem brian_breath_proof :
  breath_holding_time 10 2 0.75 2 0.1 0.5 = 46.5 := 
by
  sorry

end brian_breath_proof_l107_107148


namespace cube_root_of_neg_27_over_8_l107_107396

theorem cube_root_of_neg_27_over_8 :
  (- (3 : ℝ) / 2) ^ 3 = - (27 / 8 : ℝ) := 
by
  sorry

end cube_root_of_neg_27_over_8_l107_107396


namespace carrots_total_l107_107942

def carrots_grown_by_sally := 6
def carrots_grown_by_fred := 4
def total_carrots := carrots_grown_by_sally + carrots_grown_by_fred

theorem carrots_total : total_carrots = 10 := 
by 
  sorry  -- proof to be filled in

end carrots_total_l107_107942


namespace arithmetic_sequence_common_difference_l107_107665

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 2 = 1)
  (h2 : a 3 + a 4 = 8)
  (h3 : ∀ n, a (n + 1) = a n + d) : 
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l107_107665


namespace linear_function_passing_points_l107_107788

theorem linear_function_passing_points :
  ∃ k b : ℝ, (∀ x : ℝ, y = k * x + b) ∧ (k * 0 + b = 3) ∧ (k * (-4) + b = 0)
  →
  (∃ a : ℝ, y = -((3:ℝ) / (4:ℝ)) * x + 3 ∧ (∀ x y : ℝ, y = -((3:ℝ) / (4:ℝ)) * a + 3 → y = 6 → a = -4)) :=
by sorry

end linear_function_passing_points_l107_107788


namespace find_x_minus_y_l107_107512

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 :=
by
  have h3 : x^2 - y^2 = (x + y) * (x - y) := by sorry
  have h4 : (x + y) * (x - y) = 8 * (x - y) := by sorry
  have h5 : 16 = 8 * (x - y) := by sorry
  have h6 : 16 = 8 * (x - y) := by sorry
  have h7 : x - y = 2 := by sorry
  exact h7

end find_x_minus_y_l107_107512


namespace range_of_m_l107_107638

noncomputable def p (x : ℝ) : Prop := |x - 3| ≤ 2
noncomputable def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m {m : ℝ} (H : ∀ (x : ℝ), ¬p x → ¬q x m) :
  2 ≤ m ∧ m ≤ 4 :=
sorry

end range_of_m_l107_107638


namespace chelsea_total_time_l107_107469

def num_batches := 4
def bake_time_per_batch := 20  -- minutes
def ice_time_per_batch := 30   -- minutes
def cupcakes_per_batch := 6
def additional_time_first_batch := 10 -- per cupcake
def additional_time_second_batch := 15 -- per cupcake
def additional_time_third_batch := 12 -- per cupcake
def additional_time_fourth_batch := 20 -- per cupcake

def total_bake_ice_time := bake_time_per_batch + ice_time_per_batch
def total_bake_ice_time_all_batches := total_bake_ice_time * num_batches

def total_additional_time_first_batch := additional_time_first_batch * cupcakes_per_batch
def total_additional_time_second_batch := additional_time_second_batch * cupcakes_per_batch
def total_additional_time_third_batch := additional_time_third_batch * cupcakes_per_batch
def total_additional_time_fourth_batch := additional_time_fourth_batch * cupcakes_per_batch

def total_additional_time := 
  total_additional_time_first_batch +
  total_additional_time_second_batch +
  total_additional_time_third_batch +
  total_additional_time_fourth_batch

def total_time := total_bake_ice_time_all_batches + total_additional_time

theorem chelsea_total_time : total_time = 542 := by
  sorry

end chelsea_total_time_l107_107469


namespace race_distance_l107_107902

variable (distance : ℝ)

theorem race_distance :
  (0.25 * distance = 50) → (distance = 200) :=
by
  intro h
  sorry

end race_distance_l107_107902


namespace merill_has_30_marbles_l107_107296

variable (M E : ℕ)

-- Conditions
def merill_twice_as_many_as_elliot : Prop := M = 2 * E
def together_five_fewer_than_selma : Prop := M + E = 45

theorem merill_has_30_marbles (h1 : merill_twice_as_many_as_elliot M E) (h2 : together_five_fewer_than_selma M E) : M = 30 := 
by
  sorry

end merill_has_30_marbles_l107_107296


namespace no_four_digit_number_differs_from_reverse_by_1008_l107_107686

theorem no_four_digit_number_differs_from_reverse_by_1008 :
  ∀ a b c d : ℕ, 
  a < 10 → b < 10 → c < 10 → d < 10 → (999 * (a - d) + 90 * (b - c) ≠ 1008) :=
by
  intro a b c d ha hb hc hd h
  sorry

end no_four_digit_number_differs_from_reverse_by_1008_l107_107686


namespace find_width_l107_107850

variable (a b : ℝ)

def perimeter : ℝ := 6 * a + 4 * b
def length : ℝ := 2 * a + b
def width : ℝ := a + b

theorem find_width (h : perimeter a b = 6 * a + 4 * b)
                   (h₂ : length a b = 2 * a + b) : width a b = (perimeter a b) / 2 - length a b := by
  sorry

end find_width_l107_107850


namespace algorithm_must_have_sequential_structure_l107_107642

-- Definitions for types of structures used in algorithm definitions.
inductive Structure
| Logical
| Selection
| Loop
| Sequential

-- Predicate indicating whether a given Structure is necessary for any algorithm.
def necessary (s : Structure) : Prop :=
  match s with
  | Structure.Logical => False
  | Structure.Selection => False
  | Structure.Loop => False
  | Structure.Sequential => True

-- The theorem statement to prove that the sequential structure is necessary for any algorithm.
theorem algorithm_must_have_sequential_structure :
  necessary Structure.Sequential :=
by
  sorry

end algorithm_must_have_sequential_structure_l107_107642


namespace number_of_white_balls_l107_107263

theorem number_of_white_balls (x : ℕ) : (3 : ℕ) + x = 12 → x = 9 :=
by
  intros h
  sorry

end number_of_white_balls_l107_107263


namespace prove_m_range_l107_107931

theorem prove_m_range (m : ℝ) :
  (∀ x : ℝ, (2 * x + 5) / 3 - 1 ≤ 2 - x → 3 * (x - 1) + 5 > 5 * x + 2 * (m + x)) → m < -3 / 5 := by
  sorry

end prove_m_range_l107_107931


namespace tens_digit_of_6_pow_4_is_9_l107_107973

theorem tens_digit_of_6_pow_4_is_9 : (6 ^ 4 / 10) % 10 = 9 :=
by
  sorry

end tens_digit_of_6_pow_4_is_9_l107_107973


namespace sum_cubes_identity_l107_107648

theorem sum_cubes_identity (x y z : ℝ) (h1 : x + y + z = 10) (h2 : xy + yz + zx = 20) :
    x^3 + y^3 + z^3 - 3 * x * y * z = 400 := by
  sorry

end sum_cubes_identity_l107_107648


namespace chair_cost_l107_107121

theorem chair_cost (T P n : ℕ) (hT : T = 135) (hP : P = 55) (hn : n = 4) : 
  (T - P) / n = 20 := by
  sorry

end chair_cost_l107_107121


namespace sumata_family_miles_driven_l107_107910

def total_miles_driven (days : ℝ) (miles_per_day : ℝ) : ℝ :=
  days * miles_per_day

theorem sumata_family_miles_driven :
  total_miles_driven 5 50 = 250 :=
by
  sorry

end sumata_family_miles_driven_l107_107910


namespace find_a_l107_107877

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + 1) / (x + 1)

theorem find_a (a : ℝ) (h1 : ∃ t, t = (f a 1 - 1) / (1 - 0) ∧ t = ((3 * a - 1) / 4)) : a = -1 :=
by
  -- Auxiliary steps to frame the Lean theorem precisely
  let f1 := f a 1
  have h2 : f1 = (a + 1) / 2 := sorry
  have slope_tangent : ∀ t : ℝ, t = (3 * a - 1) / 4 := sorry
  have tangent_eq : (∀ (x y : ℝ), y - f1 = ((3 * a - 1) / 4) * (x - 1)) := sorry
  have pass_point : ∀ (x y : ℝ), (x, y) = (0, 1) -> (1 : ℝ) - ((a + 1) / 2) = ((1 - 3 * a) / 4) := sorry
  exact sorry

end find_a_l107_107877


namespace slope_of_chord_l107_107494

theorem slope_of_chord (x1 x2 y1 y2 : ℝ) (P : ℝ × ℝ)
    (hp : P = (3, 2))
    (h1 : 4 * x1 ^ 2 + 9 * y1 ^ 2 = 144)
    (h2 : 4 * x2 ^ 2 + 9 * y2 ^ 2 = 144)
    (h3 : (x1 + x2) / 2 = 3)
    (h4 : (y1 + y2) / 2 = 2) : 
    (y1 - y2) / (x1 - x2) = -2 / 3 :=
by
  sorry

end slope_of_chord_l107_107494


namespace B_and_C_finish_in_22_857_days_l107_107568

noncomputable def work_rate_A := 1 / 40
noncomputable def work_rate_B := 1 / 60
noncomputable def work_rate_C := 1 / 80

noncomputable def work_done_by_A : ℚ := 10 * work_rate_A
noncomputable def work_done_by_B : ℚ := 5 * work_rate_B

noncomputable def remaining_work : ℚ := 1 - (work_done_by_A + work_done_by_B)

noncomputable def combined_work_rate_BC : ℚ := work_rate_B + work_rate_C

noncomputable def days_BC_to_finish_remaining_work : ℚ := remaining_work / combined_work_rate_BC

theorem B_and_C_finish_in_22_857_days : days_BC_to_finish_remaining_work = 160 / 7 :=
by
  -- Proof is omitted
  sorry

end B_and_C_finish_in_22_857_days_l107_107568


namespace absent_children_count_l107_107175

theorem absent_children_count (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ)
    (absent_children : ℕ) (total_bananas : ℕ) (present_children : ℕ) :
    total_children = 640 →
    bananas_per_child = 2 →
    extra_bananas_per_child = 2 →
    total_bananas = (total_children * bananas_per_child) →
    present_children = (total_children - absent_children) →
    total_bananas = (present_children * (bananas_per_child + extra_bananas_per_child)) →
    absent_children = 320 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end absent_children_count_l107_107175


namespace children_attended_play_l107_107889

variables (A C : ℕ)

theorem children_attended_play
  (h1 : A + C = 610)
  (h2 : 2 * A + C = 960) : 
  C = 260 := 
by 
  -- Proof goes here
  sorry

end children_attended_play_l107_107889


namespace right_triangle_area_l107_107181

theorem right_triangle_area (a b : ℕ) (h1 : a = 36) (h2 : b = 48) : (1 / 2 : ℚ) * (a * b) = 864 := 
by 
  sorry

end right_triangle_area_l107_107181


namespace find_cd_product_l107_107589

open Complex

theorem find_cd_product :
  let u : ℂ := -3 + 4 * I
  let v : ℂ := 2 - I
  let c : ℂ := -5 + 5 * I
  let d : ℂ := -5 - 5 * I
  c * d = 50 :=
by
  sorry

end find_cd_product_l107_107589


namespace ellipse_equation_l107_107417

theorem ellipse_equation (a b c : ℝ) (h0 : a > b) (h1 : b > 0) (h2 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h3 : dist (3, y) (5 - 5 / 2, 0) = 6.5) (h4 : dist (3, y) (5 + 5 / 2, 0) = 3.5) : 
  ( ∀ x y, (x^2 / 25) + (y^2 / (75 / 4)) = 1 ) :=
sorry

end ellipse_equation_l107_107417


namespace probability_of_drawing_white_ball_is_zero_l107_107321

theorem probability_of_drawing_white_ball_is_zero
  (red_balls blue_balls : ℕ)
  (h1 : red_balls = 3)
  (h2 : blue_balls = 5)
  (white_balls : ℕ)
  (h3 : white_balls = 0) : 
  (0 / (red_balls + blue_balls + white_balls) = 0) :=
sorry

end probability_of_drawing_white_ball_is_zero_l107_107321


namespace liquid_left_after_evaporation_l107_107052

-- Definitions
def solution_y (total_mass : ℝ) : ℝ × ℝ :=
  (0.30 * total_mass, 0.70 * total_mass) -- liquid_x, water

def evaporate_water (initial_water : ℝ) (evaporated_mass : ℝ) : ℝ :=
  initial_water - evaporated_mass

-- Condition that new solution is 45% liquid x
theorem liquid_left_after_evaporation 
  (initial_mass : ℝ) 
  (evaporated_mass : ℝ)
  (added_mass : ℝ)
  (new_percentage_liquid_x : ℝ) :
  initial_mass = 8 → 
  evaporated_mass = 4 → 
  added_mass = 4 →
  new_percentage_liquid_x = 0.45 →
  solution_y initial_mass = (2.4, 5.6) →
  evaporate_water 5.6 evaporated_mass = 1.6 →
  solution_y added_mass = (1.2, 2.8) →
  2.4 + 1.2 = 3.6 →
  1.6 + 2.8 = 4.4 →
  0.45 * (3.6 + 4.4) = 3.6 →
  4 = 2.4 + 1.6 := sorry

end liquid_left_after_evaporation_l107_107052


namespace a100_gt_two_pow_99_l107_107533

theorem a100_gt_two_pow_99 (a : ℕ → ℤ) (h_pos : ∀ n, 0 < a n) 
  (h1 : a 1 > a 0) (h_rec : ∀ n ≥ 2, a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2 ^ 99 :=
sorry

end a100_gt_two_pow_99_l107_107533


namespace max_difference_is_62_l107_107119

open Real

noncomputable def max_difference_of_integers : ℝ :=
  let a (k : ℝ) := 2 * k + 1 + sqrt (8 * k)
  let b (k : ℝ) := 2 * k + 1 - sqrt (8 * k)
  let diff (k : ℝ) := a k - b k
  let max_k := 120 -- Maximum integer value k such that 2k + 1 + sqrt(8k) < 1000
  diff max_k

theorem max_difference_is_62 :
  max_difference_of_integers = 62 :=
sorry

end max_difference_is_62_l107_107119


namespace total_mileage_pay_l107_107604

-- Conditions
def distance_first_package : ℕ := 10
def distance_second_package : ℕ := 28
def distance_third_package : ℕ := distance_second_package / 2
def total_miles_driven : ℕ := distance_first_package + distance_second_package + distance_third_package
def pay_per_mile : ℕ := 2

-- Proof statement
theorem total_mileage_pay (X : ℕ) : 
  X + (total_miles_driven * pay_per_mile) = X + 104 := by
sorry

end total_mileage_pay_l107_107604


namespace problem_KMO_16_l107_107912

theorem problem_KMO_16
  (m : ℕ) (h_pos : m > 0) :
  (2^(m+1) + 1) ∣ (3^(2^m) + 1) ↔ Nat.Prime (2^(m+1) + 1) :=
by
  sorry

end problem_KMO_16_l107_107912


namespace pqrsum_l107_107114

-- Given constants and conditions:
variables {p q r : ℝ} -- p, q, r are real numbers
axiom Hpq : p < q -- given condition p < q
axiom Hineq : ∀ x : ℝ, (x > 5 ∨ 7 ≤ x ∧ x ≤ 15) ↔ ( (x - p) * (x - q) / (x - r) ≥ 0) -- given inequality condition

-- Values from the solution:
axiom Hp : p = 7
axiom Hq : q = 15
axiom Hr : r = 5

-- Proof statement:
theorem pqrsum : p + 2 * q + 3 * r = 52 :=
sorry 

end pqrsum_l107_107114


namespace arithmetic_sequence_a9_l107_107020

theorem arithmetic_sequence_a9 (S : ℕ → ℤ) (a : ℕ → ℤ) :
  S 8 = 4 * a 3 → a 7 = -2 → a 9 = -6 := by
  sorry

end arithmetic_sequence_a9_l107_107020


namespace sequence_general_term_l107_107120

noncomputable def a (n : ℕ) : ℝ :=
if n = 1 then 1 else (n : ℝ) / (2 ^ (n - 1))

theorem sequence_general_term (n : ℕ) (hn : n ≠ 0) : 
  a n = if n = 1 then 1 else (n : ℝ) / (2 ^ (n - 1)) :=
by
  sorry

end sequence_general_term_l107_107120


namespace find_n_value_l107_107016

theorem find_n_value : 
  ∃ (n : ℕ), ∀ (a b c : ℕ), 
    a + b + c = 200 ∧ 
    (∃ bc ca ab : ℕ, bc = b * c ∧ ca = c * a ∧ ab = a * b ∧ n = bc ∧ n = ca ∧ n = ab) → 
    n = 199 := sorry

end find_n_value_l107_107016


namespace soda_cost_l107_107355

theorem soda_cost (b s : ℕ) 
  (h₁ : 3 * b + 2 * s = 450) 
  (h₂ : 2 * b + 3 * s = 480) : 
  s = 108 := 
by
  sorry

end soda_cost_l107_107355


namespace convert_deg_to_rad_l107_107439

theorem convert_deg_to_rad (deg_to_rad : ℝ → ℝ) (conversion_factor : deg_to_rad 1 = π / 180) :
  deg_to_rad (-300) = - (5 * π) / 3 :=
by
  sorry

end convert_deg_to_rad_l107_107439


namespace unique_positive_real_solution_l107_107588

-- Define the function
def f (x : ℝ) : ℝ := x^11 + 9 * x^10 + 19 * x^9 + 2023 * x^8 - 1421 * x^7 + 5

-- Prove the statement
theorem unique_positive_real_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end unique_positive_real_solution_l107_107588


namespace sum_positive_implies_at_least_one_positive_l107_107926

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l107_107926


namespace next_volunteer_day_l107_107088

-- Definitions based on conditions.
def Alison_schedule := 5
def Ben_schedule := 3
def Carla_schedule := 9
def Dave_schedule := 8

-- Main theorem
theorem next_volunteer_day : Nat.lcm Alison_schedule (Nat.lcm Ben_schedule (Nat.lcm Carla_schedule Dave_schedule)) = 360 := by
  sorry

end next_volunteer_day_l107_107088


namespace lemons_minus_pears_l107_107841

theorem lemons_minus_pears
  (apples : ℕ)
  (pears : ℕ)
  (tangerines : ℕ)
  (lemons : ℕ)
  (watermelons : ℕ)
  (h1 : apples = 8)
  (h2 : pears = 5)
  (h3 : tangerines = 12)
  (h4 : lemons = 17)
  (h5 : watermelons = 10) :
  lemons - pears = 12 := 
sorry

end lemons_minus_pears_l107_107841


namespace parallelogram_area_l107_107818

-- Definitions
def base_cm : ℕ := 22
def height_cm : ℕ := 21

-- Theorem statement
theorem parallelogram_area : base_cm * height_cm = 462 := by
  sorry

end parallelogram_area_l107_107818


namespace vector_dot_product_l107_107834

def vector := ℝ × ℝ

def collinear (a b : vector) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

noncomputable def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product (k : ℝ) (h_collinear : collinear (3 / 2, 1) (3, k))
  (h_k : k = 2) :
  dot_product ((3 / 2, 1) - (3, k)) (2 * (3 / 2, 1) + (3, k)) = -13 :=
by
  sorry

end vector_dot_product_l107_107834


namespace product_of_roots_is_four_thirds_l107_107932

theorem product_of_roots_is_four_thirds :
  (∀ p q r s : ℚ, (∃ a b c: ℚ, (3 * a^3 - 9 * a^2 + 5 * a - 4 = 0 ∧
                                   3 * b^3 - 9 * b^2 + 5 * b - 4 = 0 ∧
                                   3 * c^3 - 9 * c^2 + 5 * c - 4 = 0)) → 
  - s / p = (4 : ℚ) / 3) := sorry

end product_of_roots_is_four_thirds_l107_107932


namespace ab_greater_than_a_plus_b_l107_107891

theorem ab_greater_than_a_plus_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a - b = a / b) : ab > a + b :=
sorry

end ab_greater_than_a_plus_b_l107_107891


namespace find_weight_b_l107_107937

theorem find_weight_b (A B C : ℕ) 
  (h1 : A + B + C = 90)
  (h2 : A + B = 50)
  (h3 : B + C = 56) : 
  B = 16 :=
sorry

end find_weight_b_l107_107937


namespace range_of_a_l107_107498

theorem range_of_a (a : ℝ) :
  ¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 ≤ 0 → -1 < a ∧ a < 3 :=
by
  intro h
  sorry

end range_of_a_l107_107498


namespace symmetry_propositions_l107_107869

noncomputable def verify_symmetry_conditions (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  Prop :=
  -- This defines the propositions to be proven
  (∀ x : ℝ, a^x - 1 = a^(-x) - 1) ∧
  (∀ x : ℝ, a^(x - 2) = a^(2 - x)) ∧
  (∀ x : ℝ, a^(x + 2) = a^(2 - x))

-- Create the problem statement
theorem symmetry_propositions (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  verify_symmetry_conditions a h1 h2 :=
sorry

end symmetry_propositions_l107_107869


namespace lcm_of_coprimes_eq_product_l107_107613

theorem lcm_of_coprimes_eq_product (a b c : ℕ) (h_coprime_ab : Nat.gcd a b = 1) (h_coprime_bc : Nat.gcd b c = 1) (h_coprime_ca : Nat.gcd c a = 1) (h_product : a * b * c = 7429) :
  Nat.lcm (Nat.lcm a b) c = 7429 :=
by 
  sorry

end lcm_of_coprimes_eq_product_l107_107613


namespace complex_plane_second_quadrant_l107_107072

theorem complex_plane_second_quadrant (x : ℝ) :
  (x ^ 2 - 6 * x + 5 < 0 ∧ x - 2 > 0) ↔ (2 < x ∧ x < 5) :=
by
  -- The proof is to be completed.
  sorry

end complex_plane_second_quadrant_l107_107072


namespace negation_of_forall_statement_l107_107584

variable (x : ℝ)

theorem negation_of_forall_statement :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by
  sorry

end negation_of_forall_statement_l107_107584


namespace sufficient_not_necessary_l107_107544

theorem sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (∃ x, 1 / x < 1 ∧ ¬(x > 1)) :=
by
  sorry

end sufficient_not_necessary_l107_107544


namespace car_speed_ratio_l107_107134

noncomputable def speed_ratio (t_round_trip t_leaves t_returns t_walk_start t_walk_end : ℕ) (meet_time : ℕ) : ℕ :=
  let one_way_time_car := t_round_trip / 2
  let total_car_time := t_returns - t_leaves
  let meeting_time_car := total_car_time / 2
  let remaining_time_to_factory := one_way_time_car - meeting_time_car
  let total_walk_time := t_walk_end - t_walk_start
  total_walk_time / remaining_time_to_factory

theorem car_speed_ratio :
  speed_ratio 60 120 160 60 140 80 = 8 :=
by
  sorry

end car_speed_ratio_l107_107134


namespace length_of_arc_l107_107204

theorem length_of_arc (S : ℝ) (α : ℝ) (hS : S = 4) (hα : α = 2) : 
  ∃ l : ℝ, l = 4 :=
by
  sorry

end length_of_arc_l107_107204


namespace nelly_earns_per_night_l107_107185

/-- 
  Nelly wants to buy pizza for herself and her 14 friends. Each pizza costs $12 and can feed 3 
  people. Nelly has to babysit for 15 nights to afford the pizza. We need to prove that Nelly earns 
  $4 per night babysitting.
--/
theorem nelly_earns_per_night 
  (total_people : ℕ) (people_per_pizza : ℕ) 
  (cost_per_pizza : ℕ) (total_nights : ℕ) (total_cost : ℕ) 
  (total_pizzas : ℕ) (cost_per_night : ℕ)
  (h1 : total_people = 15)
  (h2 : people_per_pizza = 3)
  (h3 : cost_per_pizza = 12)
  (h4 : total_nights = 15)
  (h5 : total_pizzas = total_people / people_per_pizza)
  (h6 : total_cost = total_pizzas * cost_per_pizza)
  (h7 : cost_per_night = total_cost / total_nights) :
  cost_per_night = 4 := sorry

end nelly_earns_per_night_l107_107185


namespace max_a4_l107_107363

variable (a1 d : ℝ)

theorem max_a4 (h1 : 2 * a1 + 6 * d ≥ 10) (h2 : 2.5 * a1 + 10 * d ≤ 15) :
  ∃ max_a4, max_a4 = 4 ∧ a1 + 3 * d ≤ max_a4 :=
by
  sorry

end max_a4_l107_107363


namespace part1_part2_l107_107001

-- Definitions based on the conditions
def a_i (i : ℕ) : ℕ := sorry -- Define ai's values based on the given conditions
def f (n : ℕ) : ℕ := sorry  -- Define f(n) as the number of n-digit wave numbers satisfying the given conditions

-- Prove the first part: f(10) = 3704
theorem part1 : f 10 = 3704 := sorry

-- Prove the second part: f(2008) % 13 = 10
theorem part2 : (f 2008) % 13 = 10 := sorry

end part1_part2_l107_107001


namespace compare_abc_l107_107198

noncomputable def a : ℝ := (2 / 5) ^ (3 / 5)
noncomputable def b : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def c : ℝ := (3 / 5) ^ (2 / 5)

theorem compare_abc : a < b ∧ b < c := sorry

end compare_abc_l107_107198


namespace base_k_132_eq_30_l107_107312

theorem base_k_132_eq_30 (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 30) : k = 4 :=
sorry

end base_k_132_eq_30_l107_107312


namespace log_equation_solution_l107_107067

theorem log_equation_solution (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) (h₃ : x ≠ 1/16) (h₄ : x ≠ 1/2) 
    (h_eq : (Real.log 2 / Real.log (4 * Real.sqrt x)) / (Real.log 2 / Real.log (2 * x)) 
            + (Real.log 2 / Real.log (2 * x)) * (Real.log (2 * x) / Real.log (1 / 2)) = 0) 
    : x = 4 := 
sorry

end log_equation_solution_l107_107067


namespace marbles_lost_l107_107009

def initial_marbles := 8
def current_marbles := 6

theorem marbles_lost : initial_marbles - current_marbles = 2 :=
by
  sorry

end marbles_lost_l107_107009


namespace digit_x_for_divisibility_by_29_l107_107154

-- Define the base 7 number 34x1_7 in decimal form
def base7_to_decimal (x : ℕ) : ℕ := 3 * 7^3 + 4 * 7^2 + x * 7 + 1

-- State the proof problem
theorem digit_x_for_divisibility_by_29 (x : ℕ) (h : base7_to_decimal x % 29 = 0) : x = 3 :=
by
  sorry

end digit_x_for_divisibility_by_29_l107_107154


namespace factorization_of_cubic_polynomial_l107_107499

-- Define the elements and the problem
variable (a : ℝ)

theorem factorization_of_cubic_polynomial :
  a^3 - 3 * a = a * (a + Real.sqrt 3) * (a - Real.sqrt 3) := by
  sorry

end factorization_of_cubic_polynomial_l107_107499


namespace algebraic_expression_simplification_l107_107174

theorem algebraic_expression_simplification (k x : ℝ) (h : (x - k * x) * (2 * x - k * x) - 3 * x * (2 * x - k * x) = 5 * x^2) :
  k = 3 ∨ k = -3 :=
by {
  sorry
}

end algebraic_expression_simplification_l107_107174


namespace ratio_of_larger_to_smaller_l107_107326

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx : x > y) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l107_107326


namespace find_D_l107_107406

theorem find_D (D E F : ℝ) (h : ∀ x : ℝ, x ≠ 1 → x ≠ -2 → (1 / (x^3 - 3*x^2 - 4*x + 12)) = (D / (x - 1)) + (E / (x + 2)) + (F / (x + 2)^2)) :
    D = -1 / 15 :=
by
  -- the proof is omitted as per the instructions
  sorry

end find_D_l107_107406


namespace evaluate_expression_l107_107408

variables (a b c d m : ℝ)

lemma opposite_is_zero (h1 : a + b = 0) : a + b = 0 := h1

lemma reciprocals_equal_one (h2 : c * d = 1) : c * d = 1 := h2

lemma abs_value_two (h3 : |m| = 2) : |m| = 2 := h3

theorem evaluate_expression (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 2) :
  m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 :=
by
  sorry

end evaluate_expression_l107_107408


namespace percentage_difference_l107_107338

theorem percentage_difference :
  let a1 := 0.12 * 24.2
  let a2 := 0.10 * 14.2
  a1 - a2 = 1.484 := 
by
  -- Definitions
  let a1 := 0.12 * 24.2
  let a2 := 0.10 * 14.2
  -- Proof body (skipped for this task)
  sorry

end percentage_difference_l107_107338


namespace negation_of_proposition_l107_107369

theorem negation_of_proposition :
  ¬(∃ x₀ : ℝ, 0 < x₀ ∧ Real.log x₀ = x₀ - 1) ↔ ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by
  sorry

end negation_of_proposition_l107_107369


namespace molecular_weight_l107_107804

variable (weight_moles : ℝ) (moles : ℝ)

-- Given conditions
axiom h1 : weight_moles = 699
axiom h2 : moles = 3

-- Concluding statement to prove
theorem molecular_weight : (weight_moles / moles) = 233 := sorry

end molecular_weight_l107_107804


namespace find_original_prices_and_discount_l107_107386

theorem find_original_prices_and_discount :
  ∃ x y a : ℝ,
  (6 * x + 5 * y = 1140) ∧
  (3 * x + 7 * y = 1110) ∧
  (((9 * x + 8 * y) - 1062) / (9 * x + 8 * y) = a) ∧
  x = 90 ∧
  y = 120 ∧
  a = 0.4 :=
by
  sorry

end find_original_prices_and_discount_l107_107386


namespace first_bell_weight_l107_107962

-- Given conditions from the problem
variable (x : ℕ) -- weight of the first bell in pounds
variable (total_weight : ℕ)

-- The condition as the sum of the weights
def bronze_weights (x total_weight : ℕ) : Prop :=
  x + 2 * x + 8 * 2 * x = total_weight

-- Prove that the weight of the first bell is 50 pounds given the total weight is 550 pounds
theorem first_bell_weight : bronze_weights x 550 → x = 50 := by
  intro h
  sorry

end first_bell_weight_l107_107962


namespace isosceles_triangle_base_length_l107_107436

theorem isosceles_triangle_base_length (s a b : ℕ) (h1 : 3 * s = 45)
  (h2 : 2 * a + b = 40) (h3 : a = s) : b = 10 :=
by
  sorry

end isosceles_triangle_base_length_l107_107436


namespace variance_defect_rate_l107_107689

noncomputable def defect_rate : ℝ := 0.02
noncomputable def number_of_trials : ℕ := 100
noncomputable def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem variance_defect_rate :
  variance_binomial number_of_trials defect_rate = 1.96 :=
by
  sorry

end variance_defect_rate_l107_107689


namespace sum_of_inscribed_angles_l107_107305

-- Define the circle and its division into arcs.
def circle_division (O : Type) (total_arcs : ℕ) := total_arcs = 16

-- Define the inscribed angles x and y.
def inscribed_angle (O : Type) (arc_subtended : ℕ) := arc_subtended

-- Define the conditions for angles x and y subtending 3 and 5 arcs respectively.
def angle_x := inscribed_angle ℝ 3
def angle_y := inscribed_angle ℝ 5

-- Theorem stating the sum of the inscribed angles x and y.
theorem sum_of_inscribed_angles 
  (O : Type)
  (total_arcs : ℕ)
  (h1 : circle_division O total_arcs)
  (h2 : inscribed_angle O angle_x = 3)
  (h3 : inscribed_angle O angle_y = 5) :
  33.75 + 56.25 = 90 :=
by
  sorry

end sum_of_inscribed_angles_l107_107305


namespace grocer_rows_count_l107_107739

theorem grocer_rows_count (n : ℕ) (a d S : ℕ) (h_a : a = 1) (h_d : d = 3) (h_S : S = 225)
  (h_sum : S = n * (2 * a + (n - 1) * d) / 2) : n = 16 :=
by {
  sorry
}

end grocer_rows_count_l107_107739


namespace painter_red_cells_count_l107_107029

open Nat

/-- Prove the number of red cells painted by the painter in the given 2000 x 70 grid. -/
theorem painter_red_cells_count :
  let rows := 2000
  let columns := 70
  let lcm_rc := Nat.lcm rows columns -- Calculate the LCM of row and column counts
  lcm_rc = 14000 := by
sorry

end painter_red_cells_count_l107_107029


namespace students_in_second_class_l107_107049

theorem students_in_second_class 
    (avg1 : ℝ)
    (n1 : ℕ)
    (avg2 : ℝ)
    (total_avg : ℝ)
    (x : ℕ)
    (h1 : avg1 = 40)
    (h2 : n1 = 26)
    (h3 : avg2 = 60)
    (h4 : total_avg = 53.1578947368421)
    (h5 : (n1 * avg1 + x * avg2) / (n1 + x) = total_avg) :
  x = 50 :=
by
  sorry

end students_in_second_class_l107_107049


namespace overall_labor_costs_l107_107510

noncomputable def construction_worker_daily_wage : ℝ := 100
noncomputable def electrician_daily_wage : ℝ := 2 * construction_worker_daily_wage
noncomputable def plumber_daily_wage : ℝ := 2.5 * construction_worker_daily_wage

noncomputable def total_construction_work : ℝ := 2 * construction_worker_daily_wage
noncomputable def total_electrician_work : ℝ := electrician_daily_wage
noncomputable def total_plumber_work : ℝ := plumber_daily_wage

theorem overall_labor_costs :
  total_construction_work + total_electrician_work + total_plumber_work = 650 :=
by
  sorry

end overall_labor_costs_l107_107510


namespace max_integer_a_l107_107709

theorem max_integer_a :
  ∀ (a: ℤ), (∀ x: ℝ, (a + 1) * x^2 - 2 * x + 3 = 0 → (a = -2 → (-12 * a - 8) ≥ 0)) → (∀ a ≤ -2, a ≠ -1) :=
by
  sorry

end max_integer_a_l107_107709


namespace cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths_l107_107620

theorem cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths 
  (b : ℝ)
  (h : ∀ x : ℝ, 4 * x^3 + 3 * x^2 + b * x + 27 = 0 → ∃! r : ℝ, r = x) :
  b = 3 / 4 := 
by
  sorry

end cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths_l107_107620


namespace first_place_team_wins_l107_107278

-- Define the conditions in Lean 4
variable (joe_won : ℕ := 1) (joe_draw : ℕ := 3) (fp_draw : ℕ := 2) (joe_points : ℕ := 3 * joe_won + joe_draw)
variable (fp_points : ℕ := joe_points + 2)

 -- Define the proof problem
theorem first_place_team_wins : 3 * (fp_points - fp_draw) / 3 = 2 := by
  sorry

end first_place_team_wins_l107_107278


namespace f_value_at_3_l107_107007

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

def periodic_shift (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (x + 2) = f x + 2

theorem f_value_at_3 (h_odd : odd_function f) (h_value : f (-1) = 1/2) (h_periodic : periodic_shift f) : 
  f 3 = 3 / 2 := 
sorry

end f_value_at_3_l107_107007


namespace cyclic_sum_inequality_l107_107109

theorem cyclic_sum_inequality (n : ℕ) (a : Fin n.succ -> ℕ) (h : ∀ i, a i > 0) : 
  (Finset.univ.sum fun i => a i / a ((i + 1) % n)) ≥ n :=
by
  sorry

end cyclic_sum_inequality_l107_107109


namespace red_peaches_each_basket_l107_107189

variable (TotalGreenPeachesInABasket : Nat) (TotalPeachesInABasket : Nat)

theorem red_peaches_each_basket (h1 : TotalPeachesInABasket = 10) (h2 : TotalGreenPeachesInABasket = 3) :
  (TotalPeachesInABasket - TotalGreenPeachesInABasket) = 7 := by
  sorry

end red_peaches_each_basket_l107_107189


namespace prob_of_yellow_second_l107_107909

-- Defining the probabilities based on the given conditions
def prob_white_from_X : ℚ := 5 / 8
def prob_black_from_X : ℚ := 3 / 8
def prob_yellow_from_Y : ℚ := 8 / 10
def prob_yellow_from_Z : ℚ := 3 / 7

-- Combining probabilities
def combined_prob_white_Y : ℚ := prob_white_from_X * prob_yellow_from_Y
def combined_prob_black_Z : ℚ := prob_black_from_X * prob_yellow_from_Z

-- Total probability of drawing a yellow marble in the second draw
def total_prob_yellow_second : ℚ := combined_prob_white_Y + combined_prob_black_Z

-- Proof statement
theorem prob_of_yellow_second :
  total_prob_yellow_second = 37 / 56 := 
sorry

end prob_of_yellow_second_l107_107909


namespace jose_fewer_rocks_l107_107723

theorem jose_fewer_rocks (J : ℕ) (H1 : 80 = J + 14) (H2 : J + 20 = 86) (H3 : J < 80) : J = 66 :=
by
  -- Installation of other conditions derived from the proof
  have H_albert_collected : 86 = 80 + 6 := by rfl
  have J_def : J = 86 - 20 := by sorry
  sorry

end jose_fewer_rocks_l107_107723


namespace ratio_of_speeds_l107_107252

/-- Define the conditions -/
def distance_AB : ℝ := 540 -- Distance between city A and city B is 540 km
def time_Eddy : ℝ := 3     -- Eddy takes 3 hours to travel to city B
def distance_AC : ℝ := 300 -- Distance between city A and city C is 300 km
def time_Freddy : ℝ := 4   -- Freddy takes 4 hours to travel to city C

/-- Define the average speeds -/
noncomputable def avg_speed_Eddy : ℝ := distance_AB / time_Eddy
noncomputable def avg_speed_Freddy : ℝ := distance_AC / time_Freddy

/-- The statement to prove -/
theorem ratio_of_speeds : avg_speed_Eddy / avg_speed_Freddy = 12 / 5 :=
by sorry

end ratio_of_speeds_l107_107252


namespace logarithmic_AMGM_inequality_l107_107380

theorem logarithmic_AMGM_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  2 * ((Real.log b / (a * Real.log a)) / (a + b) + 
       (Real.log c / (b * Real.log b)) / (b + c) + 
       (Real.log a / (c * Real.log c)) / (c + a)) 
  ≥ 9 / (a + b + c) := 
sorry

end logarithmic_AMGM_inequality_l107_107380


namespace intersecting_lines_a_value_l107_107237

theorem intersecting_lines_a_value :
  ∀ t a b : ℝ, (b = 12) ∧ (b = 2 * a + t) ∧ (t = 4) → a = 4 :=
by
  intros t a b h
  obtain ⟨hb1, hb2, ht⟩ := h
  sorry

end intersecting_lines_a_value_l107_107237


namespace decrease_angle_equilateral_l107_107233

theorem decrease_angle_equilateral (D E F : ℝ) (h : D = 60) (h_equilateral : D = E ∧ E = F) (h_decrease : D' = D - 20) :
  ∃ max_angle : ℝ, max_angle = 70 :=
by
  sorry

end decrease_angle_equilateral_l107_107233


namespace greatest_integer_a_l107_107253

theorem greatest_integer_a (a : ℤ) : a * a < 44 → a ≤ 6 :=
by
  intros h
  sorry

end greatest_integer_a_l107_107253


namespace find_a_if_f_is_even_l107_107773

-- Defining f as given in the problem conditions
noncomputable def f (x a : ℝ) : ℝ := (x + a) * 3 ^ (x - 2 + a ^ 2) - (x - a) * 3 ^ (8 - x - 3 * a)

-- Statement of the proof problem with the conditions
theorem find_a_if_f_is_even (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → (a = -5 ∨ a = 2) :=
by
  sorry

end find_a_if_f_is_even_l107_107773


namespace simplify_expression_zero_l107_107346

noncomputable def simplify_expression (a b c d : ℝ) : ℝ :=
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)

theorem simplify_expression_zero (a b c d : ℝ) (h : a + b + c = d)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  simplify_expression a b c d = 0 :=
by
  sorry

end simplify_expression_zero_l107_107346


namespace geometric_series_common_ratio_l107_107975

theorem geometric_series_common_ratio (a₁ q : ℝ) (S₃ : ℝ)
  (h1 : S₃ = 7 * a₁)
  (h2 : S₃ = a₁ + a₁ * q + a₁ * q^2) :
  q = 2 ∨ q = -3 :=
by
  sorry

end geometric_series_common_ratio_l107_107975


namespace sum_of_interior_angles_l107_107139

theorem sum_of_interior_angles (n : ℕ) (h1 : 180 * (n - 2) = 1800) (h2 : n = 12) : 
  180 * ((n + 4) - 2) = 2520 := 
by 
  { sorry }

end sum_of_interior_angles_l107_107139


namespace sqrt_36_eq_pm6_arith_sqrt_sqrt_16_eq_2_cube_root_minus_27_eq_minus_3_l107_107786

-- Prove that the square root of 36 equals ±6
theorem sqrt_36_eq_pm6 : ∃ y : ℤ, y * y = 36 ∧ y = 6 ∨ y = -6 :=
by
  sorry

-- Prove that the arithmetic square root of sqrt(16) equals 2
theorem arith_sqrt_sqrt_16_eq_2 : ∃ z : ℝ, z * z = 16 ∧ z = 4 ∧ 2 * 2 = z :=
by
  sorry

-- Prove that the cube root of -27 equals -3
theorem cube_root_minus_27_eq_minus_3 : ∃ x : ℝ, x * x * x = -27 ∧ x = -3 :=
by
  sorry

end sqrt_36_eq_pm6_arith_sqrt_sqrt_16_eq_2_cube_root_minus_27_eq_minus_3_l107_107786


namespace part1_part2_l107_107416

-- Define the solution set M for the inequality
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- Define the problem conditions
variables {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M)

-- First part: Prove that |(1/3)a + (1/6)b| < 1/4
theorem part1 : |(1/3 : ℝ) * a + (1/6 : ℝ) * b| < 1/4 :=
sorry

-- Second part: Prove that |1 - 4 * a * b| > 2 * |a - b|
theorem part2 : |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end part1_part2_l107_107416


namespace area_of_square_l107_107775

noncomputable def length_of_rectangle (r : ℝ) : ℝ := (2 / 5) * r
noncomputable def area_of_rectangle_given_length_and_breadth (L B : ℝ) : ℝ := L * B

theorem area_of_square (r : ℝ) (B : ℝ) (A : ℝ) 
  (h_length : length_of_rectangle r = (2 / 5) * r) 
  (h_breadth : B = 10) 
  (h_area : A = 160) 
  (h_rectangle_area : area_of_rectangle_given_length_and_breadth ((2 / 5) * r) B = 160) : 
  r = 40 → (r ^ 2 = 1600) := 
by 
  sorry

end area_of_square_l107_107775


namespace sqrt_of_quarter_l107_107757

-- Definitions as per conditions
def is_square_root (x y : ℝ) : Prop := x^2 = y

-- Theorem statement proving question == answer given conditions
theorem sqrt_of_quarter : is_square_root 0.5 0.25 ∧ is_square_root (-0.5) 0.25 ∧ (∀ x, is_square_root x 0.25 → (x = 0.5 ∨ x = -0.5)) :=
by
  -- Skipping proof with sorry
  sorry

end sqrt_of_quarter_l107_107757


namespace quadratic_range_l107_107693

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x + 7

-- Defining the range of the quadratic function for the interval -1 < x < 4
theorem quadratic_range (y : ℝ) : 3 ≤ y ∧ y < 12 ↔ ∃ x : ℝ, -1 < x ∧ x < 4 ∧ y = quadratic_function x :=
by
  sorry

end quadratic_range_l107_107693


namespace sum_of_angles_l107_107066

theorem sum_of_angles 
    (ABC_isosceles : ∃ (A B C : Type) (angleBAC : ℝ), (AB = AC) ∧ (angleBAC = 25))
    (DEF_isosceles : ∃ (D E F : Type) (angleEDF : ℝ), (DE = DF) ∧ (angleEDF = 40)) 
    (AD_parallel_CE : Prop) : 
    ∃ (angleDAC angleADE : ℝ), angleDAC = 77.5 ∧ angleADE = 70 ∧ (angleDAC + angleADE = 147.5) :=
by {
  sorry
}

end sum_of_angles_l107_107066


namespace product_of_t_values_l107_107468

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l107_107468


namespace intersect_complement_A_and_B_l107_107075

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem intersect_complement_A_and_B : (Set.compl A ∩ B) = {x | -1 ≤ x ∧ x < 3} := by
  sorry

end intersect_complement_A_and_B_l107_107075


namespace cos_neg_two_pi_over_three_eq_l107_107027

noncomputable def cos_neg_two_pi_over_three : ℝ := -2 * Real.pi / 3

theorem cos_neg_two_pi_over_three_eq :
  Real.cos cos_neg_two_pi_over_three = -1 / 2 :=
sorry

end cos_neg_two_pi_over_three_eq_l107_107027


namespace floor_problem_2020_l107_107297

-- Define the problem statement
theorem floor_problem_2020:
  2020 ^ 2021 - (Int.floor ((2020 ^ 2021 : ℝ) / 2021) * 2021) = 2020 :=
sorry

end floor_problem_2020_l107_107297


namespace average_sales_per_month_after_discount_is_93_l107_107905

theorem average_sales_per_month_after_discount_is_93 :
  let salesJanuary := 120
  let salesFebruary := 80
  let salesMarch := 70
  let salesApril := 150
  let salesMayBeforeDiscount := 50
  let discountRate := 0.10
  let discountedSalesMay := salesMayBeforeDiscount - (discountRate * salesMayBeforeDiscount)
  let totalSales := salesJanuary + salesFebruary + salesMarch + salesApril + discountedSalesMay
  let numberOfMonths := 5
  let averageSales := totalSales / numberOfMonths
  averageSales = 93 :=
by {
  -- The actual proof code would go here, but we will skip the proof steps as instructed.
  sorry
}

end average_sales_per_month_after_discount_is_93_l107_107905


namespace value_of_expression_l107_107733

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 2 * x + 5 = 9) : 3 * x^2 + 3 * x - 7 = -1 :=
by
  -- The proof would go here
  sorry

end value_of_expression_l107_107733


namespace log_9_256_eq_4_log_2_3_l107_107882

noncomputable def logBase9Base2Proof : Prop :=
  (Real.log 256 / Real.log 9 = 4 * (Real.log 3 / Real.log 2))

theorem log_9_256_eq_4_log_2_3 : logBase9Base2Proof :=
by
  sorry

end log_9_256_eq_4_log_2_3_l107_107882
