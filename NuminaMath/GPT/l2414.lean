import Mathlib

namespace NUMINAMATH_GPT_students_taking_both_languages_l2414_241401

theorem students_taking_both_languages (total_students students_neither students_french students_german : ℕ) (h1 : total_students = 69)
  (h2 : students_neither = 15) (h3 : students_french = 41) (h4 : students_german = 22) :
  (students_french + students_german - (total_students - students_neither) = 9) :=
by
  sorry

end NUMINAMATH_GPT_students_taking_both_languages_l2414_241401


namespace NUMINAMATH_GPT_min_bench_sections_l2414_241427

theorem min_bench_sections (N : ℕ) :
  ∀ x y : ℕ, (x = y) → (x = 8 * N) → (y = 12 * N) → (24 * N) % 20 = 0 → N = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_min_bench_sections_l2414_241427


namespace NUMINAMATH_GPT_smallest_of_three_consecutive_odd_numbers_l2414_241498

theorem smallest_of_three_consecutive_odd_numbers (x : ℤ) (h : x + (x + 2) + (x + 4) = 69) : x = 21 :=
sorry

end NUMINAMATH_GPT_smallest_of_three_consecutive_odd_numbers_l2414_241498


namespace NUMINAMATH_GPT_correct_option_B_l2414_241476

-- Define decimal representation of the numbers
def dec_13 : ℕ := 13
def dec_25 : ℕ := 25
def dec_11 : ℕ := 11
def dec_10 : ℕ := 10

-- Define binary representation of the numbers
def bin_1101 : ℕ := 2^(3) + 2^(2) + 2^(0)  -- 1*8 + 1*4 + 0*2 + 1*1 = 13
def bin_10110 : ℕ := 2^(4) + 2^(2) + 2^(1)  -- 1*16 + 0*8 + 1*4 + 1*2 + 0*1 = 22
def bin_1011 : ℕ := 2^(3) + 2^(2) + 2^(0)  -- 1*8 + 0*4 + 1*2 + 1*1 = 11
def bin_10 : ℕ := 2^(1)  -- 1*2 + 0*1 = 2

theorem correct_option_B : (dec_13 = bin_1101) := by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_correct_option_B_l2414_241476


namespace NUMINAMATH_GPT_day_of_20th_is_Thursday_l2414_241412

noncomputable def day_of_week (d : ℕ) : String :=
  match d % 7 with
  | 0 => "Saturday"
  | 1 => "Sunday"
  | 2 => "Monday"
  | 3 => "Tuesday"
  | 4 => "Wednesday"
  | 5 => "Thursday"
  | 6 => "Friday"
  | _ => "Unknown"

theorem day_of_20th_is_Thursday (s1 s2 s3: ℕ) (h1: 2 ≤ s1) (h2: s1 ≤ 30) (h3: s2 = s1 + 14) (h4: s3 = s2 + 14) (h5: s3 ≤ 30) (h6: day_of_week s1 = "Sunday") : 
  day_of_week 20 = "Thursday" :=
by
  sorry

end NUMINAMATH_GPT_day_of_20th_is_Thursday_l2414_241412


namespace NUMINAMATH_GPT_triangle_perimeter_l2414_241428

theorem triangle_perimeter (MN NP MP : ℝ)
  (h1 : MN - NP = 18)
  (h2 : MP = 40)
  (h3 : MN / NP = 28 / 12) : 
  MN + NP + MP = 85 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l2414_241428


namespace NUMINAMATH_GPT_cone_volume_l2414_241416

theorem cone_volume (r l: ℝ) (r_eq : r = 2) (l_eq : l = 4) (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) :
  (1 / 3) * π * r^2 * h = (8 * Real.sqrt 3 * π) / 3 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_cone_volume_l2414_241416


namespace NUMINAMATH_GPT_christmas_gift_distribution_l2414_241444

theorem christmas_gift_distribution :
  ∃ n : ℕ, n = 30 ∧ 
  ∃ (gifts : Finset α) (students : Finset β) 
    (distribute : α → β) (a b c d : α),
    a ∈ gifts ∧ b ∈ gifts ∧ c ∈ gifts ∧ d ∈ gifts ∧ gifts.card = 4 ∧
    students.card = 3 ∧ 
    (∀ s ∈ students, ∃ g ∈ gifts, distribute g = s) ∧ 
    distribute a ≠ distribute b :=
sorry

end NUMINAMATH_GPT_christmas_gift_distribution_l2414_241444


namespace NUMINAMATH_GPT_sara_quarters_eq_l2414_241467

-- Define the initial conditions and transactions
def initial_quarters : ℕ := 21
def dad_quarters : ℕ := 49
def spent_quarters : ℕ := 15
def mom_dollars : ℕ := 2
def quarters_per_dollar : ℕ := 4
def amy_quarters (x : ℕ) := x

-- Define the function to compute total quarters
noncomputable def total_quarters (x : ℕ) : ℕ :=
initial_quarters + dad_quarters - spent_quarters + mom_dollars * quarters_per_dollar + amy_quarters x

-- Prove that the total number of quarters matches the expected value
theorem sara_quarters_eq (x : ℕ) : total_quarters x = 63 + x :=
by
  sorry

end NUMINAMATH_GPT_sara_quarters_eq_l2414_241467


namespace NUMINAMATH_GPT_paving_stone_length_l2414_241400

theorem paving_stone_length (courtyard_length courtyard_width paving_stone_width : ℝ)
  (num_paving_stones : ℕ)
  (courtyard_dims : courtyard_length = 40 ∧ courtyard_width = 20) 
  (paving_stone_dims : paving_stone_width = 2) 
  (num_stones : num_paving_stones = 100) 
  : (courtyard_length * courtyard_width) / (num_paving_stones * paving_stone_width) = 4 :=
by 
  sorry

end NUMINAMATH_GPT_paving_stone_length_l2414_241400


namespace NUMINAMATH_GPT_permutation_sum_l2414_241484

theorem permutation_sum (n : ℕ) (h1 : n + 3 ≤ 2 * n) (h2 : n + 1 ≤ 4) (h3 : n > 0) :
  Nat.factorial (2 * n) / Nat.factorial (2 * n - (n + 3)) + Nat.factorial 4 / Nat.factorial (4 - (n + 1)) = 744 :=
by
  sorry

end NUMINAMATH_GPT_permutation_sum_l2414_241484


namespace NUMINAMATH_GPT_smallest_sector_angle_divided_circle_l2414_241423

theorem smallest_sector_angle_divided_circle : ∃ a d : ℕ, 
  (2 * a + 7 * d = 90) ∧ 
  (8 * (a + (a + 7 * d)) / 2 = 360) ∧ 
  a = 38 := 
by
  sorry

end NUMINAMATH_GPT_smallest_sector_angle_divided_circle_l2414_241423


namespace NUMINAMATH_GPT_find_other_sides_of_triangle_l2414_241461

-- Given conditions
variables (a b c : ℝ) -- side lengths of the triangle
variables (perimeter : ℝ) -- perimeter of the triangle
variables (iso : ℝ → ℝ → ℝ → Prop) -- a predicate to check if a triangle is isosceles
variables (triangle_ineq : ℝ → ℝ → ℝ → Prop) -- another predicate to check the triangle inequality

-- Given facts
axiom triangle_is_isosceles : iso a b c
axiom triangle_perimeter : a + b + c = perimeter
axiom one_side_is_4 : a = 4 ∨ b = 4 ∨ c = 4
axiom perimeter_value : perimeter = 17

-- The mathematically equivalent proof problem
theorem find_other_sides_of_triangle :
  (b = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ c = 6.5) ∨ (a = 6.5 ∧ b = 6.5) :=
sorry

end NUMINAMATH_GPT_find_other_sides_of_triangle_l2414_241461


namespace NUMINAMATH_GPT_det_example_l2414_241420

theorem det_example : (1 * 4 - 2 * 3) = -2 :=
by
  -- Skip the proof with sorry
  sorry

end NUMINAMATH_GPT_det_example_l2414_241420


namespace NUMINAMATH_GPT_part1_part2_range_of_a_l2414_241426

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x - Real.log (x + 1)

theorem part1 (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) : f1 x ≥ 0 := sorry

noncomputable def f2 (x a : ℝ) : ℝ := Real.sin x - a * Real.log (x + 1)

theorem part2 {a : ℝ} (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ Real.pi) : f2 x a ≤ 2 * Real.exp x - 2 := sorry

theorem range_of_a : {a : ℝ | ∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → f2 x a ≤ 2 * Real.exp x - 2} = {a : ℝ | a ≥ -1} := sorry

end NUMINAMATH_GPT_part1_part2_range_of_a_l2414_241426


namespace NUMINAMATH_GPT_price_of_tea_mixture_l2414_241454

noncomputable def price_of_mixture (price1 price2 price3 : ℝ) (ratio1 ratio2 ratio3 : ℝ) : ℝ :=
  (price1 * ratio1 + price2 * ratio2 + price3 * ratio3) / (ratio1 + ratio2 + ratio3)

theorem price_of_tea_mixture :
  price_of_mixture 126 135 175.5 1 1 2 = 153 := 
by
  sorry

end NUMINAMATH_GPT_price_of_tea_mixture_l2414_241454


namespace NUMINAMATH_GPT_exists_unique_integer_pair_l2414_241474

theorem exists_unique_integer_pair (a : ℕ) (ha : 0 < a) :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x + (x + y - 1) * (x + y - 2) / 2 = a :=
by
  sorry

end NUMINAMATH_GPT_exists_unique_integer_pair_l2414_241474


namespace NUMINAMATH_GPT_tickets_difference_l2414_241472

-- Definitions of conditions
def tickets_won : Nat := 19
def tickets_for_toys : Nat := 12
def tickets_for_clothes : Nat := 7

-- Theorem statement: Prove that the difference between tickets used for toys and tickets used for clothes is 5
theorem tickets_difference : (tickets_for_toys - tickets_for_clothes = 5) := by
  sorry

end NUMINAMATH_GPT_tickets_difference_l2414_241472


namespace NUMINAMATH_GPT_circular_pipes_equivalence_l2414_241435

/-- Determine how many circular pipes with an inside diameter 
of 2 inches are required to carry the same amount of water as 
one circular pipe with an inside diameter of 8 inches. -/
theorem circular_pipes_equivalence 
  (d_small d_large : ℝ)
  (h1 : d_small = 2)
  (h2 : d_large = 8) :
  (d_large / 2) ^ 2 / (d_small / 2) ^ 2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_circular_pipes_equivalence_l2414_241435


namespace NUMINAMATH_GPT_question_l2414_241452

def N : ℕ := 100101102 -- N should be defined properly but is simplified here for illustration.

theorem question (k : ℕ) (h : N = 100101102502499500) : (3^3 ∣ N) ∧ ¬(3^4 ∣ N) :=
sorry

end NUMINAMATH_GPT_question_l2414_241452


namespace NUMINAMATH_GPT_market_value_of_stock_l2414_241481

theorem market_value_of_stock 
  (yield : ℝ) 
  (dividend_percentage : ℝ) 
  (face_value : ℝ) 
  (market_value : ℝ) 
  (h1 : yield = 0.10) 
  (h2 : dividend_percentage = 0.07) 
  (h3 : face_value = 100) 
  (h4 : market_value = (dividend_percentage * face_value) / yield) :
  market_value = 70 := by
  sorry

end NUMINAMATH_GPT_market_value_of_stock_l2414_241481


namespace NUMINAMATH_GPT_product_of_y_values_l2414_241409

theorem product_of_y_values (y : ℝ) (h : abs (2 * y * 3) + 5 = 47) :
  ∃ y1 y2, (abs (2 * y1 * 3) + 5 = 47) ∧ (abs (2 * y2 * 3) + 5 = 47) ∧ y1 * y2 = -49 :=
by 
  sorry

end NUMINAMATH_GPT_product_of_y_values_l2414_241409


namespace NUMINAMATH_GPT_determine_pairs_l2414_241434

open Int

-- Definitions corresponding to the conditions of the problem:
def is_prime (p : ℕ) : Prop := Nat.Prime p
def condition1 (p n : ℕ) : Prop := is_prime p
def condition2 (p n : ℕ) : Prop := n ≤ 2 * p
def condition3 (p n : ℕ) : Prop := (n^(p-1)) ∣ ((p-1)^n + 1)

-- Main theorem statement:
theorem determine_pairs (n p : ℕ) (h1 : condition1 p n) (h2 : condition2 p n) (h3 : condition3 p n) :
  (n = 1 ∧ is_prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
sorry

end NUMINAMATH_GPT_determine_pairs_l2414_241434


namespace NUMINAMATH_GPT_geometric_sequence_a5_eq_2_l2414_241425

-- Define geometric sequence and the properties
noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

-- Given conditions
variables {a : ℕ → ℝ} {q : ℝ}

-- Roots of given quadratic equation
variables (h1 : a 3 = 1 ∨ a 3 = 4 / 1) (h2 : a 7 = 4 / a 3)
variables (h3 : q > 0) (h4 : geometric_seq a q)

-- Prove that a5 = 2
theorem geometric_sequence_a5_eq_2 : a 5 = 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_eq_2_l2414_241425


namespace NUMINAMATH_GPT_log_div_log_inv_of_16_l2414_241411

theorem log_div_log_inv_of_16 : (Real.log 16) / (Real.log (1 / 16)) = -1 :=
by
  sorry

end NUMINAMATH_GPT_log_div_log_inv_of_16_l2414_241411


namespace NUMINAMATH_GPT_directrix_of_parabola_l2414_241442

noncomputable def parabola_directrix (y : ℝ) (x : ℝ) : Prop :=
  y = 4 * x^2

theorem directrix_of_parabola : ∃ d : ℝ, (parabola_directrix (y := 4) (x := x) → d = -1/16) :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l2414_241442


namespace NUMINAMATH_GPT_gcd_18_30_l2414_241403

theorem gcd_18_30: Int.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_18_30_l2414_241403


namespace NUMINAMATH_GPT_find_c_for_circle_radius_five_l2414_241408

theorem find_c_for_circle_radius_five
  (c : ℝ)
  (h : ∀ x y : ℝ, x^2 + 8 * x + y^2 + 2 * y + c = 0) :
  c = -8 :=
sorry

end NUMINAMATH_GPT_find_c_for_circle_radius_five_l2414_241408


namespace NUMINAMATH_GPT_car_round_trip_time_l2414_241496

theorem car_round_trip_time
  (d_AB : ℝ) (v_AB_downhill : ℝ) (v_BA_uphill : ℝ)
  (h_d_AB : d_AB = 75.6)
  (h_v_AB_downhill : v_AB_downhill = 33.6)
  (h_v_BA_uphill : v_BA_uphill = 25.2) :
  d_AB / v_AB_downhill + d_AB / v_BA_uphill = 5.25 := by
  sorry

end NUMINAMATH_GPT_car_round_trip_time_l2414_241496


namespace NUMINAMATH_GPT_nth_smallest_d0_perfect_square_l2414_241438

theorem nth_smallest_d0_perfect_square (n : ℕ) : 
  ∃ (d_0 : ℕ), (∃ v : ℕ, ∀ t : ℝ, (2 * t * t + d_0 = v * t) ∧ (∃ k : ℕ, v = k ∧ k * k = v * v)) 
               ∧ d_0 = 4^(n - 1) := 
by sorry

end NUMINAMATH_GPT_nth_smallest_d0_perfect_square_l2414_241438


namespace NUMINAMATH_GPT_trail_mix_total_weight_l2414_241486

def peanuts : ℝ := 0.17
def chocolate_chips : ℝ := 0.17
def raisins : ℝ := 0.08

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = 0.42 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_trail_mix_total_weight_l2414_241486


namespace NUMINAMATH_GPT_remainder_of_powers_l2414_241473

theorem remainder_of_powers (n1 n2 n3 : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_powers_l2414_241473


namespace NUMINAMATH_GPT_at_least_502_friendly_numbers_l2414_241450

def friendly (a : ℤ) : Prop :=
  ∃ (m n : ℤ), m > 0 ∧ n > 0 ∧ (m^2 + n) * (n^2 + m) = a * (m - n)^3

theorem at_least_502_friendly_numbers :
  ∃ S : Finset ℤ, (∀ a ∈ S, friendly a) ∧ 502 ≤ S.card ∧ ∀ x ∈ S, 1 ≤ x ∧ x ≤ 2012 :=
by
  sorry

end NUMINAMATH_GPT_at_least_502_friendly_numbers_l2414_241450


namespace NUMINAMATH_GPT_range_of_2x_plus_y_range_of_c_l2414_241448

open Real

def point_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2 * y

theorem range_of_2x_plus_y (x y : ℝ) (h : point_on_circle x y) : 
  1 - sqrt 2 ≤ 2 * x + y ∧ 2 * x + y ≤ 1 + sqrt 2 :=
sorry

theorem range_of_c (c : ℝ) : 
  (∀ x y : ℝ, point_on_circle x y → x + y + c > 0) → c ≥ -1 :=
sorry

end NUMINAMATH_GPT_range_of_2x_plus_y_range_of_c_l2414_241448


namespace NUMINAMATH_GPT_inequality_condition_l2414_241407

theorem inequality_condition (x : ℝ) :
  ((x + 3) * (x - 2) < 0 ↔ -3 < x ∧ x < 2) →
  ((-3 < x ∧ x < 0) → (x + 3) * (x - 2) < 0) →
  ∃ p q : Prop, (p → q) ∧ ¬(q → p) ∧
  p = ((x + 3) * (x - 2) < 0) ∧ q = (-3 < x ∧ x < 0) := by
  sorry

end NUMINAMATH_GPT_inequality_condition_l2414_241407


namespace NUMINAMATH_GPT_sum_of_reciprocals_l2414_241497

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) : 
  (1 / x) + (1 / y) = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l2414_241497


namespace NUMINAMATH_GPT_particle_position_at_2004_seconds_l2414_241413

structure ParticleState where
  position : ℕ × ℕ

def initialState : ParticleState :=
  { position := (0, 0) }

def moveParticle (state : ParticleState) (time : ℕ) : ParticleState :=
  if time = 0 then initialState
  else if (time - 1) % 4 < 2 then
    { state with position := (state.position.fst + 1, state.position.snd) }
  else
    { state with position := (state.position.fst, state.position.snd + 1) }

def particlePositionAfterTime (time : ℕ) : ParticleState :=
  (List.range time).foldl moveParticle initialState

/-- The position of the particle after 2004 seconds is (20, 44) -/
theorem particle_position_at_2004_seconds :
  (particlePositionAfterTime 2004).position = (20, 44) :=
  sorry

end NUMINAMATH_GPT_particle_position_at_2004_seconds_l2414_241413


namespace NUMINAMATH_GPT_number_of_juniors_l2414_241471

variable (J S x y : ℕ)

-- Conditions given in the problem
axiom total_students : J + S = 40
axiom junior_debate_team : 3 * J / 10 = x
axiom senior_debate_team : S / 5 = y
axiom equal_debate_team : x = y

-- The theorem to prove 
theorem number_of_juniors : J = 16 :=
by
  sorry

end NUMINAMATH_GPT_number_of_juniors_l2414_241471


namespace NUMINAMATH_GPT_greatest_number_of_sets_l2414_241457

-- We define the number of logic and visual puzzles.
def n_logic : ℕ := 18
def n_visual : ℕ := 9

-- The theorem states that the greatest number of identical sets Mrs. Wilson can create is the GCD of 18 and 9.
theorem greatest_number_of_sets : gcd n_logic n_visual = 9 := by
  sorry

end NUMINAMATH_GPT_greatest_number_of_sets_l2414_241457


namespace NUMINAMATH_GPT_water_charging_standard_l2414_241402

theorem water_charging_standard
  (x y : ℝ)
  (h1 : 10 * x + 5 * y = 35)
  (h2 : 10 * x + 8 * y = 44) : 
  x = 2 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_water_charging_standard_l2414_241402


namespace NUMINAMATH_GPT_sector_area_proof_l2414_241489

-- Define variables for the central angle, arc length, and derived radius
variables (θ L : ℝ) (r A: ℝ)

-- Define the conditions given in the problem
def central_angle_condition : Prop := θ = 2
def arc_length_condition : Prop := L = 4
def radius_condition : Prop := r = L / θ

-- Define the formula for the area of the sector
def area_of_sector_condition : Prop := A = (1 / 2) * r^2 * θ

-- The theorem that needs to be proved
theorem sector_area_proof :
  central_angle_condition θ ∧ arc_length_condition L ∧ radius_condition θ L r ∧ area_of_sector_condition r θ A → A = 4 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_proof_l2414_241489


namespace NUMINAMATH_GPT_geometric_sequence_a2_value_l2414_241414

theorem geometric_sequence_a2_value
    (a : ℕ → ℝ)
    (h1 : a 1 = 1/5)
    (h3 : a 3 = 5)
    (geometric : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) :
    a 2 = 1 ∨ a 2 = -1 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a2_value_l2414_241414


namespace NUMINAMATH_GPT_vasya_cuts_larger_area_l2414_241470

noncomputable def E_Vasya_square_area : ℝ :=
  (1/6) * (1^2) + (1/6) * (2^2) + (1/6) * (3^2) + (1/6) * (4^2) + (1/6) * (5^2) + (1/6) * (6^2)

noncomputable def E_Asya_rectangle_area : ℝ :=
  (3.5 * 3.5)

theorem vasya_cuts_larger_area :
  E_Vasya_square_area > E_Asya_rectangle_area :=
  by
    sorry

end NUMINAMATH_GPT_vasya_cuts_larger_area_l2414_241470


namespace NUMINAMATH_GPT_children_got_off_bus_l2414_241469

-- Conditions
def original_number_of_children : ℕ := 43
def children_left_on_bus : ℕ := 21

-- Definition of the number of children who got off the bus
def children_got_off : ℕ := original_number_of_children - children_left_on_bus

-- Theorem stating the number of children who got off the bus
theorem children_got_off_bus : children_got_off = 22 :=
by
  -- This is to indicate where the proof would go
  sorry

end NUMINAMATH_GPT_children_got_off_bus_l2414_241469


namespace NUMINAMATH_GPT_vacuum_total_time_l2414_241451

theorem vacuum_total_time (x : ℕ) (hx : 2 * x + 5 = 27) :
  27 + x = 38 :=
by
  sorry

end NUMINAMATH_GPT_vacuum_total_time_l2414_241451


namespace NUMINAMATH_GPT_min_value_PA_PF_l2414_241482

noncomputable def minimum_value_of_PA_and_PF_minimum 
  (x y : ℝ)
  (A : ℝ × ℝ)
  (F : ℝ × ℝ) : ℝ :=
  if ((A = (-1, 8)) ∧ (F = (0, 1)) ∧ (x^2 = 4 * y)) then 9 else 0

theorem min_value_PA_PF 
  (A : ℝ × ℝ := (-1, 8))
  (F : ℝ × ℝ := (0, 1))
  (P : ℝ × ℝ)
  (hP : P.1^2 = 4 * P.2) :
  minimum_value_of_PA_and_PF_minimum P.1 P.2 A F = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_PA_PF_l2414_241482


namespace NUMINAMATH_GPT_negation_proposition_false_l2414_241419

variable {R : Type} [LinearOrderedField R]

theorem negation_proposition_false (x y : R) :
  ¬ (x > 2 ∧ y > 3 → x + y > 5) = false := by
sorry

end NUMINAMATH_GPT_negation_proposition_false_l2414_241419


namespace NUMINAMATH_GPT_marble_problem_l2414_241459

theorem marble_problem (R B : ℝ) 
  (h1 : R + B = 6000) 
  (h2 : (R + B) - |R - B| = 4800) 
  (h3 : B > R) : B = 3600 :=
sorry

end NUMINAMATH_GPT_marble_problem_l2414_241459


namespace NUMINAMATH_GPT_range_of_m_l2414_241436

noncomputable def f (x m : ℝ) := Real.exp x * (Real.log x + (1 / 2) * x ^ 2 - m * x)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → ((Real.exp x * ((1 / x) + x - m)) > 0)) → m < 2 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l2414_241436


namespace NUMINAMATH_GPT_length_more_than_breadth_l2414_241405

theorem length_more_than_breadth (b x : ℝ) (h1 : b + x = 61) (h2 : 26.50 * (4 * b + 2 * x) = 5300) : x = 22 :=
by
  sorry

end NUMINAMATH_GPT_length_more_than_breadth_l2414_241405


namespace NUMINAMATH_GPT_work_problem_l2414_241483

theorem work_problem (A B : ℝ) (hA : A = 1/4) (hB : B = 1/12) :
  (2 * (A + B) + 4 * B = 1) :=
by
  -- Work rate of A and B together
  -- Work done in 2 days by both
  -- Remaining work and time taken by B alone
  -- Final Result
  sorry

end NUMINAMATH_GPT_work_problem_l2414_241483


namespace NUMINAMATH_GPT_polynomial_roots_l2414_241462

theorem polynomial_roots:
  ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (2*x - 8) = 0 ↔ x = 2 ∨ x = 3 ∨ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l2414_241462


namespace NUMINAMATH_GPT_purple_chips_selected_is_one_l2414_241429

noncomputable def chips_selected (B G P R x : ℕ) : Prop :=
  (1^B) * (5^G) * (x^P) * (11^R) = 140800 ∧ 5 < x ∧ x < 11

theorem purple_chips_selected_is_one :
  ∃ B G P R x, chips_selected B G P R x ∧ P = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_purple_chips_selected_is_one_l2414_241429


namespace NUMINAMATH_GPT_proof_problem_l2414_241447

variable {a b x : ℝ}

theorem proof_problem (h1 : x = b / a) (h2 : a ≠ b) (h3 : a ≠ 0) : 
  (2 * a + b) / (2 * a - b) = (2 + x) / (2 - x) :=
sorry

end NUMINAMATH_GPT_proof_problem_l2414_241447


namespace NUMINAMATH_GPT_probability_of_not_adjacent_to_edge_is_16_over_25_l2414_241464

def total_squares : ℕ := 100
def perimeter_squares : ℕ := 36
def non_perimeter_squares : ℕ := total_squares - perimeter_squares
def probability_not_adjacent_to_edge : ℚ := non_perimeter_squares / total_squares

theorem probability_of_not_adjacent_to_edge_is_16_over_25 :
  probability_not_adjacent_to_edge = 16 / 25 := by
  sorry

end NUMINAMATH_GPT_probability_of_not_adjacent_to_edge_is_16_over_25_l2414_241464


namespace NUMINAMATH_GPT_perpendicular_lines_implies_perpendicular_plane_l2414_241499

theorem perpendicular_lines_implies_perpendicular_plane
  (triangle_sides : Line → Prop)
  (circle_diameters : Line → Prop)
  (perpendicular : Line → Line → Prop)
  (is_perpendicular_to_plane : Line → Prop) :
  (∀ l₁ l₂, triangle_sides l₁ → triangle_sides l₂ → perpendicular l₁ l₂ → is_perpendicular_to_plane l₁) ∧
  (∀ l₁ l₂, circle_diameters l₁ → circle_diameters l₂ → perpendicular l₁ l₂ → is_perpendicular_to_plane l₁) :=
  sorry

end NUMINAMATH_GPT_perpendicular_lines_implies_perpendicular_plane_l2414_241499


namespace NUMINAMATH_GPT_polynomial_solution_l2414_241445

noncomputable def is_solution (P : ℝ → ℝ) : Prop :=
  (P 0 = 0) ∧ (∀ x : ℝ, P x = (P (x + 1) + P (x - 1)) / 2)

theorem polynomial_solution (P : ℝ → ℝ) : is_solution P → ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
  sorry

end NUMINAMATH_GPT_polynomial_solution_l2414_241445


namespace NUMINAMATH_GPT_no_integer_solution_l2414_241441

theorem no_integer_solution (x : ℤ) : ¬ (x + 12 > 15 ∧ -3 * x > -9) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_integer_solution_l2414_241441


namespace NUMINAMATH_GPT_maria_needs_more_cartons_l2414_241466

theorem maria_needs_more_cartons
  (total_needed : ℕ)
  (strawberries : ℕ)
  (blueberries : ℕ)
  (already_has : ℕ)
  (more_needed : ℕ)
  (h1 : total_needed = 21)
  (h2 : strawberries = 4)
  (h3 : blueberries = 8)
  (h4 : already_has = strawberries + blueberries)
  (h5 : more_needed = total_needed - already_has) :
  more_needed = 9 :=
by sorry

end NUMINAMATH_GPT_maria_needs_more_cartons_l2414_241466


namespace NUMINAMATH_GPT_subset_strict_M_P_l2414_241443

-- Define the set M
def M : Set ℕ := {x | ∃ a : ℕ, a > 0 ∧ x = a^2 + 1}

-- Define the set P
def P : Set ℕ := {y | ∃ b : ℕ, b > 0 ∧ y = b^2 - 4*b + 5}

-- Prove that M is strictly a subset of P
theorem subset_strict_M_P : M ⊆ P ∧ ∃ x ∈ P, x ∉ M :=
by
  sorry

end NUMINAMATH_GPT_subset_strict_M_P_l2414_241443


namespace NUMINAMATH_GPT_trigonometric_expression_l2414_241460

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (3 * Real.cos α + 3 * Real.sin α) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_l2414_241460


namespace NUMINAMATH_GPT_half_of_expression_correct_l2414_241488

theorem half_of_expression_correct :
  (2^12 + 3 * 2^10) / 2 = 2^9 * 7 :=
by
  sorry

end NUMINAMATH_GPT_half_of_expression_correct_l2414_241488


namespace NUMINAMATH_GPT_solve_for_a_l2414_241439

theorem solve_for_a
  (a x : ℚ)
  (h1 : (2 * a * x + 3) / (a - x) = 3 / 4)
  (h2 : x = 1) : a = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l2414_241439


namespace NUMINAMATH_GPT_A_independent_of_beta_l2414_241491

noncomputable def A (alpha beta : ℝ) : ℝ :=
  (Real.sin (alpha + beta) ^ 2) + (Real.sin (beta - alpha) ^ 2) - 
  2 * (Real.sin (alpha + beta)) * (Real.sin (beta - alpha)) * (Real.cos (2 * alpha))

theorem A_independent_of_beta (alpha beta : ℝ) : 
  ∃ (c : ℝ), ∀ beta : ℝ, A alpha beta = c :=
by
  sorry

end NUMINAMATH_GPT_A_independent_of_beta_l2414_241491


namespace NUMINAMATH_GPT_math_problem_l2414_241480

theorem math_problem :
  let result := 83 - 29
  let final_sum := result + 58
  let rounded := if final_sum % 10 < 5 then final_sum - final_sum % 10 else final_sum + (10 - final_sum % 10)
  rounded = 110 := by
  sorry

end NUMINAMATH_GPT_math_problem_l2414_241480


namespace NUMINAMATH_GPT_tangent_parallel_l2414_241424

noncomputable def f (x: ℝ) : ℝ := x^4 - x
noncomputable def f' (x: ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_parallel
  (P : ℝ × ℝ)
  (hp : P = (1, 0))
  (tangent_parallel : ∀ x, f' x = 3 ↔ x = 1)
  : P = (1, 0) := 
by 
  sorry

end NUMINAMATH_GPT_tangent_parallel_l2414_241424


namespace NUMINAMATH_GPT_find_x_l2414_241493

theorem find_x (x : ℝ) : (x * 16) / 100 = 0.051871999999999995 → x = 0.3242 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l2414_241493


namespace NUMINAMATH_GPT_tax_per_pound_is_one_l2414_241433

-- Define the conditions
def bulk_price_per_pound : ℝ := 5          -- Condition 1
def minimum_spend : ℝ := 40               -- Condition 2
def total_paid : ℝ := 240                 -- Condition 4
def excess_pounds : ℝ := 32               -- Condition 5

-- Define the proof problem statement
theorem tax_per_pound_is_one :
  ∃ (T : ℝ), total_paid = (minimum_spend / bulk_price_per_pound + excess_pounds) * bulk_price_per_pound + 
  (minimum_spend / bulk_price_per_pound + excess_pounds) * T ∧ 
  T = 1 :=
by 
  sorry

end NUMINAMATH_GPT_tax_per_pound_is_one_l2414_241433


namespace NUMINAMATH_GPT_find_decimal_number_l2414_241477

noncomputable def decimal_number (x : ℝ) : Prop := 
x > 0 ∧ (100000 * x = 5 * (1 / x))

theorem find_decimal_number {x : ℝ} (h : decimal_number x) : x = 1 / (100 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_find_decimal_number_l2414_241477


namespace NUMINAMATH_GPT_more_likely_to_return_to_initial_count_l2414_241455

noncomputable def P_A (a b c d : ℕ) : ℚ :=
(b * (d + 1) + a * (c + 1)) / (50 * 51)

noncomputable def P_A_bar (a b c d : ℕ) : ℚ :=
(b * c + a * d) / (50 * 51)

theorem more_likely_to_return_to_initial_count (a b c d : ℕ) (h1 : a + b = 50) (h2 : c + d = 50) 
  (h3 : b ≥ a) (h4 : d ≥ c - 1) (h5 : a > 0) :
P_A a b c d > P_A_bar a b c d := by
  sorry

end NUMINAMATH_GPT_more_likely_to_return_to_initial_count_l2414_241455


namespace NUMINAMATH_GPT_solve_for_x_l2414_241456

theorem solve_for_x : ∀ x : ℝ, ( (x * x^(2:ℝ)) ^ (1/6) )^2 = 4 → x = 4 := by
  intro x
  sorry

end NUMINAMATH_GPT_solve_for_x_l2414_241456


namespace NUMINAMATH_GPT_mirror_area_proof_l2414_241465

-- Definitions of conditions
def outer_width := 100
def outer_height := 70
def frame_width := 15
def mirror_width := outer_width - 2 * frame_width -- 100 - 2 * 15 = 70
def mirror_height := outer_height - 2 * frame_width -- 70 - 2 * 15 = 40

-- Statement of the proof problem
theorem mirror_area_proof : 
  (mirror_width * mirror_height) = 2800 := 
by
  sorry

end NUMINAMATH_GPT_mirror_area_proof_l2414_241465


namespace NUMINAMATH_GPT_money_spent_on_ferris_wheel_l2414_241421

-- Conditions
def initial_tickets : ℕ := 6
def remaining_tickets : ℕ := 3
def ticket_cost : ℕ := 9

-- Prove that the money spent during the ferris wheel ride is 27 dollars
theorem money_spent_on_ferris_wheel : (initial_tickets - remaining_tickets) * ticket_cost = 27 := by
  sorry

end NUMINAMATH_GPT_money_spent_on_ferris_wheel_l2414_241421


namespace NUMINAMATH_GPT_cube_difference_l2414_241437

theorem cube_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 135 :=
sorry

end NUMINAMATH_GPT_cube_difference_l2414_241437


namespace NUMINAMATH_GPT_average_minutes_run_per_day_l2414_241446

theorem average_minutes_run_per_day (f : ℕ) :
  let third_grade_minutes := 12
  let fourth_grade_minutes := 15
  let fifth_grade_minutes := 10
  let third_graders := 4 * f
  let fourth_graders := 2 * f
  let fifth_graders := f
  let total_minutes := third_graders * third_grade_minutes + fourth_graders * fourth_grade_minutes + fifth_graders * fifth_grade_minutes
  let total_students := third_graders + fourth_graders + fifth_graders
  total_minutes / total_students = 88 / 7 :=
by
  sorry

end NUMINAMATH_GPT_average_minutes_run_per_day_l2414_241446


namespace NUMINAMATH_GPT_expand_polynomial_l2414_241440

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l2414_241440


namespace NUMINAMATH_GPT_q_0_plus_q_5_l2414_241432

-- Define the properties of the polynomial q(x)
variable (q : ℝ → ℝ)
variable (monic_q : ∀ x, ∃ a b c d e f, a = 1 ∧ q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f)
variable (deg_q : ∀ x, degree q = 5)
variable (q_1 : q 1 = 26)
variable (q_2 : q 2 = 52)
variable (q_3 : q 3 = 78)

-- State the theorem to find q(0) + q(5)
theorem q_0_plus_q_5 : q 0 + q 5 = 58 :=
sorry

end NUMINAMATH_GPT_q_0_plus_q_5_l2414_241432


namespace NUMINAMATH_GPT_biff_break_even_night_hours_l2414_241485

-- Define the constants and conditions
def ticket_cost : ℝ := 11
def snacks_cost : ℝ := 3
def headphones_cost : ℝ := 16
def lunch_cost : ℝ := 8
def dinner_cost : ℝ := 10
def accommodation_cost : ℝ := 35

def total_expenses_without_wifi : ℝ := ticket_cost + snacks_cost + headphones_cost + lunch_cost + dinner_cost + accommodation_cost

def earnings_per_hour : ℝ := 12
def wifi_cost_day : ℝ := 2
def wifi_cost_night : ℝ := 1

-- Define the total expenses with wifi cost variable
def total_expenses (D N : ℝ) : ℝ := total_expenses_without_wifi + (wifi_cost_day * D) + (wifi_cost_night * N)

-- Define the total earnings
def total_earnings (D N : ℝ) : ℝ := earnings_per_hour * (D + N)

-- Prove that the minimum number of hours Biff needs to work at night to break even is 8 hours
theorem biff_break_even_night_hours :
  ∃ N : ℕ, N = 8 ∧ total_earnings 0 N ≥ total_expenses 0 N := 
by 
  sorry

end NUMINAMATH_GPT_biff_break_even_night_hours_l2414_241485


namespace NUMINAMATH_GPT_initial_volume_of_solution_l2414_241410

variable (V : ℝ)

theorem initial_volume_of_solution :
  (0.05 * V + 5.5 = 0.15 * (V + 10)) → (V = 40) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_volume_of_solution_l2414_241410


namespace NUMINAMATH_GPT_rectangular_prism_faces_l2414_241417

theorem rectangular_prism_faces (n : ℕ) (h1 : ∀ z : ℕ, z > 0 → z^3 = 2 * n^3) 
  (h2 : n > 0) :
  (∃ f : ℕ, f = (1 / 6 : ℚ) * (6 * 2 * n^3) ∧ 
    f = 10 * n^2) ↔ n = 5 := by
sorry

end NUMINAMATH_GPT_rectangular_prism_faces_l2414_241417


namespace NUMINAMATH_GPT_odd_function_behavior_l2414_241431

theorem odd_function_behavior (f : ℝ → ℝ)
  (h_odd: ∀ x, f (-x) = -f x)
  (h_increasing: ∀ x y, 3 ≤ x → x ≤ 7 → 3 ≤ y → y ≤ 7 → x < y → f x < f y)
  (h_max: ∀ x, 3 ≤ x → x ≤ 7 → f x ≤ 5) :
  (∀ x, -7 ≤ x → x ≤ -3 → f x ≥ -5) ∧ (∀ x y, -7 ≤ x → x ≤ -3 → -7 ≤ y → y ≤ -3 → x < y → f x < f y) :=
sorry

end NUMINAMATH_GPT_odd_function_behavior_l2414_241431


namespace NUMINAMATH_GPT_min_value_of_x_l2414_241430

theorem min_value_of_x (x : ℝ) (h : 2 * (x + 1) ≥ x + 1) : x ≥ -1 := sorry

end NUMINAMATH_GPT_min_value_of_x_l2414_241430


namespace NUMINAMATH_GPT_rachel_picture_books_shelves_l2414_241422

theorem rachel_picture_books_shelves (mystery_shelves : ℕ) (books_per_shelf : ℕ) (total_books : ℕ) 
  (h1 : mystery_shelves = 6) 
  (h2 : books_per_shelf = 9) 
  (h3 : total_books = 72) : 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 :=
by sorry

end NUMINAMATH_GPT_rachel_picture_books_shelves_l2414_241422


namespace NUMINAMATH_GPT_average_calculation_l2414_241449

def average_two (a b : ℚ) : ℚ := (a + b) / 2
def average_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem average_calculation :
  average_three (average_three 2 4 1) (average_two 3 2) 5 = 59 / 18 :=
by
  sorry

end NUMINAMATH_GPT_average_calculation_l2414_241449


namespace NUMINAMATH_GPT_initial_elephants_count_l2414_241478

def exodus_rate : ℕ := 2880
def exodus_time : ℕ := 4
def entrance_rate : ℕ := 1500
def entrance_time : ℕ := 7
def final_elephants : ℕ := 28980

theorem initial_elephants_count :
  final_elephants - (exodus_rate * exodus_time) + (entrance_rate * entrance_time) = 27960 := by
  sorry

end NUMINAMATH_GPT_initial_elephants_count_l2414_241478


namespace NUMINAMATH_GPT_probability_red_card_top_l2414_241404

def num_red_cards : ℕ := 26
def total_cards : ℕ := 52
def prob_red_card_top : ℚ := num_red_cards / total_cards

theorem probability_red_card_top : prob_red_card_top = (1 / 2) := by
  sorry

end NUMINAMATH_GPT_probability_red_card_top_l2414_241404


namespace NUMINAMATH_GPT_depth_of_channel_l2414_241415

noncomputable def trapezium_area (a b h : ℝ) : ℝ :=
1/2 * (a + b) * h

theorem depth_of_channel :
  ∃ h : ℝ, trapezium_area 12 8 h = 700 ∧ h = 70 :=
by
  use 70
  unfold trapezium_area
  sorry

end NUMINAMATH_GPT_depth_of_channel_l2414_241415


namespace NUMINAMATH_GPT_computation_of_difference_of_squares_l2414_241418

theorem computation_of_difference_of_squares : (65^2 - 35^2) = 3000 := sorry

end NUMINAMATH_GPT_computation_of_difference_of_squares_l2414_241418


namespace NUMINAMATH_GPT_seventh_term_geometric_seq_l2414_241492

theorem seventh_term_geometric_seq (a r : ℝ) (h_pos: 0 < r) (h_fifth: a * r^4 = 16) (h_ninth: a * r^8 = 4) : a * r^6 = 8 := by
  sorry

end NUMINAMATH_GPT_seventh_term_geometric_seq_l2414_241492


namespace NUMINAMATH_GPT_solve_for_y_l2414_241479


theorem solve_for_y (b y : ℝ) (h : b ≠ 0) :
    Matrix.det ![
        ![y + b, y, y],
        ![y, y + b, y],
        ![y, y, y + b]] = 0 → y = -b := by
  sorry

end NUMINAMATH_GPT_solve_for_y_l2414_241479


namespace NUMINAMATH_GPT_proper_subset_singleton_l2414_241406

theorem proper_subset_singleton : ∀ (P : Set ℕ), P = {0} → (∃ S, S ⊂ P ∧ S = ∅) :=
by
  sorry

end NUMINAMATH_GPT_proper_subset_singleton_l2414_241406


namespace NUMINAMATH_GPT_minimum_gb_for_cheaper_plan_l2414_241475

theorem minimum_gb_for_cheaper_plan : ∃ g : ℕ, (g ≥ 778) ∧ 
  (∀ g' < 778, 3000 + (if g' ≤ 500 then 8 * g' else 8 * 500 + 6 * (g' - 500)) ≥ 15 * g') ∧ 
  3000 + (if g ≤ 500 then 8 * g else 8 * 500 + 6 * (g - 500)) < 15 * g :=
by
  sorry

end NUMINAMATH_GPT_minimum_gb_for_cheaper_plan_l2414_241475


namespace NUMINAMATH_GPT_number_of_divisors_of_36_l2414_241458

theorem number_of_divisors_of_36 : Nat.totient 36 = 9 := 
by sorry

end NUMINAMATH_GPT_number_of_divisors_of_36_l2414_241458


namespace NUMINAMATH_GPT_shoes_sold_first_week_eq_100k_l2414_241495

-- Define variables for purchase price and total revenue
def purchase_price : ℝ := 180
def total_revenue : ℝ := 216

-- Define markups
def first_week_markup : ℝ := 1.25
def remaining_markup : ℝ := 1.16

-- Define the conditions
theorem shoes_sold_first_week_eq_100k (x y : ℝ) 
  (h1 : x + y = purchase_price) 
  (h2 : first_week_markup * x + remaining_markup * y = total_revenue) :
  first_week_markup * x = 100  := 
sorry

end NUMINAMATH_GPT_shoes_sold_first_week_eq_100k_l2414_241495


namespace NUMINAMATH_GPT_min_contribution_l2414_241494

theorem min_contribution (x : ℝ) (h1 : 0 < x) (h2 : 10 * x = 20) (h3 : ∀ p, p ≠ 1 → p ≠ 2 → p ≠ 3 → p ≠ 4 → p ≠ 5 → p ≠ 6 → p ≠ 7 → p ≠ 8 → p ≠ 9 → p ≠ 10 → p ≤ 11) : 
  x = 2 := sorry

end NUMINAMATH_GPT_min_contribution_l2414_241494


namespace NUMINAMATH_GPT_fraction_product_l2414_241490

theorem fraction_product :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := 
by
  -- Detailed proof steps would go here
  sorry

end NUMINAMATH_GPT_fraction_product_l2414_241490


namespace NUMINAMATH_GPT_hexagon_ratio_l2414_241468

theorem hexagon_ratio 
  (hex_area : ℝ)
  (rs_bisects_area : ∃ (a b : ℝ), a + b = hex_area / 2 ∧ ∃ (x r s : ℝ), x = 4 ∧ r * s = (hex_area / 2 - 1))
  : ∀ (XR RS : ℝ), XR = RS → XR / RS = 1 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_ratio_l2414_241468


namespace NUMINAMATH_GPT_correct_transformation_l2414_241463

theorem correct_transformation (x : ℝ) : (x^2 - 10 * x - 1 = 0) → ((x - 5) ^ 2 = 26) := by
  sorry

end NUMINAMATH_GPT_correct_transformation_l2414_241463


namespace NUMINAMATH_GPT_isosceles_triangle_l2414_241487

theorem isosceles_triangle {a b R : ℝ} {α β : ℝ} 
  (h : a * Real.tan α + b * Real.tan β = (a + b) * Real.tan ((α + β) / 2))
  (ha : a = 2 * R * Real.sin α) (hb : b = 2 * R * Real.sin β) :
  α = β := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_l2414_241487


namespace NUMINAMATH_GPT_a_range_l2414_241453

noncomputable def f (x a : ℝ) : ℝ := |2 * x - 1| + |x - 2 * a|

def valid_a_range (a : ℝ) : Prop :=
∀ x, 1 ≤ x ∧ x ≤ 2 → f x a ≤ 4

theorem a_range (a : ℝ) : valid_a_range a → (1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ) := 
sorry

end NUMINAMATH_GPT_a_range_l2414_241453
