import Mathlib

namespace probability_of_perfect_square_l614_614046

theorem probability_of_perfect_square :
  (∑ n in (set_of (λ (n : ℕ) => n ≤ 50 ∧ is_square n)), 1/200) + 
  (∑ n in (set_of (λ (n : ℕ) => n > 50 ∧ is_square n)), 3/200) = 0.08 :=
by
  -- Proof omitted
  sorry

end probability_of_perfect_square_l614_614046


namespace eightieth_percentile_is_94_l614_614565

def scores : List ℕ := [91, 89, 90, 92, 95, 87, 93, 96, 91, 85]

def N : ℕ := scores.length

def percentile_80_position : ℕ := ((80 * N) / 100)

theorem eightieth_percentile_is_94 :
  let sorted_scores := List.sort (· ≤ ·) scores
  2 * sorted_scores.get! (percentile_80_position - 1) = 188 := sorry

end eightieth_percentile_is_94_l614_614565


namespace max_value_l614_614605

noncomputable theory

-- Define the conditions for x, y, and z being nonnegative real numbers
variables {x y z : ℝ}

-- Define the condition that x + 2y + 3z = 1
def constraint (x y z : ℝ) : Prop := x + 2 * y + 3 * z = 1

-- The statement of our theorem
theorem max_value (h : constraint x y z) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (x^2 + y^2 + z^3) ≤ 1 :=
sorry

end max_value_l614_614605


namespace badminton_players_l614_614959

theorem badminton_players (B T N Both Total: ℕ) 
  (h1: Total = 35)
  (h2: T = 18)
  (h3: N = 5)
  (h4: Both = 3)
  : B = 15 :=
by
  -- The proof block is intentionally left out.
  sorry

end badminton_players_l614_614959


namespace triangle_base_and_height_l614_614646

theorem triangle_base_and_height (h b : ℕ) (A : ℕ) (hb : b = h - 4) (hA : A = 96) 
  (hArea : A = (1 / 2) * b * h) : (b = 12 ∧ h = 16) :=
by
  sorry

end triangle_base_and_height_l614_614646


namespace conformal_2z_conformal_z_minus_2_squared_l614_614391

-- For the function w = 2z
theorem conformal_2z :
  ∀ z : ℂ, true :=
by
  intro z
  sorry

-- For the function w = (z-2)^2
theorem conformal_z_minus_2_squared :
  ∀ z : ℂ, z ≠ 2 → true :=
by
  intro z h
  sorry

end conformal_2z_conformal_z_minus_2_squared_l614_614391


namespace part_a_part_b_l614_614689

open Classical 

-- Define vertices and edges
inductive Vertex
| A | B | C | D | E | F | G | H | K | L
deriving DecidableEq

open Vertex

-- Define edges for first figure
def edges1 : List (Vertex × Vertex) := [
  (A, B), (B, H), (H, F), (F, G), (G, K), (K, L), (L, C), (C, H), (H, E), (E, L), (L, D), (D, A)
]

-- Define degree calculation
def degree (v : Vertex) (edges : List (Vertex × Vertex)) : Nat :=
  edges.countp (λ e => e.fst = v) + edges.countp (λ e => e.snd = v)

-- Define Eulerian path condition
def has_eulerian_path (edges : List (Vertex × Vertex)) : Prop :=
  (edges.foldr (λ (e : Vertex × Vertex) acc => if degree e.fst edges % 2 = 1 then acc + 1 else acc) 0 <= 2) ∧
  (edges.foldr (λ (e : Vertex × Vertex) acc => if degree e.snd edges % 2 = 1 then acc + 1 else acc) 0 <= 2)

-- Part (a):
theorem part_a : has_eulerian_path edges1 := by
  sorry

-- Define edges for second figure
def edges2 : List (Vertex × Vertex) := [
  (A, B), (B, C), (C, D), (D, A), (A, E), (E, F), (F, C)
]

-- Part (b):
theorem part_b : ¬has_eulerian_path edges2 := by
  sorry

end part_a_part_b_l614_614689


namespace compare_abc_l614_614995

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614995


namespace number_of_irrationals_l614_614438

theorem number_of_irrationals : 
  let a := 15
  let b := 22 / 7
  let c := 3 * Real.sqrt 2
  let d := -3 * Real.pi
  let e := 0.10101
  (2 = [c, d].filter (λ x, ¬ Rational.isRat x)).length :=
by
  sorry

end number_of_irrationals_l614_614438


namespace scientist_born_on_tuesday_l614_614644

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def total_leap_years (start_year end_year : ℕ) : ℕ :=
  (list.range (end_year - start_year + 1)).countp (λ y => is_leap_year (start_year + y))

def day_of_week (ref_day_of_week : ℕ) (days_difference : ℤ) : ℕ :=
  (ref_day_of_week - days_difference.to_nat % 7 + 7) % 7

def day_of_week_scientist_born (anniversary_year anniversary_day_of_week : ℕ) : ℕ :=
  let years_difference := anniversary_year - 2000
  let leap_years := total_leap_years 2000 anniversary_year
  let regular_years := years_difference - leap_years
  let total_days_difference := regular_years * 1 + leap_years * 2
  day_of_week anniversary_day_of_week total_days_difference

theorem scientist_born_on_tuesday : day_of_week_scientist_born 2300 4 = 2 :=
  sorry

end scientist_born_on_tuesday_l614_614644


namespace range_of_a_l614_614878

def f (x: ℝ) : ℝ := -2 * x + 4

def S (n: ℕ) [ne_zero n] : ℝ :=
  (∑ i in Finset.range n, f i.succ / n) + f 1

theorem range_of_a (a : ℝ) (h : ∀ (n : ℕ) [ne_zero n], (a^n / S n) < (a^(n+1) / S (n+1))) : 
  ∃ b, b = (5/2) ∧ a ∈ Ioi b :=
sorry

end range_of_a_l614_614878


namespace find_x_l614_614929

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 :=
by
  have h1 : 32 = 2^5 := by norm_num
  rw [h1, pow_mul] at h
  have h2 : 2^(10) = 2^(5*x) := by exact h
  have h3 : 10 = 5 * x := by exact (pow_inj h2).2
  linarith

end find_x_l614_614929


namespace samantha_marble_choices_l614_614297

open BigOperators

noncomputable def choose_five_with_at_least_one_red (total_marbles red_marbles marbles_needed : ℕ) : ℕ :=
choose total_marbles marbles_needed - choose (total_marbles - red_marbles) marbles_needed

theorem samantha_marble_choices :
  choose_five_with_at_least_one_red 10 1 5 = 126 :=
by
  sorry

end samantha_marble_choices_l614_614297


namespace lines_intersect_l614_614745

def line1 (t : ℝ) : ℝ × ℝ :=
  (1 - 2 * t, 2 + 4 * t)

def line2 (u : ℝ) : ℝ × ℝ :=
  (3 + u, 5 + 3 * u)

theorem lines_intersect :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (1.2, 1.6) :=
by
  sorry

end lines_intersect_l614_614745


namespace equilateral_triangle_cd_value_l614_614320

theorem equilateral_triangle_cd_value 
  (c d : ℝ)
  (h1 : (0:ℝ,0:ℝ) ≠ (c,17))
  (h2 : (0:ℝ,0:ℝ) ≠ (d,43))
  (h3 : (c, 17) ≠ (d, 43))
  (h4 : dist (0:ℝ, 0:ℝ) (c, 17) = dist (0:ℝ, 0:ℝ) (d, 43))
  (h5 : dist (0:ℝ, 0:ℝ) (c, 17) = dist (c, 17) (d, 43)) :
  c * d = -892.67 :=
by { sorry }

end equilateral_triangle_cd_value_l614_614320


namespace number_classification_l614_614654

def numbers : List ℚ := [7, -3.14, -7/2, 0, 4/3, -2, -2/5, 3/20]

def is_integer (n : ℚ) : Prop := ∃ (k : ℤ), n = k

def is_negative_fraction (n : ℚ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ n = a / b ∧ a < 0

theorem number_classification :
  (∃ (int_count neg_frac_count : ℕ), int_count = 3 ∧ neg_frac_count = 3 ∧
  (int_count = (numbers.filter is_integer).length) ∧
  (neg_frac_count = (numbers.filter is_negative_fraction).length)) :=
sorry

end number_classification_l614_614654


namespace range_of_f_on_interval_l614_614861

theorem range_of_f_on_interval (k : ℝ) (h : k > 0) : 
  set.range (λ x : ℝ, if x ≤ -1 then (|(-x)|^k) else 0) = set.Ici 1 :=
sorry

end range_of_f_on_interval_l614_614861


namespace lecture_schedules_l614_614215

theorem lecture_schedules : 
  ∃ (n : ℕ), 
    n = 120 ∧
    (∃ m_pos : {i // i < 3}, -- The Math lecture is among the first three
      ∀ a_pos c_pos : {i // i < 6}, -- Positions of Archery and Charioteering
        (abs (a_pos.val - c_pos.val) = 1) ∧ -- Archery and Charioteering are adjacent
          (6.choose 1 * (4.choose 2) * factorial 3 = 120)) := -- Calculate the total number of lectures
by sorry

end lecture_schedules_l614_614215


namespace prime_condition_l614_614795

theorem prime_condition {p x : ℕ} (h : (x^2010 + x^2009 + ... + 1) % p^2011 = p^2010 % p^2011) : 
  p % 2011 = 1 := sorry

end prime_condition_l614_614795


namespace trains_clear_time_l614_614350

theorem trains_clear_time
    (length1 length2 : ℕ)
    (speed1_kmph speed2_kmph : ℕ)
    (speed1 := speed1_kmph * 1000 / 3600 : ℕ) -- convert to m/s
    (speed2 := speed2_kmph * 1000 / 3600 : ℕ) -- convert to m/s
    (total_length := length1 + length2 : ℕ)
    (relative_speed := speed1 + speed2 : ℕ) :
    length1 = 120 → length2 = 280 →
    speed1_kmph = 42 → speed2_kmph = 30 →
    total_length / relative_speed = 20 :=
by
  intros
  sorry

end trains_clear_time_l614_614350


namespace walking_speed_of_A_l614_614065

-- Given conditions
def B_speed := 20 -- kmph
def start_delay := 10 -- hours
def distance_covered := 200 -- km

-- Prove A's walking speed
theorem walking_speed_of_A (v : ℝ) (time_A : ℝ) (time_B : ℝ) :
  distance_covered = v * time_A ∧ distance_covered = B_speed * time_B ∧ time_B = time_A - start_delay → v = 10 :=
by
  intro h
  sorry

end walking_speed_of_A_l614_614065


namespace incorrect_basis_step_three_l614_614326

theorem incorrect_basis_step_three :
  ¬ (∀ (x : ℝ), 2 * (x + 3) = 5 * x → 2 * x + 6 = 5 * x →
  (2 * x + 6) - 5 * x = -6 → -3 * x = -6 → x = 2 →
  (associative_addition  (2 * x - 5 * x) (-6) (-3 * x))) :=
sorry

end incorrect_basis_step_three_l614_614326


namespace find_original_number_l614_614742

theorem find_original_number (x : ℕ) :
  (43 * x - 34 * x = 1251) → x = 139 :=
by
  sorry

end find_original_number_l614_614742


namespace intersection_nonempty_condition_l614_614881

theorem intersection_nonempty_condition (m n : ℝ) :
  (∃ x : ℝ, (m - 1 < x ∧ x < m + 1) ∧ (3 - n < x ∧ x < 4 - n)) ↔ (2 < m + n ∧ m + n < 5) := 
by
  sorry

end intersection_nonempty_condition_l614_614881


namespace infinite_sum_b_l614_614424

def b : ℕ → ℝ 
| 0     := 0
| 1     := 2
| 2     := 3
| (n + 3) := (1 / 2) * b (n + 2) + (1 / 3) * b (n + 1)

theorem infinite_sum_b : (∑ n, b n) = 24 := by
  sorry

end infinite_sum_b_l614_614424


namespace walnut_price_l614_614687

theorem walnut_price {total_weight total_value walnut_price hazelnut_price : ℕ} 
  (h1 : total_weight = 55)
  (h2 : total_value = 1978)
  (h3 : walnut_price > hazelnut_price)
  (h4 : ∃ a b : ℕ, walnut_price = 10 * a + b ∧ hazelnut_price = 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9)
  (h5 : ∃ a b : ℕ, walnut_price = 10 * a + b ∧ b = a - 1) : 
  walnut_price = 43 := 
sorry

end walnut_price_l614_614687


namespace find_cost_price_l614_614746

theorem find_cost_price (C : ℝ)
  (h1 : ∀ C : ℝ, C > 0) -- Ensure C is positive
  (cond1 : 1.05 * C)
  (cond2 : 0.95 * C)
  (cond3 : 1.05 * C - 8)
  (cond4 : 1.045 * C) :
  C = 1600 :=
by 
  sorry

end find_cost_price_l614_614746


namespace program_count_l614_614428

noncomputable def choose_programs : ℕ :=
  let courses := {'English, 'Algebra, 'Geometry, 'History, 'Art, 'Latin, 'Biology}.to_finset in
  let math_courses := {'Algebra, 'Geometry}.to_finset in
  let english := 'English in
  let non_english_courses := courses.erase english in
  let total_choices := non_english_courses.card.choose 4 in
  let invalid_choices :=
    (finset.singleton (non_english_courses).choose 4).card + -- No math courses
    (math_courses.card.choose 1 * (non_english_courses.card - 1).choose 3) -- Exactly one math course
  in
  total_choices - invalid_choices

theorem program_count : choose_programs = 6 :=
  by
    -- Calculation details hidden
    rw [choose_programs] 
    sorry

end program_count_l614_614428


namespace five_digit_arithmetic_numbers_count_l614_614767

-- Define the problem conditions
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def without_repetition (digits : List ℕ) : Prop :=
  (digits.nodup) ∧ (∀ d ∈ digits, is_digit d)

def arithmetic_sequence (a b c : ℕ) : Prop := 2 * b = a + c

-- Define the main statement
theorem five_digit_arithmetic_numbers_count :
  ∃ (count : ℕ), count = 744 ∧
  (count = List.length { numbers : List ℕ | 
    numbers.length = 5 ∧ 
    without_repetition numbers ∧ 
    arithmetic_sequence numbers.get? 1 numbers.get? 2 numbers.get? 3 }) :=
sorry

end five_digit_arithmetic_numbers_count_l614_614767


namespace cat_meow_ratio_l614_614337

theorem cat_meow_ratio (
  (meow_rate_cat1 : ℕ) (meow_rate_cat1 = 3)
  (meow_rate_cat2 : ℕ) (meow_rate_cat2 = 6)
  (combined_meows_5min : ℕ) (combined_meows_5min = 55)
) : ∃ (meow_rate_cat3 : ℕ), (meow_rate_cat1 * 5 + meow_rate_cat2 * 5 + meow_rate_cat3 * 5 = combined_meows_5min) ∧ ((meow_rate_cat3 * 1) / (meow_rate_cat2 * 1) = 1 / 3) :=
begin
  sorry
end

end cat_meow_ratio_l614_614337


namespace number_of_polynomials_satisfying_constraint_l614_614785

-- Define the conditions of the problem
def P (x : ℤ) (e a b c d : ℤ) : ℤ :=
  e * x^4 + a * x^3 + b * x^2 + c * x + d

def valid_coefficients (n : ℤ) : Prop :=
  0 ≤ n ∧ n ≤ 9

-- Main theorem statement
theorem number_of_polynomials_satisfying_constraint : 
  let count := (∑ e, ∑ a, ∑ b, ∑ c, ∑ d, 
    if (valid_coefficients e ∧ valid_coefficients a ∧ valid_coefficients b ∧ valid_coefficients c ∧ valid_coefficients d
        ∧ P (-2) e a b c d = -16) then 1 else 0) in
  count = 4845 :=
begin
  sorry,
end

end number_of_polynomials_satisfying_constraint_l614_614785


namespace vector_angle_range_l614_614892

-- Define vectors a and b and the angle theta between them
variables (a b : ℝ^3) (θ : ℝ)

-- Conditions given in the problem
axiom condition1 : ∥a + b∥ = 2 * real.sqrt 3
axiom condition2 : ∥a - b∥ = 2

-- The goal is to show that the angle θ between vectors a and b lies within the given range
theorem vector_angle_range : θ ∈ set.Icc 0 (real.pi / 3) :=
sorry

end vector_angle_range_l614_614892


namespace relationship_among_a_b_c_l614_614264

noncomputable def a : ℝ := Real.log 4 / Real.log 5
noncomputable def b : ℝ := (Real.log 3 / Real.log 5)^2
noncomputable def c : ℝ := Real.log 5 / Real.log 4

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l614_614264


namespace min_value_of_fraction_l614_614544

theorem min_value_of_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 3 * y = 3) : 
  (3 / x + 2 / y) = 8 :=
sorry

end min_value_of_fraction_l614_614544


namespace min_value_f_l614_614517

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (4 / (1 - 2 * x))

theorem min_value_f : ∃ (x : ℝ), (0 < x ∧ x < 1 / 2) ∧ f x = 6 + 4 * sqrt 2 := by
  sorry

end min_value_f_l614_614517


namespace find_x_eq_2_l614_614922

theorem find_x_eq_2 : ∀ x : ℝ, 2^10 = 32^x → x = 2 := 
by 
  intros x h
  sorry

end find_x_eq_2_l614_614922


namespace g_is_even_l614_614248

def g (x : ℝ) : ℝ := 5 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g(-x) = g(x) := by
  sorry

end g_is_even_l614_614248


namespace length_A_l614_614595

-- Define points A, B, and C
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define A' and B', points on the line y = x
def A' : ℝ × ℝ := (7.5, 7.5)
def B' : ℝ × ℝ := (5, 5)

-- Define the length of the segment between two points as a function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem to prove
theorem length_A'B' : distance A' B' = 5 * Real.sqrt 2 :=
by
  -- Sorry, skipping the proof
  sorry

end length_A_l614_614595


namespace distance_from_center_to_line_l614_614621

theorem distance_from_center_to_line (r : ℚ) (k : ℤ) (B A : ℚ × ℚ)  
    (H_radius : r = 1 / 14) 
    (H1 : OA^2 - OB^2 > 199 * 1/7) 
    (H2 : (x + 1)^2 - x^2 > 199 / 7) 
    (H3 : OO'^2 < (100 - 1/14)^2 - (96/7)^2) 
    (H4 : 100^2 - (2 * 100 * 1/14) + (1/14)^2 ≈ 10000 - 200/14) 
    (H5 : OO'^2 < 99^2) 
    : OO' + 1 < 100 := 
by
    sorry

end distance_from_center_to_line_l614_614621


namespace no_such_n_l614_614099

theorem no_such_n (n : ℕ) (h_pos : 0 < n) :
  ¬ ∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
sorry

end no_such_n_l614_614099


namespace travel_with_one_stopover_l614_614222

def City := Type
def RedRocket : City → City → Prop
def BlueBoeing : City → City → Prop

variable (Beanville Mieliestad : City)
variable (all_cities : Set City)
variable [decidable_rel RedRocket]
variable [decidable_rel BlueBoeing]

axiom complete_travel (c1 c2 : City) : all_cities c1 → all_cities c2 → 
  (∃ route1 route2, route1 ⊆ all_cities ∧ route2 ⊆ all_cities ∧ 
  (RedRocket c1 c2 ∨ (∃ (mid_city : City), (RedRocket c1 mid_city ∧ RedRocket mid_city c2))
  ∨ BlueBoeing c1 c2 ∨ (∃ (mid_city : City), (BlueBoeing c1 mid_city ∧ BlueBoeing mid_city c2))))

axiom no_red_path_beanville_mieliestad :
  (∀ (mid_city : City), 
   ¬(RedRocket Beanville Mieliestad) ∧ 
   ¬(RedRocket Beanville mid_city ∧ RedRocket mid_city Mieliestad))

theorem travel_with_one_stopover (c1 c2 : City) :
  all_cities c1 → all_cities c2 → 
  (∃ (mid_city : City), BlueBoeing c1 c2 ∨ ((¬(c1 = c2)) ∧ BlueBoeing c1 mid_city ∧ BlueBoeing mid_city c2)) :=
by
  sorry

end travel_with_one_stopover_l614_614222


namespace hexagon_area_inscribed_in_circle_l614_614418

noncomputable def hexagon_area (r : ℝ) : ℝ :=
  let s := r
  6 * (s^2 * real.sqrt 3 / 4)

theorem hexagon_area_inscribed_in_circle :
  hexagon_area 3 = 27 * real.sqrt 3 / 2 := 
  by 
  -- The proof should be written here
  sorry

end hexagon_area_inscribed_in_circle_l614_614418


namespace modulus_z_is_sqrt_two_l614_614862

-- Defining the complex number z = 1 + i
def z : ℂ := 1 + complex.I

-- The modulus of the complex number z
theorem modulus_z_is_sqrt_two : complex.abs z = real.sqrt 2 := by
    sorry

end modulus_z_is_sqrt_two_l614_614862


namespace loss_percent_l614_614383

theorem loss_percent (CP SP Loss : ℝ) (h1 : CP = 600) (h2 : SP = 450) (h3 : Loss = CP - SP) : (Loss / CP) * 100 = 25 :=
by
  sorry

end loss_percent_l614_614383


namespace cos_sum_proof_l614_614859

theorem cos_sum_proof (x : ℝ) (h : cos (x - π / 6) = -√3 / 3) : 
  cos x + cos (x - π / 3) = -1 := 
by 
  sorry

end cos_sum_proof_l614_614859


namespace adult_ticket_cost_l614_614340

theorem adult_ticket_cost 
  (child_ticket_cost : ℕ)
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (adults_attended : ℕ)
  (children_tickets : ℕ)
  (adults_ticket_cost : ℕ)
  (h1 : child_ticket_cost = 6)
  (h2 : total_tickets = 225)
  (h3 : total_cost = 1875)
  (h4 : adults_attended = 175)
  (h5 : children_tickets = total_tickets - adults_attended)
  (h6 : total_cost = adults_attended * adults_ticket_cost + children_tickets * child_ticket_cost) :
  adults_ticket_cost = 9 :=
sorry

end adult_ticket_cost_l614_614340


namespace cube_surface_area_l614_614202

theorem cube_surface_area (V : ℝ) (h : V = 729) : ∃ SA, SA = 486 :=
by
  let s := ∛V  -- Calculate the side length from the volume
  have s_cube_vol : s^3 = V := sorry -- Since s is the real cube root of V, s^3 = V.
  have s_value : s = 9 := by
    rw [←s_cube_vol, h]
    exact real.eq_of_pow_eq_pow (by norm_num) (by norm_num)   -- The cube root of 729 is 9

  let SA := 6 * s^2  -- Calculate the surface area from the side length
  have SA_value : SA = 486 := by
    rw [s_value]
    norm_num

  exact ⟨SA, SA_value⟩

end cube_surface_area_l614_614202


namespace right_triangle_side_length_l614_614570

theorem right_triangle_side_length 
  (a c : ℕ) (h1 : c = 10) (h2 : a = 6) : ∃ b : ℕ, c^2 = a^2 + b^2 ∧ b = 8 :=
by
  use 8
  split
  · rw [h1, h2]
    norm_num
  · simp
  sorry

end right_triangle_side_length_l614_614570


namespace max_intersecting_pairs_l614_614259

open Set

-- Define the conditions of the problem
variable {P : ℕ → ℝ × ℝ} -- Points P_1, P_2, ..., P_4n as a function from natural numbers to pairs of reals.
variable (n : ℕ) (h1 : n > 0) -- n is a positive integer.
variable (h2 : ∀ i j k : ℕ, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ 4*n → ¬Collinear ℝ {P i, P j, P k}) -- No three points are collinear.

-- Pivot and rotation conditions
variable (h3 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 4*n → rotate_90deg_clockwise (P (i-1)) (P i) = P (i+1)) 
variable (h4 : P 0 = P (4*n)) (h5 : P (4*n + 1) = P 1)
variable (h6 : ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 4*n → Intersects (P i) (P (i + 1)) (P j) (P (j + 1)) ≠ Set.Empty)

-- Description of the proof problem
theorem max_intersecting_pairs : ∃ k, k = (8 * n ^ 2 - 6 * n) := 
begin
  sorry
end

end max_intersecting_pairs_l614_614259


namespace projectile_height_reach_l614_614310

theorem projectile_height_reach (t : ℝ) (h : -16 * t^2 + 64 * t = 25) : t = 3.6 :=
by
  sorry

end projectile_height_reach_l614_614310


namespace equilateral_triangle_A1_O_B1_l614_614963

noncomputable def isosceles_triangle (A B C : Point) : Prop :=
  dist A C = dist B C

noncomputable def angle_at_vertex (A B C : Point) (angle_deg : ℝ) : Prop :=
  ∠ BCA = angle_deg

noncomputable def circumcenter (A B C O : Point) : Prop :=
  ∀ (P : Point), dist O P = dist O A ↔ dist O P = dist O B ↔ dist O P = dist O C

noncomputable def angle_bisector_intersect (A B C A1 B1 : Point) : Prop :=
  ∃ I : Point, 
    Line.through A I = Line.angle_bisector ⟨A, B, C⟩ ∧
    Line.through B I = Line.angle_bisector ⟨B, A, C⟩ ∧
  Line.intersect (Line.through A C) (Line.angle_bisector ⟨A, B, C⟩) = A1 ∧
  Line.intersect (Line.through B C) (Line.angle_bisector ⟨B, A, C⟩) = B1

theorem equilateral_triangle_A1_O_B1 
  (A B C O A1 B1 : Point)
  (h1 : isosceles_triangle A B C)
  (h2 : angle_at_vertex A B C 20)
  (h3 : angle_bisector_intersect A B C A1 B1)
  (h4 : circumcenter A B C O) :
  equilateral ⟨A1, O, B1⟩ :=
sorry

end equilateral_triangle_A1_O_B1_l614_614963


namespace largest_area_l614_614966

def point := (ℝ × ℝ)

def slope (p1 p2 : point) : ℝ :=
(p2.2 - p1.2) / (p2.1 - p1.1)

def rotating_line (s : ℝ) (p : point) : set point :=
{x : point | slope p x = s}

def dist (p1 p2 : point) : ℝ :=
real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def quadrilateral_area (A B C D : point) : ℝ :=
let X := rotating_line 1 A ∩ rotating_line (-1) C,
    Y := rotating_line 1 A ∩ rotating_line 2 D,
    Z := rotating_line 0 B ∩ rotating_line (-1) C,
    W := rotating_line (-1) C ∩ rotating_line 2 D in
dist A Z * dist B C

theorem largest_area : quadrilateral_area (0, 0) (8, 0) (15, 0) (20, 0) = 110.5 := 
sorry

end largest_area_l614_614966


namespace g_monotonically_increasing_l614_614869

-- Problem setup definitions
def C1 (x : ℝ) : ℝ := sin (x - π/6)
def g (x : ℝ) : ℝ := -sin (x - π/6)
def domain : Set ℝ := Set.Icc (-π) 0 

-- To prove that g(x) is monotonically increasing in the given interval
theorem g_monotonically_increasing :
  ∃ I : Set ℝ, 
    I = Set.Icc (-2 * π / 3) (-π / 6) ∧ 
    ∀ x y ∈ I, x < y → g x < g y :=
sorry

end g_monotonically_increasing_l614_614869


namespace sum_digits_l614_614580

def distinct_digits (a b c d : ℕ) : Prop :=
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d)

def valid_equation (Y E M T : ℕ) : Prop :=
  ∃ (YE ME TTT : ℕ),
    YE = Y * 10 + E ∧
    ME = M * 10 + E ∧
    TTT = T * 111 ∧
    YE < ME ∧
    YE * ME = TTT ∧
    distinct_digits Y E M T

theorem sum_digits (Y E M T : ℕ) :
  valid_equation Y E M T → Y + E + M + T = 21 := 
sorry

end sum_digits_l614_614580


namespace range_of_sum_of_squares_l614_614601

theorem range_of_sum_of_squares
  (f : ℝ → ℝ)
  (h_increasing : ∀ x y, x ≤ y → f(x) ≤ f(y))
  (h_symmetric : ∀ x, f(-x) + f(x) = 0)
  (m n : ℝ)
  (h_inequality : f(m^2 - 6 * m + 21) + f(n^2 - 8 * n) < 0) :
  9 < m^2 + n^2 ∧ m^2 + n^2 < 49 :=
by
  sorry

end range_of_sum_of_squares_l614_614601


namespace hcf_of_two_numbers_l614_614557

-- Definitions based on conditions
def LCM (x y : ℕ) : ℕ := sorry  -- Assume some definition of LCM
def HCF (x y : ℕ) : ℕ := sorry  -- Assume some definition of HCF

-- Given conditions
axiom cond1 (x y : ℕ) : LCM x y = 600
axiom cond2 (x y : ℕ) : x * y = 18000

-- Statement to prove
theorem hcf_of_two_numbers (x y : ℕ) (h1 : LCM x y = 600) (h2 : x * y = 18000) : HCF x y = 30 :=
by {
  -- Proof omitted, hence we use sorry
  sorry
}

end hcf_of_two_numbers_l614_614557


namespace gcd_factorial_example_l614_614458

theorem gcd_factorial_example : 
  let A := 7!
  let B := 9! / 4!
  Nat.gcd A B = 2520 := 
by 
  let A := 5040
  let B := 362880 / 24
  have h : A = 7! := rfl
  have h2 : B = 9! / 4! := rfl
  let factorA := 2^4 * 3^2 * 5 * 7
  let factorB := 2^4 * 3^3 * 5 * 7
  have h3 : A = factorA := by
    calc 7! = 5040 : rfl
        ... = 2^4 * 3^2 * 5 * 7 : by norm_num
  have h4 : B = factorB := by
    calc 9! / 4! = 362880 / 24 : rfl
              ... = 2^4 * 3^3 * 5 * 7 : by norm_num
  have h5 : Nat.gcd factorA factorB = 2520 := by
    calc Nat.gcd (2^4 * 3^2 * 5 * 7) (2^4 * 3^3 * 5 * 7)
          = 2^4 * 3^2 * 5 * 7 : by norm_num
  have h6 : Nat.gcd A B = 2520 := by
    rw [h3, h4] at h5
    exact h5
  exact h6
  sorry

end gcd_factorial_example_l614_614458


namespace corresponding_angles_equal_l614_614590

theorem corresponding_angles_equal (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α = 90) :
  (180 - α = 90 ∧ β + γ = 90 ∧ α = 90) :=
by
  sorry

end corresponding_angles_equal_l614_614590


namespace bells_toll_together_l614_614018

def toll_intervals := [5, 8, 11, 15]

def prime_factors : Nat → List Nat
| 1 => []
| n => let p := (List.range (n + 1)).drop 2 |>.find (λ d => n % d = 0) |>.getD n
       (p :: prime_factors (n / p))

def lcm (a b : Nat) : Nat := a * b / Nat.gcd a b

def lcm_list (lst : List Nat) : Nat :=
List.foldr lcm 1 lst

theorem bells_toll_together : lcm_list toll_intervals = 1320 := by
  sorry

end bells_toll_together_l614_614018


namespace square_perimeter_l614_614713

theorem square_perimeter
  (length_rect : ℝ) (width_rect : ℝ) (area_square : ℝ)
  (h1 : length_rect = 50) (h2 : width_rect = 10)
  (h3 : area_square = 5 * (length_rect * width_rect)) :
  (4 * real.sqrt area_square) = 200 :=
by
  sorry

end square_perimeter_l614_614713


namespace remainder_cd_l614_614604

variable {m : ℕ} (m_pos : 0 < m)
variable {c d : ℤ} (c_invertible : c.gcd m = 1) (d_invertible : d.gcd m = 1)
variable (h : d ≡ 2 * c⁻¹ [ZMOD m])

theorem remainder_cd (m_pos : 0 < m) (c_invertible : c.gcd m = 1) (d_invertible : d.gcd m = 1) (h : d ≡ 2 * c⁻¹ [ZMOD m]) :
  (c * d) % m = 2 % m := by
  sorry

end remainder_cd_l614_614604


namespace extreme_value_f_prime_log_sum_g_zeros_l614_614169

noncomputable def f (a x : ℝ) : ℝ := a * x * (Real.log x) - x^2 - 2 * x

noncomputable def f_prime (a x : ℝ) : ℝ := a * (1 + Real.log x) - 2 * x - 2

noncomputable def g (a x : ℝ) : ℝ := f(a, x) + 2 * x

theorem extreme_value_f_prime (a : ℝ) (h : a = 4) : 
  ∃ x, f_prime a x = 4 * Real.log 2 - 2 := by 
sorry

theorem log_sum_g_zeros (a x₁ x₂ : ℝ) (h1 : g (a, x₁) = 0) (h2 : g (a, x₂) = 0) (h3 : a > 0) (h4 : (x₂ / x₁) > Real.exp 1) : 
  Real.log a + Real.log (x₁ * x₂) > 3 := by 
sorry

end extreme_value_f_prime_log_sum_g_zeros_l614_614169


namespace sum_of_distances_l614_614107

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

theorem sum_of_distances : 
  let P := (5, 1) in
  let A := (0, 0) in
  let B := (8, 0) in
  let C := (4, 6) in
  distance P A + distance P B + distance P C = 2 * Real.sqrt 26 + Real.sqrt 10 :=
by
  sorry

end sum_of_distances_l614_614107


namespace locus_of_points_l614_614024

-- Define points A and B
variable {A B : (ℝ × ℝ)}
-- Define constant d
variable {d : ℝ}

-- Definition of the distances
def distance_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem locus_of_points (A B : (ℝ × ℝ)) (d : ℝ) :
  ∀ M : (ℝ × ℝ), distance_sq M A - distance_sq M B = d ↔ 
  ∃ x : ℝ, ∃ y : ℝ, (M.1, M.2) = (x, y) ∧ 
  x = ((B.1 - A.1)^2 + d) / (2 * (B.1 - A.1)) :=
by
  sorry

end locus_of_points_l614_614024


namespace smallest_positive_period_of_f_max_value_of_f_on_interval_l614_614167

-- Define the function f
def f (x : ℝ) : ℝ := sin (2 * x + π / 6) - 2 * (cos x) ^ 2

-- Theorem for the smallest positive period of f
theorem smallest_positive_period_of_f : 
  (∀ x, f (x + π) = f x) ∧ (∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ π) :=
by
  sorry

-- Theorem for the maximum value of f on the given interval
theorem max_value_of_f_on_interval :
  ∃ x ∈ Icc (-π / 3) (π / 6), f x = -1 / 2 ∧ ∀ y ∈ Icc (-π / 3) (π / 6), f y ≤ -1 / 2 :=
by
  sorry

end smallest_positive_period_of_f_max_value_of_f_on_interval_l614_614167


namespace proof_problem_l614_614870

-- Definitions based on conditions
def proposition_1 := ∀ x, x ↔ sin x -- Interpretation: "x=" is a sufficient but not necessary condition for "sin x="
def proposition_2 (p q : Prop) := p ∨ q → p ∧ q -- Interpretation of logical connectives
def proposition_3 (a b : ℝ) := a < b → a^2 < b^2 -- Interpretation for real numbers
def proposition_4 (A B : Set ℕ) := A ∩ B = A → A ⊆ B -- Interpretation for sets

-- The equivalent Lean statement
theorem proof_problem :
  (proposition_1 ∧ proposition_4) ∧ ¬(proposition_2 true true) ∧ ¬(proposition_3 1 2) :=
sorry

end proof_problem_l614_614870


namespace inequality_proof_l614_614149

variables (a b c : ℕ)
hypothesis (h₁ : c ≥ b)

theorem inequality_proof (a b c : ℕ) (h₁ : c ≥ b) : 
  a^b * (a + b)^c >  c^b * a^c :=
sorry

end inequality_proof_l614_614149


namespace cone_bead_path_l614_614422

theorem cone_bead_path (r h : ℝ) (h_sqrt : h / r = 3 * Real.sqrt 11) : 3 + 11 = 14 := by
  sorry

end cone_bead_path_l614_614422


namespace find_x_l614_614915

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 := 
by 
  {
    sorry
  }

end find_x_l614_614915


namespace exp_log_pb_eq_log_ba_l614_614556

noncomputable def log_b (b a : ℝ) := Real.log a / Real.log b

theorem exp_log_pb_eq_log_ba (a b p : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : p = log_b b (log_b b a) / log_b b a) :
  a^p = log_b b a :=
by
  sorry

end exp_log_pb_eq_log_ba_l614_614556


namespace triangle_balls_l614_614710

theorem triangle_balls (n : ℕ) (num_tri_balls : ℕ) (num_sq_balls : ℕ) :
  (∀ n : ℕ, num_tri_balls = n * (n + 1) / 2)
  ∧ (num_sq_balls = num_tri_balls + 424)
  ∧ (∀ s : ℕ, s = n - 8 → s * s = num_sq_balls)
  → num_tri_balls = 820 :=
by sorry

end triangle_balls_l614_614710


namespace acceptable_a_interval_l614_614558

theorem acceptable_a_interval :
  ∀ (s m : ℕ), s = 120 → m = 150 →
    (∀ k, 1 ≤ k ∧ k ≤ m → (∃ a, (100 * s) / m ≤ (l k) ∧ (l k) ≤ a)) →
    let lower_bound := (100 * s) / m in
    let upper_bound := (100 * s) / (m - 1) in
    (80 : ℚ) ≤ lower_bound ∧ (lower_bound : ℚ) < upper_bound :=
by
  intros s m h₁ h₂ h₃ lower_bound upper_bound
  sorry

end acceptable_a_interval_l614_614558


namespace _l614_614541

noncomputable def unit_vectors {V : Type*} [inner_product_space ℝ V] (a b : V) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1

noncomputable theorem vector_magnitude (a b : ℝ) [inner_product_space ℝ V] (ha : unit_vectors a b) (h : ∥3 • a - 2 • b∥ = real.sqrt 7) :
  ∥3 • a + b∥ = real.sqrt 13 :=
sorry

end _l614_614541


namespace cost_per_serving_is_3_62_l614_614344

noncomputable def cost_per_serving : ℝ :=
  let beef_cost := 4 * 6
  let chicken_cost := (2.2 * 5) * 0.85
  let carrots_cost := 2 * 1.50
  let potatoes_cost := (1.5 * 1.80) * 0.85
  let onions_cost := 1 * 3
  let discounted_carrots := carrots_cost * 0.80
  let discounted_potatoes := potatoes_cost * 0.80
  let total_cost_before_tax := beef_cost + chicken_cost + discounted_carrots + discounted_potatoes + onions_cost
  let sales_tax := total_cost_before_tax * 0.07
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  total_cost_after_tax / 12

theorem cost_per_serving_is_3_62 : cost_per_serving = 3.62 :=
by
  sorry

end cost_per_serving_is_3_62_l614_614344


namespace cesaro_sum_100_term_seq_l614_614127

def cesaro_sum (l : List ℝ) : ℝ :=
  let sums := List.scanl (· + ·) 0 l
  sums.tail.sum / l.length

theorem cesaro_sum_100_term_seq (P : List ℝ) (hP_len : P.length = 99) (hP_cesaro : cesaro_sum P = 1000) :
  cesaro_sum (1 :: P) = 991 := 
  sorry

end cesaro_sum_100_term_seq_l614_614127


namespace smallest_side_length_is_5_l614_614055

theorem smallest_side_length_is_5 :
  ∃ (n : ℕ), n ≥ 1 ∧
  ∃ (squares : list ℕ), length squares = 15 ∧
  (∀ m ∈ squares, m > 0) ∧
  (∀ m ∈ squares, m * m ∈ (1 :: squares)) ∧
  (count squares 1 = 12) ∧
  (n * n = sum (map (λ m, m * m) squares)) ∧
  n = 5 :=
by
  sorry

end smallest_side_length_is_5_l614_614055


namespace find_x_eq_2_l614_614923

theorem find_x_eq_2 : ∀ x : ℝ, 2^10 = 32^x → x = 2 := 
by 
  intros x h
  sorry

end find_x_eq_2_l614_614923


namespace find_x_l614_614247

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (hxy : x + y + x * y = 80) : x = 26 :=
sorry

end find_x_l614_614247


namespace greatest_common_multiple_less_than_150_l614_614366

theorem greatest_common_multiple_less_than_150 : 
  ∀ (a b : ℕ), a = 10 → b = 15 → ∃ m, m < 150 ∧ m % a = 0 ∧ m % b = 0 ∧ ∀ n, n < 150 ∧ n % a = 0 ∧ n % b = 0 → n ≤ 120 :=
begin
  intros a b ha hb,
  rw [ha, hb],
  use 120,
  split,
  { exact lt_irrefl _ },
  split,
  { exact nat.mod_eq_zero_of_dvd },
  split,
  { exact nat.mod_eq_zero_of_dvd },
  { intros n hn,
    have h₁ : lcm 10 15 = 30 := by norm_num,
    sorry } -- the proof is left as an exercise
end

end greatest_common_multiple_less_than_150_l614_614366


namespace find_missing_number_l614_614355

theorem find_missing_number (n x : ℕ) (h : n * (n + 1) / 2 - x = 2012) : x = 4 := by
  sorry

end find_missing_number_l614_614355


namespace min_value_proof_l614_614986

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  2 / a + 2 / b + 2 / c

theorem min_value_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_abc : a + b + c = 9) : 
  minimum_value a b c ≥ 2 := 
by 
  sorry

end min_value_proof_l614_614986


namespace trigonometric_expression_eval_l614_614817

theorem trigonometric_expression_eval :
  2 * (Real.cos (5 * Real.pi / 16))^6 +
  2 * (Real.sin (11 * Real.pi / 16))^6 +
  (3 * Real.sqrt 2 / 8) = 5 / 4 :=
by
  sorry

end trigonometric_expression_eval_l614_614817


namespace range_of_x0_l614_614530

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else sqrt x

theorem range_of_x0 (x0 : ℝ) : f x0 > 1 ↔ x0 ∈ set.Iio (-1) ∪ set.Ioi 1 :=
by sorry

end range_of_x0_l614_614530


namespace correct_answers_proof_l614_614569

variable (n p q s c : ℕ)
variable (total_questions points_per_correct penalty_per_wrong total_score correct_answers : ℕ)

def num_questions := 20
def points_correct := 5
def penalty_wrong := 1
def total_points := 76

theorem correct_answers_proof :
  (total_questions * points_per_correct - (total_questions - correct_answers) * penalty_wrong) = total_points →
  correct_answers = 16 :=
by {
  sorry
}

end correct_answers_proof_l614_614569


namespace sum_f_f_inv_eq_l614_614141

noncomputable def f(x : ℕ) := (2 : ℕ) ^ x
noncomputable def f_inv(y : ℕ) := Real.log2 (y : ℝ)

theorem sum_f_f_inv_eq (n : ℕ) :
  (∑ k in Finset.range n, f k * f_inv (2 ^ k)) = (n - 1) * 2^(n + 1) + 2 := by
  sorry

end sum_f_f_inv_eq_l614_614141


namespace part1_part2_part3_l614_614508

variable {α : Type*} [LinearOrderedField α]

-- Given conditions
variable (a : ℕ → α)
variable (S : ℕ → α)
variable (h_neq_zero : ∀ n, a n ≠ 0)
variable (h_partial_sum : ∀ n, S n = ∑ i in finset.range n, a i)
variable (h_condition : ∀ m n > 0, (n - m) * S (n + m) = (n + m) * (S n - S m))

-- Prove that \( \frac{S_3}{a_2} = 3 \)
theorem part1 : S 3 / a 2 = 3 := sorry

-- Prove that \( \{a_n\} \) is an arithmetic sequence
theorem part2 : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n := sorry

-- Given p, q, r, s form a geometric sequence and a 1 ≠ a 2
variable (p q r s : ℕ)
variable (h_geo : (a q) * (a q) = (a p) * (a r) ∧ (a q) * (a s) = (a r) * (a r))
variable (h_neq : a 1 ≠ a 2)

-- Prove that \( q - p, r - q, s - r \) form a geometric sequence
theorem part3 : (q - p) * (s - r) = (r - q) * (r - q) := sorry

end part1_part2_part3_l614_614508


namespace question_1_question_2_l614_614836

noncomputable def parabola : Type := { p : ℝ × ℝ // p.1^2 = 4 * p.2 }
noncomputable def ellipse : Type := { p : ℝ × ℝ // p.1^2 / 6 + p.2^2 / 4 = 1 }

structure line (slope : ℝ) (intercept : ℝ) : Type :=
  (eqn : ∀ x : ℝ, (slope * x + intercept))
  (nonzero_slope : slope ≠ 0)

variables (l : line k m) (A B C D : ℝ × ℝ) 
          (OA OB OC OD : line)

axiom parabola_points : ∀ A B : parabola, (l.eqn A.1 = A.2) ∧ (l.eqn B.1 = B.2)
axiom ellipse_points : ∀ C D : ellipse, (l.eqn C.1 = C.2) ∧ (l.eqn D.1 = D.2)

def slope_OA : ℝ := (A.2 - 0) / (A.1 - 0)
def slope_OB : ℝ := (B.2 - 0) / (B.1 - 0)
def slope_OC : ℝ := (C.2 - 0) / (C.1 - 0)
def slope_OD : ℝ := (D.2 - 0) / (D.1 - 0)

theorem question_1 :
  (slope_OA + slope_OB) / (slope_OC + slope_OD) = -((m^2 - 4) / 8) :=
sorry

theorem question_2 :
  OA.1 * OB.1 + OA.2 * OB.2 = 0 →
  (∀ F : ℝ × ℝ, F = (0, 1)) →
  let t := sqrt (k^2 - 2) in
  18 * t / (3 * t^2 + 8) ≤ (3 * sqrt 6) / 4 :=
sorry

end question_1_question_2_l614_614836


namespace compare_abc_l614_614992

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614992


namespace lines_perpendicular_to_same_plane_are_parallel_l614_614500

variables {m n : Type} [line m] [line n]
variables {α : Type} [plane α]

def are_perpendicular_to_same_plane (m n : Type)
  (h1 : m ⟂ α) (h2 : n ⟂ α) : Prop :=
  m ∥ n

theorem lines_perpendicular_to_same_plane_are_parallel 
  (m n : Type) (α : Type) [line m] [line n] [plane α]
  (h1 : m ⟂ α) (h2 : n ⟂ α) : m ∥ n :=
sorry

end lines_perpendicular_to_same_plane_are_parallel_l614_614500


namespace total_money_l614_614748

-- Define the problem statement
theorem total_money (n : ℕ) (hn : 3 * n = 75) : (n * 1 + n * 5 + n * 10) = 400 :=
by sorry

end total_money_l614_614748


namespace perp_lines_a_value_l614_614865

theorem perp_lines_a_value :
  ∀ (a : ℝ), (∃ (p : a = 4), 
  ∀ x y : ℝ, ax + 2 * y - 1 = 0 → 2 * x - 4 * y + 5 = 0 → 
  (x + y ≠ (0, 0) → (-a / 2 * 1 / 2 = -1))) := 
sorry

end perp_lines_a_value_l614_614865


namespace projectile_height_reach_l614_614311

theorem projectile_height_reach (t : ℝ) (h : -16 * t^2 + 64 * t = 25) : t = 3.6 :=
by
  sorry

end projectile_height_reach_l614_614311


namespace area_of_triangle_PQR_l614_614054

noncomputable def len (p1 p2 : ℝ × ℝ × ℝ): ℝ :=
  ( p1.1 - p2.1 )^2 + ( p1.2 - p2.2 )^2 + ( p1.3 - p2.3 )^2

noncomputable def area_of_PQR : ℝ :=
  ∥ vecPQ × vecPR ∥ / 2

theorem area_of_triangle_PQR :
  let F := (0, 0, 0)
  let G := (4, 0, 0)
  let H := (4, 4, 0)
  let I := (0, 4, 0)
  let J := (0, 0, 10)
  let P := ((3/4)G + (1/4)J)
  let Q := ((3/4)I + (1/4)J)
  let R := ((1/2)H + (1/2)J)
  let vecPQ := Q - P
  let vecPR := R - P
  in area_of_PQR = 29 / 8 :=
by sorry

end area_of_triangle_PQR_l614_614054


namespace sum_of_remainders_mod_8_l614_614698

theorem sum_of_remainders_mod_8 
  (x y z w : ℕ)
  (hx : x % 8 = 3)
  (hy : y % 8 = 5)
  (hz : z % 8 = 7)
  (hw : w % 8 = 1) :
  (x + y + z + w) % 8 = 0 :=
by
  sorry

end sum_of_remainders_mod_8_l614_614698


namespace inequality_holds_for_positive_integers_l614_614627

theorem inequality_holds_for_positive_integers (n : ℕ) (h : n > 0) : 
  (2 * n^2 + 3 * n + 1)^n ≥ 6^n * (nat.factorial n)^2 := 
sorry

end inequality_holds_for_positive_integers_l614_614627


namespace range_of_a_l614_614532

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h : ∀ x ∈ set.Icc 0 (π / 2), f x ≤ 1) :
  ∀ x ∈ set.Icc 0 (π / 2), (sin x)^2 + a * cos x + a ≤ 1 → f x = (sin x)^2 + a * cos x + a → a <= 0 := 
begin
  intros x hx hfx f_eq,
  sorry,
end

end range_of_a_l614_614532


namespace problem_solution_l614_614832

open Real

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  (∃ (C₁ : ℝ), (2 : ℝ)^x + (4 : ℝ)^y = C₁ ∧ C₁ = 2 * sqrt 2) ∧
  (∃ (C₂ : ℝ), 1 / x + 2 / y = C₂ ∧ C₂ = 9) ∧
  (∃ (C₃ : ℝ), x^2 + 4 * y^2 = C₃ ∧ C₃ = 1 / 2) :=
by
  sorry

end problem_solution_l614_614832


namespace value_of_x_for_fn_inv_eq_l614_614453

def f (x : ℝ) : ℝ := 4 * x - 9
def f_inv (x : ℝ) : ℝ := (x + 9) / 4

theorem value_of_x_for_fn_inv_eq (x : ℝ) : f(x) = f_inv(x) → x = 3 :=
by
  sorry

end value_of_x_for_fn_inv_eq_l614_614453


namespace Jenny_walked_distance_l614_614254

-- Given: Jenny ran 0.6 mile.
-- Given: Jenny ran 0.2 miles farther than she walked.
-- Prove: Jenny walked 0.4 miles.

variable (r w : ℝ)

theorem Jenny_walked_distance
  (h1 : r = 0.6) 
  (h2 : r = w + 0.2) : 
  w = 0.4 :=
sorry

end Jenny_walked_distance_l614_614254


namespace same_focal_length_l614_614314

def hyperbola_eq : (ℝ → ℝ → Prop) := λ x y, 15 * y^2 - x^2 = 15
def ellipse_eq : (ℝ → ℝ → Prop) := λ x y, x^2 / 25 + y^2 / 9 = 1

def focal_length_hyperbola : ℝ := 2 * 4 -- focal length is 8 
def focal_length_ellipse : ℝ := 2 * 4 -- focal length is 8 

theorem same_focal_length :
  focal_length_hyperbola = focal_length_ellipse :=
by
  sorry

end same_focal_length_l614_614314


namespace log_eqn_proof_l614_614199

theorem log_eqn_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : Real.log a / Real.log 2 + Real.log b / Real.log 4 = 8)
  (h2 : Real.log a / Real.log 4 + Real.log b / Real.log 8 = 2) :
  Real.log a / Real.log 8 + Real.log b / Real.log 2 = -52 / 3 := 
by
  sorry

end log_eqn_proof_l614_614199


namespace closest_quotient_l614_614460

def closest_to_seventy (q1 q2 q3 : ℝ) :=
  if abs (q1 - 70) < abs (q2 - 70) ∧ abs (q1 - 70) < abs (q3 - 70) then q1
  else if abs (q2 - 70) < abs (q1 - 70) ∧ abs (q2 - 70) < abs (q3 - 70) then q2
  else q3

theorem closest_quotient :
  let q1 := (254.0 / 5)
  let q2 := (400.0 / 6)
  let q3 := (492.0 / 7) in
  closest_to_seventy q1 q2 q3 = q3 :=
by
  let q1 := (254.0 / 5)
  let q2 := (400.0 / 6)
  let q3 := (492.0 / 7)
  have h1 : q1 = 50.8 := by norm_num
  have h2 : q2 ≈ 66.67 := by norm_num
  have h3 : q3 ≈ 70.29 := by norm_num
  to_have : closest_to_seventy q1 q2 q3 = q3 sorry

end closest_quotient_l614_614460


namespace find_angle_BAC_l614_614579

variables (A B C D E : Type _) [EuclideanGeometry A B C D E]

noncomputable def equal_triangles (ABC EBD : Triangle A B C D E) : Prop :=
-- Two given equal triangles
ABC = EBD

noncomputable def angle_DAE := 37 -- Given angle DAE
noncomputable def angle_DEA := 37 -- Given angle DEA

theorem find_angle_BAC (ABC EBD : Triangle A B C D E)
  (h1 : equal_triangles ABC EBD)
  (h2 : angle_DAE = 37)
  (h3 : angle_DEA = 37) :
  ∠ BAC = 7 :=
sorry

end find_angle_BAC_l614_614579


namespace function_increment_l614_614877

theorem function_increment (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2 / x) : f 1.5 - f 2 = 1 / 3 := 
by {
  sorry
}

end function_increment_l614_614877


namespace slope_of_line_is_2_or_9over2_l614_614160

variable (x : ℚ)

def A := (1 : ℚ, -2 : ℚ)
def B := (3 : ℚ, 3 * x)
def C := (x, 4 : ℚ)

def collinear (A B C : ℚ × ℚ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

theorem slope_of_line_is_2_or_9over2 (x : ℚ) :
  collinear (A) (B x) (C x) → 
  (x = -2 ∨ x = 7 / 3) → 
  (let k := if x = -2 then -2 else 9 / 2 in
   k = -2 ∨ k = 9 / 2) :=
by
  sorry

end slope_of_line_is_2_or_9over2_l614_614160


namespace find_x_l614_614913

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 := 
by 
  {
    sorry
  }

end find_x_l614_614913


namespace integer_column_assignment_l614_614070

theorem integer_column_assignment (n : ℕ) (h : n > 1) : 
  ∃ col : char, col = 'E' → (n = 1025) := by
  sorry

end integer_column_assignment_l614_614070


namespace complex_in_second_quadrant_l614_614527

def sum_series : ℂ := ∑ k in Finset.range 11, complex.I ^ (k + 1)

theorem complex_in_second_quadrant (z : ℂ) (h : z = sum_series) : 
  ∃ (x y : ℝ), z = x + y*complex.I ∧ x < 0 ∧ y > 0 :=
by
  sorry

end complex_in_second_quadrant_l614_614527


namespace additional_discount_A_is_8_l614_614348

-- Define the problem conditions
def full_price_A : ℝ := 125
def full_price_B : ℝ := 130
def discount_B : ℝ := 0.10
def price_difference : ℝ := 2

-- Define the unknown additional discount of store A
def discount_A (x : ℝ) : Prop :=
  full_price_A - (full_price_A * (x / 100)) = (full_price_B - (full_price_B * discount_B)) - price_difference

-- Theorem stating that the additional discount offered by store A is 8%
theorem additional_discount_A_is_8 : discount_A 8 :=
by
  -- Proof can be filled in here
  sorry

end additional_discount_A_is_8_l614_614348


namespace alpha_beta_ways_to_leave_store_l614_614050

/-- 
The shop sells 7 different flavors of oreos and 4 different flavors of milk. 
Alpha will not order more than 1 of the same flavor, while Beta will only order oreos and allow repeats.
Prove that there are 4054 ways for Alpha and Beta to leave the store with 4 products collectively.
--/
theorem alpha_beta_ways_to_leave_store 
  (oreos : Fin 7) (milk : Fin 4) :
  let α_total_items := oreos.card + milk.card in
  α_total_items = 11 →
  let way_to_leave := 
    (choose 11 4) +                             -- Alpha 4 items
    (choose 11 3) * 7 +                          -- Alpha 3 items, Beta 1 oreo
    (choose 11 2) * (choose 7 2 + 7) +            -- Alpha 2 items, Beta 2 oreos (distinct or same)
    (choose 11 1) * (choose 7 3 + 7*6 + 7) +      -- Alpha 1 item, Beta 3 oreos (3 distinct, 2 same + 1 different, 3 same)
    (choose 7 4 + (7 * 6) / 2 + (7 * 6) + 7)      -- Alpha 0 items, Beta 4 oreos (4 distinct, two pairs, 3 same + 1 different, 4 same)
  in way_to_leave = 4054 := 
by sorry

end alpha_beta_ways_to_leave_store_l614_614050


namespace maximum_value_seq_l614_614502

def a_k (n k : ℕ) : ℝ := (↑(n - k) / ↑k) ^ (k - n / 2)

theorem maximum_value_seq (n : ℕ) (hn : n > 1) :
  (even n → ∃ k : ℕ, 0 < k ∧ k ≤ n ∧ ∀ j : ℕ, 0 < j ∧ j ≤ n → a_k n k ≥ a_k n j ∧ a_k n k = 1) ∧
  (odd n → ∃ k : ℕ, 0 < k ∧ k ≤ n ∧ ∀ j : ℕ, 0 < j ∧ j ≤ n → a_k n k ≥ a_k n j ∧ a_k n k = sqrt((n - 1) / (n + 1))) :=
sorry

end maximum_value_seq_l614_614502


namespace second_magician_can_determine_card_l614_614685

-- Define the set of 48 cards
def cards : set ℕ := {n | 1 ≤ n ∧ n ≤ 48}

-- Define what it means to be a pair that sums to 49
def pairs_to_49 (a b : ℕ) : Prop := a + b = 49

-- Assume a selection of 25 cards from the set of 48 cards
axiom selected_cards : set ℕ
axiom subset_selected : selected_cards ⊆ cards
axiom card_selected_25 : card selected_cards = 25

-- Assume the pair selection by the first magician
axiom first_magician_pair (c1 c2 : ℕ) (h₁ : c1 ∈ selected_cards) (h₂ : c2 ∈ selected_cards) : pairs_to_49 c1 c2

-- Assume the addition of a card by the spectator and forming a new set of three cards
variables (c3 : ℕ) (new_cards : set ℕ)
axiom new_cards_setup : new_cards = {c1, c2, c3}
axiom c3_not_in_pair : c3 ∉ {c1, c2}

-- Prove that the second magician can determine the new card c3
theorem second_magician_can_determine_card (c1 c2 c3 : ℕ)
  (h₁ : c1 ∈ selected_cards) (h₂ : c2 ∈ selected_cards)
  (ht : pairs_to_49 c1 c2)
  (hnew : new_cards = {c1, c2, c3})
  (hc3 : c3 ∉ {c1, c2}) :
  c3 ∈ new_cards ∧ ¬ pairs_to_49 c3 c1 ∧ ¬ pairs_to_49 c3 c2 :=
sorry

end second_magician_can_determine_card_l614_614685


namespace count_points_inside_F_l614_614653

def is_inside_F (m n : ℕ) : Prop :=
  0 < m ∧ m < 100 ∧ 0 < n ∧ n < 100 ∧ 2^n < 2^m ∧ 2^n > 2^(m-100)

def count_points : ℕ :=
  finset.card ((finset.range 100).product (finset.range 100)).filter (λ p, is_inside_F p.1 p.2)

theorem count_points_inside_F : count_points = 2401 :=
  sorry

end count_points_inside_F_l614_614653


namespace compare_abc_l614_614989

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614989


namespace total_shoes_l614_614898

variable (a b c d : Nat)

theorem total_shoes (h1 : a = 7) (h2 : b = a + 2) (h3 : c = 0) (h4 : d = 2 * (a + b + c)) :
  a + b + c + d = 48 :=
sorry

end total_shoes_l614_614898


namespace incorrect_reasoning_example_l614_614031

-- Define what it means for a function to be exponential
def is_exponential_function (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x = a^x

-- Define the property of being an increasing function
def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the major premise; incorrectly assuming all exponential functions are increasing
def incorrect_major_premise : Prop :=
  ∀ (a : ℝ) (f : ℝ → ℝ), (0 < a ∧ a ≠ 1 ∧ is_exponential_function f a) → is_increasing_function f

-- Define the specific function in question
def specific_function : ℝ → ℝ := λ x, (1/2)^x

-- State the theorem we want to prove
theorem incorrect_reasoning_example : ¬ incorrect_major_premise :=
sorry

end incorrect_reasoning_example_l614_614031


namespace irreducibility_fraction_l614_614719

theorem irreducibility_fraction (n : ℤ) : irreducible (3 * n^2 + 2 * n + 4) (n + 1) ↔ ¬ (n ≡ 4 [ZMOD 5]) :=
by
  sorry

end irreducibility_fraction_l614_614719


namespace coeff_x6_in_expansion_l614_614360

theorem coeff_x6_in_expansion :
  let f := (1 - 3 * x^2)
  polynomial.coeff (f ^ 6) 6 = -540 :=
by
  sorry

end coeff_x6_in_expansion_l614_614360


namespace mr_johnson_pill_intake_l614_614278

theorem mr_johnson_pill_intake (total_days : ℕ) (remaining_pills : ℕ) (fraction : ℚ) (dose : ℕ)
  (h1 : total_days = 30)
  (h2 : remaining_pills = 12)
  (h3 : fraction = 4 / 5) :
  dose = 2 :=
by
  sorry

end mr_johnson_pill_intake_l614_614278


namespace option_c_not_equivalent_l614_614466

theorem option_c_not_equivalent :
  ¬ (785 * 10^(-9) = 7.845 * 10^(-6)) :=
by
  sorry

end option_c_not_equivalent_l614_614466


namespace price_difference_l614_614671

variable (P E : ℝ)

noncomputable def price_of_basic_computer : ℝ := 2125
noncomputable def total_price_basic_and_printer : ℝ := 2500

axiom printer_price_equation : price_of_basic_computer + P = total_price_basic_and_printer
axiom printer_price_condition : P = (1 / 8) * (E + P)

theorem price_difference (price_of_basic_computer := 2125) : E - price_of_basic_computer = 500 := by
  have h1 : P = total_price_basic_and_printer - price_of_basic_computer,
  sorry,
  have h2 : E = (8 * P) - P,
  sorry,
  show E - price_of_basic_computer = 500,
  sorry

end price_difference_l614_614671


namespace total_shoes_tried_on_l614_614900

variable (T : Type)
variable (store1 store2 store3 store4 : T)
variable (pair_of_shoes : T → ℕ)
variable (c1 : pair_of_shoes store1 = 7)
variable (c2 : pair_of_shoes store2 = pair_of_shoes store1 + 2)
variable (c3 : pair_of_shoes store3 = 0)
variable (c4 : pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3))

theorem total_shoes_tried_on (store1 store2 store3 store4 : T) (pair_of_shoes : T → ℕ) : 
  pair_of_shoes store1 = 7 →
  pair_of_shoes store2 = pair_of_shoes store1 + 2 →
  pair_of_shoes store3 = 0 →
  pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3) →
  pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3 + pair_of_shoes store4 = 48 := by
  intro c1 c2 c3 c4
  sorry

end total_shoes_tried_on_l614_614900


namespace recurring_decimal_as_fraction_l614_614015

theorem recurring_decimal_as_fraction :
  ∃ (x : ℚ), x = 0.726726726… ∧ x = 242 / 333 := by
sorry

end recurring_decimal_as_fraction_l614_614015


namespace general_term_S_limit_S_l614_614772
noncomputable theory

-- Definitions based on conditions:
-- Initial area of P0 is 1
def initial_area : ℝ := 1

-- Function to define the sequence of areas
def S (n : ℕ) : ℝ :=
  if n = 0 then 1
  else (8 / 5) - (3 / 5) * ( (4 / 9)^(n-1) )

-- Lean Statement 1: The general term formula for the sequence {S_n}.
theorem general_term_S (n : ℕ) : S (n) = if n = 0 then 1 else (8 / 5) - (3 / 5) * ( (4 / 9)^(n-1) ) :=
sorry

-- Lean Statement 2: The limit of the sequence {S_n} as n approaches infinity.
theorem limit_S : tendsto (λ (n : ℕ), S n) at_top (𝓝 (8 / 5)) :=
sorry

end general_term_S_limit_S_l614_614772


namespace polar_line_equation_l614_614321

theorem polar_line_equation (p : ℝ × ℝ) (h1 : p = (2, π / 4)) (h2 : ∀ θ, p.1 * sin θ = √2) : ∃ ρ θ, ρ * sin θ = √2 :=
by {
  use 2, π / 4, sorry
}

end polar_line_equation_l614_614321


namespace positive_terms_count_l614_614904

theorem positive_terms_count (n : ℕ) : 
  n = 100 →
  (λ seq, seq = λ k, sin (10^k : ℝ)) n →
  (count (λ k, sin (10^k : ℝ) > 0) (range 100)) = 3 :=
sorry

end positive_terms_count_l614_614904


namespace find_x_l614_614927

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 :=
by
  have h1 : 32 = 2^5 := by norm_num
  rw [h1, pow_mul] at h
  have h2 : 2^(10) = 2^(5*x) := by exact h
  have h3 : 10 = 5 * x := by exact (pow_inj h2).2
  linarith

end find_x_l614_614927


namespace false_proposition_c_l614_614014

theorem false_proposition_c :
  ¬ (∃ (planes : list Proposition), 
    (sizeof planes = 2) ∧ 
    (∀ (plane : Proposition), plane ∈ planes → intersects plane (hd planes)) ∧ 
    (∀ (plane : Proposition), plane ∈ planes → perpendicular_to_same_line plane)) :=
  sorry

end false_proposition_c_l614_614014


namespace decreasing_on_0_inf_l614_614434

theorem decreasing_on_0_inf : ∀ x : ℝ, 0 < x → deriv (λ x, -x * (x + 2)) x < 0 := 
sorry

end decreasing_on_0_inf_l614_614434


namespace athlete_group_problem_l614_614492

theorem athlete_group_problem
  (x y : ℕ)
  (h1 : 7 * y + 3 = x)
  (h2 : 8 * y - 5 = x) :
  x = 59 ∧ y = 8 :=
begin
  sorry
end

end athlete_group_problem_l614_614492


namespace intersection_of_sets_l614_614539

open Set

theorem intersection_of_sets (p q : ℝ) :
  (M = {x : ℝ | x^2 - 5 * x < 0}) →
  (M = {x : ℝ | 0 < x ∧ x < 5}) →
  (N = {x : ℝ | p < x ∧ x < 6}) →
  (M ∩ N = {x : ℝ | 2 < x ∧ x < q}) →
  p + q = 7 :=
by
  intros h1 h2 h3 h4
  sorry

end intersection_of_sets_l614_614539


namespace slope_of_line_l614_614461

noncomputable def line_equation (x y : ℝ) : Prop := 4 * y + 2 * x = 10

theorem slope_of_line (x y : ℝ) (h : line_equation x y) : -1 / 2 = -1 / 2 :=
by
  sorry

end slope_of_line_l614_614461


namespace corresponding_angles_equal_l614_614587

theorem corresponding_angles_equal 
  (α β γ : ℝ) 
  (h1 : α + β + γ = 180) 
  (h2 : (180 - α) + β + γ = 180) : 
  α = 90 ∧ β + γ = 90 ∧ (180 - α = 90) :=
by
  sorry

end corresponding_angles_equal_l614_614587


namespace coefficient_x6_expansion_l614_614364

theorem coefficient_x6_expansion 
  (binomial_expansion : (1 - 3 * x^2) ^ 6 = ∑ k in range 7, (Nat.choose 6 k) * (1:ℝ) ^ (6 - k) * (-3 * x^2) ^ k) :
  (∑ k in range 7, (Nat.choose 6 k) * (1:ℝ) ^ (6 - k) * (-3 * x^2) ^ k).coeff 6 = -540 := 
by
  have term_with_x6 : ∑ k in range 7, (Nat.choose 6 k) * (1:ℝ) ^ (6 - k) * (-3 * x^2) ^ k 
    = (Nat.choose 6 3) * (1:ℝ) ^ 3 * (-3 * x^2) ^ 3 := sorry
  have coefficient_of_x6 : (Nat.choose 6 3 * (-3) ^ 3) = -540 := sorry
  sorry

end coefficient_x6_expansion_l614_614364


namespace frac_plus_a_ge_seven_l614_614292

theorem frac_plus_a_ge_seven (a : ℝ) (h : a > 3) : 4 / (a - 3) + a ≥ 7 := 
by
  sorry

end frac_plus_a_ge_seven_l614_614292


namespace area_ratio_limit_l614_614244

-- Definitions of the geometric conditions
variables (A B C D P Q R : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P] [Inhabited Q] [Inhabited R]
variables (triangleABC : Triangle A B C)
variables (angleA angleB angleC : Angle)
variables (altitudeAD : Altitude A D)
variables (medianBE : Median B E)
variables (angleBisectorCF : AngleBisector C F)

-- Defining intersections and conditions
variables (intersectPoints : IntersectPoints P Q R)
variables (pointDonBC : PointOnLine D B C)
variables (altitudeFromAD : IsAltitude A D)
variables (medianFromBE : IsMedian B E)
variables (angleBisectorFromCF : IsAngleBisector C F)
variables (x : Real → Real)

-- Representation of the area ratio condition
variables (areaPQR : Real) (areaABC : Real)

-- Conditions on angles
variables (angleOrdering : angleA ≥ angleB ∧ angleB ≥ angleC)
variables (xExpression : x = areaPQR / areaABC)

-- Statement to prove
theorem area_ratio_limit (h : x = areaPQR / areaABC) (hx : x < 1/6 ∧ ∀ ε > 0, ε < 1/6 → x > 1/6 - ε) :
  x = 1 / 6 :=
by
  sorry


end area_ratio_limit_l614_614244


namespace correct_reflexive_pronoun_l614_614016

def reflexive_pronoun (subject : String) : String :=
  match subject with
  | "you" => "yourself"
  | "I" => "myself"
  | "he" => "himself"
  | "she" => "herself"
  | "it" => "itself"
  | "we" => "ourselves"
  | "they" => "themselves"
  | _ => ""

theorem correct_reflexive_pronoun :
  reflexive_pronoun "you" = "yourself" :=
by
  simp [reflexive_pronoun]
  sorry

end correct_reflexive_pronoun_l614_614016


namespace perpendicular_line_through_point_l614_614103

-- Define the given line equation
def given_line (x : ℝ) : ℝ := (1 / 2) * x + 1

-- Define the perpendicular line equation through point (2,0)
noncomputable def perp_line (b : ℝ) (x : ℝ) : ℝ := -2 * x + b

-- Define the point (2,0)
def point := (2 : ℝ, 0 : ℝ)

-- The proof problem statement
theorem perpendicular_line_through_point :
  ∃ b : ℝ, perp_line b 2 = 0 ∧ ∀ x : ℝ, perp_line b x = -2 * x + 4 :=
sorry

end perpendicular_line_through_point_l614_614103


namespace sin_66_approx_l614_614154

theorem sin_66_approx (h : Real.cos (78 * Real.pi / 180) ≈ 1 / 5) : Real.sin (66 * Real.pi / 180) ≈ 0.92 :=
by
  sorry

end sin_66_approx_l614_614154


namespace part1_solution_part2_solution_l614_614504

noncomputable theory

-- Define the conditions
variable {f : ℝ → ℝ}
variable (x m a : ℝ)

-- Part 1
def part1_conditions : Prop := 
  ∀ x ∈ (0, +∞), (f 1 = 0) ∧ (f 3 = 1) ∧ (∀ x y, (x < y → f x < f y))

theorem part1_solution (h : part1_conditions) :
  (0 < f (x^2 - 1) ∧ f (x^2 - 1) < 1) ↔ (sqrt 2 < x ∧ x < 2) :=
sorry

-- Part 2
def part2_conditions : Prop :=
  ∀ (x : ℝ) (a : ℝ) (h1 : x ∈ (0, 3]) (h2 : a ∈ [-1, 1]), f x ≤ m^2 - 2*a*m + 1

theorem part2_solution (h : part2_conditions):
  ∀ m : ℝ, (1 ≤ m^2 - 2*a*m + 1) → (m ∈ (-∞, -2] ∪ {0} ∪ [2, +∞)) :=
sorry

end part1_solution_part2_solution_l614_614504


namespace fill_tank_time_l614_614715

-- Define the rates at which the pipes fill the tank
noncomputable def rate_A := (1:ℝ)/50
noncomputable def rate_B := (1:ℝ)/75

-- Define the combined rate of both pipes
noncomputable def combined_rate := rate_A + rate_B

-- Define the time to fill the tank at the combined rate
noncomputable def time_to_fill := 1 / combined_rate

-- The theorem that states the time taken to fill the tank is 30 hours
theorem fill_tank_time : time_to_fill = 30 := sorry

end fill_tank_time_l614_614715


namespace sequence_sum_example_l614_614842

noncomputable def a : ℕ → ℕ
| 1       := 1
| (n + 1) := (2^n) / (a n)

def S : ℕ → ℕ
| 0       := 0
| (n + 1) := S n + a (n + 1)

theorem sequence_sum_example :
  (S 20 = 3069) := 
begin
  sorry
end

end sequence_sum_example_l614_614842


namespace series_sum_l614_614110

-- Define the sum of the series for n from 1 to 64
noncomputable def series : ℚ :=
  ∑ n in Finset.range 64,  1 / ( (3 * (n + 1) - 1) * (3 * (n + 1) + 2))

-- State the theorem to be proven
theorem series_sum : series = 32 / 194 :=
  by sorry

end series_sum_l614_614110


namespace parabola_equation_l614_614664

-- Definitions
def parabola (p : ℝ) : Prop := ∃ (x y : ℝ), y^2 = 2 * p * x

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def origin : (ℝ × ℝ) := (0, 0)

def O_distance (F : (ℝ × ℝ)) : ℝ := distance origin F

def area (A B C : ℝ × ℝ) : ℝ := 0.5 * real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Hypotheses
axiom h_positive_p (p : ℝ) : p > 0
axiom h_M_on_parabola (p : ℝ) (M : ℝ × ℝ) : (∃ x y, M = (x, y) ∧ y^2 = 2 * p * x)
axiom h_distance_relation (p : ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) : distance M F = 4 * O_distance F
axiom h_area (M F O : ℝ × ℝ) : area M F O = 4 * real.sqrt 3

-- Theorem to prove
theorem parabola_equation (p : ℝ) (M : ℝ × ℝ) (F O : ℝ × ℝ)
  (h_focus : F = focus p)
  (h_area_eq : area M F O = 4 * real.sqrt 3)
  (h_M_parabola : ∃ x y, M = (x, y) ∧ y^2 = 2 * p * x)
  (h_distance_eq : distance M F = 4 * O_distance F)
  : parabola 4 ∧ (∃ x y, y^2 = 8 * x) :=
sorry

end parabola_equation_l614_614664


namespace real_roots_of_polynomial_l614_614487

theorem real_roots_of_polynomial :
  ∃ x : ℝ, x ∈ ({-1, 1, 3} : set ℝ) ∧ (x^5 - 3*x^4 + 3*x^2 - x - 6 = 0) :=
sorry

end real_roots_of_polynomial_l614_614487


namespace existence_of_special_numbers_l614_614802

theorem existence_of_special_numbers :
  ∃ (N : Finset ℕ), N.card = 1998 ∧ 
  ∀ (a b : ℕ), a ∈ N → b ∈ N → a ≠ b → a * b ∣ (a - b)^2 :=
sorry

end existence_of_special_numbers_l614_614802


namespace Dima_impossible_cut_l614_614393

theorem Dima_impossible_cut (n : ℕ) 
  (h1 : n % 5 = 0) 
  (h2 : n % 7 = 0) 
  (h3 : n ≤ 200) : ¬(n % 6 = 0) :=
sorry

end Dima_impossible_cut_l614_614393


namespace projections_form_equilateral_triangle_l614_614973

theorem projections_form_equilateral_triangle
  {A B C M : Point}
  (h1 : ∠ A M C = 60 + ∠ A B C)
  (h2 : ∠ C M B = 60 + ∠ C A B)
  (h3 : ∠ B M A = 60 + ∠ B C A) :
  ∃ A1 B1 C1 : Point, 
  IsProjection A1 M B C ∧
  IsProjection B1 M C A ∧
  IsProjection C1 M A B ∧
  EquilateralTriangle A1 B1 C1 :=
sorry

end projections_form_equilateral_triangle_l614_614973


namespace sum_series_l614_614450

theorem sum_series : (Finset.sum (Finset.range 101) (λ n : ℕ, if n % 2 = 0 then n + 1 else -n)) = -50 :=
by
  sorry

end sum_series_l614_614450


namespace corresponding_angles_equal_l614_614589

theorem corresponding_angles_equal (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α = 90) :
  (180 - α = 90 ∧ β + γ = 90 ∧ α = 90) :=
by
  sorry

end corresponding_angles_equal_l614_614589


namespace net_difference_expenditure_l614_614020

-- Definitions of the conditions
variable (P Q : ℝ) -- original price and quantity
variable (new_price : ℝ) (new_quantity : ℝ)
variable (original_expenditure new_expenditure : ℝ)

-- Define the conditions
def condition1 : new_price = 1.25 * P := by
  sorry

def condition2 : new_quantity = 0.64 * Q := by
  sorry

def condition3 : original_expenditure = P * Q := by
  sorry

def condition4 : new_expenditure = new_price * new_quantity := by
  sorry

-- The theorem to prove the net difference is 20% less
theorem net_difference_expenditure : (original_expenditure - new_expenditure) = 0.2 * original_expenditure := by
  have h1 : new_expenditure = 1.25 * P * 0.64 * Q := by
    rw [condition4, condition1, condition2]
  have h2 : new_expenditure = 0.8 * (P * Q) := by
    rw [mul_comm (1.25 * P) (0.64 * Q), mul_assoc, ← mul_assoc, mul_comm P 0.8, ← mul_comm 1.25 0.64, mul_comm 1.25 0.8, mul_assoc 1.25 0.64]
    assume hyp : 1.25 * 0.64 = 0.8,
    rw [hyp]
  rw [h1, condition3, mul_comm],
  by sorry

end net_difference_expenditure_l614_614020


namespace train_length_l614_614766

theorem train_length (time : ℝ) (speed_kmh : ℝ) (speed_ms : ℝ) (length : ℝ) : 
  time = 20 ∧ speed_kmh = 50.4 ∧ speed_ms = (speed_kmh * 1000 / 3600) ∧ length = speed_ms * time → length = 280 :=
by 
  intros h 
  cases h with h_time h_rest 
  cases h_rest with h_speed_kmh h_rest2 
  cases h_rest2 with h_speed_ms h_length 
  rw [h_time, h_speed_kmh, h_speed_ms, h_length] 
  sorry

end train_length_l614_614766


namespace yuna_has_greater_sum_l614_614380

theorem yuna_has_greater_sum : 
  let yoosum := 5 + 8 in 
  let yunasum := 7 + 9 in 
  yunasum > yoosum :=
by
  let yoosum := 5 + 8
  let yunasum := 7 + 9
  sorry

end yuna_has_greater_sum_l614_614380


namespace count_arithmetic_progressions_22_1000_l614_614181

def num_increasing_arithmetic_progressions (n k max_val : ℕ) : ℕ :=
  -- This is a stub for the arithmetic sequence counting function.
  sorry

theorem count_arithmetic_progressions_22_1000 :
  num_increasing_arithmetic_progressions 22 22 1000 = 23312 :=
sorry

end count_arithmetic_progressions_22_1000_l614_614181


namespace coloring_count_l614_614985

theorem coloring_count (n : ℕ) (h : 0 < n) :
  ∃ (num_colorings : ℕ), num_colorings = 2 :=
sorry

end coloring_count_l614_614985


namespace sqrt_sum_simplify_equal_l614_614012

theorem sqrt_sum_simplify_equal (a b : ℝ) :
  (a^2 + 3*b^2 = 1 ∧ a*b = -1) →
  (sqrt(16 - 8 * sqrt 3) + sqrt(16 + 8 * sqrt 3) = 8 * sqrt 3) :=
by
  -- Define the given conditions
  have h1 : sqrt(1 - 2 * sqrt 3) = a + b * sqrt 3,
    from sorry,
  have h2 : sqrt(1 + 2 * sqrt 3) = a * sqrt 3 + b * sqrt 3,
    from sorry,
  -- Prove the final expression
  rw [h1, h2],
  calc
    sqrt(16 - 8 * sqrt 3) + sqrt(16 + 8 * sqrt 3)
        = 4 * (sqrt(1 - 2 * sqrt 3) + sqrt(1 + 2 * sqrt 3)) : sorry
    ... = 8 * sqrt 3 : sorry

end sqrt_sum_simplify_equal_l614_614012


namespace airplane_seat_count_l614_614076

theorem airplane_seat_count (s : ℝ) 
  (h1 : 30 + 0.2 * s + 0.75 * s = s) : 
  s = 600 :=
sorry

end airplane_seat_count_l614_614076


namespace sqrt_approximation_l614_614694

theorem sqrt_approximation :
  2 * (Real.sqrt 130 - Real.sqrt 11) ≈ 16.16 :=
by
  sorry

end sqrt_approximation_l614_614694


namespace determine_k_for_line_through_point_l614_614464

theorem determine_k_for_line_through_point :
  ∃ k : ℚ, (∀ x y : ℚ, (x = 3 ∧ y = -4) → (2 * k * x - 5 = 4 * y)) → k = (-11) / 6 :=
by
  -- Condition: Line passes through the point (3, -4)
  have h : ∀ k : ℚ, (2 * k * 3 - 5 = 4 * (-4)) → k = (-11) / 6,
  intro k,
  intro h_eq,
  -- Now, solve the equation 6k - 5 = -16
  linarith,
  -- Therefore k = (-11) / 6
  use (-11) / 6,
  sorry

end determine_k_for_line_through_point_l614_614464


namespace number_of_centered_four_digit_numbers_l614_614356

def is_centered_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  let digits := (List.ofFn (λ i => (n / 10 ^ i % 10) : ℕ)).filter (λ x => x ≠ 0)
  in ∀ perm : List ℕ, perm ∈ digits.permutations →
  List.nth (digits.sorted nat.lt) (digits.length / 2 - 1) = some (digits.nth (digits.length / 2 - 1)) ∧ 
  List.nth (digits.sorted nat.lt) (digits.length / 2) = some (digits.nth (digits.length / 2))

theorem number_of_centered_four_digit_numbers : 
  ∃! n, is_centered_four_digit n :=
sorry

end number_of_centered_four_digit_numbers_l614_614356


namespace large_A_exists_l614_614823

noncomputable def F_n (n a : ℕ) : ℕ :=
  let q := a / n
  let r := a % n
  q + r

theorem large_A_exists : ∃ n1 n2 n3 n4 n5 n6 : ℕ,
  ∀ a : ℕ, a ≤ 53590 → 
  F_n n6 (F_n n5 (F_n n4 (F_n n3 (F_n n2 (F_n n1 a))))) = 1 :=
by
  sorry

end large_A_exists_l614_614823


namespace sum_g_inverse_4095_l614_614602

-- Definition of g(n)
def g (n : ℕ) : ℕ := 
  let m := (Real.cbrt n).toNat
  if (m : ℝ) + 0.5 > Real.cbrt n then m else m + 1

-- Statement of the theorem
theorem sum_g_inverse_4095 : (∑ k in Finset.range 4095, 1 / (g (k + 1))) = 424 := by
  sorry -- The proof is omitted

end sum_g_inverse_4095_l614_614602


namespace segment_AB_length_l614_614584

variable (h AB CD : ℝ)
variable (ratio : ℝ) (AB_plus_CD : ℝ)

axiom ratio_condition : ratio = 3 * (CD / h) / (CD / h)
axiom ab_cd_sum_condition : AB + CD = 320

theorem segment_AB_length :
  ∃ AB CD : ℝ, (3 * (CD / h) / (CD / h) = 3) ∧ (AB + CD = 320) → AB = 240 :=
by
  intro AB CD ratios_sum_conditions,
  cases ratios_sum_conditions with ratio_condition ab_cd_sum_condition,
  sorry

end segment_AB_length_l614_614584


namespace train_length_is_225_m_l614_614064

noncomputable def speed_kmph : ℝ := 90
noncomputable def time_s : ℝ := 9

noncomputable def speed_ms : ℝ := speed_kmph / 3.6
noncomputable def distance_m (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_length_is_225_m :
  distance_m speed_ms time_s = 225 := by
  sorry

end train_length_is_225_m_l614_614064


namespace maximum_rectangle_area_l614_614426

-- Define the perimeter condition
def perimeter (rectangle : ℝ × ℝ) : ℝ :=
  2 * rectangle.fst + 2 * rectangle.snd

-- Define the area function
def area (rectangle : ℝ × ℝ) : ℝ :=
  rectangle.fst * rectangle.snd

-- Define the question statement in terms of Lean
theorem maximum_rectangle_area (length_width : ℝ × ℝ) (h : perimeter length_width = 32) : 
  area length_width ≤ 64 :=
sorry

end maximum_rectangle_area_l614_614426


namespace commercials_time_l614_614623

theorem commercials_time (n : ℕ) (t : ℕ) (h : t = 30) (k : ℕ) (one_fourth : ℚ) (h_fraction : one_fourth = 1 / 4) :
  n = 6 → 
  k = 6 * t → 
  k / 4 = 45 :=
by
  intros h_n h_k
  rw [h, ← nat.cast_mul, ← nat.cast_div, h_fraction] at h_k
  norm_num at h_k
  assumption

end commercials_time_l614_614623


namespace percentage_workers_present_l614_614712

theorem percentage_workers_present (total_present: ℝ) (total_workers: ℝ) (percentage: ℝ) : 
  total_present = 72 → 
  total_workers = 86 →
  percentage = 83.7 →
  Real.round (total_present / total_workers * 100 * 10) / 10 = percentage :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end percentage_workers_present_l614_614712


namespace solve_xyz_l614_614293

theorem solve_xyz (a b c : ℝ) (h1 : a = y + z) (h2 : b = x + z) (h3 : c = x + y) 
                   (h4 : 0 < y) (h5 : 0 < z) (h6 : 0 < x)
                   (hab : b + c > a) (hbc : a + c > b) (hca : a + b > c) :
  x = (b - a + c)/2 ∧ y = (a - b + c)/2 ∧ z = (a + b - c)/2 :=
by
  sorry

end solve_xyz_l614_614293


namespace number_2007_is_2_sum_first_2007_numbers_is_3952_l614_614756

noncomputable def sequence : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => if (n + 1) % 3 == 0 then 1 else 2

theorem number_2007_is_2 :
  sequence 2006 = 2 :=
sorry

theorem sum_first_2007_numbers_is_3952 :
  (∑ i in Finset.range 2007, sequence i) = 3952 :=
sorry

end number_2007_is_2_sum_first_2007_numbers_is_3952_l614_614756


namespace central_moments_properties_l614_614628

variable (X : Type) [ProbabilityTheory X] (p : X → ℝ) (a : ℝ) 

noncomputable def zeroth_central_moment (X : Type) [ProbabilityTheory X] (p : X → ℝ) (a : ℝ) : ℝ :=
∑ i, ((λ x_i, (x_i - a)^0 * p x_i) i)

noncomputable def first_central_moment (X : Type) [ProbabilityTheory X] : ℝ := 
𝔼[X] - 𝔼[X]

noncomputable def second_central_moment (X : Type) [ProbabilityTheory X] (p : X → ℝ) (a : ℝ) : ℝ :=
∑ i, ((λ x_i, (x_i - a)^2 * p x_i) i)

theorem central_moments_properties (X : Type) [ProbabilityTheory X] (p : X → ℝ) (a : ℝ) :
  zeroth_central_moment X p a = 1 ∧
  first_central_moment X = 0 ∧
  second_central_moment X p a = 𝔼[(λ x, (x - a)^2 p x) X] :=
by
  sorry

end central_moments_properties_l614_614628


namespace magnitude_of_b_l614_614828

noncomputable def vector_magnitude_problem (a b : E) [inner_product_space ℝ E] : Prop :=
  (inner_product_space.inner a b = -12 * real.sqrt 2) ∧ 
  (norm a = 4) ∧ 
  (real.angle_between_vectors a b = 135)

theorem magnitude_of_b (a b : E) [inner_product_space ℝ E]
  (h : vector_magnitude_problem a b) : norm b = 6 :=
by sorry

end magnitude_of_b_l614_614828


namespace satisfies_conditions_l614_614376

noncomputable def m := 29 / 3

def real_part (m : ℝ) : ℝ := m^2 - 8*m + 15
def imag_part (m : ℝ) : ℝ := m^2 - 5*m - 14

theorem satisfies_conditions (m : ℝ) 
  (real_cond : m < 3 ∨ m > 5) 
  (imag_cond : -2 < m ∧ m < 7)
  (line_cond : real_part m = imag_part m): 
  m = 29 / 3 :=
by {
  sorry
}

end satisfies_conditions_l614_614376


namespace interval_is_correct_l614_614682

def total_population : ℕ := 2000
def sample_size : ℕ := 40
def interval_between_segments (N : ℕ) (n : ℕ) : ℕ := N / n

theorem interval_is_correct : interval_between_segments total_population sample_size = 50 :=
by
  sorry

end interval_is_correct_l614_614682


namespace integral_e_x_plus_2x_l614_614805

open Real

theorem integral_e_x_plus_2x : ∫ x in 0..1, (exp x + 2 * x) = exp 1 :=
by
  -- Proof goes here
  sorry

end integral_e_x_plus_2x_l614_614805


namespace expression_calculation_l614_614084

theorem expression_calculation (x : ℝ) : (2 + x^2) * (1 - x^4) = -x^6 + x^2 - 2x^4 + 2 :=
by
  sorry

end expression_calculation_l614_614084


namespace ratio_C_divides_AE_l614_614840

theorem ratio_C_divides_AE (A B C D E: Point) 
  (hABC: RightTriangle A B C) 
  (hBC: OnHypotenuse B C) 
  (ω: Circle) 
  (hCircumABC: IsCircumcircle ω A B C) 
  (hD: OnExtension BC D) 
  (hTangency: TangentToCircle AD ω A)
  (hIntersection: IntersectsCircle AC (CircumcircleOfTriangle A B D) E)
  (hAngleBisectorTangency: TangentToCircle (AngleBisector ADE) ω)
  : ratio (C, AE) = 1 / 2 := 
sorry

end ratio_C_divides_AE_l614_614840


namespace coefficient_of_x7_in_expansion_eq_15_l614_614648

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := n.choose k

theorem coefficient_of_x7_in_expansion_eq_15 (a : ℝ) (hbinom : binomial 10 3 * (-a) ^ 3 = 15) : a = -1 / 2 := by
  sorry

end coefficient_of_x7_in_expansion_eq_15_l614_614648


namespace find_min_max_of_f_l614_614408

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem find_min_max_of_f : (∀ x ∈ set.Icc (-2 : ℝ) 2, 0 ≤ f x) ∧
                           (∀ x ∈ set.Icc (-2 : ℝ) 2, f x ≤ 9) ∧
                           (∃ x ∈ set.Icc (-2 : ℝ) 2, f x = 0) ∧
                           (∃ x ∈ set.Icc (-2 : ℝ) 2, f x = 9) := by
  sorry

end find_min_max_of_f_l614_614408


namespace seating_arrangements_l614_614523

theorem seating_arrangements (n : ℕ) (max_capacity : ℕ) 
  (h_n : n = 6) (h_max : max_capacity = 4) :
  ∃ k : ℕ, k = 50 :=
by
  sorry

end seating_arrangements_l614_614523


namespace range_of_sum_l614_614987

theorem range_of_sum (a b : ℝ) (h : a^2 - a * b + b^2 = a + b) :
  0 ≤ a + b ∧ a + b ≤ 4 :=
by
  sorry

end range_of_sum_l614_614987


namespace bivalid_positions_count_l614_614416

/-- 
A position of the hands of a (12-hour, analog) clock is called valid if it occurs in the course of a day.
A position of the hands is called bivalid if it is valid and, in addition, the position formed by interchanging the hour and minute hands is valid.
-/
def is_valid (h m : ℕ) : Prop := 
  0 ≤ h ∧ h < 360 ∧ 
  0 ≤ m ∧ m < 360

def satisfies_conditions (h m : Int) (a b : Int) : Prop :=
  m = 12 * h - 360 * a ∧ h = 12 * m - 360 * b

def is_bivalid (h m : ℕ) : Prop := 
  ∃ (a b : Int), satisfies_conditions (h : Int) (m : Int) a b ∧ satisfies_conditions (m : Int) (h : Int) b a

theorem bivalid_positions_count : 
  ∃ (n : ℕ), n = 143 ∧ 
  ∀ (h m : ℕ), is_bivalid h m → n = 143 :=
sorry

end bivalid_positions_count_l614_614416


namespace find_t_l614_614577

theorem find_t (t : ℝ) :
  let OA := (-1 : ℝ, t),
      OB := (2 : ℝ, 2),
      AB := (OB.1 - OA.1, OB.2 - OA.2)
  in
  (OB.1 * AB.1 + OB.2 * AB.2 = 0) → t = 5 :=
by {
  intros OA OB AB h,
  sorry
}

end find_t_l614_614577


namespace S8_div_S16_equals_3_div_10_l614_614846

variables {a_n : ℕ → ℝ} {S : ℕ → ℝ}

-- Define the sum function of the first n terms of the sequence
def sum_to (n : ℕ) : ℝ :=
  finset.sum (finset.range n) (λ k, a_n k)

-- State the conditions
axiom arithmetic_sequence_sum (n : ℕ) : S n = sum_to n
axiom ratio_S4_S8 : S 4 / S 8 = 1 / 3

-- Main theorem we want to prove
theorem S8_div_S16_equals_3_div_10 : (S 8) / (S 16) = 3 / 10 :=
sorry

end S8_div_S16_equals_3_div_10_l614_614846


namespace inequality_of_distances_and_sides_l614_614980

variables {A B C M : Type} [triangle ABC : Type] [inside M ABC : Type] 
          (d_a d_b d_c : ℝ) (a b c : ℝ) (S : ℝ)

def area_of_triangle : ℝ := S

theorem inequality_of_distances_and_sides :
  ∀ {ABC : Type} {M : Type} [triangle ABC : Type] [inside M ABC : Type], 
  (distance M BC = d_a) → (distance M CA = d_b) → (distance M AB = d_c) → 
  (length BC = a) → (length CA = b) → (length AB = c) → 
  area_of_triangle = S → 
  abd_ad_b + bcd_bd_c + cad_cd_a ≤ (4 * S * S) / 3 :=
sorry

end inequality_of_distances_and_sides_l614_614980


namespace budget_projection_l614_614069

theorem budget_projection (f w p : ℂ) (h_f : f = 7) (h_w : w = 70 + 210 * complex.I) (h_eq : f * p - w = 15000) : 
  p = 2153 + 30 * complex.I := by
  sorry

end budget_projection_l614_614069


namespace largest_frog_weight_l614_614316

theorem largest_frog_weight (S L : ℕ) (h1 : L = 10 * S) (h2 : L = S + 108): L = 120 := by
  sorry

end largest_frog_weight_l614_614316


namespace paintings_possible_l614_614227

noncomputable def num_distinct_paintings : ℕ :=
  (1 / 12 : ℝ) * (
    (choose 12 4) * 
    (choose 8 3) * 
    (choose 5 2) 
  )

theorem paintings_possible : num_distinct_paintings = 23100 := by
  sorry

end paintings_possible_l614_614227


namespace m_perp_β_l614_614864

variables (α β γ : Plane) (m n : Line)

-- The conditions
axiom α_parallel_β : α ∥ β
axiom n_perp_α : n ⊥ α
axiom n_perp_β : n ⊥ β
axiom m_perp_α : m ⊥ α

-- The proof statement
theorem m_perp_β : m ⊥ β :=
sorry

end m_perp_β_l614_614864


namespace find_n_150_l614_614792

def special_sum (k n : ℕ) : ℕ := (n * (2 * k + n - 1)) / 2

theorem find_n_150 : ∃ n : ℕ, special_sum 3 n = 150 ∧ n = 15 :=
by
  sorry

end find_n_150_l614_614792


namespace min_sqrt_x2_y2_l614_614514

-- Define the conditions: x and y are real numbers and they satisfy the equation x^2 + y^2 - 4x + 6y + 4 = 0
variables (x y : ℝ)
axiom h : x^2 + y^2 - 4x + 6y + 4 = 0

-- Define the statement to prove: the minimum value of sqrt(x^2 + y^2) under these conditions is sqrt(13) - 3
theorem min_sqrt_x2_y2 : ∃ x y : ℝ, (x^2 + y^2 - 4x + 6y + 4 = 0) ∧ ∀ u v : ℝ, (u^2 + v^2 - 4u + 6v + 4 = 0) → sqrt(u^2 + v^2) ≥ sqrt(13) - 3 :=
begin
  -- We state the existence of such x and y satisfying both the equation and the minimum condition
  use x,
  use y,
  split,
  { exact h },
  { intros u v h_uv,
    sorry }
end

end min_sqrt_x2_y2_l614_614514


namespace area_inequality_l614_614979

open Real

structure Triangle (α : Type) :=
(A B C : α) 

variables {α : Type} [OrderedField α]

def area {α : Type} [OrderedField α] (X Y Z : α × α) : α :=
  0.5 * abs ((X.1 * (Y.2 - Z.2)) + (Y.1 * (Z.2 - X.2)) + (Z.1 * (X.2 - Y.2)))

theorem area_inequality {A B C P Q R : α × α} (hP : P.1 = B.1 + (C.1 - B.1) * P.2)
  (hQ : Q.1 = C.1 + (A.1 - C.1) * Q.2) (hR : R.1 = A.1 + (B.1 - A.1) * R.2) :
  min (area A Q R) (min (area B R P) (area C P Q)) ≤ 0.25 * area A B C :=
sorry

end area_inequality_l614_614979


namespace dot_product_solution_l614_614559

variables {K : Type} [Field K]
variables {V : Type} [AddCommGroup V] [Module K V]
variables (a b c : V) (k : K)

-- Conditions
def parallel (a b : V) : Prop := ∃ k : K, b = k • a
def perpendicular (a c : V) : Prop := a ⬝ c = 0

-- Theorem statement
theorem dot_product_solution (h1 : parallel a b) (h2 : perpendicular a c) : c ⬝ (a + 2 • b) = 0 := by
  sorry

end dot_product_solution_l614_614559


namespace alien_exclamation_on_saturday_l614_614439

def exclamation_on_day (n : ℕ) : String :=
  match n with
  | 1 => "A"
  | 2 => "AU"
  | 3 => "AUUA"
  | 4 => "AUUAUAAU"
  | _ => exclamation_on_day (n - 1) ++ swap_letters (exclamation_on_day (n - 1))
  where
    swap_letters : String → String := 
      fun s => s.map (fun c => if c = 'A' then 'U' else 'A')

theorem alien_exclamation_on_saturday :
  exclamation_on_day 6 = "AUUAUAAUUAAUAUAUUAUAAUUAUUAU" :=
  sorry

end alien_exclamation_on_saturday_l614_614439


namespace maximum_value_expression_l614_614272

theorem maximum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : x^2 - 3 * x * y + 4 * y^2 - z = 0) :
  ∃ x y z, x^2 - 3 * x * y + 4 * y^2 - z = 0 
  ∧ (x > 0 ∧ y > 0 ∧ z > 0) 
  ∧ x + 2 * y - z ≤ 2 :=
sorry

end maximum_value_expression_l614_614272


namespace sequence_general_formula_sum_first_n_terms_l614_614242

def sequence (n : ℕ) : ℤ :=
  if n = 1 then 20
  else |sequence (n - 1) - 3|

theorem sequence_general_formula (n : ℕ) : 
  (sequence n = if n ≤ 7 then 23 - 3 * n else (3 + (-1)^(n-1)) / 2) := 
sorry

def sequence_sum (n : ℕ) : ℤ := 
  if n ≤ 7 then (43 * n - 3 * n^2) / 2
  else (3 * n) / 2 + (265 + (-1)^(n-1)) / 4

theorem sum_first_n_terms (n : ℕ) : 
  (sequence_sum n = if n ≤ 7 then (43 * n - 3 * n^2) / 2 else (3 * n) / 2 + (265 + (-1)^(n-1)) / 4) := 
sorry

end sequence_general_formula_sum_first_n_terms_l614_614242


namespace hall_length_l614_614039

theorem hall_length (L : ℝ) (H : ℝ) 
  (h1 : 2 * (L * 15) = 2 * (L * H) + 2 * (15 * H)) 
  (h2 : L * 15 * H = 1687.5) : 
  L = 15 :=
by 
  sorry

end hall_length_l614_614039


namespace complex_number_quadrant_l614_614946

theorem complex_number_quadrant (z : ℂ) (h : (1 + complex.i) * z = 2 - complex.i) : 
  ∃ (x y : ℝ), z = x + y * complex.i ∧ x > 0 ∧ y < 0 := 
sorry

end complex_number_quadrant_l614_614946


namespace store_profit_is_20_percent_l614_614708

variable (C : ℝ)
variable (marked_up_price : ℝ := 1.20 * C)          -- First markup price
variable (new_year_price : ℝ := 1.50 * C)           -- Second markup price
variable (discounted_price : ℝ := 1.20 * C)         -- Discounted price in February
variable (profit : ℝ := discounted_price - C)       -- Profit on items sold in February

theorem store_profit_is_20_percent (C : ℝ) : profit = 0.20 * C := 
  sorry

end store_profit_is_20_percent_l614_614708


namespace barking_dogs_count_l614_614026

theorem barking_dogs_count (initial_dogs barking_dogs_start : ℕ) (h1 : initial_dogs = 30) (h2 : barking_dogs_start = 10) : 
  initial_dogs + barking_dogs_start = 40 :=
by
  rw [h1, h2]
  sorry

end barking_dogs_count_l614_614026


namespace min_empty_squares_eq_nine_l614_614784

-- Definition of the problem conditions
def chessboard_size : ℕ := 9
def total_squares : ℕ := chessboard_size * chessboard_size
def number_of_white_squares : ℕ := 4 * chessboard_size
def number_of_black_squares : ℕ := 5 * chessboard_size
def minimum_number_of_empty_squares : ℕ := number_of_black_squares - number_of_white_squares

-- Theorem to prove minimum number of empty squares
theorem min_empty_squares_eq_nine :
  minimum_number_of_empty_squares = 9 :=
by
  -- Placeholder for the proof
  sorry

end min_empty_squares_eq_nine_l614_614784


namespace ball_travel_distance_l614_614060

noncomputable def total_distance_after_fifth_bounce (initial_height : ℕ) (rebound_ratio : ℚ) : ℚ :=
  let descent := λ n, initial_height * (rebound_ratio ^ n)
  let ascent := λ n, initial_height * (rebound_ratio ^ (n + 1))
  initial_height + -- first descent
    descent 1 + descent 2 + descent 3 + descent 4 + descent 5 + 
    ascent 0 + ascent 1 + ascent 2 + ascent 3

theorem ball_travel_distance : total_distance_after_fifth_bounce 120 (1 / 3 : ℚ) = 5000 / 27 := by
  sorry

end ball_travel_distance_l614_614060


namespace tangent_line_equation_l614_614312

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem tangent_line_equation :
  let x1 : ℝ := 1
  let y1 : ℝ := f 1
  ∀ x y : ℝ, 
    (y - y1 = (1 / (x1 + 1)) * (x - x1)) ↔ 
    (x - 2 * y + 2 * Real.log 2 - 1 = 0) :=
by
  sorry

end tangent_line_equation_l614_614312


namespace goats_in_caravan_l614_614208

def num_hens := 60
def num_camels := 6
def num_keepers := 10
def feet_per_hen := 2
def feet_per_goat := 4
def feet_per_camel := 4
def feet_per_keeper := 2
def head_count_surplus := 193

theorem goats_in_caravan (G : ℕ) :
  let total_heads := num_hens + G + num_camels + num_keepers,
      total_feet := (num_hens * feet_per_hen) + (G * feet_per_goat) + (num_camels * feet_per_camel) + (num_keepers * feet_per_keeper)
  in total_feet = total_heads + head_count_surplus → G = 35 := 
by 
  sorry

end goats_in_caravan_l614_614208


namespace find_x_l614_614930

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 :=
by
  have h1 : 32 = 2^5 := by norm_num
  rw [h1, pow_mul] at h
  have h2 : 2^(10) = 2^(5*x) := by exact h
  have h3 : 10 = 5 * x := by exact (pow_inj h2).2
  linarith

end find_x_l614_614930


namespace triangle_side_lengths_and_sine_sum_sine_A_plus_pi_six_l614_614561

theorem triangle_side_lengths_and_sine_sum 
  (a b c : ℝ) (A B C : ℝ) (area : ℝ)
  (hC : C = π / 3) (hb : b = 5) (hArea : area = 10 * sqrt 3)
  (hAreaEq : area = (1 / 2) * a * b * sin C)
  (hCosRule : c^2 = a^2 + b^2 - 2 * a * b * cos C)
  (hCosA : A = acos ((b^2 + c^2 - a^2) / (2 * b * c))) : 
  (a = 8) ∧ (c = 7) :=
begin
  sorry
end

theorem sine_A_plus_pi_six 
  (a b c : ℝ) (A C : ℝ) (hC : C = π / 3) (hb : b = 5) (ha : a = 8) 
  (hc : c = 7) : 
  sin (A + π / 6) = 13 / 14 :=
begin
  sorry
end

end triangle_side_lengths_and_sine_sum_sine_A_plus_pi_six_l614_614561


namespace sum_of_ages_l614_614307

variables (P M Mo : ℕ)

def age_ratio_PM := 3 * M = 5 * P
def age_ratio_MMo := 3 * Mo = 5 * M
def age_difference := Mo = P + 64

theorem sum_of_ages : age_ratio_PM P M → age_ratio_MMo M Mo → age_difference P Mo → P + M + Mo = 196 :=
by
  intros h1 h2 h3
  sorry

end sum_of_ages_l614_614307


namespace five_minus_x_eight_l614_614187

theorem five_minus_x_eight (x y : ℤ) (h1 : 5 + x = 3 - y) (h2 : 2 + y = 6 + x) : 5 - x = 8 :=
by
  sorry

end five_minus_x_eight_l614_614187


namespace bridget_initial_skittles_l614_614445

theorem bridget_initial_skittles (b : ℕ) (h : b + 4 = 8) : b = 4 :=
by {
  sorry
}

end bridget_initial_skittles_l614_614445


namespace smallest_sum_of_20_consecutive_integers_is_triangular_l614_614669

theorem smallest_sum_of_20_consecutive_integers_is_triangular : 
  ∃ n, let S := 10 * (2 * n + 19) in S = 190 ∧ ∃ m, S = m * (m + 1) / 2 :=
by sorry

end smallest_sum_of_20_consecutive_integers_is_triangular_l614_614669


namespace population_of_missing_village_eq_945_l614_614323

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

end population_of_missing_village_eq_945_l614_614323


namespace value_of_k_l614_614040

theorem value_of_k :
  ∀ (k : ℝ), (∃ m : ℝ, m = 4/5 ∧ (21 - (-5)) / (k - 3) = m) →
  k = 35.5 :=
by
  intros k hk
  -- Here hk is the proof that the line through (3, -5) and (k, 21) has the same slope as 4/5
  sorry

end value_of_k_l614_614040


namespace M_value_l614_614068

noncomputable def calculate_value_of_M : ℝ :=
  let side_length_of_cube : ℝ := 3
  let surface_area_of_cube : ℝ := 6 * side_length_of_cube^2
  let volume_formula_cylinder (M : ℝ) : ℝ := M * Real.sqrt 6 / Real.sqrt π

  sorry -- proof omitted
  
theorem M_value :
  calculate_value_of_M = 9 * Real.sqrt 6 * π := 
  sorry -- proof omitted

end M_value_l614_614068


namespace prove_AM_gt_MK_and_MK_gt_KC_l614_614843

-- Definitions of the triangle and relevant points
def Triangle (A B C : Type) := ∃ (P Q R : Type), True

-- Assuming some basic geometric properties
variable (A B C : Type) [Triangle A B C]
variable (AB BC AC : ℝ)
variable (AK CM : ℝ) -- lengths for bisectors
variable (K M : Type) -- points K and M on BC and AB respectively

-- Conditions
axiom AB_greater_BC (h1 : AB > BC) -- Condition AB > BC
axiom angle_bisectors (h2 : AK = CM) -- angle bisectors are given

-- Prove the required inequalities
theorem prove_AM_gt_MK_and_MK_gt_KC (h1 : AB > BC) (h2 : AK = CM) :
  -- This line states we have to prove AM > MK
  ∃ (AM MK KC : ℝ), AM > MK ∧
  -- And here we state that MK > KC
  MK > KC := 
  sorry

end prove_AM_gt_MK_and_MK_gt_KC_l614_614843


namespace range_of_sqrt_is_meaningful_l614_614551

theorem range_of_sqrt_is_meaningful (a : ℝ) (h : ∃ x : ℝ, x = sqrt (a - 1)) : a ≥ 1 := by
  sorry

end range_of_sqrt_is_meaningful_l614_614551


namespace max_xi_value_prob_xi_max_prob_dist_xi_expected_value_xi_l614_614953

-- Cards labeled 1, 2, 3
def card_labels : Set ℕ := {1, 2, 3}

-- Define the random variable xi
noncomputable def xi (x y : ℕ) : ℕ :=
  |x - 2| + |y - x|

-- Probability function for distribution
def prob_fun (x y : ℕ) : ℚ :=
  1 / 9

-- Prove the maximum value of xi
theorem max_xi_value : ∃ x y, xi x y = 3 :=
by
  refine ⟨_, _, _⟩;
  -- Plug in the values that give xi = 3, e.g., (1, 3) and (3, 1)
  sorry

-- Prove the probability of xi attaining its maximum value
theorem prob_xi_max : prob_fun 1 3 + prob_fun 3 1 = 2 / 9 :=
by
  -- Since (1, 3) and (3, 1) give xi = 3 and each has probability 1/9
  sorry

-- Prove the probability distribution of xi
theorem prob_dist_xi : 
  (prob_fun 2 2 = 1/9) ∧
  (prob_fun 1 1 + prob_fun 2 1 + prob_fun 2 3 + prob_fun 3 3 = 4/9) ∧
  (prob_fun 1 2 + prob_fun 3 2 = 2/9) ∧
  (prob_fun 1 3 + prob_fun 3 1 = 2/9) :=
by
  sorry

-- Prove the expected value of xi
theorem expected_value_xi : 
  ∑ (x y : ℕ) in card_labels, xi x y * prob_fun x y = 14 / 9 :=
by
  -- Calculation of expected value
  sorry

end max_xi_value_prob_xi_max_prob_dist_xi_expected_value_xi_l614_614953


namespace find_x_l614_614914

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 := 
by 
  {
    sorry
  }

end find_x_l614_614914


namespace translation_correct_l614_614683

/-- Original quadratic function. -/
def original_function (x : ℝ) : ℝ := x^2 + 1

/-- Translated quadratic function (left by 2 units and down by 3 units). -/
def translated_function (x : ℝ) : ℝ := (x + 2)^2 + 1 - 3

theorem translation_correct :
  ∀ x : ℝ, translated_function x = x^2 + 4x + 2 :=
by
  intro x
  unfold translated_function
  unfold original_function
  calc
  (x + 2)^2 + 1 - 3
      = x^2 + 4x + 4 + 1 - 3 : by sorry
  ... = x^2 + 4x + 2       : by sorry

end translation_correct_l614_614683


namespace sector_area_l614_614161

theorem sector_area (r α l S : ℝ) (h1 : l + 2 * r = 8) (h2 : α = 2) (h3 : l = α * r) :
  S = 4 :=
by
  -- Let the radius be 2 as a condition derived from h1 and h2
  have r := 2
  -- Substitute and compute to find S
  have S_calculated := (1 / 2 * α * r * r)
  sorry

end sector_area_l614_614161


namespace locus_of_point_M_l614_614816

noncomputable def locus_problem (M : Type) [HasCoordinates M (ℝ × ℝ)] : Prop :=
  (∃ (x y : ℝ), 3 * x^2 - y^2 = 3 ∧ x > -1) ∨ (∃ (x : ℝ), y = 0 ∧ -1 < x ∧ x < 2)

theorem locus_of_point_M (M : Type) [HasCoordinates M (ℝ × ℝ)] :
  (∃ (x y : ℝ), ∠MBA = 2 * ∠MAB) → locus_problem M :=
sorry

end locus_of_point_M_l614_614816


namespace magnitude_unique_value_l614_614935

-- Define the quadratic equation
def quadratic_eq (z : ℂ) : Prop := z^2 - 6 * z + 20 = 0

-- Statement of the problem
theorem magnitude_unique_value (z : ℂ) (h : quadratic_eq z) : ∃! (m : ℝ), m = complex.abs z :=
by
  sorry

end magnitude_unique_value_l614_614935


namespace sum_x_coords_g2_l614_614657

noncomputable def g (x : ℝ) : ℝ :=
  if h : x ∈ Icc (-4 : ℝ) (-2 : ℝ) then x + 4
  else if h : x ∈ Icc (-2 : ℝ) 0 then -2 * x
  else if h : x ∈ Icc (0 : ℝ) 2 then 2 * x - 2
  else if h : x ∈ Icc (2 : ℝ) 4 then -2 * x + 8
  else 0

theorem sum_x_coords_g2 : ∑ (x : ℝ) in {x | g x = 2}.to_finset, x = 2 :=
by
  sorry

end sum_x_coords_g2_l614_614657


namespace magnitude_of_conjugate_plus_3i_eq_4_l614_614143

theorem magnitude_of_conjugate_plus_3i_eq_4 (z : ℂ) (h : z = (1 - complex.i) / (1 + complex.i)) :
  complex.abs (complex.conj z + 3 * complex.i) = 4 := by
  sorry

end magnitude_of_conjugate_plus_3i_eq_4_l614_614143


namespace prob1_prob2_prob3_prob4_l614_614087

theorem prob1 : (-20) + (-14) - (-18) - 13 = -29 := sorry

theorem prob2 : (-24) * (-1/2 + 3/4 - 1/3) = 2 := sorry

theorem prob3 : (- (49 + 24/25)) * 10 = -499.6 := sorry

theorem prob4 :
  -3^2 + ((-1/3) * (-3) - 8/5 / 2^2) = -8 - 2/5 := sorry

end prob1_prob2_prob3_prob4_l614_614087


namespace C_M_Y_collinear_l614_614271

-- Let ABCD be a square
structure Square (A B C D : Type) :=
  (is_square : true) -- The property that A, B, C, and D form a square

-- There is a point Z on diagonal AC such that AZ > ZC
axiom Z_on_diagonal (A C Z: Type) (AZ ZC: ℝ) : AZ > ZC

-- There is a square AXYZ with vertices in clockwise order
structure Square_AXYZ (A X Y Z : Type) :=
  (is_square : true) -- The property that A, X, Y, and Z form a square

-- Point B lies inside the square AXYZ
axiom B_inside_square (A X Y Z B : Type) : true -- Placeholder for the property that B lies inside AXYZ

-- M is defined as the point of intersection of lines BX and DZ
axiom M_intersection (B X D Z M : Type) : true -- Placeholder for the property that M is the intersection of BX and DZ

-- Prove that C, M, and Y are collinear
theorem C_M_Y_collinear (A B C D X Y Z M : Type)
  [Square A B C D]
  [Z_on_diagonal A C Z (0:ℝ) (0:ℝ)] -- Placeholder for AZ and ZC
  [Square_AXYZ A X Y Z]
  [B_inside_square A X Y Z B]
  [M_intersection B X D Z M] 
  : collinear {C, M, Y} :=
sorry

end C_M_Y_collinear_l614_614271


namespace calc_m_l614_614778

theorem calc_m (m : ℤ) (h : (64 : ℝ)^(1 / 3) = 2^m) : m = 2 :=
sorry

end calc_m_l614_614778


namespace find_angle_C_l614_614591

noncomputable def angleC (A B C a b c S : ℝ) : Prop :=
  A + C = 2 * B ∧
  A + B + C = π ∧
  S = 0.5 * a * b * sin(C) ∧
  sin(A + C) = (2 * S) / (b^2 - c^2) ∧
  sorry -- Additional conditions for side lengths

theorem find_angle_C (A B C a b c S : ℝ) 
  (h1 : A + C = 2 * B)
  (h2 : A + B + C = π)
  (h3 : S = 0.5 * a * b * sin(C))
  (h4 : sin(A + C) = (2 * S) / (b^2 - c^2)):
  C = π / 6 := sorry

end find_angle_C_l614_614591


namespace solve_equation_l614_614638

theorem solve_equation (x : ℝ) (h : x ≠ 3) (hx : x + 2 = 4 / (x - 3)) : 
    x = (1 + Real.sqrt 41) / 2 ∨ x = (1 - Real.sqrt 41) / 2 := by
sorry

end solve_equation_l614_614638


namespace fifty_third_number_is_2_pow_53_l614_614255

theorem fifty_third_number_is_2_pow_53 :
  ∀ n : ℕ, (n = 53) → ∃ seq : ℕ → ℕ, (seq 1 = 2) ∧ (∀ k : ℕ, seq (k+1) = 2 * seq k) ∧ (seq n = 2 ^ 53) :=
  sorry

end fifty_third_number_is_2_pow_53_l614_614255


namespace calculate_expression_l614_614086

theorem calculate_expression :
  ((2023 - Real.pi)^0 + ((1 : ℝ) / 2)^(-1) + abs (1 - Real.sqrt 3) - 2 * Real.sin (Real.pi / 3)) = 2 := by
  sorry

end calculate_expression_l614_614086


namespace hyperbola_foci_distance_l614_614814

theorem hyperbola_foci_distance :
  let equation := (λ x y, 3*x^2 - 18*x - 9*y^2 - 27*y = 81)
  ∃ c : ℝ, c = 6.272 ∧ (2 * c) = 12.544 :=
by
  let a_squared := 29.5
  let b_squared := 9.8333
  let c := Real.sqrt (a_squared + b_squared)
  have hc : c = 6.272 := sorry
  use c
  split
  · exact hc
  · rw [hc]
    norm_num
    exact 12.544

end hyperbola_foci_distance_l614_614814


namespace angle_at_vertex_C_l614_614733

theorem angle_at_vertex_C (A B C P Q R S : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]
  (angle_ABC : ∠B C A = 80)
  (square : IsSquare P Q R S)
  (vertex_condition : P = A)
  (C_on_arc_QR : Midpoint Q R C) :
  ∠A C B = 55 := sorry

end angle_at_vertex_C_l614_614733


namespace trucks_needed_l614_614739

theorem trucks_needed (total_apples transported_apples truck_capacity : ℕ) (h : total_apples = 42) (h1 : transported_apples = 22) (h2 : truck_capacity = 4) : 
  (total_apples - transported_apples) / truck_capacity = 5 := 
by
  rw [h, h1, h2]
  exact sorry

end trucks_needed_l614_614739


namespace tangent_at_x_2_tangent_lines_through_A_l614_614170

def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4
def df := deriv f

noncomputable def tangent_line_at (x₀ : ℝ) : ℝ × ℝ := 
  let m := df x₀
  let y₀ := f x₀
  (-m, 1), -(m * x₀ - y₀)

theorem tangent_at_x_2 : tangent_line_at 2 = ((-1, 1), -4) :=
by sorry

noncomputable def is_tangent_line_through (a : ℝ) :=
  let m := df a
  let y₀ := f a
  (∃ b, tangent_line_at a = b ∧ b (2, -2))

theorem tangent_lines_through_A : is_tangent_line_through 1 ∧ is_tangent_line_through 2 :=
by sorry

end tangent_at_x_2_tangent_lines_through_A_l614_614170


namespace ten_digit_number_difference_l614_614289

-- Definitions based on conditions
def is_ten_digit (n : ℕ) : Prop :=
  n >= 1000000000 ∧ n < 10000000000

def digits_from_0_to_9 (n : ℕ) : Prop :=
  (List.range 10).perm $ Integer.digits 10 n

def divisible_by_11 (n : ℕ) : Prop :=
  (Integer.digits 10 n).zipWithIndex.foldr (λ ⟨d, i⟩ acc => acc + (-1)^i * d) 0 % 11 = 0

-- Proof statement
theorem ten_digit_number_difference :
  ∀ n₁ n₂ : ℕ,
    is_ten_digit n₁ →
    is_ten_digit n₂ →
    digits_from_0_to_9 n₁ →
    digits_from_0_to_9 n₂ →
    divisible_by_11 n₁ →
    divisible_by_11 n₂ →
    (n₁ > n₂ ∧ largest_ten_digit_divisible_by_11 n₁ ∧ smallest_ten_digit_divisible_by_11 n₂) →
    n₁ - n₂ = 8852148261 :=
by
  sorry

end ten_digit_number_difference_l614_614289


namespace decimal_sequence_l614_614806

theorem decimal_sequence :
  ∃ (a b c d : ℝ), 5.45 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < 5.47 ∧ 
  a = 5.451 ∧ b = 5.452 ∧ c = 5.453 ∧ d = 5.454 :=
by
  use [5.451, 5.452, 5.453, 5.454]
  simp
  split; linarith
  split; linarith
  split; linarith
  split; linarith
  split; linarith
sorry

end decimal_sequence_l614_614806


namespace B_in_fourth_quadrant_l614_614218

theorem B_in_fourth_quadrant (a b : ℝ) (h_a : a > 0) (h_b : -b > 0) : (a > 0 ∧ b < 0) := 
  begin
    have h_b_neg : b < 0 := by linarith,
    exact ⟨h_a, h_b_neg⟩,
  end

end B_in_fourth_quadrant_l614_614218


namespace evenFunctionExists_l614_614658

-- Definitions based on conditions
def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def passesThroughPoints (f : ℝ → ℝ) (points : List (ℝ × ℝ)) : Prop :=
  ∀ p ∈ points, f p.1 = p.2

-- Example function
def exampleEvenFunction (x : ℝ) : ℝ := x^2 * (x - 3) * (x + 1)

-- Points to pass through
def givenPoints : List (ℝ × ℝ) := [(-1, 0), (0.5, 2.5), (3, 0)]

-- Theorem to be proven
theorem evenFunctionExists : 
  isEvenFunction exampleEvenFunction ∧ passesThroughPoints exampleEvenFunction givenPoints :=
by
  sorry

end evenFunctionExists_l614_614658


namespace range_of_a_l614_614531

open Real

noncomputable def f (x a : ℝ) : ℝ := ln x - a

theorem range_of_a (a : ℝ) :
  (∀ x > 1, f x a < x ^ 2) ↔ (a ≥ -1) :=
begin
  sorry
end

end range_of_a_l614_614531


namespace area_of_rectangle_with_diagonal_l614_614079

-- Define the dimensions of the rectangle
variables (w d : ℝ)
-- Condition: length is three times the width
def length := 3 * w
-- Pythagorean theorem relating length, width, and diagonal
def diagonal_eq : d^2 = (3 * w)^2 + w^2 := 
by {
  have h1 : (3 * w)^2 = 9 * w^2 := by ring,
  have h2 :  (3 * w)^2 + w^2 = 10 * w^2 := by rw [h1]; ring,
  rw h2
}

-- Solve for w^2 given the diagonal condition from the Pythagorean theorem
def solve_w_squared : w^2 = d^2 / 10 := 
by {
  rw [diagonal_eq],
  field_simp,
  norm_num
}

-- Area of the rectangle
def area : ℝ := 3 * w^2

-- Final proof statement: Area == (3 * d^2) / 10 given the conditions
theorem area_of_rectangle_with_diagonal (d : ℝ) : 
  (w^2 = d^2 / 10) → area = (3 * d^2) / 10 :=
by {
  intro h,
  rw [solve_w_squared],
  rw h,
  norm_num
}

end area_of_rectangle_with_diagonal_l614_614079


namespace tori_passing_additional_questions_l614_614574

theorem tori_passing_additional_questions :
  ∀ (total_problems arithmetic_problems algebra_problems geometry_problems : ℕ)
    (arithmetic_score algebra_score geometry_score : ℚ)
    (passing_mark : ℚ),
  (total_problems = 100) →
  (arithmetic_problems = 20) →
  (algebra_problems = 40) →
  (geometry_problems = 40) →
  (arithmetic_score = 0.80) →
  (algebra_score = 0.35) →
  (geometry_score = 0.55) →
  (passing_mark = 0.65) →
  let correct_arithmetic := arithmetic_score * arithmetic_problems,
      correct_algebra := algebra_score * algebra_problems,
      correct_geometry := geometry_score * geometry_problems,
      total_correct := correct_arithmetic + correct_algebra + correct_geometry,
      correct_needed := passing_mark * total_problems in
  (correct_needed.to_nat - total_correct.to_nat = 13) :=
by
  intros
  sorry

end tori_passing_additional_questions_l614_614574


namespace projection_of_b_onto_a_l614_614891
-- Import the entire library for necessary functions and definitions.

-- Define the problem in Lean 4, using relevant conditions and statement.
theorem projection_of_b_onto_a (m : ℝ) (h : (1 : ℝ) * 3 + (Real.sqrt 3) * m = 6) : m = Real.sqrt 3 :=
by
  sorry

end projection_of_b_onto_a_l614_614891


namespace find_polynomial_l614_614115

noncomputable def satisfies_conditions (P : ℝ → ℝ) :=
∀ a : ℝ, a > 1995 → (∃ n : ℕ, P = (λ x, (x - 1995 : ℝ)^n))

theorem find_polynomial (P : ℝ → ℝ) :
  (∀ a > 1995, ∃ n : ℕ, ∀ x : ℝ, P(x) = a → x > 1995 ∧ (x = x + a) = a. n ) → satisfies_conditions P :=
sorry

end find_polynomial_l614_614115


namespace problem_solution_l614_614261

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def constant_function (f : ℕ → ℕ) : Prop :=
  ∃ c ∈ A, ∀ x ∈ A, f (f (f x)) = c

def num_such_functions : ℕ :=
  8 * (7.choose 1 * 1^6 + 7.choose 2 * 2^5 + 7.choose 3 * 3^4 + 7.choose 4 * 4^3 + 7.choose 5 * 5^2 + 7.choose 6 * 6^1 + 7.choose 7 * 7^0)

theorem problem_solution : (num_such_functions % 1000) = 576 := by
  sorry

end problem_solution_l614_614261


namespace left_handed_jazz_lovers_l614_614956

theorem left_handed_jazz_lovers (total_members left_handed jazz_lovers right_handed_dislike_both : ℕ) :
  total_members = 25 →
  left_handed = 10 →
  jazz_lovers = 18 →
  right_handed_dislike_both = 3 →
  (∀ m, m ∈ {m | m = "left-handed"} ∪ {m | m = "right-handed"}) →
  (∀ m, m ∈ {m | m = "jazz"} ∪ {m | m = "classical"}) →
  (exc : vars) :=
  let x := exc.num_left_hand_jazz
  let y := 0
  exc.num_right_hand_only_classical = 0 →
  (total_members = (x + (left_handed - x - y) + (jazz_lovers - x) + (x - 6) + right_handed_dislike_both)) →
  ∃ num_left_hand_jazz : ℕ, num_left_hand_jazz = 10 :=
  
begin
  sorry
end

end left_handed_jazz_lovers_l614_614956


namespace product_nonreal_roots_l614_614120

theorem product_nonreal_roots :
  let f := λ x : ℂ, x^6 - 6 * x^5 + 15 * x^4 - 20 * x^3 + 15 * x^2 - 6 * x - 729
  let roots := {x : ℂ | f x = 0}
  let nonreal_roots := roots.filter (λ x, x.im ≠ 0)
  ∀ a b ∈ nonreal_roots, (a * b) = 1 := by
sorry

end product_nonreal_roots_l614_614120


namespace suff_not_nec_for_abs_eq_one_l614_614138

variable (m : ℝ)

theorem suff_not_nec_for_abs_eq_one (hm : m = 1) : |m| = 1 ∧ (¬(|m| = 1 → m = 1)) := by
  sorry

end suff_not_nec_for_abs_eq_one_l614_614138


namespace median_name_length_is_4_l614_614470

def name_lengths : List ℕ := [/* Insert the actual 19 lengths of names here in ascending order, e.g., 1, 2, 2, etc. */]

theorem median_name_length_is_4 (lengths : List ℕ) (h_length : lengths.length = 19) (h_sorted : lengths.sorted) :
  lengths.nth (19 / 2) = some 4 :=
by
  sorry

end median_name_length_is_4_l614_614470


namespace polynomial_multiplication_identity_l614_614358

-- Statement of the problem
theorem polynomial_multiplication_identity (x : ℝ) : 
  (25 * x^3) * (12 * x^2) * (1 / (5 * x)^3) = (12 / 5) * x^2 :=
by
  sorry

end polynomial_multiplication_identity_l614_614358


namespace gas_reduction_percentages_l614_614333

theorem gas_reduction_percentages:
  ∀ (init_price : ℝ) (increase1_A increase2_A increase1_B increase2_B increase1_C increase2_C : ℝ), 
  increase1_A = 0.30 → increase2_A = 0.15 →
  increase1_B = 0.25 → increase2_B = 0.10 →
  increase1_C = 0.20 → increase2_C = 0.05 →
  let final_price_A := (init_price * (1 + increase1_A)) * (1 + increase2_A),
      final_price_B := (init_price * (1 + increase1_B)) * (1 + increase2_B),
      final_price_C := (init_price * (1 + increase1_C)) * (1 + increase2_C)
  in (final_price_A = init_price * 1.495) ∧ 
     (final_price_B = init_price * 1.375) ∧ 
     (final_price_C = init_price * 1.26) →
     (∀ (reduction_A reduction_B reduction_C : ℝ),
      reduction_A = (init_price - final_price_A) / final_price_A * 100 ∧
      reduction_B = (init_price - final_price_B) / final_price_B * 100 ∧
      reduction_C = (init_price - final_price_C) / final_price_C * 100 →
      (reduction_A = 33.11) ∧
      (reduction_B = 27.27) ∧
      (reduction_C = 20.63)) := by
  sorry

end gas_reduction_percentages_l614_614333


namespace at_least_8_heads_probability_l614_614357

open BigOperators

noncomputable def ten_flip_coin_probability : ℚ := 
  let total_outcomes := 2^10
  let at_least_8_heads := (nat.choose 10 8) + (nat.choose 10 9) + (nat.choose 10 10)
  at_least_8_heads / total_outcomes

theorem at_least_8_heads_probability :
  ten_flip_coin_probability = 7 / 128 :=
by
  sorry

end at_least_8_heads_probability_l614_614357


namespace coeff_x6_in_expansion_l614_614361

theorem coeff_x6_in_expansion :
  let f := (1 - 3 * x^2)
  polynomial.coeff (f ^ 6) 6 = -540 :=
by
  sorry

end coeff_x6_in_expansion_l614_614361


namespace sculpture_visible_surface_area_l614_614755

-- Define the volumes of the cubes
def volumes := [1, 27, 64, 125, 216, 343]

-- Function to calculate the side length from the volume of a cube
def side_length (v : Nat) : Float := (v.toFloat) ** (1.0 / 3.0)

-- Function to calculate the visible surface area based on conditions in the problem
def visible_surface_area (s : Float) : Float :=
  if s == 7 then 4 * s^2 - s^2 - s^2 / 2
  else 4 * s^2 - s^2 - s^2 / 2

-- Calculate the total visible surface area
def total_visible_surface_area (vols : List Nat) : Float :=
  List.foldl (fun acc v => acc + visible_surface_area (side_length v)) 0 vols

-- The main statement to prove 
theorem sculpture_visible_surface_area :
  total_visible_surface_area volumes = 353.5 := by
  sorry

end sculpture_visible_surface_area_l614_614755


namespace gcd_10010_20020_l614_614474

/-- 
Given that 10,010 can be written as 10 * 1001 and 20,020 can be written as 20 * 1001,
prove that the GCD of 10,010 and 20,020 is 10010.
-/
theorem gcd_10010_20020 : gcd 10010 20020 = 10010 := by
  sorry

end gcd_10010_20020_l614_614474


namespace greatest_award_correct_l614_614382

-- Definitions and constants
def total_prize : ℕ := 600
def num_winners : ℕ := 15
def min_award : ℕ := 15
def prize_fraction_num : ℕ := 2
def prize_fraction_den : ℕ := 5
def winners_fraction_num : ℕ := 3
def winners_fraction_den : ℕ := 5

-- Conditions (translated and simplified)
def num_specific_winners : ℕ := (winners_fraction_num * num_winners) / winners_fraction_den
def specific_prize : ℕ := (prize_fraction_num * total_prize) / prize_fraction_den
def remaining_winners : ℕ := num_winners - num_specific_winners
def min_total_award_remaining : ℕ := remaining_winners * min_award
def remaining_prize : ℕ := total_prize - min_total_award_remaining
def min_award_specific : ℕ := num_specific_winners - 1
def sum_min_awards_specific : ℕ := min_award_specific * min_award

-- Correct answer
def greatest_award : ℕ := remaining_prize - sum_min_awards_specific

-- Theorem statement (Proof skipped with sorry)
theorem greatest_award_correct :
  greatest_award = 390 := sorry

end greatest_award_correct_l614_614382


namespace no_such_n_l614_614100

theorem no_such_n (n : ℕ) (h_pos : 0 < n) :
  ¬ ∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
sorry

end no_such_n_l614_614100


namespace incorrect_statement_D_l614_614700

theorem incorrect_statement_D
  (A : ∀ (X Y : Type) (r : X → Y), ¬(X = X → Y ≅ Y))
  (B : ∀ (data : List (ℝ × ℝ)), ∃ (scatter_plot : List (ℝ × ℝ)), true)
  (C : ∀ (data : List (ℝ × ℝ)), ∃ (reg_line : ℝ → ℝ), ∀ (x y : ℝ), true) :
  ¬∀ (data : List (ℝ × ℝ)), ∃ (reg_eq : ℝ → ℝ), true := sorry

end incorrect_statement_D_l614_614700


namespace ACP_angle_l614_614298

theorem ACP_angle (A B C D P : EuclideanGeometry.Point)
  (h1 : EuclideanGeometry.midpoint A B C)
  (h2 : EuclideanGeometry.midpoint B C D)
  (h3 : ∃ (semicircle1 semicircle2 : EuclideanGeometry.Circle), 
        semicircle1.diameter = EuclideanGeometry.segment A B ∧ 
        semicircle2.diameter = EuclideanGeometry.segment B C)
  (h4 : EuclideanGeometry.bisects_area CP (semicircle1 ∪ semicircle2)) :
  EuclideanGeometry.angle A C P = 112.5 :=
begin
  sorry
end

end ACP_angle_l614_614298


namespace sliced_cone_volume_ratio_l614_614421

theorem sliced_cone_volume_ratio (h R : ℝ) (h_pos : 0 < h) (R_pos : 0 < R) :
  let V4 := (1/3) * Real.pi * ((4/5) * R)^2 * h in
  let V5 := (1/3) * Real.pi * R^2 * (2 * h) in
  V5 / V4 = 25 / 8 :=
by
  sorry

end sliced_cone_volume_ratio_l614_614421


namespace three_digit_not_multiple_of_3_5_7_l614_614908

theorem three_digit_not_multiple_of_3_5_7 : 
  (900 - (let count_mult_3 := 300 in
           let count_mult_5 := 180 in
           let count_mult_7 := 128 in
           let count_mult_15 := 60 in
           let count_mult_21 := 43 in
           let count_mult_35 := 26 in
           let count_mult_105 := 9 in
           let total_mult_3_5_or_7 := 
             count_mult_3 + count_mult_5 + count_mult_7 - 
             (count_mult_15 + count_mult_21 + count_mult_35) +
             count_mult_105 in
           total_mult_3_5_or_7)) = 412 :=
by {
  -- The mathematical calculations were performed above
  -- The proof is represented by 'sorry' indicating the solution is skipped
  sorry
}

end three_digit_not_multiple_of_3_5_7_l614_614908


namespace no_reverse_pascal_triangle_l614_614028

theorem no_reverse_pascal_triangle (n : ℕ) (h : n = 2018) : 
  ¬ (∃ (a : ℕ → ℕ → ℕ), 
    (∀ i j, 1 ≤ i ∧ i < n → 1 ≤ j ∧ j ≤ i → 
      a i j = | a (i + 1) j - a (i + 1) (j + 1) |) ∧ 
    (Finset.univ.sum (λ x, Finset.univ (Fin x).sum (λ y, a x y)) 
      = (n * (n + 1)) / 2) ∧ 
    ∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ i → 
      a i j ∈ Finset.range ((n + 1) * n / 2))) :=
begin
  sorry
end

end no_reverse_pascal_triangle_l614_614028


namespace number_of_songs_l614_614212

-- Definitions for the participants and their respective songs
def Rose_participation := 3
def Sue_participation := 6
def Beth_participation := ∃ b : ℕ, b ∈ {4, 5}
def Lila_participation := ∃ l : ℕ, l ∈ {4, 5}

-- The proof problem structured in Lean 4 statement.
theorem number_of_songs (b l : ℕ) (hb : b ∈ {4, 5}) (hl : l ∈ {4, 5}) 
  (Rose_participation := 3) (Sue_participation := 6) :
  (6 + 3 + b + l) / 3 = 6 :=
by
  have h_valid : 6 + 3 + b + l = 18, sorry
  have h_divisible : 18 / 3 = 6, sorry
  exact h_divisible

end number_of_songs_l614_614212


namespace tangency_of_circumcircle_l614_614269

variables {A B C A' B' C' D M N : Point}
variables {ω : Circle}
variables [Triangle ABC]
variables [Incircle ω ABC A' B' C']
variables [Altitude D A BC]
variables [Midpoint M AD]
variables [SecondIntersection N A'M ω]

theorem tangency_of_circumcircle (h : ∠BNC.TangentAt N ω) :
  (Circle.BNC.TangentAt N ω) := sorry

end tangency_of_circumcircle_l614_614269


namespace possible_values_of_b_l614_614630

theorem possible_values_of_b (r s : ℝ) (t t' : ℝ)
  (hp : ∀ x, x^3 + a * x + b = 0 → (x = r ∨ x = s ∨ x = t))
  (hq : ∀ x, x^3 + a * x + b + 240 = 0 → (x = r + 4 ∨ x = s - 3 ∨ x = t'))
  (h_sum_p : r + s + t = 0)
  (h_sum_q : (r + 4) + (s - 3) + t' = 0)
  (ha_p : a = r * s + r * t + s * t)
  (ha_q : a = (r + 4) * (s - 3) + (r + 4) * (t' - 1) + (s - 3) * (t' - 1))
  (ht'_def : t' = t - 1)
  : b = -330 ∨ b = 90 :=
by
  sorry

end possible_values_of_b_l614_614630


namespace value_of_x_for_fn_inv_eq_l614_614454

def f (x : ℝ) : ℝ := 4 * x - 9
def f_inv (x : ℝ) : ℝ := (x + 9) / 4

theorem value_of_x_for_fn_inv_eq (x : ℝ) : f(x) = f_inv(x) → x = 3 :=
by
  sorry

end value_of_x_for_fn_inv_eq_l614_614454


namespace length_of_AB_l614_614967

noncomputable def radius : ℝ := Real.sqrt 7
noncomputable def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry
noncomputable def C : ℝ × ℝ := sorry

-- Given points A, B, and C are on the circle with radii from O
axiom hA : Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) = radius
axiom hB : Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2) = radius
axiom hC : Real.sqrt ((C.1 - O.1)^2 + (C.2 - O.2)^2) = radius

-- Given angle BOC = 120 degrees
axiom angle_BOC : ∃ θ : ℝ, θ = 120 * Real.pi / 180 ∧ θ = angle_between O B O C

-- Given AC = AB + 1
axiom AC_eq_AB_plus_1 : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 1

-- To prove: length of AB = 4
theorem length_of_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 :=
by
  sorry

end length_of_AB_l614_614967


namespace sum_first_15_odd_from_5_l614_614007

theorem sum_first_15_odd_from_5 : 
  let a₁ := 5 
  let d := 2 
  let n := 15 
  let a₁₅ := a₁ + (n - 1) * d 
  let S := n * (a₁ + a₁₅) / 2 
  S = 285 := by 
  sorry

end sum_first_15_odd_from_5_l614_614007


namespace vincent_total_is_20_l614_614686

theorem vincent_total_is_20 (a b c d : ℝ) (round_to_nearest : ℝ → ℤ) :
  a = 3.37 → b = 8.75 → c = 2.49 → d = 6.01 →
  round_to_nearest a + round_to_nearest b + round_to_nearest c + round_to_nearest d = 20 :=
by
  intros ha hb hc hd
  have ha1 : round_to_nearest a = 3 := sorry
  have hb1 : round_to_nearest b = 9 := sorry
  have hc1 : round_to_nearest c = 2 := sorry
  have hd1 : round_to_nearest d = 6 := sorry
  rw [ha1, hb1, hc1, hd1]
  exact rfl
  sorry

end vincent_total_is_20_l614_614686


namespace highest_score_l614_614647

-- Definitions based on conditions
variable (H L : ℕ)

-- Condition (1): H - L = 150
def condition1 : Prop := H - L = 150

-- Condition (2): H + L = 208
def condition2 : Prop := H + L = 208

-- Condition (3): Total runs in 46 innings at an average of 60, excluding two innings averages to 58
def total_runs := 60 * 46
def excluded_runs := total_runs - 2552

theorem highest_score
  (cond1 : condition1 H L)
  (cond2 : condition2 H L)
  : H = 179 :=
by sorry

end highest_score_l614_614647


namespace new_wattage_l614_614744

theorem new_wattage (original_wattage : ℝ) (percentage_increase : ℝ) (new_wattage : ℝ) :
  original_wattage = 60 → percentage_increase = 0.12 → new_wattage = original_wattage * (1 + percentage_increase) →
  new_wattage = 67.2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end new_wattage_l614_614744


namespace no_partition_equal_product_l614_614101

theorem no_partition_equal_product (n : ℕ) (h_pos : 0 < n) :
  ¬∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧
  A.prod id = B.prod id := sorry

end no_partition_equal_product_l614_614101


namespace magnitude_unique_value_l614_614933

-- Define the quadratic equation
def quadratic_eq (z : ℂ) : Prop := z^2 - 6 * z + 20 = 0

-- Statement of the problem
theorem magnitude_unique_value (z : ℂ) (h : quadratic_eq z) : ∃! (m : ℝ), m = complex.abs z :=
by
  sorry

end magnitude_unique_value_l614_614933


namespace smallest_positive_period_of_f_l614_614462

def f (x : ℝ) : ℝ := 2 * |sin x|

theorem smallest_positive_period_of_f : ∃ T > 0, T = π ∧ ∀ x, f(x + T) = f(x) :=
by
  sorry

end smallest_positive_period_of_f_l614_614462


namespace decreasing_function_range_of_a_l614_614499

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem decreasing_function_range_of_a :
  (∀ x y : ℝ, x < y → f a y ≤ f a x) ↔ (1/7 ≤ a ∧ a < 1/3) :=
by
  sorry

end decreasing_function_range_of_a_l614_614499


namespace correctAttitudeTowardsTraditionalCulture_l614_614964

-- Definitions for the attitudes towards traditional culture
def AbsorbEssenceDiscardDross : Prop := 
  "Absorb its essence and discard the dross"

def FaceWorldLearnStrengths : Prop :=
  "Face the world and learn from the strengths of others"

def PrioritizeSelfBenefit : Prop :=
  "Prioritize ourselves and use it for our benefit"

def CriticallyInheritApply : Prop :=
  "Critically inherit and apply the ancient wisdom to the present"

-- Proving that the correct attitude involves combining ① and ②
theorem correctAttitudeTowardsTraditionalCulture :
  AbsorbEssenceDiscardDross ∧ FaceWorldLearnStrengths ∧ ¬(PrioritizeSelfBenefit ∨ CriticallyInheritApply) :=
by
  sorry

end correctAttitudeTowardsTraditionalCulture_l614_614964


namespace more_than_500_correct_not_less_than_999_correct_l614_614342

-- Definition of the conditions
def wise_men_hats (n : ℕ) := 1000
def hat_numbers : set ℕ := set.Icc 1 1001
def hidden_hat : ℕ := 1 -- assuming 1 is the hidden hat for simplicity
def visible_hats (k : ℕ) : set ℕ := hat_numbers \ {hidden_hat}

-- Statements to be proven
theorem more_than_500_correct (strategy : ℕ → ℕ) : 
  ∃ m > 500, m = wise_men_hats 1000 ∧ (∀ k < 1000, strategy k ∈ visible_hats k) := sorry

theorem not_less_than_999_correct (strategy : ℕ → ℕ) : 
  ∃ m ≥ 999, m = wise_men_hats 1000 ∧ (∀ k < 1000, strategy k ∈ visible_hats k) := sorry

end more_than_500_correct_not_less_than_999_correct_l614_614342


namespace inequality_range_of_a_l614_614197

theorem inequality_range_of_a (x y : ℝ) (a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) :
  ( (ax^2 + 2y^2)/ (x*y) - 1 > 0 ) ↔ (a > -1) := by
  sorry

end inequality_range_of_a_l614_614197


namespace choose_president_vice_president_and_committee_l614_614217

theorem choose_president_vice_president_and_committee :
  let num_ways : ℕ := 10 * 9 * (Nat.choose 8 2)
  num_ways = 2520 :=
by
  sorry

end choose_president_vice_president_and_committee_l614_614217


namespace count_mk_lt_alpha_l614_614145

theorem count_mk_lt_alpha (n : ℕ) (a : ℕ → ℕ) (α : ℝ) (hα : α > 0) :
  ∑ i in range (n + 1), a i > α * (count (λ k, let m_k :=
    max (λ l, l ≤ k → (∑ j in range (k - l + 1, k + 1), a j) / (l + 1)) in m_k > α) (range (n + 1))) :=
sorry

end count_mk_lt_alpha_l614_614145


namespace find_a100_l614_614970

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ (∀ n, a (n + 1) = a n + n)

theorem find_a100 (a : ℕ → ℕ) (h : sequence a) : a 100 = 4952 := sorry

end find_a100_l614_614970


namespace has_zero_in_intervals_l614_614275

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x
noncomputable def f' (x : ℝ) : ℝ := (1 / 3) - (1 / x)

theorem has_zero_in_intervals : 
  (∃ x : ℝ, 0 < x ∧ x < 3 ∧ f x = 0) ∧ (∃ x : ℝ, 3 < x ∧ f x = 0) :=
sorry

end has_zero_in_intervals_l614_614275


namespace total_games_in_season_is_16715_l614_614962

structure League :=
  (teams : ℕ)
  (sub_leagues : ℕ)
  (teams_per_sub_league : ℕ)
  (games_per_pair_regular_season : ℕ)
  (teams_advance_intermediate_round : ℕ)
  (teams_advance_playoff_round : ℕ)
  (games_per_pair_playoff_round : ℕ)

def league : League :=
  { teams := 200,
    sub_leagues := 10,
    teams_per_sub_league := 20,
    games_per_pair_regular_season := 8,
    teams_advance_intermediate_round := 5,
    teams_advance_playoff_round := 2,
    games_per_pair_playoff_round := 2 }

noncomputable def number_of_games_played (league : League) : ℕ := 
  let num_games_regular_season := (league.teams_per_sub_league * (league.teams_per_sub_league - 1) / 2) * league.games_per_pair_regular_season * league.sub_leagues in
  let num_games_intermediate_round := (league.teams_advance_intermediate_round * league.sub_leagues * (league.teams_advance_intermediate_round * league.sub_leagues - 1)) / 2 in
  let num_games_playoff_round := 
    ((league.teams_advance_playoff_round * league.sub_leagues * (league.teams_advance_playoff_round * league.sub_leagues - 1)) / 2 
    - (league.teams_per_sub_league * (league.teams_per_sub_league - 1) / 2) * league.sub_leagues) * league.games_per_pair_playoff_round in
  num_games_regular_season + num_games_intermediate_round + num_games_playoff_round

theorem total_games_in_season_is_16715 : number_of_games_played league = 16715 :=
  by
  sorry

end total_games_in_season_is_16715_l614_614962


namespace monotonic_increasing_f_C_l614_614378

noncomputable def f_A (x : ℝ) : ℝ := -Real.log x
noncomputable def f_B (x : ℝ) : ℝ := 1 / (2^x)
noncomputable def f_C (x : ℝ) : ℝ := -(1 / x)
noncomputable def f_D (x : ℝ) : ℝ := 3^(abs (x - 1))

theorem monotonic_increasing_f_C : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f_C x < f_C y :=
sorry

end monotonic_increasing_f_C_l614_614378


namespace problem_1_problem_2_l614_614135

-- Define proposition p
def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

-- Define proposition q
def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Define the range of values for a in proposition p
def range_p (a : ℝ) : Prop :=
  a ≤ 1

-- Define set A and set B
def set_A (a : ℝ) : Prop := a ≤ 1
def set_B (a : ℝ) : Prop := a ≥ 1 ∨ a ≤ -2

theorem problem_1 (a : ℝ) (h : proposition_p a) : range_p a := 
sorry

theorem problem_2 (a : ℝ) : 
  (∃ h1 : proposition_p a, set_A a) ∧ (∃ h2 : proposition_q a, set_B a)
  ↔ ¬ ((∃ h1 : proposition_p a, set_B a) ∧ (∃ h2 : proposition_q a, set_A a)) :=
sorry

end problem_1_problem_2_l614_614135


namespace area_of_rhombus_eq_four_l614_614239

-- Define the situation and properties
variables {A B C D P Q : Type}

-- Assume AB CD is a rhombus
variables [comm_ring AB] [comm_ring CD]
variables [comm_ring P] [comm_ring Q]

-- Given conditions
def rhombus (AB CD : Type) : Prop :=
∀ (A B C D : Type), is_rhombus A B C D

def circle_around_triangles (ABC BCD : Type) : Prop :=
∀ (A B C D : Type), circumscribed_circle A B C ∧ circumscribed_circle B C D

def ray_intersections (BA PD : Type) : Prop :=
∀ (A B C D P Q : Type), 
  (ray_intersects BA P) ∧ (ray_intersects PD Q)

def distances (PD DQ : Type) : Prop :=
∀ (D P Q : Type), 
  (distance PD 1) ∧ (distance DQ (2 + real.sqrt 3))

-- The theorem to be proved
theorem area_of_rhombus_eq_four 
  (h1 : rhombus AB CD)
  (h2 : circle_around_triangles ABC BCD)
  (h3 : ray_intersections BA PD)
  (h4 : distances PD DQ) : 
  area AB CD = 4 :=
sorry

end area_of_rhombus_eq_four_l614_614239


namespace perpendicular_line_eq_l614_614057

theorem perpendicular_line_eq (a b : ℝ) (ha : 2 * a - 5 * b + 3 = 0) (hpt : a = 2 ∧ b = -1) : 
    ∃ c : ℝ, c = 5 * a + 2 * b - 8 := 
sorry

end perpendicular_line_eq_l614_614057


namespace verify_lines_and_planes_l614_614540

noncomputable def lines_and_planes (a b : Line) (alpha beta : Plane) : Prop :=
  ((a.parallel b ∧ a.parallel alpha) → b.parallel alpha) ∧
  ((alpha.perpendicular beta ∧ a.parallel alpha) → a.perpendicular beta) ∧
  ((alpha.perpendicular beta ∧ a.perpendicular beta) → a.parallel alpha) ∧
  ((a.perpendicular b ∧ a.perpendicular alpha ∧ b.perpendicular beta) → alpha.perpendicular beta)

theorem verify_lines_and_planes (a b : Line) (alpha beta : Plane) : lines_and_planes a b alpha beta := 
by
  -- Proof goes here
  sorry

end verify_lines_and_planes_l614_614540


namespace find_x_eq_2_l614_614924

theorem find_x_eq_2 : ∀ x : ℝ, 2^10 = 32^x → x = 2 := 
by 
  intros x h
  sorry

end find_x_eq_2_l614_614924


namespace find_angle_C_l614_614731

theorem find_angle_C 
  (A B C M Q : Point) 
  (hK : K.center = Q ∧ Q ∈ circumcircle ABC ∧ K.passes_through A ∧ K.passes_through C ∧ K.intersects_extension B A M)
  (hMA_AB : MA / AB = 2 / 5)
  (hAngleB : ∠B = arcsin (3 / 5)) : 
  ∠C = 45 :=
sorry

end find_angle_C_l614_614731


namespace find_x_l614_614931

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 :=
by
  have h1 : 32 = 2^5 := by norm_num
  rw [h1, pow_mul] at h
  have h2 : 2^(10) = 2^(5*x) := by exact h
  have h3 : 10 = 5 * x := by exact (pow_inj h2).2
  linarith

end find_x_l614_614931


namespace ruined_tomatoes_percentage_l614_614318

noncomputable def ruined_percentage (W : ℝ) (cost_per_pound : ℝ) (selling_price_per_pound : ℝ) (desired_profit_rate : ℝ): ℝ :=
  let total_cost := cost_per_pound * W
  let desired_profit := desired_profit_rate * total_cost
  let revenue := selling_price_per_pound * (1 - P) * W
  let required_revenue := total_cost + desired_profit
  (required_revenue / selling_price_per_pound) / W

theorem ruined_tomatoes_percentage (W : ℝ) (cost_per_pound : ℝ) (selling_price_per_pound : ℝ) (desired_profit_rate : ℝ) : 
  ruined_percentage W cost_per_pound selling_price_per_pound desired_profit_rate = 0.1 :=
  by
  sorry

end ruined_tomatoes_percentage_l614_614318


namespace man_l614_614747

theorem man's_speed_downstream (v : ℝ) (speed_of_stream : ℝ) (speed_upstream : ℝ) : 
  speed_upstream = v - speed_of_stream ∧ speed_of_stream = 1.5 ∧ speed_upstream = 8 → v + speed_of_stream = 11 :=
by
  sorry

end man_l614_614747


namespace trebled_principal_after_5_years_l614_614325

theorem trebled_principal_after_5_years 
(P R : ℝ) (T total_interest : ℝ) (n : ℝ) 
(h1 : T = 10) 
(h2 : total_interest = 800) 
(h3 : (P * R * 10) / 100 = 400) 
(h4 : (P * R * n) / 100 + (3 * P * R * (10 - n)) / 100 = 800) :
n = 5 :=
by
-- The Lean proof will go here
sorry

end trebled_principal_after_5_years_l614_614325


namespace slope_product_ellipse_l614_614854

def A := (2, 0)
def B := (-2, 0)

def on_ellipse (M : ℝ × ℝ) : Prop :=
  M.1 ^ 2 / 4 + M.2 ^ 2 / 3 = 1

def valid_M (M : ℝ × ℝ) : Prop :=
  on_ellipse M ∧ M ≠ (2, 0) ∧ M ≠ (-2, 0)

def calculate_slope (P Q : ℝ × ℝ) : ℝ :=
  (P.2 - Q.2) / (P.1 - Q.1)

theorem slope_product_ellipse (M : ℝ × ℝ) (h : valid_M M) :
  calculate_slope M A * calculate_slope M B = -3 / 4 :=
by sorry

end slope_product_ellipse_l614_614854


namespace complementary_events_l614_614412

open Set

-- Define the total number of male and female students
def male_students : ℕ := 3
def female_students : ℕ := 2

-- Define the event of selecting 2 students such that it includes at least 1 female student
def at_least_one_female (s : Finset (Fin (male_students + female_students))) : Prop :=
  ∃ (a : Fin male_students), a ∈ s ∧ a.val < male_students

-- Define the event of selecting 2 students such that it includes all male students
def all_male (s : Finset (Fin (male_students + female_students))) : Prop :=
  ∀ (a : Fin male_students), a ∈ s

theorem complementary_events :
  ∃ s, at_least_one_female s ∧ ¬ all_male s :=
sorry

end complementary_events_l614_614412


namespace sum_f_1_to_23_l614_614849

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom h_symm : ∀ x : ℝ, g(x) = g(2 - x)
axiom h_fg_eq : ∀ x : ℝ, f(x) - g(x) = 1
axiom h_fgp1 : ∀ x : ℝ, f(x + 1) + g(2 - x) = 1
axiom h_g1 : g(1) = 3

theorem sum_f_1_to_23 : (∑ i in finset.range 23, f (i + 1)) = 26 :=
by
  sorry

end sum_f_1_to_23_l614_614849


namespace scientific_notation_to_standard_form_l614_614196

theorem scientific_notation_to_standard_form :
  - 3.96 * 10^5 = -396000 :=
sorry

end scientific_notation_to_standard_form_l614_614196


namespace remainder_is_3_l614_614122

-- Define the polynomial p(x)
def p (x : ℝ) := x^3 - 3 * x + 5

-- Define the divisor d(x)
def d (x : ℝ) := x - 1

-- The theorem: remainder when p(x) is divided by d(x)
theorem remainder_is_3 : p 1 = 3 := by 
  sorry

end remainder_is_3_l614_614122


namespace distinct_solutions_square_l614_614932

theorem distinct_solutions_square (α β : ℝ) (h₁ : α ≠ β)
    (h₂ : α^2 = 2 * α + 2 ∧ β^2 = 2 * β + 2) : (α - β) ^ 2 = 12 := by
  sorry

end distinct_solutions_square_l614_614932


namespace salary_unspent_fraction_l614_614385

theorem salary_unspent_fraction (S : ℝ) : 
  let spent_first_week := S / 4 in
  let spent_next_three_weeks := 3 * (S / 5) in
  let total_spent := spent_first_week + spent_next_three_weeks in
  S - total_spent = 3 / 20 * S :=
by
  let spent_first_week := S / 4
  let spent_next_three_weeks := 3 * (S / 5)
  let total_spent := spent_first_week + spent_next_three_weeks
  have h1 : total_spent = 17 / 20 * S := 
    sorry -- Steps to show 17/20 S
  have h2 : S - total_spent = 3 / 20 * S :=
    sorry -- Steps to derive 3/20 S
  exact h2

end salary_unspent_fraction_l614_614385


namespace product_of_slopes_constant_area_triangle_OMN_l614_614147

section
variables {x y : ℝ}

/-- Given conditions for an ellipse -/
def ellipse (x y : ℝ) : Prop := (x^2 / 3 + y^2 / 2 = 1)

/-- Product of slopes of lines PA and PB is a constant -/
theorem product_of_slopes_constant (x₀ y₀ : ℝ) (h : ellipse x₀ y₀) :
  let A := (-real.sqrt 3, 0)
  let B := (real.sqrt 3, 0)
  let slope_PA := y₀ / (x₀ + real.sqrt 3)
  let slope_PB := y₀ / (x₀ - real.sqrt 3)
  (slope_PA * slope_PB = -2 / 3) := sorry

/-- Area of triangle OMN given OM parallel to PA and ON parallel to PB -/
theorem area_triangle_OMN (M N : ℝ × ℝ) (h₁ : ellipse M.1 M.2) (h₂ : ellipse N.1 N.2)
  (OM_parallel_PA : (M.2 / M.1 = y / (x + real.sqrt 3))) 
  (ON_parallel_PB : (N.2 / N.1 = y / (x - real.sqrt 3))) :
  let O := (0, 0)
  let area := (real.sqrt 6) / 2
  (abs ((O.1 - M.1) * (O.2 - N.2) - (O.2 - M.2) * (O.1 - N.1)) / 2 = area) := sorry

end

end product_of_slopes_constant_area_triangle_OMN_l614_614147


namespace lambda_range_l614_614537

noncomputable def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 1 else
  sequence_a (n - 1) / (sequence_a (n - 1) + 2)

noncomputable def sequence_b (lambda : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then -3/2 * lambda else
  (n - 2 * lambda) * (1 / sequence_a (n - 1) + 1)

def is_monotonically_increasing (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → seq (n+1) > seq n

theorem lambda_range (lambda : ℝ) (hn : is_monotonically_increasing (sequence_b lambda)) : lambda < 4/5 := sorry

end lambda_range_l614_614537


namespace max_crates_carried_l614_614063

theorem max_crates_carried
  (weight_per_crate : ℕ)
  (max_weight : ℕ)
  (h1 : 1250 ≤ weight_per_crate)
  (h2 : max_weight = 6250) :
  ∃ n : ℕ, n = max_weight / weight_per_crate ∧ n = 5 :=
begin
  sorry,
end

end max_crates_carried_l614_614063


namespace problem1_solution_set_problem2_range_of_a_l614_614533

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|

-- Problem 1: Solution set for f(x) >= 2
theorem problem1_solution_set : set_of (λ x, f x ≥ 2) = set_of (λ x, x ≤ 0) ∪ set_of (λ x, x ≥ 2/3) :=
by sorry

-- Problem 2: Range of values for a
theorem problem2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x < a) : a ≤ 3/2 :=
by sorry

end problem1_solution_set_problem2_range_of_a_l614_614533


namespace min_omega_value_l614_614155

theorem min_omega_value (ω : ℝ) (h : ω > 0) :
  let f := λ x, sin (ω * x + π / 3) + 2 in
  (∀ x, f (x - 4 * π / 3) = f x) → ω = 3 / 2 :=
begin
  intros f h_eq,
  sorry,
end

end min_omega_value_l614_614155


namespace incorrect_statement_D_l614_614701

theorem incorrect_statement_D :
  ¬ (-3 = real.sqrt (real.sqrt ((-3)^2))) :=
by
  sorry

end incorrect_statement_D_l614_614701


namespace relationship_y_values_l614_614162

theorem relationship_y_values (m y_1 y_2 y_3 : ℝ) :
  (∀ x y, (x, y) ∈ {(-1, y_1), (-2, y_2), (-4, y_3)} → y = -2*x^2 - 8*x + m) →
  y_3 < y_1 ∧ y_1 < y_2 :=
by 
  sorry

end relationship_y_values_l614_614162


namespace cos_seventeen_pi_over_six_l614_614782

noncomputable def cos_pi_over_six : ℝ := Real.cos (π / 6)

theorem cos_seventeen_pi_over_six : Real.cos (17 * π / 6) = - cos_pi_over_six :=
by
  -- periodic property: cos(x + 2π) = cos(x)
  have h1 : Real.cos (17 * π / 6) = Real.cos ((17 * π / 6) - 2 * π), from sorry,
  -- simplifying the expression (17π/6 - 2π)
  have h2 : (17 * π / 6) - 2 * π = 5 * π / 6, from sorry,
  -- angle subtraction formula: cos(π - x) = - cos(x)
  have h3 : Real.cos (5 * π / 6) = Real.cos (π - π / 6), from sorry,
  -- applying cos(π - x) = - cos(x)
  have h4 : Real.cos (π - π / 6) = - Real.cos (π / 6), from sorry,
  -- substitution of the known value
  have h5 : Real.cos (π / 6) = cos_pi_over_six, from sorry,
  exact eq.trans h1 (eq.trans (eq.symm h2) (eq.trans (eq.symm h3) h4))


end cos_seventeen_pi_over_six_l614_614782


namespace prove_perpendicular_l614_614240

noncomputable def rectangle := Type  -- Define the type for rectangle to be used further

-- Define conditions
structure MyConditions (ABCD : rectangle) :=
  (M : Point)
  (A B : Point)
  (arc_AB : part_circle)
  (is_on_arc : M ∈ arc_AB)
  (is_not_vertex_A : M ≠ A)
  (is_not_vertex_B : M ≠ B)
  (P Q R S : Point)
  (proj_P : Proj M AD P)
  (proj_Q : Proj M AB Q)
  (proj_R : Proj M BC R)
  (proj_S : Proj M CD S)

-- Define the theorem statement
theorem prove_perpendicular (ABCD : rectangle) (h : MyConditions ABCD) :
  isPerpendicular (line h.P h.Q) (line h.R h.S) := 
by
  sorry

end prove_perpendicular_l614_614240


namespace first_prime_year_with_digit_sum_8_l614_614205

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem first_prime_year_with_digit_sum_8 :
  ∃ y : ℕ, y > 2015 ∧ sum_of_digits y = 8 ∧ is_prime y ∧
  ∀ z : ℕ, z > 2015 ∧ sum_of_digits z = 8 ∧ is_prime z → y ≤ z :=
sorry

end first_prime_year_with_digit_sum_8_l614_614205


namespace number_of_irrationals_l614_614437

theorem number_of_irrationals : 
  let a := 15
  let b := 22 / 7
  let c := 3 * Real.sqrt 2
  let d := -3 * Real.pi
  let e := 0.10101
  (2 = [c, d].filter (λ x, ¬ Rational.isRat x)).length :=
by
  sorry

end number_of_irrationals_l614_614437


namespace earthquake_intensity_comparison_l614_614306

theorem earthquake_intensity_comparison (x y z : ℝ)
  (h1 : log 10 x = 8.9)
  (h2 : log 10 y = 8.3)
  (h3 : log 10 z = 7.1)
  (h4 : log 10 2 = 0.3) :
  x = 4 * y ∧ x = 64 * z :=
by
  -- Calculation details will come here
  sorry

end earthquake_intensity_comparison_l614_614306


namespace hyperbola_equation_l614_614880

theorem hyperbola_equation (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a^2 + b^2 = 4) (h₄ : a / b = Real.sqrt 3) :
  (d : ℝ) (h : d ∈ ({d : ℝ | ∃ i j : ℝ, i^2 - j^2 = 1 ∧ j ∈ ({a * x | x : ℝ} ∪ {b * x | x : ℝ})})) :
    d = (y^2 / 3 - x^2 = 1) := 
sorry

end hyperbola_equation_l614_614880


namespace total_shoes_tried_on_l614_614901

variable (T : Type)
variable (store1 store2 store3 store4 : T)
variable (pair_of_shoes : T → ℕ)
variable (c1 : pair_of_shoes store1 = 7)
variable (c2 : pair_of_shoes store2 = pair_of_shoes store1 + 2)
variable (c3 : pair_of_shoes store3 = 0)
variable (c4 : pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3))

theorem total_shoes_tried_on (store1 store2 store3 store4 : T) (pair_of_shoes : T → ℕ) : 
  pair_of_shoes store1 = 7 →
  pair_of_shoes store2 = pair_of_shoes store1 + 2 →
  pair_of_shoes store3 = 0 →
  pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3) →
  pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3 + pair_of_shoes store4 = 48 := by
  intro c1 c2 c3 c4
  sorry

end total_shoes_tried_on_l614_614901


namespace coefficient_of_x_in_expansion_coefficient_of_linear_term_l614_614134

noncomputable def integral_value : ℝ :=
  2 * (sin (π + π / 6) - sin (π / 6))

def a := integral_value

theorem coefficient_of_x_in_expansion :
  a = 2 * ∫ (x : ℝ) in 0..π, cos (x + π / 6) :=
by 
  unfold a
  unfold integral_value
  sorry

theorem coefficient_of_linear_term :
  a = -2 → 
  let expr := (λ (x : ℝ), (x^2 + a / x)) in
  let expanded := (expr x)^5 in
  ∑ (r : ℕ) in finset.range 6,
    (finset.choose 5 r) * (x^2)^(5 - r) * (a / x)^r = -80 :=
by 
  unfold a
  intros h
  have ha : a = -2, from h,
  sorry

end coefficient_of_x_in_expansion_coefficient_of_linear_term_l614_614134


namespace trajectory_eq_l614_614394

theorem trajectory_eq (M : Type) [MetricSpace M] : 
  (∀ (r x y : ℝ), (x + 2)^2 + y^2 = (r + 1)^2 ∧ |x - 1| = 1 → y^2 = -8 * x) :=
by sorry

end trajectory_eq_l614_614394


namespace count_positive_integers_satisfying_inequality_l614_614185

theorem count_positive_integers_satisfying_inequality :
  ∃ (n : ℕ), (n = 10) ∧ ∀ k : ℕ, (k > 0) → (k + 9) * (k - 4) * (k - 15) < 0 ↔ (4 < k ∧ k < 15) :=
begin
  sorry
end

end count_positive_integers_satisfying_inequality_l614_614185


namespace measure_of_angle_A_l614_614803

theorem measure_of_angle_A (A B C D : Prop) 
  (h1: ∀ α : Prop, ∃ β : Prop, ∃ γ : Prop, (α ↔ (β ∧ γ)) ∧ β = γ) 
  (h2: ∀ α : Prop, ∃ β : Prop, β = α / 2 ∨ β = 2 * α) : 
  ∃ α : ℝ, α = 72 ∨ α = 108 ∨ α = 720 / 7 ∨ α = 540 / 7 :=
sorry

end measure_of_angle_A_l614_614803


namespace correct_answer_is_C_l614_614263

variable (a : Vector ℝ) (λ : ℝ)
variables [NonZeroVector a] [Nonzero λ]

theorem correct_answer_is_C : Vector.same_direction a (λ^2 * a) := sorry

end correct_answer_is_C_l614_614263


namespace infinitely_many_solutions_l614_614611

variable (x y z : ℝ)

def problem_conditions : Prop :=
  x + y + z = 15 ∧ z = 2 * y

theorem infinitely_many_solutions : ∃ (x y z : ℝ), problem_conditions x y z :=
begin
  unfold problem_conditions,
  simp,
  have h : ∀ y : ℝ, ∃ x z : ℝ, x + 3 * y = 15 ∧ z = 2 * y,
  {
    intro y,
    use (15 - 3*y, 2*y),
    split,
    { simp [15 - 3*y] },
    { refl }
  },
  use [15 - 3*1, 1, 2*1],
  exact h 1,
end

end infinitely_many_solutions_l614_614611


namespace origin_on_circle_l614_614163

-- Define the points and the radius
def center : ℝ × ℝ := (3, 4)
def radius : ℝ := 5
def origin : ℝ × ℝ := (0, 0)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Translate the conditions and the proof problem into a Lean statement
theorem origin_on_circle : distance origin center = radius := 
by 
  sorry

end origin_on_circle_l614_614163


namespace number_of_children_l614_614231

-- Define the given conditions in Lean 4
variable {a : ℕ}
variable {R : ℕ}
variable {L : ℕ}
variable {k : ℕ}

-- Conditions given in the problem
def condition1 : 200 ≤ a ∧ a ≤ 300 := sorry
def condition2 : a = 25 * R + 10 := sorry
def condition3 : a = 30 * L - 15 := sorry 
def condition4 : a + 15 = 150 * k := sorry

-- The theorem to prove
theorem number_of_children : a = 285 :=
by
  assume a R L k // This assumption is for the variables needed.
  have h₁ : condition1 := sorry
  have h₂ : condition2 := sorry
  have h₃ : condition3 := sorry
  have h₄ : condition4 := sorry 
  exact sorry

end number_of_children_l614_614231


namespace seq_bounded_l614_614457

def digit_product (n : ℕ) : ℕ :=
  n.digits 10 |>.prod

def a_seq (a : ℕ → ℕ) (m : ℕ) : Prop :=
  a 0 = m ∧ (∀ n, a (n + 1) = a n + digit_product (a n))

theorem seq_bounded (m : ℕ) : ∃ B, ∀ n, a_seq a m → a n < B :=
by sorry

end seq_bounded_l614_614457


namespace num_sets_satisfying_union_is_four_l614_614176

variable (M : Set ℕ) (N : Set ℕ)

def num_sets_satisfying_union : Prop :=
  M = {1, 2} ∧ (M ∪ N = {1, 2, 6} → (N = {6} ∨ N = {1, 6} ∨ N = {2, 6} ∨ N = {1, 2, 6}))

theorem num_sets_satisfying_union_is_four :
  (∃ M : Set ℕ, M = {1, 2}) →
  (∃ N : Set ℕ, M ∪ N = {1, 2, 6}) →
  (∃ (num_sets : ℕ), num_sets = 4) :=
by
  sorry

end num_sets_satisfying_union_is_four_l614_614176


namespace minimum_k_l614_614146

noncomputable def f (n : ℕ) : ℕ :=
  ⌊Real.log 3 (2 * n)⌋ + 1

theorem minimum_k (n : ℕ) (h : n > 0) : 
  ∃ k, (∀ w : ℕ, 1 ≤ w ∧ w ≤ n →  ∃ l : ℤ, ∃ a : Fin k → ℤ, 
    w = l ∧ ∑ i, a i * (3 ^ i) = w) ∧ 
    k = ⌊Real.log 3 (2 * n)⌋ + 1 := 
sorry

end minimum_k_l614_614146


namespace non_zero_real_value_satisfies_eq_l614_614375

theorem non_zero_real_value_satisfies_eq (x : ℝ) (hx : x ≠ 0) : (5 * x) ^ 15 = (25 * x) ^ 5 ↔ x = sqrt (1 / 5) ∨ x = -sqrt (1 / 5) :=
by
  -- Proof skipped
  sorry

end non_zero_real_value_satisfies_eq_l614_614375


namespace sum_first_15_odd_starting_from_5_l614_614006

-- Definitions based on conditions in the problem.
def a : ℕ := 5    -- First term of the sequence is 5
def n : ℕ := 15   -- Number of terms is 15

-- Define the sequence of odd numbers starting from 5
def oddSeq (i : ℕ) : ℕ := a + 2 * i

-- Define the sum of the first n terms of this sequence
def sumOddSeq : ℕ := ∑ i in Finset.range n, oddSeq i

-- Key statement to prove that the sum of the sequence is 255
theorem sum_first_15_odd_starting_from_5 : sumOddSeq = 255 := by
  sorry

end sum_first_15_odd_starting_from_5_l614_614006


namespace dice_hidden_dots_l614_614130

theorem dice_hidden_dots :
  let die_sum := 21 in
  let total_dots := 4 * die_sum in
  let visible_dots := [1, 1, 2, 3, 4, 5, 6, 6, 5].sum in
  total_dots - visible_dots = 51 := by
  sorry

end dice_hidden_dots_l614_614130


namespace incorrect_option_c_l614_614603

variables {m n : Line} {α β : Plane}

-- Definitions for the conditions:
def line_subset_plane (l : Line) (π : Plane) := ∀ p, p ∈ l → p ∈ π
def line_perpendicular_plane (l : Line) (π : Plane) := ∀ p, p ∈ l → perpendicular p π
def line_parallel_plane (l : Line) (π : Plane) := ∀ p, p ∈ l → parallel p π
def line_parallel_line (l₁ l₂ : Line) := ∀ p₁ p₂, p₁ ∈ l₁ → p₂ ∈ l₂ → parallel p₁ p₂

-- The mathematical statement to be proved:
theorem incorrect_option_c (h₁ : line_subset_plane m α) (h₂ : line_parallel_plane n α) :
  ¬ (line_parallel_line m n ↔ line_parallel_plane n α ∧ ¬ line_subset_plane n α) :=
sorry

end incorrect_option_c_l614_614603


namespace points_on_inverse_proportion_l614_614291

theorem points_on_inverse_proportion (y_1 y_2 : ℝ) :
  (2:ℝ) = 5 / y_1 → (3:ℝ) = 5 / y_2 → y_1 > y_2 :=
by
  intros h1 h2
  sorry

end points_on_inverse_proportion_l614_614291


namespace powers_of_i_sum_l614_614804

theorem powers_of_i_sum :
  (complex.I ^ 21) + (complex.I ^ 103) + (complex.I ^ 50) = -1 :=
sorry

end powers_of_i_sum_l614_614804


namespace minimal_subjects_l614_614641

theorem minimal_subjects (n : Nat) (A : Fin n → Finset Fin n) :
  (n ≥ 3) →
  (∀ k : Fin n, A k.card = 3) →
  (∀ i j : Fin n, i ≠ j → (A i ∩ A j).card = 1) →
  (∃ (idx : Fin n), (∀ i : Fin n, idx ∈ A i)) ↔ (n = 7) := by {
  sorry
}

end minimal_subjects_l614_614641


namespace additional_stars_needed_l614_614726

-- Defining the number of stars required per bottle
def stars_per_bottle : Nat := 85

-- Defining the number of bottles Luke needs to fill
def bottles_to_fill : Nat := 4

-- Defining the number of stars Luke has already made
def stars_made : Nat := 33

-- Calculating the number of stars Luke still needs to make
theorem additional_stars_needed : (stars_per_bottle * bottles_to_fill - stars_made) = 307 := by
  sorry  -- Proof to be provided

end additional_stars_needed_l614_614726


namespace necessary_condition_transitivity_l614_614395

theorem necessary_condition_transitivity (A B C : Prop) 
  (hAB : A → B) (hBC : B → C) : A → C := 
by
  intro ha
  apply hBC
  apply hAB
  exact ha

-- sorry


end necessary_condition_transitivity_l614_614395


namespace y_at_x_equals_1_l614_614140

theorem y_at_x_equals_1 (a b : ℝ) (h₁ : ∀ x : ℝ, y = a * x ^ 3 + b * x + 2)
  (h₂ : y = 2009) : y = -2005 :=
by
  have h₃ : a + b = -2007,
  { sorry },  -- derived from h₁ and h₂ using substitution and simplification
  sorry  -- proving y = -2005 when x = 1 from h₃

end y_at_x_equals_1_l614_614140


namespace x_equals_neg_x_is_zero_abs_x_equals_2_is_pm_2_l614_614553

theorem x_equals_neg_x_is_zero (x : ℝ) (h : x = -x) : x = 0 := sorry

theorem abs_x_equals_2_is_pm_2 (x : ℝ) (h : |x| = 2) : x = 2 ∨ x = -2 := sorry

end x_equals_neg_x_is_zero_abs_x_equals_2_is_pm_2_l614_614553


namespace specific_certain_event_l614_614768

theorem specific_certain_event :
  ∀ (A B C D : Prop), 
    (¬ A) →
    (¬ B) →
    (¬ C) →
    D →
    D :=
by
  intros A B C D hA hB hC hD
  exact hD

end specific_certain_event_l614_614768


namespace area_rect_proof_l614_614625

noncomputable def area_rectangle (A B C D X P : Point) (h1 : Points_form_rectangle A B C D)
  (h2 : X_on_CD X C D) (h3 : Segments_intersect_at B X A C P)
  (h4 : Area_triangle B C P = 3) (h5 : Area_triangle P X C = 2) : Real :=
  15

-- The theorem statement
theorem area_rect_proof : ∀ (A B C D X P : Point),
  Points_form_rectangle A B C D →
  X_on_CD X C D →
  Segments_intersect_at B X A C P →
  Area_triangle B C P = 3 →
  Area_triangle P X C = 2 →
  area_rectangle A B C D X P = 15 :=
by
  intros A B C D X P h1 h2 h3 h4 h5
  sorry

end area_rect_proof_l614_614625


namespace total_notebooks_l614_614978

theorem total_notebooks (children : ℕ) (john_notebooks_per_child : ℕ) (wife_notebooks_per_child : ℕ) :
  children = 3 → john_notebooks_per_child = 2 → wife_notebooks_per_child = 5 →
  (john_notebooks_per_child * children) + (wife_notebooks_per_child * children) = 21 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_notebooks_l614_614978


namespace smallest_positive_period_of_f_minimum_value_of_f_in_interval_l614_614875

noncomputable def f : ℝ → ℝ := λ x, sin x - 2 * real.sqrt 3 * (sin (x / 2))^2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + 2 * real.pi) = f x := by
  sorry

theorem minimum_value_of_f_in_interval : ∃ x ∈ set.Icc (0 : ℝ) (2 * real.pi / 3), f x = -real.sqrt 3 := by
  sorry

end smallest_positive_period_of_f_minimum_value_of_f_in_interval_l614_614875


namespace unique_magnitude_of_roots_l614_614936

theorem unique_magnitude_of_roots (z : ℂ) (h : z^2 - 6 * z + 20 = 0) : 
  ∃! (c : ℝ), c = Complex.abs z := 
sorry

end unique_magnitude_of_roots_l614_614936


namespace rectangle_area_l614_614047

theorem rectangle_area (x w : ℝ) (h₁ : 3 * w = 3 * w) (h₂ : x^2 = 9 * w^2 + w^2) : 
  (3 * w) * w = (3 / 10) * x^2 := 
by
  sorry

end rectangle_area_l614_614047


namespace billy_ratio_apples_l614_614443

noncomputable def billy_ate_apples : Prop :=
  let M := 2 in              -- Apples on Monday
  let W := 9 in              -- Apples on Wednesday
  let T := 20 - (M + W + (4 * (M / 2)) + (M / 2)) in  -- Apples on Tuesday
  (T / M = 2)

theorem billy_ratio_apples : billy_ate_apples :=
by
  let M := 2      -- Apples on Monday
  let W := 9      -- Apples on Wednesday
  let F := M / 2  -- Apples on Friday
  let Th := 4 * F -- Apples on Thursday
  let T := 20 - (M + W + Th + F) -- Apples on Tuesday
  have h1 : M = 2 := rfl
  have h2 : F = M / 2 := rfl
  have h3 : Th = 4 * F := rfl
  have h4 : T = 20 - (M + W + Th + F) := rfl
  have h5 : T = 4 := by linarith [h1, h2, h3, h4]
  have h6 : T / M = 2 := by field_simp [h1, h5]
  exact h6

end billy_ratio_apples_l614_614443


namespace compare_abc_l614_614990

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614990


namespace range_of_f_l614_614319

def g (x : ℝ) : ℝ := Real.tan (π / 3 * x - π / 6)

noncomputable def M : ℝ := 3

def f (x : ℝ) : ℝ := M * Real.sin (2 * x - π / 6)

theorem range_of_f : 
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → -3 / 2 ≤ f x ∧ f x ≤ 3 :=
by
  intros x hx,
  sorry

end range_of_f_l614_614319


namespace smallest_block_with_hidden_cubes_l614_614425

def valid_block (l m n : ℕ) (N : ℕ) : Prop :=
  l ≥ 1 ∧ m ≥ 1 ∧ n ≥ 1 ∧ N = l * m * n

def hidden_cubes (l m n : ℕ) : Prop :=
  (l - 1) * (m - 1) * (n - 1) = 210

theorem smallest_block_with_hidden_cubes :
  ∃ (l m n N : ℕ), valid_block l m n N ∧ hidden_cubes l m n ∧ N = 336 :=
by
  have h1 := nat.succ_pos'
  have dim_min (x : ℕ) : 0 < x + 1 := nat.succ_pos' x
  have exists_factors_of_210 : 
    ∃ x y z : ℕ, x * y * z = 210 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z :=
  sorry
  cases exists_factors_of_210 with a factors
  cases factors with b factors_proof
  cases factors_proof with c factors_eq
  cases factors_eq with not_eq1 rhs_eq

  exists l m n, h_N.le, 
  exact ⟨7, 6, 8, 336, ⟨dim_min 6, dim_min 5, dim_min 7, rfl⟩, rfl⟩
  sorry

end smallest_block_with_hidden_cubes_l614_614425


namespace max_min_sum_of_expression_l614_614520

theorem max_min_sum_of_expression (a b : ℝ) (h : a^2 + a * b + b^2 = 3) :
  (let M := Real.sup (set_of (λ t, ∃ (a b : ℝ), a^2 + a * b + b^2 = 3 ∧ t = a^2 - a * b + b^2)),
   let m := Real.inf (set_of (λ t, ∃ (a b : ℝ), a^2 + a * b + b^2 = 3 ∧ t = a^2 - a * b + b^2)),
   M + m = 10) := sorry

end max_min_sum_of_expression_l614_614520


namespace calculate_expression_l614_614010

noncomputable def sum_odd (n : ℕ) : ℤ := ∑ k in Finset.range n, (2 * k + 1)
noncomputable def sum_even (n : ℕ) : ℤ := ∑ k in Finset.range n, 2 * (k + 1)

theorem calculate_expression :
  sum_odd 1013 - sum_even 1011 = 3057 :=
by
  sorry

end calculate_expression_l614_614010


namespace best_statistical_graph_l614_614734

def total_students := 50
def excellent_students := 10
def outstanding_student_leaders := 5

theorem best_statistical_graph (total : Nat) (excellent : Nat) (leaders : Nat) : 
  total = total_students → 
  excellent = excellent_students → 
  leaders = outstanding_student_leaders → 
  (true, "Pie chart") = (true, "Pie chart") :=
by
  intros
  trivial
  sorry

end best_statistical_graph_l614_614734


namespace number_of_C_animals_l614_614704

-- Define the conditions
def A : ℕ := 45
def B : ℕ := 32
def C : ℕ := 5

-- Define the theorem that we need to prove
theorem number_of_C_animals : B + C = A - 8 :=
by
  -- placeholder to complete the proof (not part of the problem's requirement)
  sorry

end number_of_C_animals_l614_614704


namespace product_remainder_mod_5_l614_614123

theorem product_remainder_mod_5 : (2024 * 1980 * 1848 * 1720) % 5 = 0 := by
  sorry

end product_remainder_mod_5_l614_614123


namespace tangent_length_of_midpoint_of_arc_l614_614506

-- Assume a circle with center O and radius R
noncomputable def length_of_tangent (R : ℝ) : ℝ :=
  2 * R

-- Given conditions
def is_one_fourth_sector (θ : ℝ) : Prop :=
  θ = π / 2

-- The proof problem
theorem tangent_length_of_midpoint_of_arc (R : ℝ) (θ : ℝ) (hθ : is_one_fourth_sector θ) :
  length_of_tangent R = 2 * R :=
by
  sorry

end tangent_length_of_midpoint_of_arc_l614_614506


namespace find_real_a_l614_614988

theorem find_real_a (a b c : ℂ) (ha : a ∈ ℝ) (h1 : a + b + c = 4)
  (h2 : a * b + b * c + c * a = 4) (h3 : a * b * c = 4) :
  a = 2 + Real.sqrt 2 :=
sorry

end find_real_a_l614_614988


namespace find_tangent_line_eq_l614_614839

noncomputable def tangent_line
  (A B C D E F x0 y0 : ℝ)
  (h : A * x0^2 + B * x0 * y0 + C * y0^2 + D * x0 + E * y0 + F = 0) :
  Prop :=
  let tangent_eq := 2 * A * x0 + B * y0 + D,
      tangent_ey := B * x0 + 2 * C * y0 + E in
  ∀ x y, 
    (tangent_eq * (x - x0) + tangent_ey * (y - y0) = 0)

-- Lean statement to prove the tangent line equation
theorem find_tangent_line_eq (A B C D E F x0 y0 : ℝ)
  (h : A * x0^2 + B * x0 * y0 + C * y0^2 + D * x0 + E * y0 + F = 0) :
  tangent_line A B C D E F x0 y0 h :=
sorry

end find_tangent_line_eq_l614_614839


namespace range_f_ineq_l614_614142

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2*x else x^2 - 2*x

lemma f_even (x : ℝ) : f (-x) = f x :=
by
  dsimp [f]
  split_ifs
  case h₁ h₂ => 
    linarith
  case h₃ h₄ =>
    linarith

lemma f_incr_pos (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x ≤ y) : f x ≤ f y :=
by
  dsimp [f]
  split_ifs
  case h₁ =>
    nlinarith [hx, hy, hxy]
  case h₂ =>
    linarith

theorem range_f_ineq (x : ℝ) : f (2 * x + 1) > f 2 ↔ x ∈ (Set.Ioo (-∞ : ℝ) (-3 / 2) ∪ (Set.Ioo (1 / 2) ∞)) :=
by
  sorry

end range_f_ineq_l614_614142


namespace inequality_example_l614_614299

theorem inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 = 3) : 
    1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + a * c) ≥ 3 / 2 :=
by
  sorry

end inequality_example_l614_614299


namespace probability_abs_diff_gt_half_l614_614620

def coin_flip_distribution (first_flip second_flip : Bool) : ℝ :=
  if first_flip then if second_flip then 0 else 1 else 0.5

def probability_of_event : ℚ :=
  let prob_0 : ℚ := 1/4
  let prob_0_5 : ℚ := 1/2
  let prob_1 : ℚ := 1/4
  have h_cases : ∀ x y, ((x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0)) → P(x) * P(y) = 1/16 := sorry,
  have h_add : (1/16 + 1/16 = 1/8) := by norm_num,
  1/8

theorem probability_abs_diff_gt_half :
  probability_of_event > 0.5 = 1/8 := sorry

end probability_abs_diff_gt_half_l614_614620


namespace markup_rate_l614_614776

theorem markup_rate (S : ℝ) (C : ℝ) (hS : S = 8) (h1 : 0.20 * S = 0.10 * S + (S - C)) :
  ((S - C) / C) * 100 = 42.857 :=
by
  -- Assume given conditions and reasoning to conclude the proof
  sorry

end markup_rate_l614_614776


namespace ratio_d1_d2_one_l614_614886
 
variables (A B C D P1 Q1 P2 Q2 : Type)
variable [MetricSpace A]
variable [NormedAddCommGroup A]
variable [NormedSpace ℝ A]

/-- Given a parallelogram ABCD with vertex C and B. 
    Two circles S1 and S2 going through vertices C and B respectively, 
    touching sides BA, AD, DC, BC at points P1, Q1, P2, Q2 respectively. 
    d1 and d2 are the distances from C and B to lines P1Q1 and P2Q2 respectively. -/
theorem ratio_d1_d2_one 
  (ABCD : Parallelogram A B C D)
  (S1 : Circle C)
  (S2 : Circle B)
  (Touches_S1_BA : S1.Touches BA at P1)
  (Touches_S1_AD : S1.Touches AD at Q1)
  (Touches_S2_DC : S2.Touches DC at P2)
  (Touches_S2_BC : S2.Touches BC at Q2)
  (d1 : A = dist C (line P1 Q1))
  (d2 : A = dist B (line P2 Q2))
  : d1 / d2 = 1 :=
sorry

end ratio_d1_d2_one_l614_614886


namespace max_area_triangle_ABC_l614_614512

-- Definitions: Points O, A, B, C in a plane
structure Point :=
(x : ℝ) (y : ℝ)

def O : Point := {x := 0, y := 0} -- Origin

-- Vector definitions for OA, OB, OC
def distance (p1 p2 : Point) : ℝ :=
real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

def dot_product (u v : Point) : ℝ :=
(u.x * v.x + u.y * v.y)

-- The points A, B, C with given distances from O
variables (A B C : Point)
variables (OA_ob : distance O A = 4)
variables (OB_ob : distance O B = 3)
variables (OC_ob : distance O C = 2)
variables (dot_product_ob : dot_product B C = 3)

-- The maximum value of the area S_ABC
def S_ABC_max : ℝ :=
2 * real.sqrt 7 + (3 * real.sqrt 3) / 2

-- Proof problem statement
theorem max_area_triangle_ABC :
  OA_ob →
  OB_ob →
  OC_ob →
  dot_product_ob →
  ∃ S : ℝ, S = S_ABC_max :=
sorry

end max_area_triangle_ABC_l614_614512


namespace laurent_series_around_singular_points_l614_614469

open Complex

noncomputable def f (z : ℂ) : ℂ := (2 * z - 3) / (z^2 - 3 * z + 2)

theorem laurent_series_around_singular_points :
    (∀ (z : ℂ), 0 < abs (z - 1) ∧ abs (z - 1) < 1 → 
    f z = (1 / (z - 1)) - ∑' n : ℕ, (z - 1) ^ n) ∧
    (∀ (z : ℂ), 0 < abs (z - 2) ∧ abs (z - 2) < 1 → 
    f z = (1 / (z - 2)) + ∑' n : ℕ, (-1) ^ n * (z - 2) ^ n) :=
begin
  sorry -- proof goes here
end

end laurent_series_around_singular_points_l614_614469


namespace twelve_sided_polygon_eq_triangles_l614_614177

theorem twelve_sided_polygon_eq_triangles :
  ∀ (vertices : Fin 12 → ℝ × ℝ),
  (is_regular_12_sided_polygon vertices) →
  distinct_equilateral_triangles vertices = 3 :=
by
  intros
  sorry

end twelve_sided_polygon_eq_triangles_l614_614177


namespace hexagon_area_l614_614419

noncomputable def area_of_regular_hexagon_inscribed_in_circle (r : ℝ) : ℝ :=
  6 * ( (r * r * real.sqrt 3) / 4 )

theorem hexagon_area {r : ℝ} (h : r = 3) :
  area_of_regular_hexagon_inscribed_in_circle r = 13.5 * real.sqrt 3 := 
by
  rw [h, area_of_regular_hexagon_inscribed_in_circle]
  sorry

end hexagon_area_l614_614419


namespace largest_integer_condition_l614_614976

theorem largest_integer_condition (m a b : ℤ) 
  (h1 : m < 150) 
  (h2 : m > 50) 
  (h3 : m = 9 * a - 2) 
  (h4 : m = 6 * b - 4) : 
  m = 106 := 
sorry

end largest_integer_condition_l614_614976


namespace compound_interest_rate_example_l614_614691

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) : ℝ :=
  let r := Real.root (t * 1) (A / P) - 1
  r

theorem compound_interest_rate_example :
  compound_interest_rate 10000 14500 1 5 ≈ 0.077033 :=
by
  sorry

end compound_interest_rate_example_l614_614691


namespace tire_repair_cost_without_tax_l614_614341

theorem tire_repair_cost_without_tax 
  (x : ℝ) 
  (hst: 0.50) 
  (hst_total: 50/100) 
  (total_cost : 4 * (x + hst) = 30) : 
  x = 7 := 
by 
  sorry

end tire_repair_cost_without_tax_l614_614341


namespace total_games_scheduled_l614_614645

-- Definitions based on given conditions
def num_divisions := 3
def teams_per_division := 6
def intra_division_games_per_pair := 3
def inter_division_games_per_team_pair := 2

-- Proven statement about the total number of games
theorem total_games_scheduled :
  let intra_division_games :=
        (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games_per_pair * num_divisions,
      inter_division_games :=
        (teams_per_division * num_divisions * (teams_per_division * (num_divisions - 1))) * inter_division_games_per_team_pair / 2
  in intra_division_games + inter_division_games = 351 := by
  sorry

end total_games_scheduled_l614_614645


namespace octahedron_plane_intersection_l614_614753

theorem octahedron_plane_intersection 
  (s : ℝ) 
  (a b c : ℕ) 
  (ha : Nat.Coprime a c) 
  (hb : ∀ p : ℕ, Prime p → p^2 ∣ b → False) 
  (hs : s = 2) 
  (hangle : ∀ θ, θ = 45 ∧ θ = 45) 
  (harea : ∃ A, A = (s^2 * Real.sqrt 3) / 2 ∧ A = a * Real.sqrt b / c): 
  a + b + c = 11 := 
by 
  sorry

end octahedron_plane_intersection_l614_614753


namespace pax_has_least_amount_l614_614612

def Persons := {Lex, Max, Pax, Rex, Tex}

variables (money : Persons → ℕ) 

axiom different_amounts : ∀ p1 p2 : Persons, p1 ≠ p2 → money p1 ≠ money p2
axiom tex_more_than_rex_max : money Rex < money Tex ∧ money Max < money Tex
axiom max_lex_more_than_pax : money Pax < money Max ∧ money Pax < money Lex
axiom rex_between_pax_max : money Pax < money Rex ∧ money Rex < money Max

theorem pax_has_least_amount : ∀ p : Persons, money Pax ≤ money p :=
by
  intro p
  cases p
  case Lex => sorry -- actual proof
  case Max => sorry -- actual proof
  case Pax => sorry -- actual proof
  case Rex => sorry -- actual proof
  case Tex => sorry -- actual proof

end pax_has_least_amount_l614_614612


namespace g_inv_eq_l614_614455

def g (x : ℝ) : ℝ := 2 * x ^ 2 + 3 * x - 5

theorem g_inv_eq (x : ℝ) (g_inv : ℝ → ℝ) (h_inv : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y) :
  (x = ( -1 + Real.sqrt 11 ) / 2) ∨ (x = ( -1 - Real.sqrt 11 ) / 2) :=
by
  -- proof omitted
  sorry

end g_inv_eq_l614_614455


namespace minimum_value_of_PA_PF_l614_614858

noncomputable def ellipse_min_distance : ℝ :=
  let F := (1, 0)
  let A := (1, 1)
  let a : ℝ := 3
  let F1 := (-1, 0)
  let d_A_F1 : ℝ := Real.sqrt ((-1 - 1)^2 + (0 - 1)^2)
  6 - d_A_F1

theorem minimum_value_of_PA_PF :
  ellipse_min_distance = 6 - Real.sqrt 5 :=
by
  sorry

end minimum_value_of_PA_PF_l614_614858


namespace a_4_value_l614_614949

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

theorem a_4_value (a : ℕ → ℝ) (S7 : S a 7 = 77) (h : arithmetic_sequence a) : a 4 = 11 :=
by
  sorry

end a_4_value_l614_614949


namespace num_possible_points_D_l614_614152

variables (A B C D : Point) (h s₁ s₂ : ℝ)
  -- triangle ABC is located in space

-- define the conditions for points D
def height_condition (D : Point) : Prop := 
  ∃ (Π₁ Π₂ : Plane), (D ∈ Π₁ ∨ D ∈ Π₂) 
  ∧ (∀ P : Point, P ∈ Π₁ → dist(D, Plane(ABC)) = h) 
  ∧ (∀ P : Point, P ∈ Π₂ → dist(D, Plane(ABC)) = h)

def area_condition_ACD (D : Point) : Prop := 
  ∃ (cylinder_AC : Surface), (D ∈ cylinder_AC) 
  ∧ (∀ P : Point, P ∈ cylinder_AC → area(Triangle(A,C,D)) = s₁)

def area_condition_BCD (D : Point) : Prop := 
  ∃ (cylinder_BC : Surface), (D ∈ cylinder_BC) 
  ∧ (∀ P : Point, P ∈ cylinder_BC → area(Triangle(B,C,D)) = s₂)

-- main theorem combining conditions
theorem num_possible_points_D : 
  (∃ D : Point, height_condition h D ∧ area_condition_ACD s₁ D ∧ area_condition_BCD s₂ D) → 
  ∃ n, (n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 8) := 
sorry

end num_possible_points_D_l614_614152


namespace vector_dot_product_l614_614853

noncomputable def P := (2, 1)
noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (2 * x - 4)
noncomputable def L : (ℝ × ℝ) → Prop := λ A, (∃ k : ℝ, f(A.1) = k * A.1 + f 2)

theorem vector_dot_product (A B P : ℝ × ℝ) (L : (ℝ × ℝ) → Prop)
  (hP : P = (2, 1))
  (hf : ∀ x, f x = (2 * x + 3) / (2 * x - 4))
  (hL : ∀ B, L B → L (f (B.1)))
  (h_intersect : L A ∧ L B) :
  ((λO : ℝ × ℝ, (A.1 - O.1, A.2 - O.2)) ⟨0, 0⟩ +
   (λO : ℝ × ℝ, (B.1 - O.1, B.2 - O.2)) ⟨0, 0⟩) 
   • (λO : ℝ × ℝ, (P.1 - O.1, P.2 - O.2)) ⟨0, 0⟩ = 10 := 
sorry

end vector_dot_product_l614_614853


namespace g_56_eq_497_l614_614313

def g : ℤ → ℤ 
| n := if n ≥ 500 then n - 3 else g (g (n + 4))

theorem g_56_eq_497 : g 56 = 497 :=
by
  sorry

end g_56_eq_497_l614_614313


namespace smallest_n_property_l614_614258

noncomputable def smallest_n (m : ℕ) (h : m ≥ 2) : ℕ := m^(m^(m+2))

theorem smallest_n_property (m : ℕ) (h : m ≥ 2) :
  ∀ n, n > m → (∀ A B : set ℕ, A ∪ B = {i | m ≤ i ∧ i ≤ n} →
    (∃ a b c ∈ A, c = a^b) ∨ (∃ a b c ∈ B, c = a^b)) ↔ n = smallest_n m h := 
sorry

end smallest_n_property_l614_614258


namespace tan_alpha_obtuse_l614_614827

theorem tan_alpha_obtuse (α : ℝ) (h1 : cos α = -1/2) (h2 : π/2 < α ∧ α < π) : tan α = -real.sqrt 3 :=
by
  sorry

end tan_alpha_obtuse_l614_614827


namespace circumscribed_sphere_volume_l614_614857

-- Conditions
variables {A B C D E F : Type}
variables (BC AD : ℝ) (AB : ℝ)
variables (h_BC_2 : BC = 2) (h_AB_1 : AB = 1)
variables (midpoint_E : E = (B + C) / 2)
variables (midpoint_F : F = (A + D) / 2)
variables (perpendicular_planes : Plane (ABEF) ⊥ Plane (EFDC))

-- Main statement
theorem circumscribed_sphere_volume {A B C : Type}
  (E F : Type)
  (radius : ℝ)
  (h_radius : radius = sqrt 3 / 2)
  : volume_sphere (circumscribed_sphere (triangle_prism A F E C)) = 
    (sqrt 3 / 2) * π :=
by sorry

end circumscribed_sphere_volume_l614_614857


namespace num_ways_to_select_officers_l614_614051

def ways_to_select_five_officers (n : ℕ) (k : ℕ) : ℕ :=
  (List.range' (n - k + 1) k).foldl (λ acc x => acc * x) 1

theorem num_ways_to_select_officers :
  ways_to_select_five_officers 12 5 = 95040 :=
by
  -- By definition of ways_to_select_five_officers, this is equivalent to 12 * 11 * 10 * 9 * 8.
  sorry

end num_ways_to_select_officers_l614_614051


namespace inequality_solution_l614_614200

theorem inequality_solution (f : ℝ → ℝ) (H₁ : ∀ x, f(x) ≥ 0 ↔ -1 ≤ x ∧ x ≤ 5) :
  { x | (1-x)/f x ≥ 0 } = { x | (-1 < x ∧ x ≤ 1) ∨ (5 < x) } :=
by
  sorry

end inequality_solution_l614_614200


namespace coefficients_of_polynomial_l614_614139

theorem coefficients_of_polynomial (a_5 a_4 a_3 a_2 a_1 a_0 : ℝ) :
  (∀ x : ℝ, x^5 = a_5 * (2*x + 1)^5 + a_4 * (2*x + 1)^4 + a_3 * (2*x + 1)^3 + a_2 * (2*x + 1)^2 + a_1 * (2*x + 1) + a_0) →
  a_5 = 1/32 ∧ a_4 = -5/32 :=
by sorry

end coefficients_of_polynomial_l614_614139


namespace Liked_Both_Proof_l614_614210

section DessertProblem

variable (Total_Students Liked_Apple_Pie Liked_Chocolate_Cake Did_Not_Like_Either Liked_Both : ℕ)
variable (h1 : Total_Students = 50)
variable (h2 : Liked_Apple_Pie = 25)
variable (h3 : Liked_Chocolate_Cake = 20)
variable (h4 : Did_Not_Like_Either = 10)

theorem Liked_Both_Proof :
  Liked_Both = (Liked_Apple_Pie + Liked_Chocolate_Cake) - (Total_Students - Did_Not_Like_Either) :=
by
  sorry

end DessertProblem

end Liked_Both_Proof_l614_614210


namespace minimize_cost_l614_614758

noncomputable def total_cost_per_km (speed : ℝ) : ℝ :=
  let fuel_cost := (speed ^ 3) * k + 96
  fuel_cost / speed

-- Given conditions as Lean definitions
def k : ℝ := 6 / (10 ^ 3)  -- proportional constant for fuel cost
def optimal_speed := 20  -- optimal speed in km/h

theorem minimize_cost :
  ∀ speed : ℝ, total_cost_per_km optimal_speed ≤ total_cost_per_km speed :=
sorry

end minimize_cost_l614_614758


namespace greatest_common_multiple_of_10_and_15_less_than_150_l614_614369

-- Definitions based on conditions
def lcm (a b : ℕ) : ℕ := Nat.lcm a b
def is_multiple (x y : ℕ) : Prop := ∃ k, x = y * k

-- Statement of the problem
theorem greatest_common_multiple_of_10_and_15_less_than_150 : 
  ∃ x, is_multiple x (lcm 10 15) ∧ x < 150 ∧ ∀ y, is_multiple y (lcm 10 15) ∧ y < 150 → y ≤ x :=
begin
  sorry
end

end greatest_common_multiple_of_10_and_15_less_than_150_l614_614369


namespace total_shoes_l614_614897

variable (a b c d : Nat)

theorem total_shoes (h1 : a = 7) (h2 : b = a + 2) (h3 : c = 0) (h4 : d = 2 * (a + b + c)) :
  a + b + c + d = 48 :=
sorry

end total_shoes_l614_614897


namespace main_problem_l614_614578

noncomputable def polar_line := ∃ θ ρ, ρ * (Math.cos θ) - sqrt 3 * ρ * (Math.sin θ) + 1 = 0
noncomputable def parametric_curve := ∃ α, (5 + Math.cos α, Math.sin α)

theorem main_problem :
  (∃ x y, polar_line = (x - sqrt 3 * y + 1 = 0)) ∧
  (∃ x y, parametric_curve = ((x - 5)^2 + y^2 = 1)) ∧
  (∃ α, parametric_curve α = (5 + Math.cos α, Math.sin α) ∧ 
   let d := abs (5 + Math.cos α - sqrt 3 * Math.sin α + 1) / 2 in
   (min d = 2 ∧ (5 + Math.cos α, Math.sin α) = (9/2, sqrt 3 / 2))
  ) := sorry

end main_problem_l614_614578


namespace sum_of_solutions_eq_zero_l614_614000

theorem sum_of_solutions_eq_zero :
  let equation := (6 * x) / 24 = 4 / x
  ∀ x₁ x₂ : ℝ, 
  x₁^2 = 16 → x₂^2 = 16 → 
  x₁ + x₂ = 0 := 
by
  -- Definitions of x₁ and x₂ coming from the equation
  intros x₁ x₂ h₁ h₂
  have h1 : x₁ = 4 ∨ x₁ = -4 := sorry
  have h2 : x₂ = 4 ∨ x₂ = -4 := sorry
  sorry

end sum_of_solutions_eq_zero_l614_614000


namespace asymptotes_of_hyperbola_l614_614652

def hyperbola_eq (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

theorem asymptotes_of_hyperbola :
  ∀ x y : ℝ, (hyperbola_eq x y) → (y = (2 / 3) * x) ∨ (y = -(2 / 3) * x) :=
begin
  sorry
end

end asymptotes_of_hyperbola_l614_614652


namespace three_digit_not_multiple_of_3_5_7_l614_614907

theorem three_digit_not_multiple_of_3_5_7 : 
  (900 - (let count_mult_3 := 300 in
           let count_mult_5 := 180 in
           let count_mult_7 := 128 in
           let count_mult_15 := 60 in
           let count_mult_21 := 43 in
           let count_mult_35 := 26 in
           let count_mult_105 := 9 in
           let total_mult_3_5_or_7 := 
             count_mult_3 + count_mult_5 + count_mult_7 - 
             (count_mult_15 + count_mult_21 + count_mult_35) +
             count_mult_105 in
           total_mult_3_5_or_7)) = 412 :=
by {
  -- The mathematical calculations were performed above
  -- The proof is represented by 'sorry' indicating the solution is skipped
  sorry
}

end three_digit_not_multiple_of_3_5_7_l614_614907


namespace surface_area_of_rectangular_prism_l614_614711

theorem surface_area_of_rectangular_prism :
  ∀ (length width height : ℝ), length = 8 → width = 4 → height = 2 → 
    2 * (length * width + length * height + width * height) = 112 :=
by
  intros length width height h_length h_width h_height
  rw [h_length, h_width, h_height]
  sorry

end surface_area_of_rectangular_prism_l614_614711


namespace solve_for_a_l614_614830

theorem solve_for_a {f : ℝ → ℝ} (h1 : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 2) (h2 : f a = 7) : a = 7 :=
sorry

end solve_for_a_l614_614830


namespace dataset_arith_progression_x_value_l614_614835

theorem dataset_arith_progression_x_value :
  ∀ (x : ℝ), 
    let dataset := [5, 2, 8, 2, 7, 2, x] in
    let mean := (5 + 2 + 8 + 2 + 7 + 2 + x) / 7 in
    let mode := 2 in
    let median := if x ≤ 2 then 2
                  else if 2 < x ∧ x < 5 then x
                  else if 5 ≤ x ∧ x ≤ 7 then 5
                  else 7-- Median depending on the value of x
    in
    (mode, median, mean).1 < (mode, median, mean).2 ∧ (mode, median, mean).2 < (mode, median, mean).3
    ∧ (median - mode) = (mean - median)
  → x = 40 / 13 := 
by
  sorry

end dataset_arith_progression_x_value_l614_614835


namespace sum_first_15_odd_starting_from_5_l614_614005

-- Definitions based on conditions in the problem.
def a : ℕ := 5    -- First term of the sequence is 5
def n : ℕ := 15   -- Number of terms is 15

-- Define the sequence of odd numbers starting from 5
def oddSeq (i : ℕ) : ℕ := a + 2 * i

-- Define the sum of the first n terms of this sequence
def sumOddSeq : ℕ := ∑ i in Finset.range n, oddSeq i

-- Key statement to prove that the sum of the sequence is 255
theorem sum_first_15_odd_starting_from_5 : sumOddSeq = 255 := by
  sorry

end sum_first_15_odd_starting_from_5_l614_614005


namespace population_double_in_eight_years_l614_614943

noncomputable def annualGrowthRate : ℝ :=
  (2 : ℝ)^(1 / 8) - 1

theorem population_double_in_eight_years :
  ∀ (P : ℝ) (r : ℝ), (P * (1 + r)^8 = 2 * P) ↔ r = annualGrowthRate :=
by
  -- Definitions for conditions for clarity
  intro P r
  split
  case mp =>
    intro h
    rw [mul_eq_mul_left_iff] at h
    cases h with h1 h2
    case inl =>
      exact false.elim (ne_of_lt (by norm_num) h1)
    case inr =>
      field_simp at h2
      exact h2
  case mpr =>
    intro h
    rw h
    field_simp
    norm_num
  sorry

end population_double_in_eight_years_l614_614943


namespace negation_proof_l614_614663

theorem negation_proof (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_proof_l614_614663


namespace part_I_part_II_part_III_l614_614171

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := (x / (1 + x)) - b * log (1 + x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log (1 + x) - a * x

theorem part_I : ∀ x : ℝ, g x 1 ≤ 0 := by
  sorry

theorem part_II (a : ℝ) : (∀ x ∈ set.Ici (0 : ℝ), f x a ≤ 0) ↔ (1 ≤ a) := by
  sorry

theorem part_III (n : ℕ) : (∑ i in finset.range n, i / (i^2 + 1 : ℝ) - log ↑n) ≤ (1 / 2 : ℝ) := by
  sorry

end part_I_part_II_part_III_l614_614171


namespace usage_difference_correct_l614_614703

def computerUsageLastWeek : ℕ := 91

def computerUsageThisWeek : ℕ :=
  let first4days := 4 * 8
  let last3days := 3 * 10
  first4days + last3days

def computerUsageFollowingWeek : ℕ :=
  let weekdays := 5 * (5 + 3)
  let weekends := 2 * 12
  weekdays + weekends

def differenceThisWeek : ℕ := computerUsageLastWeek - computerUsageThisWeek
def differenceFollowingWeek : ℕ := computerUsageLastWeek - computerUsageFollowingWeek

theorem usage_difference_correct :
  differenceThisWeek = 29 ∧ differenceFollowingWeek = 27 := by
  sorry

end usage_difference_correct_l614_614703


namespace monotonic_increasing_iff_k_ge_2_l614_614043

noncomputable def f (k x : ℝ) : ℝ := k * x - 2 * real.log x

theorem monotonic_increasing_iff_k_ge_2 (k : ℝ) :
  (∀ x ≥ 1, deriv (λ x, f k x) x ≥ 0) ↔ k ≥ 2 := 
by
  sorry

end monotonic_increasing_iff_k_ge_2_l614_614043


namespace parameter_a_solution_exists_l614_614810

theorem parameter_a_solution_exists (a : ℝ) : 
  (a < -2 / 3 ∨ a > 0) → ∃ b x y : ℝ, 
  x = 6 / a - abs (y - a) ∧ x^2 + y^2 + b^2 + 63 = 2 * (b * y - 8 * x) :=
by
  intro h
  sorry

end parameter_a_solution_exists_l614_614810


namespace common_difference_is_two_l614_614573

variable {α : Type*} [linear_ordered_field α]

noncomputable def proof_problem : Prop :=
  ∃ (a : ℕ → α) (d : α),
    (a 2 + a 6 = 8) ∧ (a 5 = 6) ∧ (∀ n, a (n + 1) - a n = d) ∧ d = 2

theorem common_difference_is_two :
  proof_problem :=
by
  use [λ n, 2 * n + 2] -- providing a specific arithmetic sequence example
  exists 2
  -- Conditions:
  split
  -- a_2 + a_6 = 8
  split
  linarith
  -- a_5 = 6
  split
  linarith
  -- common difference is 2
  split
  intro n
  ring
  -- proof that d = 2
  rfl

end common_difference_is_two_l614_614573


namespace max_cookies_andy_could_have_eaten_l614_614678

theorem max_cookies_andy_could_have_eaten (x k : ℕ) (hk : k > 0) 
  (h_total : x + k * x + 2 * x = 36) : x ≤ 9 :=
by
  -- Using the conditions to construct the proof (which is not required based on the instructions)
  sorry

end max_cookies_andy_could_have_eaten_l614_614678


namespace volleyball_team_selection_l614_614287

open Nat

-- Definitions based on the conditions
def numTotalPlayers := 14
def numTriplets := 3
def numStarters := 6
def triplets : Fin numTriplets → String := 
  λ i, ["Alicia", "Amanda", "Anna"].nth i sorry

-- Proof goal
theorem volleyball_team_selection :
  ∃ ways : ℕ,
    ways = (numTriplets.choose 2) * ((numTotalPlayers - numTriplets).choose (numStarters - 2)) ∧
    ways = 990 := 
by
  sorry

end volleyball_team_selection_l614_614287


namespace commission_per_car_l614_614728

theorem commission_per_car 
  (base_salary : ℕ) 
  (march_earnings : ℕ)
  (cars_sold : ℕ)
  (commission : ℕ) :
  base_salary = 1000 →
  march_earnings = 2000 →
  cars_sold = 15 →
  2 * march_earnings = 2 * (base_salary + 15 * commission) →
  commission = 200 :=
by
  intros bs_me base_salary_1000 me_2000 cars_15 double_march_earnings
  sorry

end commission_per_car_l614_614728


namespace hyperbola_foci_distance_l614_614813

theorem hyperbola_foci_distance :
  let equation := (λ x y, 3*x^2 - 18*x - 9*y^2 - 27*y = 81)
  ∃ c : ℝ, c = 6.272 ∧ (2 * c) = 12.544 :=
by
  let a_squared := 29.5
  let b_squared := 9.8333
  let c := Real.sqrt (a_squared + b_squared)
  have hc : c = 6.272 := sorry
  use c
  split
  · exact hc
  · rw [hc]
    norm_num
    exact 12.544

end hyperbola_foci_distance_l614_614813


namespace smallest_value_of_a_plus_b_l614_614525

theorem smallest_value_of_a_plus_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  3^8 * 5^2 * 2 = a^b → a + b = 812 :=
begin
  sorry
end

end smallest_value_of_a_plus_b_l614_614525


namespace largest_prime_factor_9801_l614_614479

/-- Definition to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Definition for the largest prime factor of a number -/
def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else Classical.find (Nat.exists_greatest_prime_factor n h)

/-- Condition: 241 and 41 are prime factors of 9801 -/
def prime_factors_9801 : ∀ (p : ℕ), p ∣ 9801 → is_prime p → (p = 41 ∨ p = 241) :=
λ p hdiv hprime, by {
  obtain ⟨a, ha⟩ := Nat.exists_mul_of_dvd hdiv,
  have h : 9801 = 41 * 241 := rfl,
  have ph31 : is_prime 41 := Nat.Prime_iff.2 ⟨by norm_num, by norm_num⟩,
  have ph241 : is_prime 241 := Nat.Prime_iff.2 ⟨by norm_num, by norm_num⟩,
  have h_9801 : 9801 = 41 * 241, by refl,
  rw [ha, h_9801] at *,
  cases ha with ha l,
  { exact Or.inl ha },
  { exact Or.inr ha },
}

/-- Statement: the largest prime factor of 9801 is 241 -/
theorem largest_prime_factor_9801 : largest_prime_factor 9801 = 241 :=
by {
  rw largest_prime_factor,
  have h1 : 9801 = 41 * 241 := rfl,
  exact Classical.find_spec {
    exists := 241, 
    h := _,
    obtain ⟨m, hm⟩ := Nat.exists_dvd_of_not_prime,
  sorry,
}

end largest_prime_factor_9801_l614_614479


namespace digits_equals_zeros_l614_614296

def count_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else n.digits 10 |>.length

def count_zeros (n : ℕ) : ℕ :=
  n.digits 10 |> list.count 0

def seq_sum (f : ℕ → ℕ → ℕ) (n : ℕ) : ℕ :=
  finset.sum (finset.range n) (λ x, f x 10)

def total_digits (k : ℕ) : ℕ :=
  seq_sum (λ m, m * 9 * 10^(m-1)) k

def total_zeros (k : ℕ) : ℕ :=
  seq_sum (λ m n, count_zeros (10^(m+1) - 1)) k

theorem digits_equals_zeros (k : ℕ) : 
  total_digits k = total_zeros (k + 1) :=
sorry

end digits_equals_zeros_l614_614296


namespace jake_fewer_peaches_l614_614251

variable (Jake Steven : Type)
variable (has_peaches has_apples : Steven → ℕ)
variable (has_more_apples : ∀ j : Jake, ∀ s : Steven, has_apples j = has_apples s + 84)
variable (fewer_peaches : ∀ j : Jake, ∀ s : Steven, has_peaches j < has_peaches s)
variable (steven_peaches : ∀ s : Steven, has_peaches s = 13)
variable (steven_apples : ∀ s : Steven, has_apples s = 52)

theorem jake_fewer_peaches (j : Jake) (s : Steven) : ∃ n, has_peaches s - has_peaches j = n :=
by
    unfold fewer_peaches
    unfold steven_peaches
    have h := fewer_peaches j s
    have h1 := steven_peaches s
    linarith

end jake_fewer_peaches_l614_614251


namespace original_recipe_serves_7_l614_614097

theorem original_recipe_serves_7 (x : ℕ)
  (h1 : 2 / x = 10 / 35) :
  x = 7 := by
  sorry

end original_recipe_serves_7_l614_614097


namespace find_m_for_parallel_vectors_l614_614178

theorem find_m_for_parallel_vectors (m : ℝ) :
  let a := (1, m)
  let b := (2, -1)
  (2 * a.1 + b.1, 2 * a.2 + b.2) = (k * (a.1 - 2 * b.1), k * (a.2 - 2 * b.2)) → m = -1/2 :=
by
  sorry

end find_m_for_parallel_vectors_l614_614178


namespace even_and_monotonically_increasing_f3_l614_614770

noncomputable def f1 (x : ℝ) : ℝ := x^3
noncomputable def f2 (x : ℝ) : ℝ := -x^2 + 1
noncomputable def f3 (x : ℝ) : ℝ := abs x + 1
noncomputable def f4 (x : ℝ) : ℝ := 2^(-abs x)

theorem even_and_monotonically_increasing_f3 :
  (∀ x, f3 x = f3 (-x)) ∧ (∀ x > 0, ∀ y > x, f3 y > f3 x) := 
sorry

end even_and_monotonically_increasing_f3_l614_614770


namespace derivative_f_at_zero_l614_614718

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (exp (x^2) - cos x) / x else 0

theorem derivative_f_at_zero : fderiv ℝ f 0 1 = 1.5 :=
sorry

end derivative_f_at_zero_l614_614718


namespace total_canoes_built_l614_614777

-- Defining basic variables and functions for the proof
variable (a : Nat := 5) -- Initial number of canoes in January
variable (r : Nat := 3) -- Common ratio
variable (n : Nat := 6) -- Number of months including January

-- Function to compute sum of the first n terms of a geometric series
def geometric_sum (a r n : Nat) : Nat :=
  a * (r^n - 1) / (r - 1)

-- The proposition we want to prove
theorem total_canoes_built : geometric_sum a r n = 1820 := by
  sorry

end total_canoes_built_l614_614777


namespace arithmetic_mean_of_scores_l614_614252

theorem arithmetic_mean_of_scores :
  let s1 := 85
  let s2 := 94
  let s3 := 87
  let s4 := 93
  let s5 := 95
  let s6 := 88
  let s7 := 90
  (s1 + s2 + s3 + s4 + s5 + s6 + s7) / 7 = 90.2857142857 :=
by
  sorry

end arithmetic_mean_of_scores_l614_614252


namespace smallest_N_to_prevent_white_rectangle_l614_614617

theorem smallest_N_to_prevent_white_rectangle : ∃ N : ℕ, (N = 4) ∧ (∀ grid : Matrix Bool 7 7, count_black_cells grid = N → no_white_rectangles grid 10)

end smallest_N_to_prevent_white_rectangle_l614_614617


namespace compare_abc_l614_614999

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614999


namespace solution_l614_614833

theorem solution (x y : ℝ) (h₁ : x + 3 * y = -1) (h₂ : x - 3 * y = 5) : x^2 - 9 * y^2 = -5 := 
by
  sorry

end solution_l614_614833


namespace shortest_distance_parabola_l614_614124

def point := (ℝ × ℝ)
def parabola (y : ℝ) : ℝ := y^2 / 4

def distance (p1 p2 : point) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem shortest_distance_parabola (a : ℝ) :
  ∃ P : point, P.2 = a ∧ P.1 = parabola a ∧ distance P (8, 8) = 4 * Real.sqrt 2 :=
sorry

end shortest_distance_parabola_l614_614124


namespace inscribed_quadrilateral_inequality_l614_614650

-- Definitions of points and line segments in a plane
variables {A B C D O: Type} [Is_point A] [Is_point B] [Is_point C] [Is_point D] [Is_point O]

-- Define lengths of the segments
variables (AB CD BC AD OA OC OB OD : ℝ)

-- Define the conditions for inscribed quadrilateral and intersection of diagonals
axiom InscribedQuadrilateral : is_inscribed A B C D
axiom DiagonalsIntersectAtO : intersects_at A C B D O

-- Main theorem statement
theorem inscribed_quadrilateral_inequality :
  AB / CD + CD / AB + BC / AD + AD / BC ≤ OA / OC + OC / OA + OB / OD + OD / OB :=
sorry

end inscribed_quadrilateral_inequality_l614_614650


namespace smallest_n_l614_614175

-- Definitions for sequences and initial conditions
def seq_a : ℕ → ℝ
def seq_b : ℕ → ℝ
axiom a_initial : seq_a 1 = 1
axiom seq_a_recurrence (n : ℕ) : seq_a (n + 1) / seq_a n = 1 / (3 * seq_a n + 2)
axiom seq_a_seq_b (n : ℕ) : seq_a n * seq_b n = 1

-- Main theorem statement
theorem smallest_n (n : ℕ) (H : seq_b n > 101) : n = 6 :=
sorry

end smallest_n_l614_614175


namespace height_of_tray_l614_614760

theorem height_of_tray (a : ℝ) (b : ℝ) (θ : ℝ) (length : ℝ) 
    (h₁ : a = 120) 
    (h₂ : b = 5) 
    (h₃ : θ = π / 4) 
    (h₄ : length = 5 * real.sqrt 3) : 
    length = 5 * real.sqrt 3 :=
by sorry

end height_of_tray_l614_614760


namespace length_of_PQ_and_PR_l614_614631
noncomputable theory

-- Definitions of the conditions
def isosceles_right_triangle (PQ PR QR : ℝ) : Prop :=
PQ = PR ∧ QR * √2 = PQ

def prism_volume (base_area height volume : ℝ) : Prop :=
(base_area * height) = volume

-- Given data from the problem
def PQ := √5
def PR := √5
def base_area := (1/2) * PQ^2
def height := 10
def volume := 25

-- The theorem to prove
theorem length_of_PQ_and_PR : 
  ∀ (PQ PR QR : ℝ), isosceles_right_triangle PQ PR QR 
  → prism_volume ((1/2) * PQ^2) height volume → PQ = √5 ∧ PR = √5 :=
by
  intros PQ PR QR h_triangle h_volume
  sorry

end length_of_PQ_and_PR_l614_614631


namespace unique_magnitude_of_roots_l614_614938

theorem unique_magnitude_of_roots (z : ℂ) (h : z^2 - 6 * z + 20 = 0) : 
  ∃! (c : ℝ), c = Complex.abs z := 
sorry

end unique_magnitude_of_roots_l614_614938


namespace number_of_irrationals_is_2_l614_614435

noncomputable def set_of_real_numbers := {15 : ℝ, 22 / 7, 3 * Real.sqrt 2, -3 * Real.pi, 0.10101}

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def count_irrationals (s : Set ℝ) : ℕ :=
  (s.filter is_irrational).toFinset.card

theorem number_of_irrationals_is_2 :
  count_irrationals set_of_real_numbers = 2 :=
sorry

end number_of_irrationals_is_2_l614_614435


namespace probability_even_sum_l614_614283

theorem probability_even_sum (tiles : Finset ℕ) (players : Finset (Finset ℕ)) :
  tiles = {1, 2, 3, 4, 5, 6, 7, 8, 9} →
  players.card = 3 →
  (∀ p ∈ players, p.card = 3 ∧ (∑ t in p, t) % 2 = 0) →
  ∃ m n : ℕ, Nat.coprime m n ∧ (∀ m_o n_o : ℕ, m * n_o = m_o * n → n_o = n -> m + n = 8) :=
begin
  intro h_tiles,
  intro h_players_card,
  intro h_player_conditions,
  sorry
end

end probability_even_sum_l614_614283


namespace find_a_l614_614045

theorem find_a
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (x y : ℝ) → (x + y)^8 = ∑ k in Finset.range 9, ( (Nat.choose 8 k) * x^(8-k) * y^k))
  (h4 : ∃ x y, 28 * x^6 * y^2 = 56 * x^5 * y^3)
  (h5 : a * b = 1/2) : a = 1 :=
  sorry

end find_a_l614_614045


namespace solve_equation_l614_614637

theorem solve_equation (x : ℚ) (h : (30 * x + 18).nthRoot 3 = 18) :
  x = 2907 / 15 :=
sorry

end solve_equation_l614_614637


namespace curve_equation_quadrilateral_area_l614_614576

noncomputable section

-- Define points C, D, and the segment length condition
variables (m n x y : ℝ)
variables (segment_length : ℝ := √2 + 1)
variables (condition1 : m^2 + n^2 = segment_length^2)

-- Define the ratio condition for vector magnitudes
variables (condition2 : (x - m, y) = √2 * (-x, n - y))

-- Prove the equation of the curve E
theorem curve_equation (h1 : condition1) (h2 : condition2) :
  ∀ (x y : ℝ), x^2 + y^2 / 2 = 1 :=
sorry

-- Define points A and B, and the equation of line l
variables (k : ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (h3 : y1 = k * x1 + 1)
variables (h4 : y2 = k * x2 + 1)
variables (h5 : (x1 + x2) ^ 2 + (y1 + y2) ^ 2 / 2 = 1)

-- Prove the area of quadrilateral OAMB
theorem quadrilateral_area (curv_eq : ∀ (x y : ℝ), x^2 + y^2 / 2 = 1) :
  ∀ (k : ℝ), area OAMB = √6 / 2 :=
sorry

end curve_equation_quadrilateral_area_l614_614576


namespace line_intersects_circle_l614_614732

-- Definitions
def radius : ℝ := 5
def distance_to_center : ℝ := 3

-- Theorem statement
theorem line_intersects_circle (r : ℝ) (d : ℝ) (h_r : r = radius) (h_d : d = distance_to_center) : d < r :=
by
  rw [h_r, h_d]
  exact sorry

end line_intersects_circle_l614_614732


namespace estimated_height_is_644_l614_614629

noncomputable def height_of_second_building : ℝ := 100
noncomputable def height_of_first_building : ℝ := 0.8 * height_of_second_building
noncomputable def height_of_third_building : ℝ := (height_of_first_building + height_of_second_building) - 20
noncomputable def height_of_fourth_building : ℝ := 1.15 * height_of_third_building
noncomputable def height_of_fifth_building : ℝ := 2 * |height_of_second_building - height_of_third_building|
noncomputable def total_estimated_height : ℝ := height_of_first_building + height_of_second_building + height_of_third_building + height_of_fourth_building + height_of_fifth_building

theorem estimated_height_is_644 : total_estimated_height = 644 := by
  sorry

end estimated_height_is_644_l614_614629


namespace find_k_l614_614201

theorem find_k (k : ℝ) : (∃ x : ℝ, x - 2 = 0 ∧ 1 - (x + k) / 3 = 0) → k = 1 :=
by
  sorry

end find_k_l614_614201


namespace value_a_square_binomial_l614_614798

theorem value_a_square_binomial (a : ℝ) : (∃ b : ℝ, (16 : ℝ) * x^2 + 40 * x + a = (4 * x + b)^2) → a = 25 :=
by
  intro h
  cases h with b h_eq
  rw [pow_two, mul_assoc, mul_left_comm, mul_comm, pow_two, mul_assoc, mul_right_comm] at h_eq
  sorry

end value_a_square_binomial_l614_614798


namespace b_has_infinite_solutions_l614_614799

noncomputable def b_value_satisfies_infinite_solutions : Prop :=
  ∃ b : ℚ, (∀ x : ℚ, 4 * (3 * x - b) = 3 * (4 * x + 7)) → b = -21 / 4

theorem b_has_infinite_solutions : b_value_satisfies_infinite_solutions :=
  sorry

end b_has_infinite_solutions_l614_614799


namespace first_grade_enrollment_l614_614235

theorem first_grade_enrollment (a : ℕ) (R : ℕ) (L : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
  (h3 : a = 25 * R + 10) (h4 : a = 30 * L - 15) : a = 285 :=
by
  sorry

end first_grade_enrollment_l614_614235


namespace find_x_l614_614918

theorem find_x (x : ℕ) (h : 2^10 = 32^x) (h32 : 32 = 2^5) : x = 2 :=
sorry

end find_x_l614_614918


namespace greatest_common_multiple_less_than_150_l614_614367

theorem greatest_common_multiple_less_than_150 : 
  ∀ (a b : ℕ), a = 10 → b = 15 → ∃ m, m < 150 ∧ m % a = 0 ∧ m % b = 0 ∧ ∀ n, n < 150 ∧ n % a = 0 ∧ n % b = 0 → n ≤ 120 :=
begin
  intros a b ha hb,
  rw [ha, hb],
  use 120,
  split,
  { exact lt_irrefl _ },
  split,
  { exact nat.mod_eq_zero_of_dvd },
  split,
  { exact nat.mod_eq_zero_of_dvd },
  { intros n hn,
    have h₁ : lcm 10 15 = 30 := by norm_num,
    sorry } -- the proof is left as an exercise
end

end greatest_common_multiple_less_than_150_l614_614367


namespace discount_is_20_percent_l614_614677

/-
Conditions:
1. The initial weight of the vest is 60 pounds.
2. Thomas wants to increase the weight by 60%.
3. Weights come in 2-pound steel ingots.
4. Each ingot costs $5.
5. The cost with the discount is $72.
6. More than 10 ingots were purchased.
-/

def initial_weight : ℝ := 60
def weight_increase_percentage : ℝ := 0.6
def ingot_weight : ℝ := 2
def ingot_cost : ℝ := 5
def cost_with_discount : ℝ := 72
def number_of_ingots_purchased : ℕ := 18
def original_total_cost (n : ℕ) : ℝ := n * ingot_cost
def discount_amount (original_cost discounted_cost : ℝ) : ℝ := original_cost - discounted_cost
def discount_percentage (original_cost discount : ℝ) : ℝ := (discount / original_cost) * 100

theorem discount_is_20_percent :
  let added_weight := initial_weight * weight_increase_percentage,
      ingots_needed := added_weight / ingot_weight,
      original_cost := original_total_cost number_of_ingots_purchased,
      discount := discount_amount original_cost cost_with_discount in
    discount_percentage original_cost discount = 20 :=
by
  sorry

end discount_is_20_percent_l614_614677


namespace calculate_new_average_score_l614_614955

noncomputable def new_average_score (num_students avg_score std_dev : ℕ) 
(percentiles : ℕ → (ℕ × ℕ))
(grace_marks : ℕ → ℕ) 
: ℝ :=
  let num_25th := percentiles 35).fst
  let num_50th := percentiles 35).snd in
  let num_75th := percentiles 35).fst - percentiles 35).snd
  let num_above_75th := 35 - num_25th - num_50th - num_75th in
  let total_grace_marks := num_25th * grace_marks 25 + 
                           num_50th * grace_marks 50 + 
                           num_75th * grace_marks 75 + 
                           num_above_75th * grace_marks 100 in
  let total_original_score := avg_score * num_students in
  let new_total_score := total_original_score + total_grace_marks in
  new_total_score / num_students

theorem calculate_new_average_score :
  new_average_score 35 37 6 (λ n, 
    if n <= 25 then (9,0) 
    else if n <= 50 then (9,9) 
    else if n <= 75 then (9,9) 
    else (8,17) ) 
  (λ n, 
    if n <= 25 then 6 
    else if n <= 50 then 4 
    else if n <= 75 then 2 
    else 0) ≈ 40.09 := by sorry

end calculate_new_average_score_l614_614955


namespace fractional_area_black_after_three_changes_l614_614441

theorem fractional_area_black_after_three_changes (original_area : ℝ) (h1 : original_area > 0) :
  let fraction_remaining := (8 / 9) ^ 3 in
  fraction_remaining * original_area = (512 / 729) * original_area :=
by
  sorry

end fractional_area_black_after_three_changes_l614_614441


namespace unique_number_not_in_range_of_g_l614_614095

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range_of_g 
  (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : g 5 a b c d = 5) (h6 : g 25 a b c d = 25) 
  (h7 : ∀ x, x ≠ -d/c → g (g x a b c d) a b c d = x) :
  ∃ r, r = 15 ∧ ∀ y, g y a b c d ≠ r := 
by
  sorry

end unique_number_not_in_range_of_g_l614_614095


namespace evaluate_fx_l614_614655

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x^2 else x^2 - x - 3

theorem evaluate_fx : f (1 / f 3) = 8 / 9 :=
by
  have f3_eq : f 3 = 3 :=
    by
      unfold f
      simp only [not_le, if_false]
      norm_num
  rw [f3_eq, one_div, f]
  simp only [one_div]
  have h : (1 / 3 : ℝ) ≤ 1 := by norm_num
  rw [if_pos h]
  ring

end evaluate_fx_l614_614655


namespace sandy_distance_l614_614634

theorem sandy_distance :
  ∃ d : ℝ, d = 18 * (1000 / 3600) * 99.9920006399488 := sorry

end sandy_distance_l614_614634


namespace bounded_area_l614_614779

noncomputable def y1 (x : ℝ) : ℝ := Real.acos (Real.cos x)
noncomputable def y2 (x : ℝ) : ℝ := abs (Real.sin x)

theorem bounded_area :
  let f := y1
  let g := y2
  let a := 0
  let b := 3 * Real.pi
  ∫ x in a .. b, (f x - g x) = Real.pi ^ 2 - 7 :=
by
  sorry

end bounded_area_l614_614779


namespace power_function_parity_even_l614_614526

noncomputable def f (x : ℝ) : ℝ := x^(2 / 3)

theorem power_function_parity_even (f : ℝ → ℝ) (h1 : f = (λ x, x ^ (2 / 3))) 
(h2 : f 8 = 4) : ∀ x, f (-x) = f x :=
by
  sorry

end power_function_parity_even_l614_614526


namespace median_salary_is_25000_l614_614104

def number_of_employees := 63
def CEO_count := 1
def SVP_count := 4
def Manager_count := 12
def AssistantManager_count := 8
def Clerk_count := 38

def CEO_salary := 135000
def SVP_salary := 95000
def Manager_salary := 80000
def AssistantManager_salary := 55000
def Clerk_salary := 25000

theorem median_salary_is_25000 :
  list.median (list.replicate CEO_count CEO_salary ++ 
               list.replicate SVP_count SVP_salary ++ 
               list.replicate Manager_count Manager_salary ++ 
               list.replicate AssistantManager_count AssistantManager_salary ++ 
               list.replicate Clerk_count Clerk_salary) = Clerk_salary :=
by
  sorry -- Proof is not provided.

end median_salary_is_25000_l614_614104


namespace problem_solution_l614_614757

open Nat

-- Define the sequence b_i
def b : ℕ → ℕ
| 0       => 0  -- included to satisfy definition for natural numbers
| 1       => 2
| 2       => 4
| 3       => 6
| 4       => 8
| 5       => 10
| 6       => 12
| 7       => 14
| 8       => 16
| 9       => 18
| 10      => 20
| (n + 1) => if (n + 1) > 10 then (b n) * (b (n - 1)) * ... * (b 1) + 1 else 2 * (n + 1)

-- Expression to evaluate
def expr : ℕ :=
  (Nat.prod (Finset.range 100.succ) b) + (Finset.sum (Finset.range 100.succ) (λ i => (b i) ^ 2))

-- Theorem that needs to be proved
theorem problem_solution : expr = 4226 := by
  sorry

end problem_solution_l614_614757


namespace problem_1_problem_2_l614_614516

-- Define polynomial and coefficients according to the conditions
def polynomial (x : ℝ) : ℝ := (2 * (x - 1) - 1) ^ 9

-- Definitions to state the propositions
def a_0 : ℝ := polynomial 1
def coeff_1 : ℝ := ∑ i in (finset.range 9 \ finset.singleton 0), (polynomial 2 - polynomial (2 + (i*pow 2 (i - 1))))
def a_2 : ℝ := -144
def sum_a_1_to_a_9 : ℝ := 2

-- Lean statements for the proof problems
theorem problem_1 :
  a_2 = -144 :=
by sorry

theorem problem_2 :
  coeff_1 = 2 :=
by sorry

end problem_1_problem_2_l614_614516


namespace exists_c_d_l614_614262

noncomputable def T : Set (ℝ × ℝ × ℝ) :=
  { p | ∃ (x y z : ℝ), p = (x, y, z) ∧ log 10 (x + y) = z ∧ log 10 (x ^ 2 + y ^ 2) = z + 2 }

theorem exists_c_d (c d : ℝ) :
  (∀ (x y z : ℝ), (x, y, z) ∈ T → x^3 - y^3 = c * 10^(3 * z) + d * 10^(2 * z)) →
  c + d = 45 :=
sorry

end exists_c_d_l614_614262


namespace internally_tangent_circles_distance_l614_614345

theorem internally_tangent_circles_distance
  (A B C D E : Point)
  (r₁ r₂ : ℝ)
  (r₁_pos : r₁ = 7)
  (r₂_pos : r₂ = 4)
  (tangent_A : is_tangent A C D r₁)
  (tangent_B : is_tangent B C E r₂)
  (internally_tangent : dist A B = r₁ - r₂)
  (AB_collinear : collinear A B C)
  (common_tangent : tangent_line A D intersects_line B E = C) :
  dist B C = 4 :=
sorry

end internally_tangent_circles_distance_l614_614345


namespace find_b1_l614_614049

noncomputable def sequence (b : ℕ → ℝ) : Prop :=
∀ n ≥ 2, (∑ i in Finset.range n.succ, b i) = n^3 * b n

noncomputable def b_27_eq_2 (b : ℕ → ℝ) : Prop :=
b 27 = 2

theorem find_b1 (b : ℕ → ℝ) (h_seq : sequence b) (h_b27 : b_27_eq_2 b) : b 1 = 1458 :=
sorry

end find_b1_l614_614049


namespace degree_of_my_polynomial_number_of_terms_of_my_polynomial_ascending_order_of_my_polynomial_l614_614797

noncomputable def my_polynomial : Polynomial ℤ := Polynomial.C 2 * Polynomial.X ^ 4
  + Polynomial.C 1 * Polynomial.X ^ 3 * Polynomial.Y ^ 2
  + Polynomial.C (-5) * Polynomial.X ^ 2 * Polynomial.Y ^ 3
  + Polynomial.C 1 * Polynomial.X
  + Polynomial.C (-1)

-- Proving that the degree of the polynomial is 4
theorem degree_of_my_polynomial : my_polynomial.degree = 4 := by
  sorry

-- Proving that the polynomial has 5 terms
theorem number_of_terms_of_my_polynomial : my_polynomial.coeff.support.card = 5 := by
  sorry

-- Proving that the polynomial arranged in ascending powers of a is correct
theorem ascending_order_of_my_polynomial : 
  my_polynomial.eval₂ Polynomial.C Polynomial.X Polynomial.Y = Polynomial.C (-1)
  + Polynomial.C 1 * Polynomial.X
  + Polynomial.C (-5) * Polynomial.X ^ 2 * Polynomial.Y ^ 3
  + Polynomial.C 1 * Polynomial.X ^ 3 * Polynomial.Y ^ 2
  + Polynomial.C 2 * Polynomial.X ^ 4 := by
  sorry

end degree_of_my_polynomial_number_of_terms_of_my_polynomial_ascending_order_of_my_polynomial_l614_614797


namespace collinear_vectors_l614_614133

open Vector

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def not_collinear (a b : V) : Prop :=
¬(∃ k : ℝ, k ≠ 0 ∧ a = k • b)

theorem collinear_vectors
  {a b m n : V}
  (h1 : m = a + b)
  (h2 : n = 2 • a + 2 • b)
  (h3 : not_collinear a b) :
  ∃ k : ℝ, k ≠ 0 ∧ n = k • m :=
by
  sorry

end collinear_vectors_l614_614133


namespace total_shoes_tried_on_l614_614899

variable (T : Type)
variable (store1 store2 store3 store4 : T)
variable (pair_of_shoes : T → ℕ)
variable (c1 : pair_of_shoes store1 = 7)
variable (c2 : pair_of_shoes store2 = pair_of_shoes store1 + 2)
variable (c3 : pair_of_shoes store3 = 0)
variable (c4 : pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3))

theorem total_shoes_tried_on (store1 store2 store3 store4 : T) (pair_of_shoes : T → ℕ) : 
  pair_of_shoes store1 = 7 →
  pair_of_shoes store2 = pair_of_shoes store1 + 2 →
  pair_of_shoes store3 = 0 →
  pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3) →
  pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3 + pair_of_shoes store4 = 48 := by
  intro c1 c2 c3 c4
  sorry

end total_shoes_tried_on_l614_614899


namespace michelle_travel_distance_l614_614773

-- Define the conditions
def initial_fee : ℝ := 2
def charge_per_mile : ℝ := 2.5
def total_paid : ℝ := 12

-- Define the theorem to prove the distance Michelle traveled
theorem michelle_travel_distance : (total_paid - initial_fee) / charge_per_mile = 4 := by
  sorry

end michelle_travel_distance_l614_614773


namespace find_xy_l614_614027

theorem find_xy (x y : ℝ) :
  0.75 * x - 0.40 * y = 0.20 * 422.50 →
  0.30 * x + 0.50 * y = 0.35 * 530 →
  x = 52.816 ∧ y = -112.222 :=
by
  intro h1 h2
  sorry

end find_xy_l614_614027


namespace number_of_irrationals_is_2_l614_614436

noncomputable def set_of_real_numbers := {15 : ℝ, 22 / 7, 3 * Real.sqrt 2, -3 * Real.pi, 0.10101}

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def count_irrationals (s : Set ℝ) : ℕ :=
  (s.filter is_irrational).toFinset.card

theorem number_of_irrationals_is_2 :
  count_irrationals set_of_real_numbers = 2 :=
sorry

end number_of_irrationals_is_2_l614_614436


namespace blackened_arc_length_in_polygon_l614_614203

-- Define the problem parameters
def regular_ngon (n : ℕ) : Type := sorry
def unit_circle (center : (ℝ × ℝ)) : Type := sorry
def path_length_of_rolling_circle (ngon : regular_ngon 2015) : ℝ := sorry

-- Define the main theorem to be proven
theorem blackened_arc_length_in_polygon :
  let p := 2013 in let q := 1 in
  ∃ (l : ℝ), l = 2013 * Real.pi → gcd p q = 1 ∧ p + q = 2014 :=
by {
  sorry
}

end blackened_arc_length_in_polygon_l614_614203


namespace prob_of_three_digit_divisible_by_3_l614_614524

/-- Define the exponents and the given condition --/
def a : ℕ := 5
def b : ℕ := 2
def c : ℕ := 3
def d : ℕ := 1

def condition : Prop := (2^a) * (3^b) * (5^c) * (7^d) = 252000

/-- The probability that a randomly chosen three-digit number formed by any 3 of a, b, c, d 
    is divisible by 3 and less than 250 is 1/4 --/
theorem prob_of_three_digit_divisible_by_3 :
  condition →
  ((sorry : ℝ) = 1/4) := sorry

end prob_of_three_digit_divisible_by_3_l614_614524


namespace bridget_initial_skittles_l614_614444

theorem bridget_initial_skittles (b : ℕ) (h : b + 4 = 8) : b = 4 :=
by {
  sorry
}

end bridget_initial_skittles_l614_614444


namespace unique_abs_value_of_solving_quadratic_l614_614939

theorem unique_abs_value_of_solving_quadratic :
  ∀ z : ℂ, (z^2 - 6*z + 20 = 0) → (complex.abs z = complex.sqrt 53) :=
begin
  sorry
end

end unique_abs_value_of_solving_quadratic_l614_614939


namespace length_of_faster_train_is_360_l614_614716

open Real

def kmph_to_mps (v : ℝ) : ℝ := v * (5 / 18)

noncomputable def calculate_length_of_faster_train 
  (speed_faster speed_slower : ℝ) 
  (time_taken : ℝ) : ℝ :=
  let relative_speed := kmph_to_mps (speed_faster - speed_slower)
  in relative_speed * time_taken

theorem length_of_faster_train_is_360 
  (speed_faster : ℝ) (speed_slower : ℝ) (time_taken : ℝ) 
  (h_speed_faster : speed_faster = 108) 
  (h_speed_slower : speed_slower = 54) 
  (h_time_taken : time_taken = 24) : 
  calculate_length_of_faster_train speed_faster speed_slower time_taken = 360 :=
by
  rw [h_speed_faster, h_speed_slower, h_time_taken]
  unfold calculate_length_of_faster_train kmph_to_mps
  norm_num
  sorry

end length_of_faster_train_is_360_l614_614716


namespace find_other_number_l614_614164

theorem find_other_number 
  (h : ℕ) (l : ℕ) (x : ℕ) 
  (reciprocal_h : 1 / (h : ℚ) = 1 / 17)
  (reciprocal_l : 1 / (l : ℚ) = 1 / 312)
  (number : ℕ := 24)
  (hcf_lcm_relation : h * l = number * x) :
  x = 221 :=
by
  have h_val : h = 17 := by
    rw [←nat.cast_inj, inv_eq_inv] at reciprocal_h
    exact (inv_eq_one_div 17).symm.trans reciprocal_h
  have l_val : l = 312 := by
    rw [←nat.cast_inj, inv_eq_inv] at reciprocal_l
    exact (inv_eq_one_div 312).symm.trans reciprocal_l
  rw [h_val, l_val] at hcf_lcm_relation
  rw [←nat.mul_div_assoc _ (nat.gcd_pos_of_pos_right (h * l) _)], sorry

end find_other_number_l614_614164


namespace shaded_to_non_shaded_ratio_l614_614585

structure Triangle :=
(P Q R : ℝ × ℝ)
(right_isosceles : ∃ M : (ℝ×ℝ), M = midpoint P Q ∧ distance P Q = distance P R)

structure Midpoints :=
(X Y Z M N : (ℝ×ℝ))
(midpoint_X : X = midpoint P Q)
(midpoint_Y : Y = midpoint Q R)
(midpoint_Z : Z = midpoint R P)
(midpoint_M : M = midpoint X Z)
(midpoint_N : N = midpoint Y Z)

def area (T : Triangle) : ℝ :=
  let s := distance T.P T.Q / 2
  2 * s^2

def shaded_area (m : Midpoints) : ℝ :=
  let s := distance m.X m.Y
  s^2 / 2

def non_shaded_area (T : Triangle) (m : Midpoints) : ℝ :=
  area T - shaded_area m

theorem shaded_to_non_shaded_ratio (T : Triangle) (m : Midpoints) : 
  Triangle.right_isosceles T →
  (shaded_area m) / (non_shaded_area T m) = 1 / 3 :=
by
  sorry

end shaded_to_non_shaded_ratio_l614_614585


namespace find_slant_height_l614_614116

theorem find_slant_height
  (r : ℝ) (CSA : ℝ) (π : ℝ)
  (h_r : r = 28) 
  (h_CSA : CSA = 2638.9378290154264)
  (pi_def : π = Real.pi) :
  ∃ l : ℝ, l = 30 ∧ CSA = π * r * l :=
by
  use 30
  have h_r_pos : r > 0 := by 
    rw [h_r]
    norm_num
  have h_pi_pos : π > 0 := by 
    rw [pi_def]
    exact Real.pi_pos
  have h_l_correct : 2638.9378290154264 = π * 28 * 30 := by
    rw [pi_def, ←h_CSA]
    norm_num
  split
  . exact rfl
  . exact h_l_correct

end find_slant_height_l614_614116


namespace bottles_have_200_mL_l614_614427

def liters_to_milliliters (liters : ℕ) : ℕ :=
  liters * 1000

def total_milliliters (liters : ℕ) : ℕ :=
  liters_to_milliliters liters

def milliliters_per_bottle (total_mL : ℕ) (num_bottles : ℕ) : ℕ :=
  total_mL / num_bottles

theorem bottles_have_200_mL (num_bottles : ℕ) (total_oil_liters : ℕ) (h1 : total_oil_liters = 4) (h2 : num_bottles = 20) :
  milliliters_per_bottle (total_milliliters total_oil_liters) num_bottles = 200 := 
by
  sorry

end bottles_have_200_mL_l614_614427


namespace projectile_highest_point_area_l614_614750

def u : ℝ := sorry
def g : ℝ := sorry
def φ : ℝ := sorry
def t : ℝ := sorry

theorem projectile_highest_point_area :
  (0 ≤ φ ∧ φ ≤ π / 2) →
  (∀ u g φ t, x = u * t * cos φ ∧ y = u * t * sin φ - (1 / 2) * g * t^2) →
  ∃ d : ℝ, d = π / 8 ∧ area_of_curve = d * (u^4 / g^2) :=
sorry

end projectile_highest_point_area_l614_614750


namespace circular_arc_sum_l614_614794

theorem circular_arc_sum (n : ℕ) (h₁ : n > 0) :
  ∀ s : ℕ, (1 ≤ s ∧ s ≤ (n * (n + 1)) / 2) →
  ∃ arc_sum : ℕ, arc_sum = s := 
by
  sorry

end circular_arc_sum_l614_614794


namespace probability_playing_one_instrument_l614_614567

noncomputable def total_people : ℕ := 800
noncomputable def fraction_playing_instruments : ℚ := 1 / 5
noncomputable def number_playing_two_or_more : ℕ := 32

theorem probability_playing_one_instrument :
  let number_playing_at_least_one := (fraction_playing_instruments * total_people)
  let number_playing_exactly_one := number_playing_at_least_one - number_playing_two_or_more
  (number_playing_exactly_one / total_people) = 1 / 6.25 :=
by 
  let number_playing_at_least_one := (fraction_playing_instruments * total_people)
  let number_playing_exactly_one := number_playing_at_least_one - number_playing_two_or_more
  have key : (number_playing_exactly_one / total_people) = 1 / 6.25 := sorry
  exact key

end probability_playing_one_instrument_l614_614567


namespace sum_expression_equality_l614_614596

theorem sum_expression_equality :
  let S := ∑ n in Finset.range 9800, 1 / Real.sqrt (n + 1 + Real.sqrt ((n + 1)^2 - 1))
  ∃ p q r : ℕ, 
    S = p + q * Real.sqrt r ∧
    r ≠ 0 ∧ ¬∃ k : ℕ, k * k ∣ r ∧ k > 1 ∧ 
    p + q + r = 121 := 
by
  sorry

end sum_expression_equality_l614_614596


namespace remainder_sum_is_74_l614_614449

-- Defining the values from the given conditions
def num1 : ℕ := 1234567
def num2 : ℕ := 890123
def divisor : ℕ := 256

-- We state the theorem to capture the main problem
theorem remainder_sum_is_74 : (num1 + num2) % divisor = 74 := 
sorry

end remainder_sum_is_74_l614_614449


namespace nonagon_diagonals_l614_614902

theorem nonagon_diagonals : 
  let n := 9 in
  let diagonals := n * (n - 3) / 2 in
  diagonals = 27 :=
by
  let n := 9
  let diagonals := n * (n - 3) / 2
  have h1: diagonals = 27 := by sorry
  exact h1

end nonagon_diagonals_l614_614902


namespace sum_volumes_spheres_l614_614052

theorem sum_volumes_spheres (l : ℝ) (h_l : l = 2) : 
  ∑' (n : ℕ), (4 / 3) * π * ((1 / (3 ^ (n + 1))) ^ 3) = (2 * π / 39) :=
by
  sorry

end sum_volumes_spheres_l614_614052


namespace solve_for_x_l614_614112

theorem solve_for_x (x : ℝ) (h : 5 * (x - 9) = 6 * (3 - 3 * x) + 6) : x = 3 :=
by
  sorry

end solve_for_x_l614_614112


namespace tangent_line_at_x_one_l614_614534

-- Definitions of given conditions and the function
def f (x : ℝ) : ℝ := x * Real.log x

-- Statement of the problem
theorem tangent_line_at_x_one : 
  let m := 1 + Real.log 1 -- Slope at x = 1 (which is 1)
  let y₁ := f 1          -- Value of the function at x = 1 (which is 0)
  let x₁ := 1            -- Point of tangency at x = 1
  y = m * (x - x₁) + y₁ := 
  y = x - 1 :=
sorry

end tangent_line_at_x_one_l614_614534


namespace integer_solutions_eq_l614_614475

theorem integer_solutions_eq :
  { (x, y) : ℤ × ℤ | 2 * x ^ 4 - 4 * y ^ 4 - 7 * x ^ 2 * y ^ 2 - 27 * x ^ 2 + 63 * y ^ 2 + 85 = 0 }
  = { (3, 1), (3, -1), (-3, 1), (-3, -1), (2, 3), (2, -3), (-2, 3), (-2, -3) } :=
by sorry

end integer_solutions_eq_l614_614475


namespace quadratic_function_integer_values_not_imply_integer_coefficients_l614_614250

theorem quadratic_function_integer_values_not_imply_integer_coefficients :
  ∃ (a b c : ℚ), (∀ x : ℤ, ∃ y : ℤ, (a * (x : ℚ)^2 + b * (x : ℚ) + c = (y : ℚ))) ∧
    (¬ (∃ (a_int b_int c_int : ℤ), a = (a_int : ℚ) ∧ b = (b_int : ℚ) ∧ c = (c_int : ℚ))) :=
by
  sorry

end quadratic_function_integer_values_not_imply_integer_coefficients_l614_614250


namespace length_x_PC_eq_half_a_sub_sqrt_ac_minus_b_l614_614672

theorem length_x_PC_eq_half_a_sub_sqrt_ac_minus_b 
  {a b c : ℝ} 
  (h1 : ∠C = 90°) 
  (h2 : P ∈ AC) 
  (h3 : incircle(ΔPBA) = incircle(ΔPBC)) 
  (h4 : a = BC) 
  (h5 : b = CA) 
  (h6 : c = AB) 
  : x = PC := 
    x = (1 / 2) * (a - sqrt (a * (c - b))) :=
begin
  sorry
end

end length_x_PC_eq_half_a_sub_sqrt_ac_minus_b_l614_614672


namespace find_a_value_l614_614868

open Real

noncomputable def only_one_tangent_line_through_P (a : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, P = (2, 1) ∧
  ∃ circle_eq : ℝ → ℝ → ℝ, circle_eq = (λ x y, x^2 + y^2 + 2*a*x + a*y + 2*a^2 + a - 1) ∧
  ∀ x y, circle_eq x y = 0 → P = (x, y)

theorem find_a_value :
  only_one_tangent_line_through_P (-1) :=
by
  -- Proof here
  sorry

end find_a_value_l614_614868


namespace distance_between_foci_l614_614811

theorem distance_between_foci (x y : ℝ) :
  3 * x^2 - 18 * x - 9 * y^2 - 27 * y = 81 →
  ∃ c : ℝ, 2 * c = 2 * Real.sqrt(39) :=
by
  intro h
  use Real.sqrt 39
  split
  · sorry -- Prove the equation implies the standard form of hyperbola.
  · rfl -- 2 * c = 2 * sqrt(39)

end distance_between_foci_l614_614811


namespace how_many_green_towels_l614_614616

-- Define the conditions
def initial_white_towels : ℕ := 21
def towels_given_to_mother : ℕ := 34
def towels_left_after_giving : ℕ := 22

-- Define the statement to prove
theorem how_many_green_towels (G : ℕ) (initial_white : ℕ) (given : ℕ) (left_after : ℕ) :
  initial_white = initial_white_towels →
  given = towels_given_to_mother →
  left_after = towels_left_after_giving →
  (G + initial_white) - given = left_after →
  G = 35 :=
by
  intros
  sorry

end how_many_green_towels_l614_614616


namespace number_of_rectangles_l614_614491

theorem number_of_rectangles (h_lines v_lines : ℕ) (h_lines_eq : h_lines = 5) (v_lines_eq : v_lines = 6) :
  (h_lines.choose 2) * (v_lines.choose 2) = 150 := by
  rw [h_lines_eq, v_lines_eq]
  norm_num
  exact dec_trivial
  sorry

end number_of_rectangles_l614_614491


namespace compare_abc_l614_614993

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614993


namespace least_value_q_minus_p_l614_614971

theorem least_value_q_minus_p (y : ℝ) (h1 : y > 3) (h2 : y < 6) : 
  let p := 3 in
  let q := 6 in
  q - p = 3 :=
by
  sorry

end least_value_q_minus_p_l614_614971


namespace sum_of_consecutive_integers_squared_l614_614131

theorem sum_of_consecutive_integers_squared (n : ℕ) : 
  ∑ k in finset.range (2*n - 1), (n + k) = (2*n - 1)^2 :=
by 
  sorry

end sum_of_consecutive_integers_squared_l614_614131


namespace boat_speed_in_still_water_l614_614401

theorem boat_speed_in_still_water
  (speed_of_stream : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ)
  (effective_speed : ℝ)
  (boat_speed : ℝ)
  (h1: speed_of_stream = 5)
  (h2: time_downstream = 2)
  (h3: distance_downstream = 54)
  (h4: effective_speed = boat_speed + speed_of_stream)
  (h5: distance_downstream = effective_speed * time_downstream) :
  boat_speed = 22 := by
  sorry

end boat_speed_in_still_water_l614_614401


namespace sum_of_even_subsets_of_S4_l614_614597

def capacity (X : Set ℕ) : ℕ :=
  if X = ∅ then 0 else Set.prod X id

def even_subset (X : Set ℕ) : Prop :=
  capacity X % 2 = 0

def even_subsets (S : Set ℕ) : Set (Set ℕ) :=
  {X | X ⊆ S ∧ even_subset X}

def sum_of_capacities (S : Set ℕ) (subsets : Set (Set ℕ)) : ℕ :=
  ∑ X in subsets, capacity X

theorem sum_of_even_subsets_of_S4 : 
  let S4 := {1, 2, 3, 4} in
  sum_of_capacities S4 (even_subsets S4) = 112 :=
by
  sorry

end sum_of_even_subsets_of_S4_l614_614597


namespace pills_per_day_l614_614279

theorem pills_per_day (total_days : ℕ) (prescription_days_frac : ℚ) (remaining_pills : ℕ) (days_taken : ℕ) (remaining_days : ℕ) (pills_per_day : ℕ)
  (h1 : total_days = 30)
  (h2 : prescription_days_frac = 4/5)
  (h3 : remaining_pills = 12)
  (h4 : days_taken = prescription_days_frac * total_days)
  (h5 : remaining_days = total_days - days_taken)
  (h6 : pills_per_day = remaining_pills / remaining_days) :
  pills_per_day = 2 := by
  sorry

end pills_per_day_l614_614279


namespace number_of_rectangles_l614_614488

theorem number_of_rectangles (H V : ℕ) (hH : H = 5) (hV : V = 6) :
  (nat.choose 5 2) * (nat.choose 6 2) = 150 :=
by
  rw [hH, hV]
  norm_num
  sorry

end number_of_rectangles_l614_614488


namespace train_speed_approx_18_kmph_l614_614763

/-- The length of the train in meters. -/
def length_of_train : ℝ := 100

/-- The length of the bridge in meters. -/
def length_of_bridge : ℝ := 150

/-- The time taken to cross the bridge in seconds. -/
def time_to_cross_bridge : ℝ := 49.9960003199744

/-- The total distance covered by the train in meters. -/
def total_distance : ℝ := length_of_train + length_of_bridge

/-- The speed of the train in meters per second (m/s). -/
def speed_in_mps : ℝ := total_distance / time_to_cross_bridge

/-- The conversion factor from meters per second to kilometers per hour. -/
def mps_to_kmph : ℝ := 3.6

/-- The speed of the train in kilometers per hour (km/h). -/
def speed_in_kmph : ℝ := speed_in_mps * mps_to_kmph

/-- The theorem stating that the speed of the train is approximately 18 km/h. -/
theorem train_speed_approx_18_kmph : abs (speed_in_kmph - 18) < 1 := sorry

end train_speed_approx_18_kmph_l614_614763


namespace corresponding_angles_equal_l614_614588

theorem corresponding_angles_equal 
  (α β γ : ℝ) 
  (h1 : α + β + γ = 180) 
  (h2 : (180 - α) + β + γ = 180) : 
  α = 90 ∧ β + γ = 90 ∧ (180 - α = 90) :=
by
  sorry

end corresponding_angles_equal_l614_614588


namespace total_number_of_lockers_l614_614317

theorem total_number_of_lockers (cost_per_digit: ℝ) (total_cost: ℝ) (costs: cost_per_digit = 0.03 ∧ total_cost = 287.85):
  ∃ n: ℕ, n = 2675 ∧ 
    let one_digit_cost := 9 * 1 * cost_per_digit,
        two_digit_cost := 90 * 2 * cost_per_digit,
        three_digit_cost := 900 * 3 * cost_per_digit,
        remaining_cost := total_cost - (one_digit_cost + two_digit_cost + three_digit_cost),
        cost_per_4_digit_locker := 4 * cost_per_digit,
        num_4_digit_lockers := remaining_cost / cost_per_4_digit_locker
    in (num_4_digit_lockers ≈ 1676)   :=
begin
  sorry
end

end total_number_of_lockers_l614_614317


namespace sum_inequality_l614_614888

theorem sum_inequality (n : ℕ) 
  (a b : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i ∧ 0 < b i)
  (h_sum_equal : (∑ i, a i) = (∑ i, b i)) : 
  (∑ i, a i^2 / (a i + b i)) ≥ (1/2) * (∑ i, a i) :=
by
  sorry

end sum_inequality_l614_614888


namespace sum_f_1_to_23_l614_614851

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom symmetry_g (x : ℝ) : g (1 + x) = g (1 - x)
axiom f_minus_g (x : ℝ) : f x - g x = 1
axiom f_plus_g (x : ℝ) : f (x + 1) + g (2 - x) = 1
axiom g_one : g 1 = 3

theorem sum_f_1_to_23 : (finset.range 23).sum (λ i, f (i + 1)) = 26 :=
sorry

end sum_f_1_to_23_l614_614851


namespace max_possible_sum_l614_614883

-- Define the elements of the pattern
variables (a b c d e : ℕ)

-- Given numbers
def nums : list ℕ := [2, 6, 9, 11, 14]

-- Constraint conditions
axiom constraint1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
axiom constraint2 : a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧ e ∈ nums

-- Constraint on the sums
axiom horizontal_vertical_sum : a + b + e = a + d + e
axiom diagonal_sum : a + d = b + c

-- Theorem statement
theorem max_possible_sum : ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
                          a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧ e ∈ nums ∧ 
                          (a + b + e = 31 ∨ a + d + e = 31 ∨ a + d = 31 ∨ b + c = 31) := 
begin
  sorry
end

end max_possible_sum_l614_614883


namespace find_q_l614_614581

-- Definitions and conditions:
variables (q : ℝ) (n : ℕ)
noncomputable def a : ℕ → ℝ
| 0 := 4
| n := a (n - 1) * q

-- Sum of the first n terms of the geometric sequence:
noncomputable def S : ℕ → ℝ
| 0 := 4
| n := a (n - 1) * (q ^ n - 1) / (q - 1)

-- Define the transformed sequence:
def T (n : ℕ) : ℝ := S n + 2

-- Conditions:
axiom S_geom : ∀ n, T n = S n + 2
axiom geom_seq : ∀ n, T n * T n = T (n - 1) * T (n + 1)

theorem find_q : q = 3 :=
by 
sorry

end find_q_l614_614581


namespace frac_m_over_q_l614_614188

variable (m n p q : ℚ)

theorem frac_m_over_q (h1 : m / n = 10) (h2 : p / n = 2) (h3 : p / q = 1 / 5) : m / q = 1 :=
by
  sorry

end frac_m_over_q_l614_614188


namespace disjoint_cover_exists_l614_614738

theorem disjoint_cover_exists { X : set ℝ^2 } (D : finset (set ℝ^2)) 
  (hD : ∀ x ∈ X, ∃ D_i ∈ D, x ∈ D_i) : 
  ∃ S : finset (set ℝ^2), (∀ D_1 ∈ S, ∀ D_2 ∈ S, D_1 ≠ D_2 → disjoint D_1 D_2) ∧ 
  (X ⊆ ⋃ (D_i ∈ S), { x | ∃ D_i_radius. x ∈ disk (center D_i) (3 * D_i_radius) }) :=
by
  sorry

end disjoint_cover_exists_l614_614738


namespace second_integer_is_66_l614_614327

-- Define the conditions
def are_two_units_apart (a b c : ℤ) : Prop :=
  b = a + 2 ∧ c = a + 4

def sum_of_first_and_third_is_132 (a b c : ℤ) : Prop :=
  a + c = 132

-- State the theorem
theorem second_integer_is_66 (a b c : ℤ) 
  (H1 : are_two_units_apart a b c) 
  (H2 : sum_of_first_and_third_is_132 a b c) : b = 66 :=
by
  sorry -- Proof omitted

end second_integer_is_66_l614_614327


namespace tangent_of_angle_l614_614315

/-- Define the coordinates of the point on the terminal side of angle θ -/
def coordinates : ℝ × ℝ := (-1, 2)

/-- Define the angle θ with initial side on the positive x-axis and terminal side passing through the point -/
def θ : Real.Ang := ⟨(-1, 2)⟩

/-- Prove that the tangent of θ is -2 -/
theorem tangent_of_angle : Real.tan θ = -2 := by
  sorry

end tangent_of_angle_l614_614315


namespace distinct_real_roots_l614_614493

def otimes (a b : ℝ) : ℝ := b^2 - a * b

theorem distinct_real_roots (m x : ℝ) :
  otimes (m - 2) x = m -> ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (x^2 - (m - 2) * x - m = 0) := by
  sorry

end distinct_real_roots_l614_614493


namespace minimum_queries_to_find_two_white_balls_l614_614674

-- Define the number of boxes
def num_boxes : ℕ := 2004

-- Define a function to represent whether a box contains a white ball
def contains_white_ball (box : ℕ) : Bool := sorry

-- Condition: The total number of white balls is even
def num_white_balls_even : Prop := ∃ (n : ℕ), 2 * n = (List.range num_boxes).countp contains_white_ball

-- Define a function that represents the query on two boxes
def at_least_one_white (box1 box2 : ℕ) : Bool := contains_white_ball box1 ∨ contains_white_ball box2

-- Minimum number of queries required to find two white balls
def min_queries : ℕ := 4005

theorem minimum_queries_to_find_two_white_balls :
  num_white_balls_even →
  (∃ box1 box2, box1 ≠ box2 ∧ contains_white_ball box1 ∧ contains_white_ball box2) → 
  min_queries = 4005 :=
sorry

end minimum_queries_to_find_two_white_balls_l614_614674


namespace necessary_but_not_sufficient_l614_614876

def f (x : ℝ) : ℝ := Real.log x

theorem necessary_but_not_sufficient (x : ℝ) :
  (f x > 0 → f (f x) > 0) ∧ (f (f x) > 0 → f x > 0) ∧ ¬(f x > 0 ↔ f (f x) > 0) :=
by {
  sorry
}

end necessary_but_not_sufficient_l614_614876


namespace function_symmetry_line_x_1_5_l614_614787

def floor_symm (x : ℝ) : ℝ :=
  (|⌊ x ⌋ + 1| - |⌊ 2 - x ⌋ + 1|)

noncomputable def symm_line : ℝ := 1.5

theorem function_symmetry_line_x_1_5 :
  ∀ a : ℝ, floor_symm (symm_line + a) = floor_symm (symm_line - a) :=
by
  sorry

end function_symmetry_line_x_1_5_l614_614787


namespace three_real_pairs_l614_614114

theorem three_real_pairs :
  ∃ (x y : ℝ), y = x^2 + 2x + 1 ∧ xy = x + y - 1 :=
sorry

end three_real_pairs_l614_614114


namespace find_y_in_terms_of_x_l614_614554

variable (x y : ℝ)

theorem find_y_in_terms_of_x (hx : x = 5) (hy : y = -4) (hp : ∃ k, y = k * (x - 3)) :
  y = -2 * x + 6 := by
sorry

end find_y_in_terms_of_x_l614_614554


namespace rhombus_segment_sum_l614_614495

theorem rhombus_segment_sum {a : ℝ} (CE CF : ℝ) (h1 : CE = CF)
  (h2 : ∀ A B C D : ℝ, (cos C = 1/4)) : CE + CF = 8 * a / 3 := 
sorry

end rhombus_segment_sum_l614_614495


namespace pure_imaginary_solutions_l614_614347

def is_pure_imaginary_solution (x : ℂ) : Prop :=
  (x^4 - 3*x^3 + 5*x^2 - 27*x - 36 = 0) ∧ (∃ k : ℝ, x = k * complex.I)

theorem pure_imaginary_solutions :
  is_pure_imaginary_solution (3 * complex.I) ∧ is_pure_imaginary_solution (-3 * complex.I) := 
by 
  sorry

end pure_imaginary_solutions_l614_614347


namespace total_steps_correct_l614_614622

/-- Definition of the initial number of steps on the first day --/
def steps_first_day : Nat := 200 + 300

/-- Definition of the number of steps on the second day --/
def steps_second_day : Nat := (3 / 2) * steps_first_day -- 1.5 is expressed as 3/2

/-- Definition of the number of steps on the third day --/
def steps_third_day : Nat := 2 * steps_second_day

/-- The total number of steps Eliana walked during the three days --/
def total_steps : Nat := steps_first_day + steps_second_day + steps_third_day

theorem total_steps_correct : total_steps = 2750 :=
  by
  -- provide the proof here
  sorry

end total_steps_correct_l614_614622


namespace quarter_sector_area_l614_614690

theorem quarter_sector_area (d : ℝ) (h : d = 10) : (π * (d / 2)^2) / 4 = 6.25 * π :=
by 
  sorry

end quarter_sector_area_l614_614690


namespace log_fraction_difference_l614_614372

theorem log_fraction_difference : 
  (log 160 / log 80) / (log 2 / log 80) - (log 320 / log 40) / (log 2 / log 40) = 2 := 
  sorry

end log_fraction_difference_l614_614372


namespace curve_equation_and_tangent_l614_614309

theorem curve_equation_and_tangent (k : ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (M : ℝ × ℝ)
    (hA : A = (α, 0))
    (hB : B = (0, β))
    (hDist : (A.1 - B.1)^2 + (A.2 - B.2)^2 = k^2)
    (hM : M = (α^3 / k^2, β^3 / k^2)) :
    M.1 ^ (2 / 3) + M.2 ^ (2 / 3) = k ^ (2 / 3) ∧
    is_tangent (λ x : ℝ × ℝ, x.1 ^ (2 / 3) + x.2 ^ (2 / 3)) (λ M : ℝ × ℝ, (α^3 / k^2, β^3 / k^2)) :=
sorry

end curve_equation_and_tangent_l614_614309


namespace solve_for_x_l614_614113

theorem solve_for_x (x : ℝ) (h : 5 * (x - 9) = 6 * (3 - 3 * x) + 6) : x = 3 :=
by
  sorry

end solve_for_x_l614_614113


namespace norm_of_5v_l614_614829

noncomputable def norm_scale (v : ℝ × ℝ) (c : ℝ) : ℝ := c * (Real.sqrt (v.1^2 + v.2^2))

theorem norm_of_5v (v : ℝ × ℝ) (h : Real.sqrt (v.1^2 + v.2^2) = 6) : norm_scale v 5 = 30 := by
  sorry

end norm_of_5v_l614_614829


namespace zeros_of_f_l614_614673

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- State the theorem about its roots
theorem zeros_of_f : ∃ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end zeros_of_f_l614_614673


namespace probability_sum_3_or_7_or_10_l614_614736

-- Definitions of the faces of each die
def die_1_faces : List ℕ := [1, 2, 2, 5, 5, 6]
def die_2_faces : List ℕ := [1, 2, 4, 4, 5, 6]

-- Probability of a sum being 3 (valid_pairs: (1, 2))
def probability_sum_3 : ℚ :=
  (1 / 6) * (1 / 6)

-- Probability of a sum being 7 (valid pairs: (1, 6), (2, 5))
def probability_sum_7 : ℚ :=
  ((1 / 6) * (1 / 6)) + ((1 / 3) * (1 / 6))

-- Probability of a sum being 10 (valid pairs: (5, 5))
def probability_sum_10 : ℚ :=
  (1 / 3) * (1 / 6)

-- Total probability for sums being 3, 7, or 10
def total_probability : ℚ :=
  probability_sum_3 + probability_sum_7 + probability_sum_10

-- The proof statement
theorem probability_sum_3_or_7_or_10 : total_probability = 1 / 6 :=
  sorry

end probability_sum_3_or_7_or_10_l614_614736


namespace hexagon_area_inscribed_in_circle_l614_614417

noncomputable def hexagon_area (r : ℝ) : ℝ :=
  let s := r
  6 * (s^2 * real.sqrt 3 / 4)

theorem hexagon_area_inscribed_in_circle :
  hexagon_area 3 = 27 * real.sqrt 3 / 2 := 
  by 
  -- The proof should be written here
  sorry

end hexagon_area_inscribed_in_circle_l614_614417


namespace sum_of_altitudes_of_triangle_l614_614661

noncomputable def sum_of_altitudes (a : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  let x_intercept := c/a in
  let y_intercept := c/b in
  let base := x_intercept in
  let height := y_intercept in
  let area := (1/2) * base * height in
  let third_altitude := 2 * area / real.sqrt(a^2 + b^2) in
  base + height + third_altitude

theorem sum_of_altitudes_of_triangle : sum_of_altitudes 10 8 80 = 18 + 40 * real.sqrt(41) / 41 := 
by sorry

end sum_of_altitudes_of_triangle_l614_614661


namespace true_propositions_l614_614153

noncomputable def alpha : Type := sorry
noncomputable def beta : Type := sorry
noncomputable def gamma : Type := sorry

axiom alpha_parallel_beta : ∀ (α β : Type), α = alpha → β = beta → ∀ (x y : α), x = y → ∀ (m n : β), m = n → parallel α β
axiom alpha_perpendicular_gamma : ∀ (α γ : Type), α = alpha → γ = gamma → ∀ (x : α) (y : γ), perpendicular x y

theorem true_propositions : 
  (∃ (a b : Type), a = α ∧ b = β ∧ (a ∥ b) ∧ (a ⊥ γ) ∧ (b ⊥ γ)) ∧ 
  ¬(∃ (a c : Type), a = α ∧ c = γ ∧ (a ∥ β) ∧ (a ⊥ c) ∧ (c ⊥ β)) ∧
  (∃ (b c : Type), b = β ∧ c = γ ∧ (b ∥ α) ∧ (α ⊥ c) ∧ (b ⊥ c)) →
  2 := 
by sorry

end true_propositions_l614_614153


namespace find_rolls_of_toilet_paper_l614_614442

theorem find_rolls_of_toilet_paper (visits : ℕ) (squares_per_visit : ℕ) (squares_per_roll : ℕ) (days : ℕ)
  (h_visits : visits = 3)
  (h_squares_per_visit : squares_per_visit = 5)
  (h_squares_per_roll : squares_per_roll = 300)
  (h_days : days = 20000) : (visits * squares_per_visit * days) / squares_per_roll = 1000 :=
by
  sorry

end find_rolls_of_toilet_paper_l614_614442


namespace smaller_rectangle_perimeter_l614_614053

theorem smaller_rectangle_perimeter (P : ℕ) (hP : P = 160) : ∃ k : ℕ, k = 80 :=
by
  let s := P / 4
  have hs : s = 40 := by rw [hP]; norm_num
  let w := s / 2
  have hw : w = 20 := by rw [hs]; norm_num
  have perimeter := 2 * (w + w)
  use 80
  have hp : w + w = 40 := by rw [hw]; norm_num
  rw [← hp, mul_assoc, mul_one]
  norm_num

sorry

end smaller_rectangle_perimeter_l614_614053


namespace number_of_children_l614_614233

-- Define the given conditions in Lean 4
variable {a : ℕ}
variable {R : ℕ}
variable {L : ℕ}
variable {k : ℕ}

-- Conditions given in the problem
def condition1 : 200 ≤ a ∧ a ≤ 300 := sorry
def condition2 : a = 25 * R + 10 := sorry
def condition3 : a = 30 * L - 15 := sorry 
def condition4 : a + 15 = 150 * k := sorry

-- The theorem to prove
theorem number_of_children : a = 285 :=
by
  assume a R L k // This assumption is for the variables needed.
  have h₁ : condition1 := sorry
  have h₂ : condition2 := sorry
  have h₃ : condition3 := sorry
  have h₄ : condition4 := sorry 
  exact sorry

end number_of_children_l614_614233


namespace problem_statement_l614_614238

noncomputable def curve_C1_parametric : ℝ → ℝ × ℝ := λ t, (2 * t, t^2)

def curve_C1_Cartesian (x y : ℝ) : Prop := x^2 = 4 * y

def curve_C2_polar (ρ θ : ℝ) : Prop := ρ * (sin θ + cos θ) = 5

def curve_C2_Cartesian (x y : ℝ) : Prop := x + y = 5

def point_P : ℝ × ℝ := (2, 3)

def PA (P A : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
def PB (P B : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)

theorem problem_statement :
  ( ∀ t, curve_C1_Cartesian (2*t) (t^2) ) ∧
  ( ∀ ρ θ, curve_C2_Cartesian (ρ * cos θ) (ρ * sin θ) = curve_C2_polar ρ θ ) ∧
  ( ( ∃ A B, curve_C1_Cartesian A.1 A.2 ∧ curve_C2_Cartesian A.1 A.2 ∧
            curve_C1_Cartesian B.1 B.2 ∧ curve_C2_Cartesian B.1 B.2 ∧
            (A ≠ B) ) →
    ( ∀ A B, PA point_P A ≠ 0 ∧ PB point_P B ≠ 0 →
              1 / (PA point_P A) + 1 / (PB point_P B) = real.sqrt(3) / 2 ) ) :=
begin
  sorry
end

end problem_statement_l614_614238


namespace number_of_sets_of_popcorn_l614_614042

theorem number_of_sets_of_popcorn (t p s : ℝ) (k : ℕ) 
  (h1 : t = 5)
  (h2 : p = 0.80 * t)
  (h3 : s = 0.50 * p)
  (h4 : 4 * t + 4 * s + k * p = 36) :
  k = 2 :=
by sorry

end number_of_sets_of_popcorn_l614_614042


namespace ball_distance_fifth_hit_l614_614062

-- Define initial height and rebound ratio
def initialHeight : Real := 120
def reboundRatio : Real := 1 / 3

-- Total distance traveled when the ball hits the ground the fifth time
theorem ball_distance_fifth_hit :
  let descent1 := initialHeight
  let ascent1 := reboundRatio * descent1
  let descent2 := ascent1
  let ascent2 := reboundRatio * descent2
  let descent3 := ascent2
  let ascent3 := reboundRatio * descent3
  let descent4 := ascent3
  let ascent4 := reboundRatio * descent4
  let descent5 := ascent4
  descent1 + ascent1 + descent2 + ascent2 + descent3 + ascent3 + descent4 + ascent4 + descent5 = 248.962 :=
by
  sorry

end ball_distance_fifth_hit_l614_614062


namespace unique_abs_value_of_solving_quadratic_l614_614941

theorem unique_abs_value_of_solving_quadratic :
  ∀ z : ℂ, (z^2 - 6*z + 20 = 0) → (complex.abs z = complex.sqrt 53) :=
begin
  sorry
end

end unique_abs_value_of_solving_quadratic_l614_614941


namespace sum_of_solutions_of_quadratic_l614_614301

theorem sum_of_solutions_of_quadratic :
    let a := 1;
    let b := -8;
    let c := -40;
    let discriminant := b * b - 4 * a * c;
    let root_discriminant := Real.sqrt discriminant;
    let sol1 := (-b + root_discriminant) / (2 * a);
    let sol2 := (-b - root_discriminant) / (2 * a);
    sol1 + sol2 = 8 := by
{
  sorry
}

end sum_of_solutions_of_quadratic_l614_614301


namespace not_odd_iff_exists_ne_l614_614600

open Function

variable {f : ℝ → ℝ}

theorem not_odd_iff_exists_ne : (∃ x : ℝ, f (-x) ≠ -f x) ↔ ¬ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end not_odd_iff_exists_ne_l614_614600


namespace difference_max_min_y_l614_614951

-- Define initial and final percentages of responses
def initial_yes : ℝ := 0.30
def initial_no : ℝ := 0.70
def final_yes : ℝ := 0.60
def final_no : ℝ := 0.40

-- Define the problem statement
theorem difference_max_min_y : 
  ∃ y_min y_max : ℝ, (initial_yes + initial_no = 1) ∧ (final_yes + final_no = 1) ∧
  (initial_yes + initial_no = final_yes + final_no) ∧ y_min ≤ y_max ∧ 
  y_max - y_min = 0.30 :=
sorry

end difference_max_min_y_l614_614951


namespace nadia_walks_distance_l614_614619

theorem nadia_walks_distance
    (d_flat : ℝ := 2.5)
    (v_up : ℝ := 4)
    (v_down : ℝ := 6)
    (v_flat : ℝ := 5)
    (t_NG : ℝ := 1.6)
    (t_GN : ℝ := 1.65) :
    let x := (5 * t_NG - d_flat) / (v_up + v_down),
        y := 5 - x in
    (x + y + d_flat) = 7.9 := 
by
  sorry

end nadia_walks_distance_l614_614619


namespace polynomial_coeff_square_diff_l614_614549

theorem polynomial_coeff_square_diff (a_0 a_1 a_2 a_3 a_4 : ℝ) :
  (∀ x : ℝ, (x + real.sqrt 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 1 :=
by
  sorry

end polynomial_coeff_square_diff_l614_614549


namespace pens_left_for_Lenny_l614_614257

def total_boxes := 20
def pens_per_box := 5
def percent_to_friends := 0.40
def fraction_to_classmates := 1 / 4

theorem pens_left_for_Lenny :
  let total_pens := total_boxes * pens_per_box in
  let pens_given_to_friends := total_pens * percent_to_friends in
  let pens_after_friends := total_pens - pens_given_to_friends in
  let pens_given_to_classmates := pens_after_friends * fraction_to_classmates in
  let pens_left := pens_after_friends - pens_given_to_classmates in
  pens_left = 45 :=
by
  sorry

end pens_left_for_Lenny_l614_614257


namespace revenue_increase_percentage_l614_614396

def initial_price_per_liter := 80   -- rubles
def initial_volume := 1             -- liters
def reduced_volume_per_carton := 0.9 -- liters
def new_price_per_carton := 99      -- rubles

def original_cost_9_liters := initial_price_per_liter * 9
def number_of_new_cartons := 9 / reduced_volume_per_carton
def new_cost_9_liters := number_of_new_cartons * new_price_per_carton
def increase_in_cost := new_cost_9_liters - original_cost_9_liters

def percentage_increase := (increase_in_cost / original_cost_9_liters) * 100

theorem revenue_increase_percentage :
  percentage_increase = 37.5 := by
  sorry

end revenue_increase_percentage_l614_614396


namespace determine_a_l614_614267

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z given a and the condition
def z (a : ℝ) : ℂ := (a / (1 - 2 * i)) + Complex.abs i

-- State the condition that the real part of z and the imaginary part are opposite numbers
def condition (a : ℝ) : Prop :=
  let real_part := Complex.re (z a)
  let imag_part := Complex.im (z a)
  real_part = -imag_part

-- State the main theorem to prove
theorem determine_a (a : ℝ) (h1 : condition a) : a = -5 / 3 := 
sorry

end determine_a_l614_614267


namespace exists_set_with_rational_product_subsets_l614_614249

-- Define the set and the conditions
def A : Set ℝ := {2 ^ (n / 2885) | n ∈ {1, 2^1, 2^2, ..., 2^10}}
def B : Set ℝ := {2 * 2 ^ (n / 2885) | n ∈ {1, 2^1, 2^2, ..., 2^10}}
def C : Set ℝ := {3 * 2 ^ (2 / 2885)}

-- Define the union of the sets A, B, and C
def S : Set ℝ := A ∪ B ∪ C

-- Define the proof problem: existence of 2422 subsets with rational product
theorem exists_set_with_rational_product_subsets : 
  ∃ (S : Set ℝ), S.card = 23 ∧
  (∃ (P : Finset (Finset ℝ)), P.card = 2422 ∧ 
   ∀ x ∈ P, ∃ y ∈ Finset.to_set x, y ∈ S ∧
   (∏ z in x, z).is_rational) := 
sorry

end exists_set_with_rational_product_subsets_l614_614249


namespace mobius_total_trip_time_l614_614618

theorem mobius_total_trip_time :
  ∀ (d1 d2 v1 v2 : ℝ) (n r : ℕ),
  d1 = 143 → d2 = 143 → 
  v1 = 11 → v2 = 13 → 
  n = 4 → r = (30:ℝ)/60 →
  d1 / v1 + d2 / v2 + n * r = 26 :=
by
  intros d1 d2 v1 v2 n r h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num

end mobius_total_trip_time_l614_614618


namespace margo_donation_l614_614615

variable (M J : ℤ)

theorem margo_donation (h1: J = 4700) (h2: (|J - M| / 2) = 200) : M = 4300 :=
sorry

end margo_donation_l614_614615


namespace total_shoes_l614_614896

variable (a b c d : Nat)

theorem total_shoes (h1 : a = 7) (h2 : b = a + 2) (h3 : c = 0) (h4 : d = 2 * (a + b + c)) :
  a + b + c + d = 48 :=
sorry

end total_shoes_l614_614896


namespace math_problem_l614_614346

/-
Two mathematicians take a morning coffee break each day.
They arrive at the cafeteria independently, at random times between 9 a.m. and 10:30 a.m.,
and stay for exactly m minutes.
Given the probability that either one arrives while the other is in the cafeteria is 30%,
and m = a - b√c, where a, b, and c are positive integers, and c is not divisible by the square of any prime,
prove that a + b + c = 127.

-/

noncomputable def is_square_free (c : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p * p ∣ c → False

theorem math_problem
  (m a b c : ℕ)
  (h1 : 0 < m)
  (h2 : m = a - b * Real.sqrt c)
  (h3 : is_square_free c)
  (h4 : 30 * (90 * 90) / 100 = (90 - m) * (90 - m)) :
  a + b + c = 127 :=
sorry

end math_problem_l614_614346


namespace cos_sin_quotient_l614_614519

variable (α : ℝ)
hypothesis h1 : sin α = - (real.sqrt 5) / 5
hypothesis h2 : ∃ k : ℤ, -(π / 2) < (2 * k * π + α) ∧ (2 * k * π + α) < 0 -- representing 4th quadrant

theorem cos_sin_quotient : (cos α + sin α) / (cos α - sin α) = 1 / 3 :=
sorry

end cos_sin_quotient_l614_614519


namespace remainder_when_divided_by_29_l614_614414

theorem remainder_when_divided_by_29 (k N : ℤ) (h : N = 761 * k + 173) : N % 29 = 28 :=
by
  sorry

end remainder_when_divided_by_29_l614_614414


namespace cube_volume_l614_614329

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V, V = 125 := 
by
  sorry

end cube_volume_l614_614329


namespace expected_value_of_area_of_triangle_l614_614034

noncomputable def expected_area_of_triangle_from_clock_hands : ℝ :=
  (3:ℝ) / (2 * Real.pi)

theorem expected_value_of_area_of_triangle :
  ∀ (hour_hand minute_hand second_hand : ℝ), 
    |hour_hand| = 1 → |minute_hand| = 1 → |second_hand| = 1 → 
    (∀ t : ℝ, t ∈ Icc 0 24 → (hour_hand, minute_hand, second_hand)
       = (cos (t * 2 * π / 12), cos (t * 2 * π / 60), cos (t * 2 * π / 60))) →
    (expected_area_of_triangle_from_clock_hands = 3 / (2 * Real.pi)) :=
sorry

end expected_value_of_area_of_triangle_l614_614034


namespace no_partition_equal_product_l614_614102

theorem no_partition_equal_product (n : ℕ) (h_pos : 0 < n) :
  ¬∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧
  A.prod id = B.prod id := sorry

end no_partition_equal_product_l614_614102


namespace coefficient_x6_expansion_l614_614363

theorem coefficient_x6_expansion 
  (binomial_expansion : (1 - 3 * x^2) ^ 6 = ∑ k in range 7, (Nat.choose 6 k) * (1:ℝ) ^ (6 - k) * (-3 * x^2) ^ k) :
  (∑ k in range 7, (Nat.choose 6 k) * (1:ℝ) ^ (6 - k) * (-3 * x^2) ^ k).coeff 6 = -540 := 
by
  have term_with_x6 : ∑ k in range 7, (Nat.choose 6 k) * (1:ℝ) ^ (6 - k) * (-3 * x^2) ^ k 
    = (Nat.choose 6 3) * (1:ℝ) ^ 3 * (-3 * x^2) ^ 3 := sorry
  have coefficient_of_x6 : (Nat.choose 6 3 * (-3) ^ 3) = -540 := sorry
  sorry

end coefficient_x6_expansion_l614_614363


namespace solution_exists_for_c_l614_614089

theorem solution_exists_for_c (c : ℝ) : 
  (∃ x y : ℝ, sqrt (x * y) = c^(2 * c) ∧ log c x ^ log c y + log c y ^ log c x = 6 * c^4) ↔ 
  c ∈ Set.Icc 0 (2 / Real.sqrt 3) :=
by
  sorry

end solution_exists_for_c_l614_614089


namespace complement_A_union_B_l614_614538
open set -- Open set namespace for set operations

-- Define the universal set U
def U : set ℕ := {x | x ≤ 5}

-- Define sets A and B
def A : set ℕ := {2, 4}
def B : set ℕ := {2, 3}

-- Define the statement to prove complement of the union
theorem complement_A_union_B : (U \ (A ∪ B)) = {1, 5} :=
by sorry

end complement_A_union_B_l614_614538


namespace angles_same_terminal_side_l614_614073

theorem angles_same_terminal_side (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℤ) :
  a₁ = 390 ∧ a₂ = 690 ∧ a₃ = -330 ∧ a₄ = 750 ∧ a₅ = 480 ∧ a₆ = -420 ∧ a₇ = 3000 ∧ a₈ = -840 → 
  (∃ k : ℤ, a₂ - a₁ = k * 360) ∨
  (∃ k : ℤ, a₄ - a₃ = k * 360) ∨
  (∃ k : ℤ, a₆ - a₅ = k * 360) ∨
  (∃ k : ℤ, a₈ - a₇ = k * 360) :=
by
  intro h
  cases h with h₁ h; cases h with h₂ h; cases h with h₃ h; cases h with h₄ h
  cases h with h₅ h; cases h with h₆ h; cases h with h₇ h₈
  use 3
  exact h₄.symm ▸ h₃ ▸ (by normalize <| show -330 + 750 = 3 * 360 from rfl)
  sorry -- Proof for all other cases can be added here

end angles_same_terminal_side_l614_614073


namespace sum_f_1_to_23_l614_614848

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom h_symm : ∀ x : ℝ, g(x) = g(2 - x)
axiom h_fg_eq : ∀ x : ℝ, f(x) - g(x) = 1
axiom h_fgp1 : ∀ x : ℝ, f(x + 1) + g(2 - x) = 1
axiom h_g1 : g(1) = 3

theorem sum_f_1_to_23 : (∑ i in finset.range 23, f (i + 1)) = 26 :=
by
  sorry

end sum_f_1_to_23_l614_614848


namespace together_complete_days_l614_614022

-- Define the work rates of x and y
def work_rate_x := (1 : ℚ) / 30
def work_rate_y := (1 : ℚ) / 45

-- Define the combined work rate when x and y work together
def combined_work_rate := work_rate_x + work_rate_y

-- Define the number of days to complete the work together
def days_to_complete_work := 1 / combined_work_rate

-- The theorem we want to prove
theorem together_complete_days : days_to_complete_work = 18 := by
  sorry

end together_complete_days_l614_614022


namespace probability_real_roots_eq_half_l614_614860

theorem probability_real_roots_eq_half (b c : ℝ) (hb : -1 ≤ b ∧ b ≤ 1) (hc : -1 ≤ c ∧ c ≤ 1) :
  let A := { (b, c) | b^2 ≥ c^2 }
  let S_D := 4
  let S := 2
  S / S_D = 1 / 2 :=
by
  sorry

end probability_real_roots_eq_half_l614_614860


namespace exists_x_for_integer_conditions_l614_614129

-- Define the conditions as functions in Lean
def is_int_div (a b : Int) : Prop := ∃ k : Int, a = b * k

-- The target statement in Lean 4
theorem exists_x_for_integer_conditions :
  ∃ t_1 : Int, ∃ x : Int, (x = 105 * t_1 + 52) ∧ 
    (is_int_div (x - 3) 7) ∧ 
    (is_int_div (x - 2) 5) ∧ 
    (is_int_div (x - 4) 3) :=
by 
  sorry

end exists_x_for_integer_conditions_l614_614129


namespace card_draw_probability_l614_614336

theorem card_draw_probability:
  let hearts := 13
  let diamonds := 13
  let clubs := 13
  let total_cards := 52
  let first_draw_probability := hearts / (total_cards : ℝ)
  let second_draw_probability := diamonds / (total_cards - 1 : ℝ)
  let third_draw_probability := clubs / (total_cards - 2 : ℝ)
  first_draw_probability * second_draw_probability * third_draw_probability = 2197 / 132600 :=
by
  sorry

end card_draw_probability_l614_614336


namespace calculate_spadesuit_l614_614824

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem calculate_spadesuit : spadesuit 3 (spadesuit 5 6) = -112 := by
  sorry

end calculate_spadesuit_l614_614824


namespace fraction_distance_traveled_by_bus_l614_614429

theorem fraction_distance_traveled_by_bus (D : ℝ) (hD : D = 105.00000000000003)
    (distance_by_foot : ℝ) (h_foot : distance_by_foot = (1 / 5) * D)
    (distance_by_car : ℝ) (h_car : distance_by_car = 14) :
    (D - (distance_by_foot + distance_by_car)) / D = 2 / 3 := by
  sorry

end fraction_distance_traveled_by_bus_l614_614429


namespace comparison_l614_614503

noncomputable def f : ℝ → ℝ := sorry

-- Define constants a, b, and c 
def a : ℝ := (2^0.1) * f (2^0.1)
def b : ℝ := (Real.log 2) * f (Real.log 2)
def c : ℝ := (Real.logBase 2 (1/8)) * f (Real.logBase 2 (1/8))

-- Conditions
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_deriv_cond : ∀ x : ℝ, x < 0 → f x + x * (deriv (deriv f)) x < 0

theorem comparison : c < a ∧ a < b := by
  sorry


end comparison_l614_614503


namespace first_grade_enrollment_l614_614236

theorem first_grade_enrollment (a : ℕ) (R : ℕ) (L : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
  (h3 : a = 25 * R + 10) (h4 : a = 30 * L - 15) : a = 285 :=
by
  sorry

end first_grade_enrollment_l614_614236


namespace uki_total_earnings_l614_614351

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def cupcakes_per_day : ℕ := 20
def cookies_per_day : ℕ := 10
def biscuits_per_day : ℕ := 20
def days : ℕ := 5

-- Prove the total earnings for five days
theorem uki_total_earnings : 
    (cupcakes_per_day * price_cupcake + 
     cookies_per_day * price_cookie + 
     biscuits_per_day * price_biscuit) * days = 350 := 
by
  -- The actual proof will go here, but is omitted for now.
  sorry

end uki_total_earnings_l614_614351


namespace stratified_sample_over_30_l614_614405

-- Define the total number of employees and conditions
def total_employees : ℕ := 49
def employees_over_30 : ℕ := 14
def employees_30_or_younger : ℕ := 35
def sample_size : ℕ := 7

-- State the proportion and the final required count
def proportion_over_30 (total : ℕ) (over_30 : ℕ) : ℚ := (over_30 : ℚ) / (total : ℚ)
def required_count (proportion : ℚ) (sample : ℕ) : ℚ := proportion * (sample : ℚ)

theorem stratified_sample_over_30 :
  required_count (proportion_over_30 total_employees employees_over_30) sample_size = 2 := 
by sorry

end stratified_sample_over_30_l614_614405


namespace angle_x_in_degrees_l614_614290

theorem angle_x_in_degrees
  (O A B C D : Point)
  (hO_center : is_center O)
  (h_ACD_subt_diameter : subtends_diameter A C D)
  (h_angleCDA : angle_deg C D A = 42)
  (h_radii_AO_BO_CO : AO = BO ∧ BO = CO)
  (h_angleOAB : angle_deg O A B = 10) :
  angle_deg O B C = 58 := 
sorry

end angle_x_in_degrees_l614_614290


namespace continuity_sufficient_but_not_necessary_l614_614740

variables {α : Type*} {β : Type*} [TopologicalSpace α] [TopologicalSpace β]
variables {f : α → β} {x₀ : α}

-- Definition of when a function is continuous at a point
def continuous_at (f : α → β) (x₀ : α) :=
  ∃ y ∈ (set.univ : set β), filter.tendsto f (nhds x₀) (nhds y)

-- Proof Problem: Continuity is a sufficient but not necessary condition for a function to be defined at x₀
theorem continuity_sufficient_but_not_necessary (h : continuous_at f x₀) :
  ∃ y, is_open {y} ∧ f x₀ = y :=
begin
  sorry
end

end continuity_sufficient_but_not_necessary_l614_614740


namespace geo_arith_seq_l614_614968

theorem geo_arith_seq
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : 0 < q)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : a 1 + a 3 = 2 * a 2) :
  (a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end geo_arith_seq_l614_614968


namespace points_concyclic_l614_614276

variables {C : Type} [ellipse C]
variables {F1 F2 : point}
variables {l1 l2 : line}
variables {P M1 M2 Q : point}

-- Assume P is on ellipse
axiom point_on_ellipse (P : point) (C : ellipse) : P ∈ C

-- Assume M1 and M2 are intersections
axiom line_parallel_to_foci (P : point) (F1 F2 : point) (l1 l2 : line) : 
  is_parallel (line_through P (line_through F1 F2)) ∧ 
  intersect l1 (line_through P (line_through F1 F2)) = M1 ∧ 
  intersect l2 (line_through P (line_through F1 F2)) = M2

-- Assume intersection of lines at Q
axiom lines_intersect_at_Q (M1 F1 M2 F2 : point) : intersect (line_through M1 F1) (line_through M2 F2) = Q

-- Prove P, F1, Q, F2 are concyclic
theorem points_concyclic (C : ellipse) (P F1 F2 Q : point) (l1 l2 : line) (M1 M2 : point) :
  point_on_ellipse P C →
  line_parallel_to_foci P F1 F2 l1 l2 →
  lines_intersect_at_Q M1 F1 M2 F2 →
  concyclic P F1 Q F2 :=
  sorry

end points_concyclic_l614_614276


namespace largest_in_column_smallest_in_row_l614_614105

def array : List (List Nat) := 
  [[20, 5, 9, 4, 3], 
   [22, 12, 25, 19, 15], 
   [13, 6, 9, 10, 18], 
   [23, 11, 30, 21, 7], 
   [18, 8, 10, 14, 6]]

def col_max (arr : List (List Nat)) (col : Nat) : Nat :=
  List.foldr max 0 (List.map (fun row => List.getD row col 0) arr)

def row_min (arr : List (List Nat)) (row : Nat) : Nat :=
  List.foldr min (List.headD (List.getD arr row [])) (List.getD arr row [])

theorem largest_in_column_smallest_in_row : 
  col_max array 1 = 12 ∧ row_min array 1 = 12 :=
by
  -- The proof is omitted
  sorry

end largest_in_column_smallest_in_row_l614_614105


namespace _l614_614563

variable {α : Type*} [LinearOrderedField α] {a b c : α} {C : Real}

def cosine_theorem (a b c : α) : α :=
  a^2 + b^2 - 2 * a * b * Real.cos C

lemma find_angle (h : (a^2 + b^2 - c^2) * Real.tan C = a * b) :
  C = π / 6 ∨ C = 5 * π / 6 := 
sorry

end _l614_614563


namespace probability_geometric_progression_l614_614339

theorem probability_geometric_progression :
  let outcomes := 6^3
  let favorable := 6
  (favorable : ℚ) / outcomes = 1 / 36 :=
by
  let outcomes := 6^3
  let favorable := 6
  show (favorable : ℚ) / outcomes = 1 / 36
  sorry

end probability_geometric_progression_l614_614339


namespace find_ordered_pair_l614_614483

theorem find_ordered_pair (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) := by
  sorry

end find_ordered_pair_l614_614483


namespace initial_sum_of_money_l614_614058

theorem initial_sum_of_money (A r t P : ℝ) (hA : A = 15500) (hr : r = 0.06) (ht : t = 4) :
  P = A / (1 + (r * t)) → P = 12500 :=
by
  intro h
  rw [hA, hr, ht]
  sorry

end initial_sum_of_money_l614_614058


namespace value_of_a_pow_b_l614_614550

theorem value_of_a_pow_b (a b : ℝ) (h : √(a + 2) + (b - 3)^2 = 0) : a^b = -8 :=
by {
  sorry
}

end value_of_a_pow_b_l614_614550


namespace statement_true_when_b_le_a_div_5_l614_614552

theorem statement_true_when_b_le_a_div_5
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h₀ : ∀ x : ℝ, f x = 5 * x + 3)
  (h₁ : ∀ x : ℝ, |f x + 7| < a ↔ |x + 2| < b)
  (h₂ : 0 < a)
  (h₃ : 0 < b) :
  b ≤ a / 5 :=
by
  sorry

end statement_true_when_b_le_a_div_5_l614_614552


namespace swimming_class_attendance_l614_614714

def total_students : ℕ := 1000
def chess_ratio : ℝ := 0.25
def swimming_ratio : ℝ := 0.50

def chess_students := chess_ratio * total_students
def swimming_students := swimming_ratio * chess_students

theorem swimming_class_attendance :
  swimming_students = 125 :=
by
  sorry

end swimming_class_attendance_l614_614714


namespace monotone_decreasing_range_of_a_l614_614947

noncomputable def f (a : ℝ) : ℝ → ℝ := 
λ x, if x < 2 then (a - 1) * x - 2 * a else Real.log x / Real.log a

theorem monotone_decreasing_range_of_a (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  (∀ x y : ℝ, x ≤ y → f a y ≤ f a x) ↔ (Real.sqrt 2 / 2 ≤ a ∧ a < 1) :=
sorry

end monotone_decreasing_range_of_a_l614_614947


namespace newspaper_subscription_cost_per_month_l614_614413

theorem newspaper_subscription_cost_per_month (M : ℝ)
    (monthly_cost_annual : 12 * M)
    (discounted_annual_cost : 0.80 * monthly_cost_annual = 96) :
    M = 10 := 
by
  sorry

end newspaper_subscription_cost_per_month_l614_614413


namespace ab_value_l614_614274

theorem ab_value (a b : ℝ) (h₁ : ∀ x, f 1 x = a * x + b)
  (h₂ : ∀ n x, f (n + 1) x = f (f n x))
  (h₃ : ∀ x, f 5 x = 32 * x + 93) : a * b = 6 :=
sorry

end ab_value_l614_614274


namespace diagonal_of_rectangular_prism_l614_614091

theorem diagonal_of_rectangular_prism
  (width height depth : ℕ)
  (h1 : width = 15)
  (h2 : height = 20)
  (h3 : depth = 25) : 
  (width ^ 2 + height ^ 2 + depth ^ 2).sqrt = 25 * (2 : ℕ).sqrt :=
by {
  sorry
}

end diagonal_of_rectangular_prism_l614_614091


namespace current_rate_l614_614707

variable (c : ℝ)

def still_water_speed : ℝ := 3.6

axiom rowing_time_ratio (c : ℝ) : (2 : ℝ) * (still_water_speed - c) = still_water_speed + c

theorem current_rate : c = 1.2 :=
by
  sorry

end current_rate_l614_614707


namespace sum_f_1_to_23_l614_614850

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom symmetry_g (x : ℝ) : g (1 + x) = g (1 - x)
axiom f_minus_g (x : ℝ) : f x - g x = 1
axiom f_plus_g (x : ℝ) : f (x + 1) + g (2 - x) = 1
axiom g_one : g 1 = 3

theorem sum_f_1_to_23 : (finset.range 23).sum (λ i, f (i + 1)) = 26 :=
sorry

end sum_f_1_to_23_l614_614850


namespace wastewater_pool_emptied_purification_complete_l614_614081

theorem wastewater_pool_emptied (initial_wastewater : ℕ) (monthly_addition : ℕ) (initial_discharge : ℕ) (discharge_increase : ℕ)
  (n : ℕ)
  (h_initial : initial_wastewater = 800)
  (h_monthly_addition : monthly_addition = 2)
  (h_initial_discharge : initial_discharge = 10)
  (h_discharge_increase : discharge_increase = 2)
  (h_months : n = 25) :
  let S_n := (n * (2 * initial_discharge + (n - 1) * discharge_increase)) / 2 in
  S_n ≥ initial_wastewater + n * monthly_addition :=
sorry

theorem purification_complete (initial_wastewater : ℕ) (monthly_addition : ℕ) (initial_discharge : ℕ) (discharge_increase : ℕ)
  (purification_start : ℕ) (initial_purification : ℕ) (purification_rate : ℕ) (n : ℕ)
  (h_initial : initial_wastewater = 800)
  (h_monthly_addition : monthly_addition = 2)
  (h_initial_discharge : initial_discharge = 10)
  (h_discharge_increase : discharge_increase = 2)
  (h_purification_start : purification_start = 7)
  (h_initial_purification : initial_purification = 5)
  (h_purification_rate : purification_rate = 20)
  (h_months : n = 20) :
  let a_n := 2 * (n - purification_start + 1) + 8 in
  let b_n := initial_purification * (1.2 ^ (n - purification_start)) in
  b_n ≥ a_n :=
sorry

end wastewater_pool_emptied_purification_complete_l614_614081


namespace prism_volume_l614_614338

theorem prism_volume (a b c : ℝ) (h1 : a * b = 45) (h2 : b * c = 49) (h3 : a * c = 56) : a * b * c = 1470 := by
  sorry

end prism_volume_l614_614338


namespace min_deviation_from_zero_in_interval_l614_614486

noncomputable def min_deviation_poly := x^2 - (1 / 2)

theorem min_deviation_from_zero_in_interval : 
  ∀ (P : ℝ → ℝ), (∃ p q : ℝ, ∀ x : ℝ, P x = x^2 + p * x + q) → 
  ∃ d : ℝ, (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (|P x| ≤ d)) ∧ (∃ P_min : ℝ → ℝ, 
      (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (|P_min x| = ((x^2 - (1 / 2)) x))) ∧
      (d = 1 / 2)) := 
by
  sorry

end min_deviation_from_zero_in_interval_l614_614486


namespace paco_cookies_left_l614_614624

variable (initial_cookies : ℕ)
variable (paco_eats_day1 : ℕ)
variable (maria_fraction : ℚ)
variable (paco_fraction_day2 : ℚ)

theorem paco_cookies_left :
  initial_cookies = 93 →
  paco_eats_day1 = 15 →
  maria_fraction = 2 / 3 →
  paco_fraction_day2 = 1 / 10 →
  let cookies_after_paco_day1 := initial_cookies - paco_eats_day1 in
  let cookies_after_maria := cookies_after_paco_day1 - (maria_fraction * cookies_after_paco_day1).natAbs in
  let paco_second_day_eat := (paco_fraction_day2 * cookies_after_maria).natAbs in
  let cookies_left := cookies_after_maria - paco_second_day_eat in
  cookies_left = 24 :=
by intros; sorry

end paco_cookies_left_l614_614624


namespace handshaking_mod_1000_l614_614568

-- Define the conditions as Lean definitions
def group_of_twelve_people := Fin 12
def shakes_hands_with_exactly_three (p : group_of_twelve_people) : Finset group_of_twelve_people := sorry
def number_of_handshaking_arrangements := sorry

-- We need to prove the main statement
theorem handshaking_mod_1000 : number_of_handshaking_arrangements % 1000 = 50 := by
  sorry

end handshaking_mod_1000_l614_614568


namespace max_members_sports_team_l614_614670

theorem max_members_sports_team : 
  (∀ (team : Finset ℕ), 
  (∀ (x y z ∈ team), 
    x ≠ y → x ≠ z → y ≠ z → x ≠ y + z ∧ y ≠ x + z ∧ z ≠ x + y) ∧ 
  (∀ (x y ∈ team), x ≠ y → x ≠ 2 * y ∧ y ≠ 2 * x) → 
  team.card ≤ 50) :=
by 
-- proof omitted 
sorry

end max_members_sports_team_l614_614670


namespace AM_eq_DK_plus_BM_l614_614150

section square_problem

variables {A B C D K M : Type*}

-- Definitions of the points
variables [has_point A] [has_point B] [has_point C] [has_point D]
variables [has_point K] [has_point M]

-- Conditions
variable (h1 : is_square A B C D)  -- Square ABCD
variable (h2 : M ∈ segment B C)    -- M is on the side BC
variable (h3 : is_angle_bisector (angle D A M) A K) -- Line AK bisects ∠D A M
variable (h4 : K ∈ segment D C)    -- K is on the line DC

-- Statement to prove
theorem AM_eq_DK_plus_BM : AM = DK + BM :=
sorry

end square_problem

end AM_eq_DK_plus_BM_l614_614150


namespace unique_magnitude_of_roots_l614_614937

theorem unique_magnitude_of_roots (z : ℂ) (h : z^2 - 6 * z + 20 = 0) : 
  ∃! (c : ℝ), c = Complex.abs z := 
sorry

end unique_magnitude_of_roots_l614_614937


namespace value_of_E_neg3_l614_614775

-- Define the function E : ℝ → ℝ
variable (E : ℝ → ℝ)

-- Define the condition that the point (-3, 4) is on the graph of E
axiom contains_point : E (-3) = 4

-- Lean statement to prove that E(-3) = 4
theorem value_of_E_neg3 : E (-3) = 4 := 
by
  exact contains_point

end value_of_E_neg3_l614_614775


namespace range_of_a_l614_614872

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(1-x)
  else 1 - Real.log2 x

theorem range_of_a (a : ℝ) : |f a| >= 2 → a ∈ Iic (1/2) ∨ a ∈ Ici 8 := by
  sorry

end range_of_a_l614_614872


namespace sum_of_values_of_n_l614_614371

theorem sum_of_values_of_n (n : ℚ) (h : |3 * n - 4| = 6) : 
  (n = 10 / 3 ∨ n = -2 / 3) → (10 / 3 + -2 / 3 = 8 / 3) :=
sorry

end sum_of_values_of_n_l614_614371


namespace system_of_equations_solution_l614_614639

theorem system_of_equations_solution :
  ∀ (x y: ℝ), (x + 2 * y = 2) ∧ (3 * x - 4 * y = -24) → (x = -4 ∧ y = 3) :=
by
  intros x y h
  cases h with h1 h2
  sorry

end system_of_equations_solution_l614_614639


namespace sum_of_perpendiculars_l614_614722

noncomputable def regular_pentagon (A B C D E : ℝ) : Prop :=
∃ (s : ℝ), s > 0 ∧ A = 0 ∧ B = s ∧ C = s * (1 + cos (2 * Real.pi / 5)) ∧ D = s * (1 + 2 * cos (4 * Real.pi / 5)) ∧ E = s * (1 + cos (2 * Real.pi / 5))

noncomputable def perpendiculars (A P Q R : ℝ) : Prop :=
∃ (AP AQ AR : ℝ), 
  AP = ‖A - P‖ ∧ 
  AQ = ‖A - Q‖ ∧ 
  AR = ‖A - R‖

noncomputable def center_and_side_length (O : ℝ) (s : ℝ) : Prop :=
∃ (OP AO : ℝ), OP = 2 ∧ s > 0

theorem sum_of_perpendiculars 
  (A B C D E O P Q R : ℝ)
  (h1 : regular_pentagon A B C D E)
  (h2 : perpendiculars A P Q R)
  (h3 : center_and_side_length O s) :
  (AO + AQ + AR = 8) := sorry

end sum_of_perpendiculars_l614_614722


namespace cube_sum_gt_zero_l614_614260

variable {x y z : ℝ}

theorem cube_sum_gt_zero (h1 : x < y) (h2 : y < z) : 
  (x - y)^3 + (y - z)^3 + (z - x)^3 > 0 :=
sorry

end cube_sum_gt_zero_l614_614260


namespace length_of_segment_l614_614660

theorem length_of_segment :
  let line := {p : ℝ × ℝ | p.1 + √3 * p.2 = 2}
  let circle := {p : ℝ × ℝ | (p.1 - 1) ^ 2 + (p.2) ^ 2 = 1}
  ∀ p1 p2 : ℝ × ℝ, p1 ∈ line → p2 ∈ line → p1 ∈ circle → p2 ∈ circle → dist p1 p2 = √3 :=
sorry

end length_of_segment_l614_614660


namespace min_varphi_value_l614_614224

def z1 := 2
def z2 := -3

def is_on_upper_semicircle (z : ℂ) : Prop :=
  ∃ θ : ℝ, (0 < θ) ∧ (θ < π) ∧ z = complex.exp (θ * complex.I)

noncomputable def min_varphi (z : ℂ) : ℝ :=
  let α := complex.arg (z - z1)
  let β := complex.arg (z - z2)
  α - β

theorem min_varphi_value :
  ∃ z : ℂ, is_on_upper_semicircle z ∧ min_varphi z = π - real.arctan (5 * real.sqrt 6 / 12) := sorry

end min_varphi_value_l614_614224


namespace zero_vector_if_power_zero_l614_614688

def delta (v w : Vector ℝ n) : Vector ℝ n :=
  λ j, ∑ i in Finset.range n, v[i] * w[(j + 2 - i) % n]

def vector_power (v : Vector ℝ n) : ℕ → Vector ℝ n
| 1       := v
| (k + 1) := delta v (vector_power v k)

theorem zero_vector_if_power_zero
  (v : Vector ℝ n)
  (k : ℕ)
  (h : vector_power v k = 0) : 
  v = 0 :=
sorry

end zero_vector_if_power_zero_l614_614688


namespace minimum_n_for_prime_subsets_l614_614303

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def S : Set ℕ := { x | 1 ≤ x ∧ x ≤ 2005 }

def pairwise_coprime (A : Set ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ A → b ∈ A → a ≠ b → coprime a b

def contains_prime (A : Set ℕ) : Prop :=
  ∃ p : ℕ, p ∈ A ∧ Nat.Prime p

theorem minimum_n_for_prime_subsets :
  ∀ (A : Set ℕ), A ⊆ S →
  (∀ A', pairwise_coprime A' → A' ⊆ A → contains_prime A') →
  16 ∈ A :=
begin
  sorry
end

end minimum_n_for_prime_subsets_l614_614303


namespace an_correct_bn_correct_l614_614656

noncomputable def f1 (x : ℝ) : ℝ := 4 * (x - x^2)

def fn (n : ℕ) : (ℝ → ℝ) → (ℝ → ℝ) 
| 1 := f1
| (n+1) := fn (n) ∘ f1

def an (n : ℕ) : ℕ
| 1 := 1
| (n+1) := 2 * an(n)

def bn (n : ℕ) : ℕ
| 1 := 2
| (n+1) := bn(n) + an(n)

theorem an_correct (n : ℕ) : 
  an n = 2^(n-1) := 
sorry 

theorem bn_correct (n : ℕ) : 
  bn n = 2^(n-1) + 1 := 
sorry 

end an_correct_bn_correct_l614_614656


namespace find_original_number_l614_614195

theorem find_original_number (x : ℝ) 
  (h1 : x * 16 = 3408) 
  (h2 : 1.6 * 21.3 = 34.080000000000005) : 
  x = 213 :=
sorry

end find_original_number_l614_614195


namespace geometric_sequence_178th_term_l614_614957

-- Conditions of the problem as definitions
def first_term : ℤ := 5
def second_term : ℤ := -20
def common_ratio : ℤ := second_term / first_term
def nth_term (a : ℤ) (r : ℤ) (n : ℕ) : ℤ := a * r^(n-1)

-- The translated problem statement in Lean 4
theorem geometric_sequence_178th_term :
  nth_term first_term common_ratio 178 = -5 * 4^177 :=
by
  repeat { sorry }

end geometric_sequence_178th_term_l614_614957


namespace probability_of_sums_l614_614735

def first_die := [1, 2, 3, 3, 4, 4]
def second_die := [2, 3, 5, 6, 7, 7]
def target_sums := [7, 9, 11]

noncomputable def probability_correct : Prop := 
  let total_outcomes := first_die.length * second_die.length;
  let favorable_outcomes := (first_die.map (λ x, second_die.count (λ y, x + y ∈ target_sums))).sum;
  (favorable_outcomes / total_outcomes.toRat) = (2 / 9)

theorem probability_of_sums : probability_correct := 
by
  sorry

end probability_of_sums_l614_614735


namespace annual_interest_l614_614633

noncomputable theory
open_locale big_operators

def P1 := 1800
def P2 := 3600 - P1
def rate1 := 3 / 100
def rate2 := 5 / 100
def interest1 := P1 * rate1
def interest2 := P2 * rate2
def total_interest := interest1 + interest2

theorem annual_interest : total_interest = 144 :=
by
-- Sorry is used to skip the proof
sorry

end annual_interest_l614_614633


namespace sum_of_odd_integers_divisible_by_3_between_200_and_600_l614_614693

theorem sum_of_odd_integers_divisible_by_3_between_200_and_600 :
  let is_odd := λ n, n % 2 = 1
  let is_div_by_3 := λ n, n % 3 = 0
  let in_range := λ n, 200 < n ∧ n < 600
  let satisfies_conditions := λ n, is_odd n ∧ is_div_by_3 n ∧ in_range n
  let sequence := finset.filter satisfies_conditions (finset.Ico 200 600)
  let series_sum := sequence.sum
  series_sum = 26000 :=
by
  sorry

end sum_of_odd_integers_divisible_by_3_between_200_and_600_l614_614693


namespace height_of_second_triangle_l614_614388

theorem height_of_second_triangle
  (base1 : ℝ) (height1 : ℝ) (base2 : ℝ) (height2 : ℝ)
  (h_base1 : base1 = 15)
  (h_height1 : height1 = 12)
  (h_base2 : base2 = 20)
  (h_area_relation : (base2 * height2) / 2 = 2 * (base1 * height1) / 2) :
  height2 = 18 :=
sorry

end height_of_second_triangle_l614_614388


namespace exists_geometric_seq_l614_614507

noncomputable def a_n (n : ℕ) : ℕ := 3^n - 1

def C (n k : ℕ) : ℕ := nat.choose n k

def geometric_seq_b (n : ℕ) : ℕ := 2^n

theorem exists_geometric_seq (n : ℕ) (hn : 0 < n) : 
  ∃ b : ℕ → ℕ, (∀ k : ℕ, b k = geometric_seq_b k) ∧ 
  (a_n n = ∑ k in finset.range (n + 1), b k * C n k) := 
by
  use geometric_seq_b
  split
  { intro k
    refl }
  { sorry }

end exists_geometric_seq_l614_614507


namespace smallest_possible_n_l614_614268

theorem smallest_possible_n (x : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ i, i < n → | x i | < 2)
  (h2 : (finset.range n).sum (λ i, | x i |) = 30 + | (finset.range n).sum (λ i, x i) |) :
  n ≥ 16 :=
sorry

end smallest_possible_n_l614_614268


namespace find_x_l614_614928

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 :=
by
  have h1 : 32 = 2^5 := by norm_num
  rw [h1, pow_mul] at h
  have h2 : 2^(10) = 2^(5*x) := by exact h
  have h3 : 10 = 5 * x := by exact (pow_inj h2).2
  linarith

end find_x_l614_614928


namespace point_B_in_first_quadrant_l614_614221

theorem point_B_in_first_quadrant 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : -b > 0) : 
  (a > 0) ∧ (b > 0) := 
by 
  sorry

end point_B_in_first_quadrant_l614_614221


namespace find_x_eq_2_l614_614926

theorem find_x_eq_2 : ∀ x : ℝ, 2^10 = 32^x → x = 2 := 
by 
  intros x h
  sorry

end find_x_eq_2_l614_614926


namespace new_deck_card_count_l614_614974

-- Define the conditions
def cards_per_time : ℕ := 30
def times_per_week : ℕ := 3
def weeks : ℕ := 11
def decks : ℕ := 18
def total_cards_tear_per_week : ℕ := cards_per_time * times_per_week
def total_cards_tear : ℕ := total_cards_tear_per_week * weeks
def total_cards_in_decks (cards_per_deck : ℕ) : ℕ := decks * cards_per_deck

-- Define the theorem we need to prove
theorem new_deck_card_count :
  ∃ (x : ℕ), total_cards_in_decks x = total_cards_tear ↔ x = 55 := by
  sorry

end new_deck_card_count_l614_614974


namespace inner_product_sum_zero_l614_614598

variables {𝕜 : Type*} [NormedAddCommGroup𝕜]
            [InnerProductSpace 𝕜 ℝ]

variables (a b c : ℝ)

h0 : ‖a‖ = 2 := sorry
h1 : ‖b‖ = 3 := sorry
h2 : ‖c‖ = 4 := sorry
h3 : a + b + c = 0 := sorry

theorem inner_product_sum_zero :
  a • b + a • c + b • c = -(29 / 2) := sorry

end inner_product_sum_zero_l614_614598


namespace smallest_c_over_a_plus_b_l614_614754

theorem smallest_c_over_a_plus_b (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  ∃ d : ℝ, d = (c / (a + b)) ∧ d = (Real.sqrt 2 / 2) :=
by
  sorry

end smallest_c_over_a_plus_b_l614_614754


namespace trig_identity_l614_614300

theorem trig_identity (A B : ℝ) (h1 : A = 96) (h2 : B = 24) 
  (cos_add : ∀ A B : ℝ, cos (A + B) = cos A * cos B - sin A * sin B) 
  (cos_120 : cos 120 = -1/2) :
  cos 96 * cos 24 - sin 96 * sin 24 = -1/2 :=
by
  sorry

end trig_identity_l614_614300


namespace exists_integers_A_B_C_l614_614098

theorem exists_integers_A_B_C (a b : ℚ) (N_star : Set ℕ) (Q : Set ℚ)
  (h : ∀ x ∈ N_star, (a * (x : ℚ) + b) / (x : ℚ) ∈ Q) : 
  ∃ A B C : ℤ, ∀ x ∈ N_star, 
    (a * (x : ℚ) + b) / (x : ℚ) = (A * (x : ℚ) + B) / (C * (x : ℚ)) := 
sorry

end exists_integers_A_B_C_l614_614098


namespace baby_grasshoppers_l614_614253

-- Definition for the number of grasshoppers on the plant
def grasshoppers_on_plant : ℕ := 7

-- Definition for the total number of grasshoppers found
def total_grasshoppers : ℕ := 31

-- The theorem to prove the number of baby grasshoppers under the plant
theorem baby_grasshoppers : 
  (total_grasshoppers - grasshoppers_on_plant) = 24 := 
by
  sorry

end baby_grasshoppers_l614_614253


namespace range_of_m_l614_614136

open Real

noncomputable def is_valid_m (m : ℝ) : Prop :=
  ∀ a b c x : ℝ, a^2 + b^2 + c^2 = 1 → (sqrt 2 * a + sqrt 3 * b + 2 * c) ≤ abs (x - 1) + abs (x + m)

theorem range_of_m :
  ∀ m : ℝ, is_valid_m m ↔ m ∈ set.Iic (-4) ∪ set.Ici 2 :=
  sorry

end range_of_m_l614_614136


namespace pair_sum_53_l614_614626

theorem pair_sum_53 (S : Finset ℕ) (hS : S.card = 53)
  (hS_bound : ∀ n ∈ S, n ≤ 1990) : ∃ a b ∈ S, a + b = 53 :=
begin
  sorry
end

end pair_sum_53_l614_614626


namespace total_bottles_remaining_l614_614759

def initial_small_bottles : ℕ := 6000
def initial_big_bottles : ℕ := 15000
def initial_medium_bottles : ℕ := 5000

def small_bottles_sold : ℕ := (11 * initial_small_bottles) / 100
def big_bottles_sold : ℕ := (12 * initial_big_bottles) / 100
def medium_bottles_sold : ℕ := (8 * initial_medium_bottles) / 100

def small_bottles_damaged : ℕ := (3 * initial_small_bottles) / 100
def big_bottles_damaged : ℕ := (2 * initial_big_bottles) / 100
def medium_bottles_damaged : ℕ := (4 * initial_medium_bottles) / 100

def remaining_small_bottles : ℕ := initial_small_bottles - small_bottles_sold - small_bottles_damaged
def remaining_big_bottles : ℕ := initial_big_bottles - big_bottles_sold - big_bottles_damaged
def remaining_medium_bottles : ℕ := initial_medium_bottles - medium_bottles_sold - medium_bottles_damaged

def total_remaining_bottles : ℕ := remaining_small_bottles + remaining_big_bottles + remaining_medium_bottles

theorem total_bottles_remaining :
  total_remaining_bottles = 22560 :=
by
  -- sell amounts
  have small_sold := (660 : ℕ)
  have big_sold := (1800 : ℕ)
  have medium_sold := (400 : ℕ)

  -- damaged amounts
  have small_damaged := (180 : ℕ)
  have big_damaged := (300 : ℕ)
  have medium_damaged := (200 : ℕ)

  -- remaining counts
  have rem_small := (5160 : ℕ)
  have rem_big := (12900 : ℕ)
  have rem_medium := (4400 : ℕ)

  -- total remaining
  have total := (22560 : ℕ)

  -- compute and assert equality
  calc
    total_remaining_bottles
    = remaining_small_bottles + remaining_big_bottles + remaining_medium_bottles : rfl
    ... = 5160 + 12900 + 4400 : by sorry
    ... = 22560 : rfl


end total_bottles_remaining_l614_614759


namespace false_of_p_and_q_l614_614855

def p (a : ℝ) : Prop := ∀ x ∈ set.Icc (0 : ℝ) 1, a ≥ Real.exp x
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 4 * x₀ + a = 0

theorem false_of_p_and_q (a : ℝ) :
  ¬ (p a ∧ q a) → a ∈ set.Ioc 4 0 ∪ set.Ioo 0 Real.e :=
sorry

end false_of_p_and_q_l614_614855


namespace magnitude_unique_value_l614_614934

-- Define the quadratic equation
def quadratic_eq (z : ℂ) : Prop := z^2 - 6 * z + 20 = 0

-- Statement of the problem
theorem magnitude_unique_value (z : ℂ) (h : quadratic_eq z) : ∃! (m : ℝ), m = complex.abs z :=
by
  sorry

end magnitude_unique_value_l614_614934


namespace added_water_volume_l614_614397

def initial_volume := 20
def alcohol_percentage_initial := 0.20
def alcohol_volume_initial := initial_volume * alcohol_percentage_initial
def new_alcohol_percentage := 17.391304347826086 / 100
def new_alcohol_percentage_fraction := 4 / 23

theorem added_water_volume (x : ℕ) :
  (alcohol_volume_initial / (initial_volume + x) = new_alcohol_percentage_fraction) → x = 3 :=
begin
  sorry
end

end added_water_volume_l614_614397


namespace num_distinct_differences_is_five_l614_614179

-- Condition: The set
def my_set : Set ℕ := {1, 2, 3, 4, 5, 6}

-- To Prove: The number of different positive integers that can be represented as a difference of two distinct members of the set is 5.
theorem num_distinct_differences_is_five : 
  (Finset.card (Finset.filter (λ x : ℕ, x > 0) 
  (Finset.image (λ (p : ℕ × ℕ), p.1 - p.2) 
  (Finset.filter (λ (p : ℕ × ℕ), p.1 ≠ p.2)
  (Finset.product (Finset.filter (λ n : ℕ, n ∈ my_set) (Finset.range 7)) 
  (Finset.filter (λ n : ℕ, n ∈ my_set) (Finset.range 7))))))) = 5 := by
  sorry

end num_distinct_differences_is_five_l614_614179


namespace parabola_and_hyperbola_equations_l614_614866

theorem parabola_and_hyperbola_equations 
  (a b : ℝ) (h₁ : a > b) (h₂ : b > 0)
  (vertex_origin : vertex parabola = (0, 0))
  (focus_hyperbola : ∃ (f : ℝ), (a > 0) ∧ (b > 0) ∧ (f = ((a^2+b^2)^0.5)) )
  (directrix_perpendicular : ∃ (d : ℝ), (d = -((a^2-b^2)^0.5)))
  (intersection_point : (3/2, sqrt 6) ∈ parabola ∧ (3/2, sqrt 6) ∈ hyperbola) :
  (∃ (c : ℝ), (c = 1) ∧ (parabola_eq = y^2 = 4x)) ∧ 
  (∃ (a : ℝ), (a^2 = 1/4) ∧ (hyperbola_eq = 4x^2 - 4y^2/3 = 1)) := 
sorry

end parabola_and_hyperbola_equations_l614_614866


namespace continuous_at_3_l614_614608

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ :=
if x ≤ 3 then 4 * x^2 + 1 else b * x + c

theorem continuous_at_3 (b c : ℝ) : 
  (3 * b + c = 37) → 
  ContinuousAt (λ x, f x b c) 3 :=
by
  sorry

end continuous_at_3_l614_614608


namespace last_term_arithmetic_progression_eq_62_l614_614119

theorem last_term_arithmetic_progression_eq_62
  (a : ℕ) (d : ℕ) (n : ℕ) 
  (h_a : a = 2)
  (h_d : d = 2)
  (h_n : n = 31) : 
  a + (n - 1) * d = 62 :=
by
  sorry

end last_term_arithmetic_progression_eq_62_l614_614119


namespace arrangement_exists_l614_614954

structure Student (C : Type) :=
(courses : Set C)

structure Dormitory (S : Type) :=
(roommates : S × S)

section
variables {C S : Type} [Fintype C] [Fintype S]
  (students : Finset S)
  (dormitories : Finset (Dormitory S))
  (n : ℕ)
  [DecidableEq S]

noncomputable def valid_arrangement (students : Finset S) (dormitories : Finset (Dormitory S)) : 
  Prop :=
  ∃ (circle : list S), 
    (∀ d ∈ dormitories, ∃ i j, d.roommates = (circle.nth i, circle.nth j) ∧ (i + 1 = j ∨ j + 1 = i)) ∧
    (∀ i j, (students i ∈ students ∧ students j ∈ students ∧ i ≠ j ∧ students i ∈ list.sublist circle ∧ students j ∈ list.sublist circle)
      → (is_adjacent circle i j 
        → (students i).courses ⊆ (students j).courses ∨ (students j).courses ⊆ (students i).courses)
      → |(students i).courses \ (students j).courses| = 1)

theorem arrangement_exists (h1 : Fintype.card S = 2^n)
  (h2 : Fintype.card C = n)
  (h3 : Fintype.card dormitories = 2^(n-1))
  (h4 : ∀ (s1 s2 : S), s1 ≠ s2 → (students s1).courses ≠ (students s2).courses) :
  valid_arrangement students dormitories :=
sorry
end

end arrangement_exists_l614_614954


namespace smallest_i_satisfies_condition_l614_614889

def sequence_def (i : ℕ) : ℕ :=
if i = 1 then 0
else let k := Nat.floor (Real.log2 i) in k*k

theorem smallest_i_satisfies_condition : ∃ i, (sequence_def i + sequence_def (2*i) ≥ 100) ∧ (∀ j, j < i → sequence_def j + sequence_def (2*j) < 100) :=
sorry

end smallest_i_satisfies_condition_l614_614889


namespace four_de_ge_ab_plus_ac_l614_614838

theorem four_de_ge_ab_plus_ac (A B C F D E: Type*) 
  [ordered_comm_ring F]
  [ℚ-linear_ring F]
  (angle_AFB : F) (angle_BFC : F) (angle_CFA : F) (angle_120 : angle_AFB = 120 ∧ angle_BFC = 120 ∧ angle_CFA = 120)
    (intersect_BF_CF: BF ∩ AC = D ∧ CF ∩ AB = E): 
  AB + AC ≥ 4 * DE :=
begin
  sorry
end

end four_de_ge_ab_plus_ac_l614_614838


namespace equation_of_the_circle_l614_614472

def circle_center_line (a : ℝ) : Prop := 2 * a - (2 * a - 3) - 3 = 0

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem equation_of_the_circle (a : ℝ) :
  circle_center_line a ∧ distance_squared (a, 2 * a - 3) (5, 2) = distance_squared (a, 2 * a - 3) (3, 2) → 
  (∃ (h k : ℝ) (r : ℝ), (h = 4 ∧ k = 5 ∧ r^2 = 10) ∧ (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2)) :=
by
  intros hc_eq hd_eq
  sorry

end equation_of_the_circle_l614_614472


namespace monotonic_increasing_odd_function_implies_a_eq_1_find_max_m_l614_614494

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- 1. Monotonicity of f(x)
theorem monotonic_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 := sorry

-- 2. f(x) is odd implies a = 1
theorem odd_function_implies_a_eq_1 (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 1 := sorry

-- 3. Find max m such that f(x) ≥ m / 2^x for all x ∈ [2, 3]
theorem find_max_m (h : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f 1 x ≥ m / 2^x) : m ≤ 12/5 := sorry

end monotonic_increasing_odd_function_implies_a_eq_1_find_max_m_l614_614494


namespace ordered_pair_exists_l614_614485

theorem ordered_pair_exists (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) :=
by
  sorry

end ordered_pair_exists_l614_614485


namespace smaller_molds_radius_l614_614410

open Real

noncomputable def vol_hemisphere (r : ℝ) : ℝ := (2 / 3) * π * r^3

theorem smaller_molds_radius :
  let R := 2 in
  let total_volume := vol_hemisphere R in
  let s in 
  (8 * vol_hemisphere s = total_volume) -> s = 1 :=
by
  intros R total_volume s h
  sorry

end smaller_molds_radius_l614_614410


namespace delta_k_vn_zero_l614_614128

noncomputable def v (n : ℕ) := n^4 + 2 * n^2

def Δ (k : ℕ) (v : ℕ → ℤ) : ℕ → ℤ
| 0, v => v
| (k+1), v => λ n, (Δ k v (n+1)) - (Δ k v n)

theorem delta_k_vn_zero (k : ℕ) : (Δ k v) = 0 ↔ k ≥ 4 := by
  sorry

end delta_k_vn_zero_l614_614128


namespace max_kopeyka_coins_l614_614017

def coins (n : Nat) (k : Nat) : Prop :=
  k ≤ n / 4 + 1

theorem max_kopeyka_coins : coins 2001 501 :=
by
  sorry

end max_kopeyka_coins_l614_614017


namespace water_fraction_final_l614_614029

theorem water_fraction_final (initial_volume : ℚ) (removed_volume : ℚ) (replacements : ℕ) (water_initial_fraction : ℚ) :
  initial_volume = 20 ∧ removed_volume = 5 ∧ replacements = 5 ∧ water_initial_fraction = 1 ->
  let water_fraction := water_initial_fraction * (3 / 4)^replacements in
  water_fraction = 243 / 1024 :=
by
  sorry

end water_fraction_final_l614_614029


namespace find_modulus_of_z_l614_614522

-- Definitions based on conditions
def imaginary_unit : ℂ := complex.i

def z (c : ℂ) : ℂ := c / (1 + imaginary_unit)

-- The main statement to prove
theorem find_modulus_of_z (c : ℂ) (hc : c * (1 + imaginary_unit) = imaginary_unit) : abs (z c) = real.sqrt 2 / 2 :=
by
  sorry

end find_modulus_of_z_l614_614522


namespace semicircle_radius_l614_614972

-- We state the conditions and goal using Lean
theorem semicircle_radius (P Q R : Type*) [InnerProductSpace ℝ P]
  (is_right_triangle : angle P Q R = π / 2)
  (PQ_diameter : ∃ r₁ : ℝ, 18 * π = 1 / 2 * π * r₁ * r₁)
  (PR_arc_length : ∃ r₂ : ℝ, 10 * π = π * r₂) :
  exists r₃ : ℝ, r₃ = sqrt 136 :=
sorry

end semicircle_radius_l614_614972


namespace number_of_valid_sets_l614_614610

open Finset

def M : Finset ℕ := (range 13).filter (λ n, 1 ≤ n)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def valid_subsets (S : Finset ℕ) : Finset (Finset ℕ) :=
  S.powerset.filter (λ A, A.card = 3 ∧ is_perfect_square (A.sum id))

theorem number_of_valid_sets : (valid_subsets M).card = 26 := sorry

end number_of_valid_sets_l614_614610


namespace circumscribable_1992_gon_circumscribable_1990_gon_l614_614635

theorem circumscribable_1992_gon : 
  ∃ (poly : Polygon), poly.sides = (Finset.range 1993) ∧ poly.circumscribable :=
  sorry
  
theorem circumscribable_1990_gon : 
  ∃ (poly : Polygon), poly.sides = (Finset.range 1991) ∧ poly.circumscribable :=
  sorry

end circumscribable_1992_gon_circumscribable_1990_gon_l614_614635


namespace unit_vector_b_correct_projection_onto_b_correct_l614_614497

variables (a b : ℝ × ℝ)

-- Conditions
def condition1 : Prop := 2 • a + b = (3, 3)
def condition2 : Prop := a - b = (3, 0)

-- Unit vector of b
def unit_vector_b : ℝ × ℝ := let b_norm := real.sqrt (b.1 ^ 2 + b.2 ^ 2) in (b.1 / b_norm, b.2 / b_norm)

-- Projection of a onto b
def projection_onto_b : ℝ :=
  let b_norm := real.sqrt (b.1 ^ 2 + b.2 ^ 2) in
  a.1 * (b.1 / b_norm) + a.2 * (b.2 / b_norm)

theorem unit_vector_b_correct (h1 : condition1 a b) (h2 : condition2 a b) : 
  unit_vector_b b = (-real.sqrt 2 / 2, real.sqrt 2 / 2) :=
sorry

theorem projection_onto_b_correct (h1 : condition1 a b) (h2 : condition2 a b) : 
  projection_onto_b a b = 0 :=
sorry

end unit_vector_b_correct_projection_onto_b_correct_l614_614497


namespace smallest_value_at_9_l614_614818

theorem smallest_value_at_9 :
  let x := 9 in 
  (1:ℚ/(
    (6:ℚ) / (x^2) < (6:ℚ) / (x-1) ∧
    (6:ℚ) / (x^2) < (6:ℚ) / (x+1) ∧
    (6:ℚ) / (x^2) < (6:ℚ) / x ∧
    (6:ℚ) / (x^2) < (x:ℚ) / 6 ∧
    (6:ℚ) / (x^2) < ((x:ℚ) + 1) / 6)) := sorry

end smallest_value_at_9_l614_614818


namespace point_B_in_first_quadrant_l614_614220

theorem point_B_in_first_quadrant 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : -b > 0) : 
  (a > 0) ∧ (b > 0) := 
by 
  sorry

end point_B_in_first_quadrant_l614_614220


namespace find_functions_l614_614984

variables {M : Type*} {n : ℕ}
def power_set (s : set M) := { t : set M | t ⊆ s }
def f (P: set M → ℕ) := ∀ (A B : set M),
  P A ≠ 0 ∧ P (A ∪ B) = P (A ∩ B) + P ((A ∪ B) \ (A ∩ B))

theorem find_functions (P : set M → ℕ) 
  (h1 : ∀ A, A ≠ ∅ → P A ≠ 0)
  (h2 : ∀ A B, P (A ∪ B) = P (A ∩ B) + P ((A ∪ B) \ (A ∩ B))) :
  (∀ X, P X = set.card X) :=
by
  sorry

end find_functions_l614_614984


namespace trigonometric_identity_l614_614721

theorem trigonometric_identity :
  4 * real.sin (80 * real.pi / 180) - real.cos (10 * real.pi / 180) / real.sin (10 * real.pi / 180) = -real.sqrt 3 := 
sorry

end trigonometric_identity_l614_614721


namespace minimum_stamps_l614_614446

theorem minimum_stamps (c f : ℕ) (h : 3 * c + 4 * f = 50) : c + f = 13 :=
sorry

end minimum_stamps_l614_614446


namespace bug_closest_point_l614_614727

noncomputable def bug_position : ℚ × ℚ :=
  let x_seq := [1, -1/4, 1/16, -1/64, 1/256, -1/1024, 1/4096, -1/16384, 1/65536, -1/262144, 1/1048576] in
  let y_seq := [0, 1/2, 0, -1/8, 0, 1/32, 0, -1/128, 0, 1/512, 0, -1/2048, 0, 1/8192, 0, -1/32768] in
  let sum_geom (seq : List ℚ) : ℚ :=
    (Seq.mapWith_index (λ n a, a * (r ^ n))).sum
  in
  (
    sum_geom x_seq,
    sum_geom y_seq 
  )

/-- The bug's final position is closest to (4/5, 2/5) -/
theorem bug_closest_point : bug_position = (4/5, 2/5) :=
  sorry

end bug_closest_point_l614_614727


namespace log_expression_identity_l614_614451

-- Definitions of the variables
variables (p q r s t z : ℝ)

-- The theorem statement
theorem log_expression_identity :
  log (p / q) + log (q / r) + 2 * log (r / s) - log (pt / sz) = log (rz / st) :=
by
  sorry

end log_expression_identity_l614_614451


namespace distance_from_D_to_plane_ABC_l614_614509

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p1 p2: Point3D) : Point3D := {
  x := p1.x - p2.x,
  y := p1.y - p2.y,
  z := p1.z - p2.z
}

def cross_product (v1 v2 : Point3D) : Point3D := {
  x := v1.y * v2.z - v1.z * v2.y,
  y := v1.z * v2.x - v1.x * v2.z,
  z := v1.x * v2.y - v1.y * v2.x
}

def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def length (v : Point3D) : ℝ := 
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

theorem distance_from_D_to_plane_ABC :
  let A := Point3D.mk 2 3 1
  let B := Point3D.mk 4 1 -2
  let C := Point3D.mk 6 3 7
  let D := Point3D.mk (-5) (-4) 8
  let AB := vector_sub B A
  let AC := vector_sub C A
  let AD := vector_sub D A
  let normal_vec := cross_product AB AC
  let distance := abs (dot_product normal_vec AD / length normal_vec)
  distance = 119 / Real.sqrt 277 := 
  sorry

end distance_from_D_to_plane_ABC_l614_614509


namespace jelly_bean_ratio_l614_614675

theorem jelly_bean_ratio (total_jelly_beans : ℕ) (coconut_flavored_jelly_beans : ℕ) (quarter_red_jelly_beans : ℕ) 
  (H_total : total_jelly_beans = 4000)
  (H_coconut : coconut_flavored_jelly_beans = 750)
  (H_quarter_red : quarter_red_jelly_beans = 1 / 4 * (4 * coconut_flavored_jelly_beans)) : 
  (quarter_red_jelly_beans * 4) / total_jelly_beans = 3 / 4 :=
by
  sorry

end jelly_bean_ratio_l614_614675


namespace number_of_shelves_l614_614067

/-- Adam could fit 11 action figures on each shelf -/
def action_figures_per_shelf : ℕ := 11

/-- Adam's shelves could hold a total of 44 action figures -/
def total_action_figures_on_shelves : ℕ := 44

/-- Prove the number of shelves in Adam's room -/
theorem number_of_shelves:
  total_action_figures_on_shelves / action_figures_per_shelf = 4 := 
by {
    sorry
}

end number_of_shelves_l614_614067


namespace number_of_data_points_l614_614681

-- Conditions
variables (y z : ℝ)
-- 6, y, z form an arithmetic sequence
axiom arithmetic_sequence : 6 + z = 2 * y
-- 6, y, z + 6 form a geometric sequence
axiom geometric_sequence : y^2 = 6 * (z + 6)

-- Hypothesis for stratified sampling
lemma stratified_sampling (N : ℝ) (P : ℝ) (a b c : ℝ) : (a + b + c) = N → P = (12 / N) * 12 := sorry

-- Define the variables
def number_of_data_points_Rongcheng : ℝ :=
  if (6 + y + z = 36 ∧ y = 12) then (12 / (6 + 12 + z)) * 12 else 0
  
-- The proof statement
theorem number_of_data_points (hy : y = 12) (hz : z = 18) :
  number_of_data_points_Rongcheng y z = 4 := sorry

end number_of_data_points_l614_614681


namespace icosahedron_hexagon_area_l614_614752

-- Define the side length
def side_length : ℝ := 2

-- Define the area calculation
def hexagon_area (s : ℝ) : ℝ := (3 * real.sqrt 3 / 2) * s^2

-- Theorem statement
theorem icosahedron_hexagon_area : hexagon_area side_length = 6 * real.sqrt 3 :=
by
  sorry

end icosahedron_hexagon_area_l614_614752


namespace ranking_of_scores_l614_614432

variable (scoreAlice scoreBarbara scoreCarol : ℕ)

-- Conditions from the problem
def reveals_score_to_others : Prop := scoreAlice
def barbara_statement : Prop := scoreBarbara > 75 ∧ scoreBarbara > scoreAlice ∧ scoreBarbara > scoreCarol
def carol_statement : Prop := scoreCarol < scoreAlice ∧ scoreCarol < scoreBarbara

-- The Proof Problem
theorem ranking_of_scores (h1 : reveals_score_to_others)
                          (h2 : barbara_statement)
                          (h3 : carol_statement) : 
                          scoreBarbara > scoreAlice ∧ scoreAlice > scoreCarol := 
by sorry

end ranking_of_scores_l614_614432


namespace passengers_got_off_l614_614331

/-- There are 50 passengers on a bus. At the first stop, 16 more passengers get on the bus. 
At the other stops, some passengers get off the bus and 5 more passengers get on the bus. 
There are 49 passengers on the bus in total at the last station.
Prove that 22 passengers got off the bus at the other stops. -/
theorem passengers_got_off (initial_passengers : ℕ) (first_stop_get_in : ℕ)
  (last_stop_get_in : ℕ) (final_passengers : ℕ) : 
  initial_passengers = 50 → first_stop_get_in = 16 → last_stop_get_in = 5 → final_passengers = 49 → 
  let total_after_first_stop := initial_passengers + first_stop_get_in 
  in let passengers_before_last_get_in := final_passengers - last_stop_get_in 
  in let passengers_got_off := total_after_first_stop - passengers_before_last_get_in 
  in passengers_got_off = 22 := 
begin
  intros h1 h2 h3 h4,
  simp only[],
  rw [h1, h2, h3, h4],
  exact nat.sub_eq_of_eq_add (eq.symm (nat.add_sub_assoc (nat.le_of_lt (nat.zero_lt_succ (nat.zero_lt_succ 44))) 44 5)),
  sorry
end

end passengers_got_off_l614_614331


namespace cos_of_angle_C_l614_614950

theorem cos_of_angle_C (A B C : ℝ) (k : ℝ) (hA : A = 2 * k) (hB : B = 3 * k) (hC : C = 4 * k) :
  let a := 2 * k,
      b := 3 * k,
      c := 4 * k in
  let cos_C := -1 / 4 in
  (c^2 = a^2 + b^2 - 2 * a * b * cos_C) := sorry

end cos_of_angle_C_l614_614950


namespace find_d_l614_614204

noncomputable def triangle_AB_equal (AB : ℝ) (BC : ℝ) (AC : ℝ) (P : ℝ×ℝ) (d : ℝ) : Prop :=
∃ A B C P₁ P₂,
  dist A B = 460 ∧ dist B C = 480 ∧ dist A C = 550 ∧ 
  (∃ D D' E E' F F', -- Points on the sides
  (segment D D') ∥ (segment A C) ∧ segment_length D D' = d ∧ 
  (segment E E') ∥ (segment A B) ∧ segment_length E E' = d ∧ 
  (segment F F') ∥ (segment B C) ∧ segment_length F F' = d ∧ 
  (segment P₁ P₂) ∥ (segment A C))

theorem find_d (h : triangle_AB_equal 460 480 550 (P₁) 260): 
  d = 260 :=
sorry

end find_d_l614_614204


namespace shaded_percent_of_grid_l614_614696

theorem shaded_percent_of_grid :
  let grid := finset.Icc (0, 0) (9, 9)
  let shaded := grid.filter (λ ⟨i, j⟩, (i + j) % 2 = 0)
  shaded.card / grid.card * 100 = 50 := by
  sorry

end shaded_percent_of_grid_l614_614696


namespace find_original_price_of_shirt_l614_614282

-- Define the conditions
def original_price_shirt (S : ℕ) : Prop :=
  ∃ S' J, J = 90 ∧ 5 * S' + 10 * (0.80 * J) = 960 ∧ S' = 0.80 * S

-- Lean statement to prove the original price of a shirt
theorem find_original_price_of_shirt : original_price_shirt 60 :=
by
  sorry

end find_original_price_of_shirt_l614_614282


namespace passing_time_l614_614349

def train1_speed_kmh : ℝ := 50 -- speed of the faster train in km/hr
def train2_speed_kmh : ℝ := 36 -- speed of the slower train in km/hr
def train_length_meters : ℝ := 70 -- length of each train in meters

/- Define km/hr to m/s conversion factor -/
def kmh_to_ms (kmh : ℝ) : ℝ := kmh * (5/18)

/- Define relative speed in m/s -/
def relative_speed_ms := kmh_to_ms (train1_speed_kmh - train2_speed_kmh)

/- Define total distance to be covered in meters -/
def total_distance := 2 * train_length_meters

/- Prove that the time to pass is 36 seconds -/
theorem passing_time : total_distance / relative_speed_ms = 36 :=
  sorry

end passing_time_l614_614349


namespace count_of_valid_numbers_in_range_l614_614545

def valid_digits : List ℕ := [0, 1, 3, 6, 7, 8]

def contains_invalid_digit (n : ℕ) : Prop :=
  n.digits 10 ∃ d, d ∉ valid_digits

def valid_numbers_in_range (n : ℕ) : ℕ :=
  (List.range' 1 n).count (λ m, m.digits 10 ∀ d, d ∈ valid_digits)

theorem count_of_valid_numbers_in_range : valid_numbers_in_range 8888 = 1295 :=
  sorry

end count_of_valid_numbers_in_range_l614_614545


namespace sum_first_15_odd_integers_from_5_l614_614002

theorem sum_first_15_odd_integers_from_5 :
  let a := 5
  let n := 15
  let d := 2
  let last_term := a + (n - 1) * d
  let S := n * a + (n * (n - 1) * d) / 2
  last_term = 37 ∧ S = 315 := by
  sorry

end sum_first_15_odd_integers_from_5_l614_614002


namespace original_words_count_l614_614680

theorem original_words_count :
  ∃ (W : ℕ), 
    let weekdays_in_two_years := 104 * 5 - 30 in
    let words_learned := weekdays_in_two_years * 10 in
    let words_forgotten := 104 * 2 in
    1.5 * W = W + words_learned - words_forgotten ∧
    W = 9384 := sorry

end original_words_count_l614_614680


namespace problem1_problem2_l614_614965

theorem problem1 
  (t : ℝ) (a : ℝ) (θ : ℝ) (x y : ℝ)
  (h1 : 0 ≤ a ∧ a < Real.pi)
  (h2 : x = 1 + t * Real.cos a)
  (h3 : y = Real.sqrt 3 + t * Real.sin a)
  (h4 : ∃ C1 : (ℝ × ℝ) → Prop, ∀(ρ θ : ℝ), C1 (ρ, θ) ↔ ρ = 4 * Real.cos θ) :
  C1 (x, y) ↔ (x^2 + y^2 = 4 * x) ∧ (a = Real.pi / 6) :=
sorry

theorem problem2
  (x y : ℝ) 
  (A B : ℝ × ℝ)
  (h1 : x = 1 + t * Real.cos (Real.pi / 6))
  (h2 : y = Real.sqrt 3 + t * Real.sin (Real.pi / 6))
  (h3 : ∃ Q : (ℝ × ℝ), Q = (2, 0))
  (h4 : ∃ C2 : (ℝ × ℝ) -> Prop, ∀(x y), C2 (x, y) ↔ x^2 + y^2 / 3 = 1)
  (h5 : ∀ B ∈ C2 (x, y), ∀ A ∈ C2 (x, y), B ≠ A) :
  ∃ S : ℝ, S = (1/2) * (Real.norm ⟨2, 1⟩ - Real.norm ⟨0, 0⟩) * 2 ∧ S = 6 * Real.sqrt 2 / 5 :=
sorry

end problem1_problem2_l614_614965


namespace number_of_ways_to_distribute_contracts_l614_614088

-- Defining the number of projects each company is contracted for
def projects_A : ℕ := 3
def projects_B : ℕ := 1
def projects_C : ℕ := 2
def projects_D : ℕ := 2
def total_projects : ℕ := 8

-- Statement of the problem
theorem number_of_ways_to_distribute_contracts :
  nat.choose total_projects projects_A *
  nat.choose (total_projects - projects_A) projects_B *
  nat.choose (total_projects - projects_A - projects_B) projects_C *
  nat.choose (total_projects - projects_A - projects_B - projects_C) projects_D = 1680 :=
sorry

end number_of_ways_to_distribute_contracts_l614_614088


namespace sum_of_seven_consecutive_integers_l614_614643

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_of_seven_consecutive_integers_l614_614643


namespace factorial_arithmetic_l614_614083

theorem factorial_arithmetic :
  6 * (6!) + 5 * (5!) + 2 * (5!) = 5160 :=
by
  sorry

end factorial_arithmetic_l614_614083


namespace geometric_triangle_sides_l614_614324

theorem geometric_triangle_sides (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b^2 = a * c) (h4 : a = 100 ∨ c = 100) :
  (a = 49 ∧ b = 70 ∧ c = 100) ∨
  (a = 64 ∧ b = 80 ∧ c = 100) ∨
  (a = 81 ∧ b = 90 ∧ c = 100) ∨
  (a = 100 ∧ b = 100 ∧ c = 100) ∨
  (a = 100 ∧ b = 110 ∧ c = 121) ∨
  (a = 100 ∧ b = 120 ∧ c = 144) ∨
  (a = 100 ∧ b = 130 ∧ c = 169) ∨
  (a = 100 ∧ b = 140 ∧ c = 196) ∨
  (a = 100 ∧ b = 150 ∧ c = 225) ∨
  (a = 100 ∧ b = 160 ∧ c = 256) := 
begin 
  sorry 
end

end geometric_triangle_sides_l614_614324


namespace fraction_of_speed_l614_614547

theorem fraction_of_speed (S : ℝ) (F : ℝ) (H1 : 0 < S) (H2 : 0 < F) :
  let T := 40 in
  (S * T = F * S * (T + 10)) → 
  F = 4 / 5 :=
by
  intros T h
  rw [T] at h
  sorry

end fraction_of_speed_l614_614547


namespace elf_distribution_finite_l614_614614

theorem elf_distribution_finite (infinite_rubies : ℕ → ℕ) (infinite_sapphires : ℕ → ℕ) :
  (∃ n : ℕ, ∀ i j : ℕ, i < n → j < n → (infinite_rubies i > infinite_rubies j → infinite_sapphires i < infinite_sapphires j) ∧
  (infinite_rubies i ≥ infinite_rubies j → infinite_sapphires i < infinite_sapphires j)) ↔
  ∃ k : ℕ, ∀ j : ℕ, j < k :=
sorry

end elf_distribution_finite_l614_614614


namespace first_grade_children_count_l614_614230

theorem first_grade_children_count (a : ℕ) (R L : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a = 25 * R + 10 ∧ a = 30 * L - 15 ∧ (R > 0 ∧ L > 0) → a = 285 :=
by
  sorry

end first_grade_children_count_l614_614230


namespace number_of_solutions_l614_614905

theorem number_of_solutions (n : ℕ) : 
  (1 ≤ n) → 
  ∃ (count : ℕ), count = n^2 - n + 1 ∧ 
  ∀ x, (1 ≤ x ∧ x ≤ n) → x^2 - ⌊x^2⌋ = (x - ⌊x⌋)^2 :=
sorry

end number_of_solutions_l614_614905


namespace problem_part_I_problem_part_II_l614_614873

-- Problem Part I
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_part_I (x : ℝ) :
    (f (x + 3/2) ≥ 0) ↔ (-2 ≤ x ∧ x ≤ 2) :=
  sorry

-- Problem Part II
theorem problem_part_II (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
    (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
    3*p + 2*q + r ≥ 9/4 :=
  sorry

end problem_part_I_problem_part_II_l614_614873


namespace solutions_to_equations_l614_614548

theorem solutions_to_equations :
  ∀ (p x y : ℕ), 
    prime p ∧ p > 0 ∧ x > 0 ∧ y > 0
    ∧ p * (x - 2) = x * (y - 1)
    ∧ x + y = 21
  → (x = 11 ∧ y = 10 ∧ p = 11) 
    ∨ (x = 14 ∧ y = 7 ∧ p = 7) :=
by sorry

end solutions_to_equations_l614_614548


namespace part1_part2_part3_l614_614144

variable (b c : ℝ)
noncomputable def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem part1 
  (h1 : ∀ α : ℝ, f b c (Real.sin α) ≥ 0)
  (h2 : ∀ β : ℝ, f b c (2 + Real.cos β) ≤ 0) : 
  f b c 1 = 0 :=
sorry

theorem part2 
  (h1 : ∀ α : ℝ, f b c (Real.sin α) ≥ 0)
  (h2 : ∀ β : ℝ, f b c (2 + Real.cos β) ≤ 0) 
  (h3 : f b c 1 = 0) : 
  c ≥ 3 :=
sorry

theorem part3 
  (h1 : ∀ α : ℝ, f b c (Real.sin α) ≥ 0)
  (h2 : ∀ β : ℝ, f b c (2 + Real.cos β) ≤ 0)
  (h3 : f b c 1 = 0)
  (h4 : ∀ α : ℝ, ⋂ i ∈ (Real.sin α) = 8)
  : 
  f b c = λ x, x^2 - 4 * x + 3 :=
sorry

end part1_part2_part3_l614_614144


namespace find_b_l614_614322

noncomputable def Q (x : ℝ) (a b c : ℝ) := 3 * x ^ 3 + a * x ^ 2 + b * x + c

theorem find_b (a b c : ℝ) (h₀ : c = 6) 
  (h₁ : ∃ (r₁ r₂ r₃ : ℝ), Q r₁ a b c = 0 ∧ Q r₂ a b c = 0 ∧ Q r₃ a b c = 0 ∧ (r₁ + r₂ + r₃) / 3 = -(c / 3) ∧ r₁ * r₂ * r₃ = -(c / 3))
  (h₂ : 3 + a + b + c = -(c / 3)): 
  b = -29 :=
sorry

end find_b_l614_614322


namespace cos_A_condition_is_isosceles_triangle_tan_sum_l614_614560

variable {A B C a b c : ℝ}

theorem cos_A_condition (h : (3 * b - c) * Real.cos A - a * Real.cos C = 0) :
  Real.cos A = 1 / 3 := sorry

theorem is_isosceles_triangle (ha : a = 2 * Real.sqrt 3)
  (hs : 1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 2) :
  c = 3 ∧ b = 3 := sorry

theorem tan_sum (h_sin : Real.sin B * Real.sin C = 2 / 3)
  (h_cos : Real.cos A = 1 / 3) :
  Real.tan A + Real.tan B + Real.tan C = 4 * Real.sqrt 2 := sorry

end cos_A_condition_is_isosceles_triangle_tan_sum_l614_614560


namespace coeff_x6_in_expansion_l614_614359

theorem coeff_x6_in_expansion :
  let f := (1 - 3 * x^2)
  polynomial.coeff (f ^ 6) 6 = -540 :=
by
  sorry

end coeff_x6_in_expansion_l614_614359


namespace incorrect_description_about_sampling_l614_614072

-- Define the given conditions
def condition1 := ∀ (x : Type) (n : ℕ), be_sampled_earlier_has_higher_probability x n
def condition2 := ∀ (x : Type), systematic_sampling x → equal_interval_sampling x ∧ everyone_has_equal_chance x
def condition3 := ∀ (x : Type), stratified_sampling x → type_sampling x ∧ (∀ stratum : Type, sample_with_equal_probability stratum x)
def condition4 := ∀ (x : Type), principle_of_sampling x = "stir evenly" ∧ everyone_has_equal_probability x

-- Define the statement that needs to be proven
theorem incorrect_description_about_sampling : 
  (∀ (x : Type), simple_sampling x ¬ everyone_has_equal_probability x) → ¬condition1 ∧ condition2 ∧ condition3 ∧ condition4 :=
by
  sorry

end incorrect_description_about_sampling_l614_614072


namespace trapezium_area_l614_614384

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 12) :
  (1 / 2 * (a + b) * h = 228) :=
by
  sorry

end trapezium_area_l614_614384


namespace Uki_earnings_l614_614354

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

end Uki_earnings_l614_614354


namespace largest_number_is_D_l614_614702

noncomputable def A : ℝ := 15467 + 3 / 5791
noncomputable def B : ℝ := 15467 - 3 / 5791
noncomputable def C : ℝ := 15467 * (3 / 5791)
noncomputable def D : ℝ := 15467 / (3 / 5791)
noncomputable def E : ℝ := 15467.5791

theorem largest_number_is_D :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_number_is_D_l614_614702


namespace inequality_solution_set_l614_614456

theorem inequality_solution_set {f : ℝ → ℝ} 
  (H1 : f 2 = 1)
  (H2 : ∀ x : ℝ, deriv f x < 1 / 3) :
  {x : ℝ | 0 < x ∧ x < 4} = {x : ℝ | f (Real.log 2 x) > (Real.log 2 x + 1) / 3} :=
by
  sorry

end inequality_solution_set_l614_614456


namespace find_x_l614_614917

theorem find_x (x : ℕ) (h : 2^10 = 32^x) (h32 : 32 = 2^5) : x = 2 :=
sorry

end find_x_l614_614917


namespace possible_values_of_x_l614_614304

-- Definitions based on conditions
def grid_3x3 := fin 3 → fin 3

def valid_placement (grid : grid_3x3 → ℕ) : Prop :=
  ∀ i j : fin 3, grid i j = 0 ∨ grid i j = 1

def game_ends_with (grid : grid_3x3 → ℕ) (x : ℕ) : Prop :=
  (∑ i j : fin 3, grid i j = x) ∧ 
  (grid (1, 1) = 1) ∧ 
  (∀ i j : fin 3, (i = 1 ∨ j = 1) → grid i j = 1) ∧ 
  (∀ i j j : fin 3, (i ≠ 1 ∧ j ≠ 1) → grid i j = 0)

-- The statement to prove the possible values of x
theorem possible_values_of_x (x : ℕ) : 
  (∃ grid : grid_3x3 → ℕ, valid_placement grid ∧ game_ends_with grid x) ↔ 
  x ∈ {3, 4, 5, 6, 9} :=
sorry

end possible_values_of_x_l614_614304


namespace value_of_c_plus_d_l614_614863

theorem value_of_c_plus_d (a b c d : ℝ) (h1 : a + b = 5) (h2 : b + c = 6) (h3 : a + d = 2) : c + d = 3 :=
by
  sorry

end value_of_c_plus_d_l614_614863


namespace count_three_digit_multiples_of_91_l614_614186

theorem count_three_digit_multiples_of_91 : 
  { n // 100 ≤ n ∧ n ≤ 999 ∧ n % 91 = 0 }.card = 9 :=
by
  sorry

end count_three_digit_multiples_of_91_l614_614186


namespace homothety_image_and_collinear_l614_614023

-- Definitions corresponding to given conditions
variable (α : Type) [EuclideanGeometry α]
variables (A B C : α) -- vertices of the triangle
variables (O G H A' B' C' D E F : α) -- important points

-- Hypotheses for the given conditions
-- circumcenter of triangle ABC
def is_circumcenter (O A B C : α) : Prop := sorry

-- centroid of triangle ABC
def is_centroid (G A B C : α) : Prop := sorry

-- orthocenter of triangle ABC
def is_orthocenter (H A B C : α) : Prop := sorry

-- midpoints of triangle sides
def is_midpoint (M P Q : α) : Prop := sorry

-- feet of altitudes from vertices
def is_feet_of_altitude (F P Q R : α) : Prop := sorry

-- Homothety transformation
def is_homothety (G A B C A' B' C' : α) (k : ℝ) : Prop := sorry

-- Collinearity of points
def are_collinear (P Q R : α) : Prop := sorry

-- Proving the necessary conditions
theorem homothety_image_and_collinear :
  is_circumcenter O A B C →
  is_centroid G A B C →
  is_orthocenter H A B C →
  is_midpoint A' B C →
  is_midpoint B' C A →
  is_midpoint C' A B →
  is_feet_of_altitude D A B C →
  is_feet_of_altitude E B C A →
  is_feet_of_altitude F C A B →
  is_homothety G A B C A' B' C' (1 / 2) ∧ are_collinear O G H := 
sorry

end homothety_image_and_collinear_l614_614023


namespace find_x_eq_2_l614_614925

theorem find_x_eq_2 : ∀ x : ℝ, 2^10 = 32^x → x = 2 := 
by 
  intros x h
  sorry

end find_x_eq_2_l614_614925


namespace profit_calculation_l614_614030

def purchase_price : ℝ := 100
def increase_rate : ℝ := 0.25
def discount_rate : ℝ := 0.10

def sell_price_increased : ℝ := purchase_price * (1 + increase_rate)
def sell_price_discounted : ℝ := sell_price_increased * (1 - discount_rate)

def profit_per_piece : ℝ := sell_price_discounted - purchase_price

theorem profit_calculation : profit_per_piece = 12.5 := by
  unfold profit_per_piece sell_price_discounted sell_price_increased purchase_price increase_rate discount_rate
  norm_num
  sorry

end profit_calculation_l614_614030


namespace ball_distance_fifth_hit_l614_614061

-- Define initial height and rebound ratio
def initialHeight : Real := 120
def reboundRatio : Real := 1 / 3

-- Total distance traveled when the ball hits the ground the fifth time
theorem ball_distance_fifth_hit :
  let descent1 := initialHeight
  let ascent1 := reboundRatio * descent1
  let descent2 := ascent1
  let ascent2 := reboundRatio * descent2
  let descent3 := ascent2
  let ascent3 := reboundRatio * descent3
  let descent4 := ascent3
  let ascent4 := reboundRatio * descent4
  let descent5 := ascent4
  descent1 + ascent1 + descent2 + ascent2 + descent3 + ascent3 + descent4 + ascent4 + descent5 = 248.962 :=
by
  sorry

end ball_distance_fifth_hit_l614_614061


namespace find_number_l614_614377

theorem find_number (x : ℤ) (h : 3 * x - 4 = 5) : x = 3 :=
sorry

end find_number_l614_614377


namespace maximum_value_of_expression_l614_614118

noncomputable def max_expression_value (theta : ℝ) : ℝ :=
  sin (theta / 2) ^ 2 * (1 + cos theta)

theorem maximum_value_of_expression : 
  ∀ θ : ℝ, 0 < θ ∧ θ < π → max_expression_value θ ≤ 1 / 2 :=
by
  intro θ h
  sorry

end maximum_value_of_expression_l614_614118


namespace find_circle_equation_tangent_to_parabola_l614_614117

noncomputable def conic_equation_tangent_to_parabola
  (circle : ℝ → ℝ → ℝ)
  (tangent_points : List (ℝ × ℝ))
  (parabola : ℝ → ℝ → Prop)
  (point_p : ℝ × ℝ)
  (point_q : ℝ × ℝ)
  (point_a : ℝ × ℝ) :
  Prop :=
  ∀ x y : ℝ,
  (parabola x y → ∃ λ : ℝ,
    circle x y = y^2 - 5*x - 9 - λ*(5*x - y + 3)^2) ∧
  ((x, y) ∈ tangent_points ∧ 
   ((x, y) = point_p ∨ (x, y) = point_q ∨ (x, y) = point_a))

def circle_equation : ℝ → ℝ → ℝ :=
  λ x y, 2*x^2 - 10*x*y - 31*y^2 + 175*x - 6*y + 297

def parabola_equation : ℝ → ℝ → Prop :=
  λ x y, y^2 = 5*x + 9

def points_p : (ℝ × ℝ) := (0, 3)
def points_q : (ℝ × ℝ) := (-1, -2)
def points_a : (ℝ × ℝ) := (-2, 1)

theorem find_circle_equation_tangent_to_parabola :
  conic_equation_tangent_to_parabola
    circle_equation
    [points_p, points_q]
    parabola_equation
    points_p
    points_q
    points_a :=
sorry

end find_circle_equation_tangent_to_parabola_l614_614117


namespace circle_area_ratio_not_integer_l614_614706

-- Define the problem in Lean 4
theorem circle_area_ratio_not_integer :
  ∀ (K L M : Type) (radius_K radius_L radius_M : ℝ),
    -- Conditions
    (radius_K = 2) ∧
    (radius_L = 1) ∧
    (radius_M = 2 - Real.sqrt 2) ∧
    (Circle L isTangentTo Circle K atCenterOf Circle K) ∧
    (Circle L isTangentTo line AB atCenterOf Circle K) ∧
    (Circle M isTangentTo Circle K) ∧
    (Circle M isTangentTo Circle L) ∧
    (Circle M isTangentTo line AB) →
    -- Prove the ratio of areas is not an integer
    ¬∃ (n : ℕ), 4 / ((2 - Real.sqrt 2)^2) = n := sorry

end circle_area_ratio_not_integer_l614_614706


namespace quadru_cyclic_quads_l614_614796

def is_cyclic (q : Quadrilateral) : Prop := ∃ O, ∀ v ∈ vertices q, dist O v = r

def regular_quadrilateral : Quadrilateral :=  -- Assume predefined
|--
def elongated_square : Quadrilateral := -- Assume predefined
|--
def rectangle_longer : Quadrilateral := -- Assume predefined
|--
def kite_not_square : Quadrilateral := -- Assume predefined
|--
def right_kite : Quadrilateral := -- Assume predefined
|--

def quadru_satisfy_cyclic_condition : Nat := 
  let quads := [regular_quadrilateral, elongated_square, rectangle_longer, kite_not_square, right_kite]
  quads.filter is_cyclic |>.length

theorem quadru_cyclic_quads : quadru_satisfy_cyclic_condition = 4 := 
by 
  sorry

end quadru_cyclic_quads_l614_614796


namespace parallel_line_slope_l614_614370

theorem parallel_line_slope {x y : ℝ} (h : 3 * x + 6 * y = -24) : 
  ∀ m b : ℝ, (y = m * x + b) → m = -1 / 2 :=
sorry

end parallel_line_slope_l614_614370


namespace only_triples_satisfy_conditions_l614_614809

theorem only_triples_satisfy_conditions (p q : ℕ) (n : ℕ) :
    (prime p ∧ odd p ∧ prime q ∧ odd q ∧ 0 < n) →
    (q^(n+2) % p^n = 3^(n+2) % p^n ∧ p^(n+2) % q^n = 3^(n+2) % q^n) →
    (p = 3 ∧ q = 3 ∧ 0 < n) :=
by
    intro h1 h2
    -- Sorry, proof steps omitted
    sorry

end only_triples_satisfy_conditions_l614_614809


namespace time_to_cross_bridge_l614_614398

-- Definitions based on the conditions
def length_of_train : ℝ := 300
def length_of_bridge : ℝ := 300
def speed_of_train : ℝ := 47.99999999999999

-- The proof problem statement
theorem time_to_cross_bridge :
  let total_distance := length_of_train + length_of_bridge in
  let time := total_distance / speed_of_train in
  time = 12.5 := by
    -- Definitions
    let total_distance := length_of_train + length_of_bridge
    let time := total_distance / speed_of_train
    -- Expected answer
    show time = 12.5 from sorry

end time_to_cross_bridge_l614_614398


namespace correct_initial_value_l614_614174

noncomputable def algorithm (x : ℤ) : ℤ :=
  let mut S := 0
  let mut a := x
  for I in [1, 3, 5, 7, 9] do
    S := S + a * I
    a := a * (-1)
  S

theorem correct_initial_value : 
  algorithm (-1) = -1 + 3 - 5 + 7 - 9 := 
by
  sorry

end correct_initial_value_l614_614174


namespace number_of_men_in_first_group_l614_614729

theorem number_of_men_in_first_group
  (M : ℕ)
  (h1 : M * 0.4 * 7 = 56)
  (h2 : 35 * 0.4 * 3 = 42) :
  M = 20 :=
by
  sorry

end number_of_men_in_first_group_l614_614729


namespace percent_diploma_thirty_l614_614216

-- Defining the conditions using Lean definitions

def percent_without_diploma_with_job := 0.10 -- 10%
def percent_with_job := 0.20 -- 20%
def percent_without_job_with_diploma :=
  (1 - percent_with_job) * 0.25 -- 25% of people without job is 25% of 80% which is 20%

def percent_with_diploma := percent_with_job - percent_without_diploma_with_job + percent_without_job_with_diploma

-- Theorem to prove that 30% of the people have a university diploma
theorem percent_diploma_thirty
  (H1 : percent_without_diploma_with_job = 0.10) -- condition 1
  (H2 : percent_with_job = 0.20) -- condition 3
  (H3 : percent_without_job_with_diploma = 0.20) -- evaluated from condition 2
  : percent_with_diploma = 0.30 := by
  -- prove that the percent with diploma is 30%
  sorry

end percent_diploma_thirty_l614_614216


namespace price_reduction_correct_eqn_l614_614402

theorem price_reduction_correct_eqn (x : ℝ) :
  120 * (1 - x)^2 = 85 :=
sorry

end price_reduction_correct_eqn_l614_614402


namespace arrange_numbers_in_grid_l614_614592

theorem arrange_numbers_in_grid :
  ∃ (f : Fin 10 → Fin 10 → ℕ),
  ∀ (partition : List (Fin 10 × Fin 10)) (h : partition.length = 50)
    (h₂ : ∀ (p : Fin 10 × Fin 10), p ∈ partition → ∃ i j, (p = (i, j) ∨ p = (j, i))),
  ((∃ (even_doms : Fin 7 → (Fin 10 × Fin 10) × (Fin 10 × Fin 10)),
    ∀ (k : Fin 7), (even_doms k).fst.1 = (even_doms k).fst.2 ∧
                   (even_doms k).snd.1 + (even_doms k).snd.2 % 2 = 0 ∧
                   ((even_doms k).fst.1 * 10 + (even_doms k).fst.2 % 2 = 1 ∨
                    (even_doms k).snd.1 * 10 + (even_doms k).snd.2 % 2 = 1)) ∧
   (∀ (k : Fin 43), (∀ (p : (Fin 10 × Fin 10) × (Fin 10 × Fin 10)),
     p ∉ (even_doms.toList k).toList → p.fst.1 * 10 + p.fst.2 % 2 = 1 ∨
     p.snd.1 * 10 + p.snd.2 % 2 = 1))) := sorry

end arrange_numbers_in_grid_l614_614592


namespace total_pairs_of_shoes_tried_l614_614893

theorem total_pairs_of_shoes_tried (first_store_pairs second_store_additional third_store_pairs fourth_store_factor : ℕ) 
  (h_first : first_store_pairs = 7)
  (h_second : second_store_additional = 2)
  (h_third : third_store_pairs = 0)
  (h_fourth : fourth_store_factor = 2) :
  first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs + 
    (fourth_store_factor * (first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs)) = 48 := 
  by 
    sorry

end total_pairs_of_shoes_tried_l614_614893


namespace sin_value_l614_614137

theorem sin_value (α : ℝ) (h : Real.cos (α + π / 6) = - (Real.sqrt 2) / 10) : 
  Real.sin (2 * α - π / 6) = 24 / 25 :=
by
  sorry

end sin_value_l614_614137


namespace problem_x_value_l614_614192

theorem problem_x_value (x : ℝ) (h : 0.25 * x = 0.15 * 1500 - 15) : x = 840 :=
by
  sorry

end problem_x_value_l614_614192


namespace find_x_l614_614920

theorem find_x (x : ℕ) (h : 2^10 = 32^x) (h32 : 32 = 2^5) : x = 2 :=
sorry

end find_x_l614_614920


namespace circle_diagonal_probability_l614_614404

theorem circle_diagonal_probability (rect_width rect_height radius : ℝ) 
  (hw : rect_width = 15) 
  (hh : rect_height = 36)
  (hr : radius = 1) :
  let valid_width := rect_width - 2 * radius,
      valid_height := rect_height - 2 * radius,
      valid_area := valid_width * valid_height,
      safe_area := 375 in
  valid_width * valid_height = 442 →
  (safe_area / valid_area = 375 / 442) :=
by
  intros valid_width valid_height valid_area safe_area h_valid_area,
  exact (safe_area / valid_area).symm

end circle_diagonal_probability_l614_614404


namespace smallest_difference_is_176_l614_614165

theorem smallest_difference_is_176 (a b : ℕ) 
  (h_digits_a : (2 ∈ digits a) ∧ (4 ∈ digits a) ∧ (5 ∈ digits a) ∧ a < 1000 ∧ a > 99)
  (h_digits_b : (6 ∈ digits b) ∧ (9 ∈ digits b) ∧ b < 100 ∧ b > 9)
  (h_odd_b : b % 2 = 1)
  (h_all_digits : ∀ d ∈ {2, 4, 5, 6, 9}, d ∈ digits a ∨ d ∈ digits b ∧ ¬(d ∈ digits a ∧ d ∈ digits b))
  : a - b = 176 := 
sorry

end smallest_difference_is_176_l614_614165


namespace correct_operation_A_l614_614013

-- Definitions for the problem
def division_rule (a : ℝ) (m n : ℕ) : Prop := a^m / a^n = a^(m - n)
def multiplication_rule (a : ℝ) (m n : ℕ) : Prop := a^m * a^n = a^(m + n)
def power_rule (a : ℝ) (m n : ℕ) : Prop := (a^m)^n = a^(m * n)
def addition_like_terms_rule (a : ℝ) (m : ℕ) : Prop := a^m + a^m = 2 * a^m

-- The theorem to prove
theorem correct_operation_A (a : ℝ) : division_rule a 4 2 :=
by {
  sorry
}

end correct_operation_A_l614_614013


namespace number_of_tangents_l614_614286

def r1 := 4
def r2 := 6
def d := 8

theorem number_of_tangents : ∃ k_values, k_values = {1, 2, 4} ∧ k_values.to_finset.card = 3 := 
by
  sorry

end number_of_tangents_l614_614286


namespace part1_part2_l614_614961

-- Lean 4 statement for proving A == 2B
theorem part1 (a b c : ℝ) (A B C : ℝ) (h₁ : 0 < A) (h₂ : A < π / 2) 
    (h₃ : 0 < B) (h₄ : B < π / 2) (h₅ : 0 < C) (h₆ : C < π / 2) (h₇ : A + B + C = π)
    (h₈ : c = 2 * b * Real.cos A + b) : A = 2 * B :=
by sorry

-- Lean 4 statement for finding range of area of ∆ABD
theorem part2 (B : ℝ) (c : ℝ) (h₁ : 0 < B) (h₂ : B < π / 2) 
    (h₃ : A = 2 * B) (h₄ : c = 2) : 
    (Real.tan (π / 6) < (1 / 2) * c * (1 / Real.cos B) * Real.sin B) ∧ 
    ((1 / 2) * c * (1 / Real.cos B) * Real.sin B < 1) :=
by sorry

end part1_part2_l614_614961


namespace count_incorrect_statements_l614_614243

/-!
  Given four statements about regression analysis:
  1. The narrower the horizontal band area where the residual points are located in the residual plot, the higher the accuracy of the regression equation's prediction.
  2. The closer the scatter plot is to a straight line, the stronger the linear correlation, and the larger the correlation coefficient.
  3. In the regression line equation \( \hat{y} = 2x + 3 \), when the variable \( x \) increases by 1 unit, the variable \( \hat{y} \) increases by 2 units.
  4. The model with a smaller sum of squared residuals has a better fit.

  Prove that there is exactly 1 incorrect statement among the four given statements.
-/

theorem count_incorrect_statements :
  ∃ (S1 S3 S4 : Prop) (S2 : Prop),
  (S1 ↔ True) ∧
  (S2 ↔ False) ∧
  (S3 ↔ True) ∧
  (S4 ↔ True) ∧
  (nat.sum (λ b : Bool, if b then 1 else 0) [S1, S2, S3, S4] = 1) :=
by
  exists (True), (False), (True), (True)
  simp
  sorry

end count_incorrect_statements_l614_614243


namespace find_triangle_angles_find_triangle_sides_l614_614245

noncomputable def triangle_angles : Prop :=
  ∃ (A B C : ℝ), 
    tan C = (sin A + sin B) / (cos A + cos B) ∧ 
    sin (B - A) = cos C ∧ 
    A + B + C = π ∧ 
    (C = π / 3) ∧ 
    (A = π / 4)

noncomputable def triangle_sides : Prop :=
  ∃ (a b c A B C : ℝ), 
    tan C = (sin A + sin B) / (cos A + cos B) ∧ 
    sin (B - A) = cos C ∧ 
    A + B + C = π ∧ 
    C = π / 3 ∧ 
    A = π / 4 ∧
    (2 * (1 / 2 * a * c * sin B) = 3 + sqrt 3) ∧ 
    (B = π - A - C) ∧ 
    (a = 2 * sqrt 2) ∧ 
    (c = 2 * sqrt 3)

theorem find_triangle_angles : triangle_angles :=
  sorry

theorem find_triangle_sides : triangle_sides :=
  sorry

end find_triangle_angles_find_triangle_sides_l614_614245


namespace ordered_pair_exists_l614_614484

theorem ordered_pair_exists (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) :=
by
  sorry

end ordered_pair_exists_l614_614484


namespace first_grade_children_count_l614_614229

theorem first_grade_children_count (a : ℕ) (R L : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a = 25 * R + 10 ∧ a = 30 * L - 15 ∧ (R > 0 ∧ L > 0) → a = 285 :=
by
  sorry

end first_grade_children_count_l614_614229


namespace not_solution_set_D_l614_614074

-- Definition of the polynomial equation
def polynomial (x : ℝ) : Prop := (x + 1) * (x - 2) = 0

-- Definition of the solution set
def solution_set : Set ℝ := { x | polynomial x }

-- Definitions of the given sets
def A : Set ℝ := {-1, 2}
def B : Set ℝ := {2, -1}
def C : Set ℝ := { x | polynomial x }
def D : Set (ℝ × ℝ) := {(-1, 2)}

-- The proof problem statement
theorem not_solution_set_D : D ≠ solution_set :=
by
  sorry

end not_solution_set_D_l614_614074


namespace book_distribution_l614_614801

theorem book_distribution (x : ℕ) (books : ℕ)
    (h1 : books = 3 * x + 6)
    (h2 : 3 * x + 6 ≥ 5 * (x - 1))
    (h3 : 3 * x + 6 < 5 * (x - 1) + 3) :
    books = 21 ∧ x = 5 := by
  have h4 : 11 ≥ 2 * x := by linarith
  have h5 : 8 < 2 * x := by linarith
  have h6 : 4 < x := by linarith
  have h7 : x ≤ 5 := by linarith
  have h8 : x = 5 := by linarith
  have h9 : books = 3 * 5 + 6 := by rw [h8]; simp
  have h10 : books = 21 := by simp [h9]
  exact ⟨h10, h8⟩

end book_distribution_l614_614801


namespace magnitude_calculation_l614_614543

def vec2 := ℝ × ℝ

def a : vec2 := (0, 1)
def b : vec2 := (2, -1)

def vec_add (u v : vec2) : vec2 := (u.1 + v.1, u.2 + v.2)
def vec_scale (c : ℝ) (v : vec2) : vec2 := (c * v.1, c * v.2)
def vec_mag (v : vec2) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_calculation : vec_mag (vec_add (vec_scale 2 a) b) = Real.sqrt 5 := 
by { sorry }

end magnitude_calculation_l614_614543


namespace bricks_needed_for_courtyard_l614_614407

noncomputable def total_bricks_required (courtyard_length courtyard_width : ℝ)
  (brick_length_cm brick_width_cm : ℝ) : ℝ :=
  let courtyard_area := courtyard_length * courtyard_width
  let brick_length := brick_length_cm / 100
  let brick_width := brick_width_cm / 100
  let brick_area := brick_length * brick_width
  courtyard_area / brick_area

theorem bricks_needed_for_courtyard :
  total_bricks_required 35 24 15 8 = 70000 := by
  sorry

end bricks_needed_for_courtyard_l614_614407


namespace total_pairs_of_shoes_tried_l614_614894

theorem total_pairs_of_shoes_tried (first_store_pairs second_store_additional third_store_pairs fourth_store_factor : ℕ) 
  (h_first : first_store_pairs = 7)
  (h_second : second_store_additional = 2)
  (h_third : third_store_pairs = 0)
  (h_fourth : fourth_store_factor = 2) :
  first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs + 
    (fourth_store_factor * (first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs)) = 48 := 
  by 
    sorry

end total_pairs_of_shoes_tried_l614_614894


namespace find_B_l614_614572

structure AcuteTriangle where
  A B C : ℝ
  a b c : ℝ
  cos_A : ℝ
  angle_A_is_acute : A < π / 2
  cosA_def : cos_A = sqrt 3 / 3
  a_def : a = 2 * sqrt 2
  b_def : b = 3
  -- We don't need to explicitly mention c as it doesn't affect our proof goal.

noncomputable def angle_B_measure (T : AcuteTriangle) : ℝ :=
  if h : T.A < π / 2 then
    let sin_A := sqrt (1 - (T.cos_A)^2) in
    let sin_B := T.b * sin_A / T.a in
    if sin_B = sqrt (3) / 2 then 
      π / 3
    else 
      0 -- arbitrary value for non-acute angles, should never be reached
  else 
    0 -- angle_A_is_acute should ensure this case is not needed

theorem find_B (T : AcuteTriangle) : T.A < π / 2 → T.A > 0 → angle_B_measure T = π / 3 :=
by
  intros hApos hAacute
  sorry -- proof to be completed

end find_B_l614_614572


namespace grasshopper_jumps_more_l614_614659

theorem grasshopper_jumps_more (frog_jump : ℕ) (grasshopper_jump : ℕ) (h_frog_jump : frog_jump = 15) (h_grasshopper_jump : grasshopper_jump = frog_jump + 4) : grasshopper_jump = 19 := 
by
  rw [h_frog_jump] at h_grasshopper_jump
  exact h_grasshopper_jump

sorry

end grasshopper_jumps_more_l614_614659


namespace hyperbola_eccentricity_is_sqrt_10_l614_614948

noncomputable def arithmetic_sequence (a b m : ℝ) := ∃ d : ℝ,
  (-1 = -1 + d) ∧ (a = -1 + d) ∧ (b = -1 + 2 * d) ∧ (m = -1 + 3 * d) ∧ (7 = -1 + 4 * d)

noncomputable def hyperbola_eccentricity (a b : ℝ) := sqrt (1 + b^2 / a^2)

theorem hyperbola_eccentricity_is_sqrt_10 (a b m : ℝ) (h_seq : arithmetic_sequence a b m) :
  hyperbola_eccentricity a b = sqrt 10 := sorry

end hyperbola_eccentricity_is_sqrt_10_l614_614948


namespace perpendicular_line_equation_l614_614473

theorem perpendicular_line_equation :
  (∀ (x y : ℝ), 2 * x + 3 * y + 1 = 0 → x - 3 * y + 4 = 0 →
  ∃ (l : ℝ) (m : ℝ), m = 4 / 3 ∧ y = m * x + l → y = 4 / 3 * x + 1 / 9) 
  ∧ (∀ (x y : ℝ), 3 * x + 4 * y - 7 = 0 → -3 / 4 * 4 / 3 = -1) :=
by 
  sorry

end perpendicular_line_equation_l614_614473


namespace right_triangle_area_l614_614447

/-- The area of a right-angled triangle given the lengths of the segments
created by the touchpoint of the inscribed circle on the hypotenuse. -/
theorem right_triangle_area (m n : ℝ) : 
  ∃ (area : ℝ), area = m * n :=
by
  use m * n
  sorry

end right_triangle_area_l614_614447


namespace find_f_neg4_l614_614168

def f : ℝ → ℝ :=
λ x, if x < 3 then f (x + 2) else (1/2) ^ x

theorem find_f_neg4 : f (-4) = 1/16 :=
by
  sorry

end find_f_neg4_l614_614168


namespace B_in_fourth_quadrant_l614_614219

theorem B_in_fourth_quadrant (a b : ℝ) (h_a : a > 0) (h_b : -b > 0) : (a > 0 ∧ b < 0) := 
  begin
    have h_b_neg : b < 0 := by linarith,
    exact ⟨h_a, h_b_neg⟩,
  end

end B_in_fourth_quadrant_l614_614219


namespace range_of_m_l614_614791

theorem range_of_m (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0)
  (h_inequality : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f (2 * m * x - Real.log x - 3) ≥ 2 * f 3 - f (-2 * m * x + Real.log x + 3)) :
  ∃ m, m ∈ Set.Icc (1 / (2 * Real.exp 1)) (1 + Real.log 3 / 6) :=
sorry

end range_of_m_l614_614791


namespace barbara_max_win_l614_614343

theorem barbara_max_win : 
  (∃ a b ∈ Finset.range 1025, 
     (∀ s t, (s ⊆ Finset.range 1025 ∧ s.card = 512 ∧ (∀ a ∈ s, a % 2 = 0) → 
     (t ⊆ Finset.range 1025 \ s ∧ t.card = 256 ∧ even_card t)) ∧
     ∀ s t, (s ⊆ Finset.range 1025 ∧ s.card = 128 ∧ (∀ a ∈ s, a % 4 = 0) →
     (t ⊆ Finset.range 1025 \ s ∧ t.card = 64 ∧ even_card t)) ∧
     ∀ s t, (s ⊆ Finset.range 1025 ∧ s.card = 32 ∧ (∀ a ∈ s, a % 8 = 0) →
     (t ⊆ Finset.range 1025 \ s ∧ t.card = 16 ∧ even_card t)) ∧
     ∀ s t, (s ⊆ Finset.range 1025 ∧ s.card = 8 ∧ (∀ a ∈ s, a % 16 = 0) →
     (t ⊆ Finset.range 1025 \ s ∧ t.card = 4 ∧ even_card t)) ∧
     ∀ s t, (s ⊆ Finset.range 1025 ∧ s.card = 2 ∧ (∀ a ∈ s, a % 32 = 0) →
     (t ⊆ Finset.range 1025 \ s ∧ t.card = 1 ∧ even_card t)) → 
     |a - b| = 32) :=
sorry

end barbara_max_win_l614_614343


namespace simplify_expression_l614_614636

theorem simplify_expression (tan_60 cot_60 : ℝ) (h1 : tan_60 = Real.sqrt 3) (h2 : cot_60 = 1 / Real.sqrt 3) :
  (tan_60^3 + cot_60^3) / (tan_60 + cot_60) = 31 / 3 :=
by
  -- proof will go here
  sorry

end simplify_expression_l614_614636


namespace sum_S_2017_l614_614582

def a : ℕ → ℝ
| 0       := 1
| (n + 1) := a n + Real.sin ((n + 1) * Real.pi / 2)

def S (n : ℕ) : ℝ := (Finset.range n).sum a

theorem sum_S_2017 : S 2017 = 1009 := by
  sorry

end sum_S_2017_l614_614582


namespace three_digit_numbers_not_multiple_of_3_5_7_l614_614909

theorem three_digit_numbers_not_multiple_of_3_5_7 : 
  let total_three_digit_numbers := 900
  let multiples_of_3 := (999 - 100) / 3 + 1
  let multiples_of_5 := (995 - 100) / 5 + 1
  let multiples_of_7 := (994 - 105) / 7 + 1
  let multiples_of_15 := (990 - 105) / 15 + 1
  let multiples_of_21 := (987 - 105) / 21 + 1
  let multiples_of_35 := (980 - 105) / 35 + 1
  let multiples_of_105 := (945 - 105) / 105 + 1
  let total_multiples := multiples_of_3 + multiples_of_5 + multiples_of_7 - multiples_of_15 - multiples_of_21 - multiples_of_35 + multiples_of_105
  let non_multiples_total := total_three_digit_numbers - total_multiples
  non_multiples_total = 412 :=
by
  sorry

end three_digit_numbers_not_multiple_of_3_5_7_l614_614909


namespace range_of_a_l614_614513

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0) → a ≤ -1 :=
by
  sorry

end range_of_a_l614_614513


namespace magnitude_of_b_l614_614890

noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt7 : ℝ := Real.sqrt 7
noncomputable def vec_a : ℝ × ℝ := (sqrt2, -sqrt7)
noncomputable def dot_ab : ℝ := -9
noncomputable def angle_a_b : ℝ := 120 -- in degrees

theorem magnitude_of_b : ∃ b : ℝ × ℝ, ∥b∥ = 6 ∧ vec_a.1 * b.1 + vec_a.2 * b.2 = dot_ab ∧ Real.angle vec_a b = Real.pi / 3 :=
by
  sorry

end magnitude_of_b_l614_614890


namespace g_of_x_plus_3_l614_614193

-- Definition of the function g
def g (x : ℝ) : ℝ := 3 * x + 1

-- The statement we aim to prove: g(x + 3) = 3 * x + 10
theorem g_of_x_plus_3 (x : ℝ) : g(x + 3) = 3 * x + 10 :=
by
  -- Place the proof here
  sorry

end g_of_x_plus_3_l614_614193


namespace day_of_week_299th_2005_l614_614198

theorem day_of_week_299th_2005 (day_15th: ℕ) (h : day_15th % 7 = 3) : (299 - 15) % 7 = 5 :=
by
  -- Definitions for days of the week assuming 0 for Sunday, hence 3 for Wednesday
  let day_15th := 3 -- Wednesday
  have mod_284 := (299 - 15) % 7 -- 284 % 7
  show (299 - 15) % 7 = 5 from congr_fun mod_284 sorry

end day_of_week_299th_2005_l614_614198


namespace find_num_boys_l614_614725

-- Definitions for conditions
def num_children : ℕ := 13
def num_girls (num_boys : ℕ) : ℕ := num_children - num_boys

-- We will assume we have a predicate representing the truthfulness of statements.
-- boys tell the truth to boys and lie to girls
-- girls tell the truth to girls and lie to boys

theorem find_num_boys (boys_truth_to_boys : Prop) 
                      (boys_lie_to_girls : Prop) 
                      (girls_truth_to_girls : Prop) 
                      (girls_lie_to_boys : Prop)
                      (alternating_statements : Prop) : 
  ∃ (num_boys : ℕ), num_boys = 7 := 
  sorry

end find_num_boys_l614_614725


namespace unique_polynomial_solution_l614_614808

def polynomial_homogeneous_of_degree_n (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (t x y : ℝ), P (t * x) (t * y) = t^n * P x y

def polynomial_symmetric_condition (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y z : ℝ), P (y + z) x + P (z + x) y + P (x + y) z = 0

def polynomial_value_at_point (P : ℝ → ℝ → ℝ) : Prop :=
  P 1 0 = 1

theorem unique_polynomial_solution (P : ℝ → ℝ → ℝ) (n : ℕ) :
  polynomial_homogeneous_of_degree_n P n →
  polynomial_symmetric_condition P →
  polynomial_value_at_point P →
  ∀ x y : ℝ, P x y = (x + y)^n * (x - 2 * y) := 
by
  intros h_deg h_symm h_value x y
  sorry

end unique_polynomial_solution_l614_614808


namespace positive_difference_of_complementary_angles_l614_614662

noncomputable def x : ℝ := 18
def angle1 : ℝ := 4 * x
def angle2 : ℝ := x
def sum_complementary : Prop := angle1 + angle2 = 90

-- Theorem statement: Prove that the positive difference between the two angles is 54 degrees
theorem positive_difference_of_complementary_angles : 
  sum_complementary → (angle1 - angle2 = 54) :=
by
  assume h : sum_complementary
  sorry

end positive_difference_of_complementary_angles_l614_614662


namespace incorrect_statements_A_D_l614_614882

variables {m x y : ℝ}

def line1 (m x y : ℝ) : Prop := (m + 2) * x + y + 1 = 0
def line2 (m x y : ℝ) : Prop := 3 * x + m * y + 4 * m - 3 = 0

theorem incorrect_statements_A_D (m : ℝ) : 
  (¬ ∃ (x y : ℝ), line1 (-3) x y ∧ line2 (-3) x y) ∧ 
  (¬ ∃ (d : ℝ), ∀ (x0 y0 : ℝ), x0 = 0 ∧ y0 = 0 → d = |(m + 2) * x0 + y0 + 1| / sqrt ((m + 2)^2 + 1^2) ∧ d = sqrt 17) :=
sorry

end incorrect_statements_A_D_l614_614882


namespace four_digit_integers_ending_in_5_divisible_by_15_count_l614_614546

noncomputable def count_four_digit_integers_ending_in_5_divisible_by_15 : Nat := do
  let range_start := 1005
  let range_end := 9995
  let d := 15
  let m := 5
  have h1 : ∀ n ∈ Set.range(range_start, range_end), n % 10 = 5 := sorry
  have h2 : ∀ n ∈ Set.range(range_start, range_end), n % d = 0 := sorry
  pure ((range_end - range_start) / (2 * d / m))

theorem four_digit_integers_ending_in_5_divisible_by_15_count : 
  count_four_digit_integers_ending_in_5_divisible_by_15 = 300 := 
  by 
    sorry

end four_digit_integers_ending_in_5_divisible_by_15_count_l614_614546


namespace B_work_days_l614_614399

/-- 
  A and B undertake to do a piece of work for $500.
  A alone can do it in 5 days while B alone can do it in a certain number of days.
  With the help of C, they finish it in 2 days. C's share is $200.
  Prove B alone can do the work in 10 days.
-/
theorem B_work_days (x : ℕ) (h1 : (1/5 : ℝ) + (1/x : ℝ) = 3/10) : x = 10 := 
  sorry

end B_work_days_l614_614399


namespace population_doubles_l614_614305

theorem population_doubles (initial_population: ℕ) (initial_year: ℕ) (doubling_period: ℕ) (target_population : ℕ) (target_year : ℕ) : 
  initial_population = 500 → 
  initial_year = 2023 → 
  doubling_period = 20 → 
  target_population = 8000 → 
  target_year = 2103 :=
by 
  sorry

end population_doubles_l614_614305


namespace more_candidates_selected_in_B_l614_614566

def total_candidates := 8100
def percentage_selected_A := 0.06
def percentage_selected_B := 0.07
def selected_A := percentage_selected_A * total_candidates
def selected_B := percentage_selected_B * total_candidates
def difference_in_selected := selected_B - selected_A

theorem more_candidates_selected_in_B :
    difference_in_selected = 81 :=
by
    sorry

end more_candidates_selected_in_B_l614_614566


namespace constant_term_in_binomial_expansion_l614_614365

theorem constant_term_in_binomial_expansion : 
  (∃ c : ℚ, c = (2 : ℚ) * (2 : ℚ).pow 12 * (4 : ℚ).pow (-6) ∧ c = 15 / 64) :=
by
  sorry

end constant_term_in_binomial_expansion_l614_614365


namespace total_pairs_of_shoes_tried_l614_614895

theorem total_pairs_of_shoes_tried (first_store_pairs second_store_additional third_store_pairs fourth_store_factor : ℕ) 
  (h_first : first_store_pairs = 7)
  (h_second : second_store_additional = 2)
  (h_third : third_store_pairs = 0)
  (h_fourth : fourth_store_factor = 2) :
  first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs + 
    (fourth_store_factor * (first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs)) = 48 := 
  by 
    sorry

end total_pairs_of_shoes_tried_l614_614895


namespace exists_unique_root_limit_of_sequence_x_n_l614_614820

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in Finset.range(n) + 1, 1 / (k^2 * x - 1)

theorem exists_unique_root (n : ℕ) (h : 0 < n) : ∃! (x : ℝ), 1 < x ∧ f_n n x = 1 / 2 :=
by
  sorry

theorem limit_of_sequence_x_n : tendsto (fun n => some (exists_unique_root n (by linarith)).1) atTop (𝓝 4) :=
by
  sorry

end exists_unique_root_limit_of_sequence_x_n_l614_614820


namespace three_digit_numbers_not_multiple_of_3_5_7_l614_614910

theorem three_digit_numbers_not_multiple_of_3_5_7 : 
  let total_three_digit_numbers := 900
  let multiples_of_3 := (999 - 100) / 3 + 1
  let multiples_of_5 := (995 - 100) / 5 + 1
  let multiples_of_7 := (994 - 105) / 7 + 1
  let multiples_of_15 := (990 - 105) / 15 + 1
  let multiples_of_21 := (987 - 105) / 21 + 1
  let multiples_of_35 := (980 - 105) / 35 + 1
  let multiples_of_105 := (945 - 105) / 105 + 1
  let total_multiples := multiples_of_3 + multiples_of_5 + multiples_of_7 - multiples_of_15 - multiples_of_21 - multiples_of_35 + multiples_of_105
  let non_multiples_total := total_three_digit_numbers - total_multiples
  non_multiples_total = 412 :=
by
  sorry

end three_digit_numbers_not_multiple_of_3_5_7_l614_614910


namespace simplify_expression_l614_614781

variables (x y z : ℝ)

theorem simplify_expression (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) : 
  ((x - 2) / (4 - z)) * ((y - 3) / (2 - x)) * ((z - 4) / (3 - y)) = -1 :=
by sorry

end simplify_expression_l614_614781


namespace num_odd_five_digit_numbers_l614_614825

theorem num_odd_five_digit_numbers : 
  (∃ (digits : finset ℕ), 
    digits = {1, 2, 3, 4, 5} ∧ 
    ∀ n ∈ digits, (n >= 1 ∧ n <= 5)) →
  (∃ n : ℕ, n = 72) :=
by
  sorry

end num_odd_five_digit_numbers_l614_614825


namespace part_a_part_b_l614_614982

section field_polynomial

variable {K : Type*} [field K]
variable (p n : ℕ) (q : ℕ := p ^ n) [fact (nat.prime p)] (n_ge_two : 2 ≤ n)
variable (a : K) (X : polynomial K)

-- Define the polynomials
def g (X : polynomial K) : polynomial K := X^q - X
def f (X : polynomial K) : polynomial K := (g p n X)^q - g p n X
def f_1 (X : polynomial K) : polynomial K := X^q - X + 1
def f_a (a : K) (X : polynomial K) : polynomial K := X^q - X + a

-- Part (a): Prove that f is divisible by f_1
theorem part_a : f p n X ∣ f_1 p n X := sorry

-- Part (b): Prove that f_a has at least p^(n-1) distinct irreducible factors
theorem part_b : ∃ (factors : multiset (polynomial K)), (∀ factor ∈ factors, irreducible factor) 
                ∧ f_a a X = factors.prod 
                ∧ multiset.card factors ≥ p^(n-1) := sorry

end field_polynomial

end part_a_part_b_l614_614982


namespace prove_conclusion_correct_l614_614071

theorem prove_conclusion_correct :
  (∀ k : ℝ, (∃ x, k * x^2 + 4*x + 4 = 0 → k ≠ 1 ∧ k ≠ 0)) ∧
  (∀ f : ℝ → ℝ, (f ∘ (λ x, 3^x)) = Icc 1 9 → (f ∘ id) = Icc 3 (3^9)) ∧
  (∀ x : ℝ, x ∈ Iio 1 → has_deriv_at (λ x, 1/(1-x)) ((1-x)^(-2)) x  → (1/(1-x)) < (1/(1-x))) ∧
  (∃ x : ℝ, 2^x = log (real.log 2) (abs (x+3)) → x ∈ 2) → 
  q = 3 :=
sorry

end prove_conclusion_correct_l614_614071


namespace problem_1_problem_2_problem_3_l614_614529

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := 6 * Real.log x - a * x^2 - 8 * x + b

theorem problem_1 (a b : ℝ) (h_extremum : x = 3 →
  f x a b = 6 * Real.log x - a * x^2 - 8 * x + b → 
  IsExtremum f 3) : 
  a = -1 := sorry

theorem problem_2 (a b : ℝ) (h_extremum : x = 3 →
  f x a b = 6 * Real.log x - a * x^2 - 8 * x + b → 
  IsExtremum f 3) : 
  ∀ x : ℝ, (0 < x ∧ x < 1) ∨ (3 < x ∧ x < ∞) → f' x > 0 ∧ (1 < x ∧ x < 3) → f' x < 0 := sorry

theorem problem_3 (b : ℝ) (h_intersect : 
  (f 1 (-1) b - 7) * (6 * Real.log 3 + b - 15) < 0) : 
  7 < b ∧ b < 15 - 6 * Real.log 3 := sorry

end problem_1_problem_2_problem_3_l614_614529


namespace solution_set_of_inequality_l614_614521

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_def : ∀ x : ℝ, x ≤ 0 → f x = x^2 + 2 * x) :
  {x : ℝ | f (x + 2) < 3} = {x : ℝ | -5 < x ∧ x < 1} :=
by sorry

end solution_set_of_inequality_l614_614521


namespace range_of_m_l614_614190

variable {m x : ℝ}

theorem range_of_m (h : ∀ x, -1 < x ∧ x < 4 ↔ x > 2 * m ^ 2 - 3) : m ∈ [-1, 1] :=
sorry

end range_of_m_l614_614190


namespace fewest_keystrokes_l614_614041

def fewest_keystrokes_to_reach_1458 (start: ℕ) (keys : list (ℕ → ℕ)) : ℕ :=
  -- function definition to be implemented based on the problem's requirements
  sorry

theorem fewest_keystrokes (start : ℕ) (goal : ℕ) (keys : list (ℕ → ℕ)) (h : start = 1) (g : goal = 1458) :
  fewest_keystrokes_to_reach_1458 start keys = 7 :=
sorry

end fewest_keystrokes_l614_614041


namespace raghu_investment_l614_614389

theorem raghu_investment (R T V : ℝ) (h1 : T = 0.9 * R) (h2 : V = 1.1 * T) (h3 : R + T + V = 5780) : R = 2000 :=
by
  sorry

end raghu_investment_l614_614389


namespace correct_equation_l614_614281

-- Define the initial deposit
def initial_deposit : ℝ := 2500

-- Define the total amount after one year with interest tax deducted
def total_amount : ℝ := 2650

-- Define the annual interest rate
variable (x : ℝ)

-- Define the interest tax rate
def interest_tax_rate : ℝ := 0.20

-- Define the equation for the total amount after one year considering the tax
theorem correct_equation :
  initial_deposit * (1 + (1 - interest_tax_rate) * x) = total_amount :=
sorry

end correct_equation_l614_614281


namespace solution_set_f_g_l614_614266

variable {ℝ : Type*} [LinearOrderedField ℝ]

-- Definitions of odd and even functions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x
def is_monotonically_increasing (h : ℝ → ℝ) (s : Set ℝ) := ∀ x y (hx : x ∈ s) (hy : y ∈ s), x < y → h x < h y

-- Conditions
variable (f g : ℝ → ℝ)
variable (odd_f : is_odd f)
variable (even_g : is_even g)
variable (deriv_cond : ∀ x, x < 0 → (has_deriv_at f x * g x + f x * has_deriv_at g x) > 0)
variable (g_neg3_zero : g (-3) = 0)

-- Proof problem
theorem solution_set_f_g (f g : ℝ → ℝ)
  (odd_f : is_odd f)
  (even_g : is_even g)
  (deriv_cond : ∀ x, x < 0 → (has_deriv_at f x * g x + f x * has_deriv_at g x) > 0)
  (g_neg3_zero : g (-3) = 0) :
  { x | f x * g x < 0 } = {x | -∞ < x ∧ x < -3} ∪ {x | 0 < x ∧ x < 3} :=
  sorry

end solution_set_f_g_l614_614266


namespace non_attacking_quaggas_placement_l614_614751

namespace Quagga

-- Define the quagga's movement
def quagga_move (p q : ℕ × ℕ) : Prop :=
  let dx := abs (p.1 - q.1)
  let dy := abs (p.2 - q.2)
  dx = 6 ∧ dy = 5 ∨ dx = 5 ∧ dy = 6

-- Define the board and vertices
def board := fin 8 × fin 8

-- Define the quagga problem and prove the required statement
def max_non_attacking_quaggas : ℕ :=
  68

theorem non_attacking_quaggas_placement : ∃ n, n = 51 ∧ 
  (∃ placements : fin 51 → board, ∀ i j, i ≠ j → ¬ quagga_move (placements i) (placements j)) :=
begin
  -- assertion to prove 51 non-attacking quaggas result in the correct number of configurations
  use 51,
  split,
  { refl },
  { sorry } -- skipping the proof
end

end Quagga

end non_attacking_quaggas_placement_l614_614751


namespace problem_1_solution_problem_2_solution_l614_614783

section
noncomputable def problem_1 : Real :=
  (5 + 4 / 9) ^ 0.5 + (0.008) ^ (-2 / 3) / (0.2) ^ (-1) / (0.0625) ^ 0.25

theorem problem_1_solution : problem_1 = 44 / 3 := 
by 
  unfold problem_1 
  sorry 

noncomputable def problem_2 : Real :=
  ((1 - Real.log 3 / Real.log 6) ^ 2 + Real.log 2 / Real.log 6 * Real.log 18 / Real.log 6) / (Real.log 4 / Real.log 6)

theorem problem_2_solution : problem_2 = 1 := 
by 
  unfold problem_2 
  sorry 
end

end problem_1_solution_problem_2_solution_l614_614783


namespace range_of_g_l614_614121

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arcsin (x / 3))^2 +
  (π / 2) * Real.arccos (x / 3) -
  (Real.arccos (x / 3))^2 +
  (π^2 / 18) * (x^2 - 3 * x + 9)

theorem range_of_g :
  ∀ x, -3 ≤ x ∧ x ≤ 3 → 
  ∃ y, y = g x ∧ ∀ y, y = g x → (π^2 / 4 ≤ y ∧ y ≤ 5 * π^2 / 4) :=
sorry

end range_of_g_l614_614121


namespace stacy_paper_shortage_l614_614640

theorem stacy_paper_shortage:
  let bought_sheets : ℕ := 240 + 320
  let daily_mwf : ℕ := 60
  let daily_tt : ℕ := 100
  -- Calculate sheets used in a week
  let used_one_week : ℕ := (daily_mwf * 3) + (daily_tt * 2)
  -- Calculate sheets used in two weeks
  let used_two_weeks : ℕ := used_one_week * 2
  -- Remaining sheets at the end of two weeks
  let remaining_sheets : Int := bought_sheets - used_two_weeks
  remaining_sheets = -200 :=
by sorry

end stacy_paper_shortage_l614_614640


namespace ellipse_focal_distance_m_value_l614_614440

-- Define the given conditions 
def focal_distance := 2
def ellipse_equation (x y : ℝ) (m : ℝ) := (x^2 / m) + (y^2 / 4) = 1

-- The proof statement
theorem ellipse_focal_distance_m_value :
  ∀ (m : ℝ), 
    (∃ c : ℝ, (2 * c = focal_distance) ∧ (m = 4 + c^2)) →
      m = 5 := by
  sorry

end ellipse_focal_distance_m_value_l614_614440


namespace periodic_f_l614_614270

variables {α β : Type*} [OrderedField α] [OrderedField β]

-- Definitions of the functions and conditions
noncomputable def f (x : α) : β

variables (a : α)
variables (cond : ∀ x : α, f(x + a) = 1 / 2 + sqrt (f(x) - (f(x)^2)))

-- Proof requirement: ∃ b > 0, ∀ x, f(x + b) = f(x)
theorem periodic_f : ∃ b > 0, ∀ x : α, f(x + b) = f(x) :=
sorry

-- Example for a = 1
example (a : α) (h : a = 1) : ∃ f : α → α, (∀ x : α, f(x + 1) = 1 / 2 + sqrt (f(x) - (f(x)^2))) 
    ∧ (∀ x y, x ≠ y → f(x) ≠ f(y)) :=
sorry

end periodic_f_l614_614270


namespace problem1_problem2_problem3_l614_614082

-- Problem 1
theorem problem1 : sqrt 27 + sqrt 3 - sqrt 12 = 2 * sqrt 3 := sorry

-- Problem 2
theorem problem2 : (1 / sqrt 24) + abs (sqrt 6 - 3) + (1 / 2)⁻¹ - 2016^0 = 4 - 13 * sqrt 6 / 12 := sorry

-- Problem 3
theorem problem3 : (sqrt 3 + sqrt 2)^2 - (sqrt 3 - sqrt 2)^2 = 4 * sqrt 6 := sorry

end problem1_problem2_problem3_l614_614082


namespace compound_interest_rate_l614_614762

-- Conditions from the problem
variables (P : ℝ) (r : ℝ) (n : ℕ)
variables (A : ℕ → ℝ)
axiom A_10 : A 10 = 9000
axiom A_11 : A 11 = 9990

-- Compound interest formula: A(n) = P * (1 + r / 100) ^ n
def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r / 100) ^ n

-- Assertion to be proven
theorem compound_interest_rate :
  (∀ P : ℝ, A 10 = 9000 → A 11 = 9990 → compound_interest P r 10 = 9000 → compound_interest P r 11 = 9990 → r = 11) :=
by
  intros P h10 h11 hci10 hci11
  sorry

end compound_interest_rate_l614_614762


namespace altitudes_concurrent_iff_cyclic_l614_614078

def is_midpoint {A B M : Point} (mAB : Segment A B) (m : Line A B) :=
  on_line M m ∧ dist A M = dist B M

def is_perpendicular {A B C P : Point} (lineA : Line A B) (lineB : Line A C) :=
  angle A B P = 90 ∨ angle A C P = 90

def is_cyclic {A B C D : Point} :=
  ∃ (O : Point) (r : ℝ), ∀ (P ∈ {A, B, C, D}), dist O P = r

def is_concurrent {lines : set (Line)} :=
  ∃ (P : Point), ∀ (l ∈ lines), on_line P l

theorem altitudes_concurrent_iff_cyclic
  (A B C D M N P Q : Point) (mAB : Segment A B) (mBC : Segment B C)
  (mCD : Segment C D) (mDA : Segment D A) (aM : Line)
  (aN : Line) (aP : Line) (aQ : Line) :

  (is_midpoint mAB M ∧ is_midpoint mBC N ∧ is_midpoint mCD P ∧ is_midpoint mDA Q) →
  (is_perpendicular aM mCD ∧ is_perpendicular aN mDA ∧ is_perpendicular aP mAB ∧ is_perpendicular aQ mBC) →
  (is_concurrent {aM, aN, aP, aQ} ↔ is_cyclic A B C D) :=
sorry

end altitudes_concurrent_iff_cyclic_l614_614078


namespace average_diff_of_teacher_and_student_l614_614771

noncomputable def t (class_sizes : List ℕ) : ℚ :=
  (class_sizes.sum : ℚ) / class_sizes.length

noncomputable def s (class_sizes : List ℕ) (total_students : ℚ) : ℚ :=
  (class_sizes.sum (λ c => (c : ℚ)^2)) / total_students

theorem average_diff_of_teacher_and_student (class_sizes : List ℕ) (total_students : ℚ) (total_teachers : ℚ) 
(h_sizes : class_sizes = [80, 40, 40, 20, 10, 5, 3, 2])
(h_students : total_students = 200)
(h_teachers : total_teachers = 8) :
  let t := t class_sizes
  let s := s class_sizes total_students
  t - s = -25.69 :=
by
  have h1 : t = 25 := by sorry
  have h2 : s = 50.69 := by sorry
  rw [h1, h2]
  norm_num

end average_diff_of_teacher_and_student_l614_614771


namespace distance_between_foci_l614_614812

theorem distance_between_foci (x y : ℝ) :
  3 * x^2 - 18 * x - 9 * y^2 - 27 * y = 81 →
  ∃ c : ℝ, 2 * c = 2 * Real.sqrt(39) :=
by
  intro h
  use Real.sqrt 39
  split
  · sorry -- Prove the equation implies the standard form of hyperbola.
  · rfl -- 2 * c = 2 * sqrt(39)

end distance_between_foci_l614_614812


namespace sum_first_15_odd_starting_from_5_l614_614004

-- Definitions based on conditions in the problem.
def a : ℕ := 5    -- First term of the sequence is 5
def n : ℕ := 15   -- Number of terms is 15

-- Define the sequence of odd numbers starting from 5
def oddSeq (i : ℕ) : ℕ := a + 2 * i

-- Define the sum of the first n terms of this sequence
def sumOddSeq : ℕ := ∑ i in Finset.range n, oddSeq i

-- Key statement to prove that the sum of the sequence is 255
theorem sum_first_15_odd_starting_from_5 : sumOddSeq = 255 := by
  sorry

end sum_first_15_odd_starting_from_5_l614_614004


namespace number_of_distinct_intersections_l614_614911

theorem number_of_distinct_intersections :
  (∃ x y : ℝ, 9 * x^2 + 16 * y^2 = 16 ∧ 16 * x^2 + 9 * y^2 = 9) →
  (∀ x y₁ y₂ : ℝ, 9 * x^2 + 16 * y₁^2 = 16 ∧ 16 * x^2 + 9 * y₁^2 = 9 ∧
    9 * x^2 + 16 * y₂^2 = 16 ∧ 16 * x^2 + 9 * y₂^2 = 9 → y₁ = y₂) →
  (∃! p : ℝ × ℝ, 9 * p.1^2 + 16 * p.2^2 = 16 ∧ 16 * p.1^2 + 9 * p.2^2 = 9) :=
by
  sorry

end number_of_distinct_intersections_l614_614911


namespace initial_population_l614_614960

theorem initial_population (P : ℝ) (h1 : 1.05 * (0.765 * P + 50) = 3213) : P = 3935 :=
by
  have h2 : 1.05 * (0.765 * P + 50) = 3213 := h1
  sorry

end initial_population_l614_614960


namespace polygon_E_has_largest_area_l614_614126

-- Define the areas of square and right triangle
def area_square (side : ℕ): ℕ := side * side
def area_right_triangle (leg : ℕ): ℕ := (leg * leg) / 2

-- Define the areas of each polygon
def area_polygon_A : ℕ := 2 * (area_square 2) + (area_right_triangle 2)
def area_polygon_B : ℕ := 3 * (area_square 2)
def area_polygon_C : ℕ := (area_square 2) + 4 * (area_right_triangle 2)
def area_polygon_D : ℕ := 3 * (area_right_triangle 2)
def area_polygon_E : ℕ := 4 * (area_square 2)

-- The theorem assertion
theorem polygon_E_has_largest_area : 
  area_polygon_E = 16 ∧ 
  16 > area_polygon_A ∧
  16 > area_polygon_B ∧
  16 > area_polygon_C ∧
  16 > area_polygon_D := 
sorry

end polygon_E_has_largest_area_l614_614126


namespace unique_zero_in_interval_l614_614874

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * x ^ 2

theorem unique_zero_in_interval
  (a : ℝ) (ha : a > 0)
  (x₀ : ℝ) (hx₀ : f a x₀ = 0)
  (h_interval : -1 < x₀ ∧ x₀ < 0) :
  Real.exp (-2) < x₀ + 1 ∧ x₀ + 1 < Real.exp (-1) :=
sorry

end unique_zero_in_interval_l614_614874


namespace circulation_vector_field_l614_614471

-- Define the vector field
def vectorField (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2, -v.1 * v.3, v.1 * v.2)  -- (y, -xz, xy)

-- Define the closed contour Γ
def contourGamma (v : ℝ × ℝ × ℝ) : Prop := 
  v.1^2 + v.2^2 + v.3^2 = 9 ∧ v.1^2 + v.2^2 = 9  -- x^2 + y^2 + z^2 = 9 and x^2 + y^2 = 9

-- Define the surface Σ from the vector field
def surfaceSigma := 
  {v : ℝ × ℝ × ℝ | v.3 = 0 ∧ v.1^2 + v.2^2 ≤ 9}  -- z = 0 and x^2 + y^2 ≤ 9

-- Prove that the circulation is 9π
theorem circulation_vector_field :
  ∮ (γ : ℝ × ℝ × ℝ) in contourGamma, vectorField γ = 9 * real.pi := 
by
  sorry

end circulation_vector_field_l614_614471


namespace area_relationship_l614_614403

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := π * (39/2)^2 / 2

theorem area_relationship (A B C : ℝ) (h : A + B + 270 = C) :
  A + B + 270 = C :=
sorry

end area_relationship_l614_614403


namespace min_value_of_expression_ge_9_l614_614172

theorem min_value_of_expression_ge_9 
    (x : ℝ)
    (h1 : -2 < x ∧ x < -1)
    (m n : ℝ)
    (a b : ℝ)
    (ha : a = -2)
    (hb : b = -1)
    (h2 : mn > 0)
    (h3 : m * a + n * b + 1 = 0) :
    (2 / m) + (1 / n) ≥ 9 := by
  sorry

end min_value_of_expression_ge_9_l614_614172


namespace problem_statement_l614_614800

variable (sqrt7_pos : 0 < real.sqrt 7)

def M : ℝ := 
  (real.sqrt (real.sqrt 7 + 3) - real.sqrt (real.sqrt 7 - 3)) / real.sqrt (real.sqrt 7 - 1) + real.sqrt (4 - 2 * real.sqrt 3)

theorem problem_statement : M = (3 - real.sqrt 6 + real.sqrt 42) / 6 :=
by
  sorry

end problem_statement_l614_614800


namespace largest_possible_value_l614_614856

-- Definitions for the conditions
def lower_x_bound := -4
def upper_x_bound := -2
def lower_y_bound := 2
def upper_y_bound := 4

-- The proposition to prove
theorem largest_possible_value (x y : ℝ) 
    (h1 : lower_x_bound ≤ x) (h2 : x ≤ upper_x_bound)
    (h3 : lower_y_bound ≤ y) (h4 : y ≤ upper_y_bound) :
    ∃ v, v = (x + y) / x ∧ ∀ (w : ℝ), w = (x + y) / x → w ≤ 1/2 :=
by
  sorry

end largest_possible_value_l614_614856


namespace cosine_inequality_l614_614807
-- Importing the entire Mathlib library to ensure all necessary definitions are available.

-- The main statement of our theorem
theorem cosine_inequality (x y : ℝ) (hx : 0 ≤ x) (hx' : x ≤ π / 2) (hy : 0 ≤ y) (hy' : y ≤ π / 2) :
  cos (x - y) ≥ cos x - cos y :=
begin
  sorry,
end

end cosine_inequality_l614_614807


namespace number_of_blankets_l614_614790

-- Definitions based on the conditions
def blanket_unfolded_area : ℕ := 8 * 8
def total_folded_area : ℕ := 48
def number_of_folds : ℕ := 4

-- Calculations based on the conditions
def folded_area_factor : ℚ := (1:ℚ) / (2^number_of_folds)
def blanket_folded_area : ℚ := blanket_unfolded_area * folded_area_factor

-- The theorem stating that the number of blankets is 12
theorem number_of_blankets : (total_folded_area / blanket_folded_area).to_nat = 12 :=
by
  sorry

end number_of_blankets_l614_614790


namespace distinct_cube_labelings_equal_three_l614_614108

-- Define the main problem
theorem distinct_cube_labelings_equal_three :
  ∃ (label : ℕ → ℕ), (∀ i, label i ∈ {0, 1}) ∧
  (∀ face, ∑ edge in cube_face_edges face, label edge = 2) ∧
  (∀ (a b : ℕ), diagonally_opposite a b → label a = label b) ∧
  (∑ l in distinct_labelings, 1) = 3 :=
sorry

end distinct_cube_labelings_equal_three_l614_614108


namespace greatest_common_multiple_of_10_and_15_less_than_150_l614_614368

-- Definitions based on conditions
def lcm (a b : ℕ) : ℕ := Nat.lcm a b
def is_multiple (x y : ℕ) : Prop := ∃ k, x = y * k

-- Statement of the problem
theorem greatest_common_multiple_of_10_and_15_less_than_150 : 
  ∃ x, is_multiple x (lcm 10 15) ∧ x < 150 ∧ ∀ y, is_multiple y (lcm 10 15) ∧ y < 150 → y ≤ x :=
begin
  sorry
end

end greatest_common_multiple_of_10_and_15_less_than_150_l614_614368


namespace sum_of_squares_of_roots_eq213_l614_614463

theorem sum_of_squares_of_roots_eq213 :
  (∀ r1 r2 : ℝ, (r1 + r2 = 15) ∧ (r1 * r2 = 6) → r1^2 + r2^2 = 213) :=
begin
  intros r1 r2 h,
  cases h with hsum hprod,
  have h1 : (r1 + r2)^2 = r1^2 + 2*r1*r2 + r2^2, by ring,
  rw [hsum, hprod] at h1,
  linarith,
end

end sum_of_squares_of_roots_eq213_l614_614463


namespace bean_lands_outside_inscribed_circle_l614_614575

theorem bean_lands_outside_inscribed_circle :
  let a := 8
  let b := 15
  let c := 17  -- hypotenuse computed as sqrt(a^2 + b^2)
  let area_triangle := (1 / 2) * a * b
  let s := (a + b + c) / 2  -- semiperimeter
  let r := area_triangle / s -- radius of the inscribed circle
  let area_incircle := π * r^2
  let probability_outside := 1 - area_incircle / area_triangle
  probability_outside = 1 - (3 * π) / 20 := 
by
  sorry

end bean_lands_outside_inscribed_circle_l614_614575


namespace total_weight_proof_l614_614780

-- Define molar masses
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.008

-- Define moles of elements in each compound
def moles_C4H10 : ℕ := 8
def moles_C3H8 : ℕ := 5
def moles_CH4 : ℕ := 3

-- Define the molar masses of each compound
def molar_mass_C4H10 : ℝ := 4 * molar_mass_C + 10 * molar_mass_H
def molar_mass_C3H8 : ℝ := 3 * molar_mass_C + 8 * molar_mass_H
def molar_mass_CH4 : ℝ := 1 * molar_mass_C + 4 * molar_mass_H

-- Define the total weight
def total_weight : ℝ :=
  moles_C4H10 * molar_mass_C4H10 +
  moles_C3H8 * molar_mass_C3H8 +
  moles_CH4 * molar_mass_CH4

theorem total_weight_proof :
  total_weight = 733.556 := by
  sorry

end total_weight_proof_l614_614780


namespace limit_of_given_function_l614_614390

theorem limit_of_given_function :
  (∀ x ∈ ℝ, 
    tendsto (λ x, (9 - 2 * x) / 3) (𝓝 3) (𝓝 1) ∧ 
    tendsto (λ x, tan (π * x / 6)) (𝓝 3) at_top) → 
  tendsto (λ x, ((9 - 2 * x) / 3) ^ tan (π * x / 6)) (𝓝 3) (𝓝 (exp (4 / π))) :=
begin
  sorry
end

end limit_of_given_function_l614_614390


namespace problem_statement_l614_614831

def f1(x : ℝ) : ℝ := Real.sin x + Real.cos x
def fn : ℕ → (ℝ → ℝ)
| 0 := f1
| (n + 1) := deriv (fn n)

theorem problem_statement : (Finset.range 2017).sum (λ n, (fn n) (Real.pi / 2)) = 1 := by
  sorry

end problem_statement_l614_614831


namespace angle_boc_eq_angle_aod_l614_614594

-- Define the quadrilateral
variables (A B C D : Point)
variable [convex_quadrilateral A B C D]

-- Define the intersection points
def E := intersection_opposite_sides A B C D
def F := intersection_opposite_sides B C D A

-- Define the diagonals and their intersection point P
def P := intersection_diagonals A C B D

-- Define the foot of the perpendicular from P to EF
def O := foot_perpendicular P (line_through E F)

theorem angle_boc_eq_angle_aod : ∠ (line_through B O) (line_through O C) = ∠ (line_through A O) (line_through O D) :=
by
  -- Proof goes here
  sorry

end angle_boc_eq_angle_aod_l614_614594


namespace min_sum_of_xk_l614_614459

theorem min_sum_of_xk : 
  (∃ x : Fin 50 → ℝ, (∀ i, 0 < x i) ∧ (∀ i, x i = 50) ∧ (∑ i, (1 : ℝ) / (x i) = 1)) 
  → 
  (∀ y : Fin 50 → ℝ, (∀ i, 0 < y i) ∧ (∑ i, ((1 : ℝ) / (y i)) = 1) → (∑ i, y i) ≥ 2500) :=
sorry

end min_sum_of_xk_l614_614459


namespace coefficient_of_x_squared_l614_614226

-- Given the condition:
def expr := (x + 3) ^ 40

-- The statement to prove:
theorem coefficient_of_x_squared : 
  (nat.choose 40 2) * (3 ^ 38) = 780 * 3 ^ 38 :=
by
  sorry

end coefficient_of_x_squared_l614_614226


namespace angle_B_is_pi_over_6_l614_614586

-- Given conditions
def BC : ℝ := 6
def AC : ℝ := 4
def sinA : ℝ := 3 / 4

theorem angle_B_is_pi_over_6 
  (BC AC : ℝ) 
  (sinA : ℝ)
  (hBC : BC = 6)
  (hAC : AC = 4)
  (hsinA : sinA = 3 / 4) : 
  ∃ B, B = π / 6 :=
by
  sorry

end angle_B_is_pi_over_6_l614_614586


namespace max_value_expression_l614_614157

noncomputable def f : Real → Real := λ x => 3 * Real.sin x + 4 * Real.cos x

theorem max_value_expression (θ : Real) (h_max : ∀ x, f x ≤ 5) :
  (3 * Real.sin θ + 4 * Real.cos θ = 5) →
  (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 65 / 7 := by
  sorry

end max_value_expression_l614_614157


namespace sin_double_angle_identity_l614_614515

variable (α : ℝ)

theorem sin_double_angle_identity :
  (cos (2 * α) / sin (α - π / 4) = - √2 / 2) → sin (2 * α) = -3 / 4 :=
by
  sorry

end sin_double_angle_identity_l614_614515


namespace calculate_markup_l614_614496

-- Definitions based on conditions
def cost (S : ℝ) : ℝ := S * 0.60
def markup (S : ℝ) : ℝ := (S - cost S) / (cost S) * 100

-- Given selling price S = 10
def selling_price : ℝ := 10

theorem calculate_markup : markup selling_price = 66 + 2 / 3 := by
  sorry

end calculate_markup_l614_614496


namespace eccentricity_range_l614_614847

theorem eccentricity_range (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : ∀ P : ℝ × ℝ, (P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1) → 
                      (√(a ^ 2 - b ^ 2) / a < ∂x P)) : 
  ∃ e : ℝ, 2 / 3 < e ∧ e < 1 :=
sorry

end eccentricity_range_l614_614847


namespace midpoint_locus_l614_614467

variables {A B C D E X M : Type} [linear_ordered_field A]
variables (AB CB AD CE DE AC : set (A × A))
variables (triangle_ABC : set (A × A))
variables (M : (A × A))
variables (midpoint_DE : (A × A))

-- Conditions
axiom AD_eq_CE : ∀ P Q : set (A × A), P = Q
axiom P_on_AB : ∀ P : set (A × A), P ∈ triangle_ABC
axiom Q_on_CB : ∀ Q : set (A × A), Q ∈ triangle_ABC
axiom X_midpoint_DE : ∀ X : set (A × A), X ∈ (midpoint_DE)
axiom M_midpoint_AC : ∀ M : set (A × A), M ∈ (AC)

-- Proof Statement
theorem midpoint_locus :
  ∀ (triangle_ABC : set (A × A)) (M : (A × A)) (midpoint_DE)
  (AD : set (A × A)) (CE : set (A × A)) (DE : set (A × A)) (AB : set (A × A)) (CB : set (A × A)),
  (AD_eq_CE AD CE) ∧ (P_on_AB AD) ∧ (Q_on_CB CE) ∧ (X_midpoint_DE midpoint_DE) ∧ (M_midpoint_AC M)
  -> (midpoint_DE = line_through M parallel_to angle_bisector_B)
  sorry

end midpoint_locus_l614_614467


namespace problem_1_problem_2_problem_3_problem_4_l614_614448

theorem problem_1 : 2 * Real.sqrt 7 - 6 * Real.sqrt 7 = -4 * Real.sqrt 7 :=
by sorry

theorem problem_2 : Real.sqrt (2 / 3) / Real.sqrt (8 / 27) = (3 / 2) :=
by sorry

theorem problem_3 : Real.sqrt 18 + Real.sqrt 98 - Real.sqrt 27 = (10 * Real.sqrt 2 - 3 * Real.sqrt 3) :=
by sorry

theorem problem_4 : (Real.sqrt 0.5 + Real.sqrt 6) - (Real.sqrt (1 / 8) - Real.sqrt 24) = (Real.sqrt 2 / 4) + 3 * Real.sqrt 6 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l614_614448


namespace expenditure_record_l614_614942

/-- Lean function to represent the condition and the proof problem -/
theorem expenditure_record (income expenditure : Int) (h_income : income = 500) (h_recorded_income : income = 500) (h_expenditure : expenditure = 200) : expenditure = -200 := 
by
  sorry

end expenditure_record_l614_614942


namespace largest_prime_factor_9801_l614_614478

/-- Definition to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Definition for the largest prime factor of a number -/
def largest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else Classical.find (Nat.exists_greatest_prime_factor n h)

/-- Condition: 241 and 41 are prime factors of 9801 -/
def prime_factors_9801 : ∀ (p : ℕ), p ∣ 9801 → is_prime p → (p = 41 ∨ p = 241) :=
λ p hdiv hprime, by {
  obtain ⟨a, ha⟩ := Nat.exists_mul_of_dvd hdiv,
  have h : 9801 = 41 * 241 := rfl,
  have ph31 : is_prime 41 := Nat.Prime_iff.2 ⟨by norm_num, by norm_num⟩,
  have ph241 : is_prime 241 := Nat.Prime_iff.2 ⟨by norm_num, by norm_num⟩,
  have h_9801 : 9801 = 41 * 241, by refl,
  rw [ha, h_9801] at *,
  cases ha with ha l,
  { exact Or.inl ha },
  { exact Or.inr ha },
}

/-- Statement: the largest prime factor of 9801 is 241 -/
theorem largest_prime_factor_9801 : largest_prime_factor 9801 = 241 :=
by {
  rw largest_prime_factor,
  have h1 : 9801 = 41 * 241 := rfl,
  exact Classical.find_spec {
    exists := 241, 
    h := _,
    obtain ⟨m, hm⟩ := Nat.exists_dvd_of_not_prime,
  sorry,
}

end largest_prime_factor_9801_l614_614478


namespace positive_number_property_l614_614749

theorem positive_number_property (x : ℝ) (h_pos : 0 < x) (h_eq : (x^2) / 100 = 9) : x = 30 :=
sorry

end positive_number_property_l614_614749


namespace train_length_300_l614_614765

/-- 
Proving the length of the train given the conditions on crossing times and length of the platform.
-/
theorem train_length_300 (V L : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 200 = V * 30) : 
  L = 300 := 
by
  sorry

end train_length_300_l614_614765


namespace pills_per_day_l614_614280

theorem pills_per_day (total_days : ℕ) (prescription_days_frac : ℚ) (remaining_pills : ℕ) (days_taken : ℕ) (remaining_days : ℕ) (pills_per_day : ℕ)
  (h1 : total_days = 30)
  (h2 : prescription_days_frac = 4/5)
  (h3 : remaining_pills = 12)
  (h4 : days_taken = prescription_days_frac * total_days)
  (h5 : remaining_days = total_days - days_taken)
  (h6 : pills_per_day = remaining_pills / remaining_days) :
  pills_per_day = 2 := by
  sorry

end pills_per_day_l614_614280


namespace checkerboard_covered_squares_l614_614033

theorem checkerboard_covered_squares (D : ℝ) :
  let radius := D
  let side_length := D
  let disc_area := pi * radius^2
  let square_area := side_length^2
  let total_squares := 10 * 10
  let covered_squares := 4
  ∀ (diameter := 2 * D), 
  ∀ (board_size := 10),
  ∀ (squares_covered := covered_squares),
  ∃ (disc : set (real^2)), 
  ∃ (checkerboard : set (ℕ × ℕ)), 
  (disc ∩ checkerboard).card = squares_covered :=
by
  sorry

end checkerboard_covered_squares_l614_614033


namespace a_2017_eq_12_div_7_l614_614666

-- Define the sequence \{a_n\}
def a : ℕ → ℚ
| 0       := 6 / 7
| (n + 1) := if a n > 1 then 2 * a n else a n - 1

theorem a_2017_eq_12_div_7 : a 2017 = 12 / 7 :=
sorry

end a_2017_eq_12_div_7_l614_614666


namespace expression_for_A_plus_2B_A_plus_2B_independent_of_b_l614_614826

theorem expression_for_A_plus_2B (a b : ℝ) : 
  let A := 2 * a^2 + 3 * a * b - 2 * b - 1
  let B := -a^2 - a * b + 1
  A + 2 * B = a * b - 2 * b + 1 :=
by
  sorry

theorem A_plus_2B_independent_of_b (a : ℝ) :
  (∀ b : ℝ, let A := 2 * a^2 + 3 * a * b - 2 * b - 1
            let B := -a^2 - a * b + 1
            A + 2 * B = a * b - 2 * b + 1) →
  a = 2 :=
by
  sorry

end expression_for_A_plus_2B_A_plus_2B_independent_of_b_l614_614826


namespace ratio_r_pq_l614_614387

theorem ratio_r_pq {p q r : ℕ} (h_total : p + q + r = 4000) (h_r : r = 1600) : r / (p + q) = 2 / 3 :=
by
  have h_pq : p + q = 2400 := by linarith
  rw [h_r] at h_total
  rw [h_pq]
  linarith
  /- sorry QByteArray to end -/
  sorry

end ratio_r_pq_l614_614387


namespace tangent_line_circle_l614_614465

theorem tangent_line_circle (r : ℝ) (h : 0 < r) :
    (∃ p : ℝ × ℝ, (p = (0, 0) ∧ (p.fst + p.snd = r)) ∧ ∀ q : ℝ × ℝ,
      q.fst ^ 2 + q.snd ^ 2 = 4 * r → abs ((q.fst + q.snd) - r) / real.sqrt 2 = 2 * real.sqrt r) →
    r = 8 := 
by
  sorry

end tangent_line_circle_l614_614465


namespace range_of_a_l614_614536

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ set.Icc 0 a, f x = x^2 - 2 * x + 3) ∧ 
  (∀ x ∈ set.Icc 0 a, f(x) ≤ 3) ∧
  (∃ x ∈ set.Icc 0 a, f(x) = 3) ∧ 
  (∀ x ∈ set.Icc 0 a, f(x) ≥ 2) ∧ 
  (∃ x ∈ set.Icc 0 a, f(x) = 2) → 
  1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l614_614536


namespace phone_call_answered_within_first_four_rings_l614_614958

def P1 := 0.1
def P2 := 0.3
def P3 := 0.4
def P4 := 0.1

theorem phone_call_answered_within_first_four_rings :
  P1 + P2 + P3 + P4 = 0.9 :=
by
  rw [P1, P2, P3, P4]
  norm_num
  sorry -- Proof step skipped

end phone_call_answered_within_first_four_rings_l614_614958


namespace solve_for_y_l614_614695

theorem solve_for_y (y : ℝ) : (10 - y) ^ 2 = 4 * y ^ 2 → y = 10 / 3 ∨ y = -10 :=
by
  intro h
  -- The proof steps would go here, but we include sorry to allow for compilation.
  sorry

end solve_for_y_l614_614695


namespace horse_oats_meal_l614_614288

-- Define the conditions
variables (num_horses : ℕ) (oats_per_meal : ℕ) (grain_per_day : ℕ) (total_food_needed : ℕ)
variables (days : ℕ)

-- Hypothesis
def conditions := 
  num_horses = 4 ∧ 
  grain_per_day = 3 ∧ 
  total_food_needed = 132 ∧ 
  days = 3 

-- Prove that each horse eats 4 pounds of oats per meal
theorem horse_oats_meal : 
  conditions num_horses oats_per_meal grain_per_day total_food_needed days → 
  2 * num_horses * oats_per_meal + num_horses * grain_per_day = (total_food_needed / days) → 
  oats_per_meal = 4 :=
by 
  sorry

end horse_oats_meal_l614_614288


namespace distance_A_B_l614_614815

-- Define the points A and B
def A : ℝ × ℝ := (-3, 4)
def B : ℝ × ℝ := (6, -2)

-- Define the distance formula for two points in 2D space
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Statement of the problem
theorem distance_A_B : distance A B = real.sqrt 117 := by
  sorry

end distance_A_B_l614_614815


namespace maximize_average_distance_l614_614080

open Classical

noncomputable def maximize_distance (s : ℕ) (n : ℕ) : Prop :=
  s = 4 ∧ (∀ i, if i = n then True else False) ∧ n = 4

theorem maximize_average_distance : 
  ∃ n, maximize_distance 4 n :=
by
  have hyp : maximize_distance 4 4,
  {
    split,
    {
      exact rfl,
    },
    split,
    {
      intro i,
      split_ifs,
      exact True.intro,
      contradiction,
    },
    {
      exact rfl,
    }
  },
  exact ⟨4, hyp⟩

end maximize_average_distance_l614_614080


namespace sum_first_15_odd_from_5_l614_614008

theorem sum_first_15_odd_from_5 : 
  let a₁ := 5 
  let d := 2 
  let n := 15 
  let a₁₅ := a₁ + (n - 1) * d 
  let S := n * (a₁ + a₁₅) / 2 
  S = 285 := by 
  sorry

end sum_first_15_odd_from_5_l614_614008


namespace pairs_of_old_roller_skates_l614_614256

def cars := 2
def bikes := 2
def trash_can := 1
def tricycle := 1
def car_wheels := 4
def bike_wheels := 2
def trash_can_wheels := 2
def tricycle_wheels := 3
def total_wheels := 25

def roller_skates_wheels := 2
def skates_per_pair := 2

theorem pairs_of_old_roller_skates : (total_wheels - (cars * car_wheels + bikes * bike_wheels + trash_can * trash_can_wheels + tricycle * tricycle_wheels)) / roller_skates_wheels / skates_per_pair = 2 := by
  sorry

end pairs_of_old_roller_skates_l614_614256


namespace tobacco_land_increase_l614_614037

def original_total_land : ℕ := 1350
def original_ratio : ℕ × ℕ × ℕ := (5, 2, 2)
def new_ratio : ℕ × ℕ × ℕ := (2, 2, 5)

theorem tobacco_land_increase :
  let original_total_parts := original_ratio.1 + original_ratio.2 + original_ratio.3 in
  let new_total_parts := new_ratio.1 + new_ratio.2 + new_ratio.3 in
  let acres_per_part := original_total_land / original_total_parts in
  let original_tobacco_acres := acres_per_part * original_ratio.3 in
  let new_tobacco_acres := acres_per_part * new_ratio.3 in
  new_tobacco_acres - original_tobacco_acres = 450 :=
by
  sorry

end tobacco_land_increase_l614_614037


namespace solution_l614_614609

variable (a : ℕ → ℝ)

noncomputable def pos_sequence (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → a k > 0

noncomputable def recursive_relation (n : ℕ) : Prop :=
  ∀ n : ℕ, (n > 0) → (n+1) * a (n+1)^2 - n * a n^2 + a (n+1) * a n = 0

noncomputable def sequence_condition (n : ℕ) : Prop :=
  a 1 = 1 ∧ pos_sequence a n ∧ recursive_relation a n

theorem solution : ∀ n : ℕ, n > 0 → sequence_condition a n → a n = 1 / n :=
by
  intros n hn h
  sorry

end solution_l614_614609


namespace arithmetic_sequence_general_term_sum_T_n_l614_614583

noncomputable def sequence_an : ℕ → ℝ
| 0     := 1
| (n+1) := sequence_an n / (2 * sequence_an n + 1)

def seq_recip := λ n : ℕ, 1 / (sequence_an n)

theorem arithmetic_sequence (n : ℕ) :
  seq_recip (n + 1) - seq_recip n = 2 :=
sorry

theorem general_term (n : ℕ) :
  sequence_an n = 1 / (2 * n - 1) :=
sorry

def sequence_bn (n : ℕ) : ℝ :=
  sequence_an n * sequence_an (n + 1)

def T_n (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, sequence_bn i)

theorem sum_T_n (n : ℕ) :
  T_n n = n / (2 * n + 1) :=
sorry

end arithmetic_sequence_general_term_sum_T_n_l614_614583


namespace equation_of_circle_l614_614651

-- Defining the problem conditions directly
variables (a : ℝ) (x y: ℝ)

-- Assume a ≠ 0
variable (h : a ≠ 0)

-- Prove that the circle passing through the origin with center (a, a) has the equation (x - a)^2 + (y - a)^2 = 2a^2.
theorem equation_of_circle (h : a ≠ 0) :
  (x - a)^2 + (y - a)^2 = 2 * a^2 :=
sorry

end equation_of_circle_l614_614651


namespace exponentiation_of_squares_l614_614085

theorem exponentiation_of_squares :
  ((Real.sqrt 2 + 1)^2000 * (Real.sqrt 2 - 1)^2000 = 1) :=
by
  sorry

end exponentiation_of_squares_l614_614085


namespace wire_length_before_cut_l614_614066

theorem wire_length_before_cut (S : ℝ) (L : ℝ) (h1 : S = 4) (h2 : S = (2/5) * L) : S + L = 14 :=
by 
  sorry

end wire_length_before_cut_l614_614066


namespace perimeter_of_plus_sign_shape_l614_614035

theorem perimeter_of_plus_sign_shape (area_total : ℕ) (num_squares : ℕ) (square_side : ℕ) (perimeter : ℕ)
  (H1 : area_total = 648)
  (H2 : num_squares = 8)
  (H3 : area_total / num_squares = square_side * square_side)
  (H4 : perimeter = 12 * square_side) : perimeter = 108 :=
by
  -- Given conditions
  have h_area_single : (648 : ℕ) / (8 : ℕ) = 81 := by norm_num,
  have h_side_length : Math.sqrt 81 = 9 := by norm_num,
  have h_perimeter : 12 * 9 = 108 := by norm_num,
  -- use the conditions to conclude that perimeter equals 108
  exact h_perimeter

end perimeter_of_plus_sign_shape_l614_614035


namespace cube_mono_l614_614194

theorem cube_mono {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_mono_l614_614194


namespace compare_abc_l614_614996

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614996


namespace simple_interest_time_period_l614_614430

variable (SI P R T : ℝ)

theorem simple_interest_time_period (h₁ : SI = 4016.25) (h₂ : P = 8925) (h₃ : R = 9) :
  (P * R * T) / 100 = SI ↔ T = 5 := by
  sorry

end simple_interest_time_period_l614_614430


namespace measure_weights_correct_l614_614692

noncomputable def smallest_number_of_weights : ℕ :=
  4

structure BalanceScale (n : ℕ) :=
  (weights : Finset ℕ)
  (weights_required : weights.card = smallest_number_of_weights)
  (weight_values : ∀ w ∈ weights, w = 1 ∨ w = 3 ∨ w = 9 ∨ w = 27)
  (measure_weights : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 40 → ∃ balance_left balance_right : Finset ℕ, 
                              (balance_left ⊆ weights) ∧ (balance_right ⊆ weights) ∧
                              (balance_left.sum id - balance_right.sum id = k))

theorem measure_weights_correct : 
  ∃ n : ℕ, ∃ s : BalanceScale n, s = { weights := {1, 3, 9, 27}, weights_required := by simp, weight_values := by simp, measure_weights := sorry } :=
begin
  existsi smallest_number_of_weights,
  existsi { weights := {1, 3, 9, 27}, 
            weights_required := by simp [smallest_number_of_weights],
            weight_values := by simp { contextual := true },
            measure_weights := sorry },
  refl
end

end measure_weights_correct_l614_614692


namespace prove_PO_length_l614_614844

open Real EuclideanGeometry

noncomputable def problem_statement : Prop :=
  let a := (0 : ℝ, 0 : ℝ, 0 : ℝ) in
  let b := (4 * sqrt 3 / 2 : ℝ, 2 * sqrt 3 : ℝ, 0 : ℝ) in
  let c := (-4 * sqrt 3 / 2 : ℝ, 2 * sqrt 3 : ℝ, 0 : ℝ) in
  let p := (0 : ℝ, 0 : ℝ, 3 : ℝ) in
  let o := (0 : ℝ, 2 : ℝ, 0 : ℝ) in
  dist p o = sqrt 13

theorem prove_PO_length : problem_statement := by
  sorry

end prove_PO_length_l614_614844


namespace sum_cubic_polynomial_l614_614649

-- Define the cubic polynomial q and the given conditions
def cubic_polynomial (q : ℤ → ℤ) : Prop :=
  q 3 = 3 ∧ q 8 = 23 ∧ q 16 = 13 ∧ q 21 = 33

theorem sum_cubic_polynomial (q : ℤ → ℤ) (h : cubic_polynomial q) :
  (∑ i in finset.range (22 - 2 + 1), q (i + 2)) = 378 :=
begin
  sorry
end

end sum_cubic_polynomial_l614_614649


namespace coefficient_x6_expansion_l614_614362

theorem coefficient_x6_expansion 
  (binomial_expansion : (1 - 3 * x^2) ^ 6 = ∑ k in range 7, (Nat.choose 6 k) * (1:ℝ) ^ (6 - k) * (-3 * x^2) ^ k) :
  (∑ k in range 7, (Nat.choose 6 k) * (1:ℝ) ^ (6 - k) * (-3 * x^2) ^ k).coeff 6 = -540 := 
by
  have term_with_x6 : ∑ k in range 7, (Nat.choose 6 k) * (1:ℝ) ^ (6 - k) * (-3 * x^2) ^ k 
    = (Nat.choose 6 3) * (1:ℝ) ^ 3 * (-3 * x^2) ^ 3 := sorry
  have coefficient_of_x6 : (Nat.choose 6 3 * (-3) ^ 3) = -540 := sorry
  sorry

end coefficient_x6_expansion_l614_614362


namespace sum_of_intercepts_mod_27_l614_614285

theorem sum_of_intercepts_mod_27 :
  (∃ x0 y0, 0 ≤ x0 ∧ x0 < 27 ∧ 0 ≤ y0 ∧ y0 < 27 ∧
    5 * x0 ≡ 2 [MOD 27] ∧ 3 * y0 ≡ 25 [MOD 27] ∧
    x0 + y0 = 40) :=
begin
  sorry
end

end sum_of_intercepts_mod_27_l614_614285


namespace Uki_earnings_l614_614353

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

end Uki_earnings_l614_614353


namespace sum_first_15_odd_from_5_l614_614009

theorem sum_first_15_odd_from_5 : 
  let a₁ := 5 
  let d := 2 
  let n := 15 
  let a₁₅ := a₁ + (n - 1) * d 
  let S := n * (a₁ + a₁₅) / 2 
  S = 285 := by 
  sorry

end sum_first_15_odd_from_5_l614_614009


namespace infinite_common_elements_in_sequences_l614_614788

def a (n : ℕ) : ℕ :=
  if n = 0 then 2
  else if n = 1 then 14
  else 14 * a (n-1) + a (n-2)

def b (n : ℕ) : ℕ :=
  if n = 0 then 2
  else if n = 1 then 14
  else 6 * b (n-1) - b (n-2)

theorem infinite_common_elements_in_sequences : ∃ (S : set ℕ), S.infinite ∧ (∀ s ∈ S, ∃ n : ℕ, s = a n ∧ ∃ m : ℕ, s = b m) :=
sorry

end infinite_common_elements_in_sequences_l614_614788


namespace four_digit_numbers_ending_in_5_divisible_by_25_l614_614180

theorem four_digit_numbers_ending_in_5_divisible_by_25 :
  {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ n % 10 = 5 ∧ n % 25 = 0}.card = 359 :=
begin
  sorry
end

end four_digit_numbers_ending_in_5_divisible_by_25_l614_614180


namespace compare_abc_l614_614997

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614997


namespace largest_prime_factor_9801_l614_614477

theorem largest_prime_factor_9801 : ∃ p : ℕ, Prime p ∧ p ∣ 9801 ∧ ∀ q : ℕ, Prime q ∧ q ∣ 9801 → q ≤ p :=
sorry

end largest_prime_factor_9801_l614_614477


namespace unique_solution_for_log_problem_l614_614884

noncomputable def log_problem (x : ℝ) :=
  let a := Real.log (x / 2 - 1) / Real.log (x - 11 / 4).sqrt
  let b := 2 * Real.log (x - 11 / 4) / Real.log (x / 2 - 1 / 4)
  let c := Real.log (x / 2 - 1 / 4) / (2 * Real.log (x / 2 - 1))
  a * b * c = 2 ∧ (a = b ∧ c = a + 1)

theorem unique_solution_for_log_problem :
  ∃! x, log_problem x = true := sorry

end unique_solution_for_log_problem_l614_614884


namespace find_total_students_l614_614761

theorem find_total_students
  (recorded_bio : ℕ) (actual_bio : ℕ)
  (recorded_chem : ℕ) (actual_chem : ℕ)
  (weight_bio : ℕ) (weight_chem : ℕ)
  (error_increase : ℚ) (initial_average : ℚ)
  (recorded_bio = 83) (actual_bio = 70)
  (recorded_chem = 85) (actual_chem = 75)
  (weight_bio = 50) (weight_chem = 50)
  (error_increase = 0.5) (initial_average = 80) :
  number_of_students = 23 :=
begin
  sorry
end

end find_total_students_l614_614761


namespace perimeter_ABCDE_l614_614225

-- Define the points A, B, C, D, E in R^2
noncomputable def A := (0 : ℝ, 8 : ℝ)
noncomputable def B := (4 : ℝ, 8 : ℝ)
noncomputable def C := (4 : ℝ, 4 : ℝ)
noncomputable def D := (9 : ℝ, 0 : ℝ)
noncomputable def E := (0 : ℝ, 0 : ℝ)

-- Define the distances
def dist (p q : ℝ × ℝ) : ℝ :=
  (real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))

-- The distances AB, AE, and ED
axiom AB_eq_4 : dist A B = 4
axiom AE_eq_8 : dist A E = 8
axiom ED_eq_9 : dist E D = 9

-- Define right angles
axiom AEB_right_angle : (B.2 - A.2) * (E.1 - A.1) + (B.1 - A.1) * (E.2 - A.2) = 0
axiom EAD_right_angle : (D.2 - E.2) * (A.1 - E.1) + (D.1 - E.1) * (A.2 - E.2) = 0
axiom ABC_right_angle : (C.2 - B.2) * (A.1 - B.1) + (C.1 - B.1) * (A.2 - B.2) = 0

open real

-- The final proof statement
theorem perimeter_ABCDE : 
  dist A B + dist B C + dist C D + dist D E + dist E A = 25 + sqrt 41 :=
sorry

end perimeter_ABCDE_l614_614225


namespace joe_investment_rate_l614_614386

def simple_interest_rate (P I T: ℝ) : ℝ :=
  I / (P * T)

theorem joe_investment_rate :
  ∃ (R : ℝ), 
  ∀ (P := 340) (I := 20), R = simple_interest_rate P I 3 ∧ P * 3 * R ≈ 20 / 1020 ∧ R ≈ 0.0196078431372549 :=
by
   sorry

end joe_investment_rate_l614_614386


namespace find_quadratic_polynomial_l614_614480

theorem find_quadratic_polynomial :
  ∃ (p : ℝ[X]), monic p ∧ (3 + complex.I * real.sqrt 2) ∈ p.roots ∧ p = X^2 - 6 * X + 11 :=
by
  sorry

end find_quadratic_polynomial_l614_614480


namespace AE_plus_AP_eq_PD_l614_614241

variables {A B C D E F O P : Type*}
variables {triangle_ABC : ∀ {a b c : ℝ}, (a^2 + b^2 = c^2)}
variables [incircle_O : ∀ (ABC : ∀ {p q r : Type*}, p ≠ q ∧ q ≠ r ∧ p ≠ r), (O p q r : Type*) → (touches_p_q : tangent_to p q ∧ tangent_to q r ∧ tangent_to r p)] 
variables [AD_intersects_O_at_P : ∀ p q : Type*, intersection p q]
variables [angle_BPC_90_deg : ∀ {x y z : ℝ}, (x y z : Type*), angle_eq_deg x y z 90]

theorem AE_plus_AP_eq_PD 
(AE AP PD : ℝ) 
(h_triangle_ABC : ∀ {a b c : ℝ}, a^2 + b^2 = c^2) 
(h_incircle_O : ∀ (ABC : ∀ {p q r : Type*}, p ≠ q ∧ q ≠ r ∧ p ≠ r), (O p q r : Type*) → touches_p_q : tangent_to p q ∧ tangent_to q r ∧ tangent_to r p) 
(h_AD_intersects_O_at_P : ∀ p q : Type*, intersection p q) 
(h_angle_BPC_90_deg : ∀ {x y z : ℝ}, (x y z : Type*), angle_eq_deg x y z 90) :
AE + AP = PD :=
sorry

end AE_plus_AP_eq_PD_l614_614241


namespace second_year_associates_l614_614213

theorem second_year_associates (total_associates : ℕ) (not_first_year : ℕ) (more_than_two_years : ℕ) 
  (h1 : not_first_year = 60 * total_associates / 100) 
  (h2 : more_than_two_years = 30 * total_associates / 100) :
  not_first_year - more_than_two_years = 30 * total_associates / 100 :=
by
  sorry

end second_year_associates_l614_614213


namespace airplane_seat_count_l614_614077

theorem airplane_seat_count (s : ℝ) 
  (h1 : 30 + 0.2 * s + 0.75 * s = s) : 
  s = 600 :=
sorry

end airplane_seat_count_l614_614077


namespace lowest_temperature_denotation_l614_614564

theorem lowest_temperature_denotation (T_high T_low : ℤ) : 
  (T_high = 3) ∧ (T_low = -2) → T_low = -2 := 
by 
  intro h.
  cases h with h1 h2.
  exact h2.

end lowest_temperature_denotation_l614_614564


namespace cos_A_equals_one_third_area_of_triangle_ABC_l614_614562

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (AB AC AM : E)
variable (angle_A angle_B angle_C : ℝ)
variable (S : ℝ)

-- Conditions
axiom triangle_ABC : (angle_A + angle_B + angle_C = π)
axiom a_cos_B_eq_3c_minus_b_cos_A : (a * cos B) = (3 * c - b) * cos A
axiom vector_equation : (AB + AC = 2 * AM)
axiom b_eq_3 : (b = 3)
axiom AM_magnitude : (|AM| = 3 * sqrt 2)

-- Part 1: Prove cos A = 1/3
theorem cos_A_equals_one_third : cos A = 1/3 :=
by sorry

-- Part 2: Find area of triangle ABC
theorem area_of_triangle_ABC : S = 7 * sqrt 2 :=
by sorry

end cos_A_equals_one_third_area_of_triangle_ABC_l614_614562


namespace solve_inequality_l614_614302

open Real

noncomputable def log_x_base_13 (x : ℝ) : ℝ := log x / log 13

noncomputable def lhs (x : ℝ) : ℝ :=
  x^(log_x_base_13 x) + 7 * (x^(1/3))^(log_x_base_13 x)

noncomputable def rhs (x : ℝ) : ℝ :=
  7 + (13^(1/3))^(log_x_base_13 (13))

theorem solve_inequality (x : ℝ) :
  x > 0 → 
  lhs x ≤ rhs x ↔ x ∈ Ioo 0 (13^(-sqrt(log_x_base_13 7))) ∪ {1} ∪ Ici (13^(sqrt (log_x_base_13 7))) :=
sorry

end solve_inequality_l614_614302


namespace correct_equation_l614_614011

theorem correct_equation (x : ℝ) : (-x^2)^2 = x^4 := by sorry

end correct_equation_l614_614011


namespace milk_division_possible_l614_614705

theorem milk_division_possible :
  ∃ (j1 j2 j3 : ℕ), j1 = 6 ∧ j2 = 6 ∧ j3 = 0
  ∧ ∀ a b c : ℕ, (a, b, c) = (12, 0, 0)
  ∧ (∃ steps : list (ℕ × ℕ × ℕ),
    steps.head = (12, 0, 0)
    ∧ (steps.last = (6, 6, 0) ∨ steps.last = none)
    ∧ ¬ steps.empty
    ∧ ∀ i, i < steps.length - 1 →
      let (a, b, c) := steps.nth i in
      let (a', b', c') := steps.nth (i + 1) in
      (a, b, c) = (a', b', c') ∨ (a, b, c) = if a' + b' ≤ 8 then (a' + b', 0, c') else (8, a' + b' - 8, c')) :=
  sorry

end milk_division_possible_l614_614705


namespace compare_abc_l614_614994

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614994


namespace sufficient_but_not_necessary_l614_614542

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem sufficient_but_not_necessary (m n : ℝ) :
  vectors_parallel (m, 1) (n, 1) ↔ (m = n) := sorry

end sufficient_but_not_necessary_l614_614542


namespace find_x_l614_614916

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 := 
by 
  {
    sorry
  }

end find_x_l614_614916


namespace difference_between_sums_l614_614975

noncomputable def sum_of_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def round_to_nearest_5 (x : ℕ) : ℕ :=
  let r := x % 5
  if r < 3 then x - r else x + (5 - r)

noncomputable def sum_of_rounded_integers (n : ℕ) : ℕ :=
  (List.range n).map (λ x => round_to_nearest_5 (x + 1)).sum

theorem difference_between_sums : 
  abs (
    sum_of_first_n_integers 200 - sum_of_rounded_integers 200
  ) = 0 := by
  sorry

end difference_between_sums_l614_614975


namespace sqrt_inequality_l614_614294

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) : 
  sqrt a - sqrt (a - 2) < sqrt (a - 1) - sqrt (a - 3) := 
by
  sorry

end sqrt_inequality_l614_614294


namespace necessary_but_not_sufficient_condition_l614_614769

theorem necessary_but_not_sufficient_condition (a b : ℝ) : 
  (a > b → a + 1 > b) ∧ (∃ a b : ℝ, a + 1 > b ∧ ¬ a > b) :=
by 
  sorry

end necessary_but_not_sufficient_condition_l614_614769


namespace isosceles_triangle_base_vertex_trajectory_l614_614148

theorem isosceles_triangle_base_vertex_trajectory :
  ∀ (x y : ℝ), 
  (∀ (A : ℝ × ℝ) (B : ℝ × ℝ), 
    A = (2, 4) ∧ B = (2, 8) ∧ 
    ((x-2)^2 + (y-4)^2 = 16)) → 
  ((x ≠ 2) ∧ (y ≠ 8) → (x-2)^2 + (y-4)^2 = 16) :=
sorry

end isosceles_triangle_base_vertex_trajectory_l614_614148


namespace friends_meet_probability_l614_614684

noncomputable def probability_of_meeting :=
  let duration_total := 60 -- Total duration from 14:00 to 15:00 in minutes
  let duration_meeting := 30 -- Duration they can meet from 14:00 to 14:30 in minutes
  duration_meeting / duration_total

theorem friends_meet_probability : probability_of_meeting = 1 / 2 := by
  sorry

end friends_meet_probability_l614_614684


namespace initial_video_files_l614_614075

theorem initial_video_files (V : ℕ) (h1 : 26 + V - 48 = 14) : V = 36 := 
by
  sorry

end initial_video_files_l614_614075


namespace area_ratio_cosines_l614_614720

variables (O A B C : Point)
variables (t_A t_B t_C : ℝ)
variables (α β γ σ : ℝ)
variables (OABC_tetrahedron : Tetrahedron O A B C)
variables (pairwise_perpendicular_at_O : 
  ∀ {P Q R : Point}, (P ≠ O ∧ Q ≠ O ∧ R ≠ O ∧ P ≠ Q ∧ Q ≠ R ∧ P ≠ R) → 
  (Perpendicular P O Q) ∧ (Perpendicular Q O R) ∧ (Perpendicular P O R))
variables (areas_of_faces : 
  Area (Face O B C) = t_A ∧ Area (Face O C A) = t_B ∧ Area (Face O A B) = t_C)
variables (spherical_excesses : 
  SphericalExcess (Trihedron A) = α ∧ SphericalExcess (Trihedron B) = β ∧ SphericalExcess (Trihedron C) = γ)
variables (semi_sum_spherical_excesses : 
  σ = (α + β + γ + SphericalExcess (Trihedron O)) / 2)

theorem area_ratio_cosines :
  t_A / t_B = cos (σ - α) / cos (σ - β) ∧
  t_B / t_C = cos (σ - β) / cos (σ - γ) ∧
  t_C / t_A = cos (σ - γ) / cos (σ - α) :=
sorry

end area_ratio_cosines_l614_614720


namespace man_speed_l614_614764

/-- The speed of the man walking in the opposite direction of the train -/
theorem man_speed
  (train_length : ℝ) (cross_time : ℝ) (train_speed_kmh : ℝ)
  (man_speed_kmh : ℝ) : 
  train_length = 500 ∧ cross_time = 10 ∧ train_speed_kmh = 174.98560115190784 ∧
  man_speed_kmh = 5.014396850094164 -> 
  man_speed_kmh = (1.39288801391449 * (3600 / 1000)) :=
begin
  sorry -- Proof omitted
end

end man_speed_l614_614764


namespace main_theorem_l614_614505

-- Define the arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

-- Define the conditions
lemma problem_conditions (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : is_arithmetic_seq a) (h2 : S 3 = 9)
  (h3 : ∃ k : ℝ, (a 0 + 1) * (a 2 + 3) = (a 1 + 1) ^ 2) : Prop :=
-- Prove the general formula
(∀ n, a n = 2 * n - 1) ∧
-- Prove Tn ≥ 1/3
(∀ n, ∑ i in Finset.range n, 1 / (a i * a (i + 1)) ≥ 1 / 3)

-- Main theorem statement
theorem main_theorem (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : is_arithmetic_seq a) (h2 : S 3 = 9)
  (h3 : ∃ k : ℝ, (a 0 + 1) * (a 2 + 3) = (a 1 + 1) ^ 2) : problem_conditions a S h1 h2 h3 :=
sorry

end main_theorem_l614_614505


namespace num_values_satisfy_f_f_eq_5_l614_614879

-- Define the function f : ℝ → ℝ
noncomputable def f : ℝ → ℝ := sorry

-- Conditions from the problem
axiom f_eq_5_iff_x_eq_3 : ∀ x : ℝ, f(x) = 5 ↔ x = 3
axiom f_eq_3_iff_x_in_set : ∀ x : ℝ, f(x) = 3 ↔ (x = -3 ∨ x = 1 ∨ x = 5)

-- The statement to prove
theorem num_values_satisfy_f_f_eq_5 : 
  {x : ℝ | f(f(x)) = 5}.finite.to_finset.card = 3 :=
by
  sorry

end num_values_satisfy_f_f_eq_5_l614_614879


namespace circumscribed_circle_is_nine_point_circle_l614_614709

-- Definitions based on conditions
variable {A B C : Point}
variable {Oa Ob Oc : Point}

def excircle_centers (Oa Ob Oc : Point) : Triangle := Triangle.mk Oa Ob Oc
def feet_of_altitudes (A B C : Point) (T : Triangle) : Prop := 
  -- Definition for feet of the altitudes of triangle
  sorry

-- Main statement
theorem circumscribed_circle_is_nine_point_circle 
  (h1 : excircle_centers Oa Ob Oc = ⟨Oa, Ob, Oc⟩) 
  (h2 : feet_of_altitudes A B C (Triangle.mk Oa Ob Oc)) :
  nine_point_circle (Triangle.mk Oa Ob Oc) = circumscribed_circle (Triangle.mk A B C) :=
sorry

end circumscribed_circle_is_nine_point_circle_l614_614709


namespace area_less_than_one_third_l614_614821

-- Define the absolute value function for real numbers
noncomputable def abs (x : ℝ) : ℝ := if x >= 0 then x else -x

-- Define the first piecewise function y = 1 - |x - 1|
def f (x : ℝ) : ℝ := 1 - abs (x - 1)

-- Define the second piecewise function y = |2x - a|
def g (x a : ℝ) : ℝ := abs (2 * x - a)

-- Define the conditions for a
def condition_a (a : ℝ) : Prop := 1 < a ∧ a < 2

-- Define the intersection points as a set
def intersection_points (a : ℝ) : set (ℝ × ℝ) :=
  { p | ∃ x y, p = (x, y) ∧ (f x = y) ∧ (g x a = y) }

-- Define the function for calculating area of the quadrilateral using given points
def area (a : ℝ) (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2) -
                (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p4.1 + p4.2 * p1.1))

-- Define the proof problem
theorem area_less_than_one_third (a : ℝ) (h : condition_a a) :
  ∃ p1 p2 p3 p4 : ℝ × ℝ,
    p1 ∈ intersection_points a ∧
    p2 ∈ intersection_points a ∧
    p3 ∈ intersection_points a ∧
    p4 ∈ intersection_points a ∧
    area a p1 p2 p3 p4 < 1 / 3 :=
sorry

end area_less_than_one_third_l614_614821


namespace conjugate_fraction_l614_614158

noncomputable def imaginary_unit : ℂ := complex.I
noncomputable def m : ℝ := 2
noncomputable def n : ℝ := -2

theorem conjugate_fraction (i : ℂ) (h1 : i = complex.I) (hm : m = 2) (hn : n = -2) :
  (complex.conj ((m + n * i) / (m - n * i))) = i := by
  -- Proof omitted
  sorry

end conjugate_fraction_l614_614158


namespace sum_of_dice_less_than_10_probability_l614_614737

/-
  Given:
  - A fair die with faces labeled 1, 2, 3, 4, 5, 6.
  - The die is rolled twice.

  Prove that the probability that the sum of the face values is less than 10 is 5/6.
-/

noncomputable def probability_sum_less_than_10 : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 30
  favorable_outcomes / total_outcomes

theorem sum_of_dice_less_than_10_probability :
  probability_sum_less_than_10 = 5 / 6 :=
by
  sorry

end sum_of_dice_less_than_10_probability_l614_614737


namespace sum_of_ages_l614_614977

variable (Joe Jane : ℕ)
variable (h1 : Jane = 16)
variable (h2 : Joe - Jane = 22)

theorem sum_of_ages : Joe + Jane = 54 :=
by
sry

end sum_of_ages_l614_614977


namespace find_x_in_list_l614_614090

theorem find_x_in_list (x : ℚ) :
  let lst := [3, 7, 2, 7, 5, 2, x]
  let mean := (26 + x) / 7
  let median :=
    if x ≤ 2 then 3
    else if 2 < x ∧ x < 5 then x
    else 5
  let mode := 7
  mode = 7 ∧ (if median = 3 then false
              else if median = x then (mean - x = x - 7 ∧ 13 * x = 75)
              else false)
  →
  x = 75 / 13 :=
by
  intros lst mean median mode h
  rw [List.mem_cons_iff] at h
  sorry

end find_x_in_list_l614_614090


namespace product_of_two_numbers_is_21_l614_614328

noncomputable def product_of_two_numbers (x y : ℝ) : ℝ :=
  x * y

theorem product_of_two_numbers_is_21 (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^2 + y^2 = 58) :
  product_of_two_numbers x y = 21 :=
by sorry

end product_of_two_numbers_is_21_l614_614328


namespace maximum_distance_P_to_D_l614_614415

theorem maximum_distance_P_to_D
(point P : ℝ × ℝ)
(square_side : ℝ := 2)
(vertices : list (ℝ × ℝ) := [(0, 0), (2, 0), (2, 2), (0, 2)])
(distances : (ℝ × ℝ) → list ℝ := λ P,
  [(P.1)^2 + (P.2)^2, (P.1 - 2)^2 + (P.2)^2, (P.1 - 2)^2 + (P.2 - 2)^2])
(h : let u := distances P,
         v := distances P,
         w := distances P,
         t := (P.1)^2 + (P.2 - 2)^2
       in u + v + w = t):
  t = real.sqrt 2 := by
  sorry

end maximum_distance_P_to_D_l614_614415


namespace find_third_sum_l614_614214

def arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (a 1) + (a 4) + (a 7) = 39 ∧ (a 2) + (a 5) + (a 8) = 33

theorem find_third_sum (a : ℕ → ℝ)
                       (d : ℝ)
                       (h_seq : arithmetic_sequence_sum a d)
                       (a_1 : ℝ) :
  a 1 = a_1 ∧ a 2 = a_1 + d ∧ a 3 = a_1 + 2 * d ∧
  a 4 = a_1 + 3 * d ∧ a 5 = a_1 + 4 * d ∧ a 6 = a_1 + 5 * d ∧
  a 7 = a_1 + 6 * d ∧ a 8 = a_1 + 7 * d ∧ a 9 = a_1 + 8 * d →
  a 3 + a 6 + a 9 = 27 :=
by
  sorry

end find_third_sum_l614_614214


namespace middle_rectangle_frequency_l614_614237

theorem middle_rectangle_frequency (S A : ℝ) (h1 : S + A = 100) (h2 : A = S / 3) : A = 25 :=
by
  sorry

end middle_rectangle_frequency_l614_614237


namespace length_CF_l614_614642

-- Define points
variables {A C F D B : Type}
-- Define distances
variables {AB CD AF CF : ℝ}
-- Define perpendicularity
variables (perp1 : ∀ x y : Type, x ≠ y → x = y)
variables (perp2 : ∀ x y : Type, x ≠ y → x = y)

-- Hypotheses
hypothesis h1 : C ≠ A
hypothesis h2 : ∀ x : Type, x = x
hypothesis h3 : CD = 9
hypothesis h4 : AF = 15
hypothesis h5 : AB = 6
hypothesis h6 : perp1 D (perp1 C A)
hypothesis h7 : perp2 B (perp2 A F)

-- To Prove
theorem length_CF : CF = 22.5 :=
  by
  sorry

end length_CF_l614_614642


namespace sum_of_g_of_nine_values_l614_614599

def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
def g (y : ℝ) : ℝ := 3 * y - 4

theorem sum_of_g_of_nine_values : (g 9) = 19 := by
  sorry

end sum_of_g_of_nine_values_l614_614599


namespace find_x_l614_614912

theorem find_x (x : ℝ) (h : 2^10 = 32^x) : x = 2 := 
by 
  {
    sorry
  }

end find_x_l614_614912


namespace equation_of_rotated_translated_line_l614_614632

theorem equation_of_rotated_translated_line (x y : ℝ) :
  (∀ x, y = 3 * x → y = x / -3 + 1 / -3) →
  (∀ x, y = -1/3 * (x - 1)) →
  y = -1/3 * x + 1/3 :=
sorry

end equation_of_rotated_translated_line_l614_614632


namespace complex_conjugate_quadrant_l614_614173

noncomputable def determinant (a b c d : Complex) : Complex := a * d - b * c

def complex_conjugate_in_first_quadrant (z : Complex) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem complex_conjugate_quadrant (z : Complex) (h : determinant z (1 + Complex.i) (-Complex.i) (2 * Complex.i) = 0) :
  complex_conjugate_in_first_quadrant z.conj :=
sorry

end complex_conjugate_quadrant_l614_614173


namespace machines_needed_l614_614191

theorem machines_needed (original_machines : ℕ) (original_days : ℕ) (additional_machines : ℕ) :
  original_machines = 12 → original_days = 40 → 
  additional_machines = ((original_machines * original_days) / (3 * original_days / 4)) - original_machines →
  additional_machines = 4 :=
by
  intros h_machines h_days h_additional
  rw [h_machines, h_days] at h_additional
  sorry

end machines_needed_l614_614191


namespace sum_digits_square_minus_sum_digits_plus_one_l614_614606

def S (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_digits_square_minus_sum_digits_plus_one (n : ℕ) :
  n = S(n)^2 - S(n) + 1 ↔ n = 1 ∨ n = 13 ∨ n = 43 ∨ n = 91 ∨ n = 157 :=
by
  sorry

end sum_digits_square_minus_sum_digits_plus_one_l614_614606


namespace cos_neg_two_pi_over_three_eq_l614_614111

noncomputable def cos_neg_two_pi_over_three : ℝ := -2 * Real.pi / 3

theorem cos_neg_two_pi_over_three_eq :
  Real.cos cos_neg_two_pi_over_three = -1 / 2 :=
sorry

end cos_neg_two_pi_over_three_eq_l614_614111


namespace find_x_l614_614921

theorem find_x (x : ℕ) (h : 2^10 = 32^x) (h32 : 32 = 2^5) : x = 2 :=
sorry

end find_x_l614_614921


namespace log_6_15_expression_l614_614132

theorem log_6_15_expression (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  Real.log 15 / Real.log 6 = (b + 1 - a) / (a + b) :=
sorry

end log_6_15_expression_l614_614132


namespace areas_of_cyclic_quadrilateral_and_ortho_quadrilateral_are_equal_l614_614981

noncomputable theory
open_locale classical

variables {A B C D E F G H W X Y Z : Type*}
variables [cyclic_quadrilateral A B C D]
variables [is_midpoint E A B] [is_midpoint F B C] [is_midpoint G C D] [is_midpoint H D A]
variables [is_orthocenter W A H E] [is_orthocenter X B E F] [is_orthocenter Y C F G] [is_orthocenter Z D G H]

theorem areas_of_cyclic_quadrilateral_and_ortho_quadrilateral_are_equal :
  area (quadrilateral A B C D) = area (quadrilateral W X Y Z) :=
sorry

end areas_of_cyclic_quadrilateral_and_ortho_quadrilateral_are_equal_l614_614981


namespace expansion_even_terms_count_l614_614906

theorem expansion_even_terms_count (a b : ℕ) (n : ℕ) :
  (∀ k, (binomial n k) * a^(n-k) * b^k  ∈ ((a + b)^n + (a - b)^n) → k % 2 = 0) ∧
  (∀ k, ((binomial n k) * a^(n-k) * b^k)  ∈ ((a + b)^n + (a - b)^n) → k % 2 = 0) ∧
  (∀ k, (∃ c, c = 2 * (binomial n k) * a^(n-k) * b^k) ∈ ((a + b)^n + (a - b)^n) → ∃ c, c=(binomial n (2*k)) * a^(n-(2*k)) * b^(2*k)) → 
   ∑ k in (range (n+1)), k % 2 = 0) 
   = (n / 2 ) + 1 ∧
  (∀ k, (binomial n (2*k+1))*a^(n-(2*k+1)) * b^(2*k+1)  = (binomial n (2*k+1)) * a^(n-(2*k+1)) (-1)^(2*k+1)  )
 -  ∑  k in (range (n+1)) k % 2 = 1
 =  
  (n / 1) = (left (n / 2))-1 → 
  (n -1) / 2 + 1 
  sorry
  

end expansion_even_terms_count_l614_614906


namespace debby_jogged_total_l614_614284

theorem debby_jogged_total :
  let monday_distance := 2
  let tuesday_distance := 5
  let wednesday_distance := 9
  monday_distance + tuesday_distance + wednesday_distance = 16 :=
by
  sorry

end debby_jogged_total_l614_614284


namespace find_a_l614_614837

noncomputable def tangent_to_circle_and_parallel (a : ℝ) : Prop := 
  let P := (2, 2)
  let circle_center := (1, 0)
  let on_circle := (P.1 - 1)^2 + P.2^2 = 5
  let perpendicular_slope := (P.2 - circle_center.2) / (P.1 - circle_center.1) * (1 / a) = -1
  on_circle ∧ perpendicular_slope

theorem find_a (a : ℝ) : tangent_to_circle_and_parallel a ↔ a = -2 :=
by
  sorry

end find_a_l614_614837


namespace part1_part2_l614_614841

variable {n : ℕ}

theorem part1 (h1 : ∀ (n : ℕ), S (n + 1) = S n + a n + n + 1) (ha1 : a 1 = 1) : ∀ (n : ℕ), a n = (1 + n) * n / 2 := sorry

theorem part2 (h2 : ∀ (n : ℕ), T n = 2 * (1 - 1 / (n + 1))) : ∃ n : ℕ, T n ≥ 19 / 10 ∧ ∀ m : ℕ, T m ≥ 19 / 10 → n ≤ m := sorry

end part1_part2_l614_614841


namespace first_grade_enrollment_l614_614234

theorem first_grade_enrollment (a : ℕ) (R : ℕ) (L : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
  (h3 : a = 25 * R + 10) (h4 : a = 30 * L - 15) : a = 285 :=
by
  sorry

end first_grade_enrollment_l614_614234


namespace probability_black_cubecube_approx_183_times_10_neg_37_l614_614036

theorem probability_black_cubecube_approx_183_times_10_neg_37 :
  let pos_prob := (factorial 8 * factorial 12 * factorial 6) / factorial 27
  let orient_prob := (1 / 8 ^ 8) * (1 / 12 ^ 12) * (1 / 6 ^ 6)
  let total_prob := pos_prob * orient_prob
  total_prob ≈ 1.83e-37 :=
by
  sorry

end probability_black_cubecube_approx_183_times_10_neg_37_l614_614036


namespace width_of_wide_flags_l614_614452

def total_fabric : ℕ := 1000
def leftover_fabric : ℕ := 294
def num_square_flags : ℕ := 16
def square_flag_area : ℕ := 16
def num_tall_flags : ℕ := 10
def tall_flag_area : ℕ := 15
def num_wide_flags : ℕ := 20
def wide_flag_height : ℕ := 3

theorem width_of_wide_flags :
  (total_fabric - leftover_fabric - (num_square_flags * square_flag_area + num_tall_flags * tall_flag_area)) / num_wide_flags / wide_flag_height = 5 :=
by
  sorry

end width_of_wide_flags_l614_614452


namespace log_sum_eq_two_of_exponents_l614_614159

theorem log_sum_eq_two_of_exponents (a b : ℝ) (h1 : 4 ^ a = 10) (h2 : 5 ^ b = 10) :
  (1 / a) + (2 / b) = 2 :=
sorry

end log_sum_eq_two_of_exponents_l614_614159


namespace ellipse_problem_l614_614528

theorem ellipse_problem :
  (∃ (k : ℝ) (a θ : ℝ), 
    (∀ x y : ℝ, y = k * (x + 3) → (x^2 / 25 + y^2 / 16 = 1)) ∧
    (a > -3) ∧
    (∃ x y : ℝ, (x = - (25 / 3) ∧ y = k * (x + 3)) ∧ 
                 (x = D_fst ∧ y = D_snd) ∧ -- Point D(a, θ)
                 (x = M_fst ∧ y = M_snd) ∧ -- Point M
                 (x = N_fst ∧ y = N_snd)) ∧ -- Point N
    (∃ x y : ℝ, (x = -3 ∧ y = 0))) → 
    a = 5 :=
sorry

end ellipse_problem_l614_614528


namespace last_a_transformed_to_s_l614_614044

def shift_n_places (c : char) (n : ℕ) : char :=
  let base := 'a'.val
  let pos := (c.val - base + n) % 26
  (base + pos).to_char

def encrypt_char (c : char) (occurrence : ℕ) : char :=
  shift_n_places c (3 * occurrence)

def count_occurrences (s : string) (target : char) : ℕ :=
  s.foldl (λ count c => if c = target then count + 1 else count) 0

def last_occurrence_transformed (message : string) (target : char) : char :=
  let occurrences := count_occurrences message target
  encrypt_char target occurrences

theorem last_a_transformed_to_s (message : string) (target : char) (h : message = "Java is a Java island") (a_eq_target : target = 'a') :
  last_occurrence_transformed message target = 's' :=
by
  sorry

end last_a_transformed_to_s_l614_614044


namespace esther_walks_975_yards_l614_614613

def miles_to_feet (miles : ℕ) : ℕ := miles * 5280
def feet_to_yards (feet : ℕ) : ℕ := feet / 3

variable (lionel_miles : ℕ) (niklaus_feet : ℕ) (total_feet : ℕ) (esther_yards : ℕ)
variable (h_lionel : lionel_miles = 4)
variable (h_niklaus : niklaus_feet = 1287)
variable (h_total : total_feet = 25332)
variable (h_esther : esther_yards = 975)

theorem esther_walks_975_yards :
  let lionel_distance_in_feet := miles_to_feet lionel_miles
  let combined_distance := lionel_distance_in_feet + niklaus_feet
  let esther_distance_in_feet := total_feet - combined_distance
  feet_to_yards esther_distance_in_feet = esther_yards := by {
    sorry
  }

end esther_walks_975_yards_l614_614613


namespace integers_satisfy_abs_inequality_l614_614182

theorem integers_satisfy_abs_inequality (x : ℤ) : 
  (setOf x : ℤ | abs (7 * x + 2) ≤ 10).card = 4 := 
sorry

end integers_satisfy_abs_inequality_l614_614182


namespace distinct_positives_reciprocal_sum_is_integer_l614_614125

theorem distinct_positives_reciprocal_sum_is_integer :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
    (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) := 
by
  use [2, 3, 6]
  repeat {split};
  sorry

end distinct_positives_reciprocal_sum_is_integer_l614_614125


namespace slower_train_pass_time_l614_614021

-- Define conditions
def length_of_train : ℝ := 500 -- in meters
def speed_of_faster_train : ℝ := 45 * (1000 / 3600) -- convert km/hr to m/s
def speed_of_slower_train : ℝ := 30 * (1000 / 3600) -- convert km/hr to m/s
def relative_speed : ℝ := speed_of_faster_train + speed_of_slower_train

-- Define the proof goal
theorem slower_train_pass_time : 
  time_to_pass = length_of_train / relative_speed → 
  time_to_pass ≈ 24.01 :=
by 
  let length_of_train := 500
  let speed_of_faster_train := 45 * (1000 / 3600)
  let speed_of_slower_train := 30 * (1000 / 3600)
  let relative_speed := speed_of_faster_train + speed_of_slower_train
  let time_to_pass := length_of_train / relative_speed
  have calc_time : time_to_pass = 24.01 := by sorry
  sorry

end slower_train_pass_time_l614_614021


namespace problem1_monotonic_and_extremum_problem2_range_of_k_problem3_inequality_l614_614871

noncomputable def f (x : ℝ) (k : ℝ) := (Real.log x - k - 1) * x

-- Problem 1: Interval of monotonicity and extremum.
theorem problem1_monotonic_and_extremum (k : ℝ):
  (k ≤ 0 → ∀ x, 1 < x → f x k = (Real.log x - k - 1) * x) ∧
  (k > 0 → (∀ x, 1 < x ∧ x < Real.exp k → f x k = (Real.log x - k - 1) * x) ∧
           (∀ x, Real.exp k < x → f x k = (Real.log x - k - 1) * x) ∧
           f (Real.exp k) k = -Real.exp k) := sorry

-- Problem 2: Range of k.
theorem problem2_range_of_k (k : ℝ):
  (∀ x, Real.exp 1 ≤ x ∧ x ≤ Real.exp 2 → f x k < 4 * Real.log x) ↔
  k > 1 - (8 / Real.exp 2) := sorry

-- Problem 3: Inequality involving product of x1 and x2.
theorem problem3_inequality (x1 x2 : ℝ) (k : ℝ):
  x1 ≠ x2 ∧ f x1 k = f x2 k → x1 * x2 < Real.exp (2 * k) := sorry

end problem1_monotonic_and_extremum_problem2_range_of_k_problem3_inequality_l614_614871


namespace base5_to_base2_conversion_l614_614789

theorem base5_to_base2_conversion : 
  ∀ (n : ℕ), (n = 1 * 5^2 + 1 * 5^1 + 0 * 5^0) → (n = 30) → nat.binary n = "11110" :=
by
  intros n h1 h2
  sorry

end base5_to_base2_conversion_l614_614789


namespace monotonic_decreasing_interval_l614_614481

variable {a : ℝ} (h₀ : 0 < a) (h₁ : a < 1)

def f (x : ℝ) : ℝ := Real.log a (x^2 - 5 * x - 6)

theorem monotonic_decreasing_interval : ∀ x, 6 < x → monotone_decreasing (f h₀ h₁) :=
by
  sorry

end monotonic_decreasing_interval_l614_614481


namespace olivia_wallet_l614_614335

theorem olivia_wallet (initial_amount spent_amount remaining_amount : ℕ)
  (h1 : initial_amount = 78)
  (h2 : spent_amount = 15):
  remaining_amount = initial_amount - spent_amount →
  remaining_amount = 63 :=
sorry

end olivia_wallet_l614_614335


namespace compare_abc_l614_614998

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614998


namespace ab_range_l614_614189

theorem ab_range (a b : ℝ) (h : a * b = a + b + 3) : a * b ≤ 1 ∨ a * b ≥ 9 := by
  sorry

end ab_range_l614_614189


namespace min_abc_value_l614_614607

noncomputable def minValue (a b c : ℝ) : ℝ := (a + b) / (a * b * c)

theorem min_abc_value (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
  (minValue a b c) ≥ 16 :=
by
  sorry

end min_abc_value_l614_614607


namespace percentage_students_dislike_but_enjoy_l614_614774

theorem percentage_students_dislike_but_enjoy (total_students : ℕ) 
    (percent_enjoy_painting affirm_enjoy : ℝ) 
    (affirm_dislike : ℝ) : 
    (17.5 / 43 * 100 = 40.698) :=
by
  have h_students_enjoy := total_students * (percent_enjoy_painting / 100)
  have h_students_dislike := total_students * ((100 - percent_enjoy_painting) / 100)
  have h_enjoy_but_dislike := h_students_enjoy * ((100 - affirm_enjoy) / 100)
  have h_dislike_but_affirm := h_students_dislike * (affirm_dislike / 100)
  have h_total_dislike_say := h_enjoy_but_dislike + h_dislike_but_affirm
  have h_fraction_enjoy_dislike := h_enjoy_but_dislike / h_total_dislike_say
  have h_percentage := h_fraction_enjoy_dislike * 100
  have h_solution := (17.5 / 43 : ℝ) * 100
  exact eq.trans h_solution 40.698

end percentage_students_dislike_but_enjoy_l614_614774


namespace evaluate_expression_l614_614109

theorem evaluate_expression : 
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 :=
by
  sorry

end evaluate_expression_l614_614109


namespace lowest_possible_price_l614_614381

theorem lowest_possible_price 
  (MSRP : ℝ)
  (regular_discount_percentage additional_discount_percentage : ℝ)
  (h1 : MSRP = 40)
  (h2 : regular_discount_percentage = 0.30)
  (h3 : additional_discount_percentage = 0.20) : 
  (MSRP * (1 - regular_discount_percentage) * (1 - additional_discount_percentage) = 22.40) := 
by
  sorry

end lowest_possible_price_l614_614381


namespace hexagon_area_l614_614420

noncomputable def area_of_regular_hexagon_inscribed_in_circle (r : ℝ) : ℝ :=
  6 * ( (r * r * real.sqrt 3) / 4 )

theorem hexagon_area {r : ℝ} (h : r = 3) :
  area_of_regular_hexagon_inscribed_in_circle r = 13.5 * real.sqrt 3 := 
by
  rw [h, area_of_regular_hexagon_inscribed_in_circle]
  sorry

end hexagon_area_l614_614420


namespace parametric_to_cartesian_binomial_expansion_monotonic_function_red_envelope_l614_614723

-- Problem 1
theorem parametric_to_cartesian (x y : ℝ) (θ : ℝ) 
  (h1 : x = 3 + 4 * Real.cos θ) 
  (h2 : y = -2 + 4 * Real.sin θ) : 
  (x - 3) ^ 2 + (y + 2) ^ 2 = 16 := 
by sorry

-- Problem 2
theorem binomial_expansion (a_0 a_1 a_2 a_3 a_4 : ℝ) 
  (h : (2 * x + Real.sqrt 3) ^ 4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) : 
  (a_0 + a_2 + a_4) ^ 2 - (a_1 + a_3) ^ 2 = 1 := 
by sorry

-- Problem 3
theorem monotonic_function (k : ℝ) : 
  (∀ x > 1, k - 1 / x ≥ 0) ↔ k ≥ 1 := 
by sorry

-- Problem 4
theorem red_envelope (total : ℝ) (grab_f : set ℝ) (grab_m : set ℝ) (grab_x : set ℝ)
  (h_father : grab_f = {1, 3, 10, 12})
  (h_mother : grab_m = {8, 9, 2, 7} ∨ grab_m = {8, 9, 4, 5})
  (h_sum : sum grab_f = total ∧ sum grab_m = total ∧ sum grab_x = total)
  (h_total: 3 * total = 78) :
  6 ∈ grab_x ∧ 11 ∈ grab_x := 
by sorry

end parametric_to_cartesian_binomial_expansion_monotonic_function_red_envelope_l614_614723


namespace cheese_cookies_price_is_correct_l614_614056

-- Define the problem conditions and constants
def total_boxes_per_carton : ℕ := 15
def total_packs_per_box : ℕ := 12
def discount_15_percent : ℝ := 0.15
def total_number_of_cartons : ℕ := 13
def total_cost_paid : ℝ := 2058

-- Calculate the expected price per pack
noncomputable def price_per_pack : ℝ :=
  let total_packs := total_boxes_per_carton * total_packs_per_box * total_number_of_cartons
  let total_cost_without_discount := total_cost_paid / (1 - discount_15_percent)
  total_cost_without_discount / total_packs

theorem cheese_cookies_price_is_correct : 
  abs (price_per_pack - 1.0347) < 0.0001 :=
by sorry

end cheese_cookies_price_is_correct_l614_614056


namespace probability_no_defective_pencils_l614_614207

theorem probability_no_defective_pencils : 
  let total_pencils := 9
  let defective_pencils := 2
  let chosen_pencils := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_ways := Nat.choose total_pencils chosen_pencils
  let non_defective_ways := Nat.choose non_defective_pencils chosen_pencils
  let probability := non_defective_ways / total_ways
  probability = 5 / 12 := 
by
  sorry

end probability_no_defective_pencils_l614_614207


namespace distance_squared_l614_614295

noncomputable def circumcircle_radius (R : ℝ) : Prop := sorry
noncomputable def excircle_radius (p : ℝ) : Prop := sorry
noncomputable def distance_between_centers (d : ℝ) (R : ℝ) (p : ℝ) : Prop := sorry

theorem distance_squared (R p d : ℝ) (h1 : circumcircle_radius R) (h2 : excircle_radius p) (h3 : distance_between_centers d R p) :
  d^2 = R^2 + 2 * R * p := sorry

end distance_squared_l614_614295


namespace sum_first_15_odd_integers_from_5_l614_614003

theorem sum_first_15_odd_integers_from_5 :
  let a := 5
  let n := 15
  let d := 2
  let last_term := a + (n - 1) * d
  let S := n * a + (n * (n - 1) * d) / 2
  last_term = 37 ∧ S = 315 := by
  sorry

end sum_first_15_odd_integers_from_5_l614_614003


namespace syllogistic_reasoning_l614_614699

theorem syllogistic_reasoning (a b c : Prop) (h1 : b → c) (h2 : a → b) : a → c :=
by sorry

end syllogistic_reasoning_l614_614699


namespace polygon_diagonals_l614_614944

theorem polygon_diagonals (n : ℕ) (h : n - 3 ≤ 6) : n = 9 :=
by sorry

end polygon_diagonals_l614_614944


namespace sample_var_interpretation_l614_614223

theorem sample_var_interpretation (squared_diffs : Fin 10 → ℝ) :
  (10 = 10) ∧ (∀ i, squared_diffs i = (i - 20)^2) →
  (∃ n: ℕ, n = 10 ∧ ∃ μ: ℝ, μ = 20) :=
by
  intro h
  sorry

end sample_var_interpretation_l614_614223


namespace number_of_ways_to_divide_10_staircase_l614_614793

def n_staircase (n : ℕ) : set (ℕ × ℕ) := {p | p.1 < n ∧ p.2 < n ∧ p.1 ≥ p.2}

def is_divisible_by_10_rectangles : ℕ → Prop
| 0 := true
| n + 1 := 
  if h : n + 1 = 10 then 
    let rect_sets := ({(i, j) | i = k ∧ j < k} ∪ {(i, j) | j = k ∧ i < k} : set (ℕ × ℕ)) in
    ∃ (k : ℕ), k ≤ n + 1 ∧ (∀ {p}, p ∈ rect_sets → p ∈ ({(i, j) | i < n + 1 ∧ j < n + 1 ∧ i ≥ j} : set (ℕ × ℕ)))
  else 
    false

theorem number_of_ways_to_divide_10_staircase : is_divisible_by_10_rectangles 10 = 256 := 
sorry

end number_of_ways_to_divide_10_staircase_l614_614793


namespace number_of_true_propositions_is_three_l614_614786

-- Define the original proposition
def original_proposition (l : Type) (α : Type) [line l] [plane α] :=
  (∀ (line_in_α : Type) [line_in_α ⊆ α], perpendicular l line_in_α) → perpendicular l α

-- Define the converse of the original proposition
def converse_proposition (l : Type) (α : Type) [line l] [plane α] :=
  perpendicular l α → (∀ (line_in_α : Type) [line_in_α ⊆ α], perpendicular l line_in_α)

-- Define the inverse of the original proposition
def inverse_proposition (l : Type) (α : Type) [line l] [plane α] :=
  (¬ ∀ (line_in_α : Type) [line_in_α ⊆ α], perpendicular l line_in_α) → ¬ perpendicular l α

-- Define the contrapositive of the original proposition
def contrapositive_proposition (l : Type) (α : Type) [line l] [plane α] :=
  ¬ perpendicular l α → ¬ ∀ (line_in_α : Type) [line_in_α ⊆ α], perpendicular l line_in_α

-- The Lean 4 statement to prove
theorem number_of_true_propositions_is_three (l : Type) (α : Type) [line l] [plane α] :
  original_proposition l α ∧ converse_proposition l α ∧ inverse_proposition l α ∧ contrapositive_proposition l α →
  3 = (if (converse_proposition l α) then 1 else 0) + 
      (if (inverse_proposition l α) then 1 else 0) + 
      (if (contrapositive_proposition l α) then 1 else 0) := 
sorry

end number_of_true_propositions_is_three_l614_614786


namespace speed_of_current_is_6_l614_614668

noncomputable def speed_of_current : ℝ :=
  let Vm := 18  -- speed in still water in kmph
  let distance_m := 100  -- distance covered in meters
  let time_s := 14.998800095992323  -- time taken in seconds
  let distance_km := distance_m / 1000  -- converting distance to kilometers
  let time_h := time_s / 3600  -- converting time to hours
  let Vd := distance_km / time_h  -- speed downstream in kmph
  Vd - Vm  -- speed of the current

theorem speed_of_current_is_6 :
  speed_of_current = 6 := by
  sorry -- proof is skipped

end speed_of_current_is_6_l614_614668


namespace all_are_multiples_of_3_l614_614183

theorem all_are_multiples_of_3 :
  (123 % 3 = 0) ∧
  (234 % 3 = 0) ∧
  (345 % 3 = 0) ∧
  (456 % 3 = 0) ∧
  (567 % 3 = 0) :=
by
  sorry

end all_are_multiples_of_3_l614_614183


namespace cot_product_leq_sqrt3_div9_l614_614952

variable {A B C : ℝ}

-- Definitions: A, B, and C are angles of a triangle
def is_triangle (A B C : ℝ) : Prop := A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0

theorem cot_product_leq_sqrt3_div9 (A B C : ℝ) (h : is_triangle A B C) :
  Real.cot A * Real.cot B * Real.cot C ≤ Real.sqrt 3 / 9 :=
by
  sorry

end cot_product_leq_sqrt3_div9_l614_614952


namespace pump_time_calculation_l614_614409

-- Definitions of the conditions
def depth_in_inches : ℕ := 12
def depth_in_feet : ℕ := depth_in_inches / 12
def length_in_feet : ℕ := 20
def width_in_feet : ℕ := 25
def volume_in_cubic_feet : ℕ := (depth_in_feet * length_in_feet * width_in_feet)
def cubic_feet_to_gallons : ℕ := 75 / 10
def total_gallons : ℕ := volume_in_cubic_feet * cubic_feet_to_gallons
def pumps : ℕ := 2
def pump_rate : ℕ := 10
def total_pump_rate : ℕ := pumps * pump_rate
def time_required : ℕ := total_gallons / total_pump_rate

theorem pump_time_calculation :
  time_required = 187.5 := by
  sorry

end pump_time_calculation_l614_614409


namespace fraction_arithmetic_l614_614374

theorem fraction_arithmetic : ((3 / 5 : ℚ) + (4 / 15)) * (2 / 3) = 26 / 45 := 
by
  sorry

end fraction_arithmetic_l614_614374


namespace unique_abs_value_of_solving_quadratic_l614_614940

theorem unique_abs_value_of_solving_quadratic :
  ∀ z : ℂ, (z^2 - 6*z + 20 = 0) → (complex.abs z = complex.sqrt 53) :=
begin
  sorry
end

end unique_abs_value_of_solving_quadratic_l614_614940


namespace mr_johnson_pill_intake_l614_614277

theorem mr_johnson_pill_intake (total_days : ℕ) (remaining_pills : ℕ) (fraction : ℚ) (dose : ℕ)
  (h1 : total_days = 30)
  (h2 : remaining_pills = 12)
  (h3 : fraction = 4 / 5) :
  dose = 2 :=
by
  sorry

end mr_johnson_pill_intake_l614_614277


namespace king_coin_problem_l614_614743

noncomputable def P1 := 1 - 0.99^10
noncomputable def P2 := 1 - (49 / 50)^5

theorem king_coin_problem : P1 < P2 :=
by
  sorry

end king_coin_problem_l614_614743


namespace no_positive_integer_n_satisfies_conditions_l614_614822

theorem no_positive_integer_n_satisfies_conditions :
  ¬ ∃ (n : ℕ), (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end no_positive_integer_n_satisfies_conditions_l614_614822


namespace apples_count_l614_614038

theorem apples_count :
    ∃ A, (72 + 32 = 104) ∧ (104 = A + 26) ∧ (A = 78) := 
begin
  have total_bottles := 72 + 32,
  have total_bottles_eq := total_bottles = 104,
  have bottles_apples_rel := 104 = A + 26,

  use 78,
  split,
  {
    exact total_bottles_eq,
  },
  split,
  {
    exact bottles_apples_rel,
  },
  {
    sorry,  -- Here is where the proof would go
  }
end

end apples_count_l614_614038


namespace sandbox_length_l614_614423

theorem sandbox_length (width : ℕ) (area : ℕ) (h_width : width = 146) (h_area : area = 45552) : ∃ length : ℕ, length = 312 :=
by {
  sorry
}

end sandbox_length_l614_614423


namespace part_a_part_b_l614_614983

noncomputable def alpha_eq_prod (α : ℝ) (n : ℕ → ℕ) : Prop :=
  1 < α ∧ α < 2 ∧ ∀ i, n i > 0 ∧ n i ^ 2 ≤ n (i + 1) ∧ 
    α = (1 + 1 / (n 1 : ℝ)) * (1 + 1 / (n 2 : ℝ)) * (1 + 1 / (n 3 : ℝ)) * ...

theorem part_a (α : ℝ) (n : ℕ → ℕ) : 
  1 < α ∧ α < 2 → ∃ n : ℕ → ℕ, ∀ i, n i > 0 ∧ n i ^ 2 ≤ n (i + 1) ∧ 
    α = (1 + 1 / (n 1 : ℝ)) * (1 + 1 / (n 2 : ℝ)) * (1 + 1 / (n 3 : ℝ)) * ...
:= sorry

noncomputable def rational_condition (n : ℕ → ℕ) : Prop :=
  ∃ k, ∀ i ≥ k, n (i + 1) = n i ^ 2

theorem part_b (α : ℝ) (n : ℕ → ℕ) : 
  (∃ n : ℕ → ℕ, ∀ i, n i > 0 ∧ n i ^ 2 ≤ n (i + 1) ∧ 
    α = (1 + 1 / (n 1 : ℝ)) * (1 + 1 / (n 2 : ℝ)) * (1 + 1 / (n 3 : ℝ)) * ...) ↔ rational_condition n
:= sorry

end part_a_part_b_l614_614983


namespace number_of_intersections_l614_614330

theorem number_of_intersections (lines : Fin 5 → affine_subspace ℝ (Fin 2 → ℝ)) (distinct : ∀ i j, i ≠ j → lines i ≠ lines j) :
  ∃ ns : Finset ℕ, (∀ n ∈ ns, n ≤ 10) ∧ ∀ (k : ℕ), k ∈ ns ↔ k ∈ {0, 1, 4, 5, 6, 7, 8, 9, 10} ∧ ns.card = 9 :=
by
  sorry

end number_of_intersections_l614_614330


namespace evaluate_expression_l614_614468

theorem evaluate_expression : 
  Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ) + Int.ceil (4 / 5 : ℚ) + Int.floor (-4 / 5 : ℚ) = 0 :=
by
  sorry

end evaluate_expression_l614_614468


namespace first_grade_children_count_l614_614228

theorem first_grade_children_count (a : ℕ) (R L : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a = 25 * R + 10 ∧ a = 30 * L - 15 ∧ (R > 0 ∧ L > 0) → a = 285 :=
by
  sorry

end first_grade_children_count_l614_614228


namespace ball_travel_distance_l614_614059

noncomputable def total_distance_after_fifth_bounce (initial_height : ℕ) (rebound_ratio : ℚ) : ℚ :=
  let descent := λ n, initial_height * (rebound_ratio ^ n)
  let ascent := λ n, initial_height * (rebound_ratio ^ (n + 1))
  initial_height + -- first descent
    descent 1 + descent 2 + descent 3 + descent 4 + descent 5 + 
    ascent 0 + ascent 1 + ascent 2 + ascent 3

theorem ball_travel_distance : total_distance_after_fifth_bounce 120 (1 / 3 : ℚ) = 5000 / 27 := by
  sorry

end ball_travel_distance_l614_614059


namespace uki_total_earnings_l614_614352

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def cupcakes_per_day : ℕ := 20
def cookies_per_day : ℕ := 10
def biscuits_per_day : ℕ := 20
def days : ℕ := 5

-- Prove the total earnings for five days
theorem uki_total_earnings : 
    (cupcakes_per_day * price_cupcake + 
     cookies_per_day * price_cookie + 
     biscuits_per_day * price_biscuit) * days = 350 := 
by
  -- The actual proof will go here, but is omitted for now.
  sorry

end uki_total_earnings_l614_614352


namespace johns_donation_l614_614555

theorem johns_donation (A : ℝ) (T : ℝ) (J : ℝ) (h1 : A + 0.5 * A = 75) (h2 : T = 3 * A) 
                       (h3 : (T + J) / 4 = 75) : J = 150 := by
  sorry

end johns_donation_l614_614555


namespace one_sofa_in_room_l614_614334

def num_sofas_in_room : ℕ :=
  let num_4_leg_tables := 4
  let num_4_leg_chairs := 2
  let num_3_leg_tables := 3
  let num_1_leg_table := 1
  let num_2_leg_rocking_chairs := 1
  let total_legs := 40

  let legs_of_4_leg_tables := num_4_leg_tables * 4
  let legs_of_4_leg_chairs := num_4_leg_chairs * 4
  let legs_of_3_leg_tables := num_3_leg_tables * 3
  let legs_of_1_leg_table := num_1_leg_table * 1
  let legs_of_2_leg_rocking_chairs := num_2_leg_rocking_chairs * 2

  let accounted_legs := legs_of_4_leg_tables + legs_of_4_leg_chairs + legs_of_3_leg_tables + legs_of_1_leg_table + legs_of_2_leg_rocking_chairs

  let remaining_legs := total_legs - accounted_legs

  let sofa_legs := 4
  remaining_legs / sofa_legs

theorem one_sofa_in_room : num_sofas_in_room = 1 :=
  by
    unfold num_sofas_in_room
    rfl

end one_sofa_in_room_l614_614334


namespace winner_has_305_votes_l614_614209

-- Define initial votes for each candidate
def initial_votes := {A := 220, B := 180, C := 165, D := 145, E := 90}

-- Define votes after the first elimination and redistribution
def votes_after_first := {A := 230, B := 195, C := 169, D := 155}

-- Define votes after the second elimination and redistribution
def votes_after_second := {A := 248, B := 210, C := 208}

-- Define votes after the third elimination and redistribution
def votes_after_third := {A := 295, B := 305}

-- The number of votes for the winning candidate
def winner_votes := votes_after_third.B

theorem winner_has_305_votes : winner_votes = 305 := by
  sorry

end winner_has_305_votes_l614_614209


namespace correct_number_of_propositions_l614_614433

-- Define the conditions 
def proposition1_correct : Prop :=
  ∀ (residual_sum_squares : ℝ), (residual_sum_squares < some_threshold) → (fitting_effect > another_threshold)

def proposition2_correct : Prop :=
  ∀ (R_squared : ℝ), (R_squared or (1 - R_squared) < some_threshold) → false

def proposition3_correct : Prop :=
  ∀ (scatter_points : list (ℝ × ℝ)) (regression_line : ℝ → ℝ), scatter_points_are_near_line scatter_points regression_line → false

def proposition4_correct : Prop :=
  ∀ (e : ℕ → ℝ), (E e = 0) → forecasting_accuracy_is_measurable_by_variance (variance e)

-- Define the main problem statement
theorem correct_number_of_propositions (proposition1 proposition2 proposition3 proposition4 : Prop) : 
  (proposition1 → proposition1_correct) →
  (proposition2 → proposition2_correct) →
  (proposition3 → proposition3_correct) →
  (proposition4 → proposition4_correct) →
  (count_correct_propositions [proposition1, proposition2, proposition3, proposition4] = 2) :=
  sorry

-- Helper functions and definitions (skipped for brevity)
def scatter_points_are_near_line : list (ℝ × ℝ) → (ℝ → ℝ) → Prop := sorry
def forecasting_accuracy_is_measurable_by_variance : (ℝ → ℝ) → Prop := sorry
def count_correct_propositions : list Prop → ℕ := sorry
def E : (ℕ → ℝ) → ℝ := sorry
def variance : (ℕ → ℝ) → ℝ := sorry

end correct_number_of_propositions_l614_614433


namespace perimeter_of_cube_face_is_28_l614_614665

-- Define the volume of the cube
def volume_of_cube : ℝ := 343

-- Define the side length of the cube based on the volume
def side_length_of_cube : ℝ := volume_of_cube^(1/3)

-- Define the perimeter of one face of the cube
def perimeter_of_one_face (side_length : ℝ) : ℝ := 4 * side_length

-- Theorem: Prove the perimeter of one face of the cube is 28 cm given the volume is 343 cm³
theorem perimeter_of_cube_face_is_28 : 
  perimeter_of_one_face side_length_of_cube = 28 := 
by
  sorry

end perimeter_of_cube_face_is_28_l614_614665


namespace calc_length_of_first_road_l614_614032

noncomputable def length_of_first_road (M : ℕ) : ℕ :=
  M / 30

theorem calc_length_of_first_road (M L : ℕ) :
  (L = M / 30) :=
by
  have h₁ : 96 * M = 20 * 28.8 * 10 * L / 2,
  calc ..., -- Insert the detailed calculation steps here
  have h₂ : L = M / 30,
  calc L = (M * 192) / 5760 : by ..., -- Insert simplification steps here
  exact sorry

end calc_length_of_first_road_l614_614032


namespace find_quadrant_l614_614498

def isFourthQuadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem find_quadrant {z : ℂ} (h : (conj z) / (3 + I) = 1 + I) : isFourthQuadrant z :=
by 
  sorry

end find_quadrant_l614_614498


namespace collinear_iff_l614_614156

section collinear_proof
variables (a b : Vector ℝ) (λ μ : ℝ)
-- Condition: \overrightarrow{a} and \overrightarrow{b} are non-collinear vectors.
variables (non_collinear : ¬ collinear ℝ ![a, b])
-- Condition: \overrightarrow{AB} = \lambda \overrightarrow{a} + \overrightarrow{b}
def AB : Vector ℝ := λ • a + b
-- Condition: \overrightarrow{AC} = \overrightarrow{a} + \mu \overrightarrow{b}
def AC : Vector ℝ := a + μ • b

-- The question and what we need to prove
theorem collinear_iff (hAB : AB a b λ = λ • a + b) (hAC : AC a b μ = a + μ • b) :
  collinear ℝ ![a, b, AB a b λ, AC a b μ] ↔ λ * μ = 1 :=
sorry
end collinear_proof

end collinear_iff_l614_614156


namespace find_x_l614_614919

theorem find_x (x : ℕ) (h : 2^10 = 32^x) (h32 : 32 = 2^5) : x = 2 :=
sorry

end find_x_l614_614919


namespace meaningful_expression_range_l614_614945

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = ⟦x⟧ → safe_div (sqrt (y + 4)) (y - 2)) ↔ (x ≥ -4 ∧ x ≠ 2) :=
sorry

end meaningful_expression_range_l614_614945


namespace unattainable_y_l614_614819

theorem unattainable_y (x : ℚ) (hx : x ≠ -4 / 3) : 
    ∀ y : ℚ, (y = (2 - x) / (3 * x + 4)) → y ≠ -1 / 3 :=
sorry

end unattainable_y_l614_614819


namespace real_part_exists_l614_614093

noncomputable def problem_statement : Prop :=
  ∃ (a b: ℝ), b > 0 ∧ (a + b * complex.I - 2 * complex.I) *
  (a + b * complex.I + 2 * complex.I) *
  (a + b * complex.I - 3 * complex.I) = 1234 * complex.I

theorem real_part_exists : problem_statement :=
  sorry

end real_part_exists_l614_614093


namespace compare_abc_l614_614991

def a : ℝ := 2 * log 1.01
def b : ℝ := log 1.02
def c : ℝ := real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l614_614991


namespace bicycles_in_garage_l614_614332

theorem bicycles_in_garage 
  (B : ℕ) 
  (h1 : 4 * 3 = 12) 
  (h2 : 7 * 1 = 7) 
  (h3 : 2 * B + 12 + 7 = 25) : 
  B = 3 := 
by
  sorry

end bicycles_in_garage_l614_614332


namespace u_2002_eq_2_l614_614741

noncomputable def g : ℕ → ℕ
| 1 := 5
| 2 := 3
| 3 := 4
| 4 := 2
| 5 := 1
| _ := 0 -- necessary to handle other inputs, even though in given problem only 1-5 are used.

def u : ℕ → ℕ
| 0 := 3
| (n+1) := g (u n)

-- Theorem to state the problem
theorem u_2002_eq_2 : u 2002 = 2 :=
by sorry

end u_2002_eq_2_l614_614741


namespace tangent_line_equation_l614_614834

noncomputable def circle_eq1 (x y : ℝ) := x^2 + (y - 2)^2 - 4
noncomputable def circle_eq2 (x y : ℝ) := (x - 3)^2 + (y + 2)^2 - 21
noncomputable def line_eq (x y : ℝ) := 3*x - 4*y - 4

theorem tangent_line_equation :
  ∀ (x y : ℝ), (circle_eq1 x y = 0 ∧ circle_eq2 x y = 0) ↔ line_eq x y = 0 :=
sorry

end tangent_line_equation_l614_614834


namespace quadratic_roots_value_r_l614_614265

theorem quadratic_roots_value_r
  (a b m p r : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h_root1 : a^2 - m*a + 3 = 0)
  (h_root2 : b^2 - m*b + 3 = 0)
  (h_ab : a * b = 3)
  (h_root3 : (a + 1/b) * (b + 1/a) = r) :
  r = 16 / 3 :=
sorry

end quadratic_roots_value_r_l614_614265


namespace find_ordered_pair_l614_614482

theorem find_ordered_pair (x y : ℤ) (h1 : x + y = (7 - x) + (7 - y)) (h2 : x - y = (x + 1) + (y + 1)) :
  (x, y) = (8, -1) := by
  sorry

end find_ordered_pair_l614_614482


namespace minimum_value_of_xy_is_e_l614_614501

noncomputable def minimum_value_of_xy : ℝ :=
  if hx : (x : ℝ) > 1 then
    if hy : (y : ℝ) > 1 then
      if hgeo : (ln x : ℝ) * (ln y : ℝ) = 1 / 4 then
        e
      else
        0
    else
      0
  else
    0

theorem minimum_value_of_xy_is_e (x y : ℝ) (hx : x > 1) (hy : y > 1) (hgeo: (ln x) * (ln y) = 1 / 4) : (x * y) ≥ e :=
by
  sorry

end minimum_value_of_xy_is_e_l614_614501


namespace math_competition_l614_614206

theorem math_competition :
  let Sammy_score := 20
  let Gab_score := 2 * Sammy_score
  let Cher_score := 2 * Gab_score
  let Total_score := Sammy_score + Gab_score + Cher_score
  let Opponent_score := 85
  Total_score - Opponent_score = 55 :=
by
  sorry

end math_competition_l614_614206


namespace ny_mets_fans_count_l614_614019

theorem ny_mets_fans_count (Y M R : ℕ) (h1 : 3 * M = 2 * Y) (h2 : 4 * R = 5 * M) (h3 : Y + M + R = 390) : M = 104 := 
by
  sorry

end ny_mets_fans_count_l614_614019


namespace fill_bathtub_with_drain_open_l614_614400

theorem fill_bathtub_with_drain_open :
  let fill_rate := 1 / 10
  let drain_rate := 1 / 12
  let net_fill_rate := fill_rate - drain_rate
  fill_rate = 1 / 10 ∧ drain_rate = 1 / 12 → 1 / net_fill_rate = 60 :=
by
  intros
  sorry

end fill_bathtub_with_drain_open_l614_614400


namespace Jose_age_proof_l614_614246

-- Definitions based on the conditions
def Inez_age : ℕ := 15
def Zack_age : ℕ := Inez_age + 5
def Jose_age : ℕ := Zack_age - 7

theorem Jose_age_proof : Jose_age = 13 :=
by
  -- Proof omitted
  sorry

end Jose_age_proof_l614_614246


namespace original_pencils_l614_614676

-- Define the conditions given in the problem
variable (total_pencils_now : ℕ) [DecidableEq ℕ] (pencils_by_Mike : ℕ)

-- State the problem to prove
theorem original_pencils (h1 : total_pencils_now = 71) (h2 : pencils_by_Mike = 30) : total_pencils_now - pencils_by_Mike = 41 := by
  sorry

end original_pencils_l614_614676


namespace final_pens_count_l614_614717

-- Define the initial number of pens and subsequent operations
def initial_pens : ℕ := 7
def pens_after_mike (initial : ℕ) : ℕ := initial + 22
def pens_after_cindy (pens : ℕ) : ℕ := pens * 2
def pens_after_sharon (pens : ℕ) : ℕ := pens - 19

-- Prove that the final number of pens is 39
theorem final_pens_count : pens_after_sharon (pens_after_cindy (pens_after_mike initial_pens)) = 39 := 
sorry

end final_pens_count_l614_614717


namespace even_two_digit_numbers_count_l614_614184

/-- Even positive integers less than 1000 with at most two different digits -/
def count_even_two_digit_numbers : ℕ :=
  let one_digit := [2, 4, 6, 8].length
  let two_d_same := [22, 44, 66, 88].length
  let two_d_diff := [24, 42, 26, 62, 28, 82, 46, 64, 48, 84, 68, 86].length
  let three_d_same := [222, 444, 666, 888].length
  let three_d_diff := 16 + 12
  one_digit + two_d_same + two_d_diff + three_d_same + three_d_diff

theorem even_two_digit_numbers_count :
  count_even_two_digit_numbers = 52 :=
by sorry

end even_two_digit_numbers_count_l614_614184


namespace number_of_rectangles_l614_614489

theorem number_of_rectangles (H V : ℕ) (hH : H = 5) (hV : V = 6) :
  (nat.choose 5 2) * (nat.choose 6 2) = 150 :=
by
  rw [hH, hV]
  norm_num
  sorry

end number_of_rectangles_l614_614489


namespace maximize_average_profit_l614_614406

def y1 (x : ℝ) : ℝ := 150 - (3 / 2) * x
def y2 (x : ℝ) : ℝ := 600 + 72 * x
def y_total_profit (x : ℝ) : ℝ := x * (150 - (3 / 2) * x) - (600 + 72 * x)
def y_average_profit (x : ℝ) : ℝ := (- (3 / 2) * x) - (600 / x) + 78

theorem maximize_average_profit (x : ℝ) (y : ℝ) 
  (h1 : y1 x = 150 - (3 / 2) * x) 
  (h2 : y >= 90) 
  (h3 : y2 x = 600 + 72 * x)
  (fx : 0 < x ∧ x ≤ 40) :
  x = 20 ∧ y_average_profit x = 18 :=
by
  sorry

end maximize_average_profit_l614_614406


namespace solve_for_x_l614_614697

theorem solve_for_x (x : ℝ) (h1 : 1 - x^2 = 0) (h2 : x ≠ 1) : x = -1 := 
by 
  sorry

end solve_for_x_l614_614697


namespace unique_abc_solution_l614_614151

theorem unique_abc_solution (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : ab + bc + ca = abc) : 
  a = 2 ∧ b = 3 ∧ c = 6 := 
by
  sorry

end unique_abc_solution_l614_614151


namespace hexagon_area_inequality_l614_614679

-- Define the triangle ABC and point O inside it
variables {A B C O D E F G H I : Point}
-- Define the lines parallel to the sides of the triangle
variables (h1 : parallel O D E B C) (h2 : parallel O F G C A) (h3 : parallel O H I A B)

-- Define the area of the hexagon and the triangle
noncomputable def area_hex (D G H E D F : Point) : ℝ := sorry
noncomputable def area_triangle (A B C : Point) : ℝ := sorry

-- The theorem to be proved
theorem hexagon_area_inequality (hO_inside : inside_triangle O A B C)
                                (hD_on_side : on_side D A B)
                                (hE_on_side : on_side E B C)
                                (hF_on_side : on_side F C A)
                                (hG_on_side : on_side G C A)
                                (hH_on_side : on_side H A B)
                                (hI_on_side : on_side I B C) : 
    area_hex D G H E D F ≥ (2 / 3) * area_triangle A B C := sorry

end hexagon_area_inequality_l614_614679


namespace find_coordinates_of_c_find_cosine_of_theta_l614_614887

variable {R : Type*} [IsROrC R]

-- Definitions and conditions for the first part (coordinates of c)
def vector_a : EuclideanSpace R (Fin 2) := ![1, 2]
def vector_c (λ : R) : EuclideanSpace R (Fin 2) := λ • vector_a
axiom length_c : |vector_c R| = 3 * Real.sqrt 5
axiom parallel_a_c : ∃ λ : R, vector_c λ = vector_c R

-- Definitions and conditions for the second part (cosine of the angle theta)
def vector_b (u v : R) : EuclideanSpace R (Fin 2) := ![u, v]
axiom length_b : |vector_b u v| = 3 * Real.sqrt 5
axiom perpendicular_condition : (4 • vector_a - vector_b u v) ⬝ (2 • vector_a + vector_b u v) = 0

-- Statement for finding the coordinates of vector_c
theorem find_coordinates_of_c :
  ∃ λ : R, vector_c λ = ![3, 6] ∨ vector_c λ = ![-3, -6] := 
sorry

-- Statement for finding the cosine of the angle theta
theorem find_cosine_of_theta : 
  ∃ u v : R, (vector_b u v).dot vector_a = 5 / 2 ∧ 
  Real.cos (angle vector_a (vector_b u v)) = 1 / 6 := 
sorry

end find_coordinates_of_c_find_cosine_of_theta_l614_614887


namespace af_length_is_one_l614_614885

-- Define the scenario based on given conditions
variables {p x0 : ℝ} (hp : p > 0) (hx0 : x0 > p / 2)

-- Define the parabola y^2 = 2px
def parabola (y x : ℝ) : Prop := y^2 = 2 * p * x

-- Define point M on the parabola
def on_parabola_M (y : ℝ) : Prop := parabola y x0 ∧ y = 2 * sqrt 2

-- Define the vector relations
def MF_length := x0 + p / 2
def MA_length := 2 * (x0 - p / 2)
def AF_length := MF_length / 3

-- Define specific constraints based on problem
def chord_length : Prop := (x0 = p) ∧ (MA_length / AF_length = 2)

-- Main proof statement to show |AF| = 1
theorem af_length_is_one (hp : p > 0) (hx0 : x0 > p / 2) 
    (p_val: p = 2) (x0_val: x0 = p) (h0 : parabola 4 x0) 
    (h1 : on_parabola_M 2*sqrt 2) (h2 : chord_length) : 
    AF_length = 1 :=
by
  sorry

end af_length_is_one_l614_614885


namespace sum_first_15_odd_integers_from_5_l614_614001

theorem sum_first_15_odd_integers_from_5 :
  let a := 5
  let n := 15
  let d := 2
  let last_term := a + (n - 1) * d
  let S := n * a + (n * (n - 1) * d) / 2
  last_term = 37 ∧ S = 315 := by
  sorry

end sum_first_15_odd_integers_from_5_l614_614001


namespace count_special_integers_l614_614903

theorem count_special_integers : 
  ∃ (n : ℕ), ∀ k < 500, 
  (∃ m, k = 7 * m ∧ k % 7 = 0 ∧ (k < 500 ∧ k = 7 * (m % 10 + (m / 10) % 10 + (m / 100) % 10))) → n = 6 :=
begin
  sorry
end

end count_special_integers_l614_614903


namespace meetings_percent_l614_614593

/-- Define the lengths of the meetings and total workday in minutes -/
def first_meeting : ℕ := 40
def second_meeting : ℕ := 80
def second_meeting_overlap : ℕ := 10
def third_meeting : ℕ := 30
def workday_minutes : ℕ := 8 * 60

/-- Define the effective duration of the second meeting -/
def effective_second_meeting : ℕ := second_meeting - second_meeting_overlap

/-- Define the total time spent in meetings -/
def total_meeting_time : ℕ := first_meeting + effective_second_meeting + third_meeting

/-- Define the percentage of the workday spent in meetings -/
noncomputable def percent_meeting_time : ℚ := (total_meeting_time * 100 : ℕ) / workday_minutes

/-- Theorem: Given Laura's workday and meeting durations, prove that the percent of her workday spent in meetings is approximately 29.17%. -/
theorem meetings_percent {epsilon : ℚ} (h : epsilon = 0.01) : abs (percent_meeting_time - 29.17) < epsilon :=
sorry

end meetings_percent_l614_614593


namespace angle_comparison_l614_614510

-- Definitions of the setup and given conditions
variables {A B C M : Type*}
variables [triangleABC : triangle A B C]
variables (H1 : dist A B < dist B C)
variables (M_on_median : is_median_point B A C M)

-- Statement to be proved
theorem angle_comparison : ∠ B A M > ∠ B C M :=
sorry

end angle_comparison_l614_614510


namespace range_of_a_l614_614166

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := if x ≤ 1 then (a - 3) * x - 3 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 3 < a ∧ a ≤ 6 :=
by
  sorry

end range_of_a_l614_614166


namespace rectangle_ratio_height_base_l614_614571

theorem rectangle_ratio_height_base
  (side_length : ℝ)
  (E F : ℝ × ℝ)
  (midpoints : E = (side_length / 2, side_length) ∧ F = (side_length / 2, 0))
  (AG_perpendicular_BF : ∀ A G, (A.1 = 0 ∧ A.2 = side_length / 2) ∧ (G.1 = 3 * (√5 / 2)) → A.2 = G.2)
  (rearranged_rectangle : ∀ XY YZ, (side_length ^ 2 = XY * YZ)) :
  XY / YZ = 5 :=
by
  -- Proof goes here
  sorry

end rectangle_ratio_height_base_l614_614571


namespace iterative_avg_difference_l614_614096

theorem iterative_avg_difference :
  let seq := [3, 5, 7, 9, 11] in
  let iterative_avg ls := (ls.inits.drop(1).map List.sum).reverse.map (fun x => x / ls.inits.length) in
  let avg_max := (iterative_avg [11, 9, 7, 5, 3]).reverse.head in
  let avg_min := (iterative_avg [3, 5, 7, 9, 11]).reverse.head in
  avg_max - avg_min = 4.25 :=
by 
  sorry

end iterative_avg_difference_l614_614096


namespace simplify_expression_l614_614373

theorem simplify_expression (b : ℝ) (hb : b = -1) : 
  (3 * b⁻¹ + (2 * b⁻¹) / 3) / b = 11 / 3 :=
by
  sorry

end simplify_expression_l614_614373


namespace largest_prime_factor_9801_l614_614476

theorem largest_prime_factor_9801 : ∃ p : ℕ, Prime p ∧ p ∣ 9801 ∧ ∀ q : ℕ, Prime q ∧ q ∣ 9801 → q ≤ p :=
sorry

end largest_prime_factor_9801_l614_614476


namespace sum_first_2014_mod_2012_l614_614106

theorem sum_first_2014_mod_2012 : 
  (∑ i in Finset.range 2015, i) % 2012 = 1009 :=
by
  sorry

end sum_first_2014_mod_2012_l614_614106


namespace find_term_number_l614_614667

noncomputable def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem find_term_number
  (a₁ : ℤ)
  (d : ℤ)
  (n : ℕ)
  (h₀ : a₁ = 1)
  (h₁ : d = 3)
  (h₂ : arithmetic_sequence a₁ d n = 2011) :
  n = 671 :=
  sorry

end find_term_number_l614_614667


namespace part1_part2_l614_614845

variable (x y : ℤ) (A B : ℤ)

def A_def : ℤ := 3 * x^2 - 5 * x * y - 2 * y^2
def B_def : ℤ := x^2 - 3 * y

theorem part1 : A_def x y - 2 * B_def x y = x^2 - 5 * x * y - 2 * y^2 + 6 * y := by
  sorry

theorem part2 : A_def 2 (-1) - 2 * B_def 2 (-1) = 6 := by
  sorry

end part1_part2_l614_614845


namespace never_2016_l614_614969

-- Define the sequence
def sequence (n : ℕ) : ℕ :=
  if n = 0 then 2 else
  if n = 1 then 0 else
  if n = 2 then 1 else
  if n = 3 then 7 else
  if n = 4 then 0 else
  (sequence (n - 1) + sequence (n - 2) + sequence (n - 3) + sequence (n - 4)) % 10

-- Prove that 2016 will never appear from the 5th digit onwards
theorem never_2016 (n: ℕ) : 
  (sequence n = 2 ∧ sequence (n+1) = 0 ∧ sequence (n+2) = 1 ∧ sequence (n+3) = 6) → false := 
sorry

end never_2016_l614_614969


namespace garden_area_l614_614048

theorem garden_area (length width : ℕ) (h_length : length = 175) (h_width : width = 12) : length * width = 2100 := by
  rw [h_length, h_width]
  norm_num
  sorry

end garden_area_l614_614048


namespace tan_theta_value_l614_614518

theorem tan_theta_value
  (θ : ℝ)
  (h1 : cos (θ / 2) = 4 / 5)
  (h2 : sin θ < 0) :
  tan θ = -24 / 7 :=
sorry

end tan_theta_value_l614_614518


namespace max_sum_first_n_terms_formula_sum_terms_abs_l614_614511

theorem max_sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  a 1 = 29 ∧ S 10 = S 20 →
  ∃ (n : ℕ), n = 15 ∧ S 15 = 225 := by
  sorry

theorem formula_sum_terms_abs (a : ℕ → ℤ) (S T : ℕ → ℤ) :
  a 1 = 29 ∧ S 10 = S 20 →
  (∀ n, n ≤ 15 → T n = 30 * n - n * n) ∧
  (∀ n, n ≥ 16 → T n = n * n - 30 * n + 450) := by
  sorry

end max_sum_first_n_terms_formula_sum_terms_abs_l614_614511


namespace concurrency_NF_DE_GM_l614_614092

-- Assumptions and Definitions
variable {A B C D E F G M N : Point} -- Define points
variable (triangle_ABC : Triangle A B C) -- A triangle
variable (acute : IsAcute triangle_ABC) -- Acute-angled triangle

variable (N_mid_AB : Midpoint N A B) -- N is the midpoint of segment AB
variable (M_mid_BC : Midpoint M B C) -- M is the midpoint of segment BC
variable (circAB : Circle) (diam_AB : circAB.diameter = A-B) -- Circle with diameter AB

variable (D_on_BC : lies_on D BC) (E_on_AM : lies_on E (line_through A M)) (F_on_AC : lies_on F AC) -- D on BC, E on AM, F on AC
variable (G_mid_FC : Midpoint G F C) -- G is the midpoint of FC

-- Theorem to prove concurrency
theorem concurrency_NF_DE_GM :
  Concurrent (line_through N F) (line_through D E) (line_through G M) := sorry

end concurrency_NF_DE_GM_l614_614092


namespace solution_set_of_inequality_l614_614025

theorem solution_set_of_inequality (x : ℝ) : ||x-2|-1| ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 :=
sorry

end solution_set_of_inequality_l614_614025


namespace vinegar_mixture_concentration_l614_614379

theorem vinegar_mixture_concentration :
  let c1 := 5 / 100
  let c2 := 10 / 100
  let v1 := 10
  let v2 := 10
  (v1 * c1 + v2 * c2) / (v1 + v2) = 7.5 / 100 :=
by
  sorry

end vinegar_mixture_concentration_l614_614379


namespace arc_length_is_5pi_l614_614308

-- Define the central angle and radius
def θ: ℝ := 150  -- angle in degrees
def r: ℝ := 6    -- radius in centimeters

-- Define the formula for the arc length
def arc_length (θ r : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

-- Theorem statement
theorem arc_length_is_5pi : arc_length θ r = 5 * Real.pi := by
  sorry

end arc_length_is_5pi_l614_614308


namespace expression_equivalence_l614_614094

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 :=
by sorry

end expression_equivalence_l614_614094


namespace number_of_rectangles_l614_614490

theorem number_of_rectangles (h_lines v_lines : ℕ) (h_lines_eq : h_lines = 5) (v_lines_eq : v_lines = 6) :
  (h_lines.choose 2) * (v_lines.choose 2) = 150 := by
  rw [h_lines_eq, v_lines_eq]
  norm_num
  exact dec_trivial
  sorry

end number_of_rectangles_l614_614490


namespace proof_radius_of_larger_circle_proof_area_of_shaded_region_l614_614211

noncomputable def radius_of_larger_circle (R : ℝ) :=
  let OP := 40   -- radius of the smaller circle
  let AB := 100  -- chord length
  let AP := AB / 2
  sqrt (AP^2 + OP^2)

theorem proof_radius_of_larger_circle :
  radius_of_larger_circle (sqrt 4100) = sqrt 4100 :=
by
  sorry

noncomputable def area_of_shaded_region (R : ℝ) :=
  let inner_radius := 40  -- radius of the smaller circle
  let outer_radius := radius_of_larger_circle (sqrt 4100)
  π * outer_radius^2 - π * inner_radius^2

theorem proof_area_of_shaded_region :
  area_of_shaded_region (sqrt 4100) = 2500 * π :=
by
  sorry

end proof_radius_of_larger_circle_proof_area_of_shaded_region_l614_614211


namespace num_sets_B_l614_614273

open Set

def A : Set ℕ := {1, 3}

theorem num_sets_B :
  ∃ (B : ℕ → Set ℕ), (∀ b, B b ∪ A = {1, 3, 5}) ∧ (∃ s t u v, B s = {5} ∧
                                                   B t = {1, 5} ∧
                                                   B u = {3, 5} ∧
                                                   B v = {1, 3, 5} ∧ 
                                                   s ≠ t ∧ s ≠ u ∧ s ≠ v ∧
                                                   t ≠ u ∧ t ≠ v ∧
                                                   u ≠ v) :=
sorry

end num_sets_B_l614_614273


namespace solution_l614_614730

-- Define the linear equations and their solutions
def system_of_equations (x y : ℕ) :=
  3 * x + y = 500 ∧ x + 2 * y = 250

-- Define the budget constraint
def budget_constraint (m : ℕ) :=
  150 * m + 50 * (25 - m) ≤ 2700

-- Define the purchasing plans and costs
def purchasing_plans (m n : ℕ) :=
  (m = 12 ∧ n = 13 ∧ 150 * m + 50 * n = 2450) ∨ 
  (m = 13 ∧ n = 12 ∧ 150 * m + 50 * n = 2550) ∨ 
  (m = 14 ∧ n = 11 ∧ 150 * m + 50 * n = 2650)

-- Define the Lean statement
theorem solution :
  (∃ x y, system_of_equations x y ∧ x = 150 ∧ y = 50) ∧
  (∃ m, budget_constraint m ∧ m ≤ 14) ∧
  (∃ m n, 12 ≤ m ∧ m ≤ 14 ∧ m + n = 25 ∧ purchasing_plans m n ∧ 150 * m + 50 * n = 2450) :=
sorry

end solution_l614_614730


namespace CE_perpendicular_BM_l614_614392

theorem CE_perpendicular_BM
  (a b : ℝ)
  (B C M D E A : ℝ)
  (perpendicular_from_M_on_BD_intersects_AD_at_E : Bool)
  (BC_eq_a : BC = a)
  (CM_eq_b : CM = b)
  (MD_eq_b : MD = b)
  (perpendicular_from_C_on_BM_intersects_AD_at_E1 : Bool)
  (perpendicular_from_M_on_BD_intersects_AD_at_E2 : Bool)
  (E1_eq_E2 : E1 = E2) :
  CE ⊥ BM := 
begin
  sorry
end

end CE_perpendicular_BM_l614_614392


namespace number_of_children_l614_614232

-- Define the given conditions in Lean 4
variable {a : ℕ}
variable {R : ℕ}
variable {L : ℕ}
variable {k : ℕ}

-- Conditions given in the problem
def condition1 : 200 ≤ a ∧ a ≤ 300 := sorry
def condition2 : a = 25 * R + 10 := sorry
def condition3 : a = 30 * L - 15 := sorry 
def condition4 : a + 15 = 150 * k := sorry

-- The theorem to prove
theorem number_of_children : a = 285 :=
by
  assume a R L k // This assumption is for the variables needed.
  have h₁ : condition1 := sorry
  have h₂ : condition2 := sorry
  have h₃ : condition3 := sorry
  have h₄ : condition4 := sorry 
  exact sorry

end number_of_children_l614_614232


namespace intercepts_sum_l614_614411

theorem intercepts_sum :
  let line := λ x y : ℝ, y - 2 = -3 * (x + 5)
  let x_intercept := (∃ x, line x 0 ∧ x = -13/3) in
  let y_intercept := (∃ y, line 0 y ∧ y = -13) in
  ∃ (sum : ℝ), x_intercept ∧ y_intercept ∧ sum = -52/3 := by
  sorry

end intercepts_sum_l614_614411


namespace point_trajectory_is_plane_l614_614867

theorem point_trajectory_is_plane {P : ℝ × ℝ × ℝ} (h : P.2 = 0) : 
  ∃ (a b c : ℝ), c ≠ 0 ∧ ∀ (x y : ℝ), (x, y, 0) = P :=
begin
  sorry
end

end point_trajectory_is_plane_l614_614867


namespace smallest_positive_period_2pi_max_value_a_minus_1_max_value_interval_a_minus_1_l614_614535

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Math.sin (x + (Real.pi / 6)) + Math.sin (x - (Real.pi / 6)) + Math.cos x + a

theorem smallest_positive_period_2pi (a : ℝ) :
  ∃ T : ℝ, 0 < T ∧ (∀ x : ℝ, f a (x + T) = f a x) ∧ 
  (∀ T' : ℝ, (∀ x : ℝ, f a (x + T') = f a x) → T' ≥ T) :=
sorry

theorem max_value_a_minus_1 (a : ℝ) :
  (∀ x : ℝ, f a x ≤ 1) → a = -1 :=
sorry

theorem max_value_interval_a_minus_1 (a : ℝ) :
  (∀ x : ℝ, -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2 → f a x ≤ 1) → a = -1 :=
sorry

end smallest_positive_period_2pi_max_value_a_minus_1_max_value_interval_a_minus_1_l614_614535


namespace first_worker_load_time_approx_l614_614431

theorem first_worker_load_time_approx :
  ∃ T : ℝ, A : ℝ, (A = 2.727272727272727 ∧ ∀ R1 R2 : ℝ, R1 = 1 / T ∧ R2 = 1 / 5 → R1 + R2 = 1 / A) →
  abs (T - 6) < 0.0001 :=
by
  sorry

end first_worker_load_time_approx_l614_614431


namespace maximum_value_inequality_l614_614724

theorem maximum_value_inequality (a b c : ℝ) (h : a^2 + b^2 + c^2 = 4) : 3 * a + 4 * b + 5 * c ≤ 10 * real.sqrt 2 :=
sorry

end maximum_value_inequality_l614_614724


namespace triangle_ABC_AB_length_l614_614852

noncomputable def length_AB (AC BC angle : ℝ) : ℝ :=
  let cos_angle := Real.cos angle
  let term1 := BC * BC
  let term2 := AC * AC
  let AB2 := term1 - term2 + 2 * AC * cos_angle
  Real.sqrt AB2

theorem triangle_ABC_AB_length :
  ∀ (AB AC BC : ℝ) (angle : ℝ),
    AC = 4 → BC = 2 * Real.sqrt 7 → angle = Real.pi / 3 → length_AB AC BC angle = 6 :=
begin
  intros AB AC BC angle hAC hBC hAngle,
  rw [hAC, hBC, hAngle],
  sorry
end

end triangle_ABC_AB_length_l614_614852
