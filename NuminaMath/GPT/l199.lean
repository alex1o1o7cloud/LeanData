import Mathlib

namespace percent_of_a_is_b_percent_of_d_is_c_percent_of_d_is_e_l199_19962

variables (a b c d e : ℝ)

-- Conditions
def condition1 : Prop := c = 0.25 * a
def condition2 : Prop := c = 0.50 * b
def condition3 : Prop := d = 0.40 * a
def condition4 : Prop := d = 0.20 * b
def condition5 : Prop := e = 0.35 * d
def condition6 : Prop := e = 0.15 * c

-- Proof Problem Statements
theorem percent_of_a_is_b (h1 : condition1 a c) (h2 : condition2 c b) : b = 0.5 * a := sorry

theorem percent_of_d_is_c (h1 : condition1 a c) (h3 : condition3 a d) : c = 0.625 * d := sorry

theorem percent_of_d_is_e (h5 : condition5 e d) : e = 0.35 * d := sorry

end percent_of_a_is_b_percent_of_d_is_c_percent_of_d_is_e_l199_19962


namespace sum_remainder_mod_9_l199_19953

theorem sum_remainder_mod_9 : 
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 9 = 6 :=
by
  sorry

end sum_remainder_mod_9_l199_19953


namespace race_distance_l199_19954

-- Definitions for the conditions
def A_time : ℕ := 20
def B_time : ℕ := 25
def A_beats_B_by : ℕ := 14

-- Definition of the function to calculate whether the total distance D is correct
def total_distance : ℕ := 56

-- The theorem statement without proof
theorem race_distance (D : ℕ) (A_time B_time A_beats_B_by : ℕ)
  (hA : A_time = 20)
  (hB : B_time = 25)
  (hAB : A_beats_B_by = 14)
  (h_eq : (D / A_time) * B_time = D + A_beats_B_by) : 
  D = total_distance :=
sorry

end race_distance_l199_19954


namespace samara_oil_spent_l199_19936

theorem samara_oil_spent (O : ℕ) (A_total : ℕ) (S_tires : ℕ) (S_detailing : ℕ) (diff : ℕ) (S_total : ℕ) :
  A_total = 2457 →
  S_tires = 467 →
  S_detailing = 79 →
  diff = 1886 →
  S_total = O + S_tires + S_detailing →
  A_total = S_total + diff →
  O = 25 :=
by
  sorry

end samara_oil_spent_l199_19936


namespace escher_probability_l199_19979

def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

def favorable_arrangements (total_art : ℕ) (escher_prints : ℕ) : ℕ :=
  num_arrangements (total_art - escher_prints + 1) * num_arrangements escher_prints

def total_arrangements (total_art : ℕ) : ℕ :=
  num_arrangements total_art

def prob_all_escher_consecutive (total_art : ℕ) (escher_prints : ℕ) : ℚ :=
  favorable_arrangements total_art escher_prints / total_arrangements total_art

theorem escher_probability :
  prob_all_escher_consecutive 12 4 = 1/55 :=
by
  sorry

end escher_probability_l199_19979


namespace algebraic_expression_value_l199_19931

theorem algebraic_expression_value (m: ℝ) (h: m^2 + m - 1 = 0) : 2023 - m^2 - m = 2022 := 
by 
  sorry

end algebraic_expression_value_l199_19931


namespace solve_for_k_l199_19960

theorem solve_for_k (k : ℝ) (h₁ : ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h₂ : k ≠ 0) : k = 6 :=
sorry

end solve_for_k_l199_19960


namespace value_of_a_l199_19950

theorem value_of_a 
  (a : ℝ) 
  (h : 0.005 * a = 0.85) : 
  a = 170 :=
sorry

end value_of_a_l199_19950


namespace day_crew_fraction_l199_19940

theorem day_crew_fraction (D W : ℕ) (h1 : ∀ n, n = D / 4) (h2 : ∀ w, w = 4 * W / 5) :
  (D * W) / ((D * W) + ((D / 4) * (4 * W / 5))) = 5 / 6 :=
by 
  sorry

end day_crew_fraction_l199_19940


namespace inequalities_not_simultaneous_l199_19921

theorem inequalities_not_simultaneous (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (ineq1 : a + b < c + d) (ineq2 : (a + b) * (c + d) < a * b + c * d) (ineq3 : (a + b) * c * d < (c + d) * a * b) :
  false := 
sorry

end inequalities_not_simultaneous_l199_19921


namespace four_fold_application_of_f_l199_19902

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 then
    x / 3
  else
    5 * x + 2

theorem four_fold_application_of_f : f (f (f (f 3))) = 187 := 
  by
    sorry

end four_fold_application_of_f_l199_19902


namespace total_bricks_used_l199_19995

def numberOfCoursesPerWall := 6
def bricksPerCourse := 10
def numberOfWalls := 4
def incompleteCourses := 2

theorem total_bricks_used :
  (numberOfCoursesPerWall * bricksPerCourse * (numberOfWalls - 1)) + ((numberOfCoursesPerWall - incompleteCourses) * bricksPerCourse) = 220 :=
by
  -- Proof goes here
  sorry

end total_bricks_used_l199_19995


namespace original_number_is_0_02_l199_19964

theorem original_number_is_0_02 (x : ℝ) (h : 10000 * x = 4 / x) : x = 0.02 :=
by
  sorry

end original_number_is_0_02_l199_19964


namespace edge_c_eq_3_or_5_l199_19942

noncomputable def a := 7
noncomputable def b := 8
noncomputable def A := Real.pi / 3

theorem edge_c_eq_3_or_5 (c : ℝ) (h : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) : c = 3 ∨ c = 5 :=
by
  sorry

end edge_c_eq_3_or_5_l199_19942


namespace frequency_of_sixth_group_l199_19914

theorem frequency_of_sixth_group :
  ∀ (total_data_points : ℕ)
    (freq1 freq2 freq3 freq4 : ℕ)
    (freq5_ratio : ℝ),
    total_data_points = 40 →
    freq1 = 10 →
    freq2 = 5 →
    freq3 = 7 →
    freq4 = 6 →
    freq5_ratio = 0.10 →
    (total_data_points - (freq1 + freq2 + freq3 + freq4) - (total_data_points * freq5_ratio)) = 8 :=
by
  sorry

end frequency_of_sixth_group_l199_19914


namespace consecutive_probability_l199_19945

-- Define the total number of ways to choose 2 episodes out of 6
def total_combinations : ℕ := Nat.choose 6 2

-- Define the number of ways to choose consecutive episodes
def consecutive_combinations : ℕ := 5

-- Define the probability of choosing consecutive episodes
def probability_of_consecutive : ℚ := consecutive_combinations / total_combinations

-- Theorem stating that the calculated probability should equal 1/3
theorem consecutive_probability : probability_of_consecutive = 1 / 3 :=
by
  sorry

end consecutive_probability_l199_19945


namespace initial_money_l199_19971

/-- Given the following conditions:
  (1) June buys 4 maths books at $20 each.
  (2) June buys 6 more science books than maths books at $10 each.
  (3) June buys twice as many art books as maths books at $20 each.
  (4) June spends $160 on music books.
  Prove that June had initially $500 for buying school supplies. -/
theorem initial_money (maths_books : ℕ) (science_books : ℕ) (art_books : ℕ) (music_books_cost : ℕ)
  (h_math_books : maths_books = 4) (price_per_math_book : ℕ) (price_per_science_book : ℕ) 
  (price_per_art_book : ℕ) (price_per_music_books_cost : ℕ) (h_maths_price : price_per_math_book = 20)
  (h_science_books : science_books = maths_books + 6) (h_science_price : price_per_science_book = 10)
  (h_art_books : art_books = 2 * maths_books) (h_art_price : price_per_art_book = 20)
  (h_music_books_cost : music_books_cost = 160) :
  4 * 20 + (4 + 6) * 10 + (2 * 4) * 20 + 160 = 500 :=
by sorry

end initial_money_l199_19971


namespace sum_of_digits_7_pow_11_l199_19941

/-
  Let's define the problem with the given conditions and expected answer.
  We need to prove that the sum of the tens digit and the ones digit of 7^11 is 7.
-/

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_7_pow_11 : tens_digit (7 ^ 11) + ones_digit (7 ^ 11) = 7 :=
sorry

end sum_of_digits_7_pow_11_l199_19941


namespace f_decreasing_ln_inequality_limit_inequality_l199_19919

-- Definitions of the given conditions
noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) / x

-- Statements we need to prove

-- (I) Prove that f(x) is decreasing on (0, +∞)
theorem f_decreasing : ∀ x y : ℝ, 0 < x → x < y → f y < f x := sorry

-- (II) Prove that for the inequality ln(1 + x) < ax to hold for all x in (0, +∞), a must be at least 1
theorem ln_inequality (a : ℝ) : (∀ x : ℝ, 0 < x → Real.log (1 + x) < a * x) ↔ 1 ≤ a := sorry

-- (III) Prove that (1 + 1/n)^n < e for all n in ℕ*
theorem limit_inequality (n : ℕ) (h : n ≠ 0) : (1 + 1 / n) ^ n < Real.exp 1 := sorry

end f_decreasing_ln_inequality_limit_inequality_l199_19919


namespace total_students_experimental_primary_school_l199_19987

theorem total_students_experimental_primary_school : 
  ∃ (n : ℕ), 
  n = (21 + 11) * 28 ∧ 
  n = 896 := 
by {
  -- Since the proof is not required, we use "sorry"
  sorry
}

end total_students_experimental_primary_school_l199_19987


namespace age_difference_between_two_children_l199_19999

theorem age_difference_between_two_children 
  (avg_age_10_years_ago : ℕ)
  (present_avg_age : ℕ)
  (youngest_child_present_age : ℕ)
  (initial_family_members : ℕ)
  (current_family_members : ℕ)
  (H1 : avg_age_10_years_ago = 24)
  (H2 : present_avg_age = 24)
  (H3 : youngest_child_present_age = 3)
  (H4 : initial_family_members = 4)
  (H5 : current_family_members = 6) :
  ∃ (D: ℕ), D = 2 :=
by
  sorry

end age_difference_between_two_children_l199_19999


namespace remainder_when_divided_by_7_l199_19935

theorem remainder_when_divided_by_7 (n : ℕ) (h : (2 * n) % 7 = 4) : n % 7 = 2 :=
  by sorry

end remainder_when_divided_by_7_l199_19935


namespace yellow_balls_count_l199_19906

theorem yellow_balls_count (R B G Y : ℕ) 
  (h1 : R = 2 * B) 
  (h2 : B = 2 * G) 
  (h3 : Y > 7) 
  (h4 : R + B + G + Y = 27) : 
  Y = 20 := by
  sorry

end yellow_balls_count_l199_19906


namespace automobile_travel_distance_l199_19944

theorem automobile_travel_distance
  (a r : ℝ) : 
  let feet_per_yard := 3
  let seconds_per_minute := 60
  let travel_feet := a / 4
  let travel_seconds := 2 * r
  let rate_yards_per_second := (travel_feet / travel_seconds) / feet_per_yard
  let total_seconds := 10 * seconds_per_minute
  let total_yards := rate_yards_per_second * total_seconds
  total_yards = 25 * a / r := by
  sorry

end automobile_travel_distance_l199_19944


namespace train_speed_l199_19998

-- Define the conditions in terms of distance and time
def train_length : ℕ := 160
def crossing_time : ℕ := 8

-- Define the expected speed
def expected_speed : ℕ := 20

-- The theorem stating the speed of the train given the conditions
theorem train_speed : (train_length / crossing_time) = expected_speed :=
by
  -- Note: The proof is omitted
  sorry

end train_speed_l199_19998


namespace xy_plus_one_is_perfect_square_l199_19958

theorem xy_plus_one_is_perfect_square (x y : ℕ) (h : 1 / (x : ℝ) + 1 / (y : ℝ) = 1 / (x + 2 : ℝ) + 1 / (y - 2 : ℝ)) :
  ∃ k : ℕ, xy + 1 = k^2 :=
by
  sorry

end xy_plus_one_is_perfect_square_l199_19958


namespace problem_solution_l199_19904

noncomputable def proof_problem : Prop :=
∀ x y : ℝ, y = (x + 1)^2 ∧ (x * y^2 + y = 1) → false

theorem problem_solution : proof_problem :=
by
  sorry

end problem_solution_l199_19904


namespace hexagon_perimeter_l199_19956

theorem hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) (h1 : side_length = 5) (h2 : num_sides = 6) : 
  num_sides * side_length = 30 := by
  sorry

end hexagon_perimeter_l199_19956


namespace frustum_slant_height_l199_19974

-- The setup: we are given specific conditions for a frustum resulting from cutting a cone
variable {r : ℝ} -- represents the radius of the upper base of the frustum
variable {h : ℝ} -- represents the slant height of the frustum
variable {h_removed : ℝ} -- represents the slant height of the removed cone

-- The given conditions
def upper_base_radius : ℝ := r
def lower_base_radius : ℝ := 4 * r
def slant_height_removed_cone : ℝ := 3

-- The proportion derived from similar triangles
def proportion (h r : ℝ) := (h / (4 * r)) = ((h + 3) / (5 * r))

-- The main statement: proving the slant height of the frustum is 9 cm
theorem frustum_slant_height (r : ℝ) (h : ℝ) (hr : proportion h r) : h = 9 :=
sorry

end frustum_slant_height_l199_19974


namespace max_value_m_l199_19908

theorem max_value_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2 * x - 8 > 0) -> (x < m)) -> m = -2 :=
by
  sorry

end max_value_m_l199_19908


namespace upper_bound_for_k_squared_l199_19963

theorem upper_bound_for_k_squared :
  (∃ (k : ℤ), k^2 > 121 ∧ ∀ m : ℤ, (m^2 > 121 ∧ m^2 < 323 → m = k + 1)) →
  (k ≤ 17) → (18^2 > 323) := 
by 
  sorry

end upper_bound_for_k_squared_l199_19963


namespace number_divisors_l199_19949

theorem number_divisors (p : ℕ) (h : p = 2^56 - 1) : ∃ x y : ℕ, 95 ≤ x ∧ x ≤ 105 ∧ 95 ≤ y ∧ y ≤ 105 ∧ p % x = 0 ∧ p % y = 0 ∧ x = 101 ∧ y = 127 :=
by {
  sorry
}

end number_divisors_l199_19949


namespace icosahedron_path_count_l199_19951

-- Definitions from the conditions
def vertices := 12
def edges := 30
def top_adjacent := 5
def bottom_adjacent := 5

-- Define the total paths calculation based on the given structural conditions
theorem icosahedron_path_count (v e ta ba : ℕ) (hv : v = 12) (he : e = 30) (hta : ta = 5) (hba : ba = 5) : 
  (ta * (ta - 1) * (ba - 1)) * 2 = 810 :=
by
-- Insert calculation logic here if needed or detailed structure definitions
  sorry

end icosahedron_path_count_l199_19951


namespace geometric_seq_arith_condition_half_l199_19977

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions from the problem
def geometric_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def positive_terms (a : ℕ → ℝ) := ∀ n, a n > 0
def arithmetic_condition (a : ℕ → ℝ) (q : ℝ) := 
  a 1 = q * a 0 ∧ (1/2 : ℝ) * a 2 = a 1 + 2 * a 0

-- The statement to be proven
theorem geometric_seq_arith_condition_half (a : ℕ → ℝ) (q : ℝ) :
  geometric_seq a q →
  positive_terms a →
  arithmetic_condition a q →
  q = 2 →
  (a 2 + a 3) / (a 3 + a 4) = 1 / 2 :=
by
  intros h1 h2 h3 hq
  sorry

end geometric_seq_arith_condition_half_l199_19977


namespace min_Sn_value_l199_19967

noncomputable def a (n : ℕ) (d : ℤ) : ℤ := -11 + (n - 1) * d

def Sn (n : ℕ) (d : ℤ) : ℤ := n * -11 + n * (n - 1) * d / 2

theorem min_Sn_value {d : ℤ} (h5_6 : a 5 d + a 6 d = -4) : 
  ∃ n, Sn n d = (n - 6)^2 - 36 ∧ n = 6 :=
by
  sorry

end min_Sn_value_l199_19967


namespace evening_temperature_l199_19922

-- Definitions based on conditions
def noon_temperature : ℤ := 2
def temperature_drop : ℤ := 3

-- The theorem statement
theorem evening_temperature : noon_temperature - temperature_drop = -1 := 
by
  -- The proof is omitted
  sorry

end evening_temperature_l199_19922


namespace yesterday_tomorrow_is_friday_l199_19912

-- Defining the days of the week
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to go to the next day
def next_day : Day → Day
| Sunday    => Monday
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday

-- Function to go to the previous day
def previous_day : Day → Day
| Sunday    => Saturday
| Monday    => Sunday
| Tuesday   => Monday
| Wednesday => Tuesday
| Thursday  => Wednesday
| Friday    => Thursday
| Saturday  => Friday

-- Proving the statement
theorem yesterday_tomorrow_is_friday (T : Day) (H : next_day (previous_day T) = Thursday) : previous_day (next_day (next_day T)) = Friday :=
by
  sorry

end yesterday_tomorrow_is_friday_l199_19912


namespace gibraltar_initial_population_stable_l199_19926
-- Import necessary libraries

-- Define constants based on conditions
def full_capacity := 300 * 4
def initial_population := (full_capacity / 3) - 100
def population := 300 -- This is the final answer we need to validate

-- The main theorem to prove
theorem gibraltar_initial_population_stable : initial_population = population :=
by 
  -- Proof is skipped as requested
  sorry

end gibraltar_initial_population_stable_l199_19926


namespace least_multiple_greater_than_500_l199_19920

theorem least_multiple_greater_than_500 : ∃ n : ℕ, n > 0 ∧ 35 * n > 500 ∧ 35 * n = 525 :=
by
  sorry

end least_multiple_greater_than_500_l199_19920


namespace square_of_binomial_b_value_l199_19938

theorem square_of_binomial_b_value (b : ℤ) (h : ∃ c : ℤ, 16 * (x : ℤ) * x + 40 * x + b = (4 * x + c) ^ 2) : b = 25 :=
sorry

end square_of_binomial_b_value_l199_19938


namespace sum_opposite_abs_val_eq_neg_nine_l199_19997

theorem sum_opposite_abs_val_eq_neg_nine (a b : ℤ) (h1 : a = -15) (h2 : b = 6) : a + b = -9 := 
by
  -- conditions given
  rw [h1, h2]
  -- skip the proof
  sorry

end sum_opposite_abs_val_eq_neg_nine_l199_19997


namespace flower_bee_difference_proof_l199_19925

variable (flowers bees : ℕ)

def flowers_bees_difference (flowers bees : ℕ) : ℕ :=
  flowers - bees

theorem flower_bee_difference_proof : flowers_bees_difference 5 3 = 2 :=
by
  sorry

end flower_bee_difference_proof_l199_19925


namespace geometric_sequence_a_eq_one_l199_19939

theorem geometric_sequence_a_eq_one (a : ℝ) 
  (h₁ : ∃ (r : ℝ), a = 1 / (1 - r) ∧ r = a - 1/2 ∧ r ≠ 0) : 
  a = 1 := 
sorry

end geometric_sequence_a_eq_one_l199_19939


namespace mark_donates_cans_of_soup_l199_19976

theorem mark_donates_cans_of_soup:
  let n_shelters := 6
  let p_per_shelter := 30
  let c_per_person := 10
  let total_people := n_shelters * p_per_shelter
  let total_cans := total_people * c_per_person
  total_cans = 1800 :=
by sorry

end mark_donates_cans_of_soup_l199_19976


namespace compare_flavors_l199_19957

def flavor_ratings_A := [7, 9, 8, 6, 10]
def flavor_ratings_B := [5, 6, 10, 10, 9]

def mean (l : List ℕ) : ℚ := l.sum / l.length

def variance (l : List ℕ) : ℚ :=
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem compare_flavors : 
  mean flavor_ratings_A = mean flavor_ratings_B ∧ variance flavor_ratings_A < variance flavor_ratings_B := by
  sorry

end compare_flavors_l199_19957


namespace annika_hike_distance_l199_19994

-- Define the conditions as definitions
def hiking_rate : ℝ := 10  -- rate of 10 minutes per kilometer
def total_minutes : ℝ := 35 -- total available time in minutes
def total_distance_east : ℝ := 3 -- total distance hiked east

-- Define the statement to prove
theorem annika_hike_distance : ∃ (x : ℝ), (x / hiking_rate) + ((total_distance_east - x) / hiking_rate) = (total_minutes - 30) / hiking_rate :=
by
  sorry

end annika_hike_distance_l199_19994


namespace shift_down_two_units_l199_19989

theorem shift_down_two_units (x : ℝ) : 
  (y = 2 * x) → (y - 2 = 2 * x - 2) := by
sorry

end shift_down_two_units_l199_19989


namespace starting_number_is_33_l199_19978

theorem starting_number_is_33 (n : ℕ)
  (h1 : ∀ k, (33 + k * 11 ≤ 79) → (k < 5))
  (h2 : ∀ k, (k < 5) → (33 + k * 11 ≤ 79)) :
  n = 33 :=
sorry

end starting_number_is_33_l199_19978


namespace complement_of_M_l199_19947

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x ≥ 1}

theorem complement_of_M :
  (U \ M) = {x | x < 1} :=
by
  sorry

end complement_of_M_l199_19947


namespace selected_people_take_B_l199_19930

def arithmetic_sequence (a d n : Nat) : Nat := a + (n - 1) * d

theorem selected_people_take_B (a d total sampleCount start n_upper n_lower : Nat) :
  a = 9 →
  d = 30 →
  total = 960 →
  sampleCount = 32 →
  start = 451 →
  n_upper = 25 →
  n_lower = 16 →
  (960 / 32) = d → 
  (10 = n_upper - n_lower + 1) ∧ 
  ∀ n, (n_lower ≤ n ∧ n ≤ n_upper) → (start ≤ arithmetic_sequence a d n ∧ arithmetic_sequence a d n ≤ 750) :=
by sorry

end selected_people_take_B_l199_19930


namespace no_matching_formula_l199_19984

def xy_pairs : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 35), (4, 69), (5, 119)]

def formula_a (x : ℕ) : ℕ := x^3 + x^2 + x + 2
def formula_b (x : ℕ) : ℕ := 3 * x^2 + 2 * x + 1
def formula_c (x : ℕ) : ℕ := 2 * x^3 - x + 4
def formula_d (x : ℕ) : ℕ := 3 * x^3 + 2 * x^2 + x + 1

theorem no_matching_formula :
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_a pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_b pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_c pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_d pair.fst) :=
by
  sorry

end no_matching_formula_l199_19984


namespace brenda_age_l199_19955

variable (A B J : ℕ)

theorem brenda_age :
  (A = 3 * B) →
  (J = B + 6) →
  (A = J) →
  (B = 3) :=
by
  intros h1 h2 h3
  -- condition: A = 3 * B
  -- condition: J = B + 6
  -- condition: A = J
  -- prove B = 3
  sorry

end brenda_age_l199_19955


namespace p_sufficient_condition_neg_q_l199_19981

variables (p q : Prop)

theorem p_sufficient_condition_neg_q (hnecsuff_q : ¬p → q) (hnecsuff_p : ¬q → p) : (p → ¬q) :=
by
  sorry

end p_sufficient_condition_neg_q_l199_19981


namespace simplify_expression_l199_19911

theorem simplify_expression (x : ℝ) : (3 * x + 6 - 5 * x) / 3 = - (2 / 3) * x + 2 :=
by
  sorry

end simplify_expression_l199_19911


namespace width_of_rectangle_l199_19980

-- Define the problem constants and parameters
variable (L W : ℝ)

-- State the main theorem about the width
theorem width_of_rectangle (h₁ : L * W = 50) (h₂ : L + W = 15) : W = 5 :=
sorry

end width_of_rectangle_l199_19980


namespace widget_difference_l199_19952

variable (w t : ℕ)

def monday_widgets (w t : ℕ) : ℕ := w * t
def tuesday_widgets (w t : ℕ) : ℕ := (w + 5) * (t - 3)

theorem widget_difference (h : w = 3 * t) :
  monday_widgets w t - tuesday_widgets w t = 4 * t + 15 :=
by
  sorry

end widget_difference_l199_19952


namespace tomatoes_picked_second_week_l199_19983

-- Define the constants
def initial_tomatoes : Nat := 100
def fraction_picked_first_week : Nat := 1 / 4
def remaining_tomatoes : Nat := 15

-- Theorem to prove the number of tomatoes Jane picked in the second week
theorem tomatoes_picked_second_week (x : Nat) :
  let T := initial_tomatoes
  let p := fraction_picked_first_week
  let r := remaining_tomatoes
  let first_week_pick := T * p
  let remaining_after_first := T - first_week_pick
  let total_picked := remaining_after_first - r
  let second_week_pick := total_picked / 3
  second_week_pick = 20 := 
sorry

end tomatoes_picked_second_week_l199_19983


namespace batsman_average_after_17th_inning_l199_19937

-- Definitions for the conditions
def runs_scored_in_17th_inning : ℝ := 95
def increase_in_average : ℝ := 2.5

-- Lean statement encapsulating the problem
theorem batsman_average_after_17th_inning (A : ℝ) (h : 16 * A + runs_scored_in_17th_inning = 17 * (A + increase_in_average)) :
  A + increase_in_average = 55 := 
sorry

end batsman_average_after_17th_inning_l199_19937


namespace min_value_a4b3c2_l199_19969

theorem min_value_a4b3c2 {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1/a + 1/b + 1/c = 9) :
  a ^ 4 * b ^ 3 * c ^ 2 ≥ 1 / 5184 := 
sorry

end min_value_a4b3c2_l199_19969


namespace cookie_contest_l199_19968

theorem cookie_contest (A B : ℚ) (hA : A = 5/6) (hB : B = 2/3) :
  A - B = 1/6 :=
by 
  sorry

end cookie_contest_l199_19968


namespace num_both_sports_l199_19985

def num_people := 310
def num_tennis := 138
def num_baseball := 255
def num_no_sport := 11

theorem num_both_sports : (num_tennis + num_baseball - (num_people - num_no_sport)) = 94 :=
by 
-- leave the proof out for now
sorry

end num_both_sports_l199_19985


namespace x_y_ge_two_l199_19986

open Real

theorem x_y_ge_two (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : 
  x + y ≥ 2 ∧ (x + y = 2 → x = 1 ∧ y = 1) :=
by {
 sorry
}

end x_y_ge_two_l199_19986


namespace percent_shaded_area_of_rectangle_l199_19946

theorem percent_shaded_area_of_rectangle
  (side_length : ℝ)
  (length_rectangle : ℝ)
  (width_rectangle : ℝ)
  (overlap_length : ℝ)
  (h1 : side_length = 12)
  (h2 : length_rectangle = 20)
  (h3 : width_rectangle = 12)
  (h4 : overlap_length = 4)
  : (overlap_length * width_rectangle) / (length_rectangle * width_rectangle) * 100 = 20 :=
  sorry

end percent_shaded_area_of_rectangle_l199_19946


namespace parabola_focus_to_equation_l199_19901

-- Define the focus of the parabola
def F : (ℝ × ℝ) := (5, 0)

-- Define the standard equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 20 * x

-- State the problem in Lean
theorem parabola_focus_to_equation : 
  (F = (5, 0)) → ∀ x y, parabola_equation x y :=
by
  intro h_focus_eq
  sorry

end parabola_focus_to_equation_l199_19901


namespace ellipse_properties_l199_19990

theorem ellipse_properties (h k a b : ℝ)
  (h_eq : h = 1)
  (k_eq : k = -3)
  (a_eq : a = 7)
  (b_eq : b = 4) :
  h + k + a + b = 9 :=
by
  sorry

end ellipse_properties_l199_19990


namespace radius_of_cookie_l199_19965

theorem radius_of_cookie (x y : ℝ) : 
  (x^2 + y^2 + x - 5 * y = 10) → 
  ∃ r, (r = Real.sqrt (33 / 2)) :=
by
  sorry

end radius_of_cookie_l199_19965


namespace cylinder_cut_is_cylinder_l199_19988

-- Define what it means to be a cylinder
structure Cylinder (r h : ℝ) : Prop :=
(r_pos : r > 0)
(h_pos : h > 0)

-- Define the condition of cutting a cylinder with two parallel planes
def cut_by_parallel_planes (c : Cylinder r h) (d : ℝ) : Prop :=
d > 0 ∧ d < h

-- Prove that the part between the parallel planes is still a cylinder
theorem cylinder_cut_is_cylinder (r h d : ℝ) (c : Cylinder r h) (H : cut_by_parallel_planes c d) :
  ∃ r' h', Cylinder r' h' :=
sorry

end cylinder_cut_is_cylinder_l199_19988


namespace curve_tangents_intersection_l199_19992

theorem curve_tangents_intersection (a : ℝ) :
  (∃ x₀ y₀, y₀ = Real.exp x₀ ∧ y₀ = (x₀ + a)^2 ∧ Real.exp x₀ = 2 * (x₀ + a)) → a = 2 - Real.log 4 :=
by
  sorry

end curve_tangents_intersection_l199_19992


namespace number_of_10_yuan_coins_is_1_l199_19966

theorem number_of_10_yuan_coins_is_1
  (n : ℕ) -- number of coins
  (v : ℕ) -- total value of coins
  (c1 c5 c10 c50 : ℕ) -- number of 1, 5, 10, and 50 yuan coins
  (h1 : n = 9) -- there are nine coins in total
  (h2 : v = 177) -- the total value of these coins is 177 yuan
  (h3 : c1 ≥ 1 ∧ c5 ≥ 1 ∧ c10 ≥ 1 ∧ c50 ≥ 1) -- at least one coin of each denomination
  (h4 : c1 + c5 + c10 + c50 = n) -- sum of all coins number is n
  (h5 : c1 * 1 + c5 * 5 + c10 * 10 + c50 * 50 = v) -- total value of all coins is v
  : c10 = 1 := 
sorry

end number_of_10_yuan_coins_is_1_l199_19966


namespace find_k_l199_19973

theorem find_k (k : ℝ) : (∃ b : ℝ, (x : ℝ) → x^2 - 60 * x + k = (x + b)^2) → k = 900 :=
by 
  sorry

end find_k_l199_19973


namespace max_a4b2c_l199_19918

-- Define the conditions and required statement
theorem max_a4b2c (a b c : ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a + b + c = 1) :
    a^4 * b^2 * c ≤ 1024 / 117649 :=
sorry

end max_a4b2c_l199_19918


namespace length_of_second_train_l199_19972

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (time_to_cross : ℝ)
  (h1 : length_first_train = 270)
  (h2 : speed_first_train = 120)
  (h3 : speed_second_train = 80)
  (h4 : time_to_cross = 9) :
  ∃ length_second_train : ℝ, length_second_train = 229.95 :=
by
  sorry

end length_of_second_train_l199_19972


namespace base_eight_to_base_ten_l199_19929

theorem base_eight_to_base_ten {d1 d2 d3 : ℕ} (h1 : d1 = 1) (h2 : d2 = 5) (h3 : d3 = 7) :
  d3 * 8^0 + d2 * 8^1 + d1 * 8^2 = 111 := 
by
  sorry

end base_eight_to_base_ten_l199_19929


namespace sum_of_dice_less_than_10_probability_l199_19975

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

end sum_of_dice_less_than_10_probability_l199_19975


namespace number_of_pairs_is_2_pow_14_l199_19910

noncomputable def number_of_pairs_satisfying_conditions : ℕ :=
  let fact5 := Nat.factorial 5
  let fact50 := Nat.factorial 50
  Nat.card {p : ℕ × ℕ | Nat.gcd p.1 p.2 = fact5 ∧ Nat.lcm p.1 p.2 = fact50}

theorem number_of_pairs_is_2_pow_14 :
  number_of_pairs_satisfying_conditions = 2^14 := by
  sorry

end number_of_pairs_is_2_pow_14_l199_19910


namespace find_real_solutions_l199_19961

noncomputable def polynomial_expression (x : ℝ) : ℝ := (x - 2)^2 * (x - 4) * (x - 1)

theorem find_real_solutions :
  ∀ (x : ℝ), (x ≠ 3) ∧ (x ≠ 5) ∧ (polynomial_expression x = 1) ↔ (x = 1 ∨ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2) := sorry

end find_real_solutions_l199_19961


namespace cones_slant_height_angle_l199_19907

theorem cones_slant_height_angle :
  ∀ (α: ℝ),
  α = 2 * Real.arccos (Real.sqrt (2 / (2 + Real.sqrt 2))) :=
by
  sorry

end cones_slant_height_angle_l199_19907


namespace gcd_bc_minimum_l199_19900

theorem gcd_bc_minimum
  (a b c : ℕ)
  (h1 : Nat.gcd a b = 360)
  (h2 : Nat.gcd a c = 1170)
  (h3 : ∃ k1 : ℕ, b = 5 * k1)
  (h4 : ∃ k2 : ℕ, c = 13 * k2) : Nat.gcd b c = 90 :=
by
  sorry

end gcd_bc_minimum_l199_19900


namespace complement_union_A_B_complement_A_intersection_B_l199_19991

open Set

-- Definitions of A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Proving the complement of A ∪ B
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ 2 ∨ 10 ≤ x} :=
by sorry

-- Proving the intersection of the complement of A with B
theorem complement_A_intersection_B : (Aᶜ ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
by sorry

end complement_union_A_B_complement_A_intersection_B_l199_19991


namespace largest_visits_l199_19905

theorem largest_visits (stores : ℕ) (total_visits : ℕ) (unique_visitors : ℕ) 
  (visits_two_stores : ℕ) (remaining_visitors : ℕ) : 
  stores = 7 ∧ total_visits = 21 ∧ unique_visitors = 11 ∧ visits_two_stores = 7 ∧ remaining_visitors = (unique_visitors - visits_two_stores) →
  (remaining_visitors * 2 <= total_visits - visits_two_stores * 2) → (∀ v : ℕ, v * unique_visitors = total_visits) →
  (∃ v_max : ℕ, v_max = 4) :=
by
  sorry

end largest_visits_l199_19905


namespace geometric_sequence_k_value_l199_19996

theorem geometric_sequence_k_value (a : ℕ → ℝ) (S : ℕ → ℝ) (a1_pos : 0 < a 1)
  (geometric_seq : ∀ n, a (n + 2) = a n * (a 3 / a 1)) (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) (h_Sk : S k = 63) :
  k = 6 := 
by
  sorry

end geometric_sequence_k_value_l199_19996


namespace total_customers_l199_19932

namespace math_proof

-- Definitions based on the problem's conditions.
def tables : ℕ := 9
def women_per_table : ℕ := 7
def men_per_table : ℕ := 3

-- The theorem stating the problem's question and correct answer.
theorem total_customers : tables * (women_per_table + men_per_table) = 90 := 
by
  -- This would be expanded into a proof, but we use sorry to bypass it here.
  sorry

end math_proof

end total_customers_l199_19932


namespace binom_10_8_equals_45_l199_19924

theorem binom_10_8_equals_45 : Nat.choose 10 8 = 45 := 
by
  sorry

end binom_10_8_equals_45_l199_19924


namespace vacation_cost_division_l199_19982

theorem vacation_cost_division 
  (total_cost : ℝ) 
  (initial_people : ℝ) 
  (initial_cost_per_person : ℝ) 
  (cost_difference : ℝ) 
  (new_cost_per_person : ℝ) 
  (new_people : ℝ) 
  (h1 : total_cost = 1000) 
  (h2 : initial_people = 4) 
  (h3 : initial_cost_per_person = total_cost / initial_people) 
  (h4 : initial_cost_per_person = 250) 
  (h5 : cost_difference = 50) 
  (h6 : new_cost_per_person = initial_cost_per_person - cost_difference) 
  (h7 : new_cost_per_person = 200) 
  (h8 : total_cost / new_people = new_cost_per_person) :
  new_people = 5 := 
sorry

end vacation_cost_division_l199_19982


namespace unique_nonneg_sequence_l199_19916

theorem unique_nonneg_sequence (a : List ℝ) (h_sum : 0 < a.sum) :
  ∃ b : List ℝ, (∀ x ∈ b, 0 ≤ x) ∧ 
                (∃ f : List ℝ → List ℝ, (f a = b) ∧ (∀ x y z, f (x :: y :: z :: tl) = (x + y) :: (-y) :: (z + y) :: tl)) :=
sorry

end unique_nonneg_sequence_l199_19916


namespace largest_angle_right_triangle_l199_19909

theorem largest_angle_right_triangle
  (a b c : ℝ)
  (h₁ : ∃ x : ℝ, x^2 + 4 * (c + 2) = (c + 4) * x)
  (h₂ : a + b = c + 4)
  (h₃ : a * b = 4 * (c + 2))
  : ∃ x : ℝ, x = 90 :=
by {
  sorry
}

end largest_angle_right_triangle_l199_19909


namespace inequality_sum_squares_l199_19928

theorem inequality_sum_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 :=
sorry

end inequality_sum_squares_l199_19928


namespace marissa_initial_ribbon_l199_19943

theorem marissa_initial_ribbon (ribbon_per_box : ℝ) (number_of_boxes : ℝ) (ribbon_left : ℝ) : 
  (ribbon_per_box = 0.7) → (number_of_boxes = 5) → (ribbon_left = 1) → 
  (ribbon_per_box * number_of_boxes + ribbon_left = 4.5) :=
  by
    intros
    sorry

end marissa_initial_ribbon_l199_19943


namespace fraction_value_l199_19993

variable (x y : ℚ)

theorem fraction_value (h₁ : x = 4 / 6) (h₂ : y = 8 / 12) : 
  (6 * x + 8 * y) / (48 * x * y) = 7 / 16 :=
by
  sorry

end fraction_value_l199_19993


namespace smallest_k_for_sum_of_squares_multiple_of_360_l199_19927

theorem smallest_k_for_sum_of_squares_multiple_of_360 :
  ∃ k : ℕ, k > 0 ∧ (k * (k + 1) * (2 * k + 1)) / 6 % 360 = 0 ∧ ∀ n : ℕ, n > 0 → (n * (n + 1) * (2 * n + 1)) / 6 % 360 = 0 → k ≤ n :=
by sorry

end smallest_k_for_sum_of_squares_multiple_of_360_l199_19927


namespace num_bags_of_cookies_l199_19970

theorem num_bags_of_cookies (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 703) (h2 : cookies_per_bag = 19) : total_cookies / cookies_per_bag = 37 :=
by
  sorry

end num_bags_of_cookies_l199_19970


namespace largest_possible_perimeter_l199_19948

theorem largest_possible_perimeter (x : ℕ) (h1 : 1 < x) (h2 : x < 15) : 
  (7 + 8 + x) ≤ 29 := 
sorry

end largest_possible_perimeter_l199_19948


namespace fraction_equiv_ratio_equiv_percentage_equiv_l199_19917

-- Define the problem's components and conditions.
def frac_1 : ℚ := 3 / 5
def frac_2 (a b : ℚ) : Prop := 3 / 5 = a / b
def ratio_1 (a b : ℚ) : Prop := 10 / a = b / 100
def percentage_1 (a b : ℚ) : Prop := (a / b) * 100 = 60

-- Problem statement 1: Fraction equality
theorem fraction_equiv : frac_2 12 20 := 
by sorry

-- Problem statement 2: Ratio equality
theorem ratio_equiv : ratio_1 (50 / 3) 60 := 
by sorry

-- Problem statement 3: Percentage equality
theorem percentage_equiv : percentage_1 60 100 := 
by sorry

end fraction_equiv_ratio_equiv_percentage_equiv_l199_19917


namespace percentage_disliked_by_both_l199_19903

theorem percentage_disliked_by_both (total_comics liked_by_females liked_by_males disliked_by_both : ℕ) 
  (total_comics_eq : total_comics = 300)
  (liked_by_females_eq : liked_by_females = 30 * total_comics / 100)
  (liked_by_males_eq : liked_by_males = 120)
  (disliked_by_both_eq : disliked_by_both = total_comics - (liked_by_females + liked_by_males)) :
  (disliked_by_both * 100 / total_comics) = 30 := by
  sorry

end percentage_disliked_by_both_l199_19903


namespace half_angle_quadrant_l199_19923

theorem half_angle_quadrant
  (α : ℝ) (k : ℤ)
  (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (∃ m : ℤ, m * π < α / 2 ∧ α / 2 < m * π + π / 2) :=
by
  sorry

end half_angle_quadrant_l199_19923


namespace monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_l199_19915

-- Proof Problem I
noncomputable def f1 (x : ℝ) := x^2 + x - Real.log x

theorem monotonic_intervals_a1 : 
  (∀ x, 0 < x ∧ x < 1 / 2 → f1 x < 0) ∧ (∀ x, 1 / 2 < x → f1 x > 0) := 
sorry

-- Proof Problem II
noncomputable def f2 (x : ℝ) (a : ℝ) := x^2 + a * x - Real.log x

theorem decreasing_on_1_to_2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f2 x a ≤ 0) → a ≤ -7 / 2 :=
sorry

-- Proof Problem III
noncomputable def g (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem exists_a_for_minimum_value :
  ∃ a : ℝ, (∀ x, 0 < x ∧ x ≤ Real.exp 1 → g x a = 3) ∧ a = Real.exp 2 :=
sorry

end monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_l199_19915


namespace sara_grew_4_onions_l199_19933

def onions_sally := 5
def onions_fred := 9
def total_onions := 18

def onions_sara : ℕ := total_onions - (onions_sally + onions_fred)

theorem sara_grew_4_onions : onions_sara = 4 := by
  -- proof here
  sorry

end sara_grew_4_onions_l199_19933


namespace students_came_to_school_l199_19934

theorem students_came_to_school (F M T A : ℕ) 
    (hF : F = 658)
    (hM : M = F - 38)
    (hA : A = 17)
    (hT : T = M + F - A) :
    T = 1261 := by 
sorry

end students_came_to_school_l199_19934


namespace value_of_f_neg_a_l199_19959

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = -2 := 
by 
  sorry

end value_of_f_neg_a_l199_19959


namespace boys_less_than_two_fifths_total_l199_19913

theorem boys_less_than_two_fifths_total
  (n b g n1 n2 b1 b2 : ℕ)
  (h_total: n = b + g)
  (h_first_trip: b1 < 2 * n1 / 5)
  (h_second_trip: b2 < 2 * n2 / 5)
  (h_participation: b ≤ b1 + b2)
  (h_total_participants: n ≤ n1 + n2) :
  b < 2 * n / 5 := 
sorry

end boys_less_than_two_fifths_total_l199_19913
