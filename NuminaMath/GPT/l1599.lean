import Mathlib

namespace NUMINAMATH_GPT_perfect_square_trinomial_l1599_159982

theorem perfect_square_trinomial (m : ℤ) : 
  (∃ x y : ℝ, 16 * x^2 + m * x * y + 25 * y^2 = (4 * x + 5 * y)^2 ∨ 16 * x^2 + m * x * y + 25 * y^2 = (4 * x - 5 * y)^2) ↔ (m = 40 ∨ m = -40) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1599_159982


namespace NUMINAMATH_GPT_parts_per_day_l1599_159964

noncomputable def total_parts : ℕ := 400
noncomputable def unfinished_parts_after_3_days : ℕ := 60
noncomputable def excess_parts_after_3_days : ℕ := 20

variables (x y : ℕ)

noncomputable def condition1 : Prop := (3 * x + 2 * y = total_parts - unfinished_parts_after_3_days)
noncomputable def condition2 : Prop := (3 * x + 3 * y = total_parts + excess_parts_after_3_days)

theorem parts_per_day (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 60 ∧ y = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_parts_per_day_l1599_159964


namespace NUMINAMATH_GPT_inequality_l1599_159965

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log (Real.pi)
noncomputable def c : ℝ := Real.log 0.5 / Real.log 2

theorem inequality (h1: a = Real.sqrt 2) (h2: b = Real.log 3 / Real.log Real.pi) (h3: c = Real.log 0.5 / Real.log 2) : a > b ∧ b > c := 
by 
  sorry

end NUMINAMATH_GPT_inequality_l1599_159965


namespace NUMINAMATH_GPT_students_answered_both_correct_l1599_159963

theorem students_answered_both_correct (total_students : ℕ)
  (answered_sets_correctly : ℕ) (answered_functions_correctly : ℕ)
  (both_wrong : ℕ) (total : total_students = 50)
  (sets_correct : answered_sets_correctly = 40)
  (functions_correct : answered_functions_correctly = 31)
  (wrong_both : both_wrong = 4) :
  (40 + 31 - (total_students - 4) + both_wrong = 50) → total_students - (40 + 31 - (total_students - 4)) = 29 :=
by
  sorry

end NUMINAMATH_GPT_students_answered_both_correct_l1599_159963


namespace NUMINAMATH_GPT_joe_commute_time_l1599_159968

theorem joe_commute_time
  (d : ℝ) -- total one-way distance from home to school
  (rw : ℝ) -- Joe's walking rate
  (rr : ℝ := 4 * rw) -- Joe's running rate (4 times walking rate)
  (walking_time_for_one_third : ℝ := 9) -- Joe takes 9 minutes to walk one-third distance
  (walking_time_two_thirds : ℝ := 2 * walking_time_for_one_third) -- time to walk two-thirds distance
  (running_time_two_thirds : ℝ := walking_time_two_thirds / 4) -- time to run two-thirds 
  : (2 * walking_time_two_thirds + running_time_two_thirds) = 40.5 := -- total travel time
by
  sorry

end NUMINAMATH_GPT_joe_commute_time_l1599_159968


namespace NUMINAMATH_GPT_rs_value_l1599_159991

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs: 0 < s) (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 3 / 4) :
  r * s = Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_GPT_rs_value_l1599_159991


namespace NUMINAMATH_GPT_length_of_train_B_l1599_159976

-- Given conditions
def lengthTrainA := 125  -- in meters
def speedTrainA := 54    -- in km/hr
def speedTrainB := 36    -- in km/hr
def timeToCross := 11    -- in seconds

-- Conversion factor from km/hr to m/s
def kmhr_to_mps (v : ℕ) : ℕ := v * 5 / 18

-- Relative speed of the trains in m/s
def relativeSpeed := kmhr_to_mps (speedTrainA + speedTrainB)

-- Distance covered in the given time
def distanceCovered := relativeSpeed * timeToCross

-- Proof statement
theorem length_of_train_B : distanceCovered - lengthTrainA = 150 := 
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_length_of_train_B_l1599_159976


namespace NUMINAMATH_GPT_bernoulli_inequality_l1599_159944

theorem bernoulli_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) : 1 + n * x ≤ (1 + x) ^ n :=
sorry

end NUMINAMATH_GPT_bernoulli_inequality_l1599_159944


namespace NUMINAMATH_GPT_tim_drinks_amount_l1599_159902

theorem tim_drinks_amount (H : ℚ := 2/7) (T : ℚ := 5/8) : 
  (T * H) = 5/28 :=
by sorry

end NUMINAMATH_GPT_tim_drinks_amount_l1599_159902


namespace NUMINAMATH_GPT_leaf_raking_earnings_l1599_159915

variable {S M L P : ℕ}

theorem leaf_raking_earnings (h1 : 5 * 4 + 7 * 2 + 10 * 1 + 3 * 1 = 47)
                             (h2 : 5 * 2 + 3 * 1 + 7 * 1 + 10 * 2 = 40)
                             (h3 : 163 - 87 = 76) :
  5 * S + 7 * M + 10 * L + 3 * P = 76 :=
by
  sorry

end NUMINAMATH_GPT_leaf_raking_earnings_l1599_159915


namespace NUMINAMATH_GPT_total_population_l1599_159987

theorem total_population (b g t : ℕ) (h1 : b = 2 * g) (h2 : g = 4 * t) : b + g + t = 13 * t :=
by
  sorry

end NUMINAMATH_GPT_total_population_l1599_159987


namespace NUMINAMATH_GPT_max_value_of_expression_l1599_159904

noncomputable def max_value (x : ℝ) : ℝ :=
  x * (1 + x) * (3 - x)

theorem max_value_of_expression :
  ∃ x : ℝ, 0 < x ∧ max_value x = (70 + 26 * Real.sqrt 13) / 27 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l1599_159904


namespace NUMINAMATH_GPT_average_score_of_class_l1599_159992

theorem average_score_of_class (total_students : ℕ)
  (perc_assigned_day perc_makeup_day : ℝ)
  (average_assigned_day average_makeup_day : ℝ)
  (h_total : total_students = 100)
  (h_perc_assigned_day : perc_assigned_day = 0.70)
  (h_perc_makeup_day : perc_makeup_day = 0.30)
  (h_average_assigned_day : average_assigned_day = 55)
  (h_average_makeup_day : average_makeup_day = 95) :
  ((perc_assigned_day * total_students * average_assigned_day + perc_makeup_day * total_students * average_makeup_day) / total_students) = 67 := by
  sorry

end NUMINAMATH_GPT_average_score_of_class_l1599_159992


namespace NUMINAMATH_GPT_triangle_height_l1599_159958

def width := 10
def length := 2 * width
def area_rectangle := width * length
def base_triangle := width

theorem triangle_height (h : ℝ) : (1 / 2) * base_triangle * h = area_rectangle → h = 40 :=
by
  sorry

end NUMINAMATH_GPT_triangle_height_l1599_159958


namespace NUMINAMATH_GPT_original_number_q_l1599_159929

variables (q : ℝ) (a b c : ℝ)
 
theorem original_number_q : 
  (a = 1.125 * q) → (b = 0.75 * q) → (c = 30) → (a - b = c) → q = 80 :=
by
  sorry

end NUMINAMATH_GPT_original_number_q_l1599_159929


namespace NUMINAMATH_GPT_watermelon_heavier_than_pineapple_l1599_159938

noncomputable def watermelon_weight : ℕ := 1 * 1000 + 300 -- Weight of one watermelon in grams
noncomputable def pineapple_weight : ℕ := 450 -- Weight of one pineapple in grams

theorem watermelon_heavier_than_pineapple :
    (4 * watermelon_weight = 5 * 1000 + 200) →
    (3 * watermelon_weight + 4 * pineapple_weight = 5 * 1000 + 700) →
    watermelon_weight - pineapple_weight = 850 :=
by
    intros h1 h2
    sorry

end NUMINAMATH_GPT_watermelon_heavier_than_pineapple_l1599_159938


namespace NUMINAMATH_GPT_cake_volume_icing_area_sum_l1599_159978

-- Define the conditions based on the problem description
def cube_edge_length : ℕ := 4
def volume_of_piece := 16
def icing_area := 12

-- Define the statements to be proven
theorem cake_volume_icing_area_sum : 
  volume_of_piece + icing_area = 28 := 
sorry

end NUMINAMATH_GPT_cake_volume_icing_area_sum_l1599_159978


namespace NUMINAMATH_GPT_permissible_range_n_l1599_159973

theorem permissible_range_n (n x y m : ℝ) (hn : n ≤ x) (hxy : x < y) (hy : y ≤ n+1)
  (hm_in: x < m ∧ m < y) (habs_eq : |y| = |m| + |x|): 
  -1 < n ∧ n < 1 := sorry

end NUMINAMATH_GPT_permissible_range_n_l1599_159973


namespace NUMINAMATH_GPT_batsman_total_score_l1599_159962

-- We establish our variables and conditions first
variables (T : ℕ) -- total score
variables (boundaries : ℕ := 3) -- number of boundaries
variables (sixes : ℕ := 8) -- number of sixes
variables (boundary_runs_per : ℕ := 4) -- runs per boundary
variables (six_runs_per : ℕ := 6) -- runs per six
variables (running_percentage : ℕ := 50) -- percentage of runs made by running

-- Define the amounts of runs from boundaries and sixes
def runs_from_boundaries := boundaries * boundary_runs_per
def runs_from_sixes := sixes * six_runs_per

-- Main theorem to prove
theorem batsman_total_score :
  T = runs_from_boundaries + runs_from_sixes + T / 2 → T = 120 :=
by
  sorry

end NUMINAMATH_GPT_batsman_total_score_l1599_159962


namespace NUMINAMATH_GPT_number_of_balls_sold_l1599_159984

-- Let n be the number of balls sold
variable (n : ℕ)

-- The given conditions
def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 60
def loss := 5 * cost_price_per_ball

-- Prove that if the selling price of 'n' balls is Rs. 720 and 
-- the loss is equal to the cost price of 5 balls, then the 
-- number of balls sold (n) is 17.
theorem number_of_balls_sold (h1 : selling_price = 720) 
                             (h2 : cost_price_per_ball = 60) 
                             (h3 : loss = 5 * cost_price_per_ball) 
                             (hsale : n * cost_price_per_ball - selling_price = loss) : 
  n = 17 := 
by
  sorry

end NUMINAMATH_GPT_number_of_balls_sold_l1599_159984


namespace NUMINAMATH_GPT_pedestrian_wait_probability_l1599_159911

-- Define the duration of the red light
def red_light_duration := 45

-- Define the favorable time window for the pedestrian to wait at least 20 seconds
def favorable_window := 25

-- The probability that the pedestrian has to wait at least 20 seconds
def probability_wait_at_least_20 : ℚ := favorable_window / red_light_duration

theorem pedestrian_wait_probability : probability_wait_at_least_20 = 5 / 9 := by
  sorry

end NUMINAMATH_GPT_pedestrian_wait_probability_l1599_159911


namespace NUMINAMATH_GPT_Alina_messages_comparison_l1599_159955

theorem Alina_messages_comparison 
  (lucia_day1 : ℕ) (alina_day1 : ℕ) (lucia_day2 : ℕ) (alina_day2 : ℕ) (lucia_day3 : ℕ) (alina_day3 : ℕ)
  (h1 : lucia_day1 = 120)
  (h2 : alina_day1 = lucia_day1 - 20)
  (h3 : lucia_day2 = lucia_day1 / 3)
  (h4 : lucia_day3 = lucia_day1)
  (h5 : alina_day3 = alina_day1)
  (h6 : lucia_day1 + lucia_day2 + lucia_day3 + alina_day1 + alina_day2 + alina_day3 = 680) :
  alina_day2 = alina_day1 + 100 :=
sorry

end NUMINAMATH_GPT_Alina_messages_comparison_l1599_159955


namespace NUMINAMATH_GPT_estimate_number_of_trees_l1599_159937

-- Definitions derived from the conditions
def forest_length : ℝ := 100
def forest_width : ℝ := 0.5
def plot_length : ℝ := 1
def plot_width : ℝ := 0.5
def tree_counts : List ℕ := [65110, 63200, 64600, 64700, 67300, 63300, 65100, 66600, 62800, 65500]

-- The main theorem stating the problem
theorem estimate_number_of_trees :
  let avg_trees_per_plot := tree_counts.sum / tree_counts.length
  let total_plots := (forest_length * forest_width) / (plot_length * plot_width)
  avg_trees_per_plot * total_plots = 6482100 :=
by
  sorry

end NUMINAMATH_GPT_estimate_number_of_trees_l1599_159937


namespace NUMINAMATH_GPT_Carol_saves_9_per_week_l1599_159971

variable (C : ℤ)

def Carol_savings (weeks : ℤ) : ℤ :=
  60 + weeks * C

def Mike_savings (weeks : ℤ) : ℤ :=
  90 + weeks * 3

theorem Carol_saves_9_per_week (h : Carol_savings C 5 = Mike_savings 5) : C = 9 :=
by
  dsimp [Carol_savings, Mike_savings] at h
  sorry

end NUMINAMATH_GPT_Carol_saves_9_per_week_l1599_159971


namespace NUMINAMATH_GPT_bob_cleaning_time_is_correct_l1599_159913

-- Definitions for conditions
def timeAliceTakes : ℕ := 32
def bobTimeFactor : ℚ := 3 / 4

-- Theorem to prove
theorem bob_cleaning_time_is_correct : (bobTimeFactor * timeAliceTakes : ℚ) = 24 := 
by
  sorry

end NUMINAMATH_GPT_bob_cleaning_time_is_correct_l1599_159913


namespace NUMINAMATH_GPT_factorization_correct_l1599_159900

theorem factorization_correct (c : ℝ) : (x : ℝ) → x^2 - x + c = (x + 2) * (x - 3) → c = -6 := by
  intro x h
  sorry

end NUMINAMATH_GPT_factorization_correct_l1599_159900


namespace NUMINAMATH_GPT_bucket_full_weight_l1599_159908

variable (p q r : ℚ)
variable (x y : ℚ)

-- Define the conditions
def condition1 : Prop := p = r + (3 / 4) * y
def condition2 : Prop := q = r + (1 / 3) * y
def condition3 : Prop := x = r

-- Define the conclusion
def conclusion : Prop := x + y = (4 * p - r) / 3

-- The theorem stating that the conclusion follows from the conditions
theorem bucket_full_weight (h1 : condition1 p r y) (h2 : condition2 q r y) (h3 : condition3 x r) : conclusion x y p r :=
by
  sorry

end NUMINAMATH_GPT_bucket_full_weight_l1599_159908


namespace NUMINAMATH_GPT_general_term_defines_sequence_l1599_159939

/-- Sequence definition -/
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (2 * a n + 6) / (a n + 1)

/-- General term formula -/
def general_term (n : ℕ) : ℚ :=
  (3 * 4 ^ n + 2 * (-1) ^ n) / (4 ^ n - (-1) ^ n)

/-- Theorem stating that the general term formula defines the sequence -/
theorem general_term_defines_sequence : ∀ (a : ℕ → ℚ), seq a → ∀ n, a n = general_term n :=
by
  intros a h_seq n
  sorry

end NUMINAMATH_GPT_general_term_defines_sequence_l1599_159939


namespace NUMINAMATH_GPT_contrapositive_example_l1599_159951

theorem contrapositive_example (x : ℝ) : (x > 2 → x^2 > 4) → (x^2 ≤ 4 → x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_example_l1599_159951


namespace NUMINAMATH_GPT_no_overlapping_sale_days_l1599_159994

def bookstore_sale_days (d : ℕ) : Prop :=
  d % 4 = 0 ∧ 1 ≤ d ∧ d ≤ 31

def shoe_store_sale_days (d : ℕ) : Prop :=
  ∃ k : ℕ, d = 2 + 8 * k ∧ 1 ≤ d ∧ d ≤ 31

theorem no_overlapping_sale_days : 
  ∀ d : ℕ, bookstore_sale_days d → ¬ shoe_store_sale_days d :=
by
  intros d h1 h2
  sorry

end NUMINAMATH_GPT_no_overlapping_sale_days_l1599_159994


namespace NUMINAMATH_GPT_count_integer_triangles_with_perimeter_12_l1599_159907

theorem count_integer_triangles_with_perimeter_12 : 
  ∃! (sides : ℕ × ℕ × ℕ), sides.1 + sides.2.1 + sides.2.2 = 12 ∧ sides.1 + sides.2.1 > sides.2.2 ∧ sides.1 + sides.2.2 > sides.2.1 ∧ sides.2.1 + sides.2.2 > sides.1 ∧
  (sides = (2, 5, 5) ∨ sides = (3, 4, 5) ∨ sides = (4, 4, 4)) :=
by 
  exists 3
  sorry

end NUMINAMATH_GPT_count_integer_triangles_with_perimeter_12_l1599_159907


namespace NUMINAMATH_GPT_problem_l1599_159930

theorem problem (k : ℕ) (h1 : 30^k ∣ 929260) : 3^k - k^3 = 2 :=
sorry

end NUMINAMATH_GPT_problem_l1599_159930


namespace NUMINAMATH_GPT_quadrilateral_area_BEIH_l1599_159905

-- Define the necessary points in the problem
structure Point :=
(x : ℚ)
(y : ℚ)

-- Definitions of given points and midpoints
def B : Point := ⟨0, 0⟩
def E : Point := ⟨0, 1.5⟩
def F : Point := ⟨1.5, 0⟩

-- Definitions of line equations from points
def line_DE (p : Point) : Prop := p.y = - (1 / 2) * p.x + 1.5
def line_AF (p : Point) : Prop := p.y = -2 * p.x + 3

-- Intersection points
def I : Point := ⟨3 / 5, 9 / 5⟩
def H : Point := ⟨3 / 4, 3 / 4⟩

-- Function to calculate the area using the Shoelace Theorem
def shoelace_area (a b c d : Point) : ℚ :=
  (1 / 2) * ((a.x * b.y + b.x * c.y + c.x * d.y + d.x * a.y) - (a.y * b.x + b.y * c.x + c.y * d.x + d.y * a.x))

-- The proof statement
theorem quadrilateral_area_BEIH :
  shoelace_area B E I H = 9 / 16 :=
sorry

end NUMINAMATH_GPT_quadrilateral_area_BEIH_l1599_159905


namespace NUMINAMATH_GPT_unique_integer_triplet_solution_l1599_159947

theorem unique_integer_triplet_solution (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : 
    (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end NUMINAMATH_GPT_unique_integer_triplet_solution_l1599_159947


namespace NUMINAMATH_GPT_vectors_parallel_l1599_159945

theorem vectors_parallel (m n : ℝ) (k : ℝ) (h1 : 2 = k * 1) (h2 : -1 = k * m) (h3 : 2 = k * n) : 
  m + n = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_vectors_parallel_l1599_159945


namespace NUMINAMATH_GPT_jack_jill_same_speed_l1599_159966

theorem jack_jill_same_speed (x : ℝ) (h : x^2 - 8*x - 10 = 0) :
  (x^2 - 7*x - 18) = 2 := 
sorry

end NUMINAMATH_GPT_jack_jill_same_speed_l1599_159966


namespace NUMINAMATH_GPT_remainder_when_divided_by_4x_minus_8_l1599_159926

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

end NUMINAMATH_GPT_remainder_when_divided_by_4x_minus_8_l1599_159926


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_x_gt_4_l1599_159986

theorem necessary_but_not_sufficient_for_x_gt_4 (x : ℝ) : (x^2 > 16) → ¬ (x > 4) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_x_gt_4_l1599_159986


namespace NUMINAMATH_GPT_gcd_of_polynomials_l1599_159967

theorem gcd_of_polynomials (b : ℤ) (k : ℤ) (hk : k % 2 = 0) (hb : b = 1187 * k) : 
  Int.gcd (2 * b^2 + 31 * b + 67) (b + 15) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_gcd_of_polynomials_l1599_159967


namespace NUMINAMATH_GPT_abs_expression_eq_five_l1599_159943

theorem abs_expression_eq_five : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry -- proof omitted

end NUMINAMATH_GPT_abs_expression_eq_five_l1599_159943


namespace NUMINAMATH_GPT_solve_system_of_equations_l1599_159960

theorem solve_system_of_equations (n : ℕ) (hn : n ≥ 3) (x : ℕ → ℝ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    x i ^ 3 = (x ((i % n) + 1) + x ((i % n) + 2) + 1)) →
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    (x i = -1 ∨ x i = (1 + Real.sqrt 5) / 2 ∨ x i = (1 - Real.sqrt 5) / 2)) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1599_159960


namespace NUMINAMATH_GPT_train_cross_time_l1599_159932

-- Definitions from the conditions
def length_of_train : ℤ := 600
def speed_of_man_kmh : ℤ := 2
def speed_of_train_kmh : ℤ := 56

-- Conversion factors and speed conversion
def kmh_to_mph_factor : ℤ := 1000 / 3600 -- 1 km/hr = 0.27778 m/s approximately

def speed_of_man_ms : ℤ := speed_of_man_kmh * kmh_to_mph_factor -- Convert speed of man to m/s
def speed_of_train_ms : ℤ := speed_of_train_kmh * kmh_to_mph_factor -- Convert speed of train to m/s

-- Calculating relative speed
def relative_speed_ms : ℤ := speed_of_train_ms - speed_of_man_ms

-- Calculating the time taken to cross
def time_to_cross : ℤ := length_of_train / relative_speed_ms 

-- The theorem to prove
theorem train_cross_time : time_to_cross = 40 := 
by sorry

end NUMINAMATH_GPT_train_cross_time_l1599_159932


namespace NUMINAMATH_GPT_height_of_highest_wave_l1599_159923

theorem height_of_highest_wave 
  (h_austin : ℝ) -- Austin's height
  (h_high : ℝ) -- Highest wave's height
  (h_short : ℝ) -- Shortest wave's height 
  (height_relation1 : h_high = 4 * h_austin + 2)
  (height_relation2 : h_short = h_austin + 4)
  (surfboard : ℝ) (surfboard_len : surfboard = 7)
  (short_wave_len : h_short = surfboard + 3) :
  h_high = 26 :=
by
  -- Define local variables with the values from given conditions
  let austin_height := 6        -- as per calculation: 10 - 4 = 6
  let highest_wave_height := 26 -- as per calculation: (6 * 4) + 2 = 26
  sorry

end NUMINAMATH_GPT_height_of_highest_wave_l1599_159923


namespace NUMINAMATH_GPT_trajectory_of_midpoint_l1599_159977

-- Definitions based on the conditions identified in the problem
variables {x y x1 y1 : ℝ}

-- Condition that point P is on the curve y = 2x^2 + 1
def point_on_curve (x1 y1 : ℝ) : Prop :=
  y1 = 2 * x1^2 + 1

-- Definition of the midpoint M conditions
def midpoint_def (x y x1 y1 : ℝ) : Prop :=
  x = (x1 + 0) / 2 ∧ y = (y1 - 1) / 2

-- Final theorem statement to be proved
theorem trajectory_of_midpoint (x y x1 y1 : ℝ) :
  point_on_curve x1 y1 → midpoint_def x y x1 y1 → y = 4 * x^2 :=
sorry

end NUMINAMATH_GPT_trajectory_of_midpoint_l1599_159977


namespace NUMINAMATH_GPT_ben_apples_difference_l1599_159918

theorem ben_apples_difference (B P T : ℕ) (h1 : P = 40) (h2 : T = 18) (h3 : (3 / 8) * B = T) :
  B - P = 8 :=
sorry

end NUMINAMATH_GPT_ben_apples_difference_l1599_159918


namespace NUMINAMATH_GPT_exponential_inequality_l1599_159909

theorem exponential_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (Real.exp a * Real.exp c > Real.exp b * Real.exp d) :=
by sorry

end NUMINAMATH_GPT_exponential_inequality_l1599_159909


namespace NUMINAMATH_GPT_problem_statement_l1599_159975

noncomputable def P1 (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def P2 (β : ℝ) : ℝ × ℝ := (Real.cos β, -Real.sin β)
noncomputable def P3 (α β : ℝ) : ℝ × ℝ := (Real.cos (α + β), Real.sin (α + β))
noncomputable def A : ℝ × ℝ := (1, 0)

theorem problem_statement (α β : ℝ) :
  (Prod.fst (P1 α))^2 + (Prod.snd (P1 α))^2 = 1 ∧
  (Prod.fst (P2 β))^2 + (Prod.snd (P2 β))^2 = 1 ∧
  (Prod.fst (P1 α) * Prod.fst (P2 β) + Prod.snd (P1 α) * Prod.snd (P2 β)) = Real.cos (α + β) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1599_159975


namespace NUMINAMATH_GPT_solve_quadratic_solve_inequalities_l1599_159917
open Classical

-- Define the equation for Part 1
theorem solve_quadratic (x : ℝ) : x^2 - 6 * x + 5 = 0 → (x = 1 ∨ x = 5) :=
by
  sorry

-- Define the inequalities for Part 2
theorem solve_inequalities (x : ℝ) : (x + 3 > 0) ∧ (2 * (x - 1) < 4) → (-3 < x ∧ x < 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_solve_inequalities_l1599_159917


namespace NUMINAMATH_GPT_fraction_spent_on_food_l1599_159952

theorem fraction_spent_on_food (r c f : ℝ) (l s : ℝ)
  (hr : r = 1/10)
  (hc : c = 3/5)
  (hl : l = 16000)
  (hs : s = 160000)
  (heq : f * s + r * s + c * s + l = s) :
  f = 1/5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_spent_on_food_l1599_159952


namespace NUMINAMATH_GPT_winner_collected_l1599_159940

variable (M : ℕ)
variable (last_year_rate this_year_rate : ℝ)
variable (extra_miles : ℕ)
variable (money_collected_last_year money_collected_this_year : ℝ)

axiom rate_last_year : last_year_rate = 4
axiom rate_this_year : this_year_rate = 2.75
axiom extra_miles_eq : extra_miles = 5

noncomputable def money_eq (M : ℕ) : ℝ :=
  last_year_rate * M

theorem winner_collected :
  ∃ M : ℕ, money_eq M = 44 :=
by
  sorry

end NUMINAMATH_GPT_winner_collected_l1599_159940


namespace NUMINAMATH_GPT_find_shift_b_l1599_159910

-- Define the periodic function f
variable (f : ℝ → ℝ)
-- Define the condition on f
axiom f_periodic : ∀ x, f (x - 30) = f x

-- The theorem we want to prove
theorem find_shift_b : ∃ b > 0, (∀ x, f ((x - b) / 3) = f (x / 3)) ∧ b = 90 := 
by
  sorry

end NUMINAMATH_GPT_find_shift_b_l1599_159910


namespace NUMINAMATH_GPT_total_water_bottles_needed_l1599_159925

def number_of_people : ℕ := 4
def travel_time_one_way : ℕ := 8
def number_of_way : ℕ := 2
def water_consumption_per_hour : ℚ := 1 / 2

theorem total_water_bottles_needed : (number_of_people * (travel_time_one_way * number_of_way) * water_consumption_per_hour) = 32 := by
  sorry

end NUMINAMATH_GPT_total_water_bottles_needed_l1599_159925


namespace NUMINAMATH_GPT_diana_apollo_probability_l1599_159988

theorem diana_apollo_probability :
  let outcomes := (6 * 6)
  let successful := (5 + 4 + 3 + 2 + 1)
  (successful / outcomes) = 5 / 12 := sorry

end NUMINAMATH_GPT_diana_apollo_probability_l1599_159988


namespace NUMINAMATH_GPT_ratio_brown_eyes_l1599_159927

theorem ratio_brown_eyes (total_people : ℕ) (blue_eyes : ℕ) (black_eyes : ℕ) (green_eyes : ℕ) (brown_eyes : ℕ) 
    (h1 : total_people = 100) 
    (h2 : blue_eyes = 19) 
    (h3 : black_eyes = total_people / 4) 
    (h4 : green_eyes = 6) 
    (h5 : brown_eyes = total_people - (blue_eyes + black_eyes + green_eyes)) : 
    brown_eyes / total_people = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_brown_eyes_l1599_159927


namespace NUMINAMATH_GPT_no_partition_exists_l1599_159979

noncomputable section

open Set

def partition_N (A B C : Set ℕ) : Prop := 
  A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧  -- Non-empty sets
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧  -- Disjoint sets
  A ∪ B ∪ C = univ ∧  -- Covers the whole ℕ
  (∀ a ∈ A, ∀ b ∈ B, a + b + 2008 ∈ C) ∧
  (∀ b ∈ B, ∀ c ∈ C, b + c + 2008 ∈ A) ∧
  (∀ c ∈ C, ∀ a ∈ A, c + a + 2008 ∈ B)

theorem no_partition_exists : ¬ ∃ (A B C : Set ℕ), partition_N A B C :=
by
  sorry

end NUMINAMATH_GPT_no_partition_exists_l1599_159979


namespace NUMINAMATH_GPT_emery_family_trip_l1599_159950

theorem emery_family_trip 
  (first_part_distance : ℕ) (first_part_time : ℕ) (total_time : ℕ) (speed : ℕ) (second_part_time : ℕ) :
  first_part_distance = 100 ∧ first_part_time = 1 ∧ total_time = 4 ∧ speed = 100 ∧ second_part_time = 3 →
  second_part_time * speed = 300 :=
by 
  sorry

end NUMINAMATH_GPT_emery_family_trip_l1599_159950


namespace NUMINAMATH_GPT_number_of_towers_l1599_159959

noncomputable def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def multinomial (n : ℕ) (k1 k2 k3 : ℕ) : ℕ :=
  factorial n / (factorial k1 * factorial k2 * factorial k3)

theorem number_of_towers :
  (multinomial 10 3 3 4 = 4200) :=
by
  sorry

end NUMINAMATH_GPT_number_of_towers_l1599_159959


namespace NUMINAMATH_GPT_number_of_bracelets_l1599_159995

-- Define the conditions as constants
def metal_beads_nancy := 40
def pearl_beads_nancy := 60
def crystal_beads_rose := 20
def stone_beads_rose := 40
def beads_per_bracelet := 2

-- Define the number of sets each person can make
def sets_of_metal_beads := metal_beads_nancy / beads_per_bracelet
def sets_of_pearl_beads := pearl_beads_nancy / beads_per_bracelet
def sets_of_crystal_beads := crystal_beads_rose / beads_per_bracelet
def sets_of_stone_beads := stone_beads_rose / beads_per_bracelet

-- Define the theorem to prove
theorem number_of_bracelets : min sets_of_metal_beads (min sets_of_pearl_beads (min sets_of_crystal_beads sets_of_stone_beads)) = 10 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_bracelets_l1599_159995


namespace NUMINAMATH_GPT_ratios_of_PQR_and_XYZ_l1599_159916

-- Define triangle sides
def sides_PQR : ℕ × ℕ × ℕ := (7, 24, 25)
def sides_XYZ : ℕ × ℕ × ℕ := (9, 40, 41)

-- Perimeter calculation functions
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Area calculation functions for right triangles
def area (a b : ℕ) : ℕ := (a * b) / 2

-- Required proof statement
theorem ratios_of_PQR_and_XYZ :
  let (a₁, b₁, c₁) := sides_PQR
  let (a₂, b₂, c₂) := sides_XYZ
  area a₁ b₁ * 15 = 7 * area a₂ b₂ ∧ perimeter a₁ b₁ c₁ * 45 = 28 * perimeter a₂ b₂ c₂ :=
sorry

end NUMINAMATH_GPT_ratios_of_PQR_and_XYZ_l1599_159916


namespace NUMINAMATH_GPT_roger_toys_l1599_159933

theorem roger_toys (initial_money spent_money toy_cost remaining_money toys : ℕ) 
  (h1 : initial_money = 63) 
  (h2 : spent_money = 48) 
  (h3 : toy_cost = 3) 
  (h4 : remaining_money = initial_money - spent_money) 
  (h5 : toys = remaining_money / toy_cost) : 
  toys = 5 := 
by 
  sorry

end NUMINAMATH_GPT_roger_toys_l1599_159933


namespace NUMINAMATH_GPT_evaluate_expression_l1599_159980

theorem evaluate_expression : (527 * 527 - 526 * 528) = 1 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1599_159980


namespace NUMINAMATH_GPT_probability_three_specific_cards_l1599_159924

theorem probability_three_specific_cards :
  let deck_size := 52
  let diamonds := 13
  let spades := 13
  let hearts := 13
  let p1 := diamonds / deck_size
  let p2 := spades / (deck_size - 1)
  let p3 := hearts / (deck_size - 2)
  p1 * p2 * p3 = 169 / 5100 :=
by
  sorry

end NUMINAMATH_GPT_probability_three_specific_cards_l1599_159924


namespace NUMINAMATH_GPT_length_of_AE_l1599_159961

noncomputable def AE_calculation (AB AC AD : ℝ) (h : ℝ) (AE : ℝ) : Prop :=
  AB = 3.6 ∧ AC = 3.6 ∧ AD = 1.2 ∧ 
  (0.5 * AC * h = 0.5 * AE * (1/3) * h) →
  AE = 10.8

theorem length_of_AE {h : ℝ} : AE_calculation 3.6 3.6 1.2 h 10.8 :=
sorry

end NUMINAMATH_GPT_length_of_AE_l1599_159961


namespace NUMINAMATH_GPT_zero_points_product_l1599_159921

noncomputable def f (a x : ℝ) : ℝ := abs (Real.log x / Real.log a) - (1 / 2) ^ x

theorem zero_points_product (a x1 x2 : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1)
  (hx1_zero : f a x1 = 0) (hx2_zero : f a x2 = 0) : 0 < x1 * x2 ∧ x1 * x2 < 1 :=
by
  sorry

end NUMINAMATH_GPT_zero_points_product_l1599_159921


namespace NUMINAMATH_GPT_identify_counterfeit_bag_l1599_159957

-- Definitions based on problem conditions
def num_bags := 10
def genuine_weight := 10
def counterfeit_weight := 11
def expected_total_weight := genuine_weight * ((num_bags * (num_bags + 1)) / 2 : ℕ)

-- Lean theorem for the above problem
theorem identify_counterfeit_bag (W : ℕ) (Δ := W - expected_total_weight) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ num_bags ∧ Δ = i :=
by sorry

end NUMINAMATH_GPT_identify_counterfeit_bag_l1599_159957


namespace NUMINAMATH_GPT_coefficient_of_x_l1599_159928

theorem coefficient_of_x :
  let expr := (5 * (x - 6)) + (6 * (9 - 3 * x ^ 2 + 3 * x)) - (9 * (5 * x - 4))
  (expr : ℝ) → 
  let expr' := 5 * x - 30 + 54 - 18 * x ^ 2 + 18 * x - 45 * x + 36
  (expr' : ℝ) → 
  let coeff_x := 5 + 18 - 45
  coeff_x = -22 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_x_l1599_159928


namespace NUMINAMATH_GPT_sum_abcd_l1599_159989

variable (a b c d x : ℝ)

axiom eq1 : a + 2 = x
axiom eq2 : b + 3 = x
axiom eq3 : c + 4 = x
axiom eq4 : d + 5 = x
axiom eq5 : a + b + c + d + 10 = x

theorem sum_abcd : a + b + c + d = -26 / 3 :=
by
  -- We state the condition given in the problem
  sorry

end NUMINAMATH_GPT_sum_abcd_l1599_159989


namespace NUMINAMATH_GPT_height_percentage_l1599_159912

theorem height_percentage (a b c : ℝ) 
  (h1 : a = 0.6 * b) 
  (h2 : c = 1.25 * a) : 
  (b - a) / a * 100 = 66.67 ∧ (c - a) / a * 100 = 25 := 
by 
  sorry

end NUMINAMATH_GPT_height_percentage_l1599_159912


namespace NUMINAMATH_GPT_cone_lateral_surface_area_ratio_l1599_159993

theorem cone_lateral_surface_area_ratio (r l S_lateral S_base : ℝ) (h1 : l = 3 * r)
  (h2 : S_lateral = π * r * l) (h3 : S_base = π * r^2) :
  S_lateral / S_base = 3 :=
by
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_ratio_l1599_159993


namespace NUMINAMATH_GPT_ratio_expression_value_l1599_159969

theorem ratio_expression_value (x y : ℝ) (h : x ≠ 0) (h' : y ≠ 0) (h_eq : x^2 - y^2 = x + y) : 
  x / y + y / x = 2 + 1 / (y^2 + y) :=
by
  sorry

end NUMINAMATH_GPT_ratio_expression_value_l1599_159969


namespace NUMINAMATH_GPT_cost_of_horse_l1599_159956

theorem cost_of_horse (H C : ℝ) 
  (h1 : 4 * H + 9 * C = 13400)
  (h2 : 0.4 * H + 1.8 * C = 1880) :
  H = 2000 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_horse_l1599_159956


namespace NUMINAMATH_GPT_travel_time_difference_l1599_159948

theorem travel_time_difference :
  (160 / 40) - (280 / 40) = 3 := by
  sorry

end NUMINAMATH_GPT_travel_time_difference_l1599_159948


namespace NUMINAMATH_GPT_total_cost_of_water_l1599_159972

-- Define conditions in Lean 4
def cost_per_liter : ℕ := 1
def liters_per_bottle : ℕ := 2
def number_of_bottles : ℕ := 6

-- Define the theorem to prove the total cost
theorem total_cost_of_water : (number_of_bottles * (liters_per_bottle * cost_per_liter)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_water_l1599_159972


namespace NUMINAMATH_GPT_line_through_fixed_point_fixed_points_with_constant_slope_l1599_159985

-- Point structure definition
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define curves C1 and C2
def curve_C1 (p : Point) : Prop :=
  p.x^2 + (p.y - 1/4)^2 = 1 ∧ p.y ≥ 1/4

def curve_C2 (p : Point) : Prop :=
  p.x^2 = 8 * p.y - 1 ∧ abs p.x ≥ 1

-- Line passing through fixed point for given perpendicularity condition
theorem line_through_fixed_point (A B M : Point) (l : ℝ → ℝ → Prop) :
  curve_C2 A → curve_C2 B →
  (∃ k b, ∀ x y, l x y ↔ y = k * x + b) →
  (M = ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩) →
  ((M.x = A.x ∧ M.y = (A.y + B.y) / 2) → A.x * B.x = -16) →
  ∀ x y, l x y → y = (17 / 8) := sorry

-- Existence of two fixed points on y-axis with constant slope product
theorem fixed_points_with_constant_slope (P T1 T2 M : Point) (l : ℝ → ℝ → Prop) :
  curve_C1 P →
  (T1 = ⟨0, -1⟩) →
  (T2 = ⟨0, 1⟩) →
  l P.x P.y →
  (∃ k b, ∀ x y, l x y ↔ y = k * x + b) →
  (M.y^2 - (M.x^2 / 16) = 1) →
  (M.x ≠ 0) →
  ((M.y + 1) / M.x) * ((M.y - 1) / M.x) = (1 / 16) := sorry

end NUMINAMATH_GPT_line_through_fixed_point_fixed_points_with_constant_slope_l1599_159985


namespace NUMINAMATH_GPT_common_fraction_proof_l1599_159935

def expr_as_common_fraction : Prop :=
  let numerator := (3 / 6) + (4 / 5)
  let denominator := (5 / 12) + (1 / 4)
  (numerator / denominator) = (39 / 20)

theorem common_fraction_proof : expr_as_common_fraction :=
by
  sorry

end NUMINAMATH_GPT_common_fraction_proof_l1599_159935


namespace NUMINAMATH_GPT_least_ab_value_l1599_159919

theorem least_ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h : (1 : ℚ)/a + (1 : ℚ)/(3 * b) = 1 / 6) : a * b = 98 :=
by
  sorry

end NUMINAMATH_GPT_least_ab_value_l1599_159919


namespace NUMINAMATH_GPT_amount_after_two_years_l1599_159942

noncomputable def amountAfterYears (presentValue : ℝ) (rate : ℝ) (n : ℕ) : ℝ :=
  presentValue * (1 + rate) ^ n

theorem amount_after_two_years 
  (presentValue : ℝ := 62000) 
  (rate : ℝ := 1 / 8) 
  (n : ℕ := 2) : 
  amountAfterYears presentValue rate n = 78468.75 := 
  sorry

end NUMINAMATH_GPT_amount_after_two_years_l1599_159942


namespace NUMINAMATH_GPT_linda_total_profit_is_50_l1599_159914

def total_loaves : ℕ := 60
def loaves_sold_morning (total_loaves : ℕ) : ℕ := total_loaves / 3
def loaves_sold_afternoon (loaves_left_morning : ℕ) : ℕ := loaves_left_morning / 2
def loaves_sold_evening (loaves_left_afternoon : ℕ) : ℕ := loaves_left_afternoon

def price_per_loaf_morning : ℕ := 3
def price_per_loaf_afternoon : ℕ := 150 / 100 -- Representing $1.50 as 150 cents to use integer arithmetic
def price_per_loaf_evening : ℕ := 1

def cost_per_loaf : ℕ := 1

def calculate_profit (total_loaves loaves_sold_morning loaves_sold_afternoon loaves_sold_evening price_per_loaf_morning price_per_loaf_afternoon price_per_loaf_evening cost_per_loaf : ℕ) : ℕ := 
  let revenue_morning := loaves_sold_morning * price_per_loaf_morning
  let loaves_left_morning := total_loaves - loaves_sold_morning
  let revenue_afternoon := loaves_sold_afternoon * price_per_loaf_afternoon
  let loaves_left_afternoon := loaves_left_morning - loaves_sold_afternoon
  let revenue_evening := loaves_sold_evening * price_per_loaf_evening
  let total_revenue := revenue_morning + revenue_afternoon + revenue_evening
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

theorem linda_total_profit_is_50 : calculate_profit total_loaves (loaves_sold_morning total_loaves) (loaves_sold_afternoon (total_loaves - loaves_sold_morning total_loaves)) (total_loaves - loaves_sold_morning total_loaves - loaves_sold_afternoon (total_loaves - loaves_sold_morning total_loaves)) price_per_loaf_morning price_per_loaf_afternoon price_per_loaf_evening cost_per_loaf = 50 := 
  by 
    sorry

end NUMINAMATH_GPT_linda_total_profit_is_50_l1599_159914


namespace NUMINAMATH_GPT_f_divisible_by_8_l1599_159922

-- Define the function f
def f (n : ℕ) : ℕ := 5^n + 2 * 3^(n-1) + 1

-- Theorem statement
theorem f_divisible_by_8 (n : ℕ) (hn : n > 0) : 8 ∣ f n := sorry

end NUMINAMATH_GPT_f_divisible_by_8_l1599_159922


namespace NUMINAMATH_GPT_shorter_leg_of_right_triangle_l1599_159934

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end NUMINAMATH_GPT_shorter_leg_of_right_triangle_l1599_159934


namespace NUMINAMATH_GPT_simone_fraction_per_day_l1599_159949

theorem simone_fraction_per_day 
  (x : ℚ) -- Define the fraction of an apple Simone ate each day as x.
  (h1 : 16 * x + 15 * (1/3) = 13) -- Condition: Simone and Lauri together ate 13 apples.
  : x = 1/2 := 
 by 
  sorry

end NUMINAMATH_GPT_simone_fraction_per_day_l1599_159949


namespace NUMINAMATH_GPT_total_students_in_lunchroom_l1599_159990

theorem total_students_in_lunchroom :
  (34 * 6) + 15 = 219 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_lunchroom_l1599_159990


namespace NUMINAMATH_GPT_inequality_range_m_l1599_159981

theorem inequality_range_m:
  (∀ x ∈ Set.Icc (Real.sqrt 2) 4, (5 / 2) * x^2 ≥ m * (x - 1)) → m ≤ 10 :=
by 
  intros h 
  sorry

end NUMINAMATH_GPT_inequality_range_m_l1599_159981


namespace NUMINAMATH_GPT_hamburgers_served_l1599_159999

def hamburgers_made : ℕ := 9
def hamburgers_leftover : ℕ := 6

theorem hamburgers_served : ∀ (total : ℕ) (left : ℕ), total = hamburgers_made → left = hamburgers_leftover → total - left = 3 := 
by
  intros total left h_total h_left
  rw [h_total, h_left]
  rfl

end NUMINAMATH_GPT_hamburgers_served_l1599_159999


namespace NUMINAMATH_GPT_find_f_3_l1599_159983

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ :=
- x^2 + b * x + c

theorem find_f_3 (b c : ℝ) (h1 : quadratic_function b c 2 + quadratic_function b c 4 = 12138)
                       (h2 : 3*b + c = 6079) :
  quadratic_function b c 3 = 6070 := 
by
  sorry

end NUMINAMATH_GPT_find_f_3_l1599_159983


namespace NUMINAMATH_GPT_remainder_of_a_squared_l1599_159954

theorem remainder_of_a_squared (n : ℕ) (a : ℤ) (h : a % n * a % n % n = 1) : (a * a) % n = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_of_a_squared_l1599_159954


namespace NUMINAMATH_GPT_distance_swam_against_current_l1599_159953

def swimming_speed_in_still_water : ℝ := 4
def speed_of_current : ℝ := 2
def time_taken_against_current : ℝ := 5

theorem distance_swam_against_current : ∀ distance : ℝ,
  (distance = (swimming_speed_in_still_water - speed_of_current) * time_taken_against_current) → distance = 10 :=
by
  intros distance h
  sorry

end NUMINAMATH_GPT_distance_swam_against_current_l1599_159953


namespace NUMINAMATH_GPT_parallel_lines_condition_l1599_159903

theorem parallel_lines_condition (a : ℝ) : 
  (a = -2) ↔ (∀ x y : ℝ, ax + 2 * y = 0 → y = 1 + x) := 
sorry

end NUMINAMATH_GPT_parallel_lines_condition_l1599_159903


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1599_159974

variables (a b : ℝ)

theorem value_of_a_plus_b (ha : abs a = 1) (hb : abs b = 4) (hab : a * b < 0) : a + b = 3 ∨ a + b = -3 := by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1599_159974


namespace NUMINAMATH_GPT_pieces_to_same_point_l1599_159946

theorem pieces_to_same_point :
  ∀ (x y z : ℤ), (∃ (final_pos : ℤ), (x = final_pos ∧ y = final_pos ∧ z = final_pos)) ↔ 
  (x, y, z) = (1, 2009, 2010) ∨ 
  (x, y, z) = (0, 2009, 2010) ∨ 
  (x, y, z) = (2, 2009, 2010) ∨ 
  (x, y, z) = (3, 2009, 2010) := 
by {
  sorry
}

end NUMINAMATH_GPT_pieces_to_same_point_l1599_159946


namespace NUMINAMATH_GPT_apples_fell_out_l1599_159901

theorem apples_fell_out (initial_apples stolen_apples remaining_apples : ℕ) 
  (h₁ : initial_apples = 79) 
  (h₂ : stolen_apples = 45) 
  (h₃ : remaining_apples = 8) 
  : initial_apples - stolen_apples - remaining_apples = 26 := by
  sorry

end NUMINAMATH_GPT_apples_fell_out_l1599_159901


namespace NUMINAMATH_GPT_milk_butterfat_mixture_l1599_159941

theorem milk_butterfat_mixture (x gallons_50 gall_10_perc final_gall mixture_perc: ℝ)
    (H1 : gall_10_perc = 24) 
    (H2 : mixture_perc = 0.20 * (x + gall_10_perc))
    (H3 : 0.50 * x + 0.10 * gall_10_perc = 0.20 * (x + gall_10_perc)) 
    (H4 : final_gall = 20) :
    x = 8 :=
sorry

end NUMINAMATH_GPT_milk_butterfat_mixture_l1599_159941


namespace NUMINAMATH_GPT_not_perfect_square_9n_squared_minus_9n_plus_9_l1599_159931

theorem not_perfect_square_9n_squared_minus_9n_plus_9
  (n : ℕ) (h : n > 1) : ¬ (∃ k : ℕ, 9 * n^2 - 9 * n + 9 = k * k) := sorry

end NUMINAMATH_GPT_not_perfect_square_9n_squared_minus_9n_plus_9_l1599_159931


namespace NUMINAMATH_GPT_crates_lost_l1599_159936

theorem crates_lost (total_crates : ℕ) (total_cost : ℕ) (desired_profit_percent : ℕ) 
(lost_crates remaining_crates : ℕ) (price_per_crate : ℕ) 
(h1 : total_crates = 10) (h2 : total_cost = 160) (h3 : desired_profit_percent = 25) 
(h4 : price_per_crate = 25) (h5 : remaining_crates = total_crates - lost_crates)
(h6 : price_per_crate * remaining_crates = total_cost + total_cost * desired_profit_percent / 100) :
  lost_crates = 2 :=
by
  sorry

end NUMINAMATH_GPT_crates_lost_l1599_159936


namespace NUMINAMATH_GPT_remainder_of_P_div_D_is_25158_l1599_159996

noncomputable def P (x : ℝ) := 4 * x^8 - 2 * x^6 + 5 * x^4 - x^3 + 3 * x - 15
def D (x : ℝ) := 2 * x - 6

theorem remainder_of_P_div_D_is_25158 : P 3 = 25158 := by
  sorry

end NUMINAMATH_GPT_remainder_of_P_div_D_is_25158_l1599_159996


namespace NUMINAMATH_GPT_cylinder_volume_expansion_l1599_159998

theorem cylinder_volume_expansion (r h : ℝ) :
  (π * (2 * r)^2 * h) = 4 * (π * r^2 * h) :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_expansion_l1599_159998


namespace NUMINAMATH_GPT_percentage_of_z_equals_39_percent_of_y_l1599_159920

theorem percentage_of_z_equals_39_percent_of_y
    (x y z : ℝ)
    (h1 : y = 0.75 * x)
    (h2 : z = 0.65 * x)
    (P : ℝ)
    (h3 : (P / 100) * z = 0.39 * y) :
    P = 45 :=
by sorry

end NUMINAMATH_GPT_percentage_of_z_equals_39_percent_of_y_l1599_159920


namespace NUMINAMATH_GPT_range_of_b_l1599_159906

open Real

theorem range_of_b (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → abs (y - (x + b)) = 1) ↔ -sqrt 2 < b ∧ b < sqrt 2 := 
by sorry

end NUMINAMATH_GPT_range_of_b_l1599_159906


namespace NUMINAMATH_GPT_sum_of_numbers_l1599_159970

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 12) (h2 : (1 / x) = 3 * (1 / y)) :
  x + y = 8 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l1599_159970


namespace NUMINAMATH_GPT_profit_percentage_no_initial_discount_l1599_159997

theorem profit_percentage_no_initial_discount
  (CP : ℝ := 100)
  (bulk_discount : ℝ := 0.02)
  (sales_tax : ℝ := 0.065)
  (no_discount_price : ℝ := CP - CP * bulk_discount)
  (selling_price : ℝ := no_discount_price + no_discount_price * sales_tax)
  (profit : ℝ := selling_price - CP) :
  (profit / CP) * 100 = 4.37 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_profit_percentage_no_initial_discount_l1599_159997
