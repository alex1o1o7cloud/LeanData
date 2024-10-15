import Mathlib

namespace NUMINAMATH_GPT_sam_initial_balloons_l474_47442

theorem sam_initial_balloons:
  ∀ (S : ℕ), (S - 10 + 16 = 52) → S = 46 :=
by
  sorry

end NUMINAMATH_GPT_sam_initial_balloons_l474_47442


namespace NUMINAMATH_GPT_wine_cost_increase_l474_47473

noncomputable def additional_cost (initial_price : ℝ) (num_bottles : ℕ) (month1_rate : ℝ) (month2_tariff : ℝ) (month2_discount : ℝ) (month3_tariff : ℝ) (month3_rate : ℝ) : ℝ := 
  let price_month1 := initial_price * (1 + month1_rate) 
  let cost_month1 := num_bottles * price_month1
  let price_month2 := (initial_price * (1 + month2_tariff)) * (1 - month2_discount)
  let cost_month2 := num_bottles * price_month2
  let price_month3 := (initial_price * (1 + month3_tariff)) * (1 - month3_rate)
  let cost_month3 := num_bottles * price_month3
  (cost_month1 + cost_month2 + cost_month3) - (3 * num_bottles * initial_price)

theorem wine_cost_increase : 
  additional_cost 20 5 0.05 0.25 0.15 0.35 0.03 = 42.20 :=
by sorry

end NUMINAMATH_GPT_wine_cost_increase_l474_47473


namespace NUMINAMATH_GPT_greatest_divisor_arithmetic_sum_l474_47482

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end NUMINAMATH_GPT_greatest_divisor_arithmetic_sum_l474_47482


namespace NUMINAMATH_GPT_train_speed_l474_47498

/-- 
Train A leaves the station traveling at a certain speed v. 
Two hours later, Train B leaves the same station traveling in the same direction at 36 miles per hour. 
Train A was overtaken by Train B 360 miles from the station.
We need to prove that the speed of Train A was 30 miles per hour.
-/
theorem train_speed (v : ℕ) (t : ℕ) (h1 : 36 * (t - 2) = 360) (h2 : v * t = 360) : v = 30 :=
by 
  sorry

end NUMINAMATH_GPT_train_speed_l474_47498


namespace NUMINAMATH_GPT_chemical_transport_problem_l474_47423

theorem chemical_transport_problem :
  (∀ (w r : ℕ), r = w + 420 →
  (900 / r) = (600 / (10 * w)) →
  w = 30 ∧ r = 450) ∧ 
  (∀ (x : ℕ), x + 450 * 3 * 2 + 60 * x ≥ 3600 → x = 15) := by
  sorry

end NUMINAMATH_GPT_chemical_transport_problem_l474_47423


namespace NUMINAMATH_GPT_evaluate_expression_l474_47488

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l474_47488


namespace NUMINAMATH_GPT_candy_left_l474_47427

theorem candy_left (d : ℕ) (s : ℕ) (ate : ℕ) (h_d : d = 32) (h_s : s = 42) (h_ate : ate = 35) : d + s - ate = 39 :=
by
  -- d, s, and ate are given as natural numbers
  -- h_d, h_s, and h_ate are the provided conditions
  -- The goal is to prove d + s - ate = 39
  sorry

end NUMINAMATH_GPT_candy_left_l474_47427


namespace NUMINAMATH_GPT_wall_height_to_breadth_ratio_l474_47436

theorem wall_height_to_breadth_ratio :
  ∀ (b : ℝ) (h : ℝ) (l : ℝ),
  b = 0.4 → h = n * b → l = 8 * h → l * b * h = 12.8 →
  n = 5 :=
by
  intros b h l hb hh hl hv
  sorry

end NUMINAMATH_GPT_wall_height_to_breadth_ratio_l474_47436


namespace NUMINAMATH_GPT_determine_a_l474_47432

noncomputable def f (x a : ℝ) := -9 * x^2 - 6 * a * x + 2 * a - a^2

theorem determine_a (a : ℝ) 
  (h₁ : ∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f x a ≤ f 0 a)
  (h₂ : f 0 a = -3) :
  a = 2 + Real.sqrt 6 := 
sorry

end NUMINAMATH_GPT_determine_a_l474_47432


namespace NUMINAMATH_GPT_find_m_l474_47422

-- Define the vectors and the real number m
variables {Vec : Type*} [AddCommGroup Vec] [Module ℝ Vec]
variables (e1 e2 : Vec) (m : ℝ)

-- Define the collinearity condition and non-collinearity of the basis vectors.
def non_collinear (v1 v2 : Vec) : Prop := ¬(∃ (a : ℝ), v2 = a • v1)

def collinear (v1 v2 : Vec) : Prop := ∃ (a : ℝ), v2 = a • v1

-- Given conditions
axiom e1_e2_non_collinear : non_collinear e1 e2
axiom AB_eq : ∀ (m : ℝ), Vec
axiom CB_eq : Vec

theorem find_m (h : collinear (e1 + m • e2) (e1 - e2)) : m = -1 :=
sorry

end NUMINAMATH_GPT_find_m_l474_47422


namespace NUMINAMATH_GPT_trapezium_hole_perimeter_correct_l474_47460

variable (a b : ℝ)

def trapezium_hole_perimeter (a b : ℝ) : ℝ :=
  6 * a - 3 * b

theorem trapezium_hole_perimeter_correct (a b : ℝ) :
  trapezium_hole_perimeter a b = 6 * a - 3 * b :=
by
  sorry

end NUMINAMATH_GPT_trapezium_hole_perimeter_correct_l474_47460


namespace NUMINAMATH_GPT_sum_of_digits_B_l474_47492

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_B (n : ℕ) (h : n = 4444^4444) : digit_sum (digit_sum (digit_sum n)) = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_B_l474_47492


namespace NUMINAMATH_GPT_evaluate_expression_l474_47494

theorem evaluate_expression (a b c : ℤ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  ((a^2 + b + c)^2 - (a^2 - b - c)^2) = 108 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l474_47494


namespace NUMINAMATH_GPT_find_quad_function_l474_47459

-- Define the quadratic function with the given conditions
def quad_function (a b c : ℝ) (f : ℝ → ℝ) :=
  ∀ x, f x = a * x^2 + b * x + c

-- Define the values y(-2) = -3, y(-1) = -4, y(0) = -3, y(2) = 5
def given_points (f : ℝ → ℝ) :=
  f (-2) = -3 ∧ f (-1) = -4 ∧ f 0 = -3 ∧ f 2 = 5

-- Prove that y = x^2 + 2x - 3 satisfies the given points
theorem find_quad_function : ∃ f : ℝ → ℝ, (quad_function 1 2 (-3) f) ∧ (given_points f) :=
by
  sorry

end NUMINAMATH_GPT_find_quad_function_l474_47459


namespace NUMINAMATH_GPT_initial_bacteria_count_l474_47478

theorem initial_bacteria_count 
  (double_every_30_seconds : ∀ n : ℕ, n * 2^(240 / 30) = 262144) : 
  ∃ n : ℕ, n = 1024 :=
by
  -- Define the initial number of bacteria.
  let n := 262144 / (2^8)
  -- Assert that the initial number is 1024.
  use n
  -- To skip the proof.
  sorry

end NUMINAMATH_GPT_initial_bacteria_count_l474_47478


namespace NUMINAMATH_GPT_total_monkeys_l474_47448

theorem total_monkeys (x : ℕ) (h : (1 / 8 : ℝ) * x ^ 2 + 12 = x) : x = 48 :=
sorry

end NUMINAMATH_GPT_total_monkeys_l474_47448


namespace NUMINAMATH_GPT_max_piece_length_l474_47497

theorem max_piece_length (L1 L2 L3 L4 : ℕ) (hL1 : L1 = 48) (hL2 : L2 = 72) (hL3 : L3 = 120) (hL4 : L4 = 144) 
  (h_min_pieces : ∀ L k, L = 48 ∨ L = 72 ∨ L = 120 ∨ L = 144 → k > 0 → L / k ≥ 5) : 
  ∃ k, k = 8 ∧ ∀ L, (L = L1 ∨ L = L2 ∨ L = L3 ∨ L = L4) → L % k = 0 :=
by
  sorry

end NUMINAMATH_GPT_max_piece_length_l474_47497


namespace NUMINAMATH_GPT_dasha_ate_one_bowl_l474_47420

-- Define the quantities for Masha, Dasha, Glasha, and Natasha
variables (M D G N : ℕ)

-- Given conditions
def conditions : Prop :=
  (M + D + G + N = 16) ∧
  (G + N = 9) ∧
  (M > D) ∧
  (M > G) ∧
  (M > N)

-- The problem statement rewritten in Lean: Prove that given the conditions, Dasha ate 1 bowl.
theorem dasha_ate_one_bowl (h : conditions M D G N) : D = 1 :=
sorry

end NUMINAMATH_GPT_dasha_ate_one_bowl_l474_47420


namespace NUMINAMATH_GPT_fill_sacks_times_l474_47467

-- Define the capacities of the sacks
def father_sack_capacity : ℕ := 20
def senior_ranger_sack_capacity : ℕ := 30
def volunteer_sack_capacity : ℕ := 25
def number_of_volunteers : ℕ := 2

-- Total wood gathered
def total_wood_gathered : ℕ := 200

-- Statement of the proof problem
theorem fill_sacks_times : (total_wood_gathered / (father_sack_capacity + senior_ranger_sack_capacity + (number_of_volunteers * volunteer_sack_capacity))) = 2 := by
  sorry

end NUMINAMATH_GPT_fill_sacks_times_l474_47467


namespace NUMINAMATH_GPT_cos_4theta_l474_47452

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (4 * θ) = 17 / 81 := 
by 
  sorry

end NUMINAMATH_GPT_cos_4theta_l474_47452


namespace NUMINAMATH_GPT_number_of_possible_teams_l474_47410

-- Definitions for the conditions
def num_goalkeepers := 3
def num_defenders := 5
def num_midfielders := 5
def num_strikers := 5

-- The number of ways to choose x from y
def choose (y x : ℕ) : ℕ := Nat.factorial y / (Nat.factorial x * Nat.factorial (y - x))

-- Main proof problem statement
theorem number_of_possible_teams :
  (choose num_goalkeepers 1) *
  (choose num_strikers 2) *
  (choose num_midfielders 4) *
  (choose (num_defenders + (num_midfielders - 4)) 4) = 2250 := by
  sorry

end NUMINAMATH_GPT_number_of_possible_teams_l474_47410


namespace NUMINAMATH_GPT_cost_of_paving_l474_47463

-- declaring the definitions and the problem statement
def length_of_room := 5.5
def width_of_room := 4
def rate_per_sq_meter := 700

theorem cost_of_paving (length : ℝ) (width : ℝ) (rate : ℝ) : length = 5.5 → width = 4 → rate = 700 → (length * width * rate) = 15400 :=
by
  intros h_length h_width h_rate
  rw [h_length, h_width, h_rate]
  sorry

end NUMINAMATH_GPT_cost_of_paving_l474_47463


namespace NUMINAMATH_GPT_election_margin_of_victory_l474_47474

theorem election_margin_of_victory (T : ℕ) (H_winning_votes : T * 58 / 100 = 1044) :
  1044 - (T * 42 / 100) = 288 :=
by
  sorry

end NUMINAMATH_GPT_election_margin_of_victory_l474_47474


namespace NUMINAMATH_GPT_exists_zero_in_interval_minus3_minus2_l474_47496

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x - x

theorem exists_zero_in_interval_minus3_minus2 : 
  ∃ x ∈ Set.Icc (-3 : ℝ) (-2), f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_zero_in_interval_minus3_minus2_l474_47496


namespace NUMINAMATH_GPT_multiple_of_interest_rate_l474_47424

theorem multiple_of_interest_rate (P r : ℝ) (m : ℝ) 
  (h1 : P * r^2 = 40) 
  (h2 : P * m^2 * r^2 = 360) : 
  m = 3 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_interest_rate_l474_47424


namespace NUMINAMATH_GPT_solve_for_x_l474_47401

theorem solve_for_x (x : ℝ) : (5 + x) / (8 + x) = (2 + x) / (3 + x) → x = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l474_47401


namespace NUMINAMATH_GPT_pair_ab_l474_47479

def students_activities_ways (n_students n_activities : Nat) : Nat :=
  n_activities ^ n_students

def championships_outcomes (n_championships n_students : Nat) : Nat :=
  n_students ^ n_championships

theorem pair_ab (a b : Nat) :
  a = students_activities_ways 4 3 ∧ b = championships_outcomes 3 4 →
  (a, b) = (3^4, 4^3) := by
  sorry

end NUMINAMATH_GPT_pair_ab_l474_47479


namespace NUMINAMATH_GPT_find_f2_l474_47433

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end NUMINAMATH_GPT_find_f2_l474_47433


namespace NUMINAMATH_GPT_desired_line_equation_l474_47464

-- Define the center of the circle and the equation of the given line
def center : (ℝ × ℝ) := (-1, 0)
def line1 (x y : ℝ) : Prop := x + y = 0

-- Define the desired line passing through the center of the circle and perpendicular to line1
def line2 (x y : ℝ) : Prop := x + y + 1 = 0

-- The theorem stating that the desired line equation is x + y + 1 = 0
theorem desired_line_equation : ∀ (x y : ℝ),
  (center = (-1, 0)) → (∀ x y, line1 x y → line2 x y) :=
by
  sorry

end NUMINAMATH_GPT_desired_line_equation_l474_47464


namespace NUMINAMATH_GPT_product_of_fraction_l474_47458

theorem product_of_fraction (x : ℚ) (h : x = 17 / 999) : 17 * 999 = 16983 := by sorry

end NUMINAMATH_GPT_product_of_fraction_l474_47458


namespace NUMINAMATH_GPT_medium_as_decoy_and_rational_choice_l474_47491

/-- 
  Define the prices and sizes of the popcorn containers:
  Small: 50g for 200 rubles.
  Medium: 70g for 400 rubles.
  Large: 130g for 500 rubles.
-/
structure PopcornContainer where
  size : ℕ -- in grams
  price : ℕ -- in rubles

def small := PopcornContainer.mk 50 200
def medium := PopcornContainer.mk 70 400
def large := PopcornContainer.mk 130 500

/-- 
  The medium-sized popcorn container can be considered a decoy
  in the context of asymmetric dominance.
  Additionally, under certain budget constraints and preferences, 
  rational economic agents may find the medium-sized container optimal.
-/
theorem medium_as_decoy_and_rational_choice :
  (medium.price = 400 ∧ medium.size = 70) ∧ 
  (∃ (budget : ℕ) (pref : ℕ → ℕ → Prop), (budget ≥ medium.price ∧ 
    pref medium.size (budget - medium.price))) :=
by
  sorry

end NUMINAMATH_GPT_medium_as_decoy_and_rational_choice_l474_47491


namespace NUMINAMATH_GPT_fiona_pairs_l474_47447

theorem fiona_pairs :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 15 → 45 ≤ (n * (n - 1) / 2) ∧ (n * (n - 1) / 2) ≤ 105 :=
by
  intro n
  intro h
  have h₁ : n ≥ 10 := h.left
  have h₂ : n ≤ 15 := h.right
  sorry

end NUMINAMATH_GPT_fiona_pairs_l474_47447


namespace NUMINAMATH_GPT_solve_k_l474_47499

theorem solve_k (t s : ℤ) : (∃ k m, 8 * k + 4 = 7 * m ∧ k = -4 + 7 * t ∧ m = -4 + 8 * t) →
  (∃ k m, 12 * k - 8 = 7 * m ∧ k = 3 + 7 * s ∧ m = 4 + 12 * s) →
  7 * t - 4 = 7 * s + 3 →
  ∃ k, k = 3 + 7 * s :=
by
  sorry

end NUMINAMATH_GPT_solve_k_l474_47499


namespace NUMINAMATH_GPT_find_p_plus_q_l474_47435

/--
In \(\triangle{XYZ}\), \(XY = 12\), \(\angle{X} = 45^\circ\), and \(\angle{Y} = 60^\circ\).
Let \(G, E,\) and \(L\) be points on the line \(YZ\) such that \(XG \perp YZ\), 
\(\angle{XYE} = \angle{EYX}\), and \(YL = LY\). Point \(O\) is the midpoint of 
the segment \(GL\), and point \(Q\) is on ray \(XE\) such that \(QO \perp YZ\).
Prove that \(XQ^2 = \dfrac{81}{2}\) and thus \(p + q = 83\), where \(p\) and \(q\) 
are relatively prime positive integers.
-/
theorem find_p_plus_q :
  ∃ (p q : ℕ), gcd p q = 1 ∧ XQ^2 = 81 / 2 ∧ p + q = 83 :=
sorry

end NUMINAMATH_GPT_find_p_plus_q_l474_47435


namespace NUMINAMATH_GPT_cos_double_angle_l474_47404

variable (θ : Real)

theorem cos_double_angle (h : ∑' n, (Real.cos θ)^(2 * n) = 7) : Real.cos (2 * θ) = 5 / 7 := 
  by sorry

end NUMINAMATH_GPT_cos_double_angle_l474_47404


namespace NUMINAMATH_GPT_a_plus_b_values_l474_47456

theorem a_plus_b_values (a b : ℤ) (h1 : |a + 1| = 0) (h2 : b^2 = 9) :
  a + b = 2 ∨ a + b = -4 :=
by
  have ha : a = -1 := by sorry
  have hb1 : b = 3 ∨ b = -3 := by sorry
  cases hb1 with
  | inl b_pos =>
    left
    rw [ha, b_pos]
    exact sorry
  | inr b_neg =>
    right
    rw [ha, b_neg]
    exact sorry

end NUMINAMATH_GPT_a_plus_b_values_l474_47456


namespace NUMINAMATH_GPT_rabbits_in_cage_l474_47462

theorem rabbits_in_cage (rabbits_in_cage : ℕ) (rabbits_park : ℕ) : 
  rabbits_in_cage = 13 ∧ rabbits_park = 60 → (1/3 * rabbits_park - rabbits_in_cage) = 7 :=
by
  sorry

end NUMINAMATH_GPT_rabbits_in_cage_l474_47462


namespace NUMINAMATH_GPT_compare_trig_functions_l474_47455

theorem compare_trig_functions :
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c :=
by
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  sorry

end NUMINAMATH_GPT_compare_trig_functions_l474_47455


namespace NUMINAMATH_GPT_multiplicative_inverse_exists_and_is_correct_l474_47451

theorem multiplicative_inverse_exists_and_is_correct :
  ∃ N : ℤ, N > 0 ∧ (123456 * 171717) * N % 1000003 = 1 :=
sorry

end NUMINAMATH_GPT_multiplicative_inverse_exists_and_is_correct_l474_47451


namespace NUMINAMATH_GPT_total_rainfall_2004_l474_47400

theorem total_rainfall_2004 (average_rainfall_2003 : ℝ) (increase_percentage : ℝ) (months : ℝ) :
  average_rainfall_2003 = 36 →
  increase_percentage = 0.10 →
  months = 12 →
  (average_rainfall_2003 * (1 + increase_percentage) * months) = 475.2 :=
by
  -- The proof is left as an exercise
  sorry

end NUMINAMATH_GPT_total_rainfall_2004_l474_47400


namespace NUMINAMATH_GPT_chocolates_remaining_l474_47453

def chocolates := 24
def chocolates_first_day := 4
def chocolates_eaten_second_day := (2 * chocolates_first_day) - 3
def chocolates_eaten_third_day := chocolates_first_day - 2
def chocolates_eaten_fourth_day := chocolates_eaten_third_day - 1

theorem chocolates_remaining :
  chocolates - (chocolates_first_day + chocolates_eaten_second_day + chocolates_eaten_third_day + chocolates_eaten_fourth_day) = 12 := by
  sorry

end NUMINAMATH_GPT_chocolates_remaining_l474_47453


namespace NUMINAMATH_GPT_minimum_paper_toys_is_eight_l474_47408

noncomputable def minimum_paper_toys (s_boats: ℕ) (s_planes: ℕ) : ℕ :=
  s_boats * 8 + s_planes * 6

theorem minimum_paper_toys_is_eight :
  ∀ (s_boats s_planes : ℕ), s_boats >= 1 → minimum_paper_toys s_boats s_planes = 8 → s_planes = 0 :=
by
  intros s_boats s_planes h_boats h_eq
  have h1: s_boats * 8 + s_planes * 6 = 8 := h_eq
  sorry

end NUMINAMATH_GPT_minimum_paper_toys_is_eight_l474_47408


namespace NUMINAMATH_GPT_time_for_A_to_finish_race_l474_47468

-- Definitions based on the conditions
def race_distance : ℝ := 120
def B_time : ℝ := 45
def B_beaten_distance : ℝ := 24

-- Proof statement: We need to show that A's time is 56.25 seconds
theorem time_for_A_to_finish_race : ∃ (t : ℝ), t = 56.25 ∧ (120 / t = 96 / 45)
  := sorry

end NUMINAMATH_GPT_time_for_A_to_finish_race_l474_47468


namespace NUMINAMATH_GPT_largest_odd_digit_multiple_of_5_lt_10000_l474_47481

def is_odd_digit (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), is_odd_digit d

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem largest_odd_digit_multiple_of_5_lt_10000 :
  ∃ n, n < 10000 ∧ all_odd_digits n ∧ is_multiple_of_5 n ∧
        ∀ m, m < 10000 → all_odd_digits m → is_multiple_of_5 m → m ≤ n :=
  sorry

end NUMINAMATH_GPT_largest_odd_digit_multiple_of_5_lt_10000_l474_47481


namespace NUMINAMATH_GPT_touching_line_eq_l474_47445

theorem touching_line_eq (f : ℝ → ℝ) (f_def : ∀ x, f x = 3 * x^4 - 4 * x^3) : 
  ∃ l : ℝ → ℝ, (∀ x, l x = - (8 / 9) * x - (4 / 27)) ∧ 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (f x₁ = l x₁ ∧ f x₂ = l x₂) :=
by sorry

end NUMINAMATH_GPT_touching_line_eq_l474_47445


namespace NUMINAMATH_GPT_equations_solution_l474_47446

-- Definition of the conditions
def equation1 := ∀ x : ℝ, x^2 - 2 * x - 3 = 0 -> (x = 3 ∨ x = -1)
def equation2 := ∀ x : ℝ, x * (x - 2) + x - 2 = 0 -> (x = -1 ∨ x = 2)

-- The main statement combining both problems
theorem equations_solution :
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 -> (x = 3 ∨ x = -1)) ∧
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 -> (x = -1 ∨ x = 2)) := by
  sorry

end NUMINAMATH_GPT_equations_solution_l474_47446


namespace NUMINAMATH_GPT_face_value_amount_of_bill_l474_47426

def true_discount : ℚ := 45
def bankers_discount : ℚ := 54

theorem face_value_amount_of_bill : 
  ∃ (FV : ℚ), bankers_discount = true_discount + (true_discount * bankers_discount / FV) ∧ FV = 270 :=
by
  sorry

end NUMINAMATH_GPT_face_value_amount_of_bill_l474_47426


namespace NUMINAMATH_GPT_find_d_l474_47425

theorem find_d (d : ℚ) (h_floor : ∃ x : ℤ, x^2 + 5 * x - 36 = 0 ∧ x = ⌊d⌋)
  (h_frac: ∃ y : ℚ, 3 * y^2 - 11 * y + 2 = 0 ∧ y = d - ⌊d⌋):
  d = 13 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l474_47425


namespace NUMINAMATH_GPT_number_of_video_cassettes_in_first_set_l474_47437

/-- Let A be the cost of an audio cassette, and V the cost of a video cassette.
  We are given that V = 300, and we have the following conditions:
  1. 7 * A + n * V = 1110,
  2. 5 * A + 4 * V = 1350.
  Prove that n = 3, the number of video cassettes in the first set -/
theorem number_of_video_cassettes_in_first_set 
    (A V n : ℕ) 
    (hV : V = 300)
    (h1 : 7 * A + n * V = 1110)
    (h2 : 5 * A + 4 * V = 1350) : 
    n = 3 := 
sorry

end NUMINAMATH_GPT_number_of_video_cassettes_in_first_set_l474_47437


namespace NUMINAMATH_GPT_find_a_l474_47483

theorem find_a (x y a : ℤ) (h₁ : x = 1) (h₂ : y = -1) (h₃ : 2 * x - a * y = 3) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l474_47483


namespace NUMINAMATH_GPT_number_of_correct_statements_l474_47440

theorem number_of_correct_statements (stmt1: Prop) (stmt2: Prop) (stmt3: Prop) :
  stmt1 ∧ stmt2 ∧ stmt3 → (∀ n, n = 3) :=
by
  sorry

end NUMINAMATH_GPT_number_of_correct_statements_l474_47440


namespace NUMINAMATH_GPT_range_of_a_l474_47444

theorem range_of_a (a : ℝ) :
  (∃ x_0 ∈ Set.Icc (-1 : ℝ) 1, |4^x_0 - a * 2^x_0 + 1| ≤ 2^(x_0 + 1)) →
  0 ≤ a ∧ a ≤ (9/2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l474_47444


namespace NUMINAMATH_GPT_expected_reflection_value_l474_47434

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) *
  (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem expected_reflection_value :
  expected_reflections = (2 / Real.pi) *
    (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_expected_reflection_value_l474_47434


namespace NUMINAMATH_GPT_product_of_roots_l474_47449

theorem product_of_roots :
  (let a := 36
   let b := -24
   let c := -120
   a ≠ 0) →
  let roots_product := c / a
  roots_product = -10/3 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_l474_47449


namespace NUMINAMATH_GPT_original_visual_range_l474_47454

theorem original_visual_range
  (V : ℝ)
  (h1 : 2.5 * V = 150) :
  V = 60 :=
by
  sorry

end NUMINAMATH_GPT_original_visual_range_l474_47454


namespace NUMINAMATH_GPT_quadratic_transformation_l474_47471

theorem quadratic_transformation
    (a b c : ℝ)
    (h : ℝ)
    (cond : ∀ x, a * x^2 + b * x + c = 4 * (x - 5)^2 + 16) :
    (∀ x, 5 * a * x^2 + 5 * b * x + 5 * c = 20 * (x - h)^2 + 80) → h = 5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_transformation_l474_47471


namespace NUMINAMATH_GPT_pump_without_leak_time_l474_47470

theorem pump_without_leak_time :
  ∃ T : ℝ, (1/T - 1/5.999999999999999 = 1/3) ∧ T = 2 :=
by 
  sorry

end NUMINAMATH_GPT_pump_without_leak_time_l474_47470


namespace NUMINAMATH_GPT_triangle_perimeter_range_expression_l474_47417

-- Part 1: Prove the perimeter of △ABC
theorem triangle_perimeter (a b c : ℝ) (cosB : ℝ) (area : ℝ)
  (h1 : b^2 = a * c) (h2 : cosB = 3 / 5) (h3 : area = 2) :
  a + b + c = Real.sqrt 5 + Real.sqrt 21 :=
sorry

-- Part 2: Prove the range for the given expression
theorem range_expression (a b c : ℝ) (q : ℝ)
  (h1 : b = a * q) (h2 : c = a * q^2) :
  (Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_GPT_triangle_perimeter_range_expression_l474_47417


namespace NUMINAMATH_GPT_compute_expression_l474_47450

theorem compute_expression : 7^2 - 2 * 5 + 4^2 / 2 = 47 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l474_47450


namespace NUMINAMATH_GPT_polynomial_roots_condition_l474_47465

open Real

def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem polynomial_roots_condition (a b : ℤ) (h1 : ∀ x ≠ 0, f (x + x⁻¹) a b = f x a b + f x⁻¹ a b) (h2 : ∃ p q : ℤ, f p a b = 0 ∧ f q a b = 0) : a^2 + b^2 = 13 := by
  sorry

end NUMINAMATH_GPT_polynomial_roots_condition_l474_47465


namespace NUMINAMATH_GPT_geometric_sequence_condition_l474_47461

-- Definition of a geometric sequence
def is_geometric_sequence (x y z : ℤ) : Prop :=
  y ^ 2 = x * z

-- Lean 4 statement based on the condition and correct answer tuple
theorem geometric_sequence_condition (a : ℤ) :
  is_geometric_sequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l474_47461


namespace NUMINAMATH_GPT_solve_system_eqns_l474_47476

theorem solve_system_eqns :
  ∀ x y z : ℝ, 
  (x * y + 5 * y * z - 6 * x * z = -2 * z) ∧
  (2 * x * y + 9 * y * z - 9 * x * z = -12 * z) ∧
  (y * z - 2 * x * z = 6 * z) →
  x = -2 ∧ y = 2 ∧ z = 1 / 6 ∨
  y = 0 ∧ z = 0 ∨
  x = 0 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_eqns_l474_47476


namespace NUMINAMATH_GPT_exist_coprime_sums_l474_47472

theorem exist_coprime_sums (n k : ℕ) (h1 : 0 < n) (h2 : Even (k * (n - 1))) :
  ∃ x y : ℕ, Nat.gcd x n = 1 ∧ Nat.gcd y n = 1 ∧ (x + y) % n = k % n :=
  sorry

end NUMINAMATH_GPT_exist_coprime_sums_l474_47472


namespace NUMINAMATH_GPT_samantha_mean_correct_l474_47439

-- Given data: Samantha's assignment scores
def samantha_scores : List ℕ := [84, 89, 92, 88, 95, 91, 93]

-- Definition of the arithmetic mean of a list of scores
def arithmetic_mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / (scores.length : ℚ)

-- Prove that the arithmetic mean of Samantha's scores is 90.29
theorem samantha_mean_correct :
  arithmetic_mean samantha_scores = 90.29 := 
by
  -- The proof steps would be filled in here
  sorry

end NUMINAMATH_GPT_samantha_mean_correct_l474_47439


namespace NUMINAMATH_GPT_roads_going_outside_city_l474_47428

theorem roads_going_outside_city (n : ℕ)
  (h : ∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 3 ∧
    (n + x1) % 2 = 0 ∧
    (n + x2) % 2 = 0 ∧
    (n + x3) % 2 = 0) :
  ∃ (x1 x2 x3 : ℕ), (x1 = 1) ∧ (x2 = 1) ∧ (x3 = 1) :=
by 
  sorry

end NUMINAMATH_GPT_roads_going_outside_city_l474_47428


namespace NUMINAMATH_GPT_determine_distance_l474_47443

noncomputable def distance_formula (d a b c : ℝ) : Prop :=
  (d / a = (d - 30) / b) ∧
  (d / b = (d - 15) / c) ∧
  (d / a = (d - 40) / c)

theorem determine_distance (d a b c : ℝ) (h : distance_formula d a b c) : d = 90 :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_distance_l474_47443


namespace NUMINAMATH_GPT_lesser_number_of_sum_and_difference_l474_47416

theorem lesser_number_of_sum_and_difference (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end NUMINAMATH_GPT_lesser_number_of_sum_and_difference_l474_47416


namespace NUMINAMATH_GPT_red_cards_pick_ordered_count_l474_47466

theorem red_cards_pick_ordered_count :
  let deck_size := 36
  let suits := 3
  let suit_size := 12
  let red_suits := 2
  let red_cards := red_suits * suit_size
  (red_cards * (red_cards - 1) = 552) :=
by
  let deck_size := 36
  let suits := 3
  let suit_size := 12
  let red_suits := 2
  let red_cards := red_suits * suit_size
  show (red_cards * (red_cards - 1) = 552)
  sorry

end NUMINAMATH_GPT_red_cards_pick_ordered_count_l474_47466


namespace NUMINAMATH_GPT_probability_diff_by_three_l474_47480

theorem probability_diff_by_three (r1 r2 : ℕ) (h1 : 1 ≤ r1 ∧ r1 ≤ 6) (h2 : 1 ≤ r2 ∧ r2 ≤ 6) :
  (∃ (rolls : List (ℕ × ℕ)), 
    rolls = [ (2, 5), (5, 2), (3, 6), (4, 1) ] ∧ 
    (r1, r2) ∈ rolls) →
  (4 : ℚ) / 36 = (1 / 9 : ℚ) :=
by sorry

end NUMINAMATH_GPT_probability_diff_by_three_l474_47480


namespace NUMINAMATH_GPT_determine_ab_l474_47490

theorem determine_ab :
  ∃ a b : ℝ, 
  (3 + 8 * a = 2 - 3 * b) ∧ 
  (-1 - 6 * a = 4 * b) → 
  a = -1 / 14 ∧ b = -1 / 14 := 
by 
sorry

end NUMINAMATH_GPT_determine_ab_l474_47490


namespace NUMINAMATH_GPT_regular_octagon_side_length_l474_47477

theorem regular_octagon_side_length
  (side_length_pentagon : ℕ)
  (total_wire_length : ℕ)
  (side_length_octagon : ℕ) :
  side_length_pentagon = 16 →
  total_wire_length = 5 * side_length_pentagon →
  side_length_octagon = total_wire_length / 8 →
  side_length_octagon = 10 := 
sorry

end NUMINAMATH_GPT_regular_octagon_side_length_l474_47477


namespace NUMINAMATH_GPT_largest_divisor_of_m_l474_47411

theorem largest_divisor_of_m (m : ℕ) (hm : m > 0) (h : 54 ∣ m^2) : 18 ∣ m :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_m_l474_47411


namespace NUMINAMATH_GPT_evaluate_g_at_neg_four_l474_47407

def g (x : Int) : Int := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_neg_four_l474_47407


namespace NUMINAMATH_GPT_small_triangles_count_l474_47469

theorem small_triangles_count
  (sL sS : ℝ)  -- side lengths of large (sL) and small (sS) triangles
  (hL : sL = 15)  -- condition for the large triangle's side length
  (hS : sS = 3)   -- condition for the small triangle's side length
  : sL^2 / sS^2 = 25 := 
by {
  -- Definitions to skip the proof body
  -- Further mathematical steps would usually go here
  -- but 'sorry' is used to indicate the skipped proof.
  sorry
}

end NUMINAMATH_GPT_small_triangles_count_l474_47469


namespace NUMINAMATH_GPT_geometric_sequence_property_l474_47495

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n m : ℕ), a (n + 1) * a (m + 1) = a n * a m

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
(h_condition : a 2 * a 4 = 1/2) :
  a 1 * a 3 ^ 2 * a 5 = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_property_l474_47495


namespace NUMINAMATH_GPT_arithmetic_mean_value_of_x_l474_47403

theorem arithmetic_mean_value_of_x (x : ℝ) (h : (x + 10 + 20 + 3 * x + 16 + 3 * x + 6) / 5 = 30) : x = 14 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_mean_value_of_x_l474_47403


namespace NUMINAMATH_GPT_number_of_terms_ap_l474_47486

variables (a d n : ℤ) 

def sum_of_first_thirteen_terms := (13 / 2) * (2 * a + 12 * d)
def sum_of_last_thirteen_terms := (13 / 2) * (2 * a + (2 * n - 14) * d)

def sum_excluding_first_three := ((n - 3) / 2) * (2 * a + (n - 4) * d)
def sum_excluding_last_three := ((n - 3) / 2) * (2 * a + (n - 1) * d)

theorem number_of_terms_ap (h1 : sum_of_first_thirteen_terms a d = (1 / 2) * sum_of_last_thirteen_terms a d)
  (h2 : sum_excluding_first_three a d / sum_excluding_last_three a d = 5 / 4) : n = 22 :=
sorry

end NUMINAMATH_GPT_number_of_terms_ap_l474_47486


namespace NUMINAMATH_GPT_quadratic_factorization_l474_47457

theorem quadratic_factorization (p q x_1 x_2 : ℝ) (h1 : x_1 = 2) (h2 : x_2 = -3) 
    (h3 : x_1 + x_2 = -p) (h4 : x_1 * x_2 = q) : 
    (x - 2) * (x + 3) = x^2 + p * x + q :=
by
  sorry

end NUMINAMATH_GPT_quadratic_factorization_l474_47457


namespace NUMINAMATH_GPT_claire_has_gerbils_l474_47441

-- Definitions based on conditions
variables (G H : ℕ)
variables (h1 : G + H = 90) (h2 : (1/4 : ℚ) * G + (1/3 : ℚ) * H = 25)

-- Main statement to prove
theorem claire_has_gerbils : G = 60 :=
sorry

end NUMINAMATH_GPT_claire_has_gerbils_l474_47441


namespace NUMINAMATH_GPT_number_is_4_l474_47414

theorem number_is_4 (x : ℕ) (h : x + 5 = 9) : x = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_number_is_4_l474_47414


namespace NUMINAMATH_GPT_amount_A_l474_47493

theorem amount_A (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 527) : A = 62 := by
  sorry

end NUMINAMATH_GPT_amount_A_l474_47493


namespace NUMINAMATH_GPT_max_value_of_a2b3c4_l474_47418

open Real

theorem max_value_of_a2b3c4
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a + b + c = 3) :
  a^2 * b^3 * c^4 ≤ 19683 / 472392 :=
sorry

end NUMINAMATH_GPT_max_value_of_a2b3c4_l474_47418


namespace NUMINAMATH_GPT_largest_whole_number_value_l474_47438

theorem largest_whole_number_value (n : ℕ) : 
  (1 : ℚ) / 5 + (n : ℚ) / 8 < 9 / 5 → n ≤ 12 := 
sorry

end NUMINAMATH_GPT_largest_whole_number_value_l474_47438


namespace NUMINAMATH_GPT_Megan_seashells_needed_l474_47431

-- Let x be the number of additional seashells needed
def seashells_needed (total_seashells desired_seashells : Nat) : Nat :=
  desired_seashells - total_seashells

-- Given conditions
def current_seashells : Nat := 19
def desired_seashells : Nat := 25

-- The equivalent proof problem
theorem Megan_seashells_needed : seashells_needed current_seashells desired_seashells = 6 := by
  sorry

end NUMINAMATH_GPT_Megan_seashells_needed_l474_47431


namespace NUMINAMATH_GPT_find_f_half_l474_47475

variable {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

def f_condition (f : R → R) : Prop := ∀ x : R, x < 0 → f x = 1 / (x + 1)

theorem find_f_half (f : R → R) (h_odd : odd_function f) (h_condition : f_condition f) : f (1 / 2) = -2 := by
  sorry

end NUMINAMATH_GPT_find_f_half_l474_47475


namespace NUMINAMATH_GPT_vertex_in_fourth_quadrant_l474_47485

theorem vertex_in_fourth_quadrant (a : ℝ) (ha : a < 0) :  
  let x_vertex := -a / 4
  let y_vertex := (-40 - a^2) / 8
  x_vertex > 0 ∧ y_vertex < 0 := by
  let x_vertex := -a / 4
  let y_vertex := (-40 - a^2) / 8
  have hx : x_vertex > 0 := by sorry
  have hy : y_vertex < 0 := by sorry
  exact And.intro hx hy

end NUMINAMATH_GPT_vertex_in_fourth_quadrant_l474_47485


namespace NUMINAMATH_GPT_rectangle_area_error_percentage_l474_47484

theorem rectangle_area_error_percentage (L W : ℝ) : 
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let actual_area := L * W
  let measured_area := measured_length * measured_width
  let error := measured_area - actual_area
  let error_percentage := (error / actual_area) * 100
  error_percentage = 0.7 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_error_percentage_l474_47484


namespace NUMINAMATH_GPT_sqrt_two_minus_one_pow_zero_l474_47419

theorem sqrt_two_minus_one_pow_zero : (Real.sqrt 2 - 1)^0 = 1 := by
  sorry

end NUMINAMATH_GPT_sqrt_two_minus_one_pow_zero_l474_47419


namespace NUMINAMATH_GPT_vec_subtraction_l474_47415

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (0, 1)

theorem vec_subtraction : a - 2 • b = (-1, 0) := by
  sorry

end NUMINAMATH_GPT_vec_subtraction_l474_47415


namespace NUMINAMATH_GPT_intersection_locus_is_vertical_line_l474_47402

/-- 
Given \( 0 < a < b \), lines \( l \) and \( m \) are drawn through the points \( A(a, 0) \) and \( B(b, 0) \), 
respectively, such that these lines intersect the parabola \( y^2 = x \) at four distinct points 
and these four points are concyclic. 

We want to prove that the locus of the intersection point \( P \) of lines \( l \) and \( m \) 
is the vertical line \( x = \frac{a + b}{2} \).
-/
theorem intersection_locus_is_vertical_line (a b : ℝ) (h : 0 < a ∧ a < b) :
  (∃ P : ℝ × ℝ, P.fst = (a + b) / 2) := 
sorry

end NUMINAMATH_GPT_intersection_locus_is_vertical_line_l474_47402


namespace NUMINAMATH_GPT_larger_of_two_numbers_l474_47412

theorem larger_of_two_numbers (H : Nat := 15) (f1 : Nat := 11) (f2 : Nat := 15) :
  let lcm := H * f1 * f2;
  ∃ (A B : Nat), A = H * f1 ∧ B = H * f2 ∧ A ≤ B := by
  sorry

end NUMINAMATH_GPT_larger_of_two_numbers_l474_47412


namespace NUMINAMATH_GPT_mushroom_mistake_l474_47409

theorem mushroom_mistake (p k v : ℝ) (hk : k = p + v - 10) (hp : p = k + v - 7) : 
  ∃ p k : ℝ, ∀ v : ℝ, (p = k + v - 7) ∧ (k = p + v - 10) → false :=
by
  sorry

end NUMINAMATH_GPT_mushroom_mistake_l474_47409


namespace NUMINAMATH_GPT_second_order_arithmetic_sequence_20th_term_l474_47406

theorem second_order_arithmetic_sequence_20th_term :
  (∀ a : ℕ → ℕ,
    a 1 = 1 ∧
    a 2 = 4 ∧
    a 3 = 9 ∧
    a 4 = 16 ∧
    (∀ n, 2 ≤ n → a n - a (n - 1) = 2 * n - 1) →
    a 20 = 400) :=
by 
  sorry

end NUMINAMATH_GPT_second_order_arithmetic_sequence_20th_term_l474_47406


namespace NUMINAMATH_GPT_find_subtracted_value_l474_47489

theorem find_subtracted_value (N V : ℕ) (hN : N = 12) (h : 4 * N - V = 9 * (N - 7)) : V = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_subtracted_value_l474_47489


namespace NUMINAMATH_GPT_union_of_sets_l474_47421

def setA : Set ℝ := { x : ℝ | (x - 2) / (x + 1) ≤ 0 }
def setB : Set ℝ := { x : ℝ | -2 * x^2 + 7 * x + 4 > 0 }
def unionAB : Set ℝ := { x : ℝ | -1 < x ∧ x < 4 }

theorem union_of_sets :
  ∀ x : ℝ, x ∈ setA ∨ x ∈ setB ↔ x ∈ unionAB :=
by sorry

end NUMINAMATH_GPT_union_of_sets_l474_47421


namespace NUMINAMATH_GPT_Simon_has_72_legos_l474_47405

theorem Simon_has_72_legos 
  (Kent_legos : ℕ)
  (h1 : Kent_legos = 40) 
  (Bruce_legos : ℕ) 
  (h2 : Bruce_legos = Kent_legos + 20) 
  (Simon_legos : ℕ) 
  (h3 : Simon_legos = Bruce_legos + (Bruce_legos/5)) :
  Simon_legos = 72 := 
  by
    -- Begin proof (not required for the problem)
    -- Proof steps would follow here
    sorry

end NUMINAMATH_GPT_Simon_has_72_legos_l474_47405


namespace NUMINAMATH_GPT_max_min_difference_l474_47413

def y (x : ℝ) : ℝ := x * abs (3 - x) - (x - 3) * abs x

theorem max_min_difference : (0 : ℝ) ≤ x → (x < 3 → y x ≤ y (3 / 4)) ∧ (x < 0 → y x = 0) ∧ (x ≥ 3 → y x = 0) → 
  (y (3 / 4) - (min (y 0) (min (y (-1)) (y 3)))) = 1.125 :=
by
  sorry

end NUMINAMATH_GPT_max_min_difference_l474_47413


namespace NUMINAMATH_GPT_remainder_when_divide_by_66_l474_47429

-- Define the conditions as predicates
def condition_1 (n : ℕ) : Prop := ∃ l : ℕ, n % 22 = 7
def condition_2 (n : ℕ) : Prop := ∃ m : ℕ, n % 33 = 18

-- Define the main theorem
theorem remainder_when_divide_by_66 (n : ℕ) (h1 : condition_1 n) (h2 : condition_2 n) : n % 66 = 51 :=
  sorry

end NUMINAMATH_GPT_remainder_when_divide_by_66_l474_47429


namespace NUMINAMATH_GPT_expenditure_on_house_rent_l474_47430

theorem expenditure_on_house_rent (I : ℝ) (H1 : 0.30 * I = 300) : 0.20 * (I - 0.30 * I) = 140 :=
by
  -- Skip the proof, the statement is sufficient at this stage.
  sorry

end NUMINAMATH_GPT_expenditure_on_house_rent_l474_47430


namespace NUMINAMATH_GPT_calculation_l474_47487

variable (x y z : ℕ)

theorem calculation (h1 : x + y + z = 20) (h2 : x + y - z = 8) :
  x + y = 14 :=
  sorry

end NUMINAMATH_GPT_calculation_l474_47487
