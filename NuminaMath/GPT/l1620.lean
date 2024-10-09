import Mathlib

namespace slope_of_line_l1620_162028

noncomputable def line_eq (x y : ℝ) := x / 4 + y / 5 = 1

theorem slope_of_line : ∀ (x y : ℝ), line_eq x y → (∃ m b : ℝ, y = m * x + b ∧ m = -5 / 4) :=
sorry

end slope_of_line_l1620_162028


namespace tangent_line_through_point_l1620_162042

theorem tangent_line_through_point (a : ℝ) : 
  ∃ l : ℝ → ℝ, 
    (∀ x y : ℝ, (x - 1)^2 + y^2 = 4 → y = a) ∧ 
    (∀ x y : ℝ, y = l x → (x - 1)^2 + y^2 = 4) → 
    a = 0 :=
by
  sorry

end tangent_line_through_point_l1620_162042


namespace number_of_fish_initially_tagged_l1620_162036

theorem number_of_fish_initially_tagged {N T : ℕ}
  (hN : N = 1250)
  (h_ratio : 2 / 50 = T / N) :
  T = 50 :=
by
  sorry

end number_of_fish_initially_tagged_l1620_162036


namespace nearest_integer_to_3_plus_sqrt2_pow_four_l1620_162017

open Real

theorem nearest_integer_to_3_plus_sqrt2_pow_four : 
  (∃ n : ℤ, abs (n - (3 + (sqrt 2))^4) < 0.5) ∧ 
  (abs (382 - (3 + (sqrt 2))^4) < 0.5) := 
by 
  sorry

end nearest_integer_to_3_plus_sqrt2_pow_four_l1620_162017


namespace largest_prime_y_in_triangle_l1620_162016

-- Define that a number is prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_y_in_triangle : 
  ∃ (x y z : ℕ), is_prime x ∧ is_prime y ∧ is_prime z ∧ x + y + z = 90 ∧ y < x ∧ y > z ∧ y = 47 :=
by
  sorry

end largest_prime_y_in_triangle_l1620_162016


namespace probability_exactly_one_first_class_l1620_162091

-- Define the probabilities
def prob_first_class_first_intern : ℚ := 2 / 3
def prob_first_class_second_intern : ℚ := 3 / 4
def prob_not_first_class_first_intern : ℚ := 1 - prob_first_class_first_intern
def prob_not_first_class_second_intern : ℚ := 1 - prob_first_class_second_intern

-- Define the event A, which is the event that exactly one of the two parts is of first-class quality
def prob_event_A : ℚ :=
  (prob_first_class_first_intern * prob_not_first_class_second_intern) +
  (prob_not_first_class_first_intern * prob_first_class_second_intern)

theorem probability_exactly_one_first_class (h1 : prob_first_class_first_intern = 2 / 3) 
    (h2 : prob_first_class_second_intern = 3 / 4) 
    (h3 : prob_event_A = 
          (prob_first_class_first_intern * (1 - prob_first_class_second_intern)) + 
          ((1 - prob_first_class_first_intern) * prob_first_class_second_intern)) : 
  prob_event_A = 5 / 12 := 
  sorry

end probability_exactly_one_first_class_l1620_162091


namespace math_problem_l1620_162094

theorem math_problem (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 :=
by
  -- The proof will be here
  sorry

end math_problem_l1620_162094


namespace trapezium_area_correct_l1620_162033

def a : ℚ := 20  -- Length of the first parallel side
def b : ℚ := 18  -- Length of the second parallel side
def h : ℚ := 20  -- Distance (height) between the parallel sides

def trapezium_area (a b h : ℚ) : ℚ :=
  (1/2) * (a + b) * h

theorem trapezium_area_correct : trapezium_area a b h = 380 := 
  by
    sorry  -- Proof goes here

end trapezium_area_correct_l1620_162033


namespace circumference_divided_by_diameter_l1620_162087

noncomputable def radius : ℝ := 15
noncomputable def circumference : ℝ := 90
noncomputable def diameter : ℝ := 2 * radius

theorem circumference_divided_by_diameter :
  circumference / diameter = 3 := by
  sorry

end circumference_divided_by_diameter_l1620_162087


namespace greatest_product_sum_300_l1620_162038

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l1620_162038


namespace increasing_or_decreasing_subseq_l1620_162045

theorem increasing_or_decreasing_subseq {m n : ℕ} (a : Fin (m * n + 1) → ℝ) :
  ∃ (idx_incr : Fin (m + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (idx_incr i) < a (idx_incr j)) ∨ 
  ∃ (idx_decr : Fin (n + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (idx_decr i) > a (idx_decr j)) :=
by
  sorry

end increasing_or_decreasing_subseq_l1620_162045


namespace range_of_a_l1620_162099

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + a * x + a = 0) → a ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end range_of_a_l1620_162099


namespace unknown_number_is_7_l1620_162082

theorem unknown_number_is_7 (x : ℤ) (hx : x > 0)
  (h : (1 / 4 : ℚ) * (10 * x + 7 - x ^ 2) - x = 0) : x = 7 :=
  sorry

end unknown_number_is_7_l1620_162082


namespace scientific_notation_GDP_l1620_162084

theorem scientific_notation_GDP (h : 1 = 10^9) : 32.07 * 10^9 = 3.207 * 10^10 := by
  sorry

end scientific_notation_GDP_l1620_162084


namespace largest_composite_not_written_l1620_162010

theorem largest_composite_not_written (n : ℕ) (hn : n = 2022) : ¬ ∃ d > 1, 2033 = n + d := 
by
  sorry

end largest_composite_not_written_l1620_162010


namespace total_pens_bought_l1620_162052

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l1620_162052


namespace simplify_fraction_l1620_162077

-- Define the numbers involved and state their GCD
def num1 := 90
def num2 := 8100

-- State the GCD condition using a Lean 4 statement
def gcd_condition (a b : ℕ) := Nat.gcd a b = 90

-- Define the original fraction and the simplified fraction
def original_fraction := num1 / num2
def simplified_fraction := 1 / 90

-- State the proof problem that the original fraction simplifies to the simplified fraction
theorem simplify_fraction : gcd_condition num1 num2 → original_fraction = simplified_fraction := 
by
  sorry

end simplify_fraction_l1620_162077


namespace min_time_calculation_l1620_162086

noncomputable def min_time_to_receive_keys (diameter cyclist_speed_road cyclist_speed_alley pedestrian_speed : ℝ) : ℝ :=
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let distance_pedestrian := pedestrian_speed * 1
  let min_time := (2 * Real.pi * radius - 2 * distance_pedestrian) / (cyclist_speed_road + cyclist_speed_alley)
  min_time

theorem min_time_calculation :
  min_time_to_receive_keys 4 15 20 6 = (2 * Real.pi - 2) / 21 :=
by
  sorry

end min_time_calculation_l1620_162086


namespace graduation_ceremony_chairs_l1620_162095

theorem graduation_ceremony_chairs (g p t a : ℕ) 
  (h_g : g = 50) 
  (h_p : p = 2 * g) 
  (h_t : t = 20) 
  (h_a : a = t / 2) : 
  g + p + t + a = 180 :=
by
  sorry

end graduation_ceremony_chairs_l1620_162095


namespace find_y_l1620_162049

theorem find_y
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y)
  (hr : x % y = 8)
  (hq : x / y = 96) 
  (hr_decimal : (x:ℚ) / (y:ℚ) = 96.16) :
  y = 50 := 
sorry

end find_y_l1620_162049


namespace green_peppers_weight_l1620_162046

theorem green_peppers_weight (total_weight : ℝ) (w : ℝ) (h1 : total_weight = 5.666666667)
  (h2 : 2 * w = total_weight) : w = 2.8333333335 :=
by
  sorry

end green_peppers_weight_l1620_162046


namespace area_of_triangle_AEB_l1620_162014

structure Rectangle :=
  (A B C D : Type)
  (AB : ℝ)
  (BC : ℝ)
  (F G E : Type)
  (DF : ℝ)
  (GC : ℝ)
  (AF_BG_intersect_at_E : Prop)

def rectangle_example : Rectangle := {
  A := Unit,
  B := Unit,
  C := Unit,
  D := Unit,
  AB := 8,
  BC := 4,
  F := Unit,
  G := Unit,
  E := Unit,
  DF := 2,
  GC := 3,
  AF_BG_intersect_at_E := true
}

theorem area_of_triangle_AEB (r : Rectangle) (h : r = rectangle_example) :
  ∃ area : ℝ, area = 128 / 3 :=
by
  sorry

end area_of_triangle_AEB_l1620_162014


namespace W_k_two_lower_bound_l1620_162056

-- Define W(k, 2)
def W (k : ℕ) (c : ℕ) : ℕ := -- smallest number such that for every n >= W(k, 2), 
  -- any 2-coloring of the set {1, 2, ..., n} contains a monochromatic arithmetic progression of length k
  sorry 

-- Define the statement to prove
theorem W_k_two_lower_bound (k : ℕ) : ∃ C > 0, W k 2 ≥ C * 2^(k / 2) :=
by
  sorry

end W_k_two_lower_bound_l1620_162056


namespace sphere_surface_area_diameter_4_l1620_162013

noncomputable def sphere_surface_area (d : ℝ) : ℝ :=
  4 * Real.pi * (d / 2) ^ 2

theorem sphere_surface_area_diameter_4 :
  sphere_surface_area 4 = 16 * Real.pi :=
by
  sorry

end sphere_surface_area_diameter_4_l1620_162013


namespace range_of_k_l1620_162060

theorem range_of_k (k : ℝ) : (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) ↔ k ≤ 1 :=
by sorry

end range_of_k_l1620_162060


namespace geometric_seq_condition_l1620_162063

-- Defining a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Defining an increasing sequence
def is_increasing_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- The condition to be proved
theorem geometric_seq_condition (a : ℕ → ℝ) (h_geo : is_geometric_seq a) :
  (a 0 < a 1 → is_increasing_seq a) ∧ (is_increasing_seq a → a 0 < a 1) :=
by 
  sorry

end geometric_seq_condition_l1620_162063


namespace ajay_distance_l1620_162005

/- Definitions -/
def speed : ℝ := 50 -- Ajay's speed in km/hour
def time : ℝ := 30 -- Time taken in hours

/- Theorem statement -/
theorem ajay_distance : (speed * time = 1500) :=
by
  sorry

end ajay_distance_l1620_162005


namespace watch_correction_l1620_162035

noncomputable def correction_time (loss_per_day : ℕ) (start_date : ℕ) (end_date : ℕ) (spring_forward_hour : ℕ) (correction_time_hour : ℕ) : ℝ :=
  let n_days := end_date - start_date
  let total_hours_watch := n_days * 24 + correction_time_hour - spring_forward_hour
  let loss_rate_per_hour := (loss_per_day : ℝ) / 24
  let total_loss := loss_rate_per_hour * total_hours_watch
  total_loss

theorem watch_correction :
  correction_time 3 1 5 1 6 = 6.625 :=
by
  sorry

end watch_correction_l1620_162035


namespace balloon_arrangements_l1620_162023

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l1620_162023


namespace cos_3theta_l1620_162057

theorem cos_3theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end cos_3theta_l1620_162057


namespace part1_part2_l1620_162097

noncomputable def f (x a : ℝ) : ℝ := |x - 2 * a| + |x - 3 * a|

theorem part1 (a : ℝ) (h_min : ∃ x, f x a = 2) : |a| = 2 := by
  sorry

theorem part2 (m : ℝ)
  (h_condition : ∀ x : ℝ, ∃ a : ℝ, -2 ≤ a ∧ a ≤ 2 ∧ (m^2 - |m| - f x a) < 0) :
  -1 < m ∧ m < 2 := by
  sorry

end part1_part2_l1620_162097


namespace four_times_sum_of_cubes_gt_cube_sum_l1620_162012

theorem four_times_sum_of_cubes_gt_cube_sum
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 :=
by
  sorry

end four_times_sum_of_cubes_gt_cube_sum_l1620_162012


namespace five_in_range_for_all_b_l1620_162090

noncomputable def f (x b : ℝ) := x^2 + b * x - 3

theorem five_in_range_for_all_b : ∀ (b : ℝ), ∃ (x : ℝ), f x b = 5 := by 
  sorry

end five_in_range_for_all_b_l1620_162090


namespace percentage_not_speaking_French_is_60_l1620_162098

-- Define the number of students who speak English well and those who do not.
def speakEnglishWell : Nat := 20
def doNotSpeakEnglish : Nat := 60

-- Calculate the total number of students who speak French.
def speakFrench : Nat := speakEnglishWell + doNotSpeakEnglish

-- Define the total number of students surveyed.
def totalStudents : Nat := 200

-- Calculate the number of students who do not speak French.
def doNotSpeakFrench : Nat := totalStudents - speakFrench

-- Calculate the percentage of students who do not speak French.
def percentageDoNotSpeakFrench : Float := (doNotSpeakFrench.toFloat / totalStudents.toFloat) * 100

-- Theorem asserting the percentage of students who do not speak French is 60%.
theorem percentage_not_speaking_French_is_60 : percentageDoNotSpeakFrench = 60 := by
  sorry

end percentage_not_speaking_French_is_60_l1620_162098


namespace log_inequality_l1620_162096

noncomputable def log3_2 : ℝ := Real.log 2 / Real.log 3
noncomputable def log2_3 : ℝ := Real.log 3 / Real.log 2
noncomputable def log2_5 : ℝ := Real.log 5 / Real.log 2

theorem log_inequality :
  let a := log3_2;
  let b := log2_3;
  let c := log2_5;
  a < b ∧ b < c :=
  by
  sorry

end log_inequality_l1620_162096


namespace primes_equal_l1620_162007

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_equal (p q r n : ℕ) (h_prime_p : is_prime p) (h_prime_q : is_prime q)
(h_prime_r : is_prime r) (h_pos_n : 0 < n)
(h1 : (p + n) % (q * r) = 0)
(h2 : (q + n) % (r * p) = 0)
(h3 : (r + n) % (p * q) = 0) : p = q ∧ q = r := by
  sorry

end primes_equal_l1620_162007


namespace value_of_expression_l1620_162066

theorem value_of_expression (m n : ℝ) (h : m + 2 * n = 1) : 3 * m^2 + 6 * m * n + 6 * n = 3 :=
by
  sorry -- Placeholder for the proof

end value_of_expression_l1620_162066


namespace fraction_zero_l1620_162068

theorem fraction_zero (x : ℝ) (h₁ : x - 3 = 0) (h₂ : x ≠ 0) : (x - 3) / (4 * x) = 0 :=
by
  sorry

end fraction_zero_l1620_162068


namespace opposite_number_of_sqrt_of_9_is_neg3_l1620_162029

theorem opposite_number_of_sqrt_of_9_is_neg3 :
  - (Real.sqrt 9) = -3 :=
by
  -- The proof is omitted as required.
  sorry

end opposite_number_of_sqrt_of_9_is_neg3_l1620_162029


namespace garden_width_l1620_162078

variable (W : ℝ) (L : ℝ := 225) (small_gate : ℝ := 3) (large_gate: ℝ := 10) (total_fencing : ℝ := 687)

theorem garden_width :
  2 * L + 2 * W - (small_gate + large_gate) = total_fencing → W = 125 := 
by
  sorry

end garden_width_l1620_162078


namespace inequality_solution_l1620_162041

theorem inequality_solution (x : ℝ) (h : |(x + 4) / 2| < 3) : -10 < x ∧ x < 2 :=
by
  sorry

end inequality_solution_l1620_162041


namespace sum_sublist_eq_100_l1620_162075

theorem sum_sublist_eq_100 {l : List ℕ}
  (h_len : l.length = 2 * 31100)
  (h_max : ∀ x ∈ l, x ≤ 100)
  (h_sum : l.sum = 200) :
  ∃ (s : List ℕ), s ⊆ l ∧ s.sum = 100 := 
sorry

end sum_sublist_eq_100_l1620_162075


namespace total_distance_eq_l1620_162032

def distance_traveled_by_bus : ℝ := 2.6
def distance_traveled_by_subway : ℝ := 5.98
def total_distance_traveled : ℝ := distance_traveled_by_bus + distance_traveled_by_subway

theorem total_distance_eq : total_distance_traveled = 8.58 := by
  sorry

end total_distance_eq_l1620_162032


namespace benjamin_decade_expense_l1620_162059

-- Define the constants
def yearly_expense : ℕ := 3000
def years : ℕ := 10

-- Formalize the statement
theorem benjamin_decade_expense : yearly_expense * years = 30000 := 
by
  sorry

end benjamin_decade_expense_l1620_162059


namespace find_angle_l1620_162006

theorem find_angle (A : ℝ) (h : 0 < A ∧ A < π) 
  (c : 4 * π * Real.sin A - 3 * Real.arccos (-1/2) = 0) :
  A = π / 6 ∨ A = 5 * π / 6 :=
sorry

end find_angle_l1620_162006


namespace andrew_paid_in_dollars_l1620_162015

def local_currency_to_dollars (units : ℝ) : ℝ := units * 0.25

def cost_of_fruits : ℝ :=
  let cost_grapes := 7 * 68
  let cost_mangoes := 9 * 48
  let cost_apples := 5 * 55
  let cost_oranges := 4 * 38
  let total_cost_grapes_mangoes := cost_grapes + cost_mangoes
  let total_cost_apples_oranges := cost_apples + cost_oranges
  let discount_grapes_mangoes := 0.10 * total_cost_grapes_mangoes
  let discounted_grapes_mangoes := total_cost_grapes_mangoes - discount_grapes_mangoes
  let discounted_apples_oranges := total_cost_apples_oranges - 25
  let total_discounted_cost := discounted_grapes_mangoes + discounted_apples_oranges
  let sales_tax := 0.05 * total_discounted_cost
  let total_tax := sales_tax + 15
  let total_amount_with_taxes := total_discounted_cost + total_tax
  total_amount_with_taxes

theorem andrew_paid_in_dollars : local_currency_to_dollars cost_of_fruits = 323.79 :=
  by
  sorry

end andrew_paid_in_dollars_l1620_162015


namespace sum_first_10_terms_l1620_162025

variable (a : ℕ → ℕ)

def condition (p q : ℕ) : Prop :=
  p + q = 11 ∧ p < q

axiom condition_a_p_a_q : ∀ (p q : ℕ), (condition p q) → (a p + a q = 2^p)

theorem sum_first_10_terms (a : ℕ → ℕ) (h : ∀ (p q : ℕ), condition p q → a p + a q = 2^p) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 62) :=
by 
  sorry

end sum_first_10_terms_l1620_162025


namespace ellipse_circle_inequality_l1620_162011

theorem ellipse_circle_inequality
  (a b : ℝ) (x y : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (h_ellipse1 : (x1^2) / (a^2) + (y1^2) / (b^2) = 1)
  (h_ellipse2 : (x2^2) / (a^2) + (y2^2) / (b^2) = 1)
  (h_ab : a > b ∧ b > 0)
  (h_circle : (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0) :
  x^2 + y^2 ≤ (3/2) * a^2 + (1/2) * b^2 :=
sorry

end ellipse_circle_inequality_l1620_162011


namespace David_is_8_years_older_than_Scott_l1620_162002

noncomputable def DavidAge : ℕ := 14 -- Since David was 8 years old, 6 years ago
noncomputable def RichardAge : ℕ := DavidAge + 6
noncomputable def ScottAge : ℕ := (RichardAge + 8) / 2 - 8
noncomputable def AgeDifference : ℕ := DavidAge - ScottAge

theorem David_is_8_years_older_than_Scott :
  AgeDifference = 8 :=
by
  sorry

end David_is_8_years_older_than_Scott_l1620_162002


namespace binomial_sum_l1620_162030

theorem binomial_sum :
  (Nat.choose 10 3) + (Nat.choose 10 4) = 330 :=
by
  sorry

end binomial_sum_l1620_162030


namespace men_wages_l1620_162054

theorem men_wages (W : ℕ) (wage : ℕ) :
  (5 + W + 8) * wage = 75 ∧ 5 * wage = W * wage ∧ W * wage = 8 * wage → 
  wage = 5 := 
by
  sorry

end men_wages_l1620_162054


namespace point_below_line_range_l1620_162064

theorem point_below_line_range (t : ℝ) : (2 * (-2) - 3 * t + 6 > 0) → t < (2 / 3) :=
by {
  sorry
}

end point_below_line_range_l1620_162064


namespace particle_path_count_l1620_162092

def lattice_path_count (n : ℕ) : ℕ :=
sorry -- Placeholder for the actual combinatorial function

theorem particle_path_count : lattice_path_count 7 = sorry :=
sorry -- Placeholder for the actual count

end particle_path_count_l1620_162092


namespace find_x_collinear_l1620_162043

def vec := ℝ × ℝ

def collinear (u v: vec): Prop :=
  ∃ k: ℝ, u = (k * v.1, k * v.2)

theorem find_x_collinear:
  ∀ (x: ℝ), (let a : vec := (1, 2)
              let b : vec := (x, 1)
              collinear a (a.1 - b.1, a.2 - b.2)) → x = 1 / 2 :=
by
  intros x h
  sorry

end find_x_collinear_l1620_162043


namespace complete_the_square_l1620_162070

theorem complete_the_square (m n : ℕ) :
  (∀ x : ℝ, x^2 - 6 * x = 1 → (x - m)^2 = n) → m + n = 13 :=
by
  sorry

end complete_the_square_l1620_162070


namespace negation_of_implication_l1620_162071

theorem negation_of_implication (x : ℝ) :
  ¬ (x > 1 → x^2 > 1) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by
  sorry

end negation_of_implication_l1620_162071


namespace even_function_value_l1620_162003

theorem even_function_value (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_def : ∀ x : ℝ, 0 < x → f x = 2^x + 1) :
  f (-2) = 5 :=
  sorry

end even_function_value_l1620_162003


namespace manager_salary_l1620_162027

theorem manager_salary (avg_salary_employees : ℝ) (num_employees : ℕ) (salary_increase : ℝ) (manager_salary : ℝ) :
  avg_salary_employees = 1500 →
  num_employees = 24 →
  salary_increase = 400 →
  (num_employees + 1) * (avg_salary_employees + salary_increase) - num_employees * avg_salary_employees = manager_salary →
  manager_salary = 11500 := 
by
  intros h_avg_salary_employees h_num_employees h_salary_increase h_computation
  sorry

end manager_salary_l1620_162027


namespace bakery_baguettes_l1620_162073

theorem bakery_baguettes : 
  ∃ B : ℕ, 
  (∃ B : ℕ, 3 * B - 138 = 6) ∧ 
  B = 48 :=
by
  sorry

end bakery_baguettes_l1620_162073


namespace g_correct_l1620_162080

-- Define the polynomials involved
def p1 (x : ℝ) : ℝ := 2 * x^5 + 4 * x^3 - 3 * x
def p2 (x : ℝ) : ℝ := 7 * x^3 + 5 * x - 2

-- Define g(x) as the polynomial we need to find
def g (x : ℝ) : ℝ := -2 * x^5 + 3 * x^3 + 8 * x - 2

-- Now, state the condition
def condition (x : ℝ) : Prop := p1 x + g x = p2 x

-- Prove the condition holds with the defined polynomials
theorem g_correct (x : ℝ) : condition x :=
by
  change p1 x + g x = p2 x
  sorry

end g_correct_l1620_162080


namespace eccentricity_of_given_ellipse_l1620_162001

noncomputable def eccentricity_of_ellipse : ℝ :=
  let a : ℝ := 1
  let b : ℝ := 1 / 2
  let c : ℝ := Real.sqrt (a ^ 2 - b ^ 2)
  c / a

theorem eccentricity_of_given_ellipse :
  eccentricity_of_ellipse = Real.sqrt (3) / 2 :=
by
  -- Proof is omitted.
  sorry

end eccentricity_of_given_ellipse_l1620_162001


namespace find_angle_OD_base_l1620_162020

noncomputable def angle_between_edge_and_base (α β : ℝ): ℝ :=
  Real.arctan ((Real.sin α * Real.sin β) / Real.sqrt (Real.sin (α - β) * Real.sin (α + β)))

theorem find_angle_OD_base (α β : ℝ) :
  ∃ γ : ℝ, γ = angle_between_edge_and_base α β :=
sorry

end find_angle_OD_base_l1620_162020


namespace division_sequence_l1620_162083

theorem division_sequence : (120 / 5) / 2 / 3 = 4 := by
  sorry

end division_sequence_l1620_162083


namespace coffee_last_days_l1620_162088

theorem coffee_last_days (weight : ℕ) (cups_per_lb : ℕ) (cups_per_day : ℕ) 
  (h_weight : weight = 3) 
  (h_cups_per_lb : cups_per_lb = 40) 
  (h_cups_per_day : cups_per_day = 3) : 
  (weight * cups_per_lb) / cups_per_day = 40 := 
by 
  sorry

end coffee_last_days_l1620_162088


namespace count_ball_distributions_l1620_162018

theorem count_ball_distributions : 
  ∃ (n : ℕ), n = 3 ∧
  (∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → (∀ (dist : ℕ → ℕ), (sorry: Prop))) := sorry

end count_ball_distributions_l1620_162018


namespace time_to_cross_is_correct_l1620_162039

noncomputable def train_cross_bridge_time : ℝ :=
  let length_train := 130
  let speed_train_kmh := 45
  let length_bridge := 245.03
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_train_ms
  time

theorem time_to_cross_is_correct : train_cross_bridge_time = 30.0024 :=
by
  sorry

end time_to_cross_is_correct_l1620_162039


namespace jake_peaches_is_7_l1620_162050

variable (Steven_peaches Jake_peaches Jill_peaches : ℕ)

-- Conditions:
def Steven_has_19_peaches : Steven_peaches = 19 := by sorry

def Jake_has_12_fewer_peaches_than_Steven : Jake_peaches = Steven_peaches - 12 := by sorry

def Jake_has_72_more_peaches_than_Jill : Jake_peaches = Jill_peaches + 72 := by sorry

-- Proof problem:
theorem jake_peaches_is_7 
    (Steven_peaches Jake_peaches Jill_peaches : ℕ)
    (h1 : Steven_peaches = 19)
    (h2 : Jake_peaches = Steven_peaches - 12)
    (h3 : Jake_peaches = Jill_peaches + 72) :
    Jake_peaches = 7 := by sorry

end jake_peaches_is_7_l1620_162050


namespace solve_for_a_l1620_162000

-- Definitions: Real number a, Imaginary unit i, complex number.
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem solve_for_a :
  ∀ (a : ℝ) (i : ℂ),
    i = Complex.I →
    is_purely_imaginary ( (3 * i / (1 + 2 * i)) * (1 - (a / 3) * i) ) →
    a = -6 :=
by
  sorry

end solve_for_a_l1620_162000


namespace Allen_age_difference_l1620_162093

theorem Allen_age_difference (M A : ℕ) (h1 : M = 30) (h2 : (A + 3) + (M + 3) = 41) : M - A = 25 :=
by
  sorry

end Allen_age_difference_l1620_162093


namespace smallest_positive_n_l1620_162047

theorem smallest_positive_n : ∃ n : ℕ, 3 * n ≡ 8 [MOD 26] ∧ n = 20 :=
by 
  use 20
  simp
  sorry

end smallest_positive_n_l1620_162047


namespace positive_integer_representation_l1620_162069

theorem positive_integer_representation (a b c n : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) 
  (h₄ : n = (abc + a * b + a) / (abc + c * b + c)) : n = 1 ∨ n = 2 := 
by
  sorry

end positive_integer_representation_l1620_162069


namespace trigonometric_expression_value_l1620_162034

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  2 * (Real.sin α)^2 + 4 * Real.sin α * Real.cos α - 9 * (Real.cos α)^2 = 21 / 10 :=
by
  sorry

end trigonometric_expression_value_l1620_162034


namespace balance_expenses_l1620_162062

-- Define the basic amounts paid by Alice, Bob, and Carol
def alicePaid : ℕ := 120
def bobPaid : ℕ := 150
def carolPaid : ℕ := 210

-- The total expenditure
def totalPaid : ℕ := alicePaid + bobPaid + carolPaid

-- Each person's share of the total expenses
def eachShare : ℕ := totalPaid / 3

-- Amount Alice should give to balance the expenses
def a : ℕ := eachShare - alicePaid

-- Amount Bob should give to balance the expenses
def b : ℕ := eachShare - bobPaid

-- The statement to be proven
theorem balance_expenses : a - b = 30 :=
by
  sorry

end balance_expenses_l1620_162062


namespace pace_ratio_l1620_162051

variable (P P' D : ℝ)

-- Usual time to reach the office in minutes
def T_usual := 120

-- Time to reach the office on the late day in minutes
def T_late := 140

-- Distance to the office is the same
def office_distance_usual := P * T_usual
def office_distance_late := P' * T_late

theorem pace_ratio (h : office_distance_usual = office_distance_late) : P' / P = 6 / 7 :=
by
  sorry

end pace_ratio_l1620_162051


namespace find_m_l1620_162053

open Set

def U : Set ℕ := {0, 1, 2, 3}
def A (m : ℤ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}
def complement_A (m : ℤ) : Set ℕ := {1, 2}

theorem find_m (m : ℤ) (hA : complement_A m = U \ A m) : m = -3 :=
by
  sorry

end find_m_l1620_162053


namespace num_people_in_group_l1620_162081

-- Given conditions as definitions
def cost_per_adult_meal : ℤ := 3
def num_kids : ℤ := 7
def total_cost : ℤ := 15

-- Statement to prove
theorem num_people_in_group : 
  ∃ (num_adults : ℤ), 
    total_cost = num_adults * cost_per_adult_meal ∧ 
    (num_adults + num_kids) = 12 :=
by
  sorry

end num_people_in_group_l1620_162081


namespace find_coefficients_l1620_162067

theorem find_coefficients (A B C D : ℚ) :
  (∀ x : ℚ, x ≠ -1 → 
  (A / (x + 1)) + (B / (x + 1)^2) + ((C * x + D) / (x^2 + x + 1)) = 
  1 / ((x + 1)^2 * (x^2 + x + 1))) →
  A = 1 ∧ B = 1 ∧ C = -1 ∧ D = -1 :=
sorry

end find_coefficients_l1620_162067


namespace nine_skiers_four_overtakes_impossible_l1620_162061

theorem nine_skiers_four_overtakes_impossible :
  ∀ (skiers : Fin 9 → ℝ),  -- skiers are represented by their speeds
  (∀ i j, i < j → skiers i ≤ skiers j) →  -- skiers start sequentially and maintain constant speeds
  ¬(∀ i, (∃ a b : Fin 9, (a ≠ i ∧ b ≠ i ∧ (skiers a < skiers i ∧ skiers i < skiers b ∨ skiers b < skiers i ∧ skiers i < skiers a)))) →
    false := 
by
  sorry

end nine_skiers_four_overtakes_impossible_l1620_162061


namespace father_twice_marika_age_in_2036_l1620_162048

-- Definitions of the initial conditions
def marika_age_2006 : ℕ := 10
def father_age_2006 : ℕ := 5 * marika_age_2006

-- Definition of the statement to be proven
theorem father_twice_marika_age_in_2036 : 
  ∃ x : ℕ, (2006 + x = 2036) ∧ (father_age_2006 + x = 2 * (marika_age_2006 + x)) :=
by {
  sorry 
}

end father_twice_marika_age_in_2036_l1620_162048


namespace range_of_b_l1620_162009

noncomputable def f (b x : ℝ) : ℝ := -x^3 + b * x

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → -3 * x^2 + b ≥ 0) ↔ b ≥ 3 := sorry

end range_of_b_l1620_162009


namespace track_meet_girls_short_hair_l1620_162019

theorem track_meet_girls_short_hair :
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  girls_short_hair = 10 :=
by
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  sorry

end track_meet_girls_short_hair_l1620_162019


namespace smallest_yellow_candies_l1620_162058
open Nat

theorem smallest_yellow_candies 
  (h_red : ∃ c : ℕ, 16 * c = 720)
  (h_green : ∃ c : ℕ, 18 * c = 720)
  (h_blue : ∃ c : ℕ, 20 * c = 720)
  : ∃ n : ℕ, 30 * n = 720 ∧ n = 24 := 
by
  -- Provide the proof here
  sorry

end smallest_yellow_candies_l1620_162058


namespace num_of_factorizable_poly_l1620_162021

theorem num_of_factorizable_poly : 
  ∃ (n : ℕ), (1 ≤ n ∧ n ≤ 2023) ∧ 
              (∃ (a : ℤ), n = a * (a + 1)) :=
sorry

end num_of_factorizable_poly_l1620_162021


namespace find_length_AB_l1620_162076

variables {A B C D E : Type} -- Define variables A, B, C, D, E as types, representing points

-- Define lengths of the segments AD and CD
def length_AD : ℝ := 2
def length_CD : ℝ := 2

-- Define the angles at vertices B, C, and D
def angle_B : ℝ := 30
def angle_C : ℝ := 90
def angle_D : ℝ := 120

-- The goal is to prove the length of segment AB
theorem find_length_AB : 
  (∃ (A B C D : Type) 
    (angle_B angle_C angle_D length_AD length_CD : ℝ), 
      angle_B = 30 ∧ 
      angle_C = 90 ∧ 
      angle_D = 120 ∧ 
      length_AD = 2 ∧ 
      length_CD = 2) → 
  (length_AB = 6) := by sorry

end find_length_AB_l1620_162076


namespace min_value_four_x_plus_one_over_x_l1620_162022

theorem min_value_four_x_plus_one_over_x (x : ℝ) (hx : x > 0) : 4*x + 1/x ≥ 4 := by
  sorry

end min_value_four_x_plus_one_over_x_l1620_162022


namespace cube_volume_surface_area_x_l1620_162074

theorem cube_volume_surface_area_x (x s : ℝ) (h1 : s^3 = 8 * x) (h2 : 6 * s^2 = 2 * x) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_x_l1620_162074


namespace pure_imaginary_condition_l1620_162079

theorem pure_imaginary_condition (m : ℝ) (h : (m^2 - 3 * m) = 0) : (m = 0) :=
by
  sorry

end pure_imaginary_condition_l1620_162079


namespace cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_l1620_162044

-- Part (a)
theorem cupSaucersCombination :
  (5 : ℕ) * (3 : ℕ) = 15 :=
by
  -- Proof goes here
  sorry

-- Part (b)
theorem cupSaucerSpoonCombination :
  (5 : ℕ) * (3 : ℕ) * (4 : ℕ) = 60 :=
by
  -- Proof goes here
  sorry

-- Part (c)
theorem twoDifferentItemsCombination :
  (5 * 3 + 5 * 4 + 3 * 4 : ℕ) = 47 :=
by
  -- Proof goes here
  sorry

end cupSaucersCombination_cupSaucerSpoonCombination_twoDifferentItemsCombination_l1620_162044


namespace selling_price_eq_100_l1620_162026

variable (CP SP : ℝ)

-- Conditions
def gain : ℝ := 20
def gain_percentage : ℝ := 0.25

-- The proof of the selling price
theorem selling_price_eq_100
  (h1 : gain = 20)
  (h2 : gain_percentage = 0.25)
  (h3 : gain = gain_percentage * CP)
  (h4 : SP = CP + gain) :
  SP = 100 := sorry

end selling_price_eq_100_l1620_162026


namespace necessary_and_sufficient_condition_l1620_162037

-- Define the conditions and question in Lean 4
variable (a : ℝ) 

-- State the theorem based on the conditions and the correct answer
theorem necessary_and_sufficient_condition :
  (a > 0) ↔ (
    let z := (⟨-a, -5⟩ : ℂ)
    ∃ (x y : ℝ), (z = x + y * I) ∧ x < 0 ∧ y < 0
  ) := by
  sorry

end necessary_and_sufficient_condition_l1620_162037


namespace waiter_total_customers_l1620_162024

theorem waiter_total_customers (tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) (tables_eq : tables = 6) (women_eq : women_per_table = 3) (men_eq : men_per_table = 5) :
  tables * (women_per_table + men_per_table) = 48 :=
by
  sorry

end waiter_total_customers_l1620_162024


namespace intersection_points_l1620_162055

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 15
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 10

noncomputable def x1 : ℝ := (3 + Real.sqrt 209) / 4
noncomputable def x2 : ℝ := (3 - Real.sqrt 209) / 4

noncomputable def y1 : ℝ := parabola1 x1
noncomputable def y2 : ℝ := parabola1 x2

theorem intersection_points :
  (parabola1 x1 = parabola2 x1) ∧ (parabola1 x2 = parabola2 x2) :=
by
  sorry

end intersection_points_l1620_162055


namespace min_cans_needed_l1620_162004

theorem min_cans_needed (C : ℕ → ℕ) (H : C 1 = 15) : ∃ n, C n * n >= 64 ∧ ∀ m, m < n → C 1 * m < 64 :=
by
  sorry

end min_cans_needed_l1620_162004


namespace ratio_of_perimeters_l1620_162040

theorem ratio_of_perimeters (s S : ℝ) 
  (h1 : S = 3 * s) : 
  (4 * S) / (4 * s) = 3 :=
by
  sorry

end ratio_of_perimeters_l1620_162040


namespace percentage_of_360_equals_126_l1620_162031

/-- 
  Prove that (126 / 360) * 100 equals 35.
-/
theorem percentage_of_360_equals_126 : (126 / 360 : ℝ) * 100 = 35 := by
  sorry

end percentage_of_360_equals_126_l1620_162031


namespace increasing_on_interval_l1620_162085

noncomputable def f1 (x : ℝ) : ℝ := Real.sin x
noncomputable def f2 (x : ℝ) : ℝ := x * Real.exp 2
noncomputable def f3 (x : ℝ) : ℝ := x^3 - x
noncomputable def f4 (x : ℝ) : ℝ := Real.log x - x

theorem increasing_on_interval (x : ℝ) (h : 0 < x) : 
  f2 (x) = x * Real.exp 2 ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f1 x < f1 y) ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f3 x < f3 y) ∧ 
  (∀(x y : ℝ), 0 < x → 0 < y → x < y →  f4 x < f4 y) :=
by sorry

end increasing_on_interval_l1620_162085


namespace anthony_pencils_l1620_162065

theorem anthony_pencils (P : Nat) (h : P + 56 = 65) : P = 9 :=
by
  sorry

end anthony_pencils_l1620_162065


namespace rhombus_perimeter_l1620_162008

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 40 :=
by
  sorry

end rhombus_perimeter_l1620_162008


namespace competition_scores_l1620_162089

theorem competition_scores (n d : ℕ) (h_n : 1 < n)
  (h_total_score : d * (n * (n + 1)) / 2 = 26 * n) :
  (n, d) = (3, 13) ∨ (n, d) = (12, 4) ∨ (n, d) = (25, 2) :=
by
  sorry

end competition_scores_l1620_162089


namespace bricks_in_chimney_proof_l1620_162072

noncomputable def bricks_in_chimney (h : ℕ) : Prop :=
  let brenda_rate := h / 8
  let brandon_rate := h / 12
  let combined_rate_with_decrease := (brenda_rate + brandon_rate) - 12
  (6 * combined_rate_with_decrease = h) 

theorem bricks_in_chimney_proof : ∃ h : ℕ, bricks_in_chimney h ∧ h = 288 :=
sorry

end bricks_in_chimney_proof_l1620_162072
